#!/usr/bin/env python3
"""Noisy-label minimal validation for EuroSAT + HEC-V.

Reproduces the EXACT HEC pipeline (backbone=QWENv2 + output_llm_lasttok_heads,
classifier=HEC_V k=20 tau=10 cov_reg=1 mean_prob, prompt=TD,
EpisodicNShotKWayDataset 4-shot seed=123) and only changes the SUPPORT LABELS.

For each test episode the frozen features are computed ONCE, then three label
conditions are evaluated on the SAME support/query split:
  - clean    : original labels (baseline)
  - noise25  : 25% symmetric label noise -> 1 of 4 support samples per class flipped
  - noise50  : 50% symmetric label noise -> 2 of 4 support samples per class flipped

Per condition we record accuracy, the full head-score ranking and the top-k
selected heads so the top-k overlap / rank-change statistics can be computed
offline. Nothing else in the pipeline is modified.
"""

import os
import gc
import random
import logging
import torch
import numpy as np
from tqdm import tqdm

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.fewshot.engine import FinetuneWrapper
from src.fewshot.utils import tg_tf

from src.data_clip import build_dataset
from src.data_clip.utils import EpisodicNShotKWayDataset

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TQDM_DISABLE"] = "1"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_device(cfg):
    from hydra.core.hydra_config import HydraConfig
    from omegaconf import open_dict
    assert torch.cuda.is_available()
    n_gpus = torch.cuda.device_count()
    job = HydraConfig.get().job
    try:
        job_num = job.num          # set in multirun (-m) mode
    except Exception:
        job_num = 0                # single run
    gpu_id = job_num % n_gpus
    torch.cuda.set_device(gpu_id)
    with open_dict(cfg):
        cfg.device = f"cuda:{gpu_id}"
    logger.info(f"[Hydra job #{job_num}] using GPU {gpu_id} / {n_gpus-1}")
    return gpu_id


def make_noisy_labels(clean_labels, noise_rate, rng_seed):
    """Symmetric label noise WITHIN the episode's classes.

    For each class present in the support set, flip exactly
    round(noise_rate * n_per_class) samples to a uniformly chosen OTHER class.
    Deterministic given rng_seed.
    """
    labels = clean_labels.clone()
    uniq = torch.unique(clean_labels)
    rng = random.Random(rng_seed)
    for c in uniq.tolist():
        pos = (clean_labels == c).nonzero().flatten().tolist()
        n_flip = int(round(noise_rate * len(pos)))
        if n_flip == 0:
            continue
        to_flip = rng.sample(pos, n_flip)
        others = [u for u in uniq.tolist() if u != c]
        for idx in to_flip:
            labels[idx] = rng.choice(others)
    return labels


@hydra.main(version_base="1.3.2", config_path="./config/eval", config_name="config_clip")
def main(cfg: DictConfig) -> None:
    tg_tf()
    setup_device(cfg)
    seed = cfg.seed
    device = cfg.device
    image_size = cfg.dataset.image_size
    datasets_used = cfg.dataset.datasets
    noise_rates = list(cfg.get("noise_rates", [0.25, 0.50]))

    # ---------- MODEL (identical to eval.py) ----------
    logger.info("Loading model ...")
    backbone = instantiate(cfg.backbone, tuning_config=cfg.peft_c, device=device)
    head = instantiate(cfg.head, num_features_in=cfg.backbone.num_features_out)
    model = FinetuneWrapper(backbone=backbone, head=head, device=device)
    model.init_peft(config=cfg.peft_c)
    model.to(device)

    records = []
    for dataset_name in datasets_used:

        # ---------- RANDOM SEEDING (identical to eval.py) ----------
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ---------- DATASET PIPELINE (identical to eval.py, CD branch) ----------
        BENCHMARK = 'CD'
        cfg.dataset.path_base = cfg.dataset.path_base_CD
        dataset = build_dataset({'subsample_classes': 'all'}, dataset_name,
                                cfg.dataset.path_base, -1)
        dataset_test = dataset.test

        dataloader_test = EpisodicNShotKWayDataset(
            data_source=dataset_test,
            num_query=cfg.dataset.n_query,
            num_support=cfg.dataset.n_support,
            num_ways=cfg.dataset.n_ways,
            num_episodes=cfg.dataset.n_task,
            image_size=image_size,
            seed=seed,
        )

        # ---------- CLASSIFIER (identical to eval.py test loop) ----------
        classifier_transform = instantiate(cfg.augment_c, image_size=image_size)
        classifier = instantiate(cfg.classifier, log_losses=cfg.log_losses,
                                 transforms=classifier_transform)

        iterator = iter(dataloader_test)
        pbar = tqdm(range(cfg.dataset.n_task), desc=f"Noise exp {dataset_name}")
        for it in pbar:
            batch = next(iterator)
            support_set, support_label, query_set, query_label, label_to_str = batch
            support_set, support_label = support_set.to(device), support_label.to(device)
            query_set, query_label = query_set.to(device), query_label.to(device)

            if hasattr(model.backbone, 'init_task_specific_data'):
                model.backbone.init_task_specific_data(
                    support_set, support_label, label_to_str,
                    dataset_name, cfg.prompt, BENCHMARK,
                )
            classifier.init_task_specific_data(label_to_str, dataset_name, BENCHMARK)

            # Compute frozen features ONCE per episode (shared by all conditions)
            support_features, query_features = classifier.get_frozen_features(
                model, support_set, query_set)

            A = support_features["attn_heads"].float()          # [Ns, L, H, Dh]
            Ns, L, H, Dh = A.shape
            M = L * H
            A2 = A.reshape(Ns, M, Dh)
            A2 = A2.detach()

            ep = {"episode": it, "L": L, "H": H, "M": M,
                  "n_support": int(Ns), "n_query": int(query_set.shape[0]),
                  "clean_labels": support_label.detach().cpu().numpy(),
                  "query_label": query_label.detach().cpu().numpy()}

            for cond, lbl in [("clean", support_label)] + [
                    (f"noise{int(100*r)}", make_noisy_labels(
                        support_label, r, f"noise_s{seed}_ep{it}_r{r}"))
                    for r in noise_rates]:
                lbl = lbl.to(device)
                pred = classifier.fit_pred_features(
                    model, support_features, lbl, query_features, query_label)
                # head ranking (same code path as the classifier internals)
                rs = classifier.get_rank_scores(A2, lbl)
                head_scores = rs["head_scores"].detach().cpu().numpy()   # [M]
                order = np.argsort(head_scores)[::-1]
                selected_idx = order[: min(classifier.k, M)].astype(np.int64)

                ep[cond] = {
                    "accuracy": float(pred["accuracy"]),
                    "head_scores": head_scores.astype(np.float32),
                    "selected_idx": selected_idx,
                    "labels_used": lbl.detach().cpu().numpy(),
                    "n_flipped": int((lbl != support_label).sum().item()),
                }

            records.append(ep)
            cond_names = ["clean"] + [f"noise{int(100*r)}" for r in noise_rates]
            accs = "  ".join(f"{c}={ep[c]['accuracy']*100:.1f}%"
                             for c in cond_names)
            pbar.set_postfix_str(accs)
            logger.info(f"ep {it}: {accs}")

            del support_set, query_set, support_features, query_features, batch
            gc.collect()
            torch.cuda.empty_cache()

        # ---------- SAVE ----------
        out_path = os.path.join(
            hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            'noisy_label_records.npy')
        np.save(out_path, records)
        logger.info(f"Saved {len(records)} episode records -> {out_path}")

    del backbone, head, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

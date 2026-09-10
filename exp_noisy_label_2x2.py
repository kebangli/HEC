#!/usr/bin/env python3
"""2x2 mechanism decomposition for EuroSAT + HEC-V noisy-label.

Same pipeline as exp_noisy_label.py (backbone=QWENv2 + output_llm_lasttok_heads,
classifier=HEC_V k=20, prompt=TD, EpisodicNShotKWayDataset 4-shot seed=123, 60 eps).

For each test episode the frozen features are computed ONCE, then for each noise
level (25% / 50%) we decouple the two noise-polluted components of HEC-V:

  - "heads" : the Top-k head SELECTION (order of head_scores from get_rank_scores)
  - "GDA"   : the class-conditional Gaussian parameters used for prediction
              (head_means / head_cov_inv / log_prior from get_rank_scores)

and evaluate all four combinations:
  ① clean_heads_clean_gda  = clean selection + clean GDA     (== clean baseline)
  ② clean_heads_noisy_gda  = clean selection + noisy GDA     (GDA pollution only)
  ③ noisy_heads_clean_gda  = noisy selection + clean GDA     (head pollution only)
  ④ noisy_heads_noisy_gda  = noisy selection + noisy GDA     (== previous noisy result)

No new method: prediction reuses get_rank_scores / get_pred_labels / mean_prob
ensemble exactly as ClassifierHEC_V.fit_pred_features does.
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
from omegaconf import DictConfig

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
        job_num = job.num
    except Exception:
        job_num = 0
    gpu_id = job_num % n_gpus
    torch.cuda.set_device(gpu_id)
    with open_dict(cfg):
        cfg.device = f"cuda:{gpu_id}"
    logger.info(f"[Hydra job #{job_num}] using GPU {gpu_id} / {n_gpus-1}")
    return gpu_id


def make_noisy_labels(clean_labels, noise_rate, rng_seed):
    """Symmetric label noise WITHIN the episode's classes (identical to exp_noisy_label)."""
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


def predict_with(classifier, Aq, selected_idx, rs, query_label):
    """Exactly the mean_prob prediction block of ClassifierHEC_V.fit_pred_features,
    but with decoupled (selected_idx, GDA-dict) arguments."""
    query_predictions, _ = classifier.get_pred_labels(
        Aq[:, selected_idx, :],
        rs["head_means"][selected_idx, :, :],
        rs["head_cov_inv"][selected_idx, :, :],
        rs["log_prior"],
    )
    yhat_idx = query_predictions.mean(dim=1).argmax(dim=-1).detach().cpu().numpy()
    labels_unique = rs["labels_unique"].detach().cpu().numpy().astype(np.int64)
    yhat = labels_unique[yhat_idx]
    yq = query_label.detach().cpu().numpy().astype(np.int64)
    return float((yhat == yq).mean())


@hydra.main(version_base="1.3.2", config_path="./config/eval", config_name="config_clip")
def main(cfg: DictConfig) -> None:
    tg_tf()
    setup_device(cfg)
    seed = cfg.seed
    device = cfg.device
    image_size = cfg.dataset.image_size
    datasets_used = cfg.dataset.datasets
    noise_rates = [0.25, 0.50]

    logger.info("Loading model ...")
    backbone = instantiate(cfg.backbone, tuning_config=cfg.peft_c, device=device)
    head = instantiate(cfg.head, num_features_in=cfg.backbone.num_features_out)
    model = FinetuneWrapper(backbone=backbone, head=head, device=device)
    model.init_peft(config=cfg.peft_c)
    model.to(device)

    records = []
    feat_support, feat_query, feat_labels = [], [], []
    for dataset_name in datasets_used:

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

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

        classifier_transform = instantiate(cfg.augment_c, image_size=image_size)
        classifier = instantiate(cfg.classifier, log_losses=cfg.log_losses,
                                 transforms=classifier_transform)

        iterator = iter(dataloader_test)
        pbar = tqdm(range(cfg.dataset.n_task), desc=f"2x2 {dataset_name}")
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

            # ---------- frozen features ONCE ----------
            support_features, query_features = classifier.get_frozen_features(
                model, support_set, query_set)

            A = support_features["attn_heads"].float().detach()   # [Ns, L, H, Dh]
            Aq = query_features["attn_heads"].float().detach()    # [Nq, L, H, Dh]
            Ns, L, H, Dh = A.shape
            M = L * H
            A2 = A.reshape(Ns, M, Dh)
            Aq2 = Aq.reshape(query_set.shape[0], M, Dh)
            k = min(classifier.k, M)

            # clean-side ranking / GDA (shared by all noise rates)
            rs_clean = classifier.get_rank_scores(A2, support_label)
            sel_clean = torch.argsort(rs_clean["head_scores"], descending=True)[:k]

            ep = {"episode": it, "L": L, "H": H, "M": M,
                  "clean_labels": support_label.detach().cpu().numpy(),
                  "query_label": query_label.detach().cpu().numpy()}

            for r in noise_rates:
                noisy_labels = make_noisy_labels(
                    support_label, r, f"noise_s{seed}_ep{it}_r{r}").to(device)
                rs_noisy = classifier.get_rank_scores(A2, noisy_labels)
                sel_noisy = torch.argsort(rs_noisy["head_scores"], descending=True)[:k]

                c1 = predict_with(classifier, Aq2, sel_clean, rs_clean, query_label)
                c2 = predict_with(classifier, Aq2, sel_clean, rs_noisy, query_label)
                c3 = predict_with(classifier, Aq2, sel_noisy, rs_clean, query_label)
                c4 = predict_with(classifier, Aq2, sel_noisy, rs_noisy, query_label)

                ep[f"r{int(100*r)}"] = {
                    "noisy_labels": noisy_labels.detach().cpu().numpy(),
                    "n_flipped": int((noisy_labels != support_label).sum().item()),
                    "head_scores_clean": rs_clean["head_scores"].detach().cpu().numpy().astype(np.float32),
                    "head_scores_noisy": rs_noisy["head_scores"].detach().cpu().numpy().astype(np.float32),
                    "sel_clean": sel_clean.detach().cpu().numpy().astype(np.int64),
                    "sel_noisy": sel_noisy.detach().cpu().numpy().astype(np.int64),
                    # ① Clean heads + Clean GDA   ② Clean heads + Noisy GDA
                    # ③ Noisy heads + Clean GDA   ④ Noisy heads + Noisy GDA
                    "cH_cG": c1, "cH_nG": c2, "nH_cG": c3, "nH_nG": c4,
                }

            records.append(ep)
            # keep frozen features for future reuse (fp16)
            feat_support.append(A2.half().cpu())
            feat_query.append(Aq2.half().cpu())
            feat_labels.append({
                "clean_labels": support_label.detach().cpu().numpy(),
                "query_label": query_label.detach().cpu().numpy(),
                "n_support": int(Ns), "n_query": int(query_set.shape[0]),
            })

            pbar.set_postfix_str(
                f"cHcG={c1*100:.1f}% cHnG={c2*100:.1f}% nHcG={c3*100:.1f}% nHnG={c4*100:.1f}%")
            logger.info(
                f"ep {it}: cleanGDA+cleanH={c1*100:.1f} | cH+nG={c2*100:.1f} "
                f"| nH+cG={c3*100:.1f} | nH+nG={c4*100:.1f}")

            del support_set, query_set, support_features, query_features, batch
            gc.collect()
            torch.cuda.empty_cache()

        out_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        np.save(os.path.join(out_dir, 'noisy_label_2x2_records.npy'), records)
        # frozen features, fp16, for any future reuse
        np.savez(os.path.join(out_dir, 'features.npz'),
                 support_A=np.stack(feat_support), query_A=np.stack(feat_query))
        np.save(os.path.join(out_dir, 'feat_labels.npy'), feat_labels)
        logger.info(f"Saved {len(records)} records + features -> {out_dir}")

    del backbone, head, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

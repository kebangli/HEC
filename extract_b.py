#!/usr/bin/env python3
# Module B: feature extraction for cached evaluation.
#
# Extracts the Qwen2-VL attention-head features (["attn_heads"]) for every
# evaluation episode using the EXACT official HEC pipeline: same hydra
# configs, same EpisodicNShotKWayDataset sampling (seed + episode_idx),
# same per-episode prompts (init_task_specific_data), same model.infer
# batching as eval.py. Episodes are therefore bit-identical to
# `python -m eval` runs with the same seed / n_task.
#
# Variants (Fisher ratios, power transforms, ...) then run on the cache in
# seconds via run_variants_b.py. Parity with the official harness is
# verified separately (baseline + one refined config).

import os
import gc
import random
import logging
import torch
import numpy as np
from tqdm import tqdm

# Hydra Imports
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

# Import project modules
from src.fewshot.engine import FinetuneWrapper
from src.data_clip import build_dataset
from src.data_clip.utils import EpisodicNShotKWayDataset

# Configure logging / env variables
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TQDM_DISABLE"] = "1"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Logging is set up.")

CACHE_DIR = "/data1/lbk/module_b/cache"


def to_device(x, device):
    if torch.is_tensor(x):
        return x.to(device)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(v, device) for v in x)
    return x


@hydra.main(version_base="1.3.2", config_path="./config/eval", config_name="config_clip")
def main(cfg: DictConfig) -> None:

    # ------    GET PARAMS    ------ #
    seed = cfg.seed
    device = cfg.device
    image_size = cfg.dataset.image_size
    datasets_used = cfg.dataset.datasets

    # ------    MODEL INIT    ------ #
    logger.info("Loading the model once")
    backbone = instantiate(
        cfg.backbone,
        tuning_config=cfg.peft_c,
        device=device
    )
    head = instantiate(
        cfg.head,
        num_features_in=cfg.backbone.num_features_out
    )
    model = FinetuneWrapper(
        backbone=backbone,
        head=head,
        device=device,
    )
    model.init_peft(config=cfg.peft_c)
    model.to(device)

    for dataset_name in datasets_used:

        # ------ RANDOM SEEDING (same as eval.py) ------ #
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # ----- DATASET PIPELINE (same as eval.py) ----- #
        if dataset_name in ['cu_birds', 'traffic_sign']:
            BENCHMARK = 'MD'
            cfg.dataset.path_base = cfg.dataset.path_base_MD
        else:
            BENCHMARK = 'CD'
            cfg.dataset.path_base = cfg.dataset.path_base_CD

        dataset = build_dataset({'subsample_classes': 'all'}, dataset_name, cfg.dataset.path_base, -1)
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

        logger.info(f"[{dataset_name}] test classes={dataset.num_classes} "
                    f"test images={len(dataset_test)} episodes={cfg.dataset.n_task} "
                    f"ways={cfg.dataset.n_ways} support={cfg.dataset.n_support} query={cfg.dataset.n_query}")

        # ----- EXTRACTION LOOP (mirrors eval.py testing loop) ----- #
        iterator = iter(dataloader_test)
        episodes = []
        for iteration_number in tqdm(range(cfg.dataset.n_task), desc=f"Extract {dataset_name}"):
            batch = next(iterator)

            # Data Init (same order as eval.py)
            support_set, support_label, query_set, query_label, label_to_str = batch
            support_set, support_label = support_set.to(device), support_label.to(device)
            query_set, query_label = query_set.to(device), query_label.to(device)

            if hasattr(model.backbone, 'init_task_specific_data'):
                model.backbone.init_task_specific_data(
                    support_set,
                    support_label,
                    label_to_str,
                    dataset_name,
                    cfg.prompt,
                    BENCHMARK,
                )

            with torch.no_grad():
                sf = model.infer(support_set)
                qf = model.infer(query_set)

            episodes.append({
                "support_heads": sf["attn_heads"].to(torch.float16).cpu(),
                "support_label": support_label.detach().cpu(),
                "query_heads": qf["attn_heads"].to(torch.float16).cpu(),
                "query_label": query_label.detach().cpu(),
                "label_to_str": label_to_str,
            })

            logger.info(f"[{dataset_name}] episode {iteration_number:04d} done "
                        f"(support {tuple(support_set.shape)}, query {tuple(query_set.shape)})")

            # -- Clean Up -- #
            del sf, qf, support_set, query_set, batch
            gc.collect()
            torch.cuda.empty_cache()

        # ----- SAVE CACHE ----- #
        os.makedirs(CACHE_DIR, exist_ok=True)
        out_path = os.path.join(
            CACHE_DIR,
            f"{dataset_name}_seed{seed}_ntask{cfg.dataset.n_task}.pt"
        )
        torch.save({
            "dataset_name": dataset_name,
            "seed": seed,
            "n_task": cfg.dataset.n_task,
            "n_support": cfg.dataset.n_support,
            "n_ways": cfg.dataset.n_ways,
            "n_query": cfg.dataset.n_query,
            "episodes": episodes,
        }, out_path)
        logger.info(f"[SAVED] {out_path} ({len(episodes)} episodes)")

        del episodes
        gc.collect()
        torch.cuda.empty_cache()

    # -- Final Clean Up -- #
    del backbone, head, model
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

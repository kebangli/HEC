#!/usr/bin/env python3
# Module B: variant runner on cached head features.
#
# Evaluates the baseline ClassifierHEC_V and the refinement variants
# (Fisher channel selection x power transform) on cached episodes produced
# by extract_b.py. Episodes are identical to the official harness (same
# seed / n_task); parity is verified separately. Saves per-episode
# accuracies for paired statistics against the baseline.

import argparse
import json
import os

import numpy as np
import torch

from src.fewshot.classifier import ClassifierHEC_V
from src.fewshot.classifier_refine import ClassifierHEC_V_refine

# Official HEC-V hyperparameters (config/eval/classifier/HEC_V.yaml)
BASE_PARAMS = dict(
    input_mode="cls",
    k=20,
    tau=10.0,
    cov_reg=1.0,
    ensemble_method="mean_prob",
    context="Class",
)

CACHE_DIR = "/data1/lbk/module_b/cache"
OUT_DIR = "/data2/lbk/HEC_repro/logs/module_b/variants"

FISHER_RATIOS = [0.75, 0.5, 0.25]
POWER_PS = [0.5, 0.25, 0.75]
COMBO_GRID = [(0.75, 0.5), (0.5, 0.5), (0.25, 0.5), (0.5, 0.25), (0.5, 0.75)]


def build_variants():
    """name -> factory returning a fresh classifier instance."""
    variants = {}

    def baseline_factory():
        return ClassifierHEC_V(**BASE_PARAMS)

    def refine_factory(**over):
        params = dict(BASE_PARAMS)
        params.update(over)
        return ClassifierHEC_V_refine(**params)

    variants["baseline"] = baseline_factory

    # Experiment 1: APE-style Fisher channel refinement
    for r in FISHER_RATIOS:
        variants[f"fisher_{r}"] = (lambda r=r: refine_factory(fisher_ratio=r))

    # Ablation: no renormalization after masking
    variants["fisher_0.5_norenorm"] = (
        lambda: refine_factory(fisher_ratio=0.5, mask_renorm=False)
    )

    # Experiment 2: PT-MAP power transformation
    for p in POWER_PS:
        variants[f"power_{p}"] = (lambda p=p: refine_factory(power_p=p))

    # Experiment 3: combinations
    for r, p in COMBO_GRID:
        variants[f"fisher_{r}_power_{p}"] = (
            lambda r=r, p=p: refine_factory(fisher_ratio=r, power_p=p)
        )

    return variants


def run_variant(factory, cache, device):
    """Returns the list of per-episode accuracies."""
    accs = []
    for ep in cache["episodes"]:
        support_features = {"attn_heads": ep["support_heads"].to(device).float()}
        query_features = {"attn_heads": ep["query_heads"].to(device).float()}
        support_label = ep["support_label"].to(device)
        query_label = ep["query_label"].to(device)

        classifier = factory()
        classifier.init_task_specific_data(
            ep["label_to_str"], cache["dataset_name"], "CD"
        )
        with torch.inference_mode():
            out = classifier.fit_pred_features(
                None,                     # unused by HEC-V path
                support_features,
                support_label,
                query_features,
                query_label,
            )
        accs.append(out["accuracy"])
        del support_features, query_features
        if device != "cpu":
            torch.cuda.empty_cache()
    return accs


def paired_stats(base_accs, var_accs):
    base_accs = np.asarray(base_accs, dtype=np.float64)
    var_accs = np.asarray(var_accs, dtype=np.float64)
    diff = var_accs - base_accs
    stats = {
        "mean_diff": float(diff.mean()),
        "win_rate": float((diff > 0).mean()),
        "n": int(len(diff)),
    }
    try:
        from scipy.stats import wilcoxon, ttest_rel
        if np.allclose(diff, 0):
            stats["p_wilcoxon"] = 1.0
            stats["p_ttest"] = 1.0
        else:
            stats["p_wilcoxon"] = float(wilcoxon(var_accs, base_accs).pvalue)
            stats["p_ttest"] = float(ttest_rel(var_accs, base_accs).pvalue)
    except Exception as e:  # scipy missing or degenerate
        stats["p_error"] = str(e)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="path to cache .pt file")
    parser.add_argument("--out_dir", default=OUT_DIR)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="optional subset of variant names to run")
    args = parser.parse_args()

    cache = torch.load(args.cache, map_location="cpu")
    name = cache["dataset_name"]
    n_task = cache["n_task"]
    print(f"[{name}] seed={cache['seed']} n_task={n_task} episodes={len(cache['episodes'])}")

    variants = build_variants()
    if args.variants:
        variants = {k: v for k, v in variants.items() if k in args.variants}

    results = {
        "dataset": name,
        "seed": cache["seed"],
        "n_task": n_task,
        "episodes": len(cache["episodes"]),
        "variants": {},
        "paired_vs_baseline": {},
    }

    for vname, factory in variants.items():
        accs = run_variant(factory, cache, args.device)
        accs_arr = np.asarray(accs)
        entry = {
            "accs": accs,
            "mean": float(accs_arr.mean()),
            "std": float(accs_arr.std(ddof=1)) if len(accs_arr) > 1 else 0.0,
            "first5_mean": float(accs_arr[:5].mean()) if len(accs_arr) >= 5 else None,
        }
        results["variants"][vname] = entry
        print(f"  {vname:<28s} mean={100*entry['mean']:.2f} "
              f"(first5={100*(entry['first5_mean'] or 0):.2f}) "
              f"std={100*entry['std']:.2f}")
        if vname != "baseline":
            results["paired_vs_baseline"][vname] = paired_stats(
                results["variants"]["baseline"]["accs"], accs
            )

    os.makedirs(args.out_dir, exist_ok=True)
    ds_tag = os.path.splitext(os.path.basename(args.cache))[0]
    out_path = os.path.join(args.out_dir, f"{ds_tag}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] {out_path}")


if __name__ == "__main__":
    main()

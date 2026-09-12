#!/usr/bin/env python3
# Module B diagnostic: oracle Fisher (support+query labels) vs support Fisher
# vs random mask. NOT a method (uses query labels) — purely diagnostic to
# distinguish "dimension selection cannot help HEC-V" from "Fisher is noisy".
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from src.fewshot.classifier import ClassifierHEC_V

BASE = dict(input_mode="cls", k=20, tau=10.0, cov_reg=1.0,
            ensemble_method="mean_prob", context="Class")


def fisher_mask(xs, ys, ratio):
    """xs [N, M, C] normalized, ys [N] -> mask [M, C]"""
    N, M, C = xs.shape
    labels_unique = torch.unique(ys)
    K = labels_unique.numel()
    y_idx = torch.searchsorted(labels_unique, ys)
    mu_k = torch.stack([xs[ys == lbl].mean(0) for lbl in labels_unique])
    mu = xs.mean(0)
    counts = torch.stack([(ys == lbl).sum() for lbl in labels_unique]).to(xs.dtype)
    w = (counts / counts.sum())[:, None, None]
    between = (w * (mu_k - mu[None]) ** 2).sum(0)
    centered = xs - mu_k[y_idx]
    within = (centered ** 2).sum(0) / max(N - K, 1)
    fisher = between / (within + 1e-12)
    n_keep = max(1, int(round(ratio * C)))
    mask = torch.zeros(M, C, dtype=torch.bool, device=xs.device)
    idx = fisher.topk(n_keep, dim=-1).indices
    mask.scatter_(1, idx, True)
    return mask


def run_episode(cache_ep, device, ratio, mode):
    """mode: support_fisher | oracle_fisher | random"""
    sfeat = {"attn_heads": cache_ep["support_heads"].to(device).float()}
    qfeat = {"attn_heads": cache_ep["query_heads"].to(device).float()}
    ys = cache_ep["support_label"].to(device)
    yq = cache_ep["query_label"].to(device)

    clf = ClassifierHEC_V(**BASE)
    clf.init_task_specific_data(cache_ep["label_to_str"], "diag", "CD")
    with torch.inference_mode():
        out = clf.fit_pred_features(None, sfeat, ys, qfeat, yq)
    base_acc = out["accuracy"]

    # ---- re-run with masked dims (replicating the refine path) ----
    from src.fewshot.classifier_refine import ClassifierHEC_V_refine
    clf_r = ClassifierHEC_V_refine(**BASE, fisher_ratio=1.0)  # manual mask below
    A = cache_ep["support_heads"].to(device).float()
    Aq = cache_ep["query_heads"].to(device).float()
    Ns, L, H, Dh = A.shape
    M = L * H
    A = A.reshape(Ns, M, Dh)
    Aq = Aq.reshape(Aq.shape[0], M, Dh)
    with torch.inference_mode():
        d = clf_r.get_rank_scores(A, ys)
        order = torch.argsort(d["head_scores"], descending=True)
        sel = order[:20]
        xs = F.normalize(A[:, sel, :], dim=-1)
        xq = F.normalize(Aq[:, sel, :], dim=-1)
        if mode == "support_fisher":
            mask = fisher_mask(xs, ys, ratio)
        elif mode == "oracle_fisher":
            xall = torch.cat([xs, xq], dim=0)
            yall = torch.cat([ys, yq], dim=0)
            mask = fisher_mask(xall, yall, ratio)
        elif mode == "random":
            g = torch.Generator(device="cpu").manual_seed(12345)
            mask = torch.rand(20, Dh, generator=g).to(device) < ratio
            mask[:, 0] = True  # avoid all-zero rows edge
        xs = xs * mask[None].to(xs.dtype)
        xq = xq * mask[None].to(xq.dtype)
        xs = F.normalize(xs, dim=-1)
        xq = F.normalize(xq, dim=-1)
        means, cov_inv, log_prior, _ = clf_r._fit_gda(xs, ys)
        z = clf_r._predict_gda(xq, means, cov_inv, log_prior)
        yhat = z.mean(dim=1).argmax(dim=-1).cpu().numpy()
        lu = torch.unique(ys).cpu().numpy()
        acc = float((lu[yhat] == yq.cpu().numpy()).mean())
    return base_acc, acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cache = torch.load(args.cache, map_location="cpu")
    name = cache["dataset_name"]
    print(f"=== {name} (n={len(cache['episodes'])}) ===")
    print(f"{'config':<24s} {'base':>6s} {'masked':>7s} {'diff':>7s}")

    for ratio in [0.5, 0.25]:
        for mode in ["support_fisher", "oracle_fisher", "random"]:
            base_accs, mask_accs = [], []
            for ep in cache["episodes"]:
                b, m = run_episode(ep, args.device, ratio, mode)
                base_accs.append(b)
                mask_accs.append(m)
            base_accs, mask_accs = np.array(base_accs), np.array(mask_accs)
            print(f"{mode + '_' + str(ratio):<24s} {100*base_accs.mean():6.2f} "
                  f"{100*mask_accs.mean():7.2f} {100*(mask_accs-base_accs).mean():+7.2f}")


if __name__ == "__main__":
    main()

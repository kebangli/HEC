#!/usr/bin/env python3
"""Direct paired accuracy validation of head-level uncertainty weighting.

Loads headunc records (head_probs [N,topk,K], query_label, labels_unique).
Compares, on the SAME episodes:
  baseline  : equal-weight mean of head probs
  uncertain : reliability-weighted (w_m = (1-U_m)/sum(1-U)) mean of head probs
Outputs baseline/uncertainty acc, per-episode, flip counts, AUROC(-U, correct).
"""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

L_EPS = 1e-12
W_EPS = 1e-8


def norm_entropy(P):
    P = np.clip(P, L_EPS, 1.0)
    return -(P * np.log(P)).sum(axis=-1) / np.log(P.shape[-1])


def load_records(path):
    recs = np.load(path, allow_pickle=True).tolist()
    HPs, yidxs = [], []
    for r in recs:
        HPs.append(r['head_probs'])
        yidxs.append(np.searchsorted(r['labels_unique'], r['query_label']))
    HP = np.concatenate(HPs, axis=0)
    y_idx = np.concatenate(yidxs, axis=0)
    return HP, y_idx, recs


def main(path):
    HP, y_idx, recs = load_records(path)
    N, topk, K = HP.shape
    print(f"# queries = {N}, selected heads = {topk}, classes = {K}, episodes = {len(recs)}")

    U = norm_entropy(HP)
    r = 1.0 - U
    w = r / (r.sum(-1, keepdims=True) + W_EPS)

    P_base = HP.mean(1)
    P_unc = (w[..., None] * HP).sum(1)
    pred_base = P_base.argmax(-1)
    pred_unc = P_unc.argmax(-1)
    acc_base = float((pred_base == y_idx).mean())
    acc_unc = float((pred_unc == y_idx).mean())

    print("\n================ ACCURACY (paired, same episodes) ================")
    print(f"  Baseline HEC-V (equal-weight):       {100*acc_base:.2f}%  ({int((pred_base==y_idx).sum())}/{N})")
    print(f"  HEC-V + head-level uncertainty:      {100*acc_unc:.2f}%  ({int((pred_unc==y_idx).sum())}/{N})")
    print(f"  Delta: {100*(acc_unc-acc_base):+.2f} pp")

    print("\n  per-episode:")
    off = 0
    for i, rr in enumerate(recs):
        nq = rr['head_probs'].shape[0]
        lu = rr['labels_unique']
        yi = np.searchsorted(lu, rr['query_label'])
        pb = rr['head_probs'].mean(1).argmax(-1)
        uu = norm_entropy(rr['head_probs'])
        ww = (1 - uu) / ((1 - uu).sum(-1, keepdims=True) + W_EPS)
        pu = (ww[..., None] * rr['head_probs']).sum(1).argmax(-1)
        a, c = float((pb == yi).mean()), float((pu == yi).mean())
        print(f"    ep{i}: baseline={100*a:6.2f}%  uncertainty={100*c:6.2f}%  (delta {100*(c-a):+5.2f})")
        off += nq

    # flips
    bw = (pred_base != y_idx) & (pred_unc == y_idx)     # baseline wrong -> unc correct
    bc = (pred_base == y_idx) & (pred_unc != y_idx)     # baseline correct -> unc wrong
    print("\n================ FLIPS ================")
    print(f"  baseline wrong -> uncertainty correct: {int(bw.sum())}")
    print(f"  baseline correct -> uncertainty wrong: {int(bc.sum())}")
    print(f"  (net = {int(bw.sum()) - int(bc.sum())})")

    # AUROC
    head_correct = HP.argmax(-1) == y_idx[:, None]
    auroc = roc_auc_score(head_correct.flatten().astype(int), -U.flatten())
    print("\n================ HEAD-LEVEL AUROC ================")
    print(f"  AUROC(-U_m, head_correct) = {auroc:.4f}  (samples={int(head_correct.size)})")
    Uc, Uw = U[head_correct], U[~head_correct]
    print(f"  entropy: correct heads mean={Uc.mean():.4f} med={np.median(Uc):.4f} | "
          f"wrong heads mean={Uw.mean():.4f} med={np.median(Uw):.4f}")
    print(f"  weight: correct heads mean={w[head_correct].mean():.4f} | wrong heads mean={w[~head_correct].mean():.4f} (equal {1/topk:.4f})")


if __name__ == "__main__":
    main(sys.argv[1])

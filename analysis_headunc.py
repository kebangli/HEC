#!/usr/bin/env python3
"""Head-level uncertainty analysis for EuroSAT + HEC-V.

Data: unc_records.npy with per-episode:
    head_probs     [Nq, topk, K]  per-head GDA+softmax class probabilities
    query_label    [Nq]
    labels_unique  [K]            sorted class id mapping
    acc_fixed                       baseline (equal-weight) accuracy

Metrics:
  1. Baseline (equal-weight) vs uncertainty-weighted accuracy, dAcc
  2. AUROC(-U_m, head_correct) over all query x selected-head samples
  3. Correct vs wrong head entropy (mean/median)
  4. Correct vs wrong head weight (mean/median)
  5. Does weighting reduce wrong-head contribution / boost correct-head?
  6. On head disagreement: does the low-entropy head predict correctly more often?
"""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

L_EPS = 1e-12
W_EPS = 1e-8


def norm_entropy(P):
    """U = -sum_c p_c log p_c / log C  (last axis)."""
    P = np.clip(P, L_EPS, 1.0)
    return -(P * np.log(P)).sum(axis=-1) / np.log(P.shape[-1])


def load_records(path):
    recs = np.load(path, allow_pickle=True).tolist()
    HPs, ys, yidxs = [], [], []
    for r in recs:
        HPs.append(r['head_probs'])
        ys.append(r['query_label'])
        lu = r['labels_unique']
        yidxs.append(np.searchsorted(lu, r['query_label']))   # per-episode true-class index
    HP = np.concatenate(HPs, axis=0)          # [N, topk, K]
    y = np.concatenate(ys, axis=0)            # [N]
    y_idx = np.concatenate(yidxs, axis=0)     # [N] index of true class in that episode's labels_unique
    return HP, y, y_idx, recs


def main(path):
    HP, y, y_idx, recs = load_records(path)
    N, topk, K = HP.shape
    print(f"# queries = {N}, selected heads = {topk}, classes = {K}, episodes = {len(recs)}")

    head_pred = HP.argmax(axis=-1)                                     # [N, topk]
    head_correct = head_pred == y_idx[:, None]                         # [N, topk]

    U = norm_entropy(HP)                                               # [N, topk]
    r = 1.0 - U
    w = r / (r.sum(axis=-1, keepdims=True) + W_EPS)                    # [N, topk] per-query normalized

    # ---------------- 1) accuracies ----------------
    P_base = HP.mean(axis=1)                                           # [N, K] equal-weight
    P_unc = (w[..., None] * HP).sum(axis=1)                            # [N, K] reliability-weighted
    pred_base = P_base.argmax(-1)
    pred_unc = P_unc.argmax(-1)
    acc_base = float((pred_base == y_idx).mean())
    acc_unc = float((pred_unc == y_idx).mean())
    print("\n================ 1. ACCURACY ================")
    print(f"  Baseline HEC-V (equal-weight mean_prob):   {100*acc_base:.2f}%  ({int((pred_base==y_idx).sum())}/{N})")
    print(f"  HEC-V + head-level 1-U weighting:          {100*acc_unc:.2f}%  ({int((pred_unc==y_idx).sum())}/{N})")
    print(f"  Delta: {100*(acc_unc-acc_base):+.2f} pp")
    # cross-check per-episode with logged acc_fixed
    print("  per-episode (log acc_fixed vs recomputed equal-weight):")
    off = 0
    for i, rr in enumerate(recs):
        nq = rr['head_probs'].shape[0]
        lu = rr['labels_unique']
        yi = np.searchsorted(lu, rr['query_label'])
        pb = rr['head_probs'].mean(1).argmax(-1)
        ep = float((pb == yi).mean())
        print(f"    ep{i}: log={100*rr['acc_fixed']:.2f}%  recomputed={100*ep:.2f}%")
        off += nq

    # ---------------- 2) AUROC ----------------
    flat_U = U.reshape(-1)
    flat_c = head_correct.reshape(-1)
    auroc = roc_auc_score(flat_c.astype(int), -flat_U)
    print("\n================ 2. HEAD-LEVEL AUROC ================")
    print(f"  samples = {flat_U.size} (query x selected-head)")
    print(f"  AUROC(-U_m, head_correct) = {auroc:.4f}")
    print(f"  head accuracy (fraction of heads correct) = {100*flat_c.mean():.2f}%")

    # ---------------- 3) entropy: correct vs wrong heads ----------------
    print("\n================ 3. ENTROPY: CORRECT vs WRONG HEADS ================")
    Uc, Uw = U[head_correct], U[~head_correct]
    print(f"  correct heads: mean={Uc.mean():.4f}  median={np.median(Uc):.4f}")
    print(f"  wrong   heads: mean={Uw.mean():.4f}  median={np.median(Uw):.4f}")

    # ---------------- 4) weight: correct vs wrong heads ----------------
    print("\n================ 4. WEIGHT: CORRECT vs WRONG HEADS ================")
    wc, ww = w[head_correct], w[~head_correct]
    eq_w = 1.0 / topk
    print(f"  equal weight (1/k) = {eq_w:.4f}")
    print(f"  correct heads: mean w={wc.mean():.4f}  median w={np.median(wc):.4f}")
    print(f"  wrong   heads: mean w={ww.mean():.4f}  median w={np.median(ww):.4f}")

    # ---------------- 5) does weighting shift mass correctly ----------------
    print("\n================ 5. WEIGHT MASS SHIFT ================")
    frac_wrong = (~head_correct).mean(axis=1)                    # [N] per-query fraction of wrong heads
    wrong_mass_eq = frac_wrong                                   # = mass on wrong heads under equal weight (weights sum to 1)
    wrong_mass_unc = (w * (~head_correct)).sum(axis=1)           # [N] mass on wrong heads under weighting
    print(f"  wrong-head weight mass (per query): equal-weight {wrong_mass_eq.mean():.4f} vs weighted {wrong_mass_unc.mean():.4f} "
          f"({100*(wrong_mass_unc.mean()-wrong_mass_eq.mean())/wrong_mass_eq.mean():+.1f}% rel)")
    print(f"  correct-head weight mass (per query): equal-weight {1-wrong_mass_eq.mean():.4f} vs weighted {1-wrong_mass_unc.mean():.4f}")
    # per-query: fraction of queries where weighting reduced wrong-head mass
    red = wrong_mass_unc < wrong_mass_eq
    print(f"  fraction of queries where weighting REDUCED wrong-head mass: {100*red.mean():.1f}%")

    # ---------------- 6) disagreement: low-entropy heads more correct ----------------
    print("\n================ 6. HEAD DISAGREEMENT ANALYSIS ================")
    # pairwise: among pairs of heads that disagree, does the lower-U head win more?
    n_pair, lower_wins, higher_wins = 0, 0, 0
    for i in range(N):
        Ui, pi, ci = U[i], head_pred[i], head_correct[i]
        for a in range(topk):
            for b in range(a + 1, topk):
                if pi[a] == pi[b]:
                    continue                      # agree -> no attribution
                n_pair += 1
                if ci[a] and not ci[b]:
                    win = a
                elif ci[b] and not ci[a]:
                    win = b
                else:
                    continue                      # both right / both wrong
                lower = a if Ui[a] < Ui[b] else b
                if lower == win:
                    lower_wins += 1
                else:
                    higher_wins += 1
    print(f"  disagreeing head pairs (resolvable) = {lower_wins+higher_wins}, total pairs examined = {n_pair}")
    if lower_wins + higher_wins > 0:
        rate = lower_wins / (lower_wins + higher_wins)
        print(f"  lower-entropy head is the correct one in {100*rate:.1f}% "
              f"({lower_wins}/{lower_wins+higher_wins}) of resolvable disagreements")

    # aggregate: per-query, heads that disagree with the equal-weight ensemble pred
    dis = head_pred != pred_base[:, None]
    if dis.sum() > 0:
        umed = np.median(U, axis=1, keepdims=True)
        low = (U < umed) & dis
        high = (U >= umed) & dis
        acc_low = head_correct[low].mean() if low.sum() else float('nan')
        acc_high = head_correct[high].mean() if high.sum() else float('nan')
        print(f"  heads disagreeing with ensemble: {int(dis.sum())} samples "
              f"(acc low-entropy={100*acc_low:.1f}% vs high-entropy={100*acc_high:.1f}%)")

    # ---------------- 7) conclusion summary ----------------
    print("\n================ SUMMARY ================")
    print(f"  Baseline acc={100*acc_base:.2f}%  Weighted acc={100*acc_unc:.2f}%  dAcc={100*(acc_unc-acc_base):+.2f}pp")
    print(f"  AUROC(-U_m, head_correct)={auroc:.4f}")
    print(f"  head weight shift: correct heads get {wc.mean():.4f} vs wrong heads {ww.mean():.4f} "
          f"(equal {eq_w:.4f}); wrong-head mass reduced in {100*red.mean():.1f}% of queries")
    if lower_wins + higher_wins > 0:
        print(f"  on head disagreement, lower-entropy head correct in {100*rate:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1])

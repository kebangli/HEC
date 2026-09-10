#!/usr/bin/env python3
"""Pure offline Oracle / diagnostic analysis for EuroSAT + HEC-V head uncertainty.

Reads the head-level unc_records.npy produced in round 3
(250 queries x 20 selected heads x 10 classes, GDA+softmax per-head probs).
No model re-run, no modification, no tuning.

Sections:
  1. Baseline confirmation (equal-weight mean) + per-episode accuracy
  2. Oracle single-head upper bound & correct-head availability
  3. Entropy Top-r diagnostic (r = 1,3,5,10,20; equal-weight within top-r)
  4. Confidently-wrong heads analysis
  5. Go / No-Go summary
"""
import sys
import numpy as np

L_EPS = 1e-12


def norm_entropy(P):
    P = np.clip(P, L_EPS, 1.0)
    return -(P * np.log(P)).sum(axis=-1) / np.log(P.shape[-1])


def load_records(path):
    recs = np.load(path, allow_pickle=True).tolist()
    HPs, yidxs = [], []
    for r in recs:
        HPs.append(r['head_probs'])
        yidxs.append(np.searchsorted(r['labels_unique'], r['query_label']))
    HP = np.concatenate(HPs, axis=0)          # [N, 20, 10]
    y_idx = np.concatenate(yidxs, axis=0)     # [N]
    return HP, y_idx, recs


def main(path):
    HP, y_idx, recs = load_records(path)
    N, topk, K = HP.shape
    print(f"# queries = {N}, selected heads = {topk}, classes = {K}, episodes = {len(recs)}")

    U = norm_entropy(HP)                       # [N, topk]
    head_pred = HP.argmax(-1)                  # [N, topk]
    head_correct = head_pred == y_idx[:, None] # [N, topk]

    # ---------------- 1) baseline ----------------
    P_base = HP.mean(axis=1)                   # equal-weight mean
    pred_base = P_base.argmax(-1)
    acc_base = float((pred_base == y_idx).mean())
    print("\n================ 1. BASELINE (recomputed from head probs) ================")
    print(f"  equal-weight mean P_base accuracy: {100*acc_base:.2f}%  ({int((pred_base==y_idx).sum())}/{N})")
    print("  per-episode:")
    off = 0
    for i, rr in enumerate(recs):
        nq = rr['head_probs'].shape[0]
        pb = rr['head_probs'].mean(1).argmax(-1)
        yi = np.searchsorted(rr['labels_unique'], rr['query_label'])
        ep = float((pb == yi).mean())
        print(f"    ep{i}: recomputed={100*ep:.2f}%  (logged acc_fixed={100*rr['acc_fixed']:.2f}%)")
        off += nq

    # ---------------- 2) Oracle upper bound ----------------
    print("\n================ 2. ORACLE SINGLE-HEAD UPPER BOUND ================")
    has_correct = head_correct.any(axis=1)     # [N] at least one correct selected head
    oracle = float(has_correct.mean())
    print(f"  queries with >=1 correct head among 20: {100*oracle:.2f}%  ({int(has_correct.sum())}/{N})")
    print(f"  -> Oracle single-head upper bound accuracy = {100*oracle:.2f}%   (vs baseline {100*acc_base:.2f}%, +{100*(oracle-acc_base):.2f} pp)")

    n_correct = head_correct.sum(axis=1)       # [N]
    print(f"  #correct heads per query: mean={n_correct.mean():.2f} median={np.median(n_correct):.0f} "
          f"min={n_correct.min()} max={n_correct.max()}")
    hist = np.bincount(n_correct.astype(int), minlength=topk + 1)[:topk + 1]
    print(f"  distribution (n correct -> count): {[(i, int(c)) for i, c in enumerate(hist) if c > 0]}")

    base_wrong = pred_base != y_idx
    hc_on_wrong = has_correct[base_wrong]
    print(f"\n  HEC-V baseline WRONG queries: {int(base_wrong.sum())}/{N} ({100*base_wrong.mean():.2f}%)")
    print(f"    of these, >=1 correct head exists: {int(hc_on_wrong.sum())}/{int(base_wrong.sum())} "
          f"({100*hc_on_wrong.mean():.1f}%)  -> 'correct expert exists but not exploited'")
    print(f"    of these, NO correct head in top-20: {int((~hc_on_wrong).sum())}/{int(base_wrong.sum())} "
          f"({100*(~hc_on_wrong).mean():.1f}%)  -> 'no correct expert in top-20'")

    # ---------------- 3) Entropy Top-r diagnostic ----------------
    print("\n================ 3. ENTROPY TOP-r DIAGNOSTIC (lowest U first, equal-weight) ================")
    order = np.argsort(U, axis=1)              # ascending U (low entropy first)
    print("  r   Acc       dAcc(vs baseline)")
    for r in (1, 3, 5, 10, 20):
        Pr = HP[np.arange(N)[:, None], order[:, :r], :].mean(axis=1)
        acc = float((Pr.argmax(-1) == y_idx).mean())
        d = 100 * (acc - acc_base)
        tag = "  <- baseline" if r == topk else ""
        print(f"  {r:>2}   {100*acc:6.2f}%   {d:+6.2f} pp{tag}")

    # ---------------- 4) Confidently-wrong analysis ----------------
    print("\n================ 4. CONFIDENTLY-WRONG HEADS ================")
    umed = np.median(U, axis=1, keepdims=True)     # per-query median entropy
    low = U < umed                                  # low-entropy (more confident) half
    cw = low & ~head_correct                        # confidently-wrong
    acc_low = head_correct[low].mean() if low.sum() else float('nan')
    acc_high = head_correct[~low].mean() if (~low).sum() else float('nan')
    print(f"  low-entropy heads: {int(low.sum())} ({100*low.mean():.1f}%), head-acc={100*acc_low:.1f}%")
    print(f"  high-entropy heads: {int((~low).sum())} ({100*(~low).mean():.1f}%), head-acc={100*acc_high:.1f}%")
    print(f"  confidently-wrong (low-entropy & wrong): {int(cw.sum())} ({100*cw.mean():.2f}% of all heads; "
          f"{100*cw[low].mean():.1f}% of low-entropy heads are wrong)")

    # top-r flip analysis
    print("  per-r flips vs baseline (r=20):")
    for r in (1, 3, 5, 10):
        Pr = HP[np.arange(N)[:, None], order[:, :r], :].mean(axis=1)
        pr = Pr.argmax(-1)
        ok_r = pr == y_idx
        bw = base_wrong & ok_r              # baseline wrong -> top-r correct
        bf = (~base_wrong) & (~ok_r)        # baseline correct -> top-r wrong
        print(f"    r={r:>2}: base-wrong->topr-correct {int(bw.sum()):>3}, base-correct->topr-wrong {int(bf.sum()):>3}")

    # are failures driven by a confidently-wrong low-entropy head?
    print("\n  on queries where top-r is WRONG but baseline is RIGHT (entropy selection hurt):")
    for r in (1, 3, 5):
        Pr = HP[np.arange(N)[:, None], order[:, :r], :].mean(axis=1)
        pr = Pr.argmax(-1)
        hurt = (~base_wrong) & (pr != y_idx)
        if hurt.sum() == 0:
            print(f"    r={r}: none")
            continue
        # among the r lowest-entropy heads, how many are wrong / confidently wrong?
        sub = head_correct[np.arange(N)[:, None], order[:, :r]]          # [N, r]
        wrong_sel = (~sub)[hurt].sum(axis=1)                             # wrong heads among selected per hurt query
        low_sel = low[np.arange(N)[:, None], order[:, :r]]               # low-entropy flags among selected
        cw_sel = (low_sel & (~sub))[hurt].sum(axis=1)                    # confidently-wrong among selected
        # top-1 head (lowest entropy) behaviour on hurt queries
        top1_ok = head_correct[np.arange(N), order[:, 0]][hurt]
        top1_U = U[np.arange(N), order[:, 0]][hurt]
        print(f"    r={r}: {int(hurt.sum())} hurt queries | selected-heads wrong {int(wrong_sel.sum())}/{int(hurt.sum()*r)} "
              f"({100*wrong_sel.mean()/r:.0f}% of selected) | confidently-wrong among selected {int(cw_sel.sum())} | "
              f"lowest-U head correct {100*top1_ok.mean():.0f}% (its mean U={top1_U.mean():.4f})")

    # ---------------- 5) summary ----------------
    print("\n================ 5. GO / NO-GO SUMMARY ================")
    print(f"  baseline = {100*acc_base:.2f}%, oracle single-head UB = {100*oracle:.2f}% (+{100*(oracle-acc_base):.2f}pp)")
    print(f"  on baseline-wrong queries, correct expert exists in {100*hc_on_wrong.mean():.1f}%")
    print(f"  entropy top-r best fixed size: see section 3")
    print(f"  confidently-wrong heads = {100*cw.mean():.2f}% of all heads")


if __name__ == "__main__":
    main(sys.argv[1])

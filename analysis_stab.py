#!/usr/bin/env python3
"""Support-stability diagnostic for EuroSAT + HEC-V (5th round).

Reads the new unc_records.npy (head_probs + stability per query x head).
Asks: can leave-one-out support stability separate confidently-wrong heads
from low-entropy-but-correct heads?

1. baseline confirmation + AUROC(stability, head_correct) vs entropy AUROC
2. correct vs wrong head stability (mean/median)
3. low-entropy correct vs low-entropy wrong (confidently-wrong) stability
4. the 35 baseline-wrong-but-has-correct-head queries: correct vs wrong stability
5. entropy x stability 4-quadrant head accuracy
"""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

L_EPS = 1e-12


def norm_entropy(P):
    P = np.clip(P, L_EPS, 1.0)
    return -(P * np.log(P)).sum(axis=-1) / np.log(P.shape[-1])


def load_records(path):
    recs = np.load(path, allow_pickle=True).tolist()
    HPs, STABs, yidxs = [], [], []
    for r in recs:
        HPs.append(r['head_probs'])
        STABs.append(r['stability'])
        yidxs.append(np.searchsorted(r['labels_unique'], r['query_label']))
    HP = np.concatenate(HPs, axis=0)          # [N, 20, 10]
    STAB = np.concatenate(STABs, axis=0)      # [N, 20]
    y_idx = np.concatenate(yidxs, axis=0)     # [N]
    return HP, STAB, y_idx, recs


def main(path):
    HP, STAB, y_idx, recs = load_records(path)
    N, topk, K = HP.shape
    print(f"# queries = {N}, selected heads = {topk}, classes = {K}, episodes = {len(recs)}")

    U = norm_entropy(HP)                       # [N, topk]
    head_pred = HP.argmax(-1)
    head_correct = head_pred == y_idx[:, None]
    P_base = HP.mean(1)
    pred_base = P_base.argmax(-1)
    acc_base = float((pred_base == y_idx).mean())

    print("\n================ 1. BASELINE & AUROC ================")
    print(f"  equal-weight baseline acc = {100*acc_base:.2f}%  ({int((pred_base==y_idx).sum())}/{N})")
    print("  per-episode:")
    for i, rr in enumerate(recs):
        pb = rr['head_probs'].mean(1).argmax(-1)
        yi = np.searchsorted(rr['labels_unique'], rr['query_label'])
        print(f"    ep{i}: {100*float((pb==yi).mean()):.2f}%  (logged acc_fixed={100*rr['acc_fixed']:.2f}%)")

    auroc_stab = roc_auc_score(head_correct.flatten().astype(int), STAB.flatten())
    auroc_ent = roc_auc_score(head_correct.flatten().astype(int), -U.flatten())
    print(f"  AUROC(stability, head_correct) = {auroc_stab:.4f}")
    print(f"  AUROC(-U_entropy, head_correct)= {auroc_ent:.4f}   [round-3 value 0.7462]")

    # ---------------- 2) correct vs wrong stability ----------------
    print("\n================ 2. STABILITY: CORRECT vs WRONG HEADS ================")
    Sc, Sw = STAB[head_correct], STAB[~head_correct]
    print(f"  correct heads: mean={Sc.mean():.4f}  median={np.median(Sc):.4f}")
    print(f"  wrong   heads: mean={Sw.mean():.4f}  median={np.median(Sw):.4f}")

    # ---------------- 3) confidently-wrong vs low-entropy correct ----------------
    print("\n================ 3. LOW-ENTROPY: CORRECT vs WRONG (CONFIDENTLY-WRONG) ================")
    umed = np.median(U, axis=1, keepdims=True)
    low = U < umed
    low_ok = low & head_correct
    cw = low & ~head_correct
    high_ok = (~low) & head_correct
    high_wr = (~low) & ~head_correct
    for name, m in [("low-entropy & correct", low_ok),
                    ("low-entropy & wrong (confidently-wrong)", cw),
                    ("high-entropy & correct", high_ok),
                    ("high-entropy & wrong", high_wr)]:
        s = STAB[m]
        if s.size:
            print(f"  {name:38s}: n={int(s.size):5d}  stab mean={s.mean():.4f}  median={np.median(s):.4f}")
    print(f"  -> confidently-wrong stability mean vs low-entropy-correct: "
          f"{STAB[cw].mean():.4f} vs {STAB[low_ok].mean():.4f} "
          f"(delta {STAB[cw].mean()-STAB[low_ok].mean():+.4f})")

    # ---------------- 4) the 35 baseline-wrong queries with a correct head ----------------
    print("\n================ 4. BASELINE-WRONG BUT HAS-CORRECT-HEAD QUERIES ================")
    has_correct = head_correct.any(1)
    base_wrong = pred_base != y_idx
    sel = base_wrong & has_correct
    print(f"  queries: baseline-wrong={int(base_wrong.sum())}, with >=1 correct head={int(sel.sum())}")
    if sel.sum() > 0:
        S_corr = STAB[sel][head_correct[sel]]
        S_wrong = STAB[sel][~head_correct[sel]]
        S_cw = STAB[sel][low[sel] & ~head_correct[sel]]
        print(f"  in these queries -> correct heads:    stab mean={S_corr.mean():.4f} median={np.median(S_corr):.4f} (n={S_corr.size})")
        print(f"  in these queries -> wrong heads:      stab mean={S_wrong.mean():.4f} median={np.median(S_wrong):.4f} (n={S_wrong.size})")
        if S_cw.size:
            print(f"  in these queries -> confidently-wrong: stab mean={S_cw.mean():.4f} median={np.median(S_cw):.4f} (n={S_cw.size})")
        # do these queries contain confidently-wrong heads?
        has_cw = (low[sel] & ~head_correct[sel]).any(1)
        print(f"  queries among them containing a confidently-wrong head: {int(has_cw.sum())}/{int(sel.sum())}")

    # ---------------- 5) entropy x stability quadrants ----------------
    print("\n================ 5. ENTROPY x STABILITY 4-QUADRANT HEAD ACCURACY ================")
    smed = np.median(STAB, axis=1, keepdims=True)
    lowE_highS = low & (STAB >= smed)
    lowE_lowS = low & (STAB < smed)
    highE_highS = (~low) & (STAB >= smed)
    highE_lowS = (~low) & (STAB < smed)
    for name, m in [("low entropy + high stability", lowE_highS),
                    ("low entropy + low stability", lowE_lowS),
                    ("high entropy + high stability", highE_highS),
                    ("high entropy + low stability", highE_lowS)]:
        n = int(m.sum())
        acc = head_correct[m].mean() if n else float('nan')
        print(f"  {name:28s}: n={n:5d}  head acc={100*acc:6.1f}%  stab mean={STAB[m].mean():.4f}")


if __name__ == "__main__":
    main(sys.argv[1])

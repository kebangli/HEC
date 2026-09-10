#!/usr/bin/env python3
"""Paired comparison: original HEC-V vs Multi-Prototype HEC-V (7th round, OxfordPets).

Input: unc_records.npy from the HEC_V_pets_multiprototype run, containing per
episode (same episodes / features / selected heads for both variants):
    head_probs          [Nq, topk, K]  baseline per-head GDA+softmax probs
    head_probs_mp       [Nq, topk, K]  multi-prototype per-head probs
    head_preds_base     [Nq, topk]     baseline per-head argmax
    head_preds_mp       [Nq, topk]     multi-prototype per-head argmax
    pred_prob_base      [Nq, K]        baseline equal-weight ensemble
    pred_prob_mp        [Nq, K]        multi-prototype equal-weight ensemble
    query_label         [Nq]
    labels_unique       [K]
    acc_fixed / acc_mp                 logged accuracies

Outputs: overall + per-episode accuracies, dAcc, flip counts, selected-head
mean classification-accuracy change, and the numbers behind the verdict.
"""
import sys
import numpy as np


def load(path):
    recs = np.load(path, allow_pickle=True).tolist()
    return recs


def main(path):
    recs = load(path)
    print(f"# episodes = {len(recs)}\n")

    # ---------------- self-check + aggregate predictions ----------------
    all_base, all_mp, all_y = [], [], []
    base_hits, mp_hits, both, neither = [], [], [], []
    head_acc_base = []   # per-episode per-head accuracy (topk,) each
    head_acc_mp = []
    ep_rows = []
    for i, r in enumerate(recs):
        lu = r['labels_unique']
        y_idx = np.searchsorted(lu, r['query_label'])
        pb = r['pred_prob_base'].argmax(-1)          # [Nq]
        pm = r['pred_prob_mp'].argmax(-1)            # [Nq]
        # cross-check against logged mapped predictions
        assert np.array_equal(pb, np.searchsorted(lu, r['query_label_pred'])), f"ep{i} base mismatch"
        assert np.array_equal(pm, np.searchsorted(lu, r['query_label_pred_mp'])), f"ep{i} mp mismatch"
        a, c = float((pb == y_idx).mean()), float((pm == y_idx).mean())
        ep_rows.append((a, c, c - a))
        all_base.append(pb); all_mp.append(pm); all_y.append(y_idx)

        # per-head classification accuracy (mean over queries per selected head)
        hb = (r['head_preds_base'] == y_idx[:, None]).mean(0)   # [topk]
        hm = (r['head_preds_mp'] == y_idx[:, None]).mean(0)     # [topk]
        head_acc_base.append(hb); head_acc_mp.append(hm)
        base_hits.append((pb == y_idx)); mp_hits.append((pm == y_idx))

    base = np.concatenate(all_base); mp = np.concatenate(all_mp); y = np.concatenate(all_y)
    N = len(base)
    acc_base = float((base == y).mean())
    acc_mp = float((mp == y).mean())

    print("=========== 1. OVERALL ACCURACY (paired, same episodes) ===========")
    print(f"  Original HEC-V (baseline):        {100*acc_base:.2f}%  ({int((base==y).sum())}/{N})")
    print(f"  Multi-Prototype HEC-V:            {100*acc_mp:.2f}%  ({int((mp==y).sum())}/{N})")
    print(f"  Delta Acc: {100*(acc_mp-acc_base):+.2f} pp")

    print("\n=========== 2. PER-EPISODE ACCURACY ===========")
    print(f"  {'ep':<4}{'baseline':>10}{'multi-proto':>12}{'delta':>8}")
    for i, (a, c, d) in enumerate(ep_rows):
        print(f"  {i:<4}{100*a:>9.2f}%{100*c:>11.2f}%{100*d:>+7.2f}")

    print("\n=========== 3. FLIPS ===========")
    bw = (base != y) & (mp == y)     # baseline wrong -> MP correct
    bc = (base == y) & (mp != y)     # baseline correct -> MP wrong
    print(f"  baseline wrong -> multi-proto correct: {int(bw.sum())}")
    print(f"  baseline correct -> multi-proto wrong:  {int(bc.sum())}")
    print(f"  net = {int(bw.sum()) - int(bc.sum())}")

    print("\n=========== 4. SELECTED-HEAD MEAN CLASSIFICATION ACCURACY ===========")
    HAB = np.concatenate(head_acc_base)   # [5*topk]
    HAM = np.concatenate(head_acc_mp)
    m_base, m_mp = float(HAB.mean()), float(HAM.mean())
    print(f"  per-head acc  baseline {100*m_base:.2f}%  |  multi-proto {100*m_mp:.2f}%  |  delta {100*(m_mp-m_base):+.2f} pp  (over {len(HAB)} head-instances)")
    n_heads_better = int((HAM > HAB).sum()); n_heads_worse = int((HAM < HAB).sum()); n_tie = int((HAM == HAB).sum())
    print(f"  heads improved / degraded / unchanged: {n_heads_better} / {n_heads_worse} / {n_tie}")
    # per-episode per-head means
    print("  per-episode selected-head mean acc:")
    for i, (hba, hma) in enumerate(zip(head_acc_base, head_acc_mp)):
        print(f"    ep{i}: baseline {100*hba.mean():6.2f}%  multi-proto {100*hma.mean():6.2f}%  delta {100*(hma.mean()-hba.mean()):+.2f}")

    # ---- supporting diagnostics for the verdict ----
    print("\n=========== 5. DIAGNOSTICS (supporting the verdict) ===========")
    # fraction of queries already predicted correctly by the single class-mean GDA
    # head-level: among selected-head instances, what fraction flip
    HB = np.concatenate([r['head_preds_base'] for r in recs])
    HM = np.concatenate([r['head_preds_mp'] for r in recs])
    YY = np.concatenate([np.searchsorted(r['labels_unique'], r['query_label'])[:, None] for r in recs])
    head_corr_base = HB == YY
    head_corr_mp = HM == YY
    print(f"  head-level: correct-head fraction  baseline {100*head_corr_base.mean():.2f}%  ->  MP {100*head_corr_mp.mean():.2f}%")
    # MP per-head probs sharpness vs baseline (mean max prob)
    zbase = np.concatenate([r['head_probs'] for r in recs])
    zmp = np.concatenate([r['head_probs_mp'] for r in recs])
    print(f"  per-head mean max prob: baseline {zbase.max(-1).mean():.3f}  MP {zmp.max(-1).mean():.3f}")
    # per-query ensemble sharpness
    pbase = np.concatenate([r['pred_prob_base'] for r in recs])
    pmp = np.concatenate([r['pred_prob_mp'] for r in recs])
    print(f"  ensemble mean max prob: baseline {pbase.max(-1).mean():.3f}  MP {pmp.max(-1).mean():.3f}")
    # on baseline-correct vs baseline-wrong queries, how often does MP change the argmax
    changed = mp != base
    print(f"  queries where MP changed the prediction: {int(changed.sum())}/{N} ({100*changed.mean():.1f}%)")

    print("\n=========== SUMMARY ===========")
    print(f"  Overall: baseline {100*acc_base:.2f}% -> MP {100*acc_mp:.2f}%  (dAcc {100*(acc_mp-acc_base):+.2f} pp)")
    print(f"  Flips: +{int(bw.sum())} / -{int(bc.sum())} (net {int(bw.sum())-int(bc.sum())})")


if __name__ == "__main__":
    main(sys.argv[1])

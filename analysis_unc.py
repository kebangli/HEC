#!/usr/bin/env python3
"""Offline analysis of the EuroSAT + HEC-VT uncertainty experiment.

Loads unc_records.npy (produced by eval_unc.py) and computes:
  1. Baseline (fixed-weight) vs uncertainty-aware (reliability-weighted) fusion accuracy
  2. Relationship between per-branch normalized entropy U and prediction correctness
  3. On V/T disagreement: does the reliability weight favour the correct branch?
"""
import sys
import numpy as np

EPS = 1e-8


def load_records(path):
    recs = np.load(path, allow_pickle=True).tolist()
    PV, PT, y, alph = [], [], [], []
    for r in recs:
        PV.append(r['pred_prob_V'])
        PT.append(r['pred_prob_T'])
        y.append(r['query_label'])
        alph.append(r['alpha'])
    return (np.concatenate(PV), np.concatenate(PT), np.concatenate(y),
            np.asarray(alph), recs)


def normalized_entropy(P):
    """U = -sum_c p_c log p_c / log C (per row)."""
    P = np.clip(P, 1e-12, 1.0)
    C = P.shape[-1]
    return -(P * np.log(P)).sum(axis=-1) / np.log(C)


def fixed_fusion_acc(PV, PT, alpha):
    pred = np.argmax(PT + alpha * PV, axis=-1)
    return pred


def dynamic_fusion_acc(PV, PT, y):
    UV, UT = normalized_entropy(PV), normalized_entropy(PT)
    rV, rT = 1.0 - UV, 1.0 - UT
    Pdyn = (rV[:, None] * PV + rT[:, None] * PT) / (rV + rT + EPS)[:, None]
    pred = np.argmax(Pdyn, axis=-1)
    return pred, UV, UT, rV, rT


def main(path):
    PV, PT, y, alpha_arr, recs = load_records(path)
    N, K = PV.shape
    print(f"# queries = {N},  # episodes = {len(recs)},  K = {K}")
    print(f"alphas recorded per episode: {np.unique(alpha_arr)}")

    # ---------------- baseline (fixed) & dynamic ----------------
    # per-episode alpha (should be constant after tuning; alpha=0.1)
    alpha = alpha_arr[0]
    pred_fixed = fixed_fusion_acc(PV, PT, alpha)
    pred_dyn, UV, UT, rV, rT = dynamic_fusion_acc(PV, PT, y)

    acc_fixed = float((pred_fixed == y).mean())
    acc_dyn = float((pred_dyn == y).mean())
    print("\n================ ACCURACY ================")
    print(f"  Baseline HEC-VT (fixed, alpha={alpha}):        {100*acc_fixed:.2f}%  ({int((pred_fixed==y).sum())}/{N})")
    print(f"  HEC-VT + uncertainty-aware dynamic fusion:     {100*acc_dyn:.2f}%  ({int((pred_dyn==y).sum())}/{N})")
    print(f"  Delta: {100*(acc_dyn-acc_fixed):+.2f} pp")

    # cross-check per-episode vs logged acc_fixed
    print("\n  per-episode cross-check (eval log accuracy vs recomputed):")
    for i, r in enumerate(recs):
        nq = r['pred_prob_V'].shape[0]
        ep_fixed = (np.argmax(r['pred_prob_T'] + r['alpha']*r['pred_prob_V'], axis=-1) == r['query_label']).mean()
        print(f"    ep{i}: log={100*r['acc_fixed']:.2f}%  recomputed_fixed={100*ep_fixed:.2f}%")

    # ---------------- uncertainty diagnostics ----------------
    print("\n================ UNCERTAINTY DIAGNOSTICS ================")
    for name, U, P in [("HEC-V", UV, PV), ("HEC-T", UT, PT)]:
        pmax = P.max(axis=-1)
        print(f"  {name}: U[min/p10/med/p90/max] = "
              f"{U.min():.4f}/{np.percentile(U,10):.4f}/{np.median(U):.4f}/"
              f"{np.percentile(U,90):.4f}/{U.max():.4f}")
        print(f"       max-class prob[min/p10/med/p90/max] = "
              f"{pmax.min():.4f}/{np.percentile(pmax,10):.4f}/{np.median(pmax):.4f}/"
              f"{np.percentile(pmax,90):.4f}/{pmax.max():.4f}")
    wV_all = rV / (rV + rT + EPS)
    print(f"  dynamic weight on V (w_V): fraction r_T > r_V = {(rT > rV).mean()*100:.1f}%, "
          f"w_V median={np.median(wV_all):.4f} min={wV_all.min():.4f}")
    # dynamic fusion almost == "use only V" when r_T ~ 0
    acc_V_only = float((np.argmax(PV, axis=-1) == y).mean())
    print(f"  reference: HEC-V alone acc = {100*acc_V_only:.2f}%")

    # ---------------- U vs correctness ----------------
    predV = np.argmax(PV, axis=-1)
    predT = np.argmax(PT, axis=-1)
    cV = predV == y
    cT = predT == y

    print("\n================ ENTROPY vs CORRECTNESS ================")
    for name, U, r, correct in [("HEC-V", UV, rV, cV), ("HEC-T", UT, rT, cT)]:
        m_c, m_w = U[correct].mean(), U[~correct].mean()
        r_c, r_w = r[correct].mean(), r[~correct].mean()
        # point-biserial correlation between -U (reliability) and correctness
        rb = np.corrcoef(-U, correct.astype(float))[0, 1]
        print(f"  {name}: acc={100*correct.mean():5.1f}%  "
              f"mean U(correct)={m_c:.4f}  mean U(wrong)={m_w:.4f}  "
              f"mean r(correct)={r_c:.4f}  mean r(wrong)={r_w:.4f}  "
              f"corr(-U,correct)={rb:+.3f}")

    # ---------------- disagreement analysis ----------------
    disagree = predV != predT
    print("\n================ DISAGREEMENT (pred_V != pred_T) ================")
    print(f"  disagreement rate: {100*disagree.mean():.1f}%  ({disagree.sum()}/{N})")
    if disagree.sum() == 0:
        print("  no disagreements")
        return

    dV = cV & ~cT       # V correct, T wrong
    dT = ~cV & cT       # V wrong, T correct
    dB = ~cV & ~cT      # both wrong
    print(f"  among disagree: V-correct/T-wrong={dV.sum()} ({100*dV.mean():.0f}%), "
          f"V-wrong/T-correct={dT.sum()} ({100*dT.mean():.0f}%), "
          f"both-wrong={dB.sum()} ({100*dB.mean():.0f}%)")

    # correct-branch-gets-higher-r
    ok_r = (dV & (rV > rT)) | (dT & (rT > rV)) | (dB & (rV != rT))
    correct_branch = dV | dT
    good_weight = (dV & (rV > rT)) | (dT & (rT > rV))
    print(f"  fraction where the CORRECT branch has higher reliability: "
          f"{100*good_weight.sum()/max(correct_branch.sum(),1):.1f}% "
          f"({good_weight.sum()}/{correct_branch.sum()} of resolvable cases, both-wrong excluded)")

    # accuracy on disagreement subset: fixed vs dynamic
    acc_fixed_d = float((pred_fixed[disagree] == y[disagree]).mean())
    acc_dyn_d = float((pred_dyn[disagree] == y[disagree]).mean())
    print(f"  accuracy on disagreement subset: fixed={100*acc_fixed_d:.2f}%  dynamic={100*acc_dyn_d:.2f}%  (delta {100*(acc_dyn_d-acc_fixed_d):+.2f} pp)")

    # how many queries flip label between fixed and dynamic, and net effect
    flip = pred_fixed != pred_dyn
    flip_fix2wrong = (flip & (pred_fixed == y) & (pred_dyn != y)).sum()
    flip_wrong2fix = (flip & (pred_fixed != y) & (pred_dyn == y)).sum()
    print(f"  label flips fixed->dynamic: {flip.sum()} (correct->wrong: {flip_fix2wrong}, wrong->correct: {flip_wrong2fix})")

    # correlation: weight on correct branch vs being correct (V-portion of weight)
    wV = rV / (rV + rT + EPS)
    print(f"\n  mean weight on V (w_V = r_V/(r_V+r_T)): overall={wV.mean():.3f}, "
          f"when V correct={wV[cV].mean():.3f}, when V wrong={wV[~cV].mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1])

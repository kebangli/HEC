#!/usr/bin/env python3
"""Offline validation of branch-wise calibrated uncertainty fusion (EuroSAT + HEC-VT).

Data:
  - test:  unc_records.npy      (5 episodes, 250 queries) -- per-query P_V, P_T, label
  - tuning: unc_tuning_best.npy (best-config tuning episode) -- mu/sigma estimation ONLY

Pipeline:
  1. AUROC(-U, prediction_correct) for HEC-V and HEC-T on TEST data.
  2. Branch-wise z-score calibration:
         z_b = (U_b - mu_b) / sigma_b ,  (mu_b, sigma_b) from TUNING data only
         [w_V, w_T] = softmax(-z_V, -z_T)          (temperature = 1, no tuning)
         P = w_V P_V + w_T P_T
  3. Compare: original HEC-VT (fixed alpha) vs calibrated dynamic fusion.
  4. Disagreement analysis + "V-wrong / T-correct" correction statistics.
"""
import sys
import numpy as np
from sklearn.metrics import roc_auc_score

EPS = 1e-8
L_EPS = 1e-12


def norm_entropy(P):
    P = np.clip(P, L_EPS, 1.0)
    return -(P * np.log(P)).sum(axis=-1) / np.log(P.shape[-1])


def load_test(path):
    recs = np.load(path, allow_pickle=True).tolist()
    PV, PT, y, alph = [], [], [], []
    for r in recs:
        PV.append(r['pred_prob_V'])
        PT.append(r['pred_prob_T'])
        y.append(r['query_label'])
        alph.append(r['alpha'])
    return (np.concatenate(PV), np.concatenate(PT), np.concatenate(y),
            np.asarray(alph), recs)


def load_tuning(path):
    recs = np.load(path, allow_pickle=True).tolist()
    assert len(recs) == 1, "expected one tuning record (best config)"
    r = recs[0]
    return r['pred_prob_V'], r['pred_prob_T'], r['query_label']


def main(test_path, tuning_path):
    PV, PT, y, alpha_arr, recs = load_test(test_path)
    N, K = PV.shape
    alpha = alpha_arr[0]
    print(f"# test queries = {N}, episodes = {len(recs)}, K = {K}, alpha(best) = {alpha}")

    # branch correctness
    predV, predT = np.argmax(PV, axis=-1), np.argmax(PT, axis=-1)
    cV, cT = predV == y, predT == y

    # ---------------- 1) AUROC(-U, correct) on TEST ----------------
    UV, UT = norm_entropy(PV), norm_entropy(PT)
    print("\n================ 1. UNCERTAINTY RELIABILITY (AUROC on test) ================")
    for name, U, c in [("HEC-V", UV, cV), ("HEC-T", UT, cT)]:
        a = roc_auc_score(c.astype(int), -U)
        print(f"  {name}: acc={100*c.mean():.1f}%  AUROC(-U, correct) = {a:.4f}  "
              f"(mean U correct={U[c].mean():.4f} wrong={U[~c].mean():.4f})")

    # ---------------- 2) tuning-only calibration ----------------
    print("\n================ 2. BRANCH-WISE CALIBRATION (tuning-only) ================")
    tPV, tPT, ty = load_tuning(tuning_path)
    UtV, UtT = norm_entropy(tPV), norm_entropy(tPT)
    muV, muT = UtV.mean(), UtT.mean()
    sV, sT = UtV.std(ddof=1), UtT.std(ddof=1)
    sV, sT = max(sV, 1e-9), max(sT, 1e-9)
    print(f"  tuning mu/sigma:  V: mu={muV:.4f} sigma={sV:.4f}  |  T: mu={muT:.4f} sigma={sT:.4f}")
    print(f"  (tuning n={tPV.shape[0]}, T sigma={sT:.2e} -> U_T is near-degenerate)")

    zV = (UV - muV) / sV
    zT = (UT - muT) / sT
    logits = np.stack([-zV, -zT], axis=1)                     # [N, 2]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    W = e / e.sum(axis=1, keepdims=True)                      # [N, 2]
    wV, wT = W[:, 0], W[:, 1]
    print(f"  calibrated weights: w_V med={np.median(wV):.3f} min={wV.min():.3f} max={wV.max():.3f} | "
          f"w_T med={np.median(wT):.3f}  (w_V+w_T=1)")

    # ---------------- 3) accuracy comparison ----------------
    print("\n================ 3. ACCURACY ================")
    pred_fixed = np.argmax(PT + alpha * PV, axis=-1)
    acc_fixed = float((pred_fixed == y).mean())

    rV, rT = 1 - UV, 1 - UT                                  # previous raw method (reference)
    pred_raw = np.argmax((rV[:, None] * PV + rT[:, None] * PT) / (rV + rT + EPS)[:, None], axis=-1)
    acc_raw = float((pred_raw == y).mean())

    Pcal = wV[:, None] * PV + wT[:, None] * PT
    pred_cal = np.argmax(Pcal, axis=-1)
    acc_cal = float((pred_cal == y).mean())

    print(f"  Original HEC-VT (fixed, alpha={alpha}):            {100*acc_fixed:.2f}%  ({int((pred_fixed==y).sum())}/{N})")
    print(f"  [ref] raw r=1-U dynamic fusion (prev round):        {100*acc_raw:.2f}%  (delta {100*(acc_raw-acc_fixed):+.2f})")
    print(f"  Calibrated uncertainty dynamic fusion (softmax):   {100*acc_cal:.2f}%  (delta {100*(acc_cal-acc_fixed):+.2f})")

    # per-episode for calibrated
    print("  per-episode calibrated accuracy:")
    for i, r in enumerate(recs):
        uv, ut = norm_entropy(r['pred_prob_V']), norm_entropy(r['pred_prob_T'])
        z1, z2 = (uv - muV) / sV, (ut - muT) / sT
        lg = np.stack([-z1, -z2], 1)
        ee = np.exp(lg - lg.max(1, keepdims=True))
        ww = ee / ee.sum(1, keepdims=True)
        pp = ww[:, 0, None] * r['pred_prob_V'] + ww[:, 1, None] * r['pred_prob_T']
        ep = float((np.argmax(pp, -1) == r['query_label']).mean())
        print(f"    ep{i}: {100*ep:.2f}%")

    # ---------------- 4) disagreement analysis ----------------
    print("\n================ 4. DISAGREEMENT ANALYSIS (calibrated weights) ================")
    disagree = predV != predT
    dV = cV & ~cT          # V correct, T wrong
    dT = ~cV & cT          # V wrong, T correct
    dB = ~cV & ~cT         # both wrong
    print(f"  disagreement rate: {100*disagree.mean():.1f}% ({disagree.sum()}/{N})")
    print(f"    V-ok/T-wrong={dV.sum()}, V-wrong/T-ok={dT.sum()}, both-wrong={dB.sum()}")
    resolvable = (dV | dT).sum()
    ok_weight = (dV & (wV > wT)) | (dT & (wT > wV))
    print(f"  fraction where CORRECT branch gets higher weight (calibrated): "
          f"{100*ok_weight.sum()/resolvable:.1f}% ({ok_weight.sum()}/{resolvable})")
    print(f"  [ref] raw r=1-U: {( (dV & (rV>rT)) | (dT & (rT>rV)) ).sum()}/{resolvable}")

    # disagreement subset accuracy
    a_fixed_d = float((pred_fixed[disagree] == y[disagree]).mean())
    a_cal_d = float((pred_cal[disagree] == y[disagree]).mean())
    print(f"  acc on disagreement subset: fixed={100*a_fixed_d:.2f}%  calibrated={100*a_cal_d:.2f}%")

    # ---------------- 5) V-wrong / T-ok corrections ----------------
    print("\n================ 5. 'V-WRONG / T-OK' CORRECTION STATS ================")
    n_dT = int(dT.sum())
    wT_baseline = 1.0 / (1.0 + alpha)            # effective T weight in fixed fusion (1 : alpha)
    wt_up = (wT[dT] > wT_baseline).sum()         # calibrated increased T weight vs baseline
    wt_pref = (wT[dT] > wV[dT]).sum()            # calibrated prefers T over V
    base_corr = int((pred_fixed[dT] == y[dT]).sum())
    raw_corr = int((pred_raw[dT] == y[dT]).sum())
    cal_corr = int((pred_cal[dT] == y[dT]).sum())
    cal_corr_and_wt_up = int(((wT[dT] > wT_baseline) & (pred_cal[dT] == y[dT])).sum()) if n_dT else 0
    print(f"  # V-wrong/T-ok cases: {n_dT}")
    print(f"  calibrated w_T > w_T_baseline({wT_baseline:.2f}): {wt_up}/{n_dT}")
    print(f"  calibrated w_T > w_V:                {wt_pref}/{n_dT}")
    print(f"  correct among them -> fixed={base_corr}, raw r={raw_corr}, calibrated={cal_corr}")
    print(f"  calibrated-correct AND w_T raised:   {cal_corr_and_wt_up}/{n_dT}")

    # overall flip summary
    flip = pred_fixed != pred_cal
    f2w = int((flip & (pred_fixed == y) & (pred_cal != y)).sum())
    w2f = int((flip & (pred_fixed != y) & (pred_cal == y)).sum())
    print(f"\n  flips fixed->calibrated: {int(flip.sum())} (correct->wrong {f2w}, wrong->correct {w2f})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

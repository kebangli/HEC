#!/usr/bin/env python3
"""Analysis of the EuroSAT + HEC-V noisy-label experiment.

Loads noisy_label_records.npy produced by exp_noisy_label.py and reports:
  1. accuracy   : clean vs noise25 vs noise50 (mean/std, per-episode, paired stats)
  2. topk set   : top-k overlap rate (|∩|/k) and Jaccard between clean and noisy
  3. ranking    : Spearman / Kendall of the full head-score ranking, plus
                  top-k head rank displacement
"""
import sys
import numpy as np
from scipy import stats


def load(path):
    return np.load(path, allow_pickle=True).tolist()


def main(path, k=20):
    recs = load(path)
    n_ep = len(recs)
    noise_conds = sorted([k_ for k_ in recs[0].keys() if k_.startswith("noise")],
                         key=lambda s: int(s[5:]))
    conds = ["clean"] + noise_conds
    print(f"# episodes = {n_ep}, k = {k}")
    print(f"head space: L={recs[0]['L']}, H={recs[0]['H']}, M={recs[0]['M']}")

    # ---------- 1. ACCURACY ----------
    acc = {c: np.array([r[c]["accuracy"] for r in recs]) for c in conds}
    print("\n================ 1. ACCURACY ================")
    print(f"{'cond':9s} {'mean':>7s} {'std':>7s} {'median':>7s}  per-episode")
    for c in conds:
        a = acc[c]
        print(f"{c:9s} {100*a.mean():6.2f}% {100*a.std():6.2f} {100*np.median(a):6.2f}%  "
              + " ".join(f"{100*x:.0f}" for x in a))
    # paired comparisons (clean vs each noise + consecutive noise pairs)
    print("\n  paired (per-episode) tests:")
    pairs = [("clean", c) for c in noise_conds] + \
            [(noise_conds[i], noise_conds[i + 1]) for i in range(len(noise_conds) - 1)]
    for pair in pairs:
        a, b = acc[pair[0]], acc[pair[1]]
        d = b - a
        try:
            w = stats.wilcoxon(a, b, alternative="two-sided")
            p_w = w.pvalue
        except Exception:
            p_w = float("nan")
        t = stats.ttest_rel(a, b)
        print(f"    {pair[0]:8s}->{pair[1]:8s}: mean_delta={100*d.mean():+6.2f}pt  "
              f"Wilcoxon_p={p_w:.4f}  ttest_p={t.pvalue:.4f}")

    # ---------- 2. TOP-K SET OVERLAP ----------
    print("\n================ 2. TOP-K SET OVERLAP (vs clean) ================")
    for c in noise_conds:
        ov, jac = [], []
        for r in recs:
            s_clean = set(r["clean"]["selected_idx"].tolist())
            s_noise = set(r[c]["selected_idx"].tolist())
            inter = s_clean & s_noise
            ov.append(len(inter) / k)
            jac.append(len(inter) / len(s_clean | s_noise) if s_clean | s_noise else np.nan)
        ov, jac = np.array(ov), np.array(jac)
        keep = np.array([k - len(set(r["clean"]["selected_idx"].tolist()) - set(r[c]["selected_idx"].tolist())) for r in recs])
        print(f"  {c:8s}: overlap_rate(|∩|/k) mean={ov.mean():.3f} std={ov.std():.3f} "
              f"| min={ov.min():.2f} max={ov.max():.2f}")
        print(f"  {c:8s}: Jaccard          mean={np.nanmean(jac):.3f} std={np.nanstd(jac):.3f}")
        print(f"  {c:8s}: kept-in-topk     mean={keep.mean():.1f}/{k}  "
              + " ".join(f"{int(x)}" for x in keep))

    # ---------- 3. RANKING CHANGE ----------
    print("\n================ 3. RANKING CHANGE (full head-score ranking) ================")
    for c in noise_conds:
        sp, kt = [], []
        disp = []
        for r in recs:
            s_clean = r["clean"]["head_scores"]
            s_noise = r[c]["head_scores"]
            sp.append(stats.spearmanr(s_clean, s_noise).statistic)
            kt.append(stats.kendalltau(s_clean, s_noise).statistic)
            # rank of noisy top-k in clean ranking and clean top-k in noisy ranking
            rank_clean = stats.rankdata(-s_clean)
            rank_noise = stats.rankdata(-s_noise)
            sel_clean = r["clean"]["selected_idx"]
            # displacement of clean top-k heads under noisy ranking
            d = np.abs(rank_clean[sel_clean] - rank_noise[sel_clean])
            disp.append(d)
        sp, kt = np.array(sp), np.array(kt)
        disp = np.concatenate(disp)  # over all selected heads x episodes
        print(f"  {c:8s}: Spearman rho      mean={sp.mean():.4f} std={sp.std():.4f}")
        print(f"  {c:8s}: Kendall tau       mean={kt.mean():.4f} std={kt.std():.4f}")
        print(f"  {c:8s}: top-k rank displacement |Δrank| mean={disp.mean():.1f} "
              f"median={np.median(disp):.1f}  p90={np.percentile(disp,90):.0f}")

    # per-episode detail for the report table
    print("\n================ PER-EPISODE TABLE ================")
    hdr = f"{'ep':>3s} " + "".join(f"{c:>12s}" for c in conds)
    hdr += "".join(f" {c.replace('noise','ov'):>7s}" for c in noise_conds)
    print(hdr)
    for r in recs:
        row = f"{r['episode']:3d} " + "".join(
            f"{100*r[c]['accuracy']:11.1f}%" for c in conds)
        for c in noise_conds:
            ov = len(set(r["clean"]["selected_idx"].tolist()) & set(r[c]["selected_idx"].tolist())) / k
            row += f" {ov:7.2f}"
        print(row)


if __name__ == "__main__":
    main(sys.argv[1])

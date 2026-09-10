#!/usr/bin/env python3
"""Analysis of the 2^3 GDA-component factorial decomposition.

For each noise rate, over the 8 combinations (mean, cov, prior) x {c, n}:
  1. mean accuracy per combination + drop vs (c,c,c) [CleanGDA]
  2. factorial main effects:
        E[mean]   = avg(acc | mean=n) - avg(acc | mean=c)
        E[cov]    = avg(acc | cov=n)  - avg(acc | cov=c)
        E[prior]  = avg(acc | prior=n)- avg(acc | prior=prior=c)
  3. two-way / three-way interactions (standard 2^3 factorial contrasts)
  4. cross-checks vs previous experiments (same noise realizations)
"""
import sys
import numpy as np
from scipy import stats
import itertools

COMBOS = ["".join(t) for t in itertools.product("cn", repeat=3)]


def load(path):
    return np.load(path, allow_pickle=True).tolist()


def main(path):
    recs = load(path)
    n_ep = len(recs)
    print(f"# episodes = {n_ep}")
    print(f"head space: L={recs[0]['L']}, H={recs[0]['H']}, M={recs[0]['M']}")

    for rate in (25, 50):
        key = f"r{rate}"
        print("\n" + "=" * 78)
        print(f"NOISE {rate}%   (heads = CLEAN, GDA components switched)")
        print("=" * 78)

        acc = {c: np.array([rec[key]["accs"][c] for rec in recs]) for c in COMBOS}

        # ---------------- 1. accuracy per combination ----------------
        print("\n--- 1. ACCURACY PER COMBINATION (mean ± std, drop vs ccc) ---")
        print(f"  {'combo':6s} {'mean':>8s} {'std':>6s} {'drop':>8s}   per-episode")
        for c in COMBOS:
            a = acc[c]
            d = acc["ccc"] - a
            print(f"  {c:6s} {100*a.mean():6.2f}% {100*a.std():5.2f} "
                  f"{100*d.mean():+7.2f}pt  "
                  + " ".join(f"{100*x:.0f}" for x in a))

        # cross-checks: ccc == clean accuracy; nnn == 2x2 "CleanH+NoisyGDA"
        print(f"  [check] ccc (CleanGDA) = {100*acc['ccc'].mean():.2f}%"
              f"  | nnn (NoisyGDA)  = {100*acc['nnn'].mean():.2f}%")

        # ---------------- 2. main effects ----------------
        print("\n--- 2. FACTORIAL MAIN EFFECTS (positive = damage) ---")
        effects = {}
        for i, name in enumerate(["mean", "cov", "prior"]):
            on = np.mean([acc[c] for c in COMBOS if c[i] == 'n'], axis=0)
            off = np.mean([acc[c] for c in COMBOS if c[i] == 'c'], axis=0)
            eff = off - on  # positive => component pollution hurts
            effects[name] = eff
            try:
                p = stats.wilcoxon(off, on).pvalue
            except Exception:
                p = float("nan")
            print(f"  E[{name:5s}] = {100*eff.mean():+6.2f}pt  "
                  f"(std {100*eff.std():5.2f}, Wilcoxon_p={p:.4f})")

        # ---------------- 3. interactions ----------------
        print("\n--- 3. INTERACTIONS ---")

        def contrast(sel):
            """avg over combos matching sel pattern ('c'/'n'/None)"""
            mats = [acc[c] for c in COMBOS
                    if all(s is None or c[i] == s for i, s in enumerate(sel))]
            return np.mean(mats, axis=0)

        for (i, j), (ni, nj) in zip([(0, 1), (0, 2), (1, 2)],
                                    [("mean", "cov"), ("mean", "prior"), ("cov", "prior")]):
            # E[i] at j=n minus E[i] at j=c
            ei_at_jn = contrast([None if x == j else ('n' if x == i else None)
                                 for x in range(3)])
            ei_at_jc = contrast([None if x == j else ('c' if x == i else None)
                                 for x in range(3)])
            e_j_on = contrast([None if x == j else ('n' if x == i else None)
                               for x in range(3)])
            # simpler: interaction = avg(acc[both n] + acc[both c]) - avg(acc[mixed])
            a_nn = contrast([('n' if x in (i, j) else None) for x in range(3)])
            a_cc = contrast([('c' if x in (i, j) else None) for x in range(3)])
            a_nc = contrast([('n' if x == i else 'c' if x == j else None) for x in range(3)])
            a_cn = contrast([('c' if x == i else 'n' if x == j else None) for x in range(3)])
            inter = ((a_nn + a_cc) - (a_nc + a_cn)) / 2.0
            print(f"  I[{ni:5s} x {nj:5s}] = {100*inter.mean():+6.2f}pt")

        t_nn = contrast(['n', 'n', 'n']) + contrast(['c', 'c', 'c'])
        t_mixed = (contrast(['n', 'n', 'c']) + contrast(['n', 'c', 'n'])
                   + contrast(['c', 'n', 'n']) + contrast(['c', 'c', 'n'])
                   + contrast(['c', 'n', 'c']) + contrast(['n', 'c', 'c']))
        # standard 3-way: E[mean|cov=n,prior=n] - E[mean|cov=n,prior=c] - ...
        e1 = contrast(['n', 'n', 'n']) - contrast(['n', 'n', 'c'])
        e2 = contrast(['n', 'c', 'n']) - contrast(['n', 'c', 'c'])
        e3 = contrast(['c', 'n', 'n']) - contrast(['c', 'n', 'c'])
        e4 = contrast(['c', 'c', 'n']) - contrast(['c', 'c', 'c'])
        three = ((e1 - e2) + (e3 - e4)) / 2.0
        print(f"  I[mean x cov x prior]  = {100*three.mean():+6.2f}pt")

        # ---------------- 4. per-episode table ----------------
        print("\n--- 4. PER-EPISODE (8 combos) ---")
        print("  ep | " + " ".join(f"{c:>5s}" for c in COMBOS))
        for rec in recs:
            a = rec[key]["accs"]
            print(f"  {rec['episode']:2d} | "
                  + " ".join(f"{100*a[c]:5.1f}" for c in COMBOS))


if __name__ == "__main__":
    main(sys.argv[1])

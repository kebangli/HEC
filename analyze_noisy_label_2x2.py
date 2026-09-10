#!/usr/bin/env python3
"""Analysis of the 2x2 mechanism decomposition (EuroSAT + HEC-V noisy-label).

Loads noisy_label_2x2_records.npy and reports, for 25% and 50% noise:
  1. the 4-condition accuracy table (mean/std/median, per-episode)
  2. paired (per-episode) tests between conditions
  3. variance decomposition of the total drop:
        T = cH_cG - nH_nG        (total drop)
        G = cH_cG - cH_nG        (drop due to GDA-parameter pollution only)
        H = cH_cG - nH_cG        (drop due to head-selection pollution only)
        I = T - G - H            (interaction)
     plus the significance of G vs H, to decide the dominant factor.
  4. cross-check: cH_cG must match the clean baseline (~79.7%),
     nH_nG must match the previous noisy run (67.8% / 42.2%).
"""
import sys
import numpy as np
from scipy import stats


def load(path):
    return np.load(path, allow_pickle=True).tolist()


def main(path, k=20):
    recs = load(path)
    n_ep = len(recs)
    print(f"# episodes = {n_ep}, k = {k}, M = {recs[0]['M']}")
    for r in recs:
        if r["episode"] == 0:
            print(f"head space: L={r['L']}, H={r['H']}, M={r['M']}")
            break

    for rate in (25, 50):
        key = f"r{rate}"
        print("\n" + "=" * 78)
        print(f"NOISE {rate}%  (每类错 {rate//25} of 4)")
        print("=" * 78)

        # ---------------- 1. accuracy table ----------------
        acc = {c: np.array([rec[key][c] for rec in recs])
               for c in ["cH_cG", "cH_nG", "nH_cG", "nH_nG"]}
        names = {
            "cH_cG": "① CleanH + CleanGDA",
            "cH_nG": "② CleanH + NoisyGDA",
            "nH_cG": "③ NoisyH + CleanGDA",
            "nH_nG": "④ NoisyH + NoisyGDA",
        }
        print("\n--- 1. ACCURACY (mean ± std, median) ---")
        for c in ["cH_cG", "cH_nG", "nH_cG", "nH_nG"]:
            a = acc[c]
            print(f"  {names[c]:20s}: {100*a.mean():6.2f}% ± {100*a.std():5.2f}  "
                  f"med={100*np.median(a):6.2f}%")

        # cross-check
        c_clean = acc["cH_cG"].mean()
        c_noisy = acc["nH_nG"].mean()
        print(f"  [cross-check] clean should ≈79.7% -> {100*c_clean:.2f}% ; "
              f"noisy{rate} should ≈{67.8 if rate==25 else 42.2}% -> {100*c_noisy:.2f}%")

        # ---------------- 2. paired tests ----------------
        print("\n--- 2. PAIRED TESTS (per-episode) ---")
        for a_, b_ in [("cH_cG", "cH_nG"), ("cH_cG", "nH_cG"),
                       ("cH_cG", "nH_nG"), ("cH_nG", "nH_nG"), ("nH_cG", "nH_nG")]:
            d = acc[b_] - acc[a_]
            try:
                pw = stats.wilcoxon(acc[a_], acc[b_]).pvalue
            except Exception:
                pw = float("nan")
            print(f"    {names[a_][:17]:17s} -> {names[b_][:17]:17s}: "
                  f"delta={100*d.mean():+6.2f}pt  Wilcoxon_p={pw:.4f}")

        # ---------------- 3. variance decomposition ----------------
        print("\n--- 3. DECOMPOSITION OF TOTAL DROP (positive = drop) ---")
        T = acc["cH_cG"] - acc["nH_nG"]   # total drop
        G = acc["cH_cG"] - acc["cH_nG"]   # GDA pollution
        H = acc["cH_cG"] - acc["nH_cG"]   # head-selection pollution
        I = T - G - H                     # interaction
        for name, v in [("T  total drop          ", T),
                        ("G  GDA-param pollution ", G),
                        ("H  head-sel pollution  ", H),
                        ("I  interaction T-G-H   ", I)]:
            print(f"  {name}: mean={100*v.mean():+6.2f}pt  std={100*v.std():5.2f}  "
                  f"median={100*np.median(v):+6.2f}  p_wilcoxon_0="
                  f"{stats.wilcoxon(v).pvalue if (v!=0).all() else float('nan'):.4f}")
        print(f"\n  relative share: |G|={100*abs(G.mean()):5.2f}pt "
              f"|H|={100*abs(H.mean()):5.2f}pt  |I|={100*abs(I.mean()):5.2f}pt")
        gs = np.sign(G.mean()); hs = np.sign(H.mean())
        print(f"  -> GDA dominates: {abs(G.mean()) > abs(H.mean())}   "
              f"(G{'+' if gs>=0 else '-'} {100*abs(G.mean()):.1f}pt vs "
              f"H{'+' if hs>=0 else '-'} {100*abs(H.mean()):.1f}pt)")

        # is G significantly larger than H?
        dGH = G - H
        try:
            p_gh = stats.wilcoxon(G, H).pvalue
        except Exception:
            p_gh = float("nan")
        print(f"  paired |G| vs |H|: mean diff = {100*dGH.mean():+.2f}pt  Wilcoxon_p={p_gh:.4f}")

        # ---------------- 4. per-episode table ----------------
        print("\n--- 4. PER-EPISODE ---")
        print("  ep |   cHcG    cHnG    nHcG    nHnG  | dropG  dropH  dropT")
        for rec in recs:
            a = rec[key]
            vals = [a[c] for c in ["cH_cG", "cH_nG", "nH_cG", "nH_nG"]]
            g = vals[0] - vals[1]; h = vals[0] - vals[2]; t = vals[0] - vals[3]
            print(f"  {rec['episode']:2d} | "
                  + " ".join(f"{100*v:5.1f}" for v in vals)
                  + f"  | {100*g:+5.1f} {100*h:+5.1f} {100*t:+5.1f}")


if __name__ == "__main__":
    main(sys.argv[1])

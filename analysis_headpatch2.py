#!/usr/bin/env python3
"""Supplementary Head-Patch analysis: spatial focus distribution + heatmaps for
all wrong subset queries and a few correct ones.

Quantifies whether the 20 selected heads *spatially scatter* (centroid / peak
cell / coverage), and renders per-query heatmap panels.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NIMG, GRID = 64, 8
N_MASK = 6


def load(path):
    return np.load(path, allow_pickle=True).tolist()


def main(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    recs = load(path)
    A = np.concatenate([r["attn_maps"] for r in recs])       # [Qtot, topk, 64]
    Qtot, topk = A.shape[0], A.shape[1]

    # ---------------- spatial focus distribution ----------------
    print("================ SPATIAL FOCUS OF THE 20 SELECTED HEADS ================")
    # per-head mean map over all subset queries
    mean_map = A.mean(0)                                     # [topk, 64]
    yy, xx = np.mgrid[0:GRID, 0:GRID]
    cents = []
    for h in range(topk):
        m = mean_map[h].reshape(GRID, GRID)
        mass = m.sum()
        cy = (yy * m).sum() / (mass + 1e-12)
        cx = (xx * m).sum() / (mass + 1e-12)
        cents.append((cy, cx))
    cents = np.array(cents)
    peaks = np.array([np.unravel_index(np.argmax(mean_map[h]), (GRID, GRID)) for h in range(topk)])
    print(f"  head centroid (row,col) over 8x8 grid:")
    for h in range(topk):
        print(f"    h{h:2d}: centroid=({cents[h,0]:.1f},{cents[h,1]:.1f})  peak=({peaks[h,0]},{peaks[h,1]})  "
              f"concentration={float((mean_map[h]**2).sum()):.3f}")
    cy_std, cx_std = cents[:, 0].std(), cents[:, 1].std()
    print(f"  centroid scatter: row std={cy_std:.2f}  col std={cx_std:.2f}")
    # how many distinct peak cells among the 20 heads?
    print(f"  distinct peak cells among 20 heads: {len(set(map(tuple, peaks.tolist())))} / 64")

    # coverage: per query, union of top-6 sets over 20 heads
    uniq = []
    for q in range(Qtot):
        s = set()
        for h in range(topk):
            s |= set(np.argsort(A[q, h])[::-1][:N_MASK].tolist())
        uniq.append(len(s))
    uniq = np.array(uniq)
    print(f"  per-query union of top-6 patches across 20 heads: mean {uniq.mean():.1f}  min {uniq.min()}  max {uniq.max()}  (of 64)")

    # ---------------- heatmaps: every wrong query + sample corrects ----------------
    print("\n================ HEATMAP PANELS ================")
    wrong, correct = [], []
    for ep, r in enumerate(recs):
        for qi in range(r["attn_maps"].shape[0]):
            (wrong if not bool(r["ens_correct"][qi]) else correct).append((ep, qi))
    print(f"  wrong subset queries: {wrong}  (n={len(wrong)});  correct: n={len(correct)}")

    def render(ep, qi, tag):
        r = recs[ep]
        img = r["query_imgs_sub"][qi].transpose(1, 2, 0)
        maps = r["attn_maps"][qi]
        fig, axes = plt.subplots(5, 4, figsize=(15, 17))
        for h in range(topk):
            ax = axes[h // 4, h % 4]
            ax.imshow(img)
            m = maps[h].reshape(GRID, GRID)
            m_up = np.kron(m, np.ones((28, 28)))
            ax.imshow(m_up / (m_up.max() + 1e-9), cmap="jet", alpha=0.55)
            ax.set_axis_off(); ax.set_title(f"h{h} c={int(r['head_correct'][qi, h])}", fontsize=7)
        fig.suptitle(f"ep{ep} q{qi} [{tag}]  true={int(r['query_label_sub'][qi])}  "
                     f"ens_pred={int(r['labels_unique'][np.argmax(r['head_probs'][qi].mean(0))])} "
                     f"ens_correct={bool(r['ens_correct'][qi])}", fontsize=11)
        out = os.path.join(outdir, f"panel_ep{ep}_q{qi}_{tag}.png")
        fig.savefig(out, dpi=80, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out}")

    for ep, qi in wrong:
        render(ep, qi, "wrong")
    for ep, qi in correct[:3]:
        render(ep, qi, "correct")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "headpatch_figs")

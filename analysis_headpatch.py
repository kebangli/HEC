#!/usr/bin/env python3
"""Head-Patch diagnostic analysis (round 8, OxfordPets + HEC-V).

Input: unc_records.npy from the HEC_V_pets_headpatch run.  For a 12-query
subset per episode and the 20 selected heads:
    attn_maps    [Q, topk, 64]  last-token -> image-token attention probs
    attn_consist [Q, topk]      cos(f_sdpa, sum_p attn_w[p]*V[p]) (eager rec.)
    logit_orig   [Q, topk]      correct-class GDA logit (official HEC-V)
    drop_top     [Q, topk]      correct-class logit drop when top-6 patches
                                value-contributions are removed
    drop_rnd     [Q, topk]      same for random patches (avg of 3 trials)
    head_correct [Q, topk]      whether each head's argmax == true class
    ens_correct  [Q]            whether the equal-weight ensemble is correct
    query_imgs_sub [Q,3,224,224]  subset query images (uint8)
    head_probs / query_label / labels_unique  (full batch, baseline self-check)

Outputs: 1) head diversity (pairwise attention cos / top-patch IoU vs random),
         2) heatmap figures (few correct + wrong queries),
         3) discriminative patch-ablation sanity check.
"""
import sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NIMG = 64
GRID = 8          # merged 8x8 grid
N_MASK = 6        # top-attention / random patches removed (10% of 64)
RND_IOU = N_MASK**2 / NIMG / (2 * N_MASK - N_MASK**2 / NIMG)   # theoretical random IoU


def load(path):
    return np.load(path, allow_pickle=True).tolist()


def top_set(a, k=N_MASK):
    """indices of the top-k entries of 1-D array a"""
    return np.argsort(a)[::-1][:k]


def iou(a, b):
    return len(set(a) & set(b)) / len(set(a) | set(b))


def main(path, outdir):
    os.makedirs(outdir, exist_ok=True)
    recs = load(path)
    E = len(recs)

    # ---------------- aggregate per-(query, head) tensors ----------------
    A = np.concatenate([r["attn_maps"] for r in recs])            # [Qtot, topk, 64]
    C = np.concatenate([r["attn_consist"] for r in recs])         # [Qtot, topk]
    DT = np.concatenate([r["drop_top"] for r in recs])            # [Qtot, topk]
    DR = np.concatenate([r["drop_rnd"] for r in recs])            # [Qtot, topk]
    LO = np.concatenate([r["logit_orig"] for r in recs])          # [Qtot, topk]
    HC = np.concatenate([r["head_correct"] for r in recs])        # [Qtot, topk]
    EC = np.concatenate([r["ens_correct"] for r in recs])         # [Qtot]
    Qtot, topk = A.shape[0], A.shape[1]

    print(f"# episodes={E}  # subset queries={Qtot}  heads={topk}  patches={NIMG} (8x8)  n_mask={N_MASK}")

    # ================= 1. HEAD DIVERSITY =================
    print("\n================ 1. HEAD DIVERSITY (top-patch overlap & map similarity) ================")
    cos_pairs, iou_pairs = [], []
    iou_perm = []     # permutation baseline: shuffle patch order of one map
    cos_perm = []
    np.random.seed(0)
    for q in range(Qtot):
        aq = A[q]                                   # [topk, 64]
        for m1 in range(topk):
            for m2 in range(m1 + 1, topk):
                v1, v2 = aq[m1], aq[m2]
                cos_pairs.append(float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)))
                iou_pairs.append(iou(top_set(v1), top_set(v2)))
                # permute one map (same marginal), recompute
                p = np.random.permutation(NIMG)
                v2p = v2[p]
                cos_perm.append(float(v1 @ v2p / (np.linalg.norm(v1) * np.linalg.norm(v2p) + 1e-12)))
                iou_perm.append(iou(top_set(v1), top_set(v2p)))
    cos_pairs, iou_pairs = np.array(cos_pairs), np.array(iou_pairs)
    iou_perm, cos_perm = np.array(iou_perm), np.array(cos_perm)
    print(f"  pairwise attention-map cosine similarity : mean {cos_pairs.mean():.3f}  median {np.median(cos_pairs):.3f}  "
          f"(permuted-random {cos_perm.mean():.3f})")
    print(f"  pairwise top-{N_MASK} patch IoU              : mean {iou_pairs.mean():.3f}  median {np.median(iou_pairs):.3f}  "
          f"(permuted-random {iou_perm.mean():.3f}, theory {RND_IOU:.3f})")
    print(f"  fraction of head pairs with IoU > 0.3: {100*(iou_pairs>0.3).mean():.1f}%   > 0.5: {100*(iou_pairs>0.5).mean():.1f}%   > 0.7: {100*(iou_pairs>0.7).mean():.1f}%")
    # mean map per head (how concentrated / where)
    mean_map = A.mean(0)                              # [topk, 64]
    concentr = (mean_map**2).sum(-1)                  # [topk]
    print(f"  per-head image-attn concentration (sum p^2): mean {concentr.mean():.3f}  min {concentr.min():.3f}  max {concentr.max():.3f}")
    img_mass = A.sum(-1)                              # [Qtot, topk] total attn on image
    print(f"  per-head image-attention mass: mean {img_mass.mean():.3f}  min {img_mass.min():.3f}  max {img_mass.max():.3f}")

    # ================= 2. DISCRIMINATIVE SANITY (patch ablation) =================
    print("\n================ 2. PATCH ABLATION (correct-class GDA logit drop) ================")
    good = C > 0.5
    print(f"  consistency cos(f_sdpa, rec): mean {C.mean():.3f}  median {np.median(C):.3f}  "
          f"frac>0.5 {100*good.mean():.1f}%  frac>0.7 {100*(C>0.7).mean():.1f}%")
    for name, mask in [("all heads/queries", np.ones_like(DT, bool)),
                       ("consistent heads only (consist>0.5)", good),
                       ("correct heads", HC),
                       ("wrong heads", ~HC)]:
        d1, d2 = DT[mask], DR[mask]
        if d1.size == 0:
            continue
        win = float((d1 > d2).mean())
        print(f"  [{name:42s}] n={d1.size:6d}  drop_top={d1.mean():+.4f}  drop_rnd={d2.mean():+.4f}  "
              f"delta={d1.mean()-d2.mean():+.4f}  top>rnd: {100*win:.1f}%")
    # does the top-drop scale with correctness / consistency?
    # per query: does removing top patches hurt the CORRECT head more?
    print("  correct-class logit drop when removing top-6 patches (vs removing random-6):")

    # ================= 3. VISUALIZATION =================
    print("\n================ 3. VISUALIZATION ================")
    # pick per-episode first correct and first wrong query (of the subset)
    picks = []   # (episode, query_in_subset, is_correct)
    for ep, r in enumerate(recs):
        sub_ec = r["ens_correct"]
        for tag, want in [("correct", True), ("wrong", False)]:
            idx = np.where(sub_ec == want)[0]
            if len(idx):
                picks.append((ep, idx[0], tag))
    print(f"  picked {len(picks)} queries for heatmaps: {[p[2] for p in picks]}")

    for ep, qi, tag in picks:
        r = recs[ep]
        img = r["query_imgs_sub"][qi].transpose(1, 2, 0)          # H,W,3 uint8
        maps = r["attn_maps"][qi]                                  # [topk, 64]
        fig = plt.figure(figsize=(16, 9))
        gs = fig.add_gridspec(3, 7, hspace=0.05, wspace=0.05)
        ax = fig.add_subplot(gs[0, 0]); ax.imshow(img); ax.set_axis_off(); ax.set_title("image", fontsize=8)
        # 20 heads in 5 rows x 4 cols -> use remaining cells; draw image overlay for each head
        hs = list(range(topk))
        # heatmaps overlaid on the image
        cell_idx = 0
        for row in range(3):
            for col in range(1, 7):
                if cell_idx >= len(hs):
                    ax = fig.add_subplot(gs[row, col]); ax.axis("off"); continue
                h = hs[cell_idx]; cell_idx += 1
                m = maps[h].reshape(GRID, GRID)
                m_up = np.kron(m, np.ones((28, 28)))               # 8x8 -> 224x224
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(img)
                im = ax.imshow(m_up / (m_up.max() + 1e-9), cmap="jet", alpha=0.55)
                ax.set_axis_off(); ax.set_title(f"head {h} (rank {h})", fontsize=7)
        fig.suptitle(f"ep{ep} subset-query {qi}  ({tag})  ens_correct={bool(r['ens_correct'][qi])} "
                     f"true={r['labels_unique'][np.searchsorted(r['labels_unique'], r['query_label_sub'][qi])]}", fontsize=10)
        out = os.path.join(outdir, f"heatmap_ep{ep}_q{qi}_{tag}.png")
        fig.savefig(out, dpi=90, bbox_inches="tight"); plt.close(fig)
        print(f"  saved {out}")

    print("\n================ SUMMARY ================")
    print(f"  top-patch IoU (mean): {iou_pairs.mean():.3f}  (random {iou_perm.mean():.3f})   cosine: {cos_pairs.mean():.3f} (random {cos_perm.mean():.3f})")
    print(f"  drop_top - drop_rnd (all): {DT.mean()-DR.mean():+.4f}   (consistent): {DT[good].mean()-DR[good].mean():+.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "headpatch_figs")

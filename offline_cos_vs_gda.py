#!/usr/bin/env python3
"""Offline (no backbone, CPU) mechanism check: cosine nearest-centroid vs HEC GDA
under clean / 25%-noise support labels, using the FROZEN FEATURES already saved
by exp_noisy_label_2x2.py (features.npz + feat_labels.npy + 2x2 records).

Fixed per episode (identical to the previous experiments):
  - EuroSAT 4-shot episodes 0..4, seed=123
  - 25% label noise realization (loaded from the 2x2 records, not regenerated)
  - clean Top-20 heads (sel_clean loaded from the 2x2 records)
  - class means computed from the same frozen features (clean or noisy labels)

Only the CLASSIFIER SCORE is switched:
  (1) cosine nearest-centroid:  z = softmax(cos(q, mu_k)/tau), mean_prob over heads
  (2) HEC original GDA/Mahal.:  z = softmax((-0.5*mahal + log_prior)/tau), mean_prob

Also measures prototype pollution geometry between clean and noisy means:
  cosine similarity, L2 shift, and Mahalanobis shift (in the clean GDA metric).

Cross-check: GDA accuracies must reproduce the 2x2 "CleanH" conditions
(cH_cG ~ 80.8%, cH_nG(25%) ~ 67.6% on episodes 0..4; fp16 features -> tiny
numerical deviations possible).
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

sys.path.insert(0, "/data2/lbk/HEC")
from src.fewshot.classifier import ClassifierHEC_V

FEAT_DIR = "/data2/lbk/HEC_repro/logs/repro_eurosat_noisy/n60_2x2"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 60
K = 20
TAU = 10.0


def gda_acc(clf, Aq2, sel, rs, yq_np):
    z, _ = clf.get_pred_labels(
        Aq2[:, sel, :],
        rs["head_means"][sel, :, :],
        rs["head_cov_inv"][sel, :, :],
        rs["log_prior"],
    )
    yhat_idx = z.mean(dim=1).argmax(dim=-1).cpu().numpy()
    lu = rs["labels_unique"].cpu().numpy().astype(np.int64)
    yhat = lu[yhat_idx]
    return float((yhat == yq_np).mean())


def cosine_acc(Aq2, sel, rs, yq_np, tau=TAU):
    """Nearest-centroid with cosine score; same mean_prob ensemble as HEC."""
    mu = F.normalize(rs["head_means"][sel, :, :], dim=-1)      # [t, K, C]
    q = F.normalize(Aq2[:, sel, :], dim=-1)                     # [Nq, t, C]
    sim = torch.einsum("ntc,tkc->ntk", q, mu)                   # [Nq, t, K]
    z = torch.softmax(sim / tau, dim=-1)
    yhat_idx = z.mean(dim=1).argmax(dim=-1).cpu().numpy()
    lu = rs["labels_unique"].cpu().numpy().astype(np.int64)
    yhat = lu[yhat_idx]
    return float((yhat == yq_np).mean())


def cosine_acc_meansim(Aq2, sel, rs, yq_np):
    """Plain variant: argmax of the head-averaged cosine similarity."""
    mu = F.normalize(rs["head_means"][sel, :, :], dim=-1)
    q = F.normalize(Aq2[:, sel, :], dim=-1)
    sim = torch.einsum("ntc,tkc->ntk", q, mu)
    yhat_idx = sim.mean(dim=1).argmax(dim=-1).cpu().numpy()
    lu = rs["labels_unique"].cpu().numpy().astype(np.int64)
    yhat = lu[yhat_idx]
    return float((yhat == yq_np).mean())


def main():
    records = np.load(f"{FEAT_DIR}/noisy_label_2x2_records.npy",
                      allow_pickle=True).tolist()[:N_EP]
    feat = np.load(f"{FEAT_DIR}/features.npz")
    labels_list = np.load(f"{FEAT_DIR}/feat_labels.npy",
                          allow_pickle=True).tolist()[:N_EP]
    support_A = feat["support_A"][:N_EP]   # [N_EP, Ns, M, Dh] fp16
    query_A = feat["query_A"][:N_EP]       # [N_EP, Nq, M, Dh] fp16

    clf = ClassifierHEC_V(input_mode="cls", k=K, tau=TAU, cov_reg=1,
                          ensemble_method="mean_prob")

    rows = []
    for it in range(N_EP):
        rec = records[it]
        lab = labels_list[it]
        A2 = torch.from_numpy(support_A[it].astype(np.float32))   # [Ns, M, Dh]
        Aq2 = torch.from_numpy(query_A[it].astype(np.float32))    # [Nq, M, Dh]
        clean_labels = torch.from_numpy(lab["clean_labels"]).long()
        noisy_labels = torch.from_numpy(rec["r25"]["noisy_labels"]).long()
        yq_np = lab["query_label"].astype(np.int64)
        sel = torch.from_numpy(rec["r25"]["sel_clean"]).long()    # clean Top-20

        rs_clean = clf.get_rank_scores(A2, clean_labels)
        rs_noisy = clf.get_rank_scores(A2, noisy_labels)

        # -------- accuracies --------
        acc = {
            "gda_clean": gda_acc(clf, Aq2, sel, rs_clean, yq_np),
            "gda_noisy": gda_acc(clf, Aq2, sel, rs_noisy, yq_np),
            "cos_clean": cosine_acc(Aq2, sel, rs_clean, yq_np),
            "cos_noisy": cosine_acc(Aq2, sel, rs_noisy, yq_np),
            "cosms_noisy": cosine_acc_meansim(Aq2, sel, rs_noisy, yq_np),
        }

        # -------- prototype geometry (clean vs noisy means, on the Top-20) --
        mu_c = rs_clean["head_means"][sel, :, :]     # [k, K, C]
        mu_n = rs_noisy["head_means"][sel, :, :]
        delta = mu_n - mu_c
        cos_sim = F.cosine_similarity(mu_c, mu_n, dim=-1)          # [k, K]
        l2_shift = delta.norm(dim=-1)                               # [k, K]
        cov_inv = rs_clean["head_cov_inv"][sel, :, :]               # [k, C, C]
        mahal_shift = torch.einsum("kcd,kdf,kcf->kc", delta, cov_inv, delta)

        rows.append({
            "episode": it,
            **acc,
            "proto_cos": float(cos_sim.mean()),
            "proto_cos_min": float(cos_sim.min()),
            "proto_l2": float(l2_shift.mean()),
            "proto_mahal": float(mahal_shift.mean()),
            "proto_mahal_median": float(mahal_shift.median()),
        })
        print(f"ep {it}: " + " ".join(f"{k}={100*v:.1f}" for k, v in acc.items())
              + f" | proto cos={rows[-1]['proto_cos']:.4f} "
              f"l2={rows[-1]['proto_l2']:.4f} mahal={rows[-1]['proto_mahal']:.3f}")

    # ---------------- aggregate ----------------
    print(f"\n{'='*74}")
    print(f"AGGREGATE (n = {N_EP} episodes, 25% noise, clean Top-20 heads)")
    print("=" * 74)
    A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k != "episode"}

    print("\n--- 1. ACCURACY ---")
    print(f"{'classifier':22s} {'clean':>8s} {'noisy':>8s} {'drop':>8s}")
    for name, (ck, nk) in {"cosine NC": ("cos_clean", "cos_noisy"),
                           "HEC GDA (Mahalanobis)": ("gda_clean", "gda_noisy")}.items():
        d = A[ck] - A[nk]
        print(f"{name:22s} {100*A[ck].mean():7.2f}% {100*A[nk].mean():7.2f}% "
              f"{100*d.mean():+7.2f}pt")
    # per-episode
    print("\n  per-episode:")
    print("  ep |  cos_clean  cos_noisy  gda_clean  gda_noisy | cos_drop  gda_drop")
    for r in rows:
        print(f"  {r['episode']:2d} | {100*r['cos_clean']:9.1f} {100*r['cos_noisy']:9.1f} "
              f"{100*r['gda_clean']:9.1f} {100*r['gda_noisy']:9.1f} | "
              f"{100*(r['cos_clean']-r['cos_noisy']):+8.1f} "
              f"{100*(r['gda_clean']-r['gda_noisy']):+8.1f}")

    cos_drop = A["cos_clean"] - A["cos_noisy"]
    gda_drop = A["gda_clean"] - A["gda_noisy"]
    try:
        p = stats.wilcoxon(cos_drop, gda_drop).pvalue
    except Exception:
        p = float("nan")
    print(f"\n  drop comparison: cosine={100*cos_drop.mean():+.2f}pt  "
          f"gda={100*gda_drop.mean():+.2f}pt  "
          f"(gda - cosine = {100*(gda_drop.mean()-cos_drop.mean()):+.2f}pt, "
          f"Wilcoxon_p={p:.4f})")
    print(f"  noisy-accuracy comparison: cosine={100*A['cos_noisy'].mean():.2f}%  "
          f"gda={100*A['gda_noisy'].mean():.2f}%  "
          f"(gda - cosine = {100*(A['gda_noisy'].mean()-A['cos_noisy'].mean()):+.2f}pt)")
    print(f"  [sanity] plain mean-cosine (no softmax) noisy acc = "
          f"{100*A['cosms_noisy'].mean():.2f}%  (softmax variant "
          f"{100*A['cos_noisy'].mean():.2f}%)")

    print("\n--- 2. PROTOTYPE POLLUTION GEOMETRY (clean vs noisy means) ---")
    print(f"  cosine(clean_mu, noisy_mu) mean = {A['proto_cos'].mean():.4f} "
          f"(min over heads/classes: {A['proto_cos_min'].mean():.4f})")
    print(f"  L2 shift ||mu_n - mu_c||  mean = {A['proto_l2'].mean():.4f} "
          f"(prototypes live on the unit-normalized feature space)")
    print(f"  Mahalanobis shift (clean metric) mean = {A['proto_mahal'].mean():.3f}  "
          f"median = {A['proto_mahal_median'].mean():.3f}")

    print("\n--- 3. CROSS-CHECK vs 2x2 experiment (episodes 0..4, 25%) ---")
    print(f"  gda_clean = {100*A['gda_clean'].mean():.2f}%  (2x2 cH_cG: 80.80%)")
    print(f"  gda_noisy = {100*A['gda_noisy'].mean():.2f}%  (2x2 cH_nG: 67.60%)")

    np.save("/data2/lbk/HEC_repro/logs/repro_eurosat_noisy/gda3_n5/"
            "offline_cos_vs_gda_rows.npy", rows)
    print("\nrows saved -> offline_cos_vs_gda_rows.npy")


if __name__ == "__main__":
    main()

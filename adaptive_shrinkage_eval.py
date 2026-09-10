#!/usr/bin/env python3
"""Reliability-aware Adaptive Covariance Shrinkage, fully OFFLINE (no VLM, GPU).

Setup identical to previous offline runs: EuroSAT 4-shot, 60 episodes, 25%
noise (same realization), clean Top-20 heads, frozen features, GDA flow
unchanged, no label correction.

Idea: shrinkage multiplier lambda is set PER EPISODE from the unlabeled
reliability of the support set:
  suspicion u_ep = fraction of support with margin < -0.05
      margin_i = cos(x_i, own-class LOO prototype) - max_k cos(x_i, class k
      prototype)   (negative => sample fits another class better)
      threshold -0.05 calibrated on the support margin distributions:
      u_clean=0.089, u_noisy=0.301, catches 74% of mislabeled samples
  - ada_u   : lambda = clip(1 + 8*u, 1, 5)         (no calibration)
      -> clean lambda ~1.7 (GDA mostly kept), noisy lambda ~3.4 (fix4-like)
  - ada_rank: lambda = 1 + 4*(1 - rank_pct(R_ep)) (cross-episode rank on
              R_ep = mean margin; lambda in [1,5], mean ~3)
The rule uses ONLY support features + noisy labels (no clean info).

Methods compared (prototype x shrinkage):
  orig (mean), wcos_g3 (weighted, cos^3)  x  {none, fix4, ada_u, ada_rank}
Cross-checks: orig noisy=68.47%, fix4=69.53%, wcos_g3+fix4=70.47%.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sps

sys.path.insert(0, "/data2/lbk/HEC")
from src.fewshot.classifier import ClassifierHEC_V

FEAT_DIR = "/data2/lbk/HEC_repro/logs/repro_eurosat_noisy/n60_2x2"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 60
K = 20
TAU = 10.0
COV_REG = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clf = ClassifierHEC_V(input_mode="cls", k=K, tau=TAU, cov_reg=COV_REG,
                      ensemble_method="mean_prob")


def fit_predict(Xn, y_idx, Kc, means, cov_mult, Xq, lu, yq_np):
    N, t, Cdim = Xn.shape
    centered = Xn.permute(1, 0, 2) - means[:, y_idx, :]
    cov = torch.einsum('tnc,tnf->tcf', centered, centered) / (N - 1)
    tr = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
    cov_inv = Cdim * torch.linalg.pinv(
        (N - 1) * cov + COV_REG * cov_mult * tr[:, None, None]
        * torch.eye(Cdim, device=Xn.device))
    counts = torch.bincount(y_idx, minlength=Kc).float()
    log_prior = (counts / counts.sum()).log()
    z, _ = clf.get_pred_labels(Xq, means, cov_inv, log_prior)
    yhat_idx = z.mean(dim=1).argmax(dim=-1).cpu().numpy()
    lu_np = lu.cpu().numpy().astype(np.int64)
    return float((lu_np[yhat_idx] == yq_np).mean())


def signals(Xn, y_idx, Kc):
    """Per-sample LOO cosine + per-head scores; returns (cos_sig [Ns],
    margin_sig [Ns])."""
    Ns, t, C = Xn.shape
    S = torch.zeros(t, Kc, C, device=Xn.device)
    n = torch.zeros(Kc, device=Xn.device)
    for k in range(Kc):
        m = y_idx == k
        if m.any():
            S[:, k, :] = Xn[m].sum(0)
            n[k] = int(m.sum())
    mu_full = S / n.clamp_min(1)[None, :, None]
    mu_loo_own = (S[:, y_idx, :] - Xn.permute(1, 0, 2)) / (n[y_idx] - 1).clamp_min(1)[None, :, None]
    cos_own = F.cosine_similarity(Xn.permute(1, 0, 2), mu_loo_own, dim=-1).permute(1, 0)
    sig_cos = cos_own.mean(dim=1)
    scores = torch.einsum('ntc,tkc->ntk', Xn, mu_full)
    own = torch.einsum('ntc,ntc->nt', Xn, mu_loo_own.permute(1, 0, 2))
    scores[torch.arange(Ns, device=Xn.device), :, y_idx] = own
    other = scores.clone()
    other[torch.arange(Ns, device=Xn.device), :, y_idx] = -1e9
    sig_margin = (own - other.max(dim=-1).values).mean(dim=1)
    return sig_cos, sig_margin


def proto_mean(Xn, y_idx, Kc):
    t, C = Xn.shape[1], Xn.shape[2]
    mus = torch.zeros(t, Kc, C, device=Xn.device)
    for k in range(Kc):
        m = y_idx == k
        if m.any():
            mus[:, k, :] = Xn[m].mean(0)
    return mus


def proto_weighted_w(Xn, y_idx, Kc, w):
    t, C = Xn.shape[1], Xn.shape[2]
    mus = torch.zeros(t, Kc, C, device=Xn.device)
    for k in range(Kc):
        m = y_idx == k
        if m.any():
            mus[:, k, :] = (Xn[m] * w[m][:, None, None]).sum(0) / w[m].sum()
    return mus


def main():
    print(f"device = {DEVICE}", flush=True)
    records = np.load(f"{FEAT_DIR}/noisy_label_2x2_records.npy",
                      allow_pickle=True).tolist()[:N_EP]
    feat = np.load(f"{FEAT_DIR}/features.npz")
    labels_list = np.load(f"{FEAT_DIR}/feat_labels.npy",
                          allow_pickle=True).tolist()[:N_EP]
    support_A = feat["support_A"][:N_EP]
    query_A = feat["query_A"][:N_EP]

    # ---- pass 1: episode reliability (u, R) under clean and noisy labels ----
    u_stats = {"clean": [], "noisy": []}
    R_stats = {"clean": [], "noisy": []}
    ep_cache = []
    for it in range(N_EP):
        rec, lab = records[it], labels_list[it]
        A2 = torch.from_numpy(support_A[it].astype(np.float32)).to(DEVICE)
        clean_labels = torch.from_numpy(lab["clean_labels"]).long().to(DEVICE)
        noisy_labels = torch.from_numpy(rec["r25"]["noisy_labels"]).long().to(DEVICE)
        sel = torch.from_numpy(rec["r25"]["sel_clean"]).long().to(DEVICE)
        Xn = F.normalize(A2[:, sel, :], dim=-1)
        cur = {}
        for cond, labels in [("clean", clean_labels), ("noisy", noisy_labels)]:
            lu = torch.unique(labels)
            y_idx = torch.searchsorted(lu, labels)
            sig_cos, sig_margin = signals(Xn, y_idx, lu.numel())
            u = float((sig_margin < -0.05).float().mean())
            u_stats[cond].append(u)
            R_stats[cond].append(float(sig_margin.mean()))
            cur[cond] = {"lu": lu, "y_idx": y_idx,
                         "w": sig_cos.clamp_min(1e-4) ** 3}
        ep_cache.append({"Xn": Xn, "cur": cur, "rec": rec, "lab": lab})
    # rank percentile of R within each condition (cross-episode calibration)
    rank_pct = {}
    for cond in ["clean", "noisy"]:
        R = np.array(R_stats[cond])
        order = R.argsort().argsort() / (N_EP - 1)     # 0 = worst reliability
        rank_pct[cond] = order

    print(f"\nsuspicion u (margin<0 fraction): clean = {np.mean(u_stats['clean']):.3f} "
          f"± {np.std(u_stats['clean']):.3f} | noisy = {np.mean(u_stats['noisy']):.3f} "
          f"± {np.std(u_stats['noisy']):.3f}")
    lam_u = {c: np.clip(1 + 8 * np.array(u_stats[c]), 1, 5) for c in ["clean", "noisy"]}
    lam_r = {c: 1 + 4 * (1 - rank_pct[c]) for c in ["clean", "noisy"]}
    for name, lam in [("ada_u", lam_u), ("ada_rank", lam_r)]:
        print(f"lambda[{name}]: clean mean={lam['clean'].mean():.2f} "
              f"(min {lam['clean'].min():.2f}, max {lam['clean'].max():.2f}) | "
              f"noisy mean={lam['noisy'].mean():.2f} "
              f"(min {lam['noisy'].min():.2f}, max {lam['noisy'].max():.2f})")

    # ---- pass 2: predictions ----
    METHODS = ["orig", "wcos_g3"]
    SHRINKS = ["none", "fix4", "ada_u", "ada_rank"]
    rows = []
    for it in range(N_EP):
        cache = ep_cache[it]
        rec, lab = cache["rec"], cache["lab"]
        sel = torch.from_numpy(rec["r25"]["sel_clean"]).long().to(DEVICE)
        Aq2 = torch.from_numpy(query_A[it].astype(np.float32)).to(DEVICE)
        Xq = Aq2[:, sel, :]
        yq_np = lab["query_label"].astype(np.int64)
        Xn = cache["Xn"]
        row = {"episode": it}
        for cond in ["clean", "noisy"]:
            lu = cache["cur"][cond]["lu"]
            y_idx = cache["cur"][cond]["y_idx"]
            Kc = lu.numel()
            means = {
                "orig": proto_mean(Xn, y_idx, Kc),
                "wcos_g3": proto_weighted_w(Xn, y_idx, Kc, cache["cur"][cond]["w"]),
            }
            lams = {"none": 1.0, "fix4": 4.0,
                    "ada_u": float(lam_u[cond][it]),
                    "ada_rank": float(lam_r[cond][it])}
            for m in METHODS:
                for s in SHRINKS:
                    row[f"{m}_{s}_{cond}"] = fit_predict(
                        Xn, y_idx, Kc, means[m], lams[s], Xq, lu, yq_np)
        rows.append(row)
        if it % 10 == 0 or it == N_EP - 1:
            print(f"ep {it}: orig={100*row['orig_none_noisy']:.1f} "
                  f"fix4={100*row['orig_fix4_noisy']:.1f} "
                  f"ada_u={100*row['orig_ada_u_noisy']:.1f} "
                  f"wcos+ada_u={100*row['wcos_g3_ada_u_noisy']:.1f}", flush=True)

    np.save("/data2/lbk/HEC_repro/logs/repro_eurosat_noisy/gda3_n5/"
            "adaptive_shrinkage_rows.npy", rows)

    # ---- aggregate ----
    def arr(key):
        return np.array([r[key] for r in rows])

    print("\n" + "=" * 96)
    print(f"RELIABILITY-AWARE ADAPTIVE SHRINKAGE (n = {N_EP}, 25% noise)")
    print("=" * 96)
    orig_n = arr("orig_none_noisy")
    orig_c = arr("orig_none_clean")
    print(f"{'method':26s} {'clean':>8s} {'noisy':>8s} {'drop':>8s} {'vs orig':>8s} "
          f"{'p_noisy':>8s} {'p_clean':>8s}")
    for m in METHODS:
        for s in SHRINKS:
            key = f"{m}_{s}_noisy"
            ck = f"{m}_{s}_clean"
            c, n_ = arr(ck), arr(key)
            try:
                pn = sps.wilcoxon(n_, orig_n).pvalue
            except Exception:
                pn = float("nan")
            try:
                pc = sps.wilcoxon(c, orig_c).pvalue
            except Exception:
                pc = float("nan")
            print(f"{m + '+' + s:26s} {100*c.mean():7.2f}% {100*n_.mean():7.2f}% "
                  f"{100*(c.mean()-n_.mean()):+7.2f} {100*(n_.mean()-orig_n.mean()):+7.2f} "
                  f"{pn:8.4f} {pc:8.4f}")

    # ada vs fix4 paired
    print("\npaired vs fix4 (same prototype):")
    for m in METHODS:
        for s in ["ada_u", "ada_rank"]:
            v = arr(f"{m}_{s}_noisy")
            f4 = arr(f"{m}_fix4_noisy")
            try:
                p = sps.wilcoxon(v, f4).pvalue
            except Exception:
                p = float("nan")
            print(f"  {m}+{s:9s} vs {m}+fix4 : noisy {100*(v.mean()-f4.mean()):+.2f}pt "
                  f"(p={p:.4f})")


if __name__ == "__main__":
    main()

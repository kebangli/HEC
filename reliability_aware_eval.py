#!/usr/bin/env python3
"""Reliability-aware Prototype validation, fully OFFLINE (no VLM, GPU).

Setup identical to robust_prototype_eval.py: EuroSAT 4-shot, 60 episodes,
25% noise (same realization), clean Top-20 heads, frozen features, GDA flow
unchanged. No label correction, no hard drop.

Methods compared (prototype estimator x covariance shrinkage):
  orig        : arithmetic mean (= original HEC)
  wcos_g3     : weighted prototype, reliability = LOO cosine (mean over
                Top-20 heads), weight = cos^3        (previous method)
  wfused_g3   : FUSED reliability = mean of per-episode z-scores of
                  (a) LOO cosine           (same as wcos)
                  (b) multi-head agreement (fraction of heads whose LOO
                      cosine argmax predicts the sample's current label)
                  (c) prototype margin     (cos to own LOO prototype minus
                      best other-class prototype, averaged over heads)
                reliability = sigmoid(z_fused), weight = reliability^3
  wfused_g6   : same with gamma=6 (sharper soft down-weighting)
Each of the above x {cov_mult=1 (none), cov_mult=4 (fix4 shrinkage)}.

Also reports AUROC of each reliability signal (detecting mislabeled
support) vs the previous LOO-cosine-only score.
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sps
from sklearn.metrics import roc_auc_score

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


def fit_gda(Xn, y_idx, K, means, cov_mult=1.0):
    N, t, Cdim = Xn.shape
    centered = Xn.permute(1, 0, 2) - means[:, y_idx, :]          # [t,Ns,C]
    cov = torch.einsum('tnc,tnf->tcf', centered, centered) / (N - 1)
    tr = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
    cov_inv = Cdim * torch.linalg.pinv(
        (N - 1) * cov + COV_REG * cov_mult * tr[:, None, None]
        * torch.eye(Cdim, device=Xn.device))
    counts = torch.bincount(y_idx, minlength=K).float()
    log_prior = (counts / counts.sum()).log()
    return means, cov_inv, log_prior


def predict(Aq_sel, means, cov_inv, log_prior, labels_unique, yq_np):
    z, _ = clf.get_pred_labels(Aq_sel, means, cov_inv, log_prior)
    yhat_idx = z.mean(dim=1).argmax(dim=-1).cpu().numpy()
    lu = labels_unique.cpu().numpy().astype(np.int64)
    return float((lu[yhat_idx] == yq_np).mean())


def proto_mean(Xn, y_idx, K):
    t, C = Xn.shape[1], Xn.shape[2]
    mus = torch.zeros(t, K, C, device=Xn.device)
    for k in range(K):
        m = y_idx == k
        if m.any():
            mus[:, k, :] = Xn[m].mean(0)
    return mus


def proto_weighted_w(Xn, y_idx, K, w):
    """Weighted prototype with explicit per-sample weights w [Ns]."""
    t, C = Xn.shape[1], Xn.shape[2]
    mus = torch.zeros(t, K, C, device=Xn.device)
    for k in range(K):
        m = y_idx == k
        if m.any():
            mus[:, k, :] = (Xn[m] * w[m][:, None, None]).sum(0) / w[m].sum()
    return mus


def reliability_signals(Xn, y_idx, K):
    """Return per-sample signals: LOO cosine, multi-head agreement, margin."""
    Ns, t, C = Xn.shape
    S = torch.zeros(t, K, C, device=Xn.device)
    n = torch.zeros(K, device=Xn.device)
    for k in range(K):
        m = y_idx == k
        if m.any():
            S[:, k, :] = Xn[m].sum(0)
            n[k] = int(m.sum())
    mu_full = S / n.clamp_min(1)[None, :, None]                        # [t,K,C]
    mu_loo_own = (S[:, y_idx, :] - Xn.permute(1, 0, 2)) / (n[y_idx] - 1).clamp_min(1)[None, :, None]

    # (a) LOO cosine (identical to previous weighted method)
    cos_own = F.cosine_similarity(Xn.permute(1, 0, 2), mu_loo_own, dim=-1).permute(1, 0)  # [Ns,t]
    sig_cos = cos_own.mean(dim=1)                                      # [Ns]

    # per-head class scores (own class uses LOO prototype)
    scores = torch.einsum('ntc,tkc->ntk', Xn, mu_full)                 # [Ns,t,K]
    own = torch.einsum('ntc,ntc->nt', Xn, mu_loo_own.permute(1, 0, 2)) # [Ns,t]
    scores[torch.arange(Ns, device=Xn.device), :, y_idx] = own
    pred = scores.argmax(-1)                                           # [Ns,t]

    # (b) multi-head agreement with the current label
    sig_agree = (pred == y_idx[:, None]).float().mean(dim=1)           # [Ns]

    # (c) prototype margin: own(LOO) - best other (full), averaged over heads
    other = scores.clone()
    other[torch.arange(Ns, device=Xn.device), :, y_idx] = -1e9
    sig_margin = (own - other.max(dim=-1).values).mean(dim=1)          # [Ns]

    return sig_cos, sig_agree, sig_margin


def zscore(x):
    return (x - x.mean()) / x.std().clamp_min(1e-8)


def main():
    print(f"device = {DEVICE}", flush=True)
    records = np.load(f"{FEAT_DIR}/noisy_label_2x2_records.npy",
                      allow_pickle=True).tolist()[:N_EP]
    feat = np.load(f"{FEAT_DIR}/features.npz")
    labels_list = np.load(f"{FEAT_DIR}/feat_labels.npy",
                          allow_pickle=True).tolist()[:N_EP]
    support_A = feat["support_A"][:N_EP]
    query_A = feat["query_A"][:N_EP]

    METHODS = ["orig", "wcos_g3", "wfused_g3", "wfused_g6"]
    COVS = ["none", "fix4"]

    rows = []
    sig_pool = {k: ([], []) for k in ["cos", "agree", "margin", "fused"]}  # (values, mis)
    per_ep_auroc = {k: [] for k in sig_pool}

    for it in range(N_EP):
        rec, lab = records[it], labels_list[it]
        A2 = torch.from_numpy(support_A[it].astype(np.float32)).to(DEVICE)
        Aq2 = torch.from_numpy(query_A[it].astype(np.float32)).to(DEVICE)
        clean_labels = torch.from_numpy(lab["clean_labels"]).long().to(DEVICE)
        noisy_labels = torch.from_numpy(rec["r25"]["noisy_labels"]).long().to(DEVICE)
        yq_np = lab["query_label"].astype(np.int64)
        sel = torch.from_numpy(rec["r25"]["sel_clean"]).long().to(DEVICE)
        Xn = F.normalize(A2[:, sel, :], dim=-1)
        Xq = Aq2[:, sel, :]
        lu = torch.unique(noisy_labels)
        Kc = lu.numel()
        y_idx = torch.searchsorted(lu, noisy_labels)
        mis = (noisy_labels != clean_labels)

        # reliability signals + fused score (noisy regime)
        sig_cos, sig_agree, sig_margin = reliability_signals(Xn, y_idx, Kc)
        z_fused = (zscore(sig_cos) + zscore(sig_agree) + zscore(sig_margin)) / 3.0
        rel_fused = torch.sigmoid(z_fused)                             # [Ns] in (0,1)

        # AUROC bookkeeping
        mis_np = mis.cpu().numpy()
        for k, v in [("cos", sig_cos), ("agree", sig_agree),
                     ("margin", sig_margin), ("fused", rel_fused)]:
            v_np = v.cpu().numpy()
            sig_pool[k][0].append(v_np)
            sig_pool[k][1].append(mis_np)
            per_ep_auroc[k].append(roc_auc_score(mis_np, -v_np))

        # weights
        w_cos = sig_cos.clamp_min(1e-4) ** 3
        w_f3 = rel_fused.clamp_min(1e-4) ** 3
        w_f6 = rel_fused.clamp_min(1e-4) ** 6

        row = {"episode": it}
        for cond, labels in [("clean", clean_labels), ("noisy", noisy_labels)]:
            lu_c = torch.unique(labels)
            y_c = torch.searchsorted(lu_c, labels)
            means = {
                "orig": proto_mean(Xn, y_c, Kc),
            }
            if cond == "noisy":
                means["wcos_g3"] = proto_weighted_w(Xn, y_c, Kc, w_cos)
                means["wfused_g3"] = proto_weighted_w(Xn, y_c, Kc, w_f3)
                means["wfused_g6"] = proto_weighted_w(Xn, y_c, Kc, w_f6)
            else:
                # clean regime: recompute signals with clean labels (all correct)
                sc, sa, sm = reliability_signals(Xn, y_c, Kc)
                zf = (zscore(sc) + zscore(sa) + zscore(sm)) / 3.0
                rf = torch.sigmoid(zf)
                means["wcos_g3"] = proto_weighted_w(Xn, y_c, Kc, sc.clamp_min(1e-4) ** 3)
                means["wfused_g3"] = proto_weighted_w(Xn, y_c, Kc, rf.clamp_min(1e-4) ** 3)
                means["wfused_g6"] = proto_weighted_w(Xn, y_c, Kc, rf.clamp_min(1e-4) ** 6)
            for mname in METHODS:
                if mname not in means:
                    continue
                for cov, mult in [("none", 1.0), ("fix4", 4.0)]:
                    if cond == "noisy" and mname == "orig" and cov == "fix4":
                        pass  # keep (informative)
                    _, ci, lp = fit_gda(Xn, y_c, Kc, means[mname], cov_mult=mult)
                    row[f"{mname}_{cov}_{cond}"] = predict(Xq, means[mname], ci, lp, lu_c, yq_np)
        rows.append(row)
        if it % 10 == 0 or it == N_EP - 1:
            print(f"ep {it}: orig={100*row['orig_none_noisy']:.1f} "
                  f"wcos={100*row['wcos_g3_none_noisy']:.1f} "
                  f"wfused3={100*row['wfused_g3_none_noisy']:.1f} "
                  f"wfused3+fix4={100*row['wfused_g3_fix4_noisy']:.1f}", flush=True)

    np.save("/data2/lbk/HEC_repro/logs/repro_eurosat_noisy/gda3_n5/"
            "reliability_aware_rows.npy", rows)

    # ---------------- aggregate ----------------
    def arr(key):
        return np.array([r[key] for r in rows])

    print("\n" + "=" * 92)
    print(f"RELIABILITY-AWARE PROTOTYPE (n = {N_EP}, 25% noise, clean Top-20 heads)")
    print("=" * 92)
    print(f"{'method':24s} {'clean':>8s} {'noisy':>8s} {'drop':>8s} {'vs orig':>8s} {'p':>8s}")
    orig_n = arr("orig_none_noisy")
    for mname in METHODS:
        for cov in COVS:
            key = f"{mname}_{cov}_noisy"
            ck = f"{mname}_{cov}_clean"
            c = arr(ck).mean()
            n_ = arr(key)
            try:
                p = sps.wilcoxon(n_, orig_n).pvalue
            except Exception:
                p = float("nan")
            print(f"{mname + '+' + cov:24s} {100*c:7.2f}% {100*n_.mean():7.2f}% "
                  f"{100*(c-n_.mean()):+7.2f} {100*(n_.mean()-orig_n.mean()):+7.2f} {p:8.4f}")

    # comparisons vs wcos
    wcos_n = arr("wcos_g3_none_noisy")
    print("\npaired vs wcos_g3 (noisy):")
    for key in ["wfused_g3_none_noisy", "wfused_g3_fix4_noisy", "wfused_g6_fix4_noisy"]:
        v = arr(key)
        try:
            p = sps.wilcoxon(v, wcos_n).pvalue
        except Exception:
            p = float("nan")
        print(f"  {key:28s}: {100*(v.mean()-wcos_n.mean()):+.2f}pt  p={p:.4f}")

    # ---------------- AUROC ----------------
    print("\n" + "=" * 92)
    print("RELIABILITY SIGNAL QUALITY (detecting mislabeled support)")
    print("=" * 92)
    for k in ["cos", "agree", "margin", "fused"]:
        vals = np.concatenate(sig_pool[k][0])
        mis_all = np.concatenate(sig_pool[k][1])
        pooled = roc_auc_score(mis_all, -vals)
        pe = np.array(per_ep_auroc[k])
        print(f"  {k:7s}: AUROC pooled={pooled:.4f}  per-ep={pe.mean():.4f} ± {pe.std():.4f}")
    # paired test fused vs cos
    d = np.array(per_ep_auroc["fused"]) - np.array(per_ep_auroc["cos"])
    p = sps.wilcoxon(d).pvalue
    print(f"  fused vs cos (per-episode AUROC): mean delta = {d.mean():+.4f}, Wilcoxon p = {p:.4f}")


if __name__ == "__main__":
    main()

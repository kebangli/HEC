#!/usr/bin/env python3
"""Minimal method validation, fully OFFLINE (no backbone, CPU), reusing the
frozen features saved by exp_noisy_label_2x2.py.

Fixed per episode (from archives, identical to all previous experiments):
  EuroSAT 4-shot, 60 episodes (seed=123), 25% label noise (same realization),
  clean Top-20 heads, same support/query features.

Prototype estimators compared (only the class-mean computation changes;
covariance is refit centered at the resulting prototypes, HEC shrinkage
unchanged, prior from the labels in use):
  orig        : arithmetic mean (= original HEC)
  oracle_correct : refit with clean labels (upper bound)
  oracle_drop : refit on samples where noisy==clean (oracle detector)
  oracle_means : clean means + noisy-grouped covariance ("fix means only" ceiling)
  spat_med    : geometric (spatial) median per class/head (Weiszfeld)
  trimmed     : drop the sample farthest from the class mean, average the rest
  weighted_g1 : multi-head reliability weights (LOO cosine, averaged over the
                20 heads), gamma=1
  weighted_g3 : same with gamma=3 (sharpened)

Covariance shrinkage variants (on top of every robust prototype):
  +fix4 / +fix16 : COV_REG multiplier 4 / 16 (global, fixed)
  +adaptive      : per-head multiplier from {0.25,1,4,16} chosen by support
                   leave-one-out self-classification accuracy (no query labels)

Cross-checks: orig noisy must reproduce the offline 60-ep GDA value (~68.40%),
orig clean ~79.83%.
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
COV_REG = 1.0
GAMMAS = [1, 3]
SHRINK_GRID = [0.25, 1.0, 4.0, 16.0]
FIXED_MULTS = [4.0, 16.0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

clf = ClassifierHEC_V(input_mode="cls", k=K, tau=TAU, cov_reg=COV_REG,
                      ensemble_method="mean_prob")


# ---------------- prototype estimators ----------------

def proto_mean(Xn, y_idx, K):
    t, C = Xn.shape[1], Xn.shape[2]
    mus = torch.zeros(t, K, C, device=Xn.device)
    for k in range(K):
        m = y_idx == k
        if m.any():
            mus[:, k, :] = Xn[m].mean(0)
    return mus


def proto_spatial_median(Xn, y_idx, K, iters=50):
    mus = proto_mean(Xn, y_idx, K).clone()
    for k in range(K):
        m = y_idx == k
        n = int(m.sum())
        if n <= 1:
            continue
        Xk = Xn[m]                                  # [n,t,C]
        mu = mus[:, k, :]                           # [t,C]
        for _ in range(iters):                      # Weiszfeld
            d = (Xk - mu[None]).norm(dim=-1).clamp_min(1e-8)   # [n,t]
            w = 1.0 / d
            mu = (Xk * w[..., None]).sum(0) / w.sum(0)[..., None]
        mus[:, k, :] = mu
    return mus


def proto_trimmed(Xn, y_idx, K):
    mus = proto_mean(Xn, y_idx, K)
    out = mus.clone()
    t = Xn.shape[1]
    for k in range(K):
        m = y_idx == k
        n = int(m.sum())
        if n <= 2:
            continue
        Xk = Xn[m]                                  # [n,t,C]
        d = (Xk - mus[:, k, :][None]).norm(dim=-1)  # [n,t]
        drop_idx = d.argmax(0)                      # [t] per-head argmax
        S = Xk.sum(0)                               # [t,C]
        dropped = Xk[drop_idx, torch.arange(t)]     # [t,C]
        out[:, k, :] = (S - dropped) / (n - 1)
    return out


def proto_weighted(Xn, y_idx, K, gamma):
    """Multi-head reliability weights: LOO cosine of each sample to its own
    class (noisy) prototype, averaged over the Top-20 heads -> scalar weight."""
    t, C = Xn.shape[1], Xn.shape[2]
    S = torch.zeros(t, K, C, device=Xn.device)
    n = torch.zeros(K, device=Xn.device)
    for k in range(K):
        m = y_idx == k
        if m.any():
            S[:, k, :] = Xn[m].sum(0)
            n[k] = int(m.sum())
    mu_loo = (S[:, y_idx, :] - Xn.permute(1, 0, 2)) / (n[y_idx] - 1).clamp_min(1)[None, :, None]
    cosw = F.cosine_similarity(Xn.permute(1, 0, 2), mu_loo, dim=-1).permute(1, 0)  # [Ns,t]
    w = cosw.mean(dim=1).clamp_min(1e-4) ** gamma   # [Ns] multi-head consensus
    mus = torch.zeros(t, K, C, device=Xn.device)
    for k in range(K):
        m = y_idx == k
        if m.any():
            mus[:, k, :] = (Xn[m] * w[m][:, None, None]).sum(0) / w[m].sum()
    return mus


# ---------------- GDA fit / predict (mirrors ClassifierHEC_V math) -----------

def fit_gda(Xn, y_idx, K, proto_fn=None, means_override=None, cov_mult=1.0):
    means = means_override if means_override is not None else proto_fn(Xn, y_idx, K)
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


def adaptive_covinv(Xn, y_idx, K):
    """Per-head COV_REG multiplier chosen on the support set by leave-one-out
    self-classification accuracy (mean prototypes; no query labels used)."""
    N, t, Cdim = Xn.shape
    means, _, log_prior = fit_gda(Xn, y_idx, K, proto_fn=proto_mean, cov_mult=1.0)
    S = torch.zeros(t, K, Cdim, device=Xn.device)
    n = torch.zeros(K, device=Xn.device)
    for k in range(K):
        m = y_idx == k
        if m.any():
            S[:, k, :] = Xn[m].sum(0)
            n[k] = int(m.sum())
    mu_loo_own = (S[:, y_idx, :] - Xn.permute(1, 0, 2)) / (n[y_idx] - 1).clamp_min(1)[None, :, None]
    M = means[None].expand(N, t, K, Cdim).clone()                # [Ns,t,K,C]
    M[torch.arange(N, device=Xn.device), :, y_idx, :] = mu_loo_own.permute(1, 0, 2)
    best_s = torch.full((t,), SHRINK_GRID[0], device=Xn.device)
    best_acc = torch.full((t,), -1.0, device=Xn.device)
    cov_invs = {}
    for s in SHRINK_GRID:
        _, cov_inv_s, _ = fit_gda(Xn, y_idx, K, proto_fn=proto_mean, cov_mult=s)
        xSmu = torch.einsum('ntc,tcf,ntkf->ntk', Xn, cov_inv_s, M)
        muSmu = torch.einsum('ntkc,tcf,ntkf->ntk', M, cov_inv_s, M)
        f = xSmu - 0.5 * muSmu + log_prior[None, None, :]
        acc = (f.argmax(-1) == y_idx[:, None]).float().mean(0)   # [t]
        imp = acc >= best_acc                                     # ties -> larger s
        best_s[imp] = s
        best_acc[imp] = acc[imp]
        cov_invs[s] = cov_inv_s
    cov_inv = torch.stack([cov_invs[float(best_s[i])][i] for i in range(t)], dim=0)
    return cov_inv, best_s


# ---------------- main ----------------

def main():
    print(f"device = {DEVICE}", flush=True)
    records = np.load(f"{FEAT_DIR}/noisy_label_2x2_records.npy",
                      allow_pickle=True).tolist()[:N_EP]
    feat = np.load(f"{FEAT_DIR}/features.npz")
    labels_list = np.load(f"{FEAT_DIR}/feat_labels.npy",
                          allow_pickle=True).tolist()[:N_EP]
    support_A = feat["support_A"][:N_EP]
    query_A = feat["query_A"][:N_EP]

    PROTOS = {
        "spat_med": proto_spatial_median,
        "trimmed": proto_trimmed,
        "weighted_g1": lambda X, y, K: proto_weighted(X, y, K, 1),
        "weighted_g3": lambda X, y, K: proto_weighted(X, y, K, 3),
    }

    all_rows = []
    for it in range(N_EP):
        rec, lab = records[it], labels_list[it]
        A2 = torch.from_numpy(support_A[it].astype(np.float32)).to(DEVICE)
        Aq2 = torch.from_numpy(query_A[it].astype(np.float32)).to(DEVICE)
        clean_labels = torch.from_numpy(lab["clean_labels"]).long().to(DEVICE)
        noisy_labels = torch.from_numpy(rec["r25"]["noisy_labels"]).long().to(DEVICE)
        yq_np = lab["query_label"].astype(np.int64)
        sel = torch.from_numpy(rec["r25"]["sel_clean"]).long().to(DEVICE)
        Xn = F.normalize(A2[:, sel, :], dim=-1)      # [Ns, t, C]
        Xq = Aq2[:, sel, :]

        row = {"episode": it}

        for cond, labels in [("clean", clean_labels), ("noisy", noisy_labels)]:
            lu = torch.unique(labels)
            y_idx = torch.searchsorted(lu, labels)

            # original HEC (mean prototype)
            means, cov_inv, lp = fit_gda(Xn, y_idx, K, proto_fn=proto_mean)
            row[f"orig_{cond}"] = predict(Xq, means, cov_inv, lp, lu, yq_np)

            # robust prototypes
            for name, fn in PROTOS.items():
                means_r, cov_inv_r, lp_r = fit_gda(Xn, y_idx, K, proto_fn=fn)
                row[f"{name}_{cond}"] = predict(Xq, means_r, cov_inv_r, lp_r, lu, yq_np)

            # oracles (noisy-relevant only, but evaluate under both conditions)
            if cond == "noisy":
                # Oracle-Drop: refit on the unflipped subset (noisy labels kept)
                keep = (noisy_labels == clean_labels)
                Xk = Xn[keep]
                lu_k = torch.unique(noisy_labels[keep])
                yk_idx = torch.searchsorted(lu_k, noisy_labels[keep])
                mk, ck, lk = fit_gda(Xk, yk_idx, K, proto_fn=proto_mean)
                row["oracle_drop_noisy"] = predict(Xq, mk, ck, lk, lu_k, yq_np)
                # Oracle-Correct: clean-label refit (= clean baseline)
                luc = torch.unique(clean_labels)
                yc_idx = torch.searchsorted(luc, clean_labels)
                mc, cc, lc = fit_gda(Xn, yc_idx, K, proto_fn=proto_mean)
                row["oracle_correct_noisy"] = predict(Xq, mc, cc, lc, luc, yq_np)
                # Oracle-CleanMeans: clean means + noisy-grouped covariance
                mn, cn, ln = fit_gda(Xn, y_idx, K, means_override=mc)
                row["oracle_means_noisy"] = predict(Xq, mn, cn, ln, lu, yq_np)

        # shrinkage variants on every robust prototype (noisy condition)
        lu = torch.unique(noisy_labels)
        y_idx = torch.searchsorted(lu, noisy_labels)
        for name, fn in PROTOS.items():
            means_r, _, lp_r = fit_gda(Xn, y_idx, K, proto_fn=fn)
            for mult in FIXED_MULTS:
                _, cm, _ = fit_gda(Xn, y_idx, K, proto_fn=fn, cov_mult=mult)
                row[f"{name}_fix{int(mult)}_noisy"] = predict(Xq, means_r, cm, lp_r, lu, yq_np)
            cov_inv_ad, _ = adaptive_covinv(Xn, y_idx, K)
            row[f"{name}_adaptive_noisy"] = predict(Xq, means_r, cov_inv_ad, lp_r, lu, yq_np)

        all_rows.append(row)
        if it % 10 == 0 or it == N_EP - 1:
            print(f"ep {it}: orig_noisy={100*row['orig_noisy']:.1f} "
                  f"trim={100*row['trimmed_noisy']:.1f} "
                  f"w3={100*row['weighted_g3_noisy']:.1f} "
                  f"odrop={100*row['oracle_drop_noisy']:.1f}", flush=True)

    np.save("/data2/lbk/HEC_repro/logs/repro_eurosat_noisy/gda3_n5/"
            "robust_proto_rows.npy", all_rows)

    # ---------------- aggregate ----------------
    def arr(key):
        return np.array([r[key] for r in all_rows])

    print("\n" + "=" * 86)
    print(f"MAIN TABLE (n = {N_EP}, 25% noise, clean Top-20 heads, offline)")
    print("=" * 86)
    print(f"{'method':24s} {'clean':>8s} {'noisy':>8s} {'drop':>8s} {'recov':>8s} {'p_vs_orig':>10s}")
    order = ["orig", "oracle_correct", "oracle_drop", "oracle_means",
             "spat_med", "trimmed", "weighted_g1", "weighted_g3"]
    orig_noisy = arr("orig_noisy")
    for name in order:
        ck = f"{name}_clean"
        nk = f"{name}_noisy"
        if nk not in all_rows[0]:
            continue
        c = arr(ck) if ck in all_rows[0] else np.full(N_EP, np.nan)
        n_ = arr(nk)
        drop = c - n_
        recov = n_ - orig_noisy
        try:
            p = stats.wilcoxon(n_, orig_noisy).pvalue
        except Exception:
            p = float("nan")
        print(f"{name:24s} {100*np.nanmean(c):7.2f}% {100*n_.mean():7.2f}% "
              f"{100*np.nanmean(drop):+7.2f} {100*recov.mean():+7.2f} {p:10.4f}")

    print("\n" + "=" * 86)
    print("SHRINKAGE VARIANTS (noisy condition; prototype method x cov shrinkage)")
    print("=" * 86)
    print(f"{'method':28s} {'noisy':>8s} {'recov':>8s} {'p_vs_orig':>10s}")
    for name in PROTOS:
        base = arr(f"{name}_noisy")
        for suffix, label in [("fix4", "fix4"), ("fix16", "fix16"), ("adaptive", "adaptive")]:
            key = f"{name}_{suffix}_noisy"
            v = arr(key)
            try:
                p = stats.wilcoxon(v, orig_noisy).pvalue
            except Exception:
                p = float("nan")
            print(f"{name + '+' + label:28s} {100*v.mean():7.2f}% {100*(v.mean()-orig_noisy.mean()):+7.2f} {p:10.4f}")

    print("\nper-episode orig vs best robust (see saved npy for full detail)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Detector validation, fully OFFLINE (no backbone), reusing frozen features.

Question: can Multi-head LOO Reliability (the weighting signal of the
weighted-prototype method) accurately identify mislabeled support samples
under 4-shot / 25% label noise?

Reliability (identical to proto_weighted in robust_prototype_eval.py):
    w_i = mean over the clean Top-20 heads of cos(x_i, mu_{-i}[y_i])
where mu_{-i}[k] = mean of class k's members (noisy labels) excluding sample i.

Ground truth: a sample is mislabeled iff noisy_label != clean_label
(10 per episode, 600 over 60 episodes).

Metrics:
  1. reliability distributions: mislabeled vs correctly-labeled (noisy regime)
     + reference distribution under clean labels (all correct)
  2. AUROC of (low reliability -> mislabeled), pooled + per-episode
  3. AUPRC (prevalence 25%), pooled + per-episode
  4. per-class lowest-reliability hit rate: in each noisy class containing
     >=1 mislabeled member, is the argmin-reliability member mislabeled?
     (+ chance level = mean m_k/n_k over eligible classes)
     and flagging-bottom-25% precision / recall
  5. majority-vote alternative class: per head, argmax cosine over classes
     (own class uses the LOO prototype); majority vote over the 20 heads.
     - on mislabeled samples: vote == true (clean) label ?
     - on correct samples  : vote == current label (stability) ?
     - combined relabel-all-by-vote accuracy
     - per-head vote accuracy on mislabeled samples (context)
"""
import sys
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as sps
from sklearn.metrics import roc_auc_score, average_precision_score

FEAT_DIR = "/data2/lbk/HEC_repro/logs/repro_eurosat_noisy/n60_2x2"
N_EP = int(sys.argv[1]) if len(sys.argv) > 1 else 60
K = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def class_sums(Xn, y_idx, K):
    t, C = Xn.shape[1], Xn.shape[2]
    S = torch.zeros(t, K, C, device=Xn.device)
    n = torch.zeros(K, device=Xn.device)
    for k in range(K):
        m = y_idx == k
        if m.any():
            S[:, k, :] = Xn[m].sum(0)
            n[k] = int(m.sum())
    return S, n


def loo_reliability(Xn, y_idx, K):
    """w_i = mean_t cos(x_i, mu_{-i}[y_i])  (multi-head LOO reliability)."""
    S, n = class_sums(Xn, y_idx, K)
    mu_loo = (S[:, y_idx, :] - Xn.permute(1, 0, 2)) / (n[y_idx] - 1).clamp_min(1)[None, :, None]
    cosw = F.cosine_similarity(Xn.permute(1, 0, 2), mu_loo, dim=-1).permute(1, 0)  # [Ns,t]
    return cosw.mean(dim=1)          # [Ns]


def loo_vote(Xn, y_idx, K):
    """Per-head class prediction with LOO own-class prototype (cosine argmax),
    then majority vote over heads. Returns (vote_idx [Ns], per_head [Ns,t])."""
    S, n = class_sums(Xn, y_idx, K)
    Ns, t, C = Xn.shape
    mu_full = S / n.clamp_min(1)[None, :, None]                     # [t,K,C]
    mu_loo_own = (S[:, y_idx, :] - Xn.permute(1, 0, 2)) / (n[y_idx] - 1).clamp_min(1)[None, :, None]
    # scores[i,t,k] = cos(x_i[t], mu_k) with own class replaced by LOO mean
    scores = torch.einsum('ntc,tkc->ntk', Xn, mu_full)              # [Ns,t,K]
    own_score = torch.einsum('ntc,ntc->nt', Xn, mu_loo_own.permute(1, 0, 2))  # [Ns,t]
    scores[torch.arange(Ns, device=Xn.device), :, y_idx] = own_score
    # mu_loo_own is [t,Ns,C] -> need [Ns,t,C]:
    per_head = scores.argmax(-1)                                    # [Ns,t]
    # majority vote over heads
    vote = torch.empty(Ns, dtype=torch.long, device=Xn.device)
    for i in range(Ns):
        vote[i] = torch.bincount(per_head[i], minlength=K).argmax()
    return vote, per_head


def main():
    print(f"device = {DEVICE}", flush=True)
    records = np.load(f"{FEAT_DIR}/noisy_label_2x2_records.npy",
                      allow_pickle=True).tolist()[:N_EP]
    feat = np.load(f"{FEAT_DIR}/features.npz")
    labels_list = np.load(f"{FEAT_DIR}/feat_labels.npy",
                          allow_pickle=True).tolist()[:N_EP]
    support_A = feat["support_A"][:N_EP]

    rel_mis, rel_ok, rel_clean = [], [], []      # pooled reliability values
    aurocs, auprcs = [], []
    argmin_hits, argmin_total, argmin_chance = 0, 0, []
    flag_prec, flag_rec = [], []
    vote_fix, vote_fix_n = 0, 0                   # mislabeled: vote == true label
    vote_stay, vote_stay_n = 0, 0                 # correct: vote == current label
    per_head_acc = []
    per_ep = []

    for it in range(N_EP):
        rec, lab = records[it], labels_list[it]
        A2 = torch.from_numpy(support_A[it].astype(np.float32)).to(DEVICE)
        clean_labels = torch.from_numpy(lab["clean_labels"]).long().to(DEVICE)
        noisy_labels = torch.from_numpy(rec["r25"]["noisy_labels"]).long().to(DEVICE)
        sel = torch.from_numpy(rec["r25"]["sel_clean"]).long().to(DEVICE)
        Xn = F.normalize(A2[:, sel, :], dim=-1)

        lu = torch.unique(noisy_labels)
        y_idx = torch.searchsorted(lu, noisy_labels)
        cy_idx = torch.searchsorted(lu, clean_labels)   # true class index (same space)
        mis = (noisy_labels != clean_labels)            # [Ns]

        # reliability (noisy regime)
        w = loo_reliability(Xn, y_idx, lu.numel())
        rel_mis.append(w[mis].cpu().numpy())
        rel_ok.append(w[~mis].cpu().numpy())

        # reference: reliability under clean labels (all correct)
        lu_c = torch.unique(clean_labels)
        y_idx_c = torch.searchsorted(lu_c, clean_labels)
        w_clean = loo_reliability(Xn, y_idx_c, lu_c.numel())
        rel_clean.append(w_clean.cpu().numpy())

        # AUROC / AUPRC (detect mislabeled via LOW reliability)
        w_np = w.cpu().numpy()
        mis_np = mis.cpu().numpy()
        aurocs.append(roc_auc_score(mis_np, -w_np))
        auprcs.append(average_precision_score(mis_np, -w_np))

        # per-class argmin hit
        Kc = lu.numel()
        for k in range(Kc):
            members = (y_idx == k).nonzero().flatten()
            m_in = mis[members]
            if m_in.any():
                argmin_total += 1
                argmin_chance.append(float(m_in.float().mean()))
                if bool(mis[members[w[members].argmin()]]):
                    argmin_hits += 1

        # flag bottom-25% (10 samples) as mislabeled
        n_flag = int(mis_np.sum())
        order = np.argsort(w_np)
        flagged = order[:n_flag]
        flag_prec.append(float(mis_np[flagged].mean()))
        flag_rec.append(float(mis_np[flagged].sum() / mis_np.sum()))

        # majority vote
        vote, per_head = loo_vote(Xn, y_idx, Kc)
        per_head_ok = (per_head[mis] == cy_idx[mis, None]).float().mean().item()
        per_head_acc.append(per_head_ok)
        m = mis
        vote_fix += int((vote[m] == cy_idx[m]).sum())
        vote_fix_n += int(m.sum())
        vote_stay += int((vote[~m] == y_idx[~m]).sum())
        vote_stay_n += int((~m).sum())

        per_ep.append({
            "episode": it, "auroc": aurocs[-1], "auprc": auprcs[-1],
            "flag_prec": flag_prec[-1], "flag_rec": flag_rec[-1],
        })

    # ---------------- aggregate ----------------
    rel_mis = np.concatenate(rel_mis); rel_ok = np.concatenate(rel_ok)
    rel_clean = np.concatenate(rel_clean)
    aurocs = np.array(aurocs); auprcs = np.array(auprcs)
    flag_prec = np.array(flag_prec); flag_rec = np.array(flag_rec)
    per_head_acc = np.array(per_head_acc)

    # pooled AUROC/AUPRC
    all_w = np.concatenate([rel_mis, rel_ok])
    all_m = np.concatenate([np.ones_like(rel_mis), np.zeros_like(rel_ok)])
    pooled_auroc = roc_auc_score(all_m, -all_w)
    pooled_auprc = average_precision_score(all_m, -all_w)

    print("\n" + "=" * 74)
    print(f"DETECTOR VALIDATION (n = {N_EP} episodes, 4-shot, 25% noise)")
    print("=" * 74)

    print("\n--- 1. RELIABILITY DISTRIBUTION ---")
    print(f"  mislabeled samples (n={rel_mis.size}): mean={rel_mis.mean():.4f} std={rel_mis.std():.4f} median={np.median(rel_mis):.4f}")
    print(f"  correct   samples (n={rel_ok.size}): mean={rel_ok.mean():.4f} std={rel_ok.std():.4f} median={np.median(rel_ok):.4f}")
    print(f"  clean-label reference (all correct): mean={rel_clean.mean():.4f} std={rel_clean.std():.4f}")
    d = rel_ok.mean() - rel_mis.mean()
    print(f"  separation (correct - mislabeled): {d:+.4f}  "
          f"(|d| / pooled_std = {abs(d)/all_w.std():.2f})")

    print("\n--- 2/3. DETECTION PERFORMANCE (low reliability -> mislabeled) ---")
    print(f"  AUROC : pooled={pooled_auroc:.4f} | per-episode mean={aurocs.mean():.4f} ± {aurocs.std():.4f} "
          f"(min={aurocs.min():.3f}, max={aurocs.max():.3f})")
    print(f"  AUPRC : pooled={pooled_auprc:.4f} | per-episode mean={auprcs.mean():.4f} ± {auprcs.std():.4f} "
          f"(prevalence=0.25, chance AUPRC≈0.25)")
    print(f"  episodes with AUROC >= 0.9: {(aurocs>=0.9).sum()}/{N_EP} | >= 0.8: {(aurocs>=0.8).sum()}/{N_EP}")

    print("\n--- 4. PER-CLASS LOWEST-RELIABILITY HIT ---")
    chance = float(np.mean(argmin_chance))
    print(f"  eligible (episode,class) groups with >=1 mislabeled: {argmin_total}")
    print(f"  argmin-reliability member IS mislabeled: {argmin_hits}/{argmin_total} = {argmin_hits/argmin_total:.3f} "
          f"(chance ≈ {chance:.3f})")
    print(f"  flag-bottom-{int(1)}0 (exactly 25%): precision={flag_prec.mean():.3f} recall={flag_rec.mean():.3f}")

    print("\n--- 5. MAJORITY-VOTE ALTERNATIVE CLASS (over 20 heads, cosine LOO) ---")
    print(f"  mislabeled: vote == TRUE label: {vote_fix}/{vote_fix_n} = {vote_fix/vote_fix_n:.3f}")
    print(f"  correct   : vote == current label (stays): {vote_stay}/{vote_stay_n} = {vote_stay/vote_stay_n:.3f}")
    relabel_acc = (vote_fix + vote_stay) / (vote_fix_n + vote_stay_n)
    print(f"  relabel-all-by-vote accuracy: {relabel_acc:.3f}")
    print(f"  per-head vote accuracy on mislabeled (mean over heads): {per_head_acc.mean():.3f} ± {per_head_acc.std():.3f}")

    np.save("/data2/lbk/HEC_repro/logs/repro_eurosat_noisy/gda3_n5/"
            "noise_detector_rows.npy", per_ep)
    print("\nper-episode records saved -> noise_detector_rows.npy")


if __name__ == "__main__":
    main()

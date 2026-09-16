"""RouteHead-inspired Dynamic Head Routing for HEC.

Transfers the query-dependent head selection idea of
"Learning to Route Queries to Heads for Attention-based Re-ranking with Large
Language Models" (RouteHead, SIGIR'26) to HEC's few-shot vision-language
classification setting.

RouteHead recap (what we transfer, no code copied):
  1. Restrict the search space to a candidate head pool (top-K by per-head utility).
  2. Build per-input pseudo labels with an offline search: greedy forward
     selection with early stopping, then swap-based local refinement.
  3. Train a lightweight router (head embeddings x input embedding, bilinear
     interaction + sigmoid) with BCE + L1 sparsity on the pseudo labels.
  4. At inference, select heads per input and aggregate only those heads' signals.

HEC adaptation (where the settings differ):
  * HEC queries are unlabeled, so per-query pseudo labels cannot be built
    directly. All pseudo labels are derived from SUPPORT data only:
      - sample_router: per-support-sample multi-hot labels, searched on
        cross-fit (out-of-fold) per-head GDA posteriors; a router maps query
        features (x_norm_clstoken) to head activations.
      - class_cond: per-class head sets searched with a contrastive margin
        objective (the class's ensemble margin over its support samples).
        Training-free at the query side: each class is ensembled with its own
        heads. This is the more natural fit for classification.
      - knn: training-free per-query routing: score heads by the OOF margin
        contribution of the query's nearest support neighbours.
  * Search objective: RouteHead maximises the ranking quality of the AGGREGATED
    signal. The classification analogue is the decision margin of the mean_prob
    ensemble, L(S) = P_S[y] - max_{k!=y} P_S[k] with P_S = mean of per-head
    posteriors — NOT the mean true-class probability, whose greedy search
    degenerates to a single head (any weaker head dilutes the mean).
  * Selection bias: in-sample head scores overfit the support GDA (see prior
    HEC_repro findings), so all pseudo labels use 4-fold cross-fit OOF posteriors.
  * Ensembles follow the official HEC path (per-head GDA `get_pred_labels` on
    the selected subset, then mean_prob) to stay numerically consistent with
    the baseline.

Public API:
    cross_fit_oof(clf, A, labels, n_folds)      -> OOF posteriors + OOF mean score
    greedy_margin_sets(Z, targets, budget, ...) -> per-item greedy+swap search
    class_margin_sets(Z, y_idx, budget, ...)    -> per-class greedy+swap search
    HeadRouter, train_router                    -> RouteHead-style router
    group_ensemble_official                     -> per-query head sets, official path
    class_ensemble_official                     -> per-class head sets, official path
    ClassifierHEC_V_Dynamic                     -> hydra-instantiable classifier
                                                   (routing=off reproduces HEC-V)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .classifier import ClassifierHEC_V

__all__ = [
    "cross_fit_oof",
    "greedy_margin_sets",
    "class_margin_sets",
    "head_margin_scores",
    "HeadRouter",
    "train_router",
    "group_ensemble_official",
    "class_ensemble_official",
    "oracle_topk_sets",
    "ClassifierHEC_V_Dynamic",
]


# --------------------------------------------------------------------------- #
# 1. Cross-fit OOF per-head posteriors (unbiased pseudo-label signal)         #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def cross_fit_oof(clf, A, labels, n_folds=4):
    """Cross-fitted per-head GDA posteriors on the support set.

    For each fold, the per-head GDA is fit on the other folds and evaluated on
    the held-out fold, giving out-of-fold posteriors z_oof [Ns, M, K]. The
    class-balanced fold assignment (round-robin within each class) guarantees
    every class appears in every fit fold when n_support >= n_folds.

    Returns:
        z_oof     [Ns, M, K] float32 (device of A)
        true_prob [Ns, M]    OOF posterior of each sample's true class per head
        mean_oof  [M]        OOF mean head score (unbiased static ranking)
        y_idx     [Ns]       index of each label into the sorted unique labels
    """
    Ns = A.shape[0]
    labels_unique = torch.unique(labels)
    y_idx = torch.searchsorted(labels_unique, labels)
    K = labels_unique.numel()
    M = A.shape[1]

    # class-balanced fold assignment
    fold_of = torch.empty(Ns, dtype=torch.long, device=A.device)
    for lbl in labels_unique:
        idxs = (labels == lbl).nonzero(as_tuple=True)[0]
        fold_of[idxs] = torch.arange(idxs.numel(), device=A.device) % n_folds

    z_oof = torch.zeros(Ns, M, K, device=A.device, dtype=torch.float32)
    for f in range(n_folds):
        fit_idx = (fold_of != f).nonzero(as_tuple=True)[0]
        val_idx = (fold_of == f).nonzero(as_tuple=True)[0]
        if val_idx.numel() == 0:
            continue
        hd = clf.get_rank_scores(A[fit_idx], labels[fit_idx])
        # fold's unique labels == global unique labels (each class has >=1 fit sample)
        assert hd["labels_unique"].numel() == K, "fold lost a class; reduce n_folds"
        z = clf.get_pred_labels(
            A[val_idx], hd["head_means"], hd["head_cov_inv"], hd["log_prior"]
        )[0]  # [n_val, M, K]
        z_oof[val_idx] = z.float()

    true_prob = z_oof[torch.arange(Ns, device=A.device), :, y_idx]  # [Ns, M]
    mean_oof = true_prob.mean(dim=0)                                # [M]
    return z_oof, true_prob, mean_oof, y_idx


def head_margin_scores(z_oof, y_idx):
    """Per-head decision-margin contribution: z[i,m,y_i] - max_{k!=y_i} z[i,m,k].

    z_oof: [Ns, M, K]; returns [Ns, M].
    """
    Ns, M, K = z_oof.shape
    off = z_oof.scatter(2, y_idx[:, None, None].expand(Ns, M, 1),
                        float("-inf")).max(dim=2).values
    return z_oof[torch.arange(Ns, device=z_oof.device), :, y_idx] - off


# --------------------------------------------------------------------------- #
# 2. Offline search: greedy forward selection + swap refinement               #
#    (objective = decision margin of the mean_prob ensemble)                  #
# --------------------------------------------------------------------------- #

def _margin(probs, y):
    """probs: [..., K] ensemble class probabilities; margin of class y."""
    others = probs.copy()
    others[..., y] = -np.inf
    return probs[..., y] - others.max(axis=-1)


def greedy_margin_sets(Z, targets, budget, eps=0.0, min_heads=4,
                       max_swap_iters=5, do_swap=True):
    """Per-item head-set search maximising the ensemble decision margin.

    Z: [N, P, K] per-item per-head class posteriors (numpy).
    targets: [N] class index whose margin is maximised.
    L(i, S) = P_S[i, y] - max_{k != y} P_S[i, k],  P_S[i, k] = mean_{m in S} Z[i, m, k].

    Greedy forward selection with early stopping, then swap-based local
    refinement (RouteHead Sec. 3.1). Adaptive min_heads <= |S| <= budget:
    the first min_heads heads are always taken (variance reduction of small
    ensembles is invisible to a per-sample surrogate), beyond that a head is
    added only if it strictly improves the margin.

    Returns bool array sets [N, P].
    """
    N, P, K = Z.shape
    Z = np.asarray(Z, dtype=np.float64)
    sets = np.zeros((N, P), dtype=bool)
    for i in range(N):
        Zi = Z[i]                                   # [P, K]
        y = int(targets[i])
        chosen = []
        ssum = np.zeros(K)                          # sum of posteriors over S
        cur = -np.inf
        # greedy forward selection with early stopping
        while len(chosen) < budget:
            means = (ssum[None, :] + Zi) / (len(chosen) + 1)   # [P, K]
            others = means.copy()
            others[:, y] = -np.inf
            marg = means[:, y] - others.max(axis=1)            # [P]
            if chosen:
                marg[np.asarray(chosen, dtype=np.int64)] = -np.inf
            m = int(np.argmax(marg))
            if len(chosen) < min_heads or marg[m] > cur + eps:
                chosen.append(m)
                cur = float(marg[m])
                ssum = ssum + Zi[m]
            else:
                break  # early stopping
        # swap-based local refinement
        if do_swap and len(chosen) >= 2:
            sel = np.zeros(P, dtype=bool)
            sel[chosen] = True
            for _ in range(max_swap_iters):
                means_cur = ssum / len(chosen)
                others_cur = means_cur.copy()
                others_cur[y] = -np.inf
                cur = float(means_cur[y] - others_cur.max())
                best_gain, best_pair = 0.0, None
                for u in chosen:
                    rest = ssum - Zi[u]
                    means = (rest[None, :] + Zi) / len(chosen)     # [P, K]
                    others = means.copy()
                    others[:, y] = -np.inf
                    marg = means[:, y] - others.max(axis=1)
                    marg[sel] = -np.inf
                    v = int(np.argmax(marg))
                    gain = float(marg[v] - cur)
                    if gain > best_gain:
                        best_gain, best_pair = gain, (u, v)
                if best_pair is None or best_gain <= eps:
                    break
                u, v = best_pair
                sel[u] = False
                sel[v] = True
                ssum = ssum - Zi[u] + Zi[v]
                chosen = [m for m in chosen if m != u] + [v]
        sets[i, chosen] = True
    return sets


def class_margin_sets(Z, y_idx, budget, eps=0.0, min_heads=4,
                      max_swap_iters=5, do_swap=True):
    """Per-class head sets maximising the class's mean support margin.

    Z: [Ns, P, K] OOF posteriors restricted to the pool; y_idx: [Ns].
    For class c:
        L(c, S) = mean_{i in c} [ P_S[i, c] - max_{k != c} P_S[i, k] ],
        P_S[i, k] = mean_{m in S} Z[i, m, k].
    The max is taken per sample (inside the row mean), so this is stricter and
    better aligned with the decision rule than averaging posteriors first.
    Adaptive min_heads <= |S(c)| <= budget (see greedy_margin_sets).

    Returns bool array sets [K, P].
    """
    Ns, P, K = Z.shape
    Z = np.asarray(Z, dtype=np.float64)
    y_np = np.asarray(y_idx)
    sets = np.zeros((K, P), dtype=bool)
    for c in range(K):
        rows = Z[y_np == c]                          # [n_c, P, K]
        n_c = rows.shape[0]
        chosen = []
        ssum = np.zeros((n_c, K))                    # [n_c, K]
        cur = -np.inf
        while len(chosen) < budget:
            means = (ssum[:, None, :] + rows) / (len(chosen) + 1)  # [n_c, P, K]
            others = means.copy()
            others[:, :, c] = -np.inf
            marg = (means[:, :, c] - others.max(axis=2)).mean(axis=0)  # [P]
            if chosen:
                marg[np.asarray(chosen, dtype=np.int64)] = -np.inf
            m = int(np.argmax(marg))
            if len(chosen) < min_heads or marg[m] > cur + eps:
                chosen.append(m)
                cur = float(marg[m])
                ssum = ssum + rows[:, m, :]
            else:
                break
        if do_swap and len(chosen) >= 2:
            sel = np.zeros(P, dtype=bool)
            sel[chosen] = True
            for _ in range(max_swap_iters):
                means_cur = ssum / len(chosen)
                others_cur = means_cur.copy()
                others_cur[:, c] = -np.inf
                cur = float((means_cur[:, c] - others_cur.max(axis=1)).mean())
                best_gain, best_pair = 0.0, None
                for u in chosen:
                    rest = ssum - rows[:, u, :]
                    means = (rest[:, None, :] + rows) / len(chosen)  # [n_c, P, K]
                    others = means.copy()
                    others[:, :, c] = -np.inf
                    marg = (means[:, :, c] - others.max(axis=2)).mean(axis=0)
                    marg[sel] = -np.inf
                    v = int(np.argmax(marg))
                    gain = float(marg[v] - cur)
                    if gain > best_gain:
                        best_gain, best_pair = gain, (u, v)
                if best_pair is None or best_gain <= eps:
                    break
                u, v = best_pair
                sel[u] = False
                sel[v] = True
                ssum = ssum - rows[:, u, :] + rows[:, v, :]
                chosen = [m for m in chosen if m != u] + [v]
        sets[c, chosen] = True
    return sets


# --------------------------------------------------------------------------- #
# 3. Router (RouteHead Sec. 3.2)                                              #
# --------------------------------------------------------------------------- #

class HeadRouter(nn.Module):
    """Bilinear query-to-head router:  alpha_m = w2^T (e_m * (W1 x + b)),
    p_m = sigmoid(alpha_m), independent activations across heads.

    The query embedding x (x_norm_clstoken, 3584-d for Qwen2-VL) is first
    compressed by a FROZEN random projection: with only tens of support
    samples per episode, a full 3584-d input layer would overfit; a seeded
    random projection preserves geometry (J-L) and keeps the trainable router
    tiny, mirroring RouteHead's low-rank embedding design.
    """

    def __init__(self, dim_query, n_pool, dim_head=64, dim_proj=256):
        super().__init__()
        g = torch.Generator().manual_seed(0)  # projection is deterministic
        self.register_buffer(
            "proj", torch.randn(dim_query, dim_proj, generator=g) / (dim_proj ** 0.5)
        )
        self.head_emb = nn.Parameter(torch.randn(n_pool, dim_head) * 0.02)
        self.w1 = nn.Linear(dim_proj, dim_head)
        self.w2 = nn.Parameter(torch.randn(dim_head) * 0.02)

    def forward(self, x):
        # x: [N, dim_query] (already normalised by the caller)
        h = self.w1(x @ self.proj)                                    # [N, d]
        alpha = torch.einsum("nd,d,pd->np", h, self.w2, self.head_emb)  # [N, P]
        return torch.sigmoid(alpha)


def train_router(feats, multi_hot, dim_head=64, lr=1e-3, epochs=300,
                 l1=0.05, weight_decay=1e-4, device="cuda"):
    """Train the router on support pseudo labels with BCE + L1 sparsity."""
    x = F.normalize(feats, dim=-1).float().to(device)
    y = torch.as_tensor(multi_hot, dtype=torch.float32, device=device)
    router = HeadRouter(x.shape[1], y.shape[1], dim_head=dim_head).to(device)
    opt = torch.optim.Adam(router.parameters(), lr=lr, weight_decay=weight_decay)
    router.train()
    for _ in range(epochs):
        p = router(x)
        loss = F.binary_cross_entropy(p, y) + l1 * p.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    router.eval()
    return router


# --------------------------------------------------------------------------- #
# 4. Ensembles on the official HEC path                                       #
# --------------------------------------------------------------------------- #

def group_ensemble_official(clf, Aq, hd, sets_pos):
    """Per-query head sets, evaluated group-wise on the official path.

    Queries sharing the same selected head set are evaluated together with a
    single `get_pred_labels` call on that subset (bitwise consistent with the
    official select-then-slice ensemble), then mean_prob within the set.

    sets_pos: list (len Nq) of pool-position lists/arrays.
    Returns P [Nq, K] float32.
    """
    Nq, K = Aq.shape[0], hd["log_prior"].shape[0]
    P = torch.zeros(Nq, K, device=Aq.device, dtype=torch.float32)
    groups = {}
    for n, s in enumerate(sets_pos):
        groups.setdefault(tuple(int(m) for m in s), []).append(n)
    for s, members in groups.items():
        idx = torch.as_tensor(list(s), dtype=torch.long, device=Aq.device)
        members_t = torch.as_tensor(members, dtype=torch.long, device=Aq.device)
        z = clf.get_pred_labels(
            Aq[members_t][:, idx, :], hd["head_means"][idx],
            hd["head_cov_inv"][idx], hd["log_prior"],
        )[0]                                    # [g, |S|, K]
        P[members_t] = z.mean(dim=1)
    return P


def class_ensemble_official(clf, Aq, hd, class_sets_pos):
    """Per-class head sets: P[n, c] = mean_{m in S(c)} z[n, m, c].

    One official-path `get_pred_labels` call per class on its own head subset.
    class_sets_pos: list (len K) of pool-position lists.
    Returns P [Nq, K] float32.
    """
    Nq, K = Aq.shape[0], hd["log_prior"].shape[0]
    P = torch.zeros(Nq, K, device=Aq.device, dtype=torch.float32)
    for c, s in enumerate(class_sets_pos):
        if len(s) == 0:  # should not happen (greedy always takes >=1 head)
            P[:, c] = hd["log_prior"][c]
            continue
        idx = torch.as_tensor(list(s), dtype=torch.long, device=Aq.device)
        z = clf.get_pred_labels(
            Aq[:, idx, :], hd["head_means"][idx], hd["head_cov_inv"][idx],
            hd["log_prior"],
        )[0]                                    # [Nq, |S(c)|, K]
        P[:, c] = z[:, :, c].mean(dim=1)
    return P


@torch.no_grad()
def oracle_topk_sets(clf, Aq, hd, pool, yq_idx, k):
    """DIAGNOSTIC ONLY (uses query labels): per-query top-k heads by the
    full-support GDA posterior of the query's true class. Upper bound of
    per-query routing under the same GDA models. Returns pool-position sets."""
    z = clf.get_pred_labels(Aq, hd["head_means"], hd["head_cov_inv"], hd["log_prior"])[0]
    Nq = z.shape[0]
    true_q = z[torch.arange(Nq, device=z.device), :, yq_idx]    # [Nq, M]
    return true_q[:, pool].topk(k, dim=1).indices.cpu().numpy()


@torch.no_grad()
def oracle_correct_topk_sets(clf, Aq, hd, pool, yq_idx, k):
    """DIAGNOSTIC ONLY (uses query labels): per-query top-k heads among those
    that INDIVIDUALLY classify the query correctly (ranked by true-class
    posterior), falling back to the highest true-class posterior. This is the
    realistic ceiling of per-query hard routing: a set of heads whose argmax
    agrees with the label."""
    z = clf.get_pred_labels(Aq, hd["head_means"], hd["head_cov_inv"], hd["log_prior"])[0]
    Nq = z.shape[0]
    yq = yq_idx.to(z.device)
    true_q = z[torch.arange(Nq, device=z.device), :, yq]        # [Nq, M]
    correct = (z.argmax(dim=-1) == yq[:, None])                  # [Nq, M]
    score = true_q.masked_fill(~correct, -1.0)
    return score[:, pool].topk(k, dim=1).indices.cpu().numpy()


# --------------------------------------------------------------------------- #
# 5. Hydra-instantiable classifier                                            #
# --------------------------------------------------------------------------- #

class ClassifierHEC_V_Dynamic(ClassifierHEC_V):
    """HEC-V with RouteHead-inspired dynamic head routing.

    routing:
        off           - official HEC-V static top-k (baseline, exact parent path)
        sample_router - RouteHead-style learned router (query-conditioned top-k)
        class_cond    - per-class margin head sets (training-free queries)
        knn           - training-free neighbour-conditioned top-k
    select (sample_router): topk | thresh | soft
    """

    def __init__(
        self,
        input_mode=None,
        transforms=None,
        log_losses=False,
        k=20,
        tau=10.0,
        cov_reg=1.0,
        ensemble_method="mean_prob",
        context="Class",
        # ---- dynamic routing ----
        routing="off",
        pool_size=64,
        budget=8,
        min_heads=4,
        search_eps=0.0,
        n_folds=4,
        select="topk",
        k_route=8,
        knn_neighbors=4,
        router_dim_head=64,
        router_lr=1e-3,
        router_epochs=300,
        router_l1=0.05,
        router_wd=1e-4,
        routing_seed=0,
    ):
        super().__init__(
            input_mode=input_mode,
            transforms=transforms,
            log_losses=log_losses,
            k=k,
            tau=tau,
            cov_reg=cov_reg,
            ensemble_method=ensemble_method,
            context=context,
        )
        assert routing in {"off", "sample_router", "class_cond", "knn"}
        self.routing = routing
        self.pool_size = pool_size
        self.budget = budget
        self.min_heads = min_heads
        self.search_eps = search_eps
        self.n_folds = n_folds
        self.select = select
        self.k_route = k_route
        self.knn_neighbors = knn_neighbors
        self.router_dim_head = router_dim_head
        self.router_lr = router_lr
        self.router_epochs = router_epochs
        self.router_l1 = router_l1
        self.router_wd = router_wd
        self.routing_seed = routing_seed

    # ------------------------------------------------------------------ #
    def fit_pred_features(self, model, support_set_features, support_label,
                          query_set_features, query_label):
        if self.routing == "off":
            return super().fit_pred_features(
                model, support_set_features, support_label,
                query_set_features, query_label,
            )

        A = support_set_features["attn_heads"].float()      # [Ns, L, H, Dh]
        Ns, L, H, Dh = A.shape
        M = L * H
        A = A.reshape(Ns, M, Dh)
        Aq = query_set_features["attn_heads"].float()       # [Nq, L, H, Dh]
        Nq = Aq.shape[0]
        Aq = Aq.reshape(Nq, M, Dh)

        # full-support GDA models (same as official HEC-V)
        hd = self.get_rank_scores(A, support_label)
        labels_unique = hd["labels_unique"].detach().cpu().numpy().astype(np.int64)
        yq = query_label.detach().cpu().numpy().astype(np.int64)

        # OOF signal + candidate pool (RouteHead: restrict the search space)
        z_oof, true_prob, mean_oof, y_idx = cross_fit_oof(self, A, support_label, self.n_folds)
        P = min(self.pool_size, M)
        pool = torch.argsort(mean_oof, descending=True)[:P]
        pool_np = pool.detach().cpu().numpy()
        z_pool_sup = z_oof[:, pool, :].cpu().numpy()        # [Ns, P, K]

        sup_cls = support_set_features["x_norm_clstoken"]
        qry_cls = query_set_features["x_norm_clstoken"]

        info = {"pool": pool_np, "routing": self.routing}
        device = Aq.device

        if self.routing == "sample_router":
            # pseudo labels: per-support-sample margin greedy+swap search on OOF probs
            sets_sup = greedy_margin_sets(
                z_pool_sup, y_idx.cpu().numpy(), self.budget, self.search_eps,
                min_heads=self.min_heads,
            )                                                   # [Ns, P] bool
            torch.manual_seed(self.routing_seed)
            router = train_router(
                sup_cls, sets_sup,
                dim_head=self.router_dim_head, lr=self.router_lr,
                epochs=self.router_epochs, l1=self.router_l1,
                weight_decay=self.router_wd, device=device,
            )
            with torch.no_grad():
                p_q = router(F.normalize(qry_cls, dim=-1).float())      # [Nq, P]
            p_np = p_q.detach().cpu().numpy()
            info["router_activation"] = p_np.mean(0)
            info["psup_sizes"] = sets_sup.sum(1)
            if self.select == "soft":
                idx = torch.as_tensor(pool_np, dtype=torch.long, device=device)
                z_pool = self.get_pred_labels(
                    Aq[:, idx, :], hd["head_means"][idx],
                    hd["head_cov_inv"][idx], hd["log_prior"],
                )[0]                                            # [Nq, P, K]
                w = p_q / p_q.sum(dim=1, keepdim=True).clamp(min=1e-8)
                P_q = torch.einsum("np,npk->nk", w, z_pool)
                info["set_sizes"] = np.full(Nq, P)
                info["usage_pool"] = None
            else:
                if self.select == "thresh":
                    hard = p_np > 0.5
                    empty = ~hard.any(axis=1)
                    if empty.any():  # fallback to top-k for empty sets
                        top = np.argsort(-p_np, axis=1)[:, : self.k_route]
                        hard[empty] = False
                        hard[np.where(empty)[0][:, None], top[empty]] = True
                    sets_pos = [np.flatnonzero(hard[n]) for n in range(Nq)]
                else:  # topk
                    top = np.argsort(-p_np, axis=1)[:, : self.k_route]
                    sets_pos = [top[n] for n in range(Nq)]
                P_q = group_ensemble_official(self, Aq, hd, sets_pos)
                info["set_sizes"] = np.array([len(s) for s in sets_pos])
                usage = np.zeros(P)
                for s in sets_pos:
                    usage[np.asarray(s)] += 1
                info["usage_pool"] = usage

        elif self.routing == "class_cond":
            sets_c = class_margin_sets(
                z_pool_sup, y_idx.cpu().numpy(), self.budget, self.search_eps,
                min_heads=self.min_heads,
            )                                                   # [K, P] bool
            class_sets_pos = [np.flatnonzero(sets_c[c]) for c in range(sets_c.shape[0])]
            P_q = class_ensemble_official(self, Aq, hd, class_sets_pos)
            info["class_sets_pool"] = class_sets_pos
            info["class_sets_global"] = [pool_np[s] for s in class_sets_pos]
            info["set_sizes"] = np.array([len(s) for s in class_sets_pos])
            usage = np.zeros(P)
            for s in class_sets_pos:
                usage[np.asarray(s)] += 1
            info["usage_pool"] = usage

        elif self.routing == "knn":
            sim = F.normalize(qry_cls, dim=-1).float() @ \
                F.normalize(sup_cls, dim=-1).float().t()        # [Nq, Ns]
            nn_idx = sim.topk(self.knn_neighbors, dim=1).indices  # [Nq, knn]
            marg = head_margin_scores(z_oof, y_idx)             # [Ns, M]
            scores_q = marg[nn_idx][:, :, pool].mean(dim=1)     # [Nq, P]
            top = scores_q.topk(self.k_route, dim=1).indices.cpu().numpy()
            sets_pos = [top[n] for n in range(Nq)]
            P_q = group_ensemble_official(self, Aq, hd, sets_pos)
            info["set_sizes"] = np.array([len(s) for s in sets_pos])
            usage = np.zeros(P)
            for s in sets_pos:
                usage[np.asarray(s)] += 1
            info["usage_pool"] = usage

        yhat = labels_unique[P_q.argmax(dim=-1).detach().cpu().numpy()]
        accuracy_final = float((yhat == yq).mean())
        if info.get("usage_pool") is not None:
            usage_global = np.zeros(M)
            usage_global[pool_np] = info["usage_pool"]
            info["usage_global"] = usage_global
        pred_prob = P_q.detach().cpu().numpy().astype(np.float32)

        return {
            "accuracy": accuracy_final,
            "pred_prob": pred_prob,
            "routing_info": info,
        }

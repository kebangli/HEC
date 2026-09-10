"""HEC-V with a Multi-Prototype query stage (minimal validation, 7th round).

Head selection is FULLY unchanged (identical to ``ClassifierHEC_V``): GDA rank
scores, Top-K, tau, context=Class.  Only the query classification of the
SELECTED heads is replaced:

  baseline (unchanged): per-class GDA uses the single class-mean centroid with
                        the shared per-head precision matrix ``head_cov_inv``.
  multi-prototype:      each class keeps its 4 support head features as 4
                        prototypes; the query uses the SAME shared per-head
                        precision matrix; class logit = max over the 4
                        prototypes of -0.5 * Mahalanobis(q, prototype_i);
                        softmax with the same tau; top-k heads equal-weight
                        averaged (mean of per-head probs).

Both the baseline and the multi-prototype predictions are computed on the same
episodes / features / selected heads and are both logged (per-head probs,
per-head argmax preds, class probs, mapped preds) for a paired comparison.

No training, no hyper-parameter change, no extra datasets.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .classifier import ClassifierHEC_V, get_top_heads_V


class ClassifierHEC_V_multiprototype(ClassifierHEC_V):
    """Official HEC-V (baseline) + Multi-Prototype query classification (paired)."""

    def get_pred_labels_mp(
            self,
            query_features,   # [Nq, topk, C] raw query head features
            prototypes,       # [topk, K, S, C] normalized support prototypes (S = shots)
            head_cov_inv,     # [topk, C, C] shared per-head precision (original GDA)
    ):
        """Per-head, per-class max-over-prototypes Mahalanobis similarity logits."""
        q = F.normalize(query_features, p=2, dim=-1)                        # [Nq, topk, C]
        # Mahalanobis: (q - p)^T cov_inv (q - p) = qcvq - 2*qcvP + PcvP
        qcvq = torch.einsum('ntd,tdf,ntf->nt', q, head_cov_inv, q)          # [Nq, topk]
        qcvP = torch.einsum('ntd,tdf,tkif->ntki', q, head_cov_inv, prototypes)  # [Nq, topk, K, S]
        PcvP = torch.einsum('tkid,tdf,tkif->tki', prototypes, head_cov_inv, prototypes)  # [topk, K, S]
        mahal = qcvq[:, :, None, None] - 2.0 * qcvP + PcvP[None, :, :, :]   # [Nq, topk, K, S]
        sim = -0.5 * mahal                                                  # similarity
        logits = sim.max(dim=-1).values                                     # [Nq, topk, K]
        z = torch.softmax(logits / self.tau, dim=-1)                        # [Nq, topk, K]
        return z, logits

    def _build_prototypes(
            self,
            A,                   # [Ns, M, C] raw support attn-head features
            support_label,       # [Ns]
            labels_unique_tensor,# [K] sorted class ids
            selected_idx,        # [topk]
    ):
        """Keep the per-shot normalized support features of the SELECTED heads."""
        An = F.normalize(A, p=2, dim=-1)                 # [Ns, M, C]
        An_sel = An[:, selected_idx, :]                  # [Ns, topk, C]
        topk = selected_idx.shape[0]
        protos = torch.stack([
            torch.stack([An_sel[(support_label == lbl), m, :] for lbl in labels_unique_tensor])
            for m in range(topk)
        ])                                               # [topk, K, S, C]
        return protos

    def fit_pred_features(
            self,
            model,
            support_set_features,
            support_label,
            query_set_features,
            query_label,
    ):
        # ---------------- TRAINING PHASE (verbatim official HEC-V) ----------------
        A = support_set_features["attn_heads"].float()   # (Ns, L, H, Dh)
        Ns, L, H, Dh = A.shape
        M = L * H
        A = A.reshape(Ns, M, Dh)

        Aq = query_set_features["attn_heads"].float()    # (Nq, L, H, Dh)
        Nq, _, _, _ = Aq.shape
        Aq = Aq.reshape(Nq, L * H, Dh)

        support_heat_models_dict = self.get_rank_scores(A, support_label)

        labels_unique_tensor = support_heat_models_dict["labels_unique"]
        labels_unique = labels_unique_tensor.detach().cpu().numpy().astype(np.int64)
        head_scores = support_heat_models_dict["head_scores"]  # [L*H]
        order = torch.argsort(head_scores, descending=True)

        if self.context == "Class":
            selected_idx = order[: min(self.k, M)]
        elif self.context in {"Task", "Domain"}:
            selected_idx = torch.as_tensor(
                get_top_heads_V(dataset_name=self.current_dataset_name, mode=self.context),
                dtype=torch.long,
                device=order.device,
            )[: min(self.k, M)]

        # ---------------- INFERENCE PHASE - baseline (verbatim official HEC-V) ----------------
        query_predictions_base, query_logits_base = self.get_pred_labels(
            Aq[:, selected_idx, :],
            support_heat_models_dict["head_means"][selected_idx, :, :],
            support_heat_models_dict["head_cov_inv"][selected_idx, :, :],
            support_heat_models_dict["log_prior"],
        )  # [Nq, topk, K]

        yhat = self.ensemble(
            head_scores=head_scores[selected_idx],
            query_predictions=query_predictions_base,
            query_logits=query_logits_base,
            support_predictions=support_heat_models_dict["head_predictions"][:, selected_idx, :],
            support_logits=support_heat_models_dict["head_logits"][:, selected_idx, :],
            support_labels=support_label,
        )
        yq = query_label.detach().cpu().numpy().astype(np.int64)
        pred_prob_base = query_predictions_base.mean(dim=1).detach().cpu().numpy().astype(np.float32)  # [Nq, K]
        yhat = labels_unique[yhat]
        acc_base = float((yhat == yq).mean())

        # ---------------- MULTI-PROTOTYPE query stage (only classification changes) ----------------
        prototypes = self._build_prototypes(
            A, support_label, labels_unique_tensor, selected_idx,
        )  # [topk, K, S, C]
        query_predictions_mp, query_logits_mp = self.get_pred_labels_mp(
            Aq[:, selected_idx, :],
            prototypes,
            support_heat_models_dict["head_cov_inv"][selected_idx, :, :],
        )  # [Nq, topk, K]
        # top-k heads equal-weight average (unchanged rule)
        pred_prob_mp = query_predictions_mp.mean(dim=1).detach().cpu().numpy().astype(np.float32)  # [Nq, K]
        yhat_mp = labels_unique[pred_prob_mp.argmax(axis=-1)]
        acc_mp = float((yhat_mp == yq).mean())

        # per-head predictions for both variants (head-accuracy comparison)
        head_preds_base = query_predictions_base.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)  # [Nq, topk]
        head_preds_mp = query_predictions_mp.argmax(dim=-1).detach().cpu().numpy().astype(np.int64)       # [Nq, topk]

        return {
            "accuracy": acc_base,                 # baseline = official HEC-V (self-check vs round 6)
            "pred_prob": pred_prob_base,
            # --- logging for the paired analysis ---
            "pred_prob_base": pred_prob_base,     # [Nq, K]
            "pred_prob_mp": pred_prob_mp,         # [Nq, K]
            "head_probs": query_predictions_base.detach().cpu().numpy().astype(np.float32),   # [Nq, topk, K]
            "head_probs_mp": query_predictions_mp.detach().cpu().numpy().astype(np.float32),  # [Nq, topk, K]
            "head_preds_base": head_preds_base,   # [Nq, topk]
            "head_preds_mp": head_preds_mp,       # [Nq, topk]
            "query_label": yq,                    # [Nq]
            "labels_unique": labels_unique,       # [K]
            "query_label_pred": yhat,             # baseline pred mapped to class ids
            "query_label_pred_mp": yhat_mp,       # multi-prototype pred mapped to class ids
            "acc_mp": acc_mp,
        }

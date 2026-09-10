"""HEC-V with leave-one-out support-stability logging (5th-round diagnostic).

The official HEC-V behaviour is left UNCHANGED: head selection, GDA, top-k,
tau, equal-weight "mean_prob" ensemble and accuracy are computed exactly as in
``ClassifierHEC_V`` (``fit_pred_features`` is replicated verbatim).  This
subclass additionally computes, for every query x selected head, a support
stability score:

    for each of the 4 leave-one-out perturbations of the 4-shot support
    (drop the 1st / 2nd / 3rd / 4th support image of every class), recompute
    the GDA (class means + covariance) for the SAME selected heads and
    re-predict the query class per head.  stability = fraction of the 4
    perturbations whose per-head prediction equals the original 4-shot
    per-head prediction.

No head re-selection, no training, no hyper-parameter change.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .classifier import ClassifierHEC_V


class ClassifierHEC_V_stab(ClassifierHEC_V):
    """Official HEC-V + per-query per-head support stability."""

    def _compute_stability(
            self,
            A,                   # [Ns, M, C] support attn-head features
            Aq,                  # [Nq, M, C] query attn-head features
            support_label,       # [Ns]
            selected_idx,        # [topk]
            labels_unique_tensor,  # [K]
            orig_z,              # [Nq, topk, K] original per-head probs
    ):
        Ns, M, Cdim = A.shape
        Nq = Aq.shape[0]
        topk = selected_idx.shape[0]
        classes = labels_unique_tensor.tolist()
        device = A.device

        Aq_sel = Aq[:, selected_idx, :]            # [Nq, topk, C]
        orig_pred = orig_z.argmax(dim=-1)          # [Nq, topk]

        # support indices per class, in batch order
        idx_by_class = {
            c: (support_label == c).nonzero(as_tuple=True)[0].tolist()
            for c in classes
        }

        eye = torch.eye(Cdim, device=device, dtype=A.dtype)[None, :, :]
        stab = torch.zeros(Nq, topk, device=device, dtype=torch.float32)
        for p in range(4):
            keep = []
            for c in classes:
                ids = idx_by_class[c]
                keep += [ids[j] for j in range(len(ids)) if j != p]   # drop (p+1)-th shot
            keep = torch.tensor(keep, device=device, dtype=torch.long)
            A_p = A[keep]                            # [Ns-K, M, C]
            lab_p = support_label[keep]
            Np = A_p.shape[0]

            # GDA for the SELECTED heads only (mirrors get_rank_scores)
            An = F.normalize(A_p, p=2, dim=-1)
            An_sel = An[:, selected_idx, :]                        # [Np, topk, C]
            y_idx = torch.searchsorted(labels_unique_tensor, lab_p)  # [Np]
            class_mean = torch.stack(
                [An_sel[lab_p == c].mean(0) for c in classes]
            )                                                      # [K, topk, C]
            centred = (An_sel - class_mean[y_idx]).transpose(0, 1) # [topk, Np, C]
            cov = (centred.transpose(1, 2) @ centred) / (Np - 1)   # [topk, C, C]
            tr = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
            cov_inv = Cdim * torch.linalg.pinv(
                (Np - 1) * cov + self.cov_reg * tr[:, None, None] * eye
            )                                                      # [topk, C, C]
            counts = torch.stack([(lab_p == c).sum() for c in classes]).to(A.dtype)
            log_prior = (counts / counts.sum()).log()              # [K]

            z_p, _ = self.get_pred_labels(
                Aq_sel,
                class_mean.permute(1, 0, 2),                       # [topk, K, C]
                cov_inv,
                log_prior,
            )                                                      # [Nq, topk, K]
            pred_p = z_p.argmax(dim=-1)
            stab += (pred_p == orig_pred).float()
        stab /= 4.0
        return stab.detach().cpu().numpy()                         # [Nq, topk]

    def fit_pred_features(
            self,
            model,
            support_set_features,
            support_label,
            query_set_features,
            query_label,
    ):
        # ---------------- VERBATIM copy of the official ClassifierHEC_V ----------------
        A = support_set_features["attn_heads"].float()   # (Ns, L, H, Dh)
        Ns, L, H, Dh = A.shape
        M = L * H
        A = A.reshape(Ns, M, Dh)

        Aq = query_set_features["attn_heads"].float()    # (Nq, L, H, Dh)
        Nq, _, _, _ = Aq.shape
        Aq = Aq.reshape(Nq, L * H, Dh)

        # TRAINING PHASE
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

        # INFERENCE PHASE
        query_predictions, query_logits = self.get_pred_labels(
            Aq[:, selected_idx, :],
            support_heat_models_dict["head_means"][selected_idx, :, :],
            support_heat_models_dict["head_cov_inv"][selected_idx, :, :],
            support_heat_models_dict["log_prior"],
        )  # [Nq, topk, K]

        yhat = self.ensemble(
            head_scores=head_scores[selected_idx],
            query_predictions=query_predictions,
            query_logits=query_logits,
            support_predictions=support_heat_models_dict["head_predictions"][:, selected_idx, :],
            support_logits=support_heat_models_dict["head_logits"][:, selected_idx, :],
            support_labels=support_label,
        )
        yq = query_label.detach().cpu().numpy().astype(np.int64)
        pred_prob = query_predictions.mean(dim=1).detach().cpu().numpy().astype(np.float32)  # [Nq, K]

        yhat = labels_unique[yhat]
        accuracy_final = float((yhat == yq).mean())
        # ---------------- end of verbatim copy ----------------

        # ADDITIONAL: leave-one-out support stability
        stability = self._compute_stability(
            A, Aq, support_label, selected_idx, labels_unique_tensor, query_predictions,
        )  # [Nq, topk]

        return {
            "accuracy": accuracy_final,
            "pred_prob": pred_prob,
            "head_probs": query_predictions.detach().cpu().numpy().astype(np.float32),  # [Nq, topk, K]
            "stability": stability,                                                     # [Nq, topk]
            "query_label": yq,                                                          # [Nq]
            "labels_unique": labels_unique,                                             # [K]
            "query_label_pred": labels_unique[np.argmax(pred_prob, axis=-1)],           # [Nq]
        }

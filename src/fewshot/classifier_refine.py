# Module B: Feature Refinement for HEC-V
#
# Minimal-invasive subclass of ClassifierHEC_V adding two orthogonal,
# training-free refinements applied AFTER head selection (the ranking and
# the selected heads stay strictly identical to the baseline):
#
#   1. APE-style channel refinement (fisher_ratio < 1.0):
#      For each of the k selected heads, compute a per-dimension Fisher
#      score (between-class variance / within-class variance) on the
#      SUPPORT set only, keep the top `fisher_ratio` fraction of the
#      head-dimension, and run the per-head GDA on the masked features.
#      Reference: "Not All Features Matter: Enhancing Few-shot CLIP with
#      Adaptive Prior Refinement" (APE). Their criterion relies on CLIP
#      text weights; here it is replaced by a support-only Fisher score,
#      the natural vision-only counterpart.
#
#   2. PT-MAP-style power transformation (power_p != 1.0):
#      Signed power transform x * |x|^(p-1) applied to the RAW head
#      features before L2 normalization, making the per-dimension
#      distribution closer to Gaussian before the GDA.
#      Reference: "Leveraging the Feature Distribution in Transfer-based
#      Few-Shot Learning" (PT-MAP), beta = 0.5.
#
# With fisher_ratio = 1.0 and power_p = 1.0 the classifier executes the
# original code path, bit-identical to ClassifierHEC_V.

import numpy as np
import torch
import torch.nn.functional as F

from .classifier import ClassifierHEC_V

import logging
logger = logging.getLogger(__name__)


class ClassifierHEC_V_refine(ClassifierHEC_V):
    frozen_classifier = True

    def __init__(
            self,
            *args,
            fisher_ratio: float = 1.0,   # 1.0 = off; 0.75/0.5/0.25 = keep top Fisher dims per head
            power_p: float = 1.0,        # 1.0 = off; 0.5 = PT-MAP paper setting
            mask_renorm: bool = True,    # renormalize after dimension masking (APE-style)
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        assert 0.0 < fisher_ratio <= 1.0, f"fisher_ratio must be in (0, 1], got {fisher_ratio}"
        assert 0.0 < power_p <= 2.0, f"power_p must be in (0, 2], got {power_p}"
        self.fisher_ratio = fisher_ratio
        self.power_p = power_p
        self.mask_renorm = mask_renorm

    # ------------------------------------------------------------------ #
    # Refinement helpers                                                  #
    # ------------------------------------------------------------------ #

    def _prep_head_features(self, x):
        """Raw head features [B, M, C] -> GDA representation [B, M, C].

        PT-MAP order: transform raw features first, then L2-normalize.
        """
        if self.power_p != 1.0:
            x = x * x.abs().clamp_min(1e-12).pow(self.power_p - 1.0)
        x = F.normalize(x, p=2, dim=-1)
        return x

    def _fisher_mask(self, xs, ys):
        """Per-head dimension importance from support features.

        xs: [N, M, C] prepared (normalized) support features.
        ys: [N] support labels.
        Returns boolean mask [M, C] (True = keep).

        Fisher score per dimension d of head m:
            F = between-class variance / within-class variance
        """
        N, M, C = xs.shape
        labels_unique = torch.unique(ys)
        K = labels_unique.numel()
        y_idx = torch.searchsorted(labels_unique, ys)  # [N]

        mu_k = torch.stack([xs[ys == lbl].mean(0) for lbl in labels_unique])  # [K, M, C]
        mu = xs.mean(0)                                                       # [M, C]

        counts = torch.stack([(ys == lbl).sum() for lbl in labels_unique]).to(xs.dtype)  # [K]
        w = (counts / counts.sum())[:, None, None]                             # [K, 1, 1]
        between = (w * (mu_k - mu[None]) ** 2).sum(0)                          # [M, C]

        centered = xs - mu_k[y_idx]                                            # [N, M, C]
        within = (centered ** 2).sum(0) / max(N - K, 1)                        # [M, C]

        fisher = between / (within + 1e-12)                                   # [M, C]

        n_keep = max(1, int(round(self.fisher_ratio * C)))
        mask = torch.zeros(M, C, dtype=torch.bool, device=xs.device)
        if n_keep >= C:
            mask[:] = True
        else:
            idx = fisher.topk(n_keep, dim=-1).indices                          # [M, n_keep]
            mask.scatter_(1, idx, True)
        return mask

    def _apply_mask(self, x, mask):
        """x: [B, M, C], mask: [M, C]. Zero out dropped dims; optionally
        renormalize the kept subspace to the unit sphere (APE-style)."""
        x = x * mask[None].to(x.dtype)
        if self.mask_renorm:
            x = F.normalize(x, p=2, dim=-1)
        return x

    def _fit_gda(self, xs, ys):
        """Per-head GDA fit on ALREADY prepared (normalized) features.

        Identical math to ClassifierHEC_V.get_rank_scores but skips its
        internal F.normalize. Returns (head_means, head_cov_inv, log_prior,
        labels_unique)."""
        N, M, C = xs.shape
        labels_unique = torch.unique(ys)
        y_idx = torch.searchsorted(labels_unique, ys)

        class_mean = torch.stack([xs[ys == lbl].mean(0) for lbl in labels_unique])  # [K, M, C]

        centered = (xs - class_mean[y_idx]).transpose(0, 1)                     # [M, N, C]
        cov = (centered.transpose(1, 2) @ centered) / (N - 1)                   # [M, C, C]
        tr = cov.diagonal(dim1=-2, dim2=-1).sum(-1)
        cov_inv = C * torch.linalg.pinv(                                        # [M, C, C]
            (N - 1) * cov
            + self.cov_reg * tr[:, None, None]
            * torch.eye(C, device=xs.device, dtype=xs.dtype)[None, :, :]
        )

        counts = torch.stack([(ys == lbl).sum() for lbl in labels_unique]).to(xs.dtype)
        log_prior = (counts / counts.sum()).log()                               # [K]

        return class_mean.permute(1, 0, 2), cov_inv, log_prior, labels_unique

    def _predict_gda(self, xq, head_means, head_cov_inv, log_prior):
        """xq: [Nq, M, C] prepared query features -> probs [Nq, M, K].

        Identical math to ClassifierHEC_V.get_pred_labels."""
        mahal = (
            torch.einsum('nmd,mdf,nmf->nm', xq, head_cov_inv, xq)[:, :, None]
            - 2.0 * torch.einsum('nmd,mdf,mkf->nmk', xq, head_cov_inv, head_means)
            + torch.einsum('mkd,mdf,mkf->mk', head_means, head_cov_inv, head_means)[None, :, :]
        )
        logits = -0.5 * mahal + log_prior[None, None, :]
        return torch.softmax(logits / self.tau, dim=-1)

    # ------------------------------------------------------------------ #
    # Main entry (mirrors ClassifierHEC_V.fit_pred_features)               #
    # ------------------------------------------------------------------ #

    def fit_pred_features(
            self,
            model,
            support_set_features: torch.Tensor,
            support_label: torch.Tensor,
            query_set_features: torch.Tensor,
            query_label: torch.Tensor,
        ):

        A = support_set_features["attn_heads"].float()  # (Ns, L, H, Dh)
        Ns, L, H, Dh = A.shape
        M = L * H
        A = A.reshape(Ns, M, Dh)

        Aq = query_set_features["attn_heads"].float()   # (Nq, L, H, Dh)
        Nq = Aq.shape[0]
        Aq = Aq.reshape(Nq, M, Dh)

        # ---- TRAINING PHASE: head ranking (identical to baseline) ---- #
        support_heat_models_dict = self.get_rank_scores(A, support_label)

        labels_unique_tensor = support_heat_models_dict["labels_unique"]
        labels_unique = labels_unique_tensor.detach().cpu().numpy().astype(np.int64)
        head_scores = support_heat_models_dict["head_scores"]  # [L*H]
        order = torch.argsort(head_scores, descending=True)

        if self.context == "Class":
            selected_idx = order[: min(self.k, M)]
        elif self.context in {"Task", "Domain"}:
            from .utils import get_top_heads_V
            selected_idx = torch.as_tensor(
                get_top_heads_V(dataset_name=self.current_dataset_name, mode=self.context),
                dtype=torch.long,
                device=order.device
            )[: min(self.k, M)]
        else:
            raise ValueError(self.context)

        refinement_off = (self.fisher_ratio >= 1.0) and (self.power_p == 1.0)

        if refinement_off:
            # ---- BASELINE PATH: bit-identical to ClassifierHEC_V ---- #
            query_predictions, query_logits = self.get_pred_labels(
                Aq[:, selected_idx, :],
                support_heat_models_dict["head_means"][selected_idx, :, :],
                support_heat_models_dict["head_cov_inv"][selected_idx, :, :],
                support_heat_models_dict["log_prior"],
            )  # [Nq, topk, K]
        else:
            # ---- REFINEMENT PATH ---- #
            xs = self._prep_head_features(A[:, selected_idx, :])   # [Ns, k, C]
            xq = self._prep_head_features(Aq[:, selected_idx, :])   # [Nq, k, C]

            if self.fisher_ratio < 1.0:
                mask = self._fisher_mask(xs, support_label)          # [k, C]
                xs = self._apply_mask(xs, mask)
                xq = self._apply_mask(xq, mask)

            head_means, head_cov_inv, log_prior, labels_unique_tensor = self._fit_gda(
                xs, support_label
            )
            query_predictions = self._predict_gda(
                xq, head_means, head_cov_inv, log_prior
            )  # [Nq, k, K]

        yhat = self.ensemble(
            head_scores=head_scores[selected_idx],
            query_predictions=query_predictions,
            query_logits=None,
            support_predictions=support_heat_models_dict["head_predictions"][:, selected_idx, :],
            support_logits=support_heat_models_dict["head_logits"][:, selected_idx, :],
            support_labels=support_label,
        )
        yq = query_label.detach().cpu().numpy().astype(np.int64)
        pred_prob = query_predictions.mean(dim=1).detach().cpu().numpy().astype(np.float32)  # [Nq, K]

        yhat = labels_unique[yhat]
        accuracy_final = float((yhat == yq).mean())

        return {
            "accuracy": accuracy_final,
            "pred_prob": pred_prob,
        }

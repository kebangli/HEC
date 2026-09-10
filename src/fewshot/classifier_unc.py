"""HEC-VT with per-query logging of branch probabilities (uncertainty experiment).

The fixed-weight fusion used by the official ``ClassifierHEC_VT``
(``final_prob = P_T + alpha * P_V``) and the whole tuning / head-selection /
data-sampling protocol are left UNCHANGED.  This subclass only *records* the
per-query HEC-V / HEC-T probability vectors and the ground-truth labels so the
uncertainty-aware (reliability-weighted) fusion and its analyses can be
computed offline.

The uncertainty-aware fusion evaluated offline:
    U_i  = -sum_c p_ic log p_ic / log C      (normalized entropy, branch i)
    r_i  = 1 - U_i                           (reliability)
    P    = (r_V * P_V + r_T * P_T) / (r_V + r_T + eps)
"""

import numpy as np
import torch

from .classifier import ClassifierHEC_VT


class ClassifierHEC_VT_unc(ClassifierHEC_VT):
    """Official HEC-VT + per-query branch-probability logging."""

    def fit_pred(
            self,
            model,
            support_set,
            support_label,
            query_set,
            query_label,
    ):
        # Inference (identical to the official ClassifierHEC_VT.fit_pred)
        support_set_features, query_set_features = self.get_frozen_features(model, support_set, query_set)

        self.hec_v.current_label_to_str = self.current_label_to_str
        self.hec_v.current_dataset_name = self.current_dataset_name
        self.hec_t.current_label_to_str = self.current_label_to_str
        self.hec_t.current_dataset_name = self.current_dataset_name

        # Classification
        hec_v_results = self.hec_v.fit_pred_features(
            model,
            support_set_features,
            support_label,
            query_set_features,
            query_label,
        )
        hec_t_results = self.hec_t.fit_pred_features(
            model,
            support_set_features,
            support_label,
            query_set_features,
            query_label,
        )

        P_V = hec_v_results["pred_prob"]  # [Nq, K] float32 numpy
        P_T = hec_t_results["pred_prob"]  # [Nq, K] float32 numpy

        # Original fixed-weight fusion (unchanged)
        final_prob = P_T + self.alpha * P_V
        final_pred = final_prob.argmax(axis=-1)

        labels_unique = torch.unique(support_label).detach().cpu().numpy().astype(np.int64)
        yhat = labels_unique[final_pred]
        yq = query_label.detach().cpu().numpy().astype(np.int64)
        accuracy_final = float((yhat == yq).mean())

        return {
            "accuracy": accuracy_final,
            "pred_prob": final_prob,
            "pred_prob_V": P_V,
            "pred_prob_T": P_T,
            "query_label": yq,             # [Nq] ground-truth class ids
            "query_label_pred": yhat,      # [Nq] fixed-fusion predictions
        }

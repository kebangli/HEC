"""HEC-V with per-query per-head probability logging (head-level uncertainty).

The official HEC-V behaviour (head selection, GDA, tau, top-k, equal-weight
"mean_prob" ensemble, accuracy) is left UNCHANGED.  This subclass only
*additionally* stashes the per-query per-selected-head class probabilities
``P_m(c|x)`` (the GDA + softmax output for each selected head), the
ground-truth labels and the sorted label mapping, so the head-level
entropy/reliability analyses can be computed offline.
"""

import numpy as np
import torch

from .classifier import ClassifierHEC_V


class ClassifierHEC_V_headunc(ClassifierHEC_V):
    """Official HEC-V + per-query per-head probability logging."""

    def ensemble(
            self,
            head_scores,         # [topk]
            query_predictions,   # [Nq, topk, K]  per-head GDA+softmax probs
            query_logits,        # [Nq, topk, K]
            support_predictions, # [Ns, topk, K]
            support_logits,      # [Ns, topk, K]
            support_labels,      # [Ns]
    ):
        # Capture the per-head probabilities before the (unchanged) ensemble.
        self._head_probs = query_predictions.detach().float().clone()
        return super().ensemble(
            head_scores, query_predictions, query_logits,
            support_predictions, support_logits, support_labels,
        )

    def fit_pred(
            self,
            model,
            support_set,
            support_label,
            query_set,
            query_label,
    ):
        self._head_probs = None
        # Inference (identical to the official ClassifierHEC_V.fit_pred)
        support_set_features, query_set_features = self.get_frozen_features(model, support_set, query_set)
        with torch.inference_mode():
            result = self.fit_pred_features(
                model,
                support_set_features,
                support_label,
                query_set_features,
                query_label,
            )

        result["head_probs"] = (
            self._head_probs.cpu().numpy().astype(np.float32)
            if self._head_probs is not None else None
        )  # [Nq, topk, K]
        result["query_label"] = query_label.detach().cpu().numpy().astype(np.int64)      # [Nq]
        result["labels_unique"] = torch.unique(support_label).detach().cpu().numpy().astype(np.int64)  # [K]
        # Equal-weight (baseline) prediction mapped back to original labels
        result["query_label_pred"] = result["labels_unique"][np.argmax(result["pred_prob"], axis=-1)]
        return result

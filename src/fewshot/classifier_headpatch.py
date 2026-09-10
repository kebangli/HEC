"""HEC-V Head-Patch diagnostic (round 8, OxfordPets).

Goal: for the Top-K HEC-V selected heads, extract the last-token -> image-token
attention weights and map them back to image patches, then check (a) head
diversity of attention maps, (b) visual heatmaps, (c) a discriminative sanity
check by removing high-attention patches' value contribution.

HEC-V is kept FULLY unchanged: ``get_rank_scores`` (GDA head selection), Top-K,
tau, context=Class, support/query, prompt and the equal-weight mean_prob
ensemble are byte-identical to the official ``ClassifierHEC_V``
(``fit_pred_features`` is replicated verbatim and returns the official
baseline).  The diagnostic is purely additive:

  * A second forward pass on a small subset of queries (``MINI_N``) with
    ``output_attentions=True`` (HF auto-eager fallback) captures per-layer
    attention probabilities [B, H, T, T] and (via v_proj / o_proj hooks) the
    value vectors.  The sdpa path that produced the official HEC-V features is
    untouched.
  * Image tokens are the ``image_token_id`` placeholder positions (64 for a
    224x224 OxfordPets image), mapped to the 8x8 merged patch grid (2x2
    spatial merge of the 16x16 patch grid from ``image_grid_thw``).
  * Per selected head (layer=idx//H, head=idx%H) and per subset query:
      - attention map over the 64 image tokens (flattened 8x8),
      - approximate intervention: ablate the value contribution of the top
        attention patches vs the same number of random patches,
        ``f' = normalize(f - sum_{p in mask} attn_w[p] * V[p])``, and record
        the drop of the correct-class GDA logit (same shared precision matrix
        and class means as the original HEC-V),
      - per-head prediction correctness and a consistency check between the
        sdpa feature f and the full eager reconstruction sum_p attn_w[p]*V[p].

No training, no hyper-parameter change, no new classifier.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .classifier import ClassifierHEC_V, get_top_heads_V


class ClassifierHEC_V_headpatch(ClassifierHEC_V):
    """Official HEC-V + per-head last-token->patch attention diagnostic."""

    MINI_N = 30          # query subset per episode for attention capture
    TOP_FRAC = 0.10      # fraction of patches ablated (top-attention vs random)
    N_RANDOM_TRIALS = 3  # random patch masks averaged per (query, head)

    # ---------------------------------------------------------------- #
    #  fit_pred: keep the raw query images for the attention pass
    # ---------------------------------------------------------------- #
    def fit_pred(self, model, support_set, support_label, query_set, query_label):
        self._query_set_raw = query_set.detach()
        support_set_features, query_set_features = self.get_frozen_features(model, support_set, query_set)
        with torch.inference_mode():
            return self.fit_pred_features(
                model, support_set_features, support_label, query_set_features, query_label,
            )

    # ---------------------------------------------------------------- #
    #  attention capture helpers
    # ---------------------------------------------------------------- #
    def _build_attention_inputs(self, backbone, imgs):
        """Mirror QwenV2Wrapper.forward input construction for a sub-batch."""
        processor = backbone.processor
        text_prompt = backbone.current_text_prompt
        B = imgs.shape[0]
        chat = processor.apply_chat_template(
            conversation=[{
                "role": "user",
                "content": [
                    {"type": "image", "image": "Any"},
                    {"type": "text", "text": text_prompt},
                ],
            }],
            tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(text=[chat] * B, images=(255 * imgs).to(torch.uint8),
                           padding=True, return_tensors="pt")
        inputs = {k: (v.to(imgs.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        return inputs

    CAPTURE_CHUNK = 8   # images per eager attention forward (bounded GPU memory)

    def _capture_attention(self, model, imgs, sel_layers):
        """Eager forward with output_attentions + value hooks (chunked).

        Returns (attns, v_acts, img_positions, grid) where
          attns[l]   [B, H, T, T] attention probs (all layers),
          v_acts[l]  [B, T, H, Dh] value states (only needed layers),
          img_positions list[int] positions of the image tokens,
          grid = (mh, mw) merged patch grid.
        """
        backbone = model.backbone
        model0 = backbone.model
        sel_layers = sorted(set(int(l) for l in sel_layers))

        img_positions = None
        thw = None
        attns_chunks, v_chunks = [], {i: [] for i in sel_layers}

        for s in range(0, imgs.shape[0], self.CAPTURE_CHUNK):
            sub = imgs[s:s + self.CAPTURE_CHUNK]
            inputs = self._build_attention_inputs(backbone, sub)
            ids = inputs["input_ids"]
            if img_positions is None:
                img_id = model0.config.image_token_id
                img_positions = [p for p in range(ids.shape[1]) if int(ids[0, p].item()) == img_id]
                thw = inputs["image_grid_thw"][0]            # [1, h, w]

            v_acts, hooks = {}, []
            for i in sel_layers:
                attn = model0.model.layers[i].self_attn
                nkv = attn.num_key_value_heads
                nkg = attn.num_heads // nkv
                Dh = attn.head_dim

                def _vh(module, inp, out, i=i, nkv=nkv, nkg=nkg, Dh=Dh):
                    v = out.view(out.shape[0], out.shape[1], nkv, Dh).repeat_interleave(nkg, dim=2)
                    v_acts[i] = v.detach()

                hooks.append(attn.v_proj.register_forward_hook(_vh))

            with torch.no_grad():
                res = model0(**inputs, output_attentions=True)
            for h in hooks:
                h.remove()

            attns_chunks.append([a.detach() if a is not None else None for a in res.attentions])
            for i in sel_layers:
                v_chunks[i].append(v_acts[i])
            del inputs, v_acts, res
            torch.cuda.empty_cache()

        attns = [torch.cat([c[l] for c in attns_chunks], 0) for l in range(len(attns_chunks[0]))]
        v_acts = {i: torch.cat(v_chunks[i], 0) for i in sel_layers}
        grid_h, grid_w = int(thw[1].item()), int(thw[2].item())
        mh, mw = grid_h // 2, grid_w // 2            # spatial_merge_size = 2
        return attns, v_acts, img_positions, (mh, mw)

    @staticmethod
    def _head_logit(q, mu, cov_inv, lp, c):
        """Correct-class GDA logit: -0.5*(q-mu_c)^T cov_inv (q-mu_c) + lp_c.
        q: [C] normalized feature; mu: [K,C]; cov_inv: [C,C]; lp: [K]."""
        d = q - mu[c]
        return -0.5 * (d @ cov_inv @ d) + lp[c]

    # ---------------------------------------------------------------- #
    #  main
    # ---------------------------------------------------------------- #
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
                dtype=torch.long, device=order.device,
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
        pred_prob_base = query_predictions_base.mean(dim=1).detach().cpu().numpy().astype(np.float32)
        yhat = labels_unique[yhat]
        acc_base = float((yhat == yq).mean())

        # ---------------- HEAD-PATCH DIAGNOSTIC (additive) ----------------
        topk = selected_idx.shape[0]
        Q = min(self.MINI_N, Nq)
        sel_layers = selected_idx // H
        sel_heads = selected_idx % H

        attns, v_acts, img_pos, (mh, mw) = self._capture_attention(
            model, self._query_set_raw[:Q], sel_layers,
        )
        Nimg = len(img_pos)
        N_mask = max(1, int(self.TOP_FRAC * Nimg))

        # subset true-class index (0..K-1)
        sub_yidx = torch.searchsorted(labels_unique_tensor, query_label[:Q])  # [Q]

        attn_maps = torch.zeros(Q, topk, Nimg, dtype=torch.float32)
        consist = torch.zeros(Q, topk, dtype=torch.float32)
        logit_orig = torch.zeros(Q, topk, dtype=torch.float32)
        drop_top = torch.zeros(Q, topk, dtype=torch.float32)
        drop_rnd = torch.zeros(Q, topk, dtype=torch.float32)
        drop_eqmass = torch.zeros(Q, topk, dtype=torch.float32)
        head_preds = torch.zeros(Q, topk, dtype=torch.long)

        mean_mu = support_heat_models_dict["head_means"][selected_idx]       # [topk, K, C]
        cov_inv = support_heat_models_dict["head_cov_inv"][selected_idx]     # [topk, C, C]
        log_prior = support_heat_models_dict["log_prior"]                    # [K]

        for m in range(topk):
            ell, hh = int(sel_layers[m]), int(sel_heads[m])
            aw_all = attns[ell][:, hh, -1, :]               # [Q, T] attention probs over all tokens
            v_all = v_acts[ell][:, :, hh, :].float()        # [Q, T, Dh]
            f_sdpa = Aq[:Q, int(selected_idx[m]), :]        # [Q, Dh] official (sdpa) feature

            # attention to image tokens + value contribution
            aw_img = aw_all[:, img_pos]                     # [Q, Nimg]
            v_img = v_all[:, img_pos]                       # [Q, Nimg, Dh]
            attn_maps[:, m, :] = aw_img.float()

            # consistency: f_sdpa vs full eager reconstruction sum_p aw[p]*V[p]
            rec = (aw_all[:, :, None] * v_all).sum(dim=1)   # [Q, Dh]
            fs = F.normalize(f_sdpa, p=2, dim=-1)
            rc = F.normalize(rec, p=2, dim=-1)
            consist[:, m] = (fs * rc).sum(-1).float()       # cosine similarity

            for b in range(Q):
                c = int(sub_yidx[b])
                q_orig = F.normalize(f_sdpa[b], p=2, dim=-1)
                lo = self._head_logit(q_orig, mean_mu[m], cov_inv[m], log_prior, c)
                logit_orig[b, m] = lo
                head_preds[b, m] = torch.argmax(
                    torch.stack([self._head_logit(q_orig, mean_mu[m], cov_inv[m], log_prior, k)
                                 for k in range(mean_mu.shape[1])]))

                # top-attention patches
                order_p = torch.argsort(aw_img[b], descending=True)
                mask_top = order_p[:N_mask]
                f_top = f_sdpa[b] - (aw_img[b, mask_top][:, None] * v_img[b, mask_top]).sum(0)
                drop_top[b, m] = lo - self._head_logit(
                    F.normalize(f_top, p=2, dim=-1), mean_mu[m], cov_inv[m], log_prior, c)
                top_mass = float(aw_img[b, mask_top].sum())

                # random patches, equal COUNT (average over trials, seeded)
                rnd_drop = 0.0
                # random patches, equal MASS (same attention-weighted value mass removed)
                eq_drop = 0.0
                g = torch.Generator(device=Aq.device).manual_seed(int(1000 + b * topk + m))
                for _ in range(self.N_RANDOM_TRIALS):
                    perm = torch.randperm(Nimg, device=Aq.device, generator=g)
                    mask_rnd = perm[:N_mask]
                    f_rnd = f_sdpa[b] - (aw_img[b, mask_rnd][:, None] * v_img[b, mask_rnd]).sum(0)
                    rnd_drop += lo - self._head_logit(
                        F.normalize(f_rnd, p=2, dim=-1), mean_mu[m], cov_inv[m], log_prior, c)
                    # equal-mass: pick random patches until their attention weight >= top_mass
                    picked = []
                    cum = 0.0
                    for pp in perm.tolist():
                        if cum >= top_mass or len(picked) >= Nimg:
                            break
                        cum += float(aw_img[b, pp])
                        picked.append(pp)
                    if not picked:
                        picked = [int(perm[0])]
                    f_eq = f_sdpa[b] - (aw_img[b, picked][:, None] * v_img[b][picked]).sum(0)
                    eq_drop += lo - self._head_logit(
                        F.normalize(f_eq, p=2, dim=-1), mean_mu[m], cov_inv[m], log_prior, c)
                drop_rnd[b, m] = rnd_drop / self.N_RANDOM_TRIALS
                drop_eqmass[b, m] = eq_drop / self.N_RANDOM_TRIALS

        # ensemble correctness on the subset (official baseline pred)
        ens_pred = pred_prob_base[:Q].argmax(axis=-1)
        ens_correct = (ens_pred == sub_yidx.detach().cpu().numpy())

        head_correct = (head_preds.detach().cpu().numpy() == sub_yidx.detach().cpu().numpy()[:, None])

        return {
            "accuracy": acc_base,                 # official baseline (self-check vs rounds 6/7)
            "pred_prob": pred_prob_base,
            # --- standard headunc-style logging (full batch) ---
            "head_probs": query_predictions_base.detach().cpu().numpy().astype(np.float32),
            "query_label": yq,
            "labels_unique": labels_unique,
            "query_label_pred": yhat,
            # --- head-patch diagnostic (subset of queries) ---
            "query_imgs_sub": (255 * self._query_set_raw[:Q]).to(torch.uint8).cpu().numpy(),  # [Q,3,224,224]
            "query_label_sub": query_label[:Q].detach().cpu().numpy().astype(np.int64),
            "attn_maps": attn_maps.detach().cpu().numpy().astype(np.float32),        # [Q, topk, Nimg]
            "attn_consist": consist.detach().cpu().numpy().astype(np.float32),       # [Q, topk]
            "logit_orig": logit_orig.detach().cpu().numpy().astype(np.float32),      # [Q, topk]
            "drop_top": drop_top.detach().cpu().numpy().astype(np.float32),          # [Q, topk]
            "drop_rnd": drop_rnd.detach().cpu().numpy().astype(np.float32),          # [Q, topk]
            "drop_eqmass": drop_eqmass.detach().cpu().numpy().astype(np.float32),    # [Q, topk]
            "head_correct": head_correct,                                            # [Q, topk]
            "ens_correct": ens_correct,                                              # [Q]
            "selected_idx": selected_idx.detach().cpu().numpy().astype(np.int64),    # [topk] HEC-V head ids
            "n_mask": N_mask, "img_grid": (mh, mw),
        }

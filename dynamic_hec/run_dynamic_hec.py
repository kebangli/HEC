#!/usr/bin/env python3
"""Dynamic Head Routing HEC vs Static HEC — small-dataset validation.

RouteHead-inspired experiment: on identical per-episode features, compare
  [static]  static_hec20 / static_hec8 : official ClassifierHEC_V top-k (in-sample score)
            static_cf20               : cross-fit OOF-mean top-20 (unbiased static)
  [dynamic] knn8                       : training-free neighbour-conditioned top-8
            class_cond8               : per-class contrastive head sets (budget 8)
            router8                   : RouteHead-style learned router, top-8
            router_soft               : router soft probabilities over the pool
  [diag]    oracle8                    : per-query top-8 by true-class posterior
                                       (uses query labels — diagnostic ceiling only)

Outputs accuracies per (dataset, seed), head-usage distributions, router/class-set
stats, and a markdown report.

Usage: python run_dynamic_hec.py [gpu_id] [--datasets oxford_pets,dtd] [--seeds 123,148,173]
"""
import os
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TQDM_DISABLE'] = '1'
import argparse
import sys
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/data2/lbk/HEC')
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.fewshot.engine import FinetuneWrapper
from src.data_clip import build_dataset
from src.data_clip.utils import EpisodicNShotKWayDataset
from src.fewshot.classifier import ClassifierHEC_V
from src.fewshot.head_router import (
    cross_fit_oof, greedy_margin_sets, class_margin_sets, head_margin_scores,
    train_router, group_ensemble_official, class_ensemble_official,
    oracle_topk_sets, oracle_correct_topk_sets,
)

OUT_DIR = '/data2/lbk/HEC/dynamic_hec/results'
PATH_BASE = '/data2/lbk/HEC_repro/data/clip_datasets'
WEIGHTS = '/data2/lbk/HEC_repro/weights/qwenv2'
POOL_SIZE = 64
BUDGET = 8

ACC_KEYS = ['static_hec20', 'static_hec8', 'static_cf20', 'pool_uniform64', 'knn8',
            'class_cond8', 'router8', 'router_thresh', 'router_soft', 'hybrid_static_router',
            'router_soft_p20', 'oracle8', 'oracle_corr8', 'oracle_soft']


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('gpu', nargs='?', default='1')
    p.add_argument('--datasets', default='oxford_pets,dtd')
    p.add_argument('--seeds', default='123,148,173')
    p.add_argument('--ways', default='20')          # eurosat overridden to 10
    p.add_argument('--n_query', type=int, default=5)
    p.add_argument('--n_support', type=int, default=4)
    p.add_argument('--tag', default='')             # suffix for parallel runs
    return p.parse_args()


def accuracy(P_q, labels_unique, yq_np):
    yhat = labels_unique[P_q.argmax(dim=-1).cpu().numpy()]
    return float((yhat == yq_np).mean())


def usage_from_sets(sets_pos, P):
    u = np.zeros(P)
    for s in sets_pos:
        u[np.asarray(s)] += 1
    return u


def mean_pairwise_jaccard(sets_pos, sample=200):
    n = len(sets_pos)
    rng = np.random.RandomState(0)
    pairs = rng.randint(0, n, size=min(sample, n * 2) * 2).reshape(-1, 2)
    js = []
    for a, b in pairs:
        if a == b:
            continue
        sa, sb = set(sets_pos[a].tolist()), set(sets_pos[b].tolist())
        js.append(len(sa & sb) / max(1, len(sa | sb)))
    return float(np.mean(js)) if js else 1.0


def main():
    args = parse_args()
    DEVICE = f'cuda:{args.gpu}'
    torch.cuda.set_device(DEVICE)
    seeds = [int(s) for s in args.seeds.split(',')]
    datasets = args.datasets.split(',')
    tag = f'_{args.tag}' if args.tag else ''
    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = OmegaConf.create({
        'backbone': OmegaConf.create({'_target_': 'src.fewshot.backbone.QwenV2Wrapper',
            'model_name': 'QWENv2', 'weights': WEIGHTS, 'num_features_out': 1280,
            'output_llm_lasttok_heads': True}),
        'peft_c': {'batch_size_max_infer': 8, 'batch_size_max_train': 8,
                   'mode_head': 'none', 'mode_backbone': 'none'},
        'head': OmegaConf.create({'_target_': 'src.fewshot.utils.NoneHead', 'input_mode': 'cls'}),
        'device': DEVICE,
        'prompt': OmegaConf.create({'type': 'lettre', 'skip_classes': True}),
    })

    backbone = instantiate(cfg.backbone, tuning_config=cfg.peft_c, device=DEVICE)
    head = instantiate(cfg.head, num_features_in=cfg.backbone.num_features_out)
    model = FinetuneWrapper(backbone=backbone, head=head, device=DEVICE)
    model.init_peft(config=cfg.peft_c)
    model.to(DEVICE)

    # transformers-version compat shim for the Qwen last-layer hook: on recent
    # transformers the decoder layer returns a tuple, so `hs = out` breaks.
    # Mirrors the official LLaVA wrapper's tuple handling. No HEC file is modified.
    def _compat_last_layer_hook(mdl):
        acts = {}
        layer = mdl.model.language_model.layers[-1]

        def f(module, inp, out):
            hs = out[0] if isinstance(out, (tuple, list)) else out
            if hs.shape[1] > 1 and "x_norm_clstoken" not in acts:
                acts["x_norm_clstoken"] = hs[:, -1, :].detach()

        h = layer.register_forward_hook(f)

        def remove():
            h.remove()

        return acts, remove

    backbone.register_llm_last_layer_prompt_lasttok_hook = _compat_last_layer_hook
    print(f'model loaded on {DEVICE}', flush=True)

    records = []
    for ds_name in datasets:
        ways = 10 if ds_name == 'eurosat' else int(args.ways)
        ds = build_dataset({'subsample_classes': 'all'}, ds_name, PATH_BASE, -1)
        for seed in seeds:
            dl = EpisodicNShotKWayDataset(ds.test, num_query=args.n_query,
                                          num_support=args.n_support, num_ways=ways,
                                          num_episodes=1, image_size=224, seed=seed)
            sup, sl, qry, ql, l2s = dl[0]
            backbone.init_task_specific_data(sup, sl, l2s, ds_name, cfg.prompt, 'CD')
            sup, sl = sup.to(DEVICE), sl.to(DEVICE)
            qry, ql = qry.to(DEVICE), ql.to(DEVICE)
            with torch.no_grad():
                fsup = model.infer(sup)
                fqry = model.infer(qry)
            A = fsup['attn_heads'].float()
            Aq = fqry['attn_heads'].float()
            Ns, L, H, Dh = A.shape
            M = L * H
            A = A.reshape(Ns, M, Dh)
            Aq = Aq.reshape(Aq.shape[0], M, Dh)
            sup_cls, qry_cls = fsup['x_norm_clstoken'], fqry['x_norm_clstoken']
            Nq = Aq.shape[0]
            print(f'[{ds_name} seed {seed}] Ns={Ns} Nq={Nq} ways={ways} M={M}', flush=True)

            # ---------------- official static baselines ---------------- #
            sup_feats = {'attn_heads': fsup['attn_heads'], 'x_norm_clstoken': sup_cls}
            qry_feats = {'attn_heads': fqry['attn_heads'], 'x_norm_clstoken': qry_cls}
            rec = {'dataset': ds_name, 'seed': seed, 'M': M, 'ways': ways, 'acc': {}}
            for name, k in [('static_hec20', 20), ('static_hec8', 8)]:
                c = ClassifierHEC_V(input_mode='cls', k=k, tau=10, cov_reg=1.0,
                                    ensemble_method='mean_prob', context='Class')
                rec['acc'][name] = c.fit_pred_features(
                    model, sup_feats, sl, qry_feats, ql)['accuracy']

            # ---------------- shared OOF + GDA ---------------- #
            clf = ClassifierHEC_V(input_mode='cls', k=20, tau=10, cov_reg=1.0,
                                  ensemble_method='mean_prob', context='Class')
            hd = clf.get_rank_scores(A, sl)
            labels_unique = hd['labels_unique'].detach().cpu().numpy().astype(np.int64)
            yq_np = ql.cpu().numpy()
            z_oof, true_prob, mean_oof, y_idx = cross_fit_oof(clf, A, sl, n_folds=4)
            pool = torch.argsort(mean_oof, descending=True)[:POOL_SIZE]
            pool_np = pool.cpu().numpy()
            P = len(pool_np)
            rec['pool'] = pool_np
            rec['mean_oof'] = mean_oof.cpu().numpy()

            # static CF-20: OOF-mean top-20, uniform, official-path ensemble
            cf20 = np.argsort(-rec['mean_oof'])[:20]
            P_q = group_ensemble_official(clf, Aq, hd, [cf20] * Nq)
            rec['acc']['static_cf20'] = accuracy(P_q, labels_unique, yq_np)

            # pool_uniform64: control — uniform mean over the SAME 64-head pool the
            # router uses; isolates router value from pool-size value
            P_q = group_ensemble_official(clf, Aq, hd, [np.arange(P)] * Nq)
            rec['acc']['pool_uniform64'] = accuracy(P_q, labels_unique, yq_np)

            # ---------------- dynamic methods ---------------- #
            z_pool_sup = z_oof[:, pool, :].cpu().numpy()              # [Ns, P, K]
            y_idx_np = y_idx.cpu().numpy()

            # knn8: neighbour-conditioned top-8 (training-free)
            sim = F.normalize(qry_cls, dim=-1).float() @ F.normalize(sup_cls, dim=-1).float().t()
            nn_idx = sim.topk(4, dim=1).indices
            marg = head_margin_scores(z_oof, y_idx)                   # [Ns, M]
            scores_q = marg[nn_idx][:, :, pool].mean(dim=1)           # [Nq, P]
            sets_pos = list(scores_q.topk(BUDGET, dim=1).indices.cpu().numpy())
            P_q = group_ensemble_official(clf, Aq, hd, sets_pos)
            rec['acc']['knn8'] = accuracy(P_q, labels_unique, yq_np)
            rec['usage_knn8'] = usage_from_sets(sets_pos, P)

            # class_cond8: per-class margin head sets (training-free)
            sets_c = class_margin_sets(z_pool_sup, y_idx_np, budget=BUDGET, eps=0.0)
            class_sets_pos = [np.flatnonzero(sets_c[c]) for c in range(sets_c.shape[0])]
            P_q = class_ensemble_official(clf, Aq, hd, class_sets_pos)
            rec['acc']['class_cond8'] = accuracy(P_q, labels_unique, yq_np)
            rec['usage_class_cond8'] = usage_from_sets(class_sets_pos, P)
            rec['class_sets_sizes'] = np.array([len(s) for s in class_sets_pos])

            # router8: RouteHead-style learned router (pseudo labels per support sample)
            sets_sup = greedy_margin_sets(z_pool_sup, y_idx_np, budget=BUDGET, eps=0.0)
            torch.manual_seed(0)
            router = train_router(sup_cls, sets_sup, dim_head=64, lr=1e-3,
                                  epochs=300, l1=0.05, weight_decay=1e-4, device=DEVICE)
            with torch.no_grad():
                p_q = router(F.normalize(qry_cls, dim=-1).float())
            p_np = p_q.cpu().numpy()
            sets_pos = list(np.argsort(-p_np, axis=1)[:, :BUDGET])
            P_q = group_ensemble_official(clf, Aq, hd, sets_pos)
            rec['acc']['router8'] = accuracy(P_q, labels_unique, yq_np)
            rec['usage_router8'] = usage_from_sets(sets_pos, P)
            rec['router_activation'] = p_np.mean(0)
            rec['router_psup_sizes'] = sets_sup.sum(1)

            # router_soft: soft router probabilities over the whole pool
            idx = torch.as_tensor(pool_np, dtype=torch.long, device=Aq.device)
            z_pool = clf.get_pred_labels(Aq[:, idx, :], hd['head_means'][idx],
                                         hd['head_cov_inv'][idx], hd['log_prior'])[0]
            w = p_q / p_q.sum(dim=1, keepdim=True).clamp(min=1e-8)
            P_q = torch.einsum('np,npk->nk', w, z_pool)
            rec['acc']['router_soft'] = accuracy(P_q, labels_unique, yq_np)

            # router_thresh: hard activation where p > 0.5 (adaptive |S|, fallback top-8)
            hard = p_np > 0.5
            empty = ~hard.any(axis=1)
            if empty.any():
                topk8 = np.argsort(-p_np, axis=1)[:, :BUDGET]
                hard[empty] = False
                hard[np.where(empty)[0][:, None], topk8[empty]] = True
            sets_pos_t = [np.flatnonzero(hard[n]) for n in range(Nq)]
            P_q = group_ensemble_official(clf, Aq, hd, sets_pos_t)
            rec['acc']['router_thresh'] = accuracy(P_q, labels_unique, yq_np)
            rec['thresh_sizes'] = np.array([len(s) for s in sets_pos_t])

            # hybrid: average of static top-20 (official selection) and router_soft probs
            order20_in = torch.argsort(hd['head_scores'], descending=True)[:20]
            P_static = group_ensemble_official(clf, Aq, hd, [order20_in.cpu().numpy()] * Nq)
            P_q = 0.5 * P_static + 0.5 * torch.einsum('np,npk->nk', w, z_pool)
            rec['acc']['hybrid_static_router'] = accuracy(P_q, labels_unique, yq_np)

            # router_soft_p20: RouteHead-style small pool — router soft over OOF top-20
            pool20 = np.argsort(-rec['mean_oof'])[:20]
            sets_sup20 = greedy_margin_sets(z_oof[:, pool20].cpu().numpy(), y_idx_np,
                                            budget=BUDGET, eps=0.0, min_heads=4)
            torch.manual_seed(0)
            router20 = train_router(sup_cls, sets_sup20, dim_head=64, lr=1e-3,
                                    epochs=300, l1=0.05, weight_decay=1e-4, device=DEVICE)
            with torch.no_grad():
                p20 = router20(F.normalize(qry_cls, dim=-1).float())
            idx20 = torch.as_tensor(pool20, dtype=torch.long, device=Aq.device)
            z_pool20 = clf.get_pred_labels(Aq[:, idx20, :], hd['head_means'][idx20],
                                           hd['head_cov_inv'][idx20], hd['log_prior'])[0]
            w20 = p20 / p20.sum(dim=1, keepdim=True).clamp(min=1e-8)
            P_q = torch.einsum('np,npk->nk', w20, z_pool20)
            rec['acc']['router_soft_p20'] = accuracy(P_q, labels_unique, yq_np)

            # oracle_soft (DIAGNOSTIC — query labels): weight pool heads by their
            # true-class posterior; ceiling of per-query SOFT routing
            yq_idx_t = torch.searchsorted(hd['labels_unique'], ql)
            true_q_pool = z_pool[torch.arange(Nq, device=Aq.device), :, yq_idx_t]  # [Nq, P]
            w_or = true_q_pool / true_q_pool.sum(dim=1, keepdim=True).clamp(min=1e-8)
            P_q = torch.einsum('np,npk->nk', w_or, z_pool)
            rec['acc']['oracle_soft'] = accuracy(P_q, labels_unique, yq_np)

            # oracle8 (DIAGNOSTIC ONLY — uses query labels)
            o_sets = oracle_topk_sets(clf, Aq, hd, pool,
                                      torch.searchsorted(hd['labels_unique'], ql), BUDGET)
            P_q = group_ensemble_official(clf, Aq, hd, list(o_sets))
            rec['acc']['oracle8'] = accuracy(P_q, labels_unique, yq_np)
            # oracle_corr8: per-query top-8 among individually-correct heads (ceiling)
            o_sets = oracle_correct_topk_sets(clf, Aq, hd, pool,
                                              torch.searchsorted(hd['labels_unique'], ql), BUDGET)
            P_q = group_ensemble_official(clf, Aq, hd, list(o_sets))
            rec['acc']['oracle_corr8'] = accuracy(P_q, labels_unique, yq_np)

            # diversity diagnostics
            rec['jaccard_router8'] = mean_pairwise_jaccard(sets_pos)
            sup_top = np.argsort(-true_prob[:, pool].cpu().numpy(), axis=1)[:, :BUDGET]
            rec['jaccard_support_static8'] = mean_pairwise_jaccard(
                [sup_top[i] for i in range(sup_top.shape[0])])

            print(f'[{ds_name} seed {seed}] '
                  + ' '.join(f'{k}={100 * rec["acc"][k]:.1f}' for k in ACC_KEYS), flush=True)

            # global head-usage vectors (length M, zero outside pool)
            for key in ['usage_knn8', 'usage_class_cond8', 'usage_router8']:
                g = np.zeros(M)
                g[pool_np] = rec[key]
                rec[key + '_global'] = g
            records.append(rec)

            del A, Aq, hd, z_oof, sup, qry, sup_feats, qry_feats
            torch.cuda.empty_cache()

    # ---------------- summary ---------------- #
    out_npz = os.path.join(OUT_DIR, f'dynamic_hec_results{tag}.npy')
    np.save(out_npz, records, allow_pickle=True)

    lines = ['# Dynamic Head Routing HEC — results', '',
             '| dataset | seed | ways | ' + ' | '.join(ACC_KEYS) + ' |',
             '|' + '---|' * (3 + len(ACC_KEYS))]
    by_ds = {}
    for r in records:
        lines.append('| ' + ' | '.join([r['dataset'], str(r['seed']), str(r['ways'])]
                     + [f"{100 * r['acc'][k]:.1f}" for k in ACC_KEYS]) + ' |')
        by_ds.setdefault(r['dataset'], []).append(r)
    lines += ['', '## Mean over seeds',
              '| dataset | ' + ' | '.join(ACC_KEYS) + ' |',
              '|' + '---|' * (1 + len(ACC_KEYS))]
    for ds_name, rs in by_ds.items():
        vals = [100 * np.mean([r['acc'][k] for r in rs]) for k in ACC_KEYS]
        lines.append('| ' + ' | '.join([ds_name] + [f'{v:.1f}' for v in vals]) + ' |')

    lines += ['', '## Head usage distribution (top-10 heads per dynamic method)']
    H = 28  # Qwen2-VL-7B: 28 layers x 28 heads
    for ds_name, rs in by_ds.items():
        r = rs[0]
        for key in ['usage_knn8_global', 'usage_class_cond8_global', 'usage_router8_global']:
            u = r[key]
            idx = np.argsort(-u)[:10]
            idx = idx[u[idx] > 0]
            if len(idx) == 0:
                continue
            lines.append(f'- **{ds_name} / {key}**: '
                         + ', '.join(f'{i} (L{i // H}.H{i % H})' for i in idx[:10]))
        lines.append(f'- **{ds_name} / router8 query-set diversity** (mean pairwise jaccard) = '
                     f"{r['jaccard_router8']:.2f} vs support-static-8 "
                     f"{r['jaccard_support_static8']:.2f} (lower = more input-adaptive)")
        lines.append(f'- **{ds_name} / class_cond8 set sizes**: mean '
                     f"{r['class_sets_sizes'].mean():.1f}, min {r['class_sets_sizes'].min()}, "
                     f"max {r['class_sets_sizes'].max()} (adaptive budget <= {BUDGET})")

    report = '\n'.join(lines)
    with open(os.path.join(OUT_DIR, f'report{tag}.md'), 'w', encoding='utf-8') as f:
        f.write(report)
    print('\n' + report)
    print('\nSAVED', out_npz)


if __name__ == '__main__':
    main()

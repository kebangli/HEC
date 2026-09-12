#!/usr/bin/env python3
# Module B smoke test:
# 1) ClassifierHEC_V_refine with default params (fisher_ratio=1.0, power_p=1.0)
#    must be BIT-IDENTICAL to ClassifierHEC_V on random episodes.
# 2) Refinement paths (fisher / power / combo) must run and stay finite.
import sys
import torch
import numpy as np

sys.path.insert(0, "/data2/lbk/HEC")

from src.fewshot.classifier import ClassifierHEC_V
from src.fewshot.classifier_refine import ClassifierHEC_V_refine

BASE = dict(input_mode="cls", k=20, tau=10.0, cov_reg=1.0,
            ensemble_method="mean_prob", context="Class")

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device={device}")

L, H, Dh = 28, 28, 128
M = L * H

def make_episode(Ns=40, Nq=50, K=10):
    sfeat = {"attn_heads": torch.randn(Ns, L, H, Dh, device=device)}
    qfeat = {"attn_heads": torch.randn(Nq, L, H, Dh, device=device)}
    y_s = torch.repeat_interleave(torch.arange(K, device=device), Ns // K)
    y_q = torch.randint(0, K, (Nq,), device=device)
    l2s = {i: f"class_{i}" for i in range(K)}
    return sfeat, y_s, qfeat, y_q, l2s

def run(clf, ep):
    sfeat, y_s, qfeat, y_q, l2s = ep
    clf.init_task_specific_data(l2s, "smoke", "CD")
    with torch.inference_mode():
        out = clf.fit_pred_features(None, sfeat, y_s, qfeat, y_q)
    return out

# --- 1) bit-parity on 3 random episodes ---
ok = True
for trial in range(3):
    torch.manual_seed(100 + trial)
    ep = make_episode()
    a = run(ClassifierHEC_V(**BASE), ep)
    b = run(ClassifierHEC_V_refine(**BASE), ep)  # defaults: refinement off
    same_acc = a["accuracy"] == b["accuracy"]
    same_prob = np.array_equal(a["pred_prob"], b["pred_prob"])
    print(f"trial {trial}: base acc={a['accuracy']:.6f} refine-default acc={b['accuracy']:.6f} "
          f"acc_equal={same_acc} prob_equal={same_prob}")
    ok &= same_acc and same_prob
assert ok, "BIT-PARITY FAILED"
print("[PASS] default params reproduce baseline bit-identically\n")

# --- 2) refinement paths run and produce finite outputs ---
for name, over in [
    ("fisher_0.75", dict(fisher_ratio=0.75)),
    ("fisher_0.5",  dict(fisher_ratio=0.5)),
    ("fisher_0.25", dict(fisher_ratio=0.25)),
    ("fisher_0.5_norenorm", dict(fisher_ratio=0.5, mask_renorm=False)),
    ("power_0.5",   dict(power_p=0.5)),
    ("power_0.25",  dict(power_p=0.25)),
    ("combo_0.5_0.5", dict(fisher_ratio=0.5, power_p=0.5)),
    ("combo_0.25_0.5", dict(fisher_ratio=0.25, power_p=0.5)),
]:
    torch.manual_seed(200)
    ep = make_episode()
    out = run(ClassifierHEC_V_refine(**BASE, **over), ep)
    finite = np.isfinite(out["pred_prob"]).all()
    print(f"{name:<22s} acc={out['accuracy']:.6f} finite={finite}")
    assert finite, f"{name} produced non-finite probabilities"
print("[PASS] all refinement paths run and stay finite")

# --- 3) per-head Fisher mask keeps the right number of dims ---
torch.manual_seed(300)
ep = make_episode()
clf = ClassifierHEC_V_refine(**BASE, fisher_ratio=0.25)
sfeat = ep[0]["attn_heads"]
xs = torch.nn.functional.normalize(
    sfeat.reshape(sfeat.shape[0], L * H, Dh), dim=-1
)
mask = clf._fisher_mask(xs, ep[1])
assert mask.shape == (M, Dh)
assert mask.sum(dim=-1).unique().tolist() == [int(round(0.25 * Dh))], \
    f"unexpected kept dims: {mask.sum(dim=-1).unique().tolist()}"
print(f"[PASS] Fisher mask keeps {int(round(0.25*Dh))}/{Dh} dims per head independently")

print("\nALL SMOKE TESTS PASSED")

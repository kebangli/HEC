#!/usr/bin/env python3
"""Mechanism validation for the Head-Patch diagnostic (round 8).

Checks, on the real Qwen2VL backbone:
  1. model(**inputs, output_attentions=True) returns per-layer attention tensors
     even though the model was loaded with sdpa (HF auto-eager fallback?).
  2. image token positions in input_ids and the 2D grid size from image_grid_thw.
  3. consistency: per-head last-token feature f_m (captured by the official
     attn_heads pre-o_proj hook) ~= sum_p attn_w[m,p] * V_m(p)  (attention
     weights from the eager pass, V from a v_proj hook).
"""
import os, torch, numpy as np
import torch.nn.functional as F

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TQDM_DISABLE"] = "1"

torch.manual_seed(0)

from src.fewshot.backbone import QwenV2Wrapper

DEV = "cuda:0"  # with CUDA_VISIBLE_DEVICES=1, visible index 0 == physical GPU 1
bb = QwenV2Wrapper(
    model_name="QWENv2",
    weights="/data2/lbk/HEC_repro/weights/qwenv2",
    num_features_out=1280,
    output_llm_lasttok_heads=True,
    device=DEV,
)

# dummy task data -> TD prompt (skip_classes=True -> only domain prompt)
label_to_str = {i: f"c{i}" for i in range(10)}
bb.init_task_specific_data(
    torch.zeros(2, 3, 224, 224), torch.tensor([0, 1]),
    label_to_str, "oxford_pets",
    {"type": "lettre", "skip_classes": True}, "CD",
)
print("PROMPT:", repr(bb.current_text_prompt))

imgs = (torch.randint(0, 255, (2, 3, 224, 224)).float() / 255.0).to(DEV)

# ---------- official forward (as HEC-V does) ----------
with torch.no_grad():
    out = bb.forward(imgs)
attn_heads = out["attn_heads"]           # [B, L, H, Dh]
B, L, H, Dh = attn_heads.shape
print(f"attn_heads shape: {attn_heads.shape}  (L={L}, H={H}, Dh={Dh})")

# ---------- attention-capture forward ----------
model, processor = bb.model, bb.processor
chat = processor.apply_chat_template(
    conversation=[{
        "role": "user",
        "content": [
            {"type": "image", "image": "Any"},
            {"type": "text", "text": bb.current_text_prompt},
        ],
    }],
    tokenize=False, add_generation_prompt=True,
)
inputs = processor(text=[chat] * B, images=(255 * imgs).to(torch.uint8),
                   padding=True, return_tensors="pt")
inputs = {k: (v.to(DEV) if hasattr(v, "to") else v) for k, v in inputs.items()}

v_acts = {}
hooks = []
for i, layer in enumerate(model.model.layers):
    attn = layer.self_attn
    num_kv = attn.num_key_value_heads
    Dhh = attn.head_dim
    num_kv_groups = attn.num_heads // num_kv
    def _h(module, inp, out, i=i, num_kv=num_kv, Dhh=Dhh, nkg=num_kv_groups):
        # v_proj -> [B, T, num_kv*Dh]; repeat_kv to num_heads (GQA)
        v = out.view(out.shape[0], out.shape[1], num_kv, Dhh)
        v = v.repeat_interleave(nkg, dim=2)            # [B, T, num_heads, Dh]
        v_acts[i] = v.detach()
    hooks.append(attn.v_proj.register_forward_hook(_h))

with torch.no_grad():
    res = model(**inputs, output_attentions=True)
for h in hooks:
    h.remove()

attn = res.attentions
print(f"res.attentions len={len(attn)}")
for a in attn[:2]:
    print("   attn tensor:", None if a is None else tuple(a.shape))

# image token detection
img_id = model.config.image_token_id
ids0 = inputs["input_ids"][0].tolist()
img_pos = [p for p, t in enumerate(ids0) if t == img_id]
thw = inputs["image_grid_thw"]          # [B, 3]
print(f"image_token_id={img_id}  #img tokens={len(img_pos)}  grid_thw[0]={thw[0].tolist()}")
print(f"first 8 ids: {ids0[:8]}")
print(f"last 8 ids:  {ids0[-8:]}")

# ---------- consistency check f_m vs sum attn*w * v ----------
grid_h, grid_w = int(thw[0, 1]), int(thw[0, 2])
Nimg = grid_h * grid_w
print(f"grid = {grid_h}x{grid_w}  Nimg={Nimg}")

def check(ell, hh):
    f = attn_heads[0, ell, hh].float()                       # [Dh]
    aw = attn[ell][0, hh, -1, img_pos].float()               # [Nimg]
    v = v_acts[ell][0, img_pos, hh].float()                  # [Nimg, Dh]
    f_rec = (aw[:, None] * v).sum(0)
    cos = float(F.cosine_similarity(f[None], f_rec[None]).item())
    corr = float(np.corrcoef(f.cpu().numpy(), f_rec.cpu().numpy())[0, 1])
    return cos, corr, float(aw.sum().item())

for ell, hh in [(0, 0), (L // 2, H // 2), (L - 1, H - 1)]:
    cos, corr, awsum = check(ell, hh)
    print(f"layer {ell:2d} head {hh:2d}: cos(f, sum attn*v)={cos:.4f} corr={corr:.4f} sum(attn_img)={awsum:.3f}")

# per-layer image attention mass
for ell in [0, L // 2, L - 1]:
    aws = [attn[ell][0, hh, -1, img_pos].float().sum().item() for hh in range(H)]
    print(f"layer {ell}: mean image-attn mass over heads = {np.mean(aws):.3f} (min {np.min(aws):.3f} max {np.max(aws):.3f})")

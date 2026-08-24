r"""
Reproduces the attention math from tests/suites/attention-parity.ts in PyTorch
with the same synthetic inputs. If PyTorch output matches the JS output to
fp32 precision, our attention math == reference math. If it diverges,
something structural is wrong on the JS side.

Usage:
    python scripts/attention_parity_py.py > /tmp/py.json
    # In browser: run 'Attention parity (WebGPU vs Workers)' at T=50, dump `finalOut`
    # Diff the arrays.
"""
import json
import math
import sys

import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")

B, NH, DH = 1, 12, 64
H = NH * DH


def gen(shape, coef):
    n = 1
    for s in shape:
        n *= s
    return torch.tensor([math.sin(i * coef) for i in range(n)]).reshape(shape).float()


def gen_cos(shape, coef):
    n = 1
    for s in shape:
        n *= s
    return torch.tensor([math.cos(i * coef) for i in range(n)]).reshape(shape).float()


def run(T: int):
    x = gen((B, T, H), 0.017)
    wq = gen((H, H), 0.011) * 0.05
    wk = gen_cos((H, H), 0.013) * 0.05
    wv = gen((H, H), 0.019) * 0.05
    wd = gen_cos((H, H), 0.021) * 0.05

    q = (x.reshape(-1, H) @ wq.T).reshape(B, T, H).reshape(B, T, NH, DH).transpose(1, 2)
    k = (x.reshape(-1, H) @ wk.T).reshape(B, T, H).reshape(B, T, NH, DH).transpose(1, 2)
    v = (x.reshape(-1, H) @ wv.T).reshape(B, T, H).reshape(B, T, NH, DH).transpose(1, 2)

    scale = 1.0 / math.sqrt(DH)
    qk = torch.matmul(q, k.transpose(-1, -2)) * scale
    attn = F.softmax(qk, dim=-1)
    av = torch.matmul(attn, v)
    ctx = av.transpose(1, 2).reshape(B, T, H)
    final = (ctx.reshape(-1, H) @ wd.T).reshape(B, T, H)

    def stats(t: torch.Tensor):
        x = t.detach().numpy()
        return {
            "shape": list(x.shape),
            "mean": float(x.mean()),
            "std": float(x.std()),
            "min": float(x.min()),
            "max": float(x.max()),
        }

    return {
        "T": T,
        "qk": stats(qk),
        "softmax": stats(attn),
        "context": stats(ctx),
        "final": stats(final),
    }


if __name__ == "__main__":
    for T in (15, 50):
        r = run(T)
        print(json.dumps(r, indent=2))

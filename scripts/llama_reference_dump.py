r"""
Reference forward pass for Llama 3.2 1B via HuggingFace transformers.
Runs a text prompt through the model, dumps per-layer hidden states + logits
to a safetensors archive, and decodes the top-K next-token predictions.

The torchic demo's parity suite loads this dump and diffs per layer.

Usage:
    .venv\Scripts\python.exe scripts/llama_reference_dump.py \
        [--text "Hello world, I am a"] \
        [--repo unsloth/Llama-3.2-1B] \
        [--out ext/llama_ref_dump.safetensors]

Dumped tensors:
    token_ids                     [T]     float32   (int-valued)
    hidden_00                     [T, H]  float32   embed output
    hidden_01..hidden_{N-1}       [T, H]  float32   layer (i-1) output (pre-norm)
    pre_final_norm                [T, H]  float32   layer (N-1) output (pre-norm)
    hidden_{N}                    [T, H]  float32   post-final-norm state
    logits                        [T, V]  float32

HF's out.hidden_states convention: index i (i<N) is the input to layer i (or
equivalently, layer (i-1)'s output for i>=1); the final entry is post-final-norm.
Layer N-1's pre-norm output isn't in that tuple, so we capture it separately via
a forward-pre-hook on model.model.norm.
"""
import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="Hello world, I am a")
    ap.add_argument("--repo", default="unsloth/Llama-3.2-1B")
    ap.add_argument("--out", default="ext/llama_ref_dump.safetensors")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    print(f"loading tokenizer + model from {args.repo} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.repo)
    model = AutoModelForCausalLM.from_pretrained(
        args.repo,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.eval()

    enc = tokenizer(args.text, return_tensors="pt")
    ids = enc["input_ids"]
    tokens = ids[0].tolist()
    print(f"prompt: {args.text!r}")
    print(f"tokens ({len(tokens)}): {tokens}")
    print(f"decoded: {tokenizer.decode(tokens)!r}")

    pre_final_norm: dict[str, torch.Tensor] = {}
    def _capture_prenorm(_module, inputs):
        pre_final_norm["x"] = inputs[0].detach().clone()
    hook = model.model.norm.register_forward_pre_hook(_capture_prenorm)

    with torch.no_grad():
        out = model(ids, output_hidden_states=True, return_dict=True)
    hook.remove()

    dumps: dict[str, torch.Tensor] = {
        "token_ids": ids[0].to(torch.float32).contiguous(),
    }
    for i, h in enumerate(out.hidden_states):
        dumps[f"hidden_{i:02d}"] = h.squeeze(0).contiguous()
    dumps["pre_final_norm"] = pre_final_norm["x"].squeeze(0).contiguous()
    dumps["logits"] = out.logits.squeeze(0).contiguous()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_file(dumps, args.out)
    total_mb = sum(v.numel() * v.element_size() for v in dumps.values()) / 1e6
    print(f"wrote {args.out} ({total_mb:.1f} MB, {len(dumps)} tensors)")

    top = torch.topk(out.logits[0, -1], args.topk)
    print(f"\ntop-{args.topk} next-token predictions:")
    for logit, tok_id in zip(top.values.tolist(), top.indices.tolist()):
        text = tokenizer.decode([tok_id])
        print(f"  {tok_id:>7}  logit={logit:+.3f}  {text!r}")


if __name__ == "__main__":
    main()

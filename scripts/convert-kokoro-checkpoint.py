"""
Convert Kokoro-82M .pth checkpoint (torch pickle) to .safetensors.

Kokoro's checkpoint is a dict of {module_name: sub_state_dict} — one entry per
top-level nn.Module (bert, bert_encoder, predictor, text_encoder, decoder).
We flatten that into dotted paths matching what our Module.load_safetensors
consumes, strip "module." DataParallel prefixes, and save.

Voice packs (voices/*.pt) are single tensors of shape [511, 1, 256] holding
per-token style vectors. Save one under the "voice" key.

Usage:
    python scripts/convert-kokoro-checkpoint.py path/to/kokoro-v1_0.pth output.safetensors
    python scripts/convert-kokoro-checkpoint.py path/to/af_bella.pt output.safetensors --voice

Setup (uv, recommended):
    uv sync                 # installs torch + safetensors from pyproject.toml
    uv run scripts/convert-kokoro-checkpoint.py ...

Or plain pip / venv:
    python -m venv .venv && . .venv/bin/activate
    pip install torch safetensors
"""
import torch
from safetensors.torch import save_file

import argparse
from pathlib import Path


def strip_module_prefix(k: str) -> str:
    return k[len("module."):] if k.startswith("module.") else k


def convert_model(pth_path: str, out_path: str) -> None:
    print(f"loading {pth_path} ...")
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise SystemExit(f"expected dict at top level, got {type(checkpoint).__name__}")

    flat: dict[str, "torch.Tensor"] = {}
    for module_name, sub in checkpoint.items():
        if not isinstance(sub, dict):
            print(f"  skipping non-dict entry: {module_name} ({type(sub).__name__})")
            continue
        for k, v in sub.items():
            k = strip_module_prefix(k)
            full = f"{module_name}.{k}"
            if not torch.is_tensor(v):
                print(f"  skipping non-tensor: {full}")
                continue
            flat[full] = v.contiguous()

    print(f"flattened {len(flat)} tensors.")
    keys = list(flat.keys())
    for key in keys[:5] + ["..."] + keys[-5:]:
        if key == "...":
            print("  ...")
            continue
        t = flat[key]
        print(f"  {key:60s}  {list(t.shape)}  {t.dtype}")

    save_file(flat, out_path)
    print(f"wrote {out_path} ({Path(out_path).stat().st_size / 1e6:.1f} MB)")


def convert_voice(pt_path: str, out_path: str) -> None:
    print(f"loading voice {pt_path} ...")
    tensor = torch.load(pt_path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(tensor):
        raise SystemExit(f"expected a single tensor, got {type(tensor).__name__}")
    print(f"  shape={list(tensor.shape)}  dtype={tensor.dtype}")
    save_file({"voice": tensor.contiguous()}, out_path)
    print(f"wrote {out_path} ({Path(out_path).stat().st_size / 1024:.1f} KB)")


def main() -> None:
    assert __doc__
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("input", help="path to .pth (model) or .pt (voice) input")
    p.add_argument("output", help="path to .safetensors output")
    p.add_argument("--voice", action="store_true", help="input is a voice tensor (not a model)")
    args = p.parse_args()

    if args.voice:
        convert_voice(args.input, args.output)
    else:
        convert_model(args.input, args.output)


if __name__ == "__main__":
    main()

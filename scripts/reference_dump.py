r"""
Run reference Kokoro on the demo phoneme strings and dump per-stage tensor
stats. Compare against our JS side to find where they diverge.

Usage:
    python scripts/reference_dump.py --model path/to/kokoro-v1_0.pth \
        --voice path/to/af_bella.pt \
        [--samples tests/demos/kokoro/samples.json]

Prints:
  * pred_dur per phoneme (compare to our `durations:` log)
  * raw sigmoid.sum(-1) values (compare to our `raw sigmoid.sum:` log)
  * mean/std of intermediates (bert_out, d_en, d, x_after_lstm, duration_logits)

If pred_dur roughly matches ours, quality problem is downstream (F0/decoder).
If pred_dur differs materially, the predictor path is wrong somewhere.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")


def stats(name: str, t: torch.Tensor) -> str:
    x = t.detach().float().cpu().numpy()
    return f"{name:30s} shape={tuple(x.shape)}  mean={x.mean():+.4f}  std={x.std():.4f}  min={x.min():+.4f}  max={x.max():+.4f}"


def run_one(kmodel, phonemes: str, ref_s: torch.Tensor, speed: float = 1.0) -> None:
    vocab = kmodel.vocab
    ids = [vocab.get(p) for p in phonemes]
    ids = [i for i in ids if i is not None]
    input_ids = torch.LongTensor([[0, *ids, 0]]).to(kmodel.device)
    T = input_ids.shape[-1]

    input_lengths = torch.full((1,), T, device=kmodel.device, dtype=torch.long)
    text_mask = torch.arange(T).unsqueeze(0).expand(1, -1).type_as(input_lengths).to(kmodel.device)
    text_mask = torch.gt(text_mask + 1, input_lengths.unsqueeze(1))

    # Hook every ALBERT layer to print its output stats.
    layer_stats: list[str] = []
    layer_count = [0]
    def hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        layer_stats.append(stats(f"bert_layer_{layer_count[0]}", h))
        layer_count[0] += 1

    def embed_hook(module, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        layer_stats.append(stats("embeddings", h))

    handle = kmodel.bert.encoder.albert_layer_groups[0].albert_layers[0].register_forward_hook(hook)
    handle_e = kmodel.bert.embeddings.register_forward_hook(embed_hook)
    try:
        with torch.no_grad():
            bert_out = kmodel.bert(input_ids, attention_mask=(~text_mask).int())
            d_en = kmodel.bert_encoder(bert_out).transpose(-1, -2)
            s_prosody = ref_s[:, 128:]
            d = kmodel.predictor.text_encoder(d_en, s_prosody, input_lengths, text_mask)
            x, _ = kmodel.predictor.lstm(d)
            duration_logits = kmodel.predictor.duration_proj(x)
            raw = torch.sigmoid(duration_logits).sum(axis=-1) / speed
            pred_dur = torch.round(raw).clamp(min=1).long().squeeze()
    finally:
        handle.remove()
        handle_e.remove()

    for s in layer_stats:
        print(f"  {s}")
    print(f"  {stats('bert_out',        bert_out)}")
    print(f"  {stats('d_en',            d_en)}")
    print(f"  {stats('d (DurEnc out)',  d)}")
    print(f"  {stats('x (LSTM out)',    x)}")
    print(f"  {stats('duration_logits', duration_logits)}")
    print(f"  raw sigmoid.sum: {', '.join(f'{v:.2f}' for v in raw.squeeze().tolist())}")
    print(f"  pred_dur:        {', '.join(str(v) for v in pred_dur.tolist())}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="Path to kokoro-v1_0.pth (auto-download from HF if omitted).")
    ap.add_argument("--voice", default="af_bella", help="Voice name (auto-download) or path to .pt.")
    ap.add_argument("--samples", default="tests/demos/kokoro/samples.json")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()

    try:
        from kokoro import KModel
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("pip install kokoro>=0.9", file=sys.stderr)
        sys.exit(1)

    kmodel = KModel(repo_id="hexgrad/Kokoro-82M", model=args.model).eval()
    voice_path = args.voice
    if not Path(voice_path).exists():
        voice_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename=f"voices/{args.voice}.pt")
    pack = torch.load(voice_path, weights_only=True, map_location="cpu")

    with open(args.samples, "r", encoding="utf-8") as f:
        samples = json.load(f)

    for key, sample in samples.items():
        phonemes = sample["phonemes"]
        idx = len(phonemes) - 1
        ref_s = pack[idx]
        print(f"\n=== {key} :: '{sample['label']}' ===")
        print(f"phonemes = {phonemes!r}  (voice_idx={idx})")
        run_one(kmodel, phonemes, ref_s, speed=args.speed)


if __name__ == "__main__":
    main()

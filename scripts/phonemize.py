"""
Convert English text to Kokoro-82M input_ids via misaki (the same G2P library
Kokoro uses at training/inference time).

The demo currently hand-encodes phoneme sequences using expanded IPA, but the
checkpoint expects misaki's shorthand: `O` (not `oʊ`), `A` (not `eɪ`), `I`
(not `aɪ`), `W` (not `aʊ`), `Y` (not `ɔɪ`). Feeding the model expanded IPA is
out-of-distribution.

Also emits `voice_idx = len(phoneme_string) - 1`, matching kokoro's
`pack[len(ps)-1]` indexing — the JS side previously used `input_ids.length - 1`,
which is off by 2 (BOS + EOS wrap).

Usage:
    # Regenerate JSON for the 3 built-in demo samples:
    python scripts/phonemize.py --output tests/demos/kokoro/samples.json

    # One-off text -> tokens:
    python scripts/phonemize.py --text "The quick brown fox."

    # Read one line per stdin:
    echo "Hello world." | python scripts/phonemize.py -
"""
import argparse
import json
import sys
from typing import TypedDict

# Mirrors hexgrad/Kokoro-82M/config.json "vocab" exactly. Keeping it inline
# avoids a network roundtrip and keeps the script self-contained.
VOCAB: dict[str, int] = {
    ";": 1, ":": 2, ",": 3, ".": 4, "!": 5, "?": 6,
    "—": 9, "…": 10, "\"": 11, "(": 12, ")": 13, "“": 14, "”": 15, " ": 16,
    "\u0303": 17,
    "ʣ": 18, "ʥ": 19, "ʦ": 20, "ʨ": 21, "ᵝ": 22, "\uAB67": 23,
    "A": 24, "I": 25, "O": 31, "Q": 33, "S": 35, "T": 36, "W": 39, "Y": 41,
    "ᵊ": 42,
    "a": 43, "b": 44, "c": 45, "d": 46, "e": 47, "f": 48, "h": 50, "i": 51,
    "j": 52, "k": 53, "l": 54, "m": 55, "n": 56, "o": 57, "p": 58, "q": 59,
    "r": 60, "s": 61, "t": 62, "u": 63, "v": 64, "w": 65, "x": 66, "y": 67,
    "z": 68,
    "ɑ": 69, "ɐ": 70, "ɒ": 71, "æ": 72, "β": 75, "ɔ": 76, "ɕ": 77, "ç": 78,
    "ɖ": 80, "ð": 81, "ʤ": 82, "ə": 83, "ɚ": 85, "ɛ": 86, "ɜ": 87, "ɟ": 90,
    "ɡ": 92, "ɥ": 99, "ɨ": 101, "ɪ": 102, "ʝ": 103, "ɯ": 110, "ɰ": 111,
    "ŋ": 112, "ɳ": 113, "ɲ": 114, "ɴ": 115, "ø": 116, "ɸ": 118, "θ": 119,
    "œ": 120, "ɹ": 123, "ɾ": 125, "ɻ": 126, "ʁ": 128, "ɽ": 129, "ʂ": 130,
    "ʃ": 131, "ʈ": 132, "ʧ": 133, "ʊ": 135, "ʋ": 136, "ʌ": 138, "ɣ": 139,
    "ɤ": 140, "χ": 142, "ʎ": 143, "ʒ": 147, "ʔ": 148,
    "ˈ": 156, "ˌ": 157, "ː": 158, "ʰ": 162, "ʲ": 164,
    "↓": 169, "→": 171, "↗": 172, "↘": 173,
    "ᵻ": 177,
}

# Sentences the demo currently ships.
DEFAULT_SAMPLES: dict[str, str] = {
    "hello_world": "Hello world.",
    "pangram": "The quick brown fox jumps over the lazy dog.",
    "torchic_tagline": "Torchic runs entirely in your browser.",
    "lighthouse": (
        "A lonely lighthouse keeper found a glowing glass bottle washed ashore "
        "during the midnight storm. When he pulled the cork, a tiny cloud of "
        "silver dust escaped and spelled out his forgotten name in the air. "
        "He smiled for the first time in ten years and poured a cup of tea."
    ),
}


class Sample(TypedDict):
    label: str
    phonemes: str
    ids: list[int]
    voice_idx: int


def build_g2p(british: bool = False):
    # Deferred import so `--help` works without spacy model downloaded.
    from misaki import en, espeak
    try:
        fallback = espeak.EspeakFallback(british=british)
    except Exception as e:
        print(f"warning: espeak fallback disabled ({e}); OOD words will be dropped", file=sys.stderr)
        fallback = None
    return en.G2P(trf=False, british=british, fallback=fallback, unk="")


def tokenize(phonemes: str) -> list[int]:
    ids: list[int] = []
    dropped: list[str] = []
    for p in phonemes:
        i = VOCAB.get(p)
        if i is None:
            dropped.append(p)
            continue
        ids.append(i)
    if dropped:
        print(f"warning: dropped {len(dropped)} chars not in vocab: {''.join(dropped)!r}", file=sys.stderr)
    # BOS + EOS. Matches KModel.forward: input_ids = [0, *ids, 0].
    return [0, *ids, 0]


def phonemize_one(g2p, text: str) -> Sample:
    phonemes, _ = g2p(text)
    ids = tokenize(phonemes)
    return {
        "label": text,
        "phonemes": phonemes,
        "ids": ids,
        # kokoro/pipeline.py: pack[len(ps)-1], where ps is the phoneme string.
        "voice_idx": max(0, len(phonemes) - 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--text", help="Phonemize a single string.")
    ap.add_argument("--stdin", action="store_true", help="Read one line per sample from stdin (blank line = EOF).")
    ap.add_argument("--british", action="store_true", help="British English (default: American).")
    ap.add_argument("--output", "-o", help="Write JSON to this path (default: stdout).")
    ap.add_argument("--samples", action="store_true", help="Emit the 3 built-in demo samples as a keyed dict (default when no other input).")
    args = ap.parse_args()

    g2p = build_g2p(british=args.british)

    if args.text:
        out: object = phonemize_one(g2p, args.text)
    elif args.stdin:
        out = [phonemize_one(g2p, line) for line in (l.strip() for l in sys.stdin) if line]
    else:
        out = {k: phonemize_one(g2p, t) for k, t in DEFAULT_SAMPLES.items()}

    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"wrote {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()

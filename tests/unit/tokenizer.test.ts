import { readFileSync, existsSync } from "node:fs";
import { describe, it, expect } from "vitest";

import { LlamaTokenizer, type TokenizerSpec } from "../demos/llama/tokenizer";

const TOKENIZER_PATH = "ext/tokenizer/tokenizer.json";

// Fixtures were generated against HuggingFace's tokenizer on the same
// tokenizer.json spec (unsloth/Llama-3.2-1B). Each case's `ids` include
// the auto-BOS.
const KNOWN_ENCODINGS: { text: string; ids: number[] }[] = [
  { text: "Hello world, I am a", ids: [128000, 9906, 1917, 11, 358, 1097, 264] },
];

describe("LlamaTokenizer", () => {
  if (!existsSync(TOKENIZER_PATH)) {
    it.skip(`tokenizer.json not found at ${TOKENIZER_PATH} — download via scripts/llama_reference_dump.py`, () => {});
    return;
  }

  const spec = JSON.parse(readFileSync(TOKENIZER_PATH, "utf-8")) as TokenizerSpec;
  const tok = new LlamaTokenizer(spec);

  for (const { text, ids } of KNOWN_ENCODINGS) {
    it(`encodes ${JSON.stringify(text)} to the exact HF token IDs`, () => {
      expect(tok.encode(text)).toEqual(ids);
    });
  }

  it("prepends BOS by default and skips it when addBos=false", () => {
    const withBos = tok.encode("Hello");
    const withoutBos = tok.encode("Hello", false);
    expect(withBos[0]).toBe(tok.bosId);
    expect(withBos.slice(1)).toEqual(withoutBos);
  });

  const ROUND_TRIP_CASES = [
    "Hello world",
    "hello",
    "The quick brown fox jumps over the lazy dog.",
    "Numbers: 123 4567 89",
    "Punctuation!?...,",
    "Newlines\nare\ntricky",
    "\ttabs\there",
    "café résumé naïve",
    "😀🎉🚀 emoji",
    "print('hello')\nreturn 42",
    "     ",
    "\n\n\n",
    "\u2028\u2029weird whitespace",
    "Hello world".repeat(20),
  ];

  for (const text of ROUND_TRIP_CASES) {
    it(`round-trips ${JSON.stringify(text.slice(0, 40))}${text.length > 40 ? "..." : ""}`, () => {
      const ids = tok.encode(text, false);
      expect(tok.decode(ids)).toBe(text);
    });
  }

  it("handles embedded special tokens as opaque single IDs", () => {
    const ids = tok.encode("<|end_of_text|> after eot", false);
    expect(ids[0]).toBe(128001); // <|end_of_text|>
    // No BPE artifacts sneaking into the special token bytes.
    expect(tok.decode(ids)).toBe("<|end_of_text|> after eot");
  });

  it("BOS + double special sequence round-trips", () => {
    const text = "<|begin_of_text|>hi<|end_of_text|>";
    const ids = tok.encode(text, false);
    expect(ids[0]).toBe(128000);
    expect(ids[ids.length - 1]).toBe(128001);
    expect(tok.decode(ids)).toBe(text);
  });

  it("empty string encodes to just BOS", () => {
    expect(tok.encode("")).toEqual([tok.bosId]);
    expect(tok.encode("", false)).toEqual([]);
  });

  it("single space is one BPE token", () => {
    const ids = tok.encode(" ", false);
    expect(ids.length).toBe(1);
    expect(tok.decode(ids)).toBe(" ");
  });

  it("encode is deterministic", () => {
    const a = tok.encode("Determinism test 12345 !@#");
    const b = tok.encode("Determinism test 12345 !@#");
    expect(a).toEqual(b);
  });

  it("decode rejects unknown token id", () => {
    expect(() => tok.decode([tok.bosId! + 10_000_000])).toThrow(/unknown token id/);
  });
});

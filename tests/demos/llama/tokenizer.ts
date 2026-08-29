// Llama 3 tokenizer, hand-rolled from tokenizer.json — no dependencies.
// Byte-level BPE with the Llama-3-specific regex pre-tokenizer, ignore_merges
// short-circuit, and auto-BOS on encode. Verified against HuggingFace on the
// same tokenizer.json spec.

export interface TokenizerSpec {
  added_tokens: { id: number; content: string }[];
  pre_tokenizer: {
    type: "Sequence";
    pretokenizers: (
      | { type: "Split"; pattern: { Regex: string } }
      | { type: "ByteLevel" }
      | { type: string }
    )[];
  };
  model: {
    type: "BPE";
    vocab: Record<string, number>;
    merges: [string, string][];
    ignore_merges?: boolean;
  };
}

// GPT-2's byte-to-unicode mapping: every byte 0..255 to a printable Unicode
// char, keeping printable ASCII/Latin-1 as-is and shifting control chars and
// whitespace out of the way so they can appear inside vocab entries.
function bytesToUnicode(): { encoder: string[]; decoder: Map<string, number> } {
  const bs: number[] = [];
  for (let c = 0x21; c <= 0x7e; c++) bs.push(c);
  for (let c = 0xa1; c <= 0xac; c++) bs.push(c);
  for (let c = 0xae; c <= 0xff; c++) bs.push(c);
  const cs = [...bs];
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const encoder = new Array<string>(256);
  const decoder = new Map<string, number>();
  for (let i = 0; i < bs.length; i++) {
    const ch = String.fromCodePoint(cs[i]);
    encoder[bs[i]] = ch;
    decoder.set(ch, bs[i]);
  }
  return { encoder, decoder };
}

const MERGE_SEP = "\u0000";

export class LlamaTokenizer {
  private readonly vocab: Map<string, number>;
  private readonly invVocab: (string | undefined)[];
  private readonly mergeRank: Map<string, number>;
  private readonly ignoreMerges: boolean;
  private readonly specialContentToId: Map<string, number>;
  private readonly specialIdToContent: Map<number, string>;
  private readonly byteEncoder: string[];
  private readonly byteDecoder: Map<string, number>;
  private readonly pat: RegExp;
  private readonly specialSplitPat: RegExp;
  private readonly textEncoder = new TextEncoder();
  private readonly textDecoder = new TextDecoder("utf-8");
  readonly bosId: number | undefined;

  constructor(spec: TokenizerSpec) {
    if (spec.model.type !== "BPE") {
      throw new Error(`LlamaTokenizer: expected BPE model, got ${spec.model.type}`);
    }
    this.vocab = new Map(Object.entries(spec.model.vocab));
    this.ignoreMerges = spec.model.ignore_merges === true;

    this.mergeRank = new Map();
    for (let i = 0; i < spec.model.merges.length; i++) {
      const [a, b] = spec.model.merges[i];
      this.mergeRank.set(a + MERGE_SEP + b, i);
    }

    this.specialContentToId = new Map();
    this.specialIdToContent = new Map();
    for (const item of spec.added_tokens) {
      this.specialContentToId.set(item.content, item.id);
      this.specialIdToContent.set(item.id, item.content);
    }

    let maxId = -1;
    for (const v of this.vocab.values()) if (v > maxId) maxId = v;
    for (const k of this.specialIdToContent.keys()) if (k > maxId) maxId = k;
    this.invVocab = new Array<string | undefined>(maxId + 1);
    for (const [s, i] of this.vocab) this.invVocab[i] = s;
    for (const [i, s] of this.specialIdToContent) this.invVocab[i] = s;

    const { encoder, decoder } = bytesToUnicode();
    this.byteEncoder = encoder;
    this.byteDecoder = decoder;

    const seq = spec.pre_tokenizer;
    if (seq.type !== "Sequence") {
      throw new Error(`LlamaTokenizer: expected Sequence pre_tokenizer, got ${seq.type}`);
    }
    const split = seq.pretokenizers[0];
    if (split.type !== "Split" || !("pattern" in split)) {
      throw new Error(`LlamaTokenizer: expected first pretokenizer to be Split with Regex pattern`);
    }
    // The pattern uses Rust regex's `(?i:...)` inline modifier which JS RegExp
    // doesn't accept; convert each letter inside to a case-insensitive char class.
    const rustPattern = (split as { pattern: { Regex: string } }).pattern.Regex;
    const jsPattern = rustPattern.replace(/\(\?i:([^)]*)\)/g, (_, inner: string) => {
      const expanded = inner.replace(/[a-zA-Z]/g, (c) => `[${c.toLowerCase()}${c.toUpperCase()}]`);
      return `(?:${expanded})`;
    });
    this.pat = new RegExp(jsPattern, "gu");

    this.bosId = this.specialContentToId.get("<|begin_of_text|>");

    // Sort special tokens longest-first so alternation regex prefers longer matches.
    const specials = [...this.specialContentToId.keys()].sort((a, b) => b.length - a.length);
    const escaped = specials.map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    this.specialSplitPat = new RegExp(`(${escaped.join("|")})`, "g");
  }

  encode(text: string, addBos = true): number[] {
    const ids: number[] = [];
    if (addBos && this.bosId !== undefined) ids.push(this.bosId);
    for (const part of text.split(this.specialSplitPat)) {
      if (!part) continue;
      const specialId = this.specialContentToId.get(part);
      if (specialId !== undefined) {
        ids.push(specialId);
      } else {
        this.encodeOrdinary(part, ids);
      }
    }
    return ids;
  }

  private encodeOrdinary(text: string, ids: number[]): void {
    for (const match of text.matchAll(this.pat)) {
      const bytes = this.textEncoder.encode(match[0]);
      let bytelevel = "";
      for (const b of bytes) bytelevel += this.byteEncoder[b];
      for (const piece of this.bpe(bytelevel)) {
        const tid = this.vocab.get(piece);
        if (tid === undefined) throw new Error(`LlamaTokenizer: unknown BPE piece ${JSON.stringify(piece)}`);
        ids.push(tid);
      }
    }
  }

  private bpe(word: string): string[] {
    if (this.ignoreMerges && this.vocab.has(word)) return [word];
    // Iterate Unicode code points, not UTF-16 code units — some byte-level
    // chars sit above 0xFFFF and would otherwise split into surrogate pairs.
    let pieces = [...word];
    while (pieces.length >= 2) {
      let bestRank = -1;
      for (let i = 0; i < pieces.length - 1; i++) {
        const r = this.mergeRank.get(pieces[i] + MERGE_SEP + pieces[i + 1]);
        if (r !== undefined && (bestRank === -1 || r < bestRank)) bestRank = r;
      }
      if (bestRank === -1) break;
      const merged: string[] = [];
      let i = 0;
      while (i < pieces.length) {
        if (
          i < pieces.length - 1 &&
          this.mergeRank.get(pieces[i] + MERGE_SEP + pieces[i + 1]) === bestRank
        ) {
          merged.push(pieces[i] + pieces[i + 1]);
          i += 2;
        } else {
          merged.push(pieces[i]);
          i += 1;
        }
      }
      pieces = merged;
    }
    return pieces;
  }

  decode(ids: number[]): string {
    const segments: string[] = [];
    let buf = "";
    for (const tid of ids) {
      const s = this.invVocab[tid];
      if (s === undefined) throw new Error(`LlamaTokenizer: unknown token id ${tid}`);
      if (this.specialIdToContent.has(tid)) {
        if (buf) {
          segments.push(this.decodeBytelevel(buf));
          buf = "";
        }
        segments.push(s);
      } else {
        buf += s;
      }
    }
    if (buf) segments.push(this.decodeBytelevel(buf));
    return segments.join("");
  }

  private decodeBytelevel(s: string): string {
    const cps = [...s];
    const bytes = new Uint8Array(cps.length);
    for (let i = 0; i < cps.length; i++) {
      const b = this.byteDecoder.get(cps[i]);
      if (b === undefined) throw new Error(`LlamaTokenizer: unknown byte-level char ${JSON.stringify(cps[i])}`);
      bytes[i] = b;
    }
    return this.textDecoder.decode(bytes);
  }
}

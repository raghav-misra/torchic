import { parseSafetensors } from "../../../src/nn/safetensors";
import type { SafetensorsMap } from "../../../src/nn/safetensors";

interface HeaderEntry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

// Reads the safetensors header only from the first few KB of the file.
async function readHeader(file: File): Promise<{ header: Record<string, HeaderEntry>; dataStart: number }> {
  const lenBuf = await file.slice(0, 8).arrayBuffer();
  const headerLen = Number(new DataView(lenBuf).getBigUint64(0, true));
  const headerBuf = await file.slice(8, 8 + headerLen).arrayBuffer();
  const raw = JSON.parse(new TextDecoder().decode(headerBuf)) as Record<string, HeaderEntry>;
  return { header: raw, dataStart: 8 + headerLen };
}

// Parse only the tensors whose name matches `filter`, streaming byte ranges
// straight from a File object. Peak memory stays around the wanted subset —
// the 2.5 GB safetensors is never held whole in an ArrayBuffer (which
// browser File API can refuse for buffers near/over 2 GB).
export async function parsePartialFromFile(
  file: File,
  filter: (name: string) => boolean,
): Promise<SafetensorsMap> {
  const { header, dataStart } = await readHeader(file);

  const wanted: { name: string; entry: HeaderEntry }[] = [];
  for (const [name, entry] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    if (filter(name)) wanted.push({ name, entry });
  }

  const encoder = new TextEncoder();
  // Rewrite offsets to be dense within the mini-file we'll build below.
  let cursor = 0;
  const rewritten = wanted.map(({ name, entry }) => {
    const bytes = entry.data_offsets[1] - entry.data_offsets[0];
    const newEntry: HeaderEntry = { ...entry, data_offsets: [cursor, cursor + bytes] };
    cursor += bytes;
    return { name, oldBegin: entry.data_offsets[0], bytes, entry: newEntry };
  });
  const miniHeader = Object.fromEntries(rewritten.map((w) => [w.name, w.entry]));
  const miniHeaderBytes = encoder.encode(JSON.stringify(miniHeader));

  const bodyStart = 8 + miniHeaderBytes.byteLength;
  const out = new ArrayBuffer(bodyStart + cursor);
  new DataView(out).setBigUint64(0, BigInt(miniHeaderBytes.byteLength), true);
  new Uint8Array(out, 8, miniHeaderBytes.byteLength).set(miniHeaderBytes);
  const dst = new Uint8Array(out, bodyStart);

  for (const w of rewritten) {
    const chunk = await file.slice(dataStart + w.oldBegin, dataStart + w.oldBegin + w.bytes).arrayBuffer();
    dst.set(new Uint8Array(chunk), w.entry.data_offsets[0]);
  }

  return parseSafetensors(out);
}

// Predicates for the per-layer parity path.
export const isEmbedKey = (k: string) => k === "model.embed_tokens.weight";
export const isFinalNormKey = (k: string) => k === "model.norm.weight";
export const isLayerKey = (i: number) => (k: string) => k.startsWith(`model.layers.${i}.`);

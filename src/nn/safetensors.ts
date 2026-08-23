// Minimal safetensors reader with BF16 -> F32 upcast.
// Format: 8-byte little-endian u64 = header length, then a UTF-8 JSON header
// describing each tensor's dtype/shape/byte range, then the raw tensor data.
// Spec: https://github.com/huggingface/safetensors

export interface  SafetensorsEntry {
  shape: number[];
  data: Float32Array;
}

export type SafetensorsMap = Record<string, SafetensorsEntry>;

interface RawHeaderEntry {
  dtype: string;
  shape: number[];
  data_offsets: [number, number];
}

// bf16_bits << 16 == f32_bits (sign, exp, high 7 bits of mantissa all in place).
function upcastBF16(raw: Uint8Array, count: number): Float32Array {
  const out = new Float32Array(count);
  const u32 = new Uint32Array(out.buffer);
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  for (let i = 0; i < count; i++) {
    u32[i] = dv.getUint16(i * 2, true) << 16;
  }
  return out;
}

// f16 (half) -> f32. Not SIMD but Kokoro ships as bf16 so this path is rare.
function upcastF16(raw: Uint8Array, count: number): Float32Array {
  const out = new Float32Array(count);
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  for (let i = 0; i < count; i++) {
    const h = dv.getUint16(i * 2, true);
    const sign = (h & 0x8000) << 16;
    const exp = (h >> 10) & 0x1f;
    const mant = h & 0x3ff;
    let f: number;
    if (exp === 0) {
      // Subnormal or zero.
      if (mant === 0) f = 0;
      else {
        // Normalize: shift mantissa left until the implicit bit is at bit 10.
        let e = 1;
        let m = mant;
        while ((m & 0x400) === 0) {
          m <<= 1;
          e--;
        }
        m &= 0x3ff;
        const bits = sign | ((e + 112) << 23) | (m << 13);
        f = new Float32Array(new Uint32Array([bits]).buffer)[0];
      }
    } else if (exp === 0x1f) {
      // Inf or NaN.
      const bits = sign | 0x7f800000 | (mant << 13);
      f = new Float32Array(new Uint32Array([bits]).buffer)[0];
    } else {
      const bits = sign | ((exp + 112) << 23) | (mant << 13);
      f = new Float32Array(new Uint32Array([bits]).buffer)[0];
    }
    out[i] = f;
  }
  return out;
}

export function parseSafetensors(buffer: ArrayBuffer): SafetensorsMap {
  if (buffer.byteLength < 8) throw new Error(`safetensors: buffer too small (${buffer.byteLength})`);
  const dv = new DataView(buffer);
  const headerLen = Number(dv.getBigUint64(0, true));
  if (headerLen <= 0 || 8 + headerLen > buffer.byteLength) {
    throw new Error(`safetensors: invalid header length ${headerLen}`);
  }

  const headerBytes = new Uint8Array(buffer, 8, headerLen);
  const headerJson = new TextDecoder().decode(headerBytes);
  const header = JSON.parse(headerJson) as Record<string, RawHeaderEntry>;

  const dataStart = 8 + headerLen;
  const out: SafetensorsMap = {};
  for (const [name, entry] of Object.entries(header)) {
    if (name === "__metadata__") continue;
    const [begin, end] = entry.data_offsets;
    const raw = new Uint8Array(buffer, dataStart + begin, end - begin);
    const count = entry.shape.reduce((a, b) => a * b, 1);
    let data: Float32Array;
    switch (entry.dtype) {
      case "F32":
      case "float32": {
        // Aligned reinterpret when possible; copy otherwise to avoid alignment traps.
        const byteOffset = dataStart + begin;
        if (byteOffset % 4 === 0) {
          data = new Float32Array(buffer, byteOffset, count);
        } else {
          data = new Float32Array(count);
          new Uint8Array(data.buffer).set(raw);
        }
        break;
      }
      case "BF16":
      case "bfloat16":
        data = upcastBF16(raw, count);
        break;
      case "F16":
      case "float16":
        data = upcastF16(raw, count);
        break;
      default:
        throw new Error(`safetensors: unsupported dtype '${entry.dtype}' for tensor '${name}'`);
    }
    out[name] = { shape: entry.shape, data };
  }
  return out;
}

// Fetches and parses a safetensors file from a URL. In the browser we get
// SharedArrayBuffer / MessagePort headers baked in via COOP/COEP; regular
// same-origin fetches are the common case for local model files.
export async function fetchSafetensors(url: string): Promise<SafetensorsMap> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`safetensors: fetch ${url} failed: ${res.status} ${res.statusText}`);
  const buf = await res.arrayBuffer();
  return parseSafetensors(buf);
}

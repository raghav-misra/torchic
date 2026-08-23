import { describe, it, expect } from "vitest";
import { parseSafetensors, saveSafetensors } from "../../src/nn/safetensors";

describe("safetensors parse", () => {
  it("round-trips F32 save -> parse", () => {
    const w = new Float32Array([1, 2, 3, 4, 5, 6]);
    const b = new Float32Array([10, 20]);
    const buf = saveSafetensors({
      "layer.weight": { shape: [2, 3], data: w },
      "layer.bias": { shape: [2], data: b },
    });
    const parsed = parseSafetensors(buf);
    expect(Object.keys(parsed).sort()).toEqual(["layer.bias", "layer.weight"]);
    expect(parsed["layer.weight"].shape).toEqual([2, 3]);
    expect(Array.from(parsed["layer.weight"].data)).toEqual([1, 2, 3, 4, 5, 6]);
    expect(parsed["layer.bias"].shape).toEqual([2]);
    expect(Array.from(parsed["layer.bias"].data)).toEqual([10, 20]);
  });

  it("upcasts BF16 to F32", () => {
    // Build a tiny handcrafted safetensors buffer with one BF16 tensor.
    const values = [1.0, -2.5, 3.5];
    const bf16Bits = values.map((v) => {
      const u32 = new Uint32Array(new Float32Array([v]).buffer)[0];
      return u32 >>> 16;
    });
    const dataBytes = new Uint8Array(bf16Bits.length * 2);
    const dv = new DataView(dataBytes.buffer);
    for (let i = 0; i < bf16Bits.length; i++) dv.setUint16(i * 2, bf16Bits[i], true);

    const header = { t: { dtype: "BF16", shape: [3], data_offsets: [0, dataBytes.length] } };
    const headerJson = new TextEncoder().encode(JSON.stringify(header));
    const padded = (headerJson.length + 7) & ~7;
    const buf = new ArrayBuffer(8 + padded + dataBytes.length);
    new DataView(buf).setBigUint64(0, BigInt(padded), true);
    new Uint8Array(buf, 8, headerJson.length).set(headerJson);
    new Uint8Array(buf, 8 + headerJson.length, padded - headerJson.length).fill(0x20);
    new Uint8Array(buf, 8 + padded, dataBytes.length).set(dataBytes);

    const parsed = parseSafetensors(buf);
    for (let i = 0; i < values.length; i++) {
      // BF16 keeps sign + exp + top 7 mantissa bits, so tiny values are exact.
      expect(parsed.t.data[i]).toBeCloseTo(values[i], 2);
    }
  });

  it("throws on unknown dtype", () => {
    const header = { t: { dtype: "INT8", shape: [1], data_offsets: [0, 1] } };
    const headerJson = new TextEncoder().encode(JSON.stringify(header));
    const padded = (headerJson.length + 7) & ~7;
    const buf = new ArrayBuffer(8 + padded + 1);
    new DataView(buf).setBigUint64(0, BigInt(padded), true);
    new Uint8Array(buf, 8, headerJson.length).set(headerJson);
    new Uint8Array(buf, 8 + headerJson.length, padded - headerJson.length).fill(0x20);
    expect(() => parseSafetensors(buf)).toThrow(/unsupported dtype/);
  });
});

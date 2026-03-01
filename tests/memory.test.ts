import { describe, it, expect } from "vitest";
import { MemoryAllocator } from "../src/backend/workers/memory";

function makeSAB(bytes: number) {
  return new SharedArrayBuffer(bytes);
}

describe("MemoryAllocator", () => {
  it("returns 4-byte aligned offsets", () => {
    const alloc = new MemoryAllocator(makeSAB(4096));
    for (let i = 0; i < 10; i++) {
      const off = alloc.allocate(13); // deliberately non-aligned request
      expect(off % 4).toBe(0);
    }
  });

  it("recycles freed bucket blocks (LIFO)", () => {
    const alloc = new MemoryAllocator(makeSAB(4096));
    const a = alloc.allocate(16);
    alloc.free(a, 16);
    const b = alloc.allocate(16);
    expect(b).toBe(a);
  });

  it("bucket recycling preserves capacity across many alloc/free cycles", () => {
    const alloc = new MemoryAllocator(makeSAB(4096));
    const size = 64;
    const offsets: number[] = [];

    // Fill up with same-size blocks
    for (let i = 0; i < 4096 / size; i++) {
      offsets.push(alloc.allocate(size));
    }

    // Free all
    for (const off of offsets) alloc.free(off, size);

    // Re-allocate the same count — should succeed without OOM
    for (let i = 0; i < 4096 / size; i++) {
      expect(() => alloc.allocate(size)).not.toThrow();
    }
  });

  it("throws on OOM", () => {
    const alloc = new MemoryAllocator(makeSAB(64));
    alloc.allocate(64);
    expect(() => alloc.allocate(4)).toThrow(/Out of memory/);
  });

  it("keeps stats consistent through mixed alloc/free", () => {
    const total = 8192;
    const alloc = new MemoryAllocator(makeSAB(total));

    const offsets: { offset: number; size: number }[] = [];
    const sizes = [16, 256, 32, 1024, 64];
    for (const s of sizes) {
      offsets.push({ offset: alloc.allocate(s), size: s });
    }

    // Free every other one
    for (let i = 0; i < offsets.length; i += 2) {
      alloc.free(offsets[i].offset, offsets[i].size);
    }

    const stats = alloc.getStats();
    expect(stats.total).toBe(total);
    expect(stats.used + stats.free).toBe(total);
    expect(stats.used).toBeGreaterThan(0);
    expect(stats.free).toBeGreaterThan(0);
  });
});

// Segregated free-list allocator.
//
// Small allocations (≤ MAX_BUCKET_SIZE) are rounded up to the next power-of-2
// and served from per-size-class LIFO stacks: O(1) alloc and free.
// NN workloads reuse the same tensor shapes each iteration, so recycled
// blocks satisfy most requests without falling through to the large path.
//
// Large allocations fall back to a coalescing sorted free-list (first-fit).

const MIN_BUCKET_BITS = 2; // 4 bytes = one Float32
const MAX_BUCKET_BITS = 20; // 1 MiB
const NUM_BUCKETS = MAX_BUCKET_BITS - MIN_BUCKET_BITS + 1;

function sizeClassIndex(size: number): number {
  // 32 - clz32(size - 1) gives ceil(log2(size)), clamped to MIN_BUCKET_BITS
  const bits = Math.max(MIN_BUCKET_BITS, 32 - Math.clz32(size - 1));
  return bits - MIN_BUCKET_BITS;
}

function alignUp(size: number): number {
  return (size + 3) & ~3;
}

interface FreeBlock {
  offset: number;
  size: number;
}

export class MemoryAllocator {
  private totalSize: number;

  // Segregated buckets: index i holds blocks of size 2^(i + MIN_BUCKET_BITS).
  // Each bucket is a stack of byte-offsets (LIFO).
  private buckets: number[][];

  // Fallback sorted free-list for allocations > MAX_BUCKET_SIZE
  private largeList: FreeBlock[] = [];

  constructor(sizeOrBuffer: number | SharedArrayBuffer | ArrayBuffer, startOffset = 0) {
    this.totalSize =
      typeof sizeOrBuffer === "number" ? sizeOrBuffer : sizeOrBuffer.byteLength;
    this.buckets = Array.from({ length: NUM_BUCKETS }, () => []);
    // Reserve [0, startOffset) for external tenants (e.g. WASM's data section +
    // stack region). All returned offsets are guaranteed >= startOffset.
    const usable = this.totalSize - startOffset;
    this.largeList = usable > 0 ? [{ offset: startOffset, size: usable }] : [];
  }

  allocate(size: number): number {
    const aligned = alignUp(size);

    if (aligned <= 1 << MAX_BUCKET_BITS) {
      return this.allocSmall(aligned);
    }
    return this.allocLarge(aligned);
  }

  free(offset: number, size: number): void {
    const aligned = alignUp(size);

    if (aligned <= 1 << MAX_BUCKET_BITS) {
      const bucket = sizeClassIndex(aligned);
      const bucketSize = 1 << (bucket + MIN_BUCKET_BITS);
      // Rounded-up allocations return at bucket granularity, not the caller's
      // original size.
      if (aligned <= bucketSize) {
        this.buckets[bucket].push(offset);
        return;
      }
    }

    this.freeLarge(offset, aligned);
  }

  private allocSmall(aligned: number): number {
    const bucket = sizeClassIndex(aligned);
    const stack = this.buckets[bucket];

    if (stack.length > 0) {
      return stack.pop() as number;
    }

    // Nothing recycled; carve from the large free-list
    const bucketSize = 1 << (bucket + MIN_BUCKET_BITS);
    return this.allocLarge(bucketSize);
  }

  private allocLarge(size: number): number {
    const off = this.tryAllocLarge(size);
    if (off !== -1) return off;

    // Segregated buckets never coalesce, so long-running loops that carve
    // many small-then-large allocations can starve the large free-list even
    // with plenty of total free memory. Drain the bucket stacks back into
    // the large list (which coalesces) and retry once before giving up.
    this.drainBucketsToLarge();
    const retry = this.tryAllocLarge(size);
    if (retry !== -1) return retry;

    const s = this.getStats();
    throw new Error(
      `Out of memory: requested ${size} bytes, but no block large enough found. ` +
        `heap=${this.totalSize} used=${s.used} free=${s.free} largestFree=${s.largestFree} fragments=${s.fragments}`,
    );
  }

  private tryAllocLarge(size: number): number {
    for (let i = 0; i < this.largeList.length; i++) {
      const block = this.largeList[i];
      if (block.size >= size) {
        const offset = block.offset;
        if (block.size === size) {
          this.largeList.splice(i, 1);
        } else {
          block.offset += size;
          block.size -= size;
        }
        return offset;
      }
    }
    return -1;
  }

  private drainBucketsToLarge(): void {
    for (let bi = 0; bi < NUM_BUCKETS; bi++) {
      const stack = this.buckets[bi];
      if (stack.length === 0) continue;
      const bucketSize = 1 << (bi + MIN_BUCKET_BITS);
      for (const offset of stack) this.freeLarge(offset, bucketSize);
      stack.length = 0;
    }
  }

  private freeLarge(offset: number, size: number): void {
    let lo = 0;
    let hi = this.largeList.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (this.largeList[mid].offset < offset) lo = mid + 1;
      else hi = mid;
    }
    this.largeList.splice(lo, 0, { offset, size });
    this.coalesceLarge(lo);
  }

  private coalesceLarge(index: number): void {
    // Merge with right neighbour
    if (index + 1 < this.largeList.length) {
      const curr = this.largeList[index];
      const next = this.largeList[index + 1];
      if (curr.offset + curr.size === next.offset) {
        curr.size += next.size;
        this.largeList.splice(index + 1, 1);
      }
    }
    // Merge with left neighbour
    if (index > 0) {
      const prev = this.largeList[index - 1];
      const curr = this.largeList[index];
      if (prev.offset + prev.size === curr.offset) {
        prev.size += curr.size;
        this.largeList.splice(index, 1);
      }
    }
  }

  getStats() {
    let bucketFree = 0;
    for (let i = 0; i < NUM_BUCKETS; i++) {
      bucketFree += this.buckets[i].length * (1 << (i + MIN_BUCKET_BITS));
    }
    const largeFree = this.largeList.reduce((acc, b) => acc + b.size, 0);
    const freeBytes = bucketFree + largeFree;
    let largest = 0;
    for (const b of this.largeList) if (b.size > largest) largest = b.size;
    return {
      total: this.totalSize,
      used: this.totalSize - freeBytes,
      free: freeBytes,
      largestFree: largest,
      fragments: this.largeList.length,
    };
  }
}

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

  constructor(sizeOrBuffer: number | SharedArrayBuffer | ArrayBuffer) {
    this.totalSize =
      typeof sizeOrBuffer === "number" ? sizeOrBuffer : sizeOrBuffer.byteLength;
    this.buckets = Array.from({ length: NUM_BUCKETS }, () => []);
    this.largeList = [{ offset: 0, size: this.totalSize }];
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

    throw new Error(
      `Out of memory: requested ${size} bytes, but no block large enough found.`,
    );
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
    return {
      total: this.totalSize,
      used: this.totalSize - freeBytes,
      free: freeBytes,
      fragments: this.largeList.length,
    };
  }
}

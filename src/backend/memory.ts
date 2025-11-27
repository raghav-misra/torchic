// Memory management using SharedArrayBuffer
export class MemoryAllocator {
    private buffer: SharedArrayBuffer;
    private freeList: { offset: number; size: number }[];
    private totalSize: number;

    constructor(buffer: SharedArrayBuffer) {
        this.buffer = buffer;
        this.totalSize = buffer.byteLength;
        // Initial state: One giant free block covering the whole buffer
        this.freeList = [{ offset: 0, size: this.totalSize }];
    }

    allocate(size: number): number {
        // Align to 4 bytes (Float32) to ensure we don't have unaligned access issues
        // (size + 3) & ~3 rounds up to the nearest multiple of 4
        const alignedSize = (size + 3) & ~3;

        for (let i = 0; i < this.freeList.length; i++) {
            const block = this.freeList[i];
            
            // First Fit Strategy
            if (block.size >= alignedSize) {
                const offset = block.offset;

                if (block.size === alignedSize) {
                    // Exact fit: remove the block entirely
                    this.freeList.splice(i, 1);
                } else {
                    // Split the block: shrink the free space
                    block.offset += alignedSize;
                    block.size -= alignedSize;
                }

                return offset;
            }
        }

        throw new Error(`Out of memory: requested ${size} bytes (aligned to ${alignedSize}), but no block large enough found.`);
    }

    free(offset: number, size: number): void {
        const alignedSize = (size + 3) & ~3;
        
        // Find insertion point to keep list sorted by offset
        // This makes coalescing (merging) much easier
        let insertIndex = 0;
        while (insertIndex < this.freeList.length && this.freeList[insertIndex].offset < offset) {
            insertIndex++;
        }

        // Insert the new free block
        this.freeList.splice(insertIndex, 0, { offset, size: alignedSize });

        // Coalesce (merge) adjacent blocks to reduce fragmentation
        this.coalesce();
    }

    private coalesce(): void {
        for (let i = 0; i < this.freeList.length - 1; i++) {
            const current = this.freeList[i];
            const next = this.freeList[i + 1];

            // If the end of the current block touches the start of the next block
            if (current.offset + current.size === next.offset) {
                // Merge next into current
                current.size += next.size;
                // Remove next from the list
                this.freeList.splice(i + 1, 1);
                // Decrement i so we check this merged block against the *new* next block
                i--;
            }
        }
    }

    getStats() {
        const freeBytes = this.freeList.reduce((acc, block) => acc + block.size, 0);
        return {
            total: this.totalSize,
            used: this.totalSize - freeBytes,
            free: freeBytes,
            fragments: this.freeList.length
        };
    }
}

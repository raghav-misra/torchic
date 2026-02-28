# TypeScript Library Style Guide & Code Review Standards

## Core Philosophy

* **Optimize for reading, not writing.** Code is read 10x more than it is written.
* **The code is the source of truth.** If your code needs a paragraph to explain *what* it does, rewrite the code.
* **Embrace simplicity.** Clever code is a liability. Boring, predictable code is an asset.

## Comments & Documentation

We have a zero-tolerance policy for noise. With the rise of AI-assisted coding, redundant comments have become a plague.

* **Ban "LLM Fluff":** Never comment the obvious. If a comment simply restates the code in English, delete it.
* **The "Why", Not the "What":** Comments are strictly reserved for explaining *why* a decision was made, documenting weird edge cases, or explaining complex business logic/math.
* **JSDoc is for Public APIs Only:** Do not pollute internal functions, types, or utilities with massive JSDoc blocks. Use them exclusively for exports that consumers of the library will interact with.

**Bad:**

```typescript
// Fetches the user data
const data = await fetchUserData(); // await the promise

/**
 * Adds two numbers together.
 * @param a - The first number
 * @param b - The second number
 * @returns The sum of a and b
 */
function add(a: number, b: number): number {
  return a + b;
}

```

**Good:**

```typescript
const data = await fetchUserData();

// Using bitwise OR to truncate towards zero (faster than Math.trunc for small ints)
function fastTruncate(val: number): number {
  return val | 0;
}

```

## Control Flow & Function Architecture

Deeply nested code (the "Arrow Anti-Pattern") is where bugs hide and readability dies. We prioritize a flat execution path.

* **Guard Clauses & Early Returns:** Handle the garbage first. Check for invalid states, missing arguments, or error conditions at the very top of your function and `return` or `throw` immediately. The "happy path" should always be at the lowest possible indentation level.
* **Avoid Deep Nesting:** If your code is indented more than two or three levels deep (e.g., an `if` inside a `for` inside an `if`), it is fundamentally broken.
* **Extract Subfunctions:** The solution to deep nesting is extraction. Break complex, nested logic out into strictly scoped, well-named subfunctions. Let the parent function act as a high-level orchestrator.

**Bad:**

```typescript
function processWorkerData(data?: Float32Array, isReady?: boolean) {
  if (isReady) {
    if (data) {
      for (let i = 0; i < data.length; i++) {
        if (data[i] > 0) {
          data[i] = Math.sqrt(data[i]);
        }
      }
      return true;
    } else {
      throw new Error("Data missing");
    }
  }
  return false;
}

```

**Good:**

```typescript
function processWorkerData(data?: Float32Array, isReady?: boolean) {
  if (!isReady) return false;
  if (!data) throw new Error("Data missing");

  normalizePositiveValues(data);
  return true;
}

function normalizePositiveValues(data: Float32Array): void {
  for (let i = 0; i < data.length; i++) {
    if (data[i] > 0) {
      data[i] = Math.sqrt(data[i]);
    }
  }
}

```

## Error Handling

Stop wrapping everything in massive `try/catch` blocks. It breaks scoping, makes code harder to read, and often swallows errors that should crash the program.

* **Fail Fast, Fail Loud:** If an unrecoverable error happens, let it throw.
* **Targeted Catching:** Only `try/catch` around the exact asynchronous or risky operation that might fail, not the entire function body.

## Async & Await

Avoid the "await-waterfall". If independent async operations can be run concurrently, they must be.

* **No Awaits in Loops:** Never `await` inside a `map` or `forEach` loop unless you specifically need sequential execution (and if you do, leave a comment explaining *why*). Use `Promise.all` for concurrent execution.
* **Don't Wrap Returns:** Returning a promise from an `async` function implicitly wraps it in another promise. Just return the promise directly if you aren't doing anything else with it.

**Bad:**

```typescript
async function fetchAll(ids: string[]) {
  const results = [];
  // Blocks the thread on every iteration
  for (const id of ids) {
    results.push(await fetchItem(id));
  }
  return results;
}

```

**Good:**

```typescript
async function fetchAll(ids: string[]) {
  // Concurrent execution
  return Promise.all(ids.map(fetchItem));
}

```

## Parallelism: Web Workers & SharedArrayBuffers

Concurrency in JS is messy. Passing data between the main thread and workers can easily become a bottleneck or introduce race conditions.

* **Structured Messaging:** Do not send raw primitives or untyped objects via `postMessage`. Always use a discriminated union for message types to keep the boundaries strictly typed.
* **Zero-Copy is King:** When passing large datasets to workers, *always* use `SharedArrayBuffer` or transfer ownership (`Transferable` objects) to avoid structured cloning overhead.
* **Always use Atomics for SABs:** Never read/write to a `SharedArrayBuffer` using standard array indexing if another thread might be accessing it. Always use `Atomics.load`, `Atomics.store`, and `Atomics.wait`/`notify` to prevent data races.

**Bad:**

```typescript
// Worker thread
self.onmessage = (e) => {
  if (e.data.type === 'calculate') {
    // Standard assignment on a SAB is a race condition!
    e.data.sharedArray[0] = Math.random(); 
  }
};

```

**Good:**

```typescript
type WorkerMessage = 
  | { type: 'INIT'; buffer: SharedArrayBuffer }
  | { type: 'CALCULATE'; offset: number };

self.onmessage = (e: MessageEvent<WorkerMessage>) => {
  const msg = e.data;
  if (msg.type === 'CALCULATE') {
     // Thread-safe operation
     Atomics.store(sharedView, msg.offset, 1);
     Atomics.notify(sharedView, msg.offset, 1);
  }
};

```

## General TypeScript & Math

* **Trust Inference:** Stop explicitly typing every variable. If TS can infer it, let it. Only explicitly type function arguments, return types, and complex object boundaries.
* **Typed Arrays for Math:** When doing heavy number crunching, stick to typed arrays (`Float32Array`, `Float64Array`). JavaScript engines handle these much better than standard JS arrays, maximizing memory efficiency and CPU caching.

**Bad:**

```typescript
const isReady: boolean = false; 
const coordinates: number[] = [1.2, 3.4, 5.6]; 

```

**Good:**

```typescript
const isReady = false; 
const coordinates = new Float64Array([1.2, 3.4, 5.6]); 

```

## Linting & Type Errors

We expect strict adherence to the baseline configuration, not blind ignorance of it.

* **Read the Errors:** Do not simply ignore red squiggles. If the linter or TypeScript compiler yells at you, take the time to read the actual error message and understand *why* it is complaining.
* **Fix, Don't Suppress:** Using `// eslint-disable-next-line` or `// @ts-ignore` is a last resort, not a tool for convenience. If you must use a suppression comment, it **must** be accompanied by a brief explanation of why the rule is being bypassed. Usually, a lint error indicates a flaw in your control flow or typing that should be refactored, not swept under the rug.
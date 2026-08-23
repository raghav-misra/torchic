import type {
  BenchOptions,
  BenchSuite,
  FreeformOptions,
  FreeformSuite,
  Suite,
  TestOptions,
  TestSuite,
} from "./types";

const registry: Suite[] = [];

export function defineTest<TParam>(options: TestOptions<TParam>): void {
  registry.push({ kind: "test", ...options } as TestSuite);
}

export function defineBench<TParam>(options: BenchOptions<TParam>): void {
  registry.push({ kind: "bench", ...options } as BenchSuite);
}

export function defineFreeform(options: FreeformOptions): void {
  registry.push({ kind: "freeform", ...options } as FreeformSuite);
}

export function getRegistry(): readonly Suite[] {
  return registry;
}

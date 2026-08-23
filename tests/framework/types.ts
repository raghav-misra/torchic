export type Log = (msg: string) => void;

export interface RunContext {
  log: Log;
}

export interface TestResult { pass: boolean; message?: string }
export type BenchMetrics = Record<string, number | string>;

export interface TestOptions<TParam> {
  name: string;
  runner: (param: TParam, ctx: RunContext) => Promise<TestResult>;
  params: TParam[];
  paramName?: string;
  description?: string;
}

export interface BenchOptions<TParam> {
  name: string;
  runner: (param: TParam, ctx: RunContext) => Promise<BenchMetrics>;
  params: TParam[];
  paramName?: string;
  description?: string;
  highlight?: string[];
}

export interface TestSuite<TParam = unknown> extends TestOptions<TParam> {
  kind: "test";
}

export interface BenchSuite<TParam = unknown> extends BenchOptions<TParam> {
  kind: "bench";
}

export type Suite = TestSuite | BenchSuite;

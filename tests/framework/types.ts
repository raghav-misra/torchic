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

// A "freeform" suite is a list of named actions rendered as buttons. Each
// action can enable/disable other actions via its context, letting you model
// a small state machine (e.g. run inference -> play the resulting audio).
export interface FreeformContext {
  log: Log;
  enable: (id: string) => void;
  disable: (id: string) => void;
}

export interface FreeformAction {
  id: string;
  label: string;
  disabled?: boolean;
  run: (ctx: FreeformContext) => Promise<void>;
}

export interface FreeformOptions {
  name: string;
  description?: string;
  actions: FreeformAction[];
}

export interface FreeformSuite extends FreeformOptions {
  kind: "freeform";
}

export type Suite = TestSuite | BenchSuite | FreeformSuite;

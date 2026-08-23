import "./suites/wasm-parity";
import "./suites/webgpu-parity";
import "./suites/matmul-bench";
import "./suites/layernorm-gelu";
import "./suites/bmm";
import "./suites/attention";
import "./demos/makemore";
import { getRegistry } from "./framework/define";
import type { BenchSuite, Suite, TestSuite } from "./framework/types";

// Message envelope: any line starting with "__BENCH__ " on the browser console
// is a JSON payload consumed by the Node driver (scripts/bench.mjs).
type Envelope =
  | { kind: "meta"; suite: string; suiteKind: "test" | "bench"; params: string[] }
  | { kind: "log"; msg: string }
  | { kind: "test-result"; param: string; pass: boolean; message?: string }
  | { kind: "bench-result"; param: string; metrics: Record<string, string | number> }
  | { kind: "error"; param?: string; message: string }
  | { kind: "done" };

function emit(env: Envelope): void {
  console.log("__BENCH__ " + JSON.stringify(env));
  if (env.kind === "done") setStatus("done");
}

const statusEl = document.getElementById("status") as HTMLDivElement;
const logEl = document.getElementById("log") as HTMLPreElement;

function setStatus(s: string): void {
  statusEl.textContent = s;
}
function appendLog(m: string): void {
  logEl.textContent += m + "\n";
}

function findSuite(registry: readonly Suite[], query: string): Suite | undefined {
  const q = query.toLowerCase();
  return (
    registry.find((s) => s.name.toLowerCase() === q) ??
    registry.find((s) => s.name.toLowerCase().includes(q))
  );
}

function paramLabel(suite: Suite, param: unknown, index: number): string {
  const val =
    param === null || param === undefined
      ? `#${index}`
      : typeof param === "string" || typeof param === "number" || typeof param === "boolean"
        ? String(param)
        : `#${index}`;
  return suite.paramName ? `${suite.paramName}=${val}` : val;
}

async function runOne(suite: Suite, param: unknown, index: number): Promise<void> {
  const label = paramLabel(suite, param, index);
  setStatus(`running ${suite.name} — ${label}`);
  const ctx = {
    log: (m: string) => {
      appendLog(m);
      emit({ kind: "log", msg: m });
    },
  };

  if (suite.kind === "test") {
    try {
      const r = await (suite as TestSuite).runner(param, ctx);
      emit({ kind: "test-result", param: label, pass: r.pass, message: r.message });
    } catch (e) {
      emit({ kind: "error", param: label, message: String(e) });
    }
  } else {
    try {
      const metrics = await (suite as BenchSuite).runner(param, ctx);
      emit({ kind: "bench-result", param: label, metrics });
    } catch (e) {
      emit({ kind: "error", param: label, message: String(e) });
    }
  }
}

async function main(): Promise<void> {
  const params = new URLSearchParams(window.location.search);
  const suiteQuery = params.get("suite");
  const paramFilter = params.get("param");

  const registry = getRegistry();
  if (!suiteQuery) {
    emit({
      kind: "error",
      message:
        "missing ?suite=<substring>. available: " + registry.map((s) => s.name).join(" | "),
    });
    emit({ kind: "done" });
    return;
  }

  const suite = findSuite(registry, suiteQuery);
  if (!suite) {
    emit({
      kind: "error",
      message:
        `no suite matches '${suiteQuery}'. available: ` + registry.map((s) => s.name).join(" | "),
    });
    emit({ kind: "done" });
    return;
  }

  // Pick which params to run.
  let indices: number[];
  if (!paramFilter || paramFilter === "all") {
    indices = suite.params.map((_, i) => i);
  } else {
    indices = suite.params
      .map((p, i) => ({ p, i }))
      .filter(({ p, i }) => String(p) === paramFilter || paramLabel(suite, p, i) === paramFilter)
      .map(({ i }) => i);
    if (indices.length === 0) {
      const avail = suite.params.map((p, i) => paramLabel(suite, p, i)).join(", ");
      emit({
        kind: "error",
        message: `param '${paramFilter}' not in suite '${suite.name}'. available: ${avail}`,
      });
      emit({ kind: "done" });
      return;
    }
  }

  emit({
    kind: "meta",
    suite: suite.name,
    suiteKind: suite.kind,
    params: indices.map((i) => paramLabel(suite, suite.params[i], i)),
  });

  for (const i of indices) {
    await runOne(suite, suite.params[i], i);
  }
  setStatus("done");
  emit({ kind: "done" });
}

main().catch((e) => {
  emit({ kind: "error", message: `top-level: ${String(e)}` });
  emit({ kind: "done" });
});

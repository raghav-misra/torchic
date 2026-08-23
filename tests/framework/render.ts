import { getRegistry } from "./define";
import type { BenchMetrics, BenchSuite, Suite, TestResult, TestSuite } from "./types";

export function mount(root: HTMLElement): void {
  for (const suite of getRegistry()) {
    root.appendChild(renderSuite(suite));
  }
}

function renderSuite(suite: Suite): HTMLElement {
  const section = el("section", "suite");
  section.appendChild(el("h2", "", suite.name));
  if (suite.description) section.appendChild(el("p", "desc", suite.description));

  const [logEl, log] = makeLog();
  const result = el("div", "result");
  const bench = suite.kind === "bench" ? makeBenchRunner(suite, result) : null;
  const runner = bench ? bench.run : makeTestRunner(suite as TestSuite, result);

  const buttons = el("div", "buttons");
  const paramButtons: HTMLButtonElement[] = suite.params.map((param, i) => {
    const btn = document.createElement("button");
    btn.textContent = buttonLabel(suite, param, i);
    buttons.appendChild(btn);
    return btn;
  });

  const runAll = document.createElement("button");
  runAll.textContent = suite.params.length === 1 ? "Run" : "Run all";
  runAll.className = "primary";
  if (suite.params.length > 1) buttons.appendChild(runAll);

  const allButtons: HTMLButtonElement[] = [...paramButtons, runAll];

  if (bench) {
    const copyBtn = document.createElement("button");
    copyBtn.textContent = "Copy Markdown";
    copyBtn.className = "secondary";
    copyBtn.addEventListener("click", async () => {
      const md = bench.toMarkdown();
      const original = copyBtn.textContent;
      try {
        await navigator.clipboard.writeText(md);
        copyBtn.textContent = "Copied!";
      } catch {
        copyBtn.textContent = "Copy failed";
      }
      setTimeout(() => (copyBtn.textContent = original), 1200);
    });
    buttons.appendChild(copyBtn);
  }

  paramButtons.forEach((btn, i) => {
    btn.addEventListener("click", async () => {
      setDisabled(allButtons, true);
      try {
        await runner(suite.params[i], i, log);
      } finally {
        setDisabled(allButtons, false);
      }
    });
  });

  runAll.addEventListener("click", async () => {
    setDisabled(allButtons, true);
    try {
      for (let i = 0; i < suite.params.length; i++) {
        await runner(suite.params[i], i, log);
      }
    } finally {
      setDisabled(allButtons, false);
    }
  });

  section.appendChild(buttons);
  section.appendChild(result);
  section.appendChild(logEl);
  return section;
}

function makeTestRunner(suite: TestSuite, container: HTMLElement) {
  const badges: HTMLElement[] = [];
  return async (param: unknown, index: number, log: (m: string) => void): Promise<void> => {
    let badge = badges[index];
    if (!badge) {
      badge = el("div", "test-badge running");
      badges[index] = badge;
      container.appendChild(badge);
    }
    const label = paramValueLabel(suite, param, index);
    badge.className = "test-badge running";
    badge.textContent = `${label}: running…`;

    let result: TestResult;
    try {
      result = await suite.runner(param, { log });
    } catch (e) {
      result = { pass: false, message: String(e) };
    }

    badge.className = `test-badge ${result.pass ? "pass" : "fail"}`;
    badge.textContent = result.message
      ? `${label}: ${result.pass ? "PASS" : "FAIL"} (${result.message})`
      : `${label}: ${result.pass ? "PASS" : "FAIL"}`;
  };
}

function makeBenchRunner(suite: BenchSuite, container: HTMLElement) {
  const table = document.createElement("table");
  table.className = "bench-table";
  const thead = document.createElement("thead");
  const tbody = document.createElement("tbody");
  table.appendChild(thead);
  table.appendChild(tbody);
  container.appendChild(table);

  const firstCol = suite.paramName ?? "param";
  const columns: string[] = [firstCol];
  const highlight = new Set(suite.highlight ?? []);
  const rows: (HTMLTableRowElement | null)[] = [];
  const paramLabels: string[] = suite.params.map((p, i) => paramValueLabel(suite, p, i));
  const cellData: (BenchMetrics | null)[] = suite.params.map(() => null);

  const isHighlighted = (col: string) => highlight.has(col);

  const rebuildHeader = () => {
    thead.textContent = "";
    const tr = document.createElement("tr");
    for (const c of columns) tr.appendChild(el("th", isHighlighted(c) ? "highlight" : "", c));
    thead.appendChild(tr);
  };

  const upsertRow = (index: number, metrics: BenchMetrics | null): void => {
    let tr = rows[index];
    if (!tr) {
      tr = document.createElement("tr");
      rows[index] = tr;
      tbody.appendChild(tr);
    }
    tr.textContent = "";
    tr.classList.toggle("running", metrics === null);

    tr.appendChild(el("td", "", paramLabels[index]));
    for (let c = 1; c < columns.length; c++) {
      const key = columns[c];
      const v = metrics && key in metrics ? String(metrics[key]) : metrics ? "" : "…";
      tr.appendChild(el("td", isHighlighted(key) ? "highlight" : "", v));
    }
  };

  rebuildHeader();
  for (let i = 0; i < suite.params.length; i++) upsertRow(i, { [firstCol]: paramLabels[i] });

  const run = async (
    param: unknown,
    index: number,
    log: (m: string) => void,
  ): Promise<void> => {
    upsertRow(index, null);
    let metrics: BenchMetrics;
    try {
      metrics = await suite.runner(param, { log });
    } catch (e) {
      metrics = { error: String(e) };
    }
    cellData[index] = metrics;
    for (const k of Object.keys(metrics)) {
      if (!columns.includes(k) && k !== firstCol) columns.push(k);
    }
    rebuildHeader();
    for (let i = 0; i < suite.params.length; i++) {
      if (i === index) upsertRow(i, metrics);
      else if (rows[i]) {
        const existing = rows[i]!;
        const existingCells = existing.children.length;
        for (let c = existingCells; c < columns.length; c++) {
          existing.appendChild(el("td", isHighlighted(columns[c]) ? "highlight" : "", ""));
        }
      }
    }
  };

  const toMarkdown = (): string => {
    const escapeCell = (s: string) =>
      s.replace(/\|/g, "\\|").replace(/\r?\n/g, "<br>");
    const headerCells = columns.map((c) =>
      isHighlighted(c) ? `**${escapeCell(c)}**` : escapeCell(c),
    );
    const header = `| ${headerCells.join(" | ")} |`;
    const sep = `| ${columns.map(() => "---").join(" | ")} |`;

    const bodyRows: string[] = [];
    for (let i = 0; i < suite.params.length; i++) {
      const cells: string[] = [escapeCell(paramLabels[i])];
      const metrics = cellData[i];
      for (let c = 1; c < columns.length; c++) {
        const key = columns[c];
        const raw = metrics && key in metrics ? String(metrics[key]) : "";
        const cell = escapeCell(raw);
        cells.push(isHighlighted(key) && raw ? `**${cell}**` : cell);
      }
      bodyRows.push(`| ${cells.join(" | ")} |`);
    }

    return [`## ${suite.name}`, "", header, sep, ...bodyRows].join("\n");
  };

  return { run, toMarkdown };
}

// e.g. "threads=4" when paramName is set; "4" otherwise.
function buttonLabel(suite: Suite, p: unknown, i: number): string {
  if (suite.params.length === 1) return "Run";
  return paramValueLabel(suite, p, i);
}

function paramValueLabel(suite: Suite, p: unknown, i: number): string {
  const val =
    p == null
      ? `#${i}`
      : typeof p === "number" || typeof p === "string" || typeof p === "boolean"
        ? String(p)
        : `#${i}`;
  return suite.paramName ? `${suite.paramName}=${val}` : val;
}

function makeLog(): [HTMLElement, (msg: string) => void] {
  const details = document.createElement("details");
  details.className = "log-details";
  const summary = document.createElement("summary");
  summary.textContent = "log";

  const copyBtn = document.createElement("button");
  copyBtn.type = "button";
  copyBtn.className = "log-copy";
  copyBtn.textContent = "copy";
  summary.appendChild(copyBtn);

  const pre = document.createElement("pre");
  pre.className = "log";
  details.appendChild(summary);
  details.appendChild(pre);

  copyBtn.addEventListener("click", async (e) => {
    // Don't toggle the <details> when clicking the button.
    e.preventDefault();
    e.stopPropagation();
    const original = copyBtn.textContent;
    try {
      await navigator.clipboard.writeText(pre.textContent ?? "");
      copyBtn.textContent = "copied!";
    } catch {
      copyBtn.textContent = "copy failed";
    }
    setTimeout(() => (copyBtn.textContent = original), 1200);
  });

  const log = (msg: string) => {
    pre.textContent += msg + "\n";
    pre.scrollTop = pre.scrollHeight;
  };
  return [details, log];
}

function el(tag: string, className = "", text = ""): HTMLElement {
  const e = document.createElement(tag);
  if (className) e.className = className;
  if (text) e.textContent = text;
  return e;
}

function setDisabled(btns: HTMLButtonElement[], disabled: boolean): void {
  for (const b of btns) b.disabled = disabled;
}

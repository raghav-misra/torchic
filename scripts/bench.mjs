#!/usr/bin/env node
// Headless bench harness. Boots vite on a free port, launches puppeteer,
// navigates to tests/headless.html?suite=<...>&param=<...>, and pipes
// __BENCH__ envelope messages from the browser console to stdout.
//
// Usage:
//   node scripts/bench.mjs <suite> [param]           # preferred
//   node scripts/bench.mjs makemore wasm
//   node scripts/bench.mjs --list
//   node scripts/bench.mjs makemore --headed --timeout 300000
//   npm run bench -- makemore wasm                   # positionals survive npm
//   npm run bench -- --list                          # boolean flags via npm env

import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { setTimeout as sleep } from "node:timers/promises";

// npm 11 swallows --suite / --param / --list / --headed as its own config,
// dumping values into positional argv and setting npm_config_<flag>=true env
// vars for the booleans. So we accept positionals first and fall back to env
// for booleans.
function parseArgs(argv) {
  const out = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--") continue;
    if (a.startsWith("--")) {
      const eq = a.indexOf("=");
      if (eq !== -1) {
        out[a.slice(2, eq)] = a.slice(eq + 1);
        continue;
      }
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith("--")) {
        out[key] = next;
        i++;
      } else {
        out[key] = true;
      }
    } else {
      out._.push(a);
    }
  }

  const envBool = (k) => process.env[`npm_config_${k}`] === "true";
  if (!out.list && envBool("list")) out.list = true;
  if (!out.headed && envBool("headed")) out.headed = true;
  if (!out.json && envBool("json")) out.json = true;
  if (!out.help && envBool("help")) out.help = true;

  if (!out.suite && out._[0]) out.suite = out._[0];
  if (!out.param && out._[1]) out.param = out._[1];

  // If npm swallowed --suite/--param into positionals AND --list, the user
  // clearly meant to run a suite; --list only applies without a suite.
  if (out.list && out.suite) out.list = false;

  return out;
}

async function pickFreePort() {
  return await new Promise((resolve, reject) => {
    const srv = createServer();
    srv.unref();
    srv.on("error", reject);
    srv.listen(0, () => {
      const port = srv.address().port;
      srv.close(() => resolve(port));
    });
  });
}

async function waitForServer(url, timeoutMs) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const r = await fetch(url);
      if (r.ok) return;
    } catch {
      // not up yet
    }
    await sleep(200);
  }
  throw new Error(`Server did not start within ${timeoutMs}ms: ${url}`);
}

function startVite(port) {
  // Windows needs shell:true so `npx.cmd` resolves.
  const proc = spawn(
    "npx",
    ["vite", "--port", String(port), "--strictPort", "--clearScreen", "false"],
    { stdio: ["ignore", "pipe", "pipe"], shell: true },
  );
  proc.stdout.on("data", (d) => process.stderr.write(`[vite] ${d}`));
  proc.stderr.on("data", (d) => process.stderr.write(`[vite] ${d}`));
  proc.on("error", (e) => process.stderr.write(`[vite spawn error] ${e}\n`));
  return proc;
}

// child.kill() on a shell-spawned npm process leaks node.exe children on Windows.
function killTree(child) {
  if (!child || child.killed) return;
  if (process.platform === "win32" && child.pid) {
    try {
      spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], {
        stdio: "ignore",
        windowsHide: true,
      });
    } catch {
      child.kill();
    }
  } else {
    child.kill();
  }
}

function registerCleanup(child) {
  const cleanup = () => killTree(child);
  process.on("exit", cleanup);
  process.on("SIGINT", () => {
    cleanup();
    process.exit(130);
  });
  process.on("SIGTERM", () => {
    cleanup();
    process.exit(143);
  });
}

function fmtMetrics(m) {
  const entries = Object.entries(m);
  const keyPad = Math.max(...entries.map(([k]) => k.length));
  return entries.map(([k, v]) => `    ${k.padEnd(keyPad)}  ${v}`).join("\n");
}

function renderMarkdownTable(suite, rows) {
  if (rows.length === 0) return "";
  const cols = ["param"];
  for (const r of rows) {
    for (const k of Object.keys(r.metrics ?? {})) {
      if (!cols.includes(k)) cols.push(k);
    }
  }
  const head = `| ${cols.join(" | ")} |`;
  const sep = `| ${cols.map(() => "---").join(" | ")} |`;
  const body = rows.map((r) => {
    const cells = [r.param, ...cols.slice(1).map((c) => String(r.metrics?.[c] ?? ""))];
    return `| ${cells.join(" | ")} |`;
  });
  return [`## ${suite}`, "", head, sep, ...body].join("\n");
}

async function listSuites() {
  const port = await pickFreePort();
  const vite = startVite(port);
  registerCleanup(vite);
  try {
    await waitForServer(`http://localhost:${port}/headless.html`, 30000);
    const { default: puppeteer } = await import("puppeteer");
    const browser = await puppeteer.launch({
      headless: "shell",
      args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan"],
    });
    const page = await browser.newPage();
    let printed = false;
    page.on("console", (msg) => {
      const text = msg.text();
      if (!text.startsWith("__BENCH__ ")) return;
      const env = JSON.parse(text.slice("__BENCH__ ".length));
      if (env.kind === "error" && !printed) {
        printed = true;
        // The error text embeds the available list.
        const marker = "available: ";
        const idx = env.message.indexOf(marker);
        const list = idx >= 0 ? env.message.slice(idx + marker.length).split(" | ") : [];
        console.log("Registered suites:");
        for (const s of list) console.log("  - " + s);
      }
    });
    await page.goto(`http://localhost:${port}/headless.html`, { waitUntil: "load" });
    await page.waitForFunction(() => document.getElementById("status")?.textContent === "done", {
      timeout: 15000,
    });
    await browser.close();
  } finally {
    killTree(vite);
  }
}

async function runSuite(suiteQuery, paramFilter, opts) {
  const port = await pickFreePort();
  const vite = startVite(port);
  registerCleanup(vite);
  const timeoutMs = Number(opts.timeout ?? 600000);
  const url = new URL(`http://localhost:${port}/headless.html`);
  url.searchParams.set("suite", suiteQuery);
  if (paramFilter) url.searchParams.set("param", paramFilter);

  try {
    await waitForServer(`http://localhost:${port}/headless.html`, 30000);
    const { default: puppeteer } = await import("puppeteer");
    const browser = await puppeteer.launch({
      // "shell" is the stripped chrome-headless-shell (no WebGPU support).
      // `true` uses the full new-headless Chrome, which honors the WebGPU flags.
      headless: opts.headed ? false : true,
      args: [
        "--enable-unsafe-webgpu",
        "--enable-features=Vulkan",
        "--use-vulkan=swiftshader",
        "--enable-webgpu-developer-features",
      ],
    });
    const page = await browser.newPage();

    const state = {
      suiteName: null,
      suiteKind: null,
      benchRows: [],
      testRows: [],
      done: false,
      exitCode: 0,
    };

    page.on("pageerror", (err) => {
      process.stderr.write(`[pageerror] ${err.message}\n`);
      state.exitCode = 2;
    });

    page.on("console", (msg) => {
      const text = msg.text();
      if (!text.startsWith("__BENCH__ ")) return;
      let env;
      try {
        env = JSON.parse(text.slice("__BENCH__ ".length));
      } catch {
        return;
      }
      if (opts.json) {
        // Machine-readable path: one JSON line per envelope.
        process.stdout.write(JSON.stringify(env) + "\n");
      }
      switch (env.kind) {
        case "meta":
          state.suiteName = env.suite;
          state.suiteKind = env.suiteKind;
          if (!opts.json)
            process.stderr.write(
              `[meta] suite='${env.suite}' kind=${env.suiteKind} params=${env.params.join(",")}\n`,
            );
          break;
        case "log":
          if (!opts.json) process.stderr.write(`[log] ${env.msg}\n`);
          break;
        case "test-result":
          state.testRows.push(env);
          if (!opts.json)
            process.stderr.write(
              `[test] ${env.param}: ${env.pass ? "PASS" : "FAIL"}${env.message ? ` (${env.message})` : ""}\n`,
            );
          if (!env.pass) state.exitCode = 1;
          break;
        case "bench-result":
          state.benchRows.push(env);
          if (!opts.json)
            process.stderr.write(`[bench] ${env.param}:\n${fmtMetrics(env.metrics)}\n`);
          break;
        case "error":
          process.stderr.write(
            `[err]${env.param ? ` ${env.param}:` : ""} ${env.message}\n`,
          );
          state.exitCode = 1;
          break;
        case "done":
          state.done = true;
          break;
      }
    });

    await page.goto(url.toString(), { waitUntil: "load" });
    await page.waitForFunction(() => document.getElementById("status")?.textContent === "done", {
      timeout: timeoutMs,
    });
    await browser.close();

    if (!opts.json && state.suiteKind === "bench" && state.benchRows.length > 0) {
      process.stdout.write("\n" + renderMarkdownTable(state.suiteName, state.benchRows) + "\n");
    }
    process.exit(state.exitCode);
  } finally {
    killTree(vite);
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));

  if (args.help || args.h) {
    process.stdout.write(
      [
        "torchic bench harness",
        "",
        "Usage:",
        "  node scripts/bench.mjs <suite> [param] [--headed] [--json] [--timeout <ms>]",
        "  node scripts/bench.mjs --list",
        "  npm run bench -- <suite> [param]",
        "",
        "Positionals:",
        "  suite     Case-insensitive substring match against defineTest/defineBench name.",
        "  param     Specific param value (e.g. 'wasm', 'workers', '4'). Omit for all.",
        "",
        "Flags:",
        "  --list     Print registered suites and exit.",
        "  --headed   Launch a visible Chrome window instead of headless.",
        "  --json     Emit one JSON envelope per line on stdout, suppress human output.",
        "  --timeout  Maximum wall-clock ms for the whole run (default 600000).",
        "",
        "Note: npm 11+ swallows --suite/--param as its own config flags. Use positional",
        "args when invoking through npm, or run scripts/bench.mjs directly with node.",
        "",
      ].join("\n"),
    );
    return;
  }

  if (args.list) {
    await listSuites();
    return;
  }

  if (!args.suite || typeof args.suite !== "string") {
    process.stderr.write(
      "error: suite name is required. Try: node scripts/bench.mjs <suite> [param]\n" +
        "      or: node scripts/bench.mjs --list\n",
    );
    process.exit(2);
  }

  await runSuite(args.suite, typeof args.param === "string" ? args.param : null, {
    headed: !!args.headed,
    json: !!args.json,
    timeout: args.timeout,
  });
}

main().catch((e) => {
  process.stderr.write(`fatal: ${e?.stack ?? e}\n`);
  process.exit(2);
});

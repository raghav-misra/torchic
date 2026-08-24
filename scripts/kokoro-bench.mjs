#!/usr/bin/env node
// Headless Kokoro synthesis harness. Boots vite + puppeteer, uploads the
// model + voice safetensors via file chooser, runs the requested phoneme
// sample end-to-end, and reports RTF + peak amplitude + pass/fail.
//
// Purpose: regression check after perf changes (fused kernels, disposes,
// allocator tweaks). Same page as tests/kokoro-bench.html; puppeteer just
// automates the two file uploads and reads a __RESULT__ envelope.
//
// Usage:
//   node scripts/kokoro-bench.mjs --model <path> --voice <path>
//     [--sample pangram]        # any key from tests/demos/kokoro/samples.json
//     [--backend webgpu]        # webgpu | workers
//     [--memory 1536]           # heap MB
//     [--timeout 180000]        # ms
//     [--headed]                # show the browser window
//     [--json]                  # emit the result envelope on stdout
//
// Exit codes:
//   0  ok
//   1  bad args
//   2  page error / thrown exception
//   3  timeout
//   4  sanity check fail (NaN / all-zero / RTF regression)

import { spawn } from "node:child_process";
import { createServer } from "node:net";
import { existsSync } from "node:fs";
import { setTimeout as sleep } from "node:timers/promises";
import path from "node:path";

function parseArgs(argv) {
  const out = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) { out._.push(a); continue; }
    const key = a.slice(2);
    const next = argv[i + 1];
    if (next === undefined || next.startsWith("--")) {
      out[key] = true;
    } else {
      out[key] = next;
      i++;
    }
  }
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
    } catch {}
    await sleep(200);
  }
  throw new Error(`server not up at ${url} within ${timeoutMs}ms`);
}

function startVite(port) {
  const proc = spawn(
    "npx",
    ["vite", "--port", String(port), "--strictPort", "--clearScreen", "false"],
    { stdio: ["ignore", "pipe", "pipe"], shell: true },
  );
  proc.stdout.on("data", (d) => process.stderr.write(`[vite] ${d}`));
  proc.stderr.on("data", (d) => process.stderr.write(`[vite] ${d}`));
  return proc;
}

function killTree(child) {
  if (!child || child.killed) return;
  if (process.platform === "win32" && child.pid) {
    try { spawn("taskkill", ["/pid", String(child.pid), "/T", "/F"], { stdio: "ignore" }); }
    catch { child.kill(); }
  } else {
    child.kill();
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (!args.model || !args.voice) {
    console.error("--model <path> and --voice <path> are required.");
    console.error("run with --help for full usage.");
    process.exit(1);
  }
  const modelPath = path.resolve(String(args.model));
  const voicePath = path.resolve(String(args.voice));
  if (!existsSync(modelPath)) { console.error(`model not found: ${modelPath}`); process.exit(1); }
  if (!existsSync(voicePath)) { console.error(`voice not found: ${voicePath}`); process.exit(1); }

  const sample = args.sample ?? "pangram";
  const backend = args.backend ?? "webgpu";
  const memory = args.memory ?? "1536";
  const timeoutMs = Number(args.timeout ?? 180000);

  console.error(`[bench] model  = ${modelPath}`);
  console.error(`[bench] voice  = ${voicePath}`);
  console.error(`[bench] sample = ${sample}  backend = ${backend}  memory = ${memory}MB`);

  const port = await pickFreePort();
  const vite = startVite(port);
  const cleanup = () => killTree(vite);
  process.on("exit", cleanup);
  process.on("SIGINT", () => { cleanup(); process.exit(130); });
  process.on("SIGTERM", () => { cleanup(); process.exit(143); });

  await waitForServer(`http://localhost:${port}/kokoro-bench.html`, 30000);

  const { default: puppeteer } = await import("puppeteer");
  const browser = await puppeteer.launch({
    headless: args.headed ? false : true,
    args: [
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan",
      "--use-vulkan=swiftshader",
      "--enable-webgpu-developer-features",
    ],
  });
  const page = await browser.newPage();

  // Filechooser events don't fire reliably for JS-synthesized clicks in
  // headless Chromium. Upload directly to persistent <input> elements
  // instead — page reads the files via change events.
  let result = null;
  let error = null;
  page.on("pageerror", (err) => { error = err.message; });
  page.on("console", (msg) => {
    const text = msg.text();
    if (text.startsWith("__RESULT__")) {
      try { result = JSON.parse(text.slice("__RESULT__".length)); } catch (e) { error = String(e); }
    } else if (text.startsWith("__ERROR__")) {
      error = text.slice("__ERROR__".length);
    } else if (text.startsWith("[bench] ")) {
      process.stderr.write(text + "\n");
    } else if (msg.type() === "error" || msg.type() === "warning") {
      // Silent swallow patterns (allocator OOM, missing metadata) go through
      // console.error inside dispatcher code — surface those too.
      process.stderr.write(`[browser ${msg.type()}] ${text}\n`);
    }
  });

  const url = new URL(`http://localhost:${port}/kokoro-bench.html`);
  url.searchParams.set("sample", sample);
  url.searchParams.set("backend", backend);
  url.searchParams.set("memory", String(memory));
  if (args.verbose) url.searchParams.set("verbose", "1");
  await page.goto(url.toString(), { waitUntil: "load" });

  // Wait until the bench page requests each file (via status text), then
  // upload directly to the corresponding input element.
  await page.waitForFunction(() => document.getElementById("status")?.textContent === "await-model", { timeout: 30000 });
  const modelInput = await page.$("#model-input");
  if (!modelInput) throw new Error("#model-input not found");
  await modelInput.uploadFile(modelPath);

  await page.waitForFunction(() => document.getElementById("status")?.textContent === "await-voice", { timeout: 60000 });
  const voiceInput = await page.$("#voice-input");
  if (!voiceInput) throw new Error("#voice-input not found");
  await voiceInput.uploadFile(voicePath);

  const deadline = Date.now() + timeoutMs;
  while (!result && !error && Date.now() < deadline) {
    await sleep(200);
  }

  await browser.close();
  killTree(vite);

  if (error) { console.error(`[bench] FAIL: ${error}`); process.exit(2); }
  if (!result) { console.error(`[bench] TIMEOUT after ${timeoutMs}ms`); process.exit(3); }

  if (args.json) {
    console.log(JSON.stringify(result));
  } else {
    console.error("");
    console.error(`[bench] result:`);
    console.error(`  tokens        ${result.tokens}`);
    console.error(`  audio         ${result.audioSamples} samples (${result.audioSec.toFixed(2)}s)`);
    console.error(`  synth time    ${result.elapsedSec.toFixed(2)}s`);
    console.error(`  RTF           ${result.rtf.toFixed(3)}`);
    console.error(`  peak amp      ${result.peak.toExponential(3)}`);
    console.error(`  nans          ${result.nans}`);
    console.error(`  duration sum  ${result.durSum}`);
  }

  // Sanity: catches all-zero output (dispose bug), NaNs, and clear RTF regressions.
  const failures = [];
  if (result.nans > 0) failures.push(`${result.nans} NaN samples`);
  if (result.peak < 1e-3) failures.push(`peak too low (${result.peak.toExponential(2)}) -- silent output`);
  if (result.rtf > 5) failures.push(`RTF regressed (${result.rtf.toFixed(2)})`);

  if (failures.length > 0) {
    console.error(`[bench] FAIL: ${failures.join(", ")}`);
    process.exit(4);
  }
  console.error(`[bench] PASS`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});

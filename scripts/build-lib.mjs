import { copyFile, mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve, extname } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..");
const srcRoot = join(repoRoot, "src");
const distRoot = join(repoRoot, "dist");

async function copyAsset(relPath) {
  const src = join(srcRoot, relPath);
  const dst = join(distRoot, relPath);
  await mkdir(dirname(dst), { recursive: true });
  await copyFile(src, dst);
  console.log(`asset ${relPath}`);
}

async function copyDirAssets(relDir, ext) {
  const absDir = join(srcRoot, relDir);
  const entries = await readdir(absDir);
  for (const name of entries) {
    if (extname(name) === ext) {
      await copyAsset(join(relDir, name));
    }
  }
}

// Vite-style `new Worker(new URL("./worker.ts", import.meta.url), ...)` survives
// tsc unchanged, but the shipped file is worker.js. Rewrite in-place.
async function rewriteWorkerUrls(dir) {
  const entries = await readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    const full = join(dir, entry.name);
    if (entry.isDirectory()) {
      await rewriteWorkerUrls(full);
      continue;
    }
    if (!/\.(js|d\.ts)$/.test(entry.name)) continue;
    const before = await readFile(full, "utf8");
    const after = before.replace(
      /new URL\(\s*(['"])(\.[^'"]*?)\.ts\1\s*,\s*import\.meta\.url\s*\)/g,
      (_m, q, path) => `new URL(${q}${path}.js${q}, import.meta.url)`,
    );
    if (after !== before) {
      await writeFile(full, after, "utf8");
      console.log(`rewrote worker URL in ${full.substring(repoRoot.length + 1)}`);
    }
  }
}

await copyAsset(join("backend", "wasm", "kernels.wasm"));
await copyDirAssets(join("backend", "webgpu", "shaders"), ".wgsl");
await rewriteWorkerUrls(distRoot);

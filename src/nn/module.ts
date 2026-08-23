import { Tensor, noGrad, GradMode } from "../frontend/tensor";
import type { SafetensorsMap, SafetensorsEntry } from "./safetensors";

export type StateDict = Record<string, Tensor>;

export abstract class Module {
  private _params = new Map<string, Tensor>();
  private _buffers = new Map<string, Tensor>();
  private _children = new Map<string, Module | Module[]>();
  training = true;

  protected param(name: string, tensor: Tensor): Tensor {
    if (!tensor.requiresGrad) {
      throw new Error(`Module parameter '${name}' must have requires_grad=true`);
    }
    if (this._params.has(name)) {
      throw new Error(`Module already has parameter '${name}'`);
    }
    this._params.set(name, tensor);
    return tensor;
  }

  // Ordered list of leaf parameters — state dict keys become `${name}.0`, `${name}.1`, ...
  // matching PyTorch's nn.ParameterList convention.
  protected paramList(name: string, tensors: Tensor[]): Tensor[] {
    tensors.forEach((t, i) => {
      const key = `${name}.${i}`;
      if (!t.requiresGrad) {
        throw new Error(`Module parameter '${key}' must have requires_grad=true`);
      }
      if (this._params.has(key)) {
        throw new Error(`Module already has parameter '${key}'`);
      }
      this._params.set(key, t);
    });
    return tensors;
  }

  // Non-trainable state (running stats, positional caches). Included in state_dict.
  protected buffer(name: string, tensor: Tensor): Tensor {
    if (this._buffers.has(name)) {
      throw new Error(`Module already has buffer '${name}'`);
    }
    this._buffers.set(name, tensor);
    return tensor;
  }

  protected child<M extends Module>(name: string, module: M): M {
    if (this._children.has(name)) {
      throw new Error(`Module already has child '${name}'`);
    }
    this._children.set(name, module);
    return module;
  }

  // Ordered list — state dict keys become `${name}.0`, `${name}.1`, ...
  protected childList<M extends Module>(name: string, modules: M[]): M[] {
    if (this._children.has(name)) {
      throw new Error(`Module already has child list '${name}'`);
    }
    this._children.set(name, modules);
    return modules;
  }

  parameters(): Tensor[] {
    const out: Tensor[] = [...this._params.values()];
    for (const c of this._children.values()) {
      if (Array.isArray(c)) for (const m of c) out.push(...m.parameters());
      else out.push(...c.parameters());
    }
    return out;
  }

  state_dict(prefix = ""): StateDict {
    const sd: StateDict = {};
    for (const [name, p] of this._params) sd[prefix + name] = p;
    for (const [name, b] of this._buffers) sd[prefix + name] = b;
    for (const [name, c] of this._children) {
      if (Array.isArray(c)) {
        c.forEach((m, i) => Object.assign(sd, m.state_dict(`${prefix}${name}.${i}.`)));
      } else {
        Object.assign(sd, c.state_dict(`${prefix}${name}.`));
      }
    }
    return sd;
  }

  async load_state_dict(sd: StateDict, opts: { strict?: boolean } = {}): Promise<void> {
    const strict = opts.strict ?? true;
    const own = this.state_dict();
    const missing: string[] = [];
    const unexpected: string[] = [];

    await noGrad(async () => {
      for (const key of Object.keys(own)) {
        const src = sd[key];
        if (!src) {
          missing.push(key);
          continue;
        }
        const dst = own[key];
        if (!shapeEquals(src.shape, dst.shape)) {
          throw new Error(
            `load_state_dict shape mismatch at '${key}': dst=${dst.shape} src=${src.shape}`,
          );
        }
        dst.write(await src.toArray());
      }
      for (const key of Object.keys(sd)) if (!(key in own)) unexpected.push(key);
    });

    if (strict && (missing.length || unexpected.length)) {
      const parts: string[] = [];
      if (missing.length) parts.push(`missing (${missing.length}): ${missing.slice(0, 5).join(", ")}`);
      if (unexpected.length) parts.push(`unexpected (${unexpected.length}): ${unexpected.slice(0, 5).join(", ")}`);
      throw new Error(`load_state_dict: ${parts.join("; ")}`);
    }
  }

  // Writes Float32Array data straight into destination tensors, skipping the
  // intermediate Tensor allocation that load_state_dict does per parameter.
  //
  // Also fuses torch.nn.utils.weight_norm on the fly: a `weight_g` + `weight_v`
  // pair in the checkpoint gets combined into `weight = g * v / ||v||` before
  // being written into the destination tensor. Layers that use weight_norm at
  // training time can then run with a plain `weight` tensor at inference.
  load_safetensors(sd: SafetensorsMap, opts: { strict?: boolean; renameMap?: Record<string, string> } = {}): void {
    const strict = opts.strict ?? true;
    const rename = opts.renameMap;
    const own = this.state_dict();
    const missing: string[] = [];
    const unexpected: string[] = [];

    const sdKeys = new Set(Object.keys(sd));
    for (const key of Object.keys(own)) {
      const srcKey = rename?.[key] ?? key;
      let src = sd[srcKey];
      let consumedKeys: string[] = [srcKey];

      if (!src && srcKey.endsWith(".weight")) {
        const prefix = srcKey.slice(0, -".weight".length);
        const wg = sd[prefix + ".weight_g"];
        const wv = sd[prefix + ".weight_v"];
        if (wg && wv) {
          src = { shape: wv.shape.slice(), data: fuseWeightNorm(wg, wv) };
          consumedKeys = [prefix + ".weight_g", prefix + ".weight_v"];
        }
      }

      if (!src) {
        missing.push(key);
        continue;
      }
      const dst = own[key];
      if (!shapeEquals(src.shape, dst.shape)) {
        throw new Error(
          `load_safetensors shape mismatch at '${key}' (source '${srcKey}'): dst=${dst.shape} src=${src.shape}`,
        );
      }
      dst.write(src.data);
      for (const k of consumedKeys) sdKeys.delete(k);
    }
    for (const key of sdKeys) unexpected.push(key);

    if (strict && (missing.length || unexpected.length)) {
      const parts: string[] = [];
      if (missing.length) parts.push(`missing (${missing.length}): ${missing.slice(0, 5).join(", ")}`);
      if (unexpected.length) parts.push(`unexpected (${unexpected.length}): ${unexpected.slice(0, 5).join(", ")}`);
      throw new Error(`load_safetensors: ${parts.join("; ")}`);
    }
  }

  eval(): this {
    this.setMode(false);
    GradMode.enabled = false;
    return this;
  }

  train(): this {
    this.setMode(true);
    GradMode.enabled = true;
    return this;
  }

  private setMode(training: boolean): void {
    this.training = training;
    for (const c of this._children.values()) {
      if (Array.isArray(c)) c.forEach((m) => m.setMode(training));
      else c.setMode(training);
    }
  }
}

function shapeEquals(a: number[], b: number[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// Reconstructs a weight tensor from a torch.nn.utils.weight_norm (g, v) pair.
// g is per-output-channel magnitude (shape [C_out, 1, ...]); v holds the
// unnormalized weight. Output has v's shape and equals g * v / ||v||_row.
function fuseWeightNorm(wg: SafetensorsEntry, wv: SafetensorsEntry): Float32Array {
  const shape = wv.shape;
  if (shape.length === 0) throw new Error("weight_norm fuse: weight_v has rank 0");
  const Cout = shape[0];
  const perOut = wv.data.length / Cout;
  const out = new Float32Array(wv.data.length);
  for (let o = 0; o < Cout; o++) {
    let sq = 0;
    const base = o * perOut;
    for (let i = 0; i < perOut; i++) {
      const v = wv.data[base + i];
      sq += v * v;
    }
    const norm = Math.sqrt(sq);
    const g = wg.data[o];
    const scale = norm > 0 ? g / norm : 0;
    for (let i = 0; i < perOut; i++) out[base + i] = wv.data[base + i] * scale;
  }
  return out;
}

import binaryWgsl from "./shaders/binary.wgsl?raw";
import unaryWgsl from "./shaders/unary.wgsl?raw";
import fillWgsl from "./shaders/fill.wgsl?raw";
import matmulWgsl from "./shaders/matmul.wgsl?raw";
import softmaxWgsl from "./shaders/softmax.wgsl?raw";
import transposeWgsl from "./shaders/transpose.wgsl?raw";
import materializeWgsl from "./shaders/materialize.wgsl?raw";
import reductionsWgsl from "./shaders/reductions.wgsl?raw";
import randnWgsl from "./shaders/randn.wgsl?raw";
import embeddingWgsl from "./shaders/embedding.wgsl?raw";

const MODULE_SOURCE: Record<string, string> = {
  binary: binaryWgsl,
  unary: unaryWgsl,
  fill: fillWgsl,
  matmul: matmulWgsl,
  softmax: softmaxWgsl,
  transpose: transposeWgsl,
  materialize: materializeWgsl,
  reductions: reductionsWgsl,
  randn: randnWgsl,
  embedding: embeddingWgsl,
};

// Op → (shader module, entry point). Entry points in WGSL can't shadow certain
// keywords (`exp`, `log`, `tanh`, `copy` are builtin), so unary variants get an
// underscore suffix that the dispatcher accounts for.
const OP_TO_ENTRY: Record<string, { module: string; entry: string }> = {
  MATMUL: { module: "matmul", entry: "main" },

  ADD: { module: "binary", entry: "add" },
  SUB: { module: "binary", entry: "sub" },
  MUL: { module: "binary", entry: "mul" },
  DIV: { module: "binary", entry: "div" },
  RELU_BACKWARD: { module: "binary", entry: "relu_backward" },
  TANH_BACKWARD: { module: "binary", entry: "tanh_backward" },
  GELU_BACKWARD: { module: "binary", entry: "gelu_backward" },
  SQRT_BACKWARD: { module: "binary", entry: "sqrt_backward" },
  RSQRT_BACKWARD: { module: "binary", entry: "rsqrt_backward" },
  ADD_SCALAR_TENSOR: { module: "binary", entry: "add_scalar_tensor" },

  NEG: { module: "unary", entry: "neg" },
  RELU: { module: "unary", entry: "relu" },
  EXP: { module: "unary", entry: "exp_" },
  LOG: { module: "unary", entry: "log_" },
  TANH: { module: "unary", entry: "tanh_" },
  GELU: { module: "unary", entry: "gelu" },
  SQRT: { module: "unary", entry: "sqrt_" },
  RSQRT: { module: "unary", entry: "rsqrt_" },
  COPY: { module: "unary", entry: "copy_" },

  FILL: { module: "fill", entry: "main" },
  SOFTMAX: { module: "softmax", entry: "softmax" },
  SOFTMAX_BACKWARD: { module: "softmax", entry: "softmax_backward" },
  TRANSPOSE: { module: "transpose", entry: "main" },
  MATERIALIZE: { module: "materialize", entry: "main" },
  SUM_PARTIAL: { module: "reductions", entry: "sum_partial" },
  SUM_FINAL: { module: "reductions", entry: "sum_final" },
  SUM_AXIS: { module: "reductions", entry: "sum_axis" },
  RANDN: { module: "randn", entry: "main" },
  EMBEDDING: { module: "embedding", entry: "embedding" },
  EMBEDDING_BACKWARD: { module: "embedding", entry: "embedding_backward" },
};

export interface Pipelines {
  bindGroupLayout: GPUBindGroupLayout;
  byOp: Map<string, GPUComputePipeline>;
}

export async function buildPipelines(device: GPUDevice): Promise<Pipelines> {
  const bindGroupLayout = device.createBindGroupLayout({
    entries: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "storage" },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: "uniform" },
      },
    ],
  });
  const layout = device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] });

  const modules = new Map<string, GPUShaderModule>();
  for (const [name, code] of Object.entries(MODULE_SOURCE)) {
    modules.set(name, device.createShaderModule({ code, label: `${name}.wgsl` }));
  }

  const byOp = new Map<string, GPUComputePipeline>();
  const compiles = Object.entries(OP_TO_ENTRY).map(async ([op, spec]) => {
    const shaderModule = modules.get(spec.module);
    if (!shaderModule) throw new Error(`shader module '${spec.module}' not registered for op '${op}'`);
    const pipeline = await device.createComputePipelineAsync({
      layout,
      compute: { module: shaderModule, entryPoint: spec.entry },
      label: op,
    });
    byOp.set(op, pipeline);
  });
  await Promise.all(compiles);
  return { bindGroupLayout, byOp };
}

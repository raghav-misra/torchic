// Binary elementwise ops: 3 pointers + start/end range.
// Also housing the ops whose uniform shape happens to match (backward ops,
// add_scalar_tensor). Each entry point interprets p0/p1/p2 for its needs.

struct BinaryU {
  p0: u32,
  p1: u32,
  p2: u32,
  start: u32,
  end: u32,
  param0: f32, // op-specific scalar (e.g. LeakyReLU negative_slope)
}

@group(0) @binding(0) var<storage, read_write> heap: array<f32>;
@group(0) @binding(1) var<uniform> u: BinaryU;

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  heap[u.p2 + i] = heap[u.p0 + i] + heap[u.p1 + i];
}

@compute @workgroup_size(256)
fn sub(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  heap[u.p2 + i] = heap[u.p0 + i] - heap[u.p1 + i];
}

@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  heap[u.p2 + i] = heap[u.p0 + i] * heap[u.p1 + i];
}

@compute @workgroup_size(256)
fn div(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  heap[u.p2 + i] = heap[u.p0 + i] / heap[u.p1 + i];
}

// relu_backward: p0=input, p1=grad_output, p2=grad_input.
// grad_input[i] = input[i] > 0 ? grad_output[i] : 0
@compute @workgroup_size(256)
fn relu_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  let x = heap[u.p0 + i];
  heap[u.p2 + i] = select(0.0, heap[u.p1 + i], x > 0.0);
}

// tanh_backward: p0=output, p1=grad_output, p2=grad_input.
// grad_input[i] = grad_output[i] * (1 - output[i]^2)
@compute @workgroup_size(256)
fn tanh_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  let o = heap[u.p0 + i];
  heap[u.p2 + i] = heap[u.p1 + i] * (1.0 - o * o);
}

// gelu_backward: p0=input, p1=grad_output, p2=grad_input.
@compute @workgroup_size(256)
fn gelu_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  let x = heap[u.p0 + i];
  let x2 = x * x;
  let u_ = 0.7978845608028654 * (x + 0.044715 * x * x2);
  let t = tanh(u_);
  let dudx = 0.7978845608028654 * (1.0 + 3.0 * 0.044715 * x2);
  let dgelu = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dudx;
  heap[u.p2 + i] = heap[u.p1 + i] * dgelu;
}

// sqrt_backward: p0=output, p1=grad_output, p2=grad_input.
// d/dx sqrt(x) = 0.5 / sqrt(x) = 0.5 / y
@compute @workgroup_size(256)
fn sqrt_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  heap[u.p2 + i] = heap[u.p1 + i] * 0.5 / heap[u.p0 + i];
}

// rsqrt_backward: p0=output, p1=grad_output, p2=grad_input.
// d/dx x^(-1/2) = -0.5 * y^3
@compute @workgroup_size(256)
fn rsqrt_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  let y = heap[u.p0 + i];
  heap[u.p2 + i] = heap[u.p1 + i] * -0.5 * y * y * y;
}

// sigmoid_backward: p0=output, p1=grad_output, p2=grad_input.
// d/dx sigmoid(x) = y * (1 - y)
@compute @workgroup_size(256)
fn sigmoid_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  let y = heap[u.p0 + i];
  heap[u.p2 + i] = heap[u.p1 + i] * y * (1.0 - y);
}

// leaky_relu_backward: p0=input, p1=grad_output, p2=grad_input, param0=slope.
@compute @workgroup_size(256)
fn leaky_relu_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  let x = heap[u.p0 + i];
  let s = select(u.param0, 1.0, x >= 0.0);
  heap[u.p2 + i] = heap[u.p1 + i] * s;
}

// silu_backward: p0=input, p1=grad_output, p2=grad_input.
// d/dx (x*sigmoid(x)) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
@compute @workgroup_size(256)
fn silu_backward(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  let x = heap[u.p0 + i];
  let s = 1.0 / (1.0 + exp(-x));
  heap[u.p2 + i] = heap[u.p1 + i] * s * (1.0 + x * (1.0 - s));
}

// add_scalar_tensor: p0=a (elementwise), p1=scalar (length 1), p2=out.
@compute @workgroup_size(256)
fn add_scalar_tensor(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = u.start + gid.x;
  if (i >= u.end) { return; }
  heap[u.p2 + i] = heap[u.p0 + i] + heap[u.p1];
}

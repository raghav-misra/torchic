export interface WebGPUContext {
  device: GPUDevice;
  queue: GPUQueue;
}

export async function requestContext(): Promise<WebGPUContext> {
  if (!("gpu" in navigator) || !navigator.gpu) {
    throw new Error(
      "WebGPU not available in this browser. Requires Chrome/Edge 113+ or Safari 18+.",
    );
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("No suitable GPU adapter found.");
  }

  const info = adapter.info;
  console.log(
    `%cWebGPU adapter%c ${info.vendor || "?"} · ${info.device || info.architecture || "?"} · ${info.description || ""}`,
    "color: #63b3ed; font-weight: bold",
    "color: inherit",
  );

  // Default maxBufferSize is 256 MiB; ask for whatever the adapter actually
  // supports so we can hold larger tensor heaps.
  const requiredLimits: Record<string, number> = {
    maxBufferSize: adapter.limits.maxBufferSize,
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
  };
  const device = await adapter.requestDevice({ requiredLimits });

  device.lost.then((info) => {
    // device.destroy() fires this; not an error path.
    if (info.reason === "destroyed") return;
    console.error("WebGPU device lost:", info.reason, info.message);
  });

  return { device, queue: device.queue };
}

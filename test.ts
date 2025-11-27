import { Tensor } from "./src/index";

async function run() {
  console.log("Initializing...");
  await Tensor.init(4);
  console.log("Initialized.");

  console.log("Creating tensors...");
  const a = Tensor.randn([100]);
  const b = Tensor.randn([100]);

  console.log("a", (await a.toArray()).slice(0, 10));
  console.log("b", (await b.toArray()).slice(0, 10));

  console.log("Computing c = a + b...");
  const c = a.add(b);

  console.log("Fetching result...");
  const result = await c.toArray();

  console.log("Result (first 10):", result.slice(0, 10));
  console.log("Done.");
}

run().catch(console.error);

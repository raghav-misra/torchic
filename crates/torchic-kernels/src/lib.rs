#![no_std]
#![allow(clippy::missing_safety_doc)]

mod elementwise;
mod embedding;
mod matmul;
mod reductions;
mod rms_norm;
mod rope;
mod softmax;
mod causal_softmax;
mod copy_range;
mod repeat_interleave;
mod transpose;
mod conv;
mod concat;
mod lstm;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // wasm32 has no OS to report to; loop is the standard bare-metal move
    loop {}
}

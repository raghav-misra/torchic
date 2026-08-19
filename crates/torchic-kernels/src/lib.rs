#![no_std]
#![allow(clippy::missing_safety_doc)]

mod elementwise;
mod matmul;

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo) -> ! {
    // wasm32 has no OS to report to; loop is the standard bare-metal move
    loop {}
}

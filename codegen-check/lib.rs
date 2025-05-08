//! Codegen checks for the `glamour` crate.
//!
//! This is intended to be run manually when checking that glamour types
//! generate specific assembly output.
//!
//! 1. Install `cargo-show-asm` (`cargo install cargo-show-asm`).
//! 2. Run `cargo show-asm --release --target-cpu=native --target codegen-check
//!    <symbol>`, where `<symbol>` is one of the functions below.
//!
//! Note that the `#[inline(never)]` is required to force the compiler to
//! generate the symbol.

#![cfg_attr(coverage, feature(coverage_attribute))]
#![cfg_attr(coverage, coverage(off))]

use glamour::Vector4;

#[inline(never)]
pub fn sum_f32x4(v: &[Vector4<f32>]) -> Vector4<f32> {
    // This should generate a tight SIMD loop using unaligned loads or `vaddps`
    // when the CPU supports it.
    v.iter().copied().sum()
}

#[inline(never)]
pub fn sum_u8x4(v: &[Vector4<u8>]) -> Vector4<u8> {
    v.iter().copied().sum()
}

//! Marker traits for shorter trait bounds.
//!
//! These traits add no functionality, but vastly simplify trait bounds in other
//! places.

use bytemuck::Pod;

use core::fmt::Debug;

/// Convenience marker trait for any POD type that has value semantics.
pub trait PodValue:
    Copy + Debug + Default + PartialEq + Pod + Send + Sync + Sized + Serializable + 'static
{
}
impl<T> PodValue for T where
    T: Copy + Debug + Default + PartialEq + Pod + Send + Sync + Serializable + 'static
{
}

/// When compiled with the `serde` feature, this trait marks every type that
/// implements `Serialize` and `Deserialize`. When `serde` is disabled, it is an
/// empty trait with no bounds.
#[cfg(not(feature = "serde"))]
pub trait Serializable {}

/// When compiled with the `serde` feature, this trait marks every type that
/// implements `Serialize` and `Deserialize`. When `serde` is disabled, it is an
/// empty trait with no bounds.
#[cfg(feature = "serde")]
pub trait Serializable: serde::Serialize + for<'de> serde::Deserialize<'de> {}

#[cfg(feature = "serde")]
const _: () = {
    impl<T> Serializable for T where T: serde::Serialize + for<'de> serde::Deserialize<'de> {}
};

#[cfg(not(feature = "serde"))]
const _: () = {
    impl<T> Serializable for T {}
};

/// When compiled with the `wasmtime` feature, this trait marks every type that
/// implements `wasmtime::component::ComponentType`. When `wasmtime` is
/// disabled, it is an empty trait with no bounds.
#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime"))]
pub trait WasmComponentType:
    wasmtime::component::ComponentType + wasmtime::component::Lower + wasmtime::component::Lift
{
}

/// When compiled with the `wasmtime` feature, this trait marks every type that
/// implements `wasmtime::component::ComponentType`. When `wasmtime` is
/// disabled, it is an empty trait with no bounds.
#[cfg(any(target_arch = "wasm32", not(feature = "wasmtime")))]
pub trait WasmComponentType {}

#[cfg(all(not(target_arch = "wasm32"), feature = "wasmtime"))]
impl<T> WasmComponentType for T where
    T: wasmtime::component::ComponentType + wasmtime::component::Lower + wasmtime::component::Lift
{
}
#[cfg(any(target_arch = "wasm32", not(feature = "wasmtime")))]
impl<T> WasmComponentType for T {}

#[cfg(feature = "facet")]
#[doc(no_inline)]
pub use facet::Facet;

/// Marker trait for when `#[cfg(feature = "facet")]` is not enabled.
#[cfg(not(feature = "facet"))]
pub trait Facet<'a> {}
#[cfg(not(feature = "facet"))]
impl<'a, T: 'a> Facet<'a> for T {}

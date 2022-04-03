//! Marker traits for shorter trait bounds.
//!
//! These traits add no functionality, but vastly simplify trait bounds in other
//! places.

use bytemuck::Pod;

use core::fmt::Debug;

/// Convenience marker trait for any type that has value semantics.
pub trait ValueSemantics:
    Copy + Debug + Default + PartialEq + Pod + Send + Sync + Sized + 'static
{
}
impl<T> ValueSemantics for T where
    T: Copy + Debug + Default + PartialEq + Pod + Send + Sync + 'static
{
}

/// When compiled with the `serde` feature, this trait marks every type that
/// implements `Serialize` and `Deserialize`. When `serde` is disabled, it is an
/// empty trait with no bounds.
#[cfg(feature = "serde")]
pub trait Serializable: serde::Serialize + for<'de> serde::Deserialize<'de> {}

#[cfg(feature = "serde")]
const _: () = {
    impl<T> Serializable for T where T: serde::Serialize + for<'de> serde::Deserialize<'de> {}
};

/// When compiled with the `serde` feature, this trait marks every type that
/// implements `Serialize` and `Deserialize`. When `serde` is disabled, it is an
/// empty trait with no bounds.
#[cfg(not(feature = "serde"))]
pub trait Serializable {}
#[cfg(not(feature = "serde"))]
const _: () = {
    impl<T> Serializable for T {}
};

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

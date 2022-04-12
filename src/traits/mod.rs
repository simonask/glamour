//! Vector traits for binding typed abstractions to implementations from `glam`.
//!
//! **CAUTION: Here be dragons.**
//!
//! This rather nightmarish collection of traits describes the mapping from
//! [`Unit`]s to `glam` vector/matrix types, as well as marker traits and
//! shorthand traits that exist in order to simplify trait bounds.
//!
//! Many traits must be conditionally available on various linear algebra types
//! dependent on the raw representation for that type. For example, it doesn't
//! make sense to ask if a vector of ints is `NaN`, but it *does* make sense to
//! ask if a vector of custom scalar (e.g. [`Angle<f32>`](crate::Angle))
//! contains `NaN`s.
//!
//! The most important trait is [`Unit`], which determines how user-defined
//! units map to scalars.
//!
//! The second-most important trait is [`Scalar`], which determines how scalar
//! values are mapped to primitive scalars (i.e., `f32`, `f64`, `i32`, or
//! `u32`), and then in turn which `glam` vector and matrix types are used for
//! that scalar.
//!
//! The trait [`Primitive`] determines which `glam` vector type should be used
//! for each primitive scalar. To support user-defined scalars based on
//! primitives, the [`ScalarVectors`] trait exists as a blanket impl that always
//! maps to the corresponding vector type in [`Primitive`], based on the
//! scalar's [`Scalar::Primitive`] type.
//!
//! The trait [`SimdVec`] is implemented for each `glam` vector type,
//! generalizing methods that all `glam` vector types share. Additionally,
//! floating point vector types implement [`SimdVecFloat`].
//!
//! The trait [`SimdMatrix`] is implemented for each `glam` matrix type. Note
//! that `glam` only supports floating point matrices.

pub mod marker;
mod scalar;
mod simd_matrix;
mod simd_quat;
mod simd_vec;
mod unit;

pub use scalar::*;
pub use simd_matrix::*;
pub use simd_quat::*;
pub use simd_vec::*;
pub use unit::*;

/// General trait for the `contains()` method.
///
/// Coordinates exactly on the upper bound are considered to be contained.
///
/// See also: [`Intersection`], which is different in this regard.
pub trait Contains<T> {
    /// Returns `true` if `thing` is inside `self`.
    fn contains(&self, thing: &T) -> bool;
}

/// The `intersection()` method.
///
/// Coordinates exactly on the upper bound are _not_ considered to be
/// intersecting.
///
/// For example, each adjacent tile of a grid of [`crate::Rect`]s or
/// [`crate::Box2`]s will not be considered intersecting, while
/// [`Contains::contains()`] returns true for both tiles with coordinates that
/// fall exactly on the upper bound of one tile.
///
/// See also: [`Contains`], which is different in this regard.
pub trait Intersection<T = Self> {
    /// The type of intersection.
    ///
    /// For example, a rect/rect intersection is another rect, while a box/box
    /// intersection is a box, and a rect/box intersection is a rect, but a
    /// box/rect intersection is a box.
    type Intersection;

    /// True if `thing` intersects with `self`.
    fn intersects(&self, thing: &T) -> bool;

    /// If `thing` intersects with `self`, return the intersection. Otherwise,
    /// returns `None`.
    fn intersection(&self, thing: &T) -> Option<Self::Intersection>;
}

/// The `union()` operation.
pub trait Union<T = Self> {
    /// The type of the union.
    type Union;

    /// Compute the union of two things.
    fn union(self, other: T) -> Self::Union;
}

/// Linear interpolation between scalars.
pub trait Lerp<T> {
    /// Linear interpolation.
    ///
    /// `self` will be interpolated towards `other`, with progress indicated by
    /// `t`, where `t == 0.0` returns `self` and `t == 1.0` returns `other`.
    #[must_use]
    fn lerp(self, other: Self, t: T) -> Self;
}

/// Computes the absolute value of `Self`.
pub trait Abs {
    /// Computes the absolute value of `Self`.
    #[must_use]
    fn abs(self) -> Self;

    /// Returns a vector with elements representing the sign of `self`.
    #[must_use]
    fn signum(self) -> Self;
}

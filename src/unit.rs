use crate::{FloatScalar, SignedScalar, bindings, scalar::IntScalar};

use super::Scalar;

/// The name of a coordinate space.
///
/// The unit is used to give vector types a "tag" so they can be distinguished at compile time.
///
/// The unit also determines which type is used as the scalar for that coordinate space. This is in contrast to a crate
/// like [euclid](https://docs.rs/euclid/latest/euclid), where the unit and the scalar are separate type parameters to
/// each vector type.
///
/// Note that primitive scalars (`f32`, `f64`, `i32`, `u32`, `i64`, `u64`, `i16`, `u16`) also implement `Unit`. This
/// allows them to function as the "untyped" vector variants.
///
/// #### Example
/// ```rust
/// # use glamour::prelude::*;
/// struct MyTag;
///
/// impl Unit for MyTag {
///     // All types that have `MyTag` as the unit will internally be based
///     // on `f32`.
///     type Scalar = f32;
/// }
///
/// // This will be `glam::Vec2` internally, and can be constructed directly
/// // with `f32` literals.
/// let v: Vector2<MyTag> = Vector2 { x: 1.0, y: 2.0 };
/// assert_eq!(v[0], v.x);
/// ```
pub trait Unit: 'static {
    /// One of the vector component types of glam: `f32`, `f64`, `i32`, or
    /// `u32`.
    type Scalar: Scalar;
}

/// Convenience trait implemented for all [`Unit`]s with an integer scalar type.
///
/// Due to implied associated type bounds, this can be used in trait bounds to enable all integer operations in generic
/// code.
pub trait IntUnit:
    Unit<
    Scalar: IntScalar<
        Vec2: bindings::IntegerVector,
        Vec3: bindings::IntegerVector,
        Vec4: bindings::IntegerVector,
    >,
>
{
}
impl<T> IntUnit for T where
    T: Unit<
        Scalar: IntScalar<
            Vec2: bindings::IntegerVector,
            Vec3: bindings::IntegerVector,
            Vec4: bindings::IntegerVector,
        >,
    >
{
}

/// Convenience trait implemented for all [`Unit`]s with a signed scalar type.
///
/// Due to implied associated type bounds, this can be used in trait bounds to enable all signed operations in generic
/// code.
pub trait SignedUnit: Unit<Scalar: SignedScalar> {}
impl<T> SignedUnit for T where T: Unit<Scalar: SignedScalar> {}

/// Convenience trait implemented for all [`Unit`]s with a floating-point scalar type.
///
/// Due to implied associated type bounds, this can be used in trait bounds to enable all floating-point operations in
/// generic code.
pub trait FloatUnit: Unit<Scalar: FloatScalar> {}
impl<T> FloatUnit for T where T: Unit<Scalar: FloatScalar> {}

impl Unit for f32 {
    type Scalar = f32;
}

impl Unit for f64 {
    type Scalar = f64;
}

impl Unit for i16 {
    type Scalar = i16;
}

impl Unit for i32 {
    type Scalar = i32;
}

impl Unit for u16 {
    type Scalar = u16;
}

impl Unit for u32 {
    type Scalar = u32;
}

impl Unit for i64 {
    type Scalar = i64;
}

impl Unit for u64 {
    type Scalar = u64;
}

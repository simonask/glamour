use super::Scalar;
use crate::bindings::{Primitive, PrimitiveMatrices, Vector};

/// The name of a coordinate space.
///
/// The unit is used to give vector types a "tag" so they can be distinguished
/// at compile time.
///
/// The unit also determines which type is used as the scalar for that
/// coordinate space. This is in contrast to a crate like
/// [euclid](https://docs.rs/euclid/latest/euclid), where the unit and the
/// scalar are separate type parameters to each vector type.
///
/// Note that primitive scalars (`f32`, `f64`, `i32`, and `u32`) also implement
/// `Unit`. This allows them to function as the "untyped" vector variants.
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
    /// The type of scalar in this space. This can be any type that implements
    /// `Scalar`, such as the primitives `i32`, `u32`, `f32`, or `f64`, but also
    /// custom scalar types that implement [`Scalar`].
    type Scalar: Scalar;

    /// Human-readable name of this coordinate space, used in debugging.
    #[must_use]
    fn name() -> Option<&'static str> {
        None
    }
}

impl Unit for f32 {
    type Scalar = f32;
    fn name() -> Option<&'static str> {
        Some("f32")
    }
}

impl Unit for f64 {
    type Scalar = f64;
    fn name() -> Option<&'static str> {
        Some("f64")
    }
}

impl Unit for i32 {
    type Scalar = i32;
    fn name() -> Option<&'static str> {
        Some("i32")
    }
}

impl Unit for u32 {
    type Scalar = u32;
    fn name() -> Option<&'static str> {
        Some("u32")
    }
}

/// Shorthand to go from a `Unit` type to its associated primitives (through the
/// scalar).
pub trait UnitTypes: Unit<Scalar = Self::UnitScalar> {
    /// `Self::Scalar` needed to define trait bounds on the associated type.
    type UnitScalar: Scalar<Primitive = Self::Primitive>;
    /// Shorthand for `<Self::Scalar as Scalar>::Primitive`.
    type Primitive: Primitive<Vec2 = Self::Vec2, Vec3 = Self::Vec3, Vec4 = Self::Vec4>;

    /// Fundamental 2D vector type.
    type Vec2: Vector<2, Scalar = Self::Primitive, Mask = glam::BVec2>;
    /// Fundamental 2D vector type.
    type Vec3: Vector<3, Scalar = Self::Primitive, Mask = glam::BVec3A>;
    /// Fundamental 2D vector type.
    type Vec4: Vector<4, Scalar = Self::Primitive, Mask = glam::BVec4A>;
}

impl<T> UnitTypes for T
where
    T: Unit,
{
    type UnitScalar = T::Scalar;
    type Primitive = <T::Scalar as Scalar>::Primitive;
    type Vec2 = <Self::Primitive as Primitive>::Vec2;
    type Vec3 = <Self::Primitive as Primitive>::Vec3;
    type Vec4 = <Self::Primitive as Primitive>::Vec4;
}

/// Shorthand to go from a `Unit` to its corresponding `Matrix` implementations
/// (through `Scalar::Primitive`).
pub trait UnitMatrices:
    UnitTypes<
    Scalar = Self::UnitMatrixScalar,
    UnitScalar = Self::UnitMatrixScalar,
    Primitive = Self::UnitMatrixPrimitive,
>
{
    /// Same as `Unit::Scalar`.
    type UnitMatrixScalar: Scalar<Primitive = Self::UnitMatrixPrimitive>;
    /// Same as `UnitTypes::Primitive` and `Unit::Scalar::Primitive`.
    type UnitMatrixPrimitive: Primitive<Vec2 = Self::Vec2, Vec3 = Self::Vec3, Vec4 = Self::Vec4>
        + PrimitiveMatrices<Mat2 = Self::Mat2, Mat3 = Self::Mat3, Mat4 = Self::Mat4>;
    /// Either [`glam::Mat2`] or [`glam::DMat2`].
    type Mat2: crate::bindings::Matrix2<Scalar = Self::Primitive>;
    /// Either [`glam::Mat3`] or [`glam::DMat3`].
    type Mat3: crate::bindings::Matrix3<Scalar = Self::Primitive>;
    /// Either [`glam::Mat4`] or [`glam::DMat4`].
    type Mat4: crate::bindings::Matrix4<Scalar = Self::Primitive>;
}

impl<T> UnitMatrices for T
where
    T: UnitTypes,
    T::Primitive: PrimitiveMatrices,
{
    type UnitMatrixScalar = Self::UnitScalar;
    type UnitMatrixPrimitive = T::Primitive;
    type Mat2 = <T::Primitive as PrimitiveMatrices>::Mat2;
    type Mat3 = <T::Primitive as PrimitiveMatrices>::Mat3;
    type Mat4 = <T::Primitive as PrimitiveMatrices>::Mat4;
}

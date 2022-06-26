use core::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign};

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

/// Map unit to glam types by dimension.
pub trait UnitDimTypes<const D: usize>: Unit {
    type Vec: Vector<D, Scalar = <Self::Scalar as Scalar>::Primitive, Mask = Self::BVec>;
    type BVec: Copy + Eq + BitAnd + BitAndAssign + BitOr + BitOrAssign;
}

impl<T: Unit> UnitDimTypes<2> for T {
    type Vec = <<Self::Scalar as Scalar>::Primitive as Primitive>::Vec2;
    type BVec = <<Self::Scalar as Scalar>::Primitive as Primitive>::BVec2;
}

impl<T: Unit> UnitDimTypes<3> for T {
    type Vec = <<Self::Scalar as Scalar>::Primitive as Primitive>::Vec3;
    type BVec = <<Self::Scalar as Scalar>::Primitive as Primitive>::BVec3;
}

impl<T: Unit> UnitDimTypes<4> for T {
    type Vec = <<Self::Scalar as Scalar>::Primitive as Primitive>::Vec4;
    type BVec = <<Self::Scalar as Scalar>::Primitive as Primitive>::BVec4;
}

/// Shorthand to go from a `Unit` type to its associated primitives (through the
/// scalar).
pub trait UnitTypes:
    UnitDimTypes<2, Scalar = Self::Scalar_, Vec = Self::Vec2, BVec = Self::BVec2>
    + UnitDimTypes<3, Scalar = Self::Scalar_, Vec = Self::Vec3, BVec = Self::BVec3>
    + UnitDimTypes<4, Scalar = Self::Scalar_, Vec = Self::Vec4, BVec = Self::BVec4>
{
    /// `Self::Scalar` needed to define trait bounds on the associated type.
    #[doc(hidden)]
    type Scalar_: Scalar<Primitive = Self::Primitive>;
    /// Shorthand for `<Self::Scalar as Scalar>::Primitive`.
    type Primitive: Primitive<
        Vec2 = Self::Vec2,
        Vec3 = Self::Vec3,
        Vec4 = Self::Vec4,
        BVec2 = Self::BVec2,
        BVec3 = Self::BVec3,
        BVec4 = Self::BVec4,
    >;

    /// Fundamental 2D vector type ([`glam::Vec2`], [`glam::DVec2`], [`glam::IVec2`], or [`glam::UVec2`]).
    type Vec2: Vector<2, Scalar = Self::Primitive, Mask = Self::BVec2>;
    /// Fundamental 3D vector type ([`glam::Vec3`], [`glam::DVec3`], [`glam::IVec3`], or [`glam::UVec3`]).
    type Vec3: Vector<3, Scalar = Self::Primitive, Mask = Self::BVec3>;
    /// Fundamental 4D vector type ([`glam::Vec4`], [`glam::DVec4`], [`glam::IVec4`], or [`glam::UVec4`]).
    type Vec4: Vector<4, Scalar = Self::Primitive, Mask = Self::BVec4>;

    /// 2D boolean vector type. Always `glam::BVec2`.
    type BVec2: Copy + Eq + BitAnd + BitAndAssign + BitOr + BitOrAssign;
    /// 3D boolean vector type. Either [`glam::BVec3`] or [`glam::BVec3A`].
    type BVec3: Copy + Eq + BitAnd + BitAndAssign + BitOr + BitOrAssign;
    /// 4D boolean vector type. Either [`glam::BVec4`] or [`glam::BVec4A`].
    type BVec4: Copy + Eq + BitAnd + BitAndAssign + BitOr + BitOrAssign;
}

impl<T> UnitTypes for T
where
    T: Unit,
{
    type Scalar_ = T::Scalar;
    type Primitive = <T::Scalar as Scalar>::Primitive;
    type Vec2 = <Self::Primitive as Primitive>::Vec2;
    type Vec3 = <Self::Primitive as Primitive>::Vec3;
    type Vec4 = <Self::Primitive as Primitive>::Vec4;
    type BVec2 = <Self::Primitive as Primitive>::BVec2;
    type BVec3 = <Self::Primitive as Primitive>::BVec3;
    type BVec4 = <Self::Primitive as Primitive>::BVec4;
}

/// Shorthand to go from a `Unit` to its corresponding `Matrix` implementations
/// (through `Scalar::Primitive`).
pub trait UnitMatrices: UnitTypes<Scalar_ = Self::Scalar__, Primitive = Self::Primitive_> {
    /// Same as `Unit::Scalar`.
    #[doc(hidden)]
    type Scalar__: Scalar<Primitive = Self::Primitive>;
    /// Same as `UnitTypes::Primitive` and `Unit::Scalar::Primitive`.
    #[doc(hidden)]
    type Primitive_: Primitive<Vec2 = Self::Vec2, Vec3 = Self::Vec3, Vec4 = Self::Vec4>
        + PrimitiveMatrices<Mat2 = Self::Mat2, Mat3 = Self::Mat3, Mat4 = Self::Mat4>;

    /// Either [`glam::Mat2`] or [`glam::DMat2`].
    type Mat2: crate::bindings::Matrix2<
        Scalar = Self::Primitive,
        Vec2 = Self::Vec2,
        Vec3 = Self::Vec3,
        Vec4 = Self::Vec4,
    >;
    /// Either [`glam::Mat3`] or [`glam::DMat3`].
    type Mat3: crate::bindings::Matrix3<
        Scalar = Self::Primitive,
        Vec2 = Self::Vec2,
        Vec3 = Self::Vec3,
        Vec4 = Self::Vec4,
    >;
    /// Either [`glam::Mat4`] or [`glam::DMat4`].
    type Mat4: crate::bindings::Matrix4<
        Scalar = Self::Primitive,
        Vec2 = Self::Vec2,
        Vec3 = Self::Vec3,
        Vec4 = Self::Vec4,
    >;
}

impl<T> UnitMatrices for T
where
    T: UnitTypes,
    T::Primitive: PrimitiveMatrices,
{
    type Scalar__ = <Self as Unit>::Scalar;
    type Primitive_ = T::Primitive;
    type Mat2 = <T::Primitive as PrimitiveMatrices>::Mat2;
    type Mat3 = <T::Primitive as PrimitiveMatrices>::Mat3;
    type Mat4 = <T::Primitive as PrimitiveMatrices>::Mat4;
}

use num_traits::AsPrimitive;

use crate::bindings;

use crate::traits::marker::{PodValue, WasmComponentType};

/// All types that can serve as components of a SIMD type in [`glam`].
///
/// This is implemented for `f32`, `f64`, `i32`, `u32`, `i16`, `u16`, `i64`, and `u64`.
///
/// # Safety
///
/// The associated types `Vec2`, `Vec3`, and `Vec4` define how this crate's vector/point/size types are mapped to `glam`
/// vectors. Vectorlike types using a [`T: Unit`](crate::Unit) go through `T::Scalar` to find which `glam` vector type
/// to use. Since those types are must be bitwise layout-compatible, this trait is unsafe to implement.
pub unsafe trait Scalar:
    // This bound is important because it implies `Pod`, which means that types
    // that are `#[repr(C)]` and only have `Scalar` members can safely implement
    // `Pod` as well.
    PodValue
    + WasmComponentType
    + PartialOrd
    + core::fmt::Display
    + crate::Unit<Scalar = Self>
    + num_traits::NumOps
    + num_traits::NumAssignOps
    + num_traits::NumCast
    + num_traits::ConstOne
    + num_traits::ConstZero
    + approx::AbsDiffEq<Epsilon = Self>
    + From<bool>
{
    /// The underlying 2D vector type for this scalar (like [`glam::Vec2`],
    /// [`glam::DVec2`], [`glam::IVec2`], or [`glam::UVec2`]).
    type Vec2: bindings::Vector2<Scalar = Self>;
    /// The underlying 3D vector type for this scalar (like [`glam::Vec3`],
    /// [`glam::DVec3`], [`glam::IVec3`], or [`glam::UVec3`]).
    type Vec3: bindings::Vector3<Scalar = Self>;
    /// The underlying 4D vector type for this scalar (like [`glam::Vec4`],
    /// [`glam::DVec4`], [`glam::IVec4`], or [`glam::UVec4`]).
    type Vec4: bindings::Vector4<Scalar = Self>;

    /// Try casting to another scalar type.
    ///
    /// The cast always succeeds if the scalars have the same underlying type
    /// (i.e., the same `Primitive` associated type).
    ///
    /// The cast fails under the same conditions where
    /// `num_traits::NumCast::from()` fails.
    #[inline]
    #[must_use]
    fn try_cast<T2: Scalar>(self) -> Option<T2> {
        num_traits::NumCast::from(self)
    }

    /// Cast the scalar through the `as`operator.
    ///
    /// This is a convenience method that just calls [`num_traits::AsPrimitive`].
    #[inline]
    #[must_use]
    fn as_<T2>(self) -> T2 where Self: AsPrimitive<T2>, T2: Scalar {
        AsPrimitive::as_(self)
    }
}

/// Signed scalar types (`i32`, `f32`, `f64`, etc.).
pub trait SignedScalar:
    Scalar<
        Vec2: bindings::SignedVector2,
        Vec3: bindings::SignedVector,
        Vec4: bindings::SignedVector,
    > + num_traits::Signed
{
    /// Negative one.
    const NEG_ONE: Self;
}

/// Floating-point scalar types (`f32` and `f64`).
///
/// # Safety
///
/// The associated types `Mat2`, `Mat3`, and `Mat4` define how this crate's matrix types are mapped to `glam` matrices.
/// Matrixlike types using a [`T: Unit`](crate::Unit) go through `T::Scalar` to find which `glam` matrix type to use.
/// Since those types are must be bitwise layout-compatible, this trait is unsafe to implement.
pub unsafe trait FloatScalar:
    Scalar<
        Vec2: bindings::FloatVector2<Scalar = Self>,
        Vec3: bindings::FloatVector3<Scalar = Self>,
        Vec4: bindings::FloatVector4<Scalar = Self>,
    > + num_traits::Float
    + approx::RelativeEq<Epsilon = Self>
    + approx::UlpsEq<Epsilon = Self>
{
    /// A NaN value for this floating-point scalar type.
    const NAN: Self;
    /// Infinity.
    const INFINITY: Self;
    /// Negative infinity.
    const NEG_INFINITY: Self;

    /// The underlying 2x2 matrix type for this scalar ([`glam::Mat2`] or [`glam::DMat2`]).
    type Mat2: bindings::Matrix2<Scalar = Self>;
    /// The underlying 3x3 matrix type for this scalar ([`glam::Mat3`] or [`glam::DMat3`]).
    type Mat3: bindings::Matrix3<Scalar = Self>;
    /// The underlying 4x4 matrix type for this scalar ([`glam::Mat4`] or [`glam::DMat4`]).
    type Mat4: bindings::Matrix4<Scalar = Self>;
    /// The underlying quaternion type for this scalar ([`glam::Quat`] or [`glam::DQuat`]).
    type Quat: bindings::Quat<Self>;
}

/// Integer scalar types.
pub trait IntScalar:
    Scalar<Vec2: bindings::IntegerVector, Vec3: bindings::IntegerVector, Vec4: bindings::IntegerVector>
{
}

unsafe impl Scalar for f32 {
    type Vec2 = glam::Vec2;
    type Vec3 = glam::Vec3;
    type Vec4 = glam::Vec4;
}

impl SignedScalar for f32 {
    const NEG_ONE: Self = -1.0;
}

unsafe impl FloatScalar for f32 {
    type Mat2 = glam::Mat2;
    type Mat3 = glam::Mat3;
    type Mat4 = glam::Mat4;
    type Quat = glam::Quat;
    const NAN: Self = f32::NAN;
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
}

unsafe impl Scalar for f64 {
    type Vec2 = glam::DVec2;
    type Vec3 = glam::DVec3;
    type Vec4 = glam::DVec4;
}

impl SignedScalar for f64 {
    const NEG_ONE: Self = -1.0;
}

unsafe impl FloatScalar for f64 {
    type Mat2 = glam::DMat2;
    type Mat3 = glam::DMat3;
    type Mat4 = glam::DMat4;
    type Quat = glam::DQuat;
    const NAN: Self = f64::NAN;
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
}

unsafe impl Scalar for i16 {
    type Vec2 = glam::I16Vec2;
    type Vec3 = glam::I16Vec3;
    type Vec4 = glam::I16Vec4;
}

impl IntScalar for i16 {}

impl SignedScalar for i16 {
    const NEG_ONE: Self = -1;
}

unsafe impl Scalar for i32 {
    type Vec2 = glam::IVec2;
    type Vec3 = glam::IVec3;
    type Vec4 = glam::IVec4;
}

impl IntScalar for i32 {}

impl SignedScalar for i32 {
    const NEG_ONE: Self = -1;
}

unsafe impl Scalar for i64 {
    type Vec2 = glam::I64Vec2;
    type Vec3 = glam::I64Vec3;
    type Vec4 = glam::I64Vec4;
}

impl IntScalar for i64 {}

impl SignedScalar for i64 {
    const NEG_ONE: Self = -1;
}

unsafe impl Scalar for u16 {
    type Vec2 = glam::U16Vec2;
    type Vec3 = glam::U16Vec3;
    type Vec4 = glam::U16Vec4;
}

impl IntScalar for u16 {}

unsafe impl Scalar for u32 {
    type Vec2 = glam::UVec2;
    type Vec3 = glam::UVec3;
    type Vec4 = glam::UVec4;
}

impl IntScalar for u32 {}

unsafe impl Scalar for u64 {
    type Vec2 = glam::U64Vec2;
    type Vec3 = glam::U64Vec3;
    type Vec4 = glam::U64Vec4;
}

impl IntScalar for u64 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_cast() {
        assert_eq!(f32::NAN.try_cast::<i32>(), None);
        assert_eq!(f32::INFINITY.try_cast::<i32>(), None);
        assert_eq!(f32::NEG_INFINITY.try_cast::<i32>(), None);
        assert_eq!(1.0f32.try_cast::<i32>(), Some(1));

        assert_eq!(f32::NAN.try_cast::<u32>(), None);
        assert_eq!(f32::INFINITY.try_cast::<u32>(), None);
        assert_eq!(f32::NEG_INFINITY.try_cast::<u32>(), None);
        assert_eq!(1.0f32.try_cast::<u32>(), Some(1));
        assert_eq!((-1.0f32).try_cast::<u32>(), None);

        assert_eq!(f64::NAN.try_cast::<i32>(), None);
        assert_eq!(f64::INFINITY.try_cast::<i32>(), None);
        assert_eq!(f64::NEG_INFINITY.try_cast::<i32>(), None);
        assert_eq!(1.0f64.try_cast::<i32>(), Some(1));

        assert_eq!(f64::NAN.try_cast::<u32>(), None);
        assert_eq!(f64::INFINITY.try_cast::<u32>(), None);
        assert_eq!(f64::NEG_INFINITY.try_cast::<u32>(), None);
        assert_eq!(1.0f64.try_cast::<u32>(), Some(1));

        assert!(f64::NAN.try_cast::<f32>().unwrap().is_nan());

        assert_eq!(f64::INFINITY.try_cast(), Some(f32::INFINITY));
        assert_eq!(f64::NEG_INFINITY.try_cast(), Some(f32::NEG_INFINITY));
        assert_eq!(1.0f64.try_cast::<f32>(), Some(1.0));
    }
}

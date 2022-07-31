use crate::traits::marker::PodValue;
use crate::{bindings, AsRaw, ToRaw, FromRaw};

/// All types that can serve as components of a SIMD type in [`glam`].
///
/// This is implemented for `f32`, `f64`, `i32`, and `u32`.
pub trait Scalar:
    // This bound is important because it implies `Pod`, which means that types
    // that are `#[repr(C)]` and only have `Scalar` members can safely implement
    // `Pod` as well.
    PodValue
    + PartialOrd
    + ToRaw<Raw = Self> + FromRaw + AsRaw
    + core::fmt::Display

    + crate::Unit<Scalar = Self>
    + num_traits::NumOps
    + num_traits::NumCast
    + num_traits::ToPrimitive
    + num_traits::AsPrimitive<f32>
    + num_traits::AsPrimitive<f64>
    + num_traits::AsPrimitive<i32>
    + num_traits::AsPrimitive<u32>

    + approx::AbsDiffEq<Epsilon = Self>
{
    /// The underlying 2D vector type for this scalar ([`glam::Vec2`],
    /// [`glam::DVec2`], [`glam::IVec2`], or [`glam::UVec2`]).
    type Vec2: bindings::Vector2<Scalar = Self>;
    /// The underlying 3D vector type for this scalar ([`glam::Vec3`],
    /// [`glam::DVec3`], [`glam::IVec3`], or [`glam::UVec3`]).
    type Vec3: bindings::Vector3<Scalar = Self>;
    /// The underlying 4D vector type for this scalar ([`glam::Vec4`],
    /// [`glam::DVec4`], [`glam::IVec4`], or [`glam::UVec4`]).
    type Vec4: bindings::Vector4<Scalar = Self>;

    /// Zero.
    const ZERO: Self;
    /// One.
    const ONE: Self;

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

    /// True if the number is not NaN and not infinity.
    /// 
    /// Always true for integer scalars.
    #[must_use]
    fn is_finite(self) -> bool;
}

/// Signed scalar types (`i32`, `f32`, `f64`).
pub trait SignedScalar: Scalar<Vec2 = Self::Vec2s, Vec3 = Self::Vec3s, Vec4 = Self::Vec4s> {
    const NEG_ONE: Self;
    type Vec2s: bindings::SignedVector2<Scalar = Self>;
    type Vec3s: bindings::SignedVector<Scalar = Self>;
    type Vec4s: bindings::SignedVector<Scalar = Self>;
}

pub trait FloatScalar:
    SignedScalar<Vec2s = Self::Vec2f, Vec3s = Self::Vec3f, Vec4s = Self::Vec4f>
    + num_traits::Float
    + approx::RelativeEq<Epsilon = Self>
    + approx::UlpsEq<Epsilon = Self>
{
    type Vec2f: bindings::FloatVector2<Scalar = Self>;
    type Vec3f: bindings::FloatVector3<Scalar = Self>;
    type Vec4f: bindings::FloatVector4<Scalar = Self>;

    type Mat2: bindings::Matrix2<
        Scalar = Self,
        Vec2 = Self::Vec2f,
        Vec3 = Self::Vec3f,
        Vec4 = Self::Vec4f,
    >;
    type Mat3: bindings::Matrix3<
        Scalar = Self,
        Vec2 = Self::Vec2f,
        Vec3 = Self::Vec3f,
        Vec4 = Self::Vec4f,
    >;
    type Mat4: bindings::Matrix4<
        Scalar = Self,
        Vec2 = Self::Vec2f,
        Vec3 = Self::Vec3f,
        Vec4 = Self::Vec4f,
    >;
    type Quat: bindings::Quat<Scalar = Self, Vec3 = Self::Vec3f, Vec4 = Self::Vec4f>
        + ToRaw<Raw = Self::Quat>;
    const NAN: Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    type Vec2 = glam::Vec2;
    type Vec3 = glam::Vec3;
    type Vec4 = glam::Vec4;

    fn is_finite(self) -> bool {
        num_traits::Float::is_finite(self)
    }
}

impl SignedScalar for f32 {
    const NEG_ONE: Self = -1.0;

    type Vec2s = glam::Vec2;
    type Vec3s = glam::Vec3;
    type Vec4s = glam::Vec4;
}

impl FloatScalar for f32 {
    type Vec2f = glam::Vec2;
    type Vec3f = glam::Vec3;
    type Vec4f = glam::Vec4;

    type Mat2 = glam::Mat2;
    type Mat3 = glam::Mat3;
    type Mat4 = glam::Mat4;
    type Quat = glam::Quat;
    const NAN: Self = f32::NAN;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    type Vec2 = glam::DVec2;
    type Vec3 = glam::DVec3;
    type Vec4 = glam::DVec4;

    fn is_finite(self) -> bool {
        num_traits::Float::is_finite(self)
    }
}

impl SignedScalar for f64 {
    const NEG_ONE: Self = -1.0;

    type Vec2s = glam::DVec2;
    type Vec3s = glam::DVec3;
    type Vec4s = glam::DVec4;
}

impl FloatScalar for f64 {
    type Vec2f = glam::DVec2;
    type Vec3f = glam::DVec3;
    type Vec4f = glam::DVec4;

    type Mat2 = glam::DMat2;
    type Mat3 = glam::DMat3;
    type Mat4 = glam::DMat4;
    type Quat = glam::DQuat;
    const NAN: Self = f64::NAN;
}

impl Scalar for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;

    type Vec2 = glam::IVec2;
    type Vec3 = glam::IVec3;
    type Vec4 = glam::IVec4;

    fn is_finite(self) -> bool {
        true
    }
}

impl SignedScalar for i32 {
    const NEG_ONE: Self = -1;

    type Vec2s = glam::IVec2;
    type Vec3s = glam::IVec3;
    type Vec4s = glam::IVec4;
}

impl Scalar for u32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;

    type Vec2 = glam::UVec2;
    type Vec3 = glam::UVec3;
    type Vec4 = glam::UVec4;

    fn is_finite(self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn try_cast() {
        assert_eq!(core::f32::NAN.try_cast::<i32>(), None);
        assert_eq!(core::f32::INFINITY.try_cast::<i32>(), None);
        assert_eq!(core::f32::NEG_INFINITY.try_cast::<i32>(), None);
        assert_eq!(1.0f32.try_cast::<i32>(), Some(1));

        assert_eq!(core::f32::NAN.try_cast::<u32>(), None);
        assert_eq!(core::f32::INFINITY.try_cast::<u32>(), None);
        assert_eq!(core::f32::NEG_INFINITY.try_cast::<u32>(), None);
        assert_eq!(1.0f32.try_cast::<u32>(), Some(1));
        assert_eq!((-1.0f32).try_cast::<u32>(), None);

        assert_eq!(core::f64::NAN.try_cast::<i32>(), None);
        assert_eq!(core::f64::INFINITY.try_cast::<i32>(), None);
        assert_eq!(core::f64::NEG_INFINITY.try_cast::<i32>(), None);
        assert_eq!(1.0f64.try_cast::<i32>(), Some(1));

        assert_eq!(core::f64::NAN.try_cast::<u32>(), None);
        assert_eq!(core::f64::INFINITY.try_cast::<u32>(), None);
        assert_eq!(core::f64::NEG_INFINITY.try_cast::<u32>(), None);
        assert_eq!(1.0f64.try_cast::<u32>(), Some(1));

        assert!(core::f64::NAN.try_cast::<f32>().unwrap().is_nan());

        assert_eq!(core::f64::INFINITY.try_cast(), Some(core::f32::INFINITY));
        assert_eq!(
            core::f64::NEG_INFINITY.try_cast(),
            Some(core::f32::NEG_INFINITY)
        );
        assert_eq!(1.0f64.try_cast::<f32>(), Some(1.0));
    }
}

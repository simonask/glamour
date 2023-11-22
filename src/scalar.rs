use core::ops::{BitAnd, BitOr, BitXor, Not};

use crate::traits::marker::PodValue;
use crate::{bindings, AsRaw, FromRaw, ToRaw};

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
    type Vec2: bindings::Vector2<Scalar = Self, Mask = Self::BVec2>;
    /// The underlying 3D vector type for this scalar ([`glam::Vec3`],
    /// [`glam::DVec3`], [`glam::IVec3`], or [`glam::UVec3`]).
    type Vec3: bindings::Vector3<Scalar = Self, Mask = Self::BVec3>;
    /// The underlying 4D vector type for this scalar ([`glam::Vec4`],
    /// [`glam::DVec4`], [`glam::IVec4`], or [`glam::UVec4`]).
    type Vec4: bindings::Vector4<Scalar = Self, Mask = Self::BVec4>;

    /// The bitmask used for comparison of 2D vectors of this type.
    type BVec2: BitAnd<Output = Self::BVec2>
    + BitOr<Output = Self::BVec2>
    + Not<Output = Self::BVec2>
    + BitXor<Output = Self::BVec2>
    + FromRaw<Raw = Self::BVec2>;
    /// The bitmask used for comparison of 3D vectors of this type.
    type BVec3: BitAnd<Output = Self::BVec3>
    + BitOr<Output = Self::BVec3>
    + Not<Output = Self::BVec3>
    + BitXor<Output = Self::BVec3>
    + FromRaw<Raw = Self::BVec3>;
    /// The bitmask used for comparison of 4D vectors of this type.
    type BVec4: BitAnd<Output = Self::BVec4>
    + BitOr<Output = Self::BVec4>
    + Not<Output = Self::BVec4>
    + BitXor<Output = Self::BVec4>
    + FromRaw<Raw = Self::BVec4>;

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
    type Vec2s: bindings::SignedVector2<Scalar = Self, Mask = Self::BVec2>;
    type Vec3s: bindings::SignedVector<Scalar = Self, Mask = Self::BVec3>;
    type Vec4s: bindings::SignedVector<Scalar = Self, Mask = Self::BVec4>;
}

pub trait FloatScalar:
    SignedScalar<Vec2s = Self::Vec2f, Vec3s = Self::Vec3f, Vec4s = Self::Vec4f>
    + num_traits::Float
    + approx::RelativeEq<Epsilon = Self>
    + approx::UlpsEq<Epsilon = Self>
{
    type Vec2f: bindings::FloatVector2<Scalar = Self, Mask = Self::BVec2>;
    type Vec3f: bindings::FloatVector3<Scalar = Self, Mask = Self::BVec3>;
    type Vec4f: bindings::FloatVector4<Scalar = Self, Mask = Self::BVec4>;

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
    const INFINITY: Self;
    const NEG_INFINITY: Self;
}

impl Scalar for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    type Vec2 = glam::Vec2;
    type Vec3 = glam::Vec3;
    type Vec4 = glam::Vec4;
    type BVec2 = glam::BVec2;
    type BVec3 = glam::BVec3;

    #[cfg(all(
        any(
            target_feature = "sse2",
            target_feature = "simd128",
            feature = "core-simd"
        ),
        not(feature = "scalar-math"),
    ))]
    type BVec4 = glam::BVec4A;

    #[cfg(not(all(
        any(
            target_feature = "sse2",
            target_feature = "simd128",
            feature = "core-simd"
        ),
        not(feature = "scalar-math"),
    )))]
    type BVec4 = glam::BVec4;

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
    const INFINITY: Self = f32::INFINITY;
    const NEG_INFINITY: Self = f32::NEG_INFINITY;
}

impl Scalar for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;

    type Vec2 = glam::DVec2;
    type Vec3 = glam::DVec3;
    type Vec4 = glam::DVec4;
    type BVec2 = glam::BVec2;
    type BVec3 = glam::BVec3;
    type BVec4 = glam::BVec4;

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
    const INFINITY: Self = f64::INFINITY;
    const NEG_INFINITY: Self = f64::NEG_INFINITY;
}

impl Scalar for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;

    type Vec2 = glam::IVec2;
    type Vec3 = glam::IVec3;
    type Vec4 = glam::IVec4;
    type BVec2 = glam::BVec2;
    type BVec3 = glam::BVec3;
    type BVec4 = glam::BVec4;

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

impl Scalar for i64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;

    type Vec2 = glam::I64Vec2;
    type Vec3 = glam::I64Vec3;
    type Vec4 = glam::I64Vec4;
    type BVec2 = glam::BVec2;
    type BVec3 = glam::BVec3;
    type BVec4 = glam::BVec4;

    fn is_finite(self) -> bool {
        true
    }
}

impl SignedScalar for i64 {
    const NEG_ONE: Self = -1;

    type Vec2s = glam::I64Vec2;
    type Vec3s = glam::I64Vec3;
    type Vec4s = glam::I64Vec4;
}

impl Scalar for u32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;

    type Vec2 = glam::UVec2;
    type Vec3 = glam::UVec3;
    type Vec4 = glam::UVec4;
    type BVec2 = glam::BVec2;
    type BVec3 = glam::BVec3;
    type BVec4 = glam::BVec4;

    fn is_finite(self) -> bool {
        true
    }
}

impl Scalar for u64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;

    type Vec2 = glam::U64Vec2;
    type Vec3 = glam::U64Vec3;
    type Vec4 = glam::U64Vec4;
    type BVec2 = glam::BVec2;
    type BVec3 = glam::BVec3;
    type BVec4 = glam::BVec4;

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

    #[test]
    fn finite() {
        // dummy tests for coverage
        assert!(0i32.is_finite());
        assert!(0u32.is_finite());
        assert!(0i64.is_finite());
        assert!(0u64.is_finite());
        assert!(0.0f32.is_finite());
        assert!(0.0f64.is_finite());
        assert!(!Scalar::is_finite(f32::NAN));
        assert!(!Scalar::is_finite(f32::INFINITY));
        assert!(!Scalar::is_finite(f32::NEG_INFINITY));
        assert!(!Scalar::is_finite(f64::NAN));
        assert!(!Scalar::is_finite(f64::INFINITY));
        assert!(!Scalar::is_finite(f64::NEG_INFINITY));
    }
}

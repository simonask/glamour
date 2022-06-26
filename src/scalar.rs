use crate::bindings::Primitive;
use crate::traits::marker::{PodValue, Serializable};

/// All types that can serve as components of a SIMD type.
///
/// All scalars must be representable by one of `f32`, `f64`, `i32`, or `u32`.
/// However, all types that are bitwise compatible with one of those may be used
/// as a scalar.
///
/// These are the things that you say when you implement `Scalar` for your
/// custom unit:
///
/// - It is a [`Pod`](bytemuck::Pod) type.
/// - It is bitwise compatible with its `Primitive` associated type (same size
///   and alignment).
/// - All bit patterns are valid values.
/// - No new invariants are introduced by the type.
/// - No conversion logic is required to turn `Primitive` into `Self` -
///   reinterpreting the bits is enough.
/// - Arithmetic operations on the scalar have the same effect as the equivalent
///   operations on the `Primitive`. When the scalar is used in a vector type,
///   the primitive operation will be used in vector math.
///
/// In short, custom scalars should be trivial newtypes containing a primitive.
///
/// You can use a crate like
/// [`derive_more`](https://docs.rs/derive_more/latest/derive_more) to derive
/// the arithmetic operations in a convenient and less error-prone way.
pub trait Scalar:
    // This bound is important because it implies `Pod`, which means that types
    // that are `#[repr(C)]` and only have `Scalar` members can safely implement
    // `Pod` as well.
    PodValue
    + PartialOrd
    + Serializable
{
    /// The underlying primitive type of this scalar (`f32`, `f64`, `i32`, or
    /// `u32`).
    /// 
    /// Must be bitwise compatible with `Self`. If it isn't, you will get
    /// nonsensical results for arithmetic operations, and in case of a
    /// size/alignment difference, panics.
    type Primitive: Primitive;

    /// Zero.
    const ZERO: Self;
    /// One.
    const ONE: Self;

    /// Bitwise cast from `Primitive` to `Self`.
    #[inline]
    #[must_use]
    fn from_raw(raw: Self::Primitive) -> Self {
        bytemuck::cast(raw)
    }

    /// Bitwise cast from `Self` to `Primitive`.
    #[inline]
    #[must_use]
    fn to_raw(self) -> Self::Primitive {
        bytemuck::cast(self)
    }

    /// Reinterpret a reference to `Self` as a reference to `Primitive`.
    #[inline]
    #[must_use]
    fn as_raw(&self) -> &Self::Primitive {
        bytemuck::cast_ref(self)
    }

    /// Reinterpret a mutable reference to `Self` as a mutable reference to
    /// `Primitive`.
    #[inline]
    #[must_use]
    fn as_raw_mut(&mut self) -> &mut Self::Primitive {
        bytemuck::cast_mut(self)
    }

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
        let primitive = self.to_raw();
        let primitive: T2::Primitive = num_traits::NumCast::from(primitive)?;
        Some(T2::from_raw(primitive))
    }

    /// True if the value is finite (not infinity and not NaN).
    /// 
    /// This is always true for integers.
    #[inline]
    #[must_use]
    fn is_finite(self) -> bool {
        Primitive::is_finite(self.to_raw())
    }
}

/// Signed scalar types (`i32`, `f32`, `f64`).
pub trait SignedScalar: Scalar {
    const NEG_ONE: Self;
}

impl Scalar for f32 {
    type Primitive = f32;

    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl SignedScalar for f32 {
    const NEG_ONE: Self = -1.0;
}

impl Scalar for f64 {
    type Primitive = f64;

    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl SignedScalar for f64 {
    const NEG_ONE: Self = -1.0;
}

impl Scalar for i32 {
    type Primitive = i32;

    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl SignedScalar for i32 {
    const NEG_ONE: Self = -1;
}

impl Scalar for u32 {
    type Primitive = u32;

    const ZERO: Self = 0;
    const ONE: Self = 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_finite() {
        assert!(Scalar::is_finite(1i32));
        assert!(Scalar::is_finite(1u32));
        assert!(Scalar::is_finite(1.0f32));
        assert!(Scalar::is_finite(1.0f64));
        assert!(!Scalar::is_finite(core::f32::NAN));
        assert!(!Scalar::is_finite(core::f32::INFINITY));
        assert!(!Scalar::is_finite(core::f32::NEG_INFINITY));
        assert!(!Scalar::is_finite(core::f64::NAN));
        assert!(!Scalar::is_finite(core::f64::INFINITY));
        assert!(!Scalar::is_finite(core::f64::NEG_INFINITY));
    }

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

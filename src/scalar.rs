use crate::bindings::Primitive;
use crate::traits::marker::{Serializable, ValueSemantics};

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
    ValueSemantics
    + PartialOrd
    + Serializable
{
    /// The underlying primitive type of this scalar (`f32`, `f64`, `i32`, or
    /// `u32`).
    /// 
    /// Must be bitwise compatible with `Self`. If it isn't, you will get
    /// nonsensical results for arithmetic operations, and in case of a
    /// size/alignment difference, panics.
    type Primitive: Primitive
        + num_traits::ToPrimitive
        + num_traits::Num
        + num_traits::NumCast;


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

    /// Get the minimum of two scalar values through `PartialOrd`. If the values
    /// are not meaningfully comparable (such as NaN), either one may be
    /// returned.
    #[inline]
    #[must_use]
    fn min(self, other: Self) -> Self {
        if self < other {
            self
        } else {
            other
        }
    }

    /// Get the maximum of two scalar values through `PartialOrd`. If the values
    /// are not meaningfully comparable (such as NaN), either one may be
    /// returned.
    #[inline]
    #[must_use]
    fn max(self, other: Self) -> Self {
        if self > other {
            self
        } else {
            other
        }
    }

    /// "Zero" from the `Primitive` type's implementation of `num_traits::Zero`.
    #[inline]
    #[must_use]
    fn zero() -> Self {
        Self::from_raw(<Self::Primitive as num_traits::Zero>::zero())
    }

    /// "One" from the `Primitive` type's implementation of `num_traits::One`.
    #[inline]
    #[must_use]
    fn one() -> Self {
        Self::from_raw(<Self::Primitive as num_traits::One>::one())
    }

    /// Convenience method for `Self::from_raw(Self::Primitive::one() +
    /// Self::Primitive::one())`.
    #[inline]
    #[must_use]
    fn two() -> Self {
        let one = <Self::Primitive as num_traits::One>::one();
        Self::from_raw(one + one)
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

impl Scalar for f32 {
    type Primitive = f32;
}

impl Scalar for f64 {
    type Primitive = f64;
}

impl Scalar for i32 {
    type Primitive = i32;
}

impl Scalar for u32 {
    type Primitive = u32;
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

    #[test]
    fn minmax() {
        assert_eq!(Scalar::min(1.0, 2.0), 1.0);
        assert_eq!(Scalar::max(1.0, 2.0), 2.0);
    }

    #[test]
    fn two() {
        assert_eq!(f32::two(), 2.0f32);
        assert_eq!(f64::two(), 2.0f64);
        assert_eq!(i32::two(), 2i32);
        assert_eq!(u32::two(), 2u32);
    }
}

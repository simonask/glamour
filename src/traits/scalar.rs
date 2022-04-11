use super::marker::{Serializable, ValueSemantics};
use super::{SimdMatrix2, SimdMatrix3, SimdMatrix4, SimdVec, Unit};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

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

/// Mapping from primitive scalar type to `glam` vector types.
///
/// Depending on the base type of the scalar, vectors of that scalar are mapped
/// to `glam` vectors in the following way:
///
/// | Primitive  | 2D                     | 3D                     | 4D                     |
/// | ---------- | ---------------------- | ---------------------- | ---------------------- |
/// | `f32`      | [`Vec2`](glam::Vec2)   | [`Vec3`](glam::Vec3)   | [`Vec4`](glam::Vec4)   |
/// | `f64`      | [`DVec2`](glam::DVec2) | [`DVec3`](glam::DVec3) | [`DVec4`](glam::DVec4) |
/// | `i32`      | [`IVec2`](glam::IVec2) | [`IVec3`](glam::IVec3) | [`IVec4`](glam::IVec4) |
/// | `u32`      | [`UVec2`](glam::UVec2) | [`UVec3`](glam::UVec3) | [`UVec4`](glam::UVec4) |
///
/// See also [the documentation module](crate::docs#how).
pub trait Primitive:
    Scalar<Primitive = Self>
    + Unit<Scalar = Self>
    + core::fmt::Debug
    + core::fmt::Display
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Rem<Self, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + RemAssign<Self>
    + Sized
    + Send
    + Sync
    + 'static
{
    /// 2D vector type
    type Vec2: SimdVec<2, Scalar = Self, Mask = glam::BVec2>;
    /// 3D vector type
    type Vec3: SimdVec<3, Scalar = Self, Mask = glam::BVec3A>;
    /// 4D vector type
    type Vec4: SimdVec<4, Scalar = Self, Mask = glam::BVec4A>;

    /// True if the value is finite (not infinity, not NaN).
    ///
    /// This is always true for integers.
    #[must_use]
    fn is_finite(self) -> bool;
}

impl Scalar for f32 {
    type Primitive = f32;
}
impl Primitive for f32 {
    type Vec2 = glam::Vec2;
    type Vec3 = glam::Vec3;
    type Vec4 = glam::Vec4;

    fn is_finite(self) -> bool {
        <f32>::is_finite(self)
    }
}

impl Scalar for f64 {
    type Primitive = f64;
}
impl Primitive for f64 {
    type Vec2 = glam::DVec2;
    type Vec3 = glam::DVec3;
    type Vec4 = glam::DVec4;

    fn is_finite(self) -> bool {
        <f64>::is_finite(self)
    }
}

impl Scalar for i32 {
    type Primitive = i32;
}
impl Primitive for i32 {
    type Vec2 = glam::IVec2;
    type Vec3 = glam::IVec3;
    type Vec4 = glam::IVec4;

    fn is_finite(self) -> bool {
        true
    }
}

impl Scalar for u32 {
    type Primitive = u32;
}
impl Primitive for u32 {
    type Vec2 = glam::UVec2;
    type Vec3 = glam::UVec3;
    type Vec4 = glam::UVec4;

    fn is_finite(self) -> bool {
        true
    }
}

/// Mapping from primitive scalar type to `glam` matrix types.
///
/// Depending on the base type of the scalar, matrices of that scalar are mapped
/// to `glam` matrices in the following way:
///
/// | Primitive  | 2D                     | 3D                     | 4D                     |
/// | ---------- | ---------------------- | ---------------------- | ---------------------- |
/// | `f32`      | [`Mat2`](glam::Mat2)   | [`Mat3`](glam::Mat3)   | [`Mat4`](glam::Mat4)   |
/// | `f64`      | [`DMat2`](glam::DMat2) | [`DMat3`](glam::DMat3) | [`DMat4`](glam::DMat4) |
///
/// Note that `glam` does not support integer matrices.
///
/// See also [the documentation module](crate::docs#how).
pub trait PrimitiveMatrices: Primitive {
    /// [`glam::Mat2`] or [`glam::DMat2`].
    type Mat2: SimdMatrix2<Scalar = Self>;
    /// [`glam::Mat3`] or [`glam::DMat3`].
    type Mat3: SimdMatrix3<Scalar = Self>;
    /// [`glam::Mat4`] or [`glam::DMat4`].
    type Mat4: SimdMatrix4<Scalar = Self>;
}

impl PrimitiveMatrices for f32 {
    type Mat2 = glam::Mat2;
    type Mat3 = glam::Mat3;
    type Mat4 = glam::Mat4;
}

impl PrimitiveMatrices for f64 {
    type Mat2 = glam::DMat2;
    type Mat3 = glam::DMat3;
    type Mat4 = glam::DMat4;
}

/// Convenience trait to go from a (potentially non-primitive) scalar to its
/// corresponding underlying vector type.
///
/// See [`Primitive`].
pub trait ScalarVectors: Scalar {
    /// Fundamental 2D vector type.
    type Vec2: SimdVec<2, Scalar = Self::Primitive, Mask = glam::BVec2>;
    /// Fundamental 3D vector type.
    type Vec3: SimdVec<3, Scalar = Self::Primitive, Mask = glam::BVec3A>;
    /// Fundamental 4D vector type.
    type Vec4: SimdVec<4, Scalar = Self::Primitive, Mask = glam::BVec4A>;
}

impl<T> ScalarVectors for T
where
    T: Scalar,
{
    type Vec2 = <T::Primitive as Primitive>::Vec2;
    type Vec3 = <T::Primitive as Primitive>::Vec3;
    type Vec4 = <T::Primitive as Primitive>::Vec4;
}

/// Convenience trait to go from a (potentially non-primitive) scalar to its
/// corresponding underlying matrix type.
///
/// See [`PrimitiveMatrices`].
pub trait ScalarMatrices: Scalar {
    /// [`glam::Mat2`] or [`glam::DMat2`].
    type Mat2: SimdMatrix2<Scalar = Self::Primitive>;
    /// [`glam::Mat3`] or [`glam::DMat3`].
    type Mat3: SimdMatrix3<Scalar = Self::Primitive>;
    /// [`glam::Mat4`] or [`glam::DMat4`].
    type Mat4: SimdMatrix4<Scalar = Self::Primitive>;
}

impl<T> ScalarMatrices for T
where
    T: Scalar,
    T::Primitive: PrimitiveMatrices,
{
    type Mat2 = <T::Primitive as PrimitiveMatrices>::Mat2;
    type Mat3 = <T::Primitive as PrimitiveMatrices>::Mat3;
    type Mat4 = <T::Primitive as PrimitiveMatrices>::Mat4;
}

use super::{marker::ValueSemantics, Abs, Primitive};

use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

/// Basic `glam` types used to back strongly typed vectors.
#[allow(missing_docs)]
pub trait SimdVec<const D: usize>:
    ValueSemantics
    + Into<[Self::Scalar; D]>
    + From<[Self::Scalar; D]>
    + AsRef<[Self::Scalar; D]>
    + AsMut<[Self::Scalar; D]>
    + Mul<Self::Scalar, Output = Self>
    + MulAssign<Self::Scalar>
    + Div<Self::Scalar, Output = Self>
    + DivAssign<Self::Scalar>
    + Rem<Self::Scalar, Output = Self>
    + RemAssign<Self::Scalar>
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Div<Self, Output = Self>
    + DivAssign<Self>
    + Rem<Self, Output = Self>
    + RemAssign<Self>
{
    /// The component type of this `glam` vector.
    type Scalar: Primitive;
    /// The corresponding boolean vector of the same dimensionality as `Self`.
    /// Always one of [`glam::BVec2`], [`glam::BVec3`], or [`glam::BVec4`].
    type Mask;

    fn zero() -> Self;
    fn one() -> Self;
    fn splat(scalar: Self::Scalar) -> Self;
    fn get(&self, index: usize) -> Self::Scalar {
        self.as_ref()[index]
    }
    fn set(&mut self, index: usize, value: Self::Scalar) {
        self.as_mut()[index] = value;
    }
    #[inline]
    fn const_get<const N: usize>(&self) -> Self::Scalar {
        self.get(N)
    }
    #[inline]
    fn const_set<const N: usize>(&mut self, value: Self::Scalar) {
        self.set(N, value);
    }

    fn cmpeq(self, other: Self) -> Self::Mask;
    fn cmpne(self, other: Self) -> Self::Mask;
    fn cmpge(self, other: Self) -> Self::Mask;
    fn cmpgt(self, other: Self) -> Self::Mask;
    fn cmple(self, other: Self) -> Self::Mask;
    fn cmplt(self, other: Self) -> Self::Mask;
    fn clamp(self, min: Self, max: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min_element(self) -> Self::Scalar;
    fn max_element(self) -> Self::Scalar;

    #[must_use]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;

    fn dot(self, other: Self) -> Self::Scalar;
}

/// Implemented for `glam` vectors with floating-point scalars.
#[allow(missing_docs)]
pub trait SimdVecFloat<const D: usize>: SimdVec<D> {
    fn nan() -> Self;
    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
    fn is_nan_mask(self) -> Self::Mask;
    fn ceil(self) -> Self;
    fn floor(self) -> Self;
    fn round(self) -> Self;
    fn normalize(self) -> Self;
    fn length(self) -> Self::Scalar;
    fn exp(self) -> Self;
    fn powf(self, n: Self::Scalar) -> Self;
    fn recip(self) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
}

macro_rules! impl_base {
    ($scalar:ty => $glam_ty:ty { $x:ident, $y:ident }) => {
        impl_base!(
            @impl
            $scalar
            [2]
            => $glam_ty
            { $x, $y }
            glam::BVec2
        );
    };
    ($scalar:ty => $glam_ty:ty { $x:ident, $y:ident, $z:ident }) => {
        impl_base!(
            @impl
            $scalar
            [3]
            => $glam_ty
            { $x, $y, $z }
            glam::BVec3
        );
    };
    ($scalar:ty => $glam_ty:ty { $x:ident, $y:ident, $z:ident, $w:ident }) => {
        impl_base!(
            @impl
            $scalar
            [4]
            => $glam_ty
            { $x, $y, $z, $w }
            glam::BVec4
        );
    };
    (@impl $scalar:ty [$dimensions:tt] => $glam_ty:ty { $($fields:ident),* } $mask:ty) => {
        impl SimdVec<$dimensions> for $glam_ty {
            type Scalar = $scalar;
            type Mask = $mask;

            fn zero() -> Self {
                <$glam_ty>::ZERO
            }

            fn one() -> Self {
                <$glam_ty>::ONE
            }

            #[inline]
            fn splat(scalar: $scalar) -> Self {
                $(
                    let $fields = scalar;
                )*
                <$glam_ty>::new($($fields),*)
            }

            #[inline]
            fn get(&self, lane: usize) -> $scalar {
                let array: &[$scalar; $dimensions] = self.as_ref();
                array[lane]
            }

            #[inline]
            fn set(&mut self, lane: usize, value: $scalar) {
                let array: &mut [$scalar; $dimensions] = self.as_mut();
                array[lane] = value;
            }

            fn cmpeq(self, other: Self) -> Self::Mask {
                <$glam_ty>::cmpeq(self, other)
            }
            fn cmpne(self, other: Self) -> Self::Mask {
                <$glam_ty>::cmpne(self, other)
            }
            fn cmpge(self, other: Self) -> Self::Mask {
                <$glam_ty>::cmpge(self, other)
            }
            fn cmpgt(self, other: Self) -> Self::Mask {
                <$glam_ty>::cmpgt(self, other)
            }
            fn cmple(self, other: Self) -> Self::Mask {
                <$glam_ty>::cmple(self, other)
            }
            fn cmplt(self, other: Self) -> Self::Mask {
                <$glam_ty>::cmplt(self, other)
            }
            fn clamp(self, min: Self, max: Self) -> Self {
                <$glam_ty>::clamp(self, min, max)
            }
            fn min(self, other: Self) -> Self {
                <$glam_ty>::min(self, other)
            }
            fn max(self, other: Self) -> Self {
                <$glam_ty>::max(self, other)
            }
            fn min_element(self) -> $scalar {
                <$glam_ty>::min_element(self)
            }
            fn max_element(self) -> $scalar {
                <$glam_ty>::max_element(self)
            }

            fn select(mask: $mask, if_true: Self, if_false: Self) -> Self {
                <$glam_ty>::select(mask.into(), if_true, if_false)
            }

            fn dot(self, other: Self) -> $scalar {
                <$glam_ty>::dot(self, other)
            }
        }
    };
}

macro_rules! impl_base_float {
    ($scalar:ty => $glam_ty:ty { $x:ident, $y:ident }) => {
        impl_base_float!(
            @impl
            $scalar
            [2]
            => $glam_ty
            { $x, $y }
            glam::BVec2
        );
    };
    ($scalar:ty => $glam_ty:ty { $x:ident, $y:ident, $z:ident }) => {
        impl_base_float!(
            @impl
            $scalar
            [3]
            => $glam_ty
            { $x, $y, $z }
            glam::BVec3
        );
    };
    ($scalar:ty => $glam_ty:ty { $x:ident, $y:ident, $z:ident, $w:ident }) => {
        impl_base_float!(
            @impl
            $scalar
            [4]
            => $glam_ty
            { $x, $y, $z, $w }
            glam::BVec4
        );
    };
    (@impl $scalar:ty [$dimensions:tt] => $glam_ty:ty { $($fields:ident),* } $mask:ty) => {
        impl SimdVecFloat<$dimensions> for $glam_ty {
            fn nan() -> Self {
                <$glam_ty>::NAN
            }
            fn is_finite(self) -> bool {
                <$glam_ty>::is_finite(self)
            }
            fn is_nan(self) -> bool {
                <$glam_ty>::is_nan(self)
            }
            fn is_nan_mask(self) -> Self::Mask {
                <$glam_ty>::is_nan_mask(self)
            }
            fn ceil(self) -> Self {
                <$glam_ty>::ceil(self)
            }
            fn floor(self) -> Self {
                <$glam_ty>::floor(self)
            }
            fn round(self) -> Self {
                <$glam_ty>::round(self)
            }
            fn normalize(self) -> Self {
                <$glam_ty>::normalize(self)
            }
            fn length(self) -> $scalar {
                <$glam_ty>::length(self)
            }
            fn exp(self) -> Self {
                <$glam_ty>::exp(self)
            }
            fn powf(self, n: $scalar) -> Self {
                <$glam_ty>::powf(self, n)
            }
            fn recip(self) -> Self {
                <$glam_ty>::recip(self)
            }
            fn mul_add(self, a: Self, b: Self) -> Self {
                <$glam_ty>::mul_add(self, a, b)
            }
        }

        impl crate::traits::Lerp<$scalar> for $glam_ty {
            fn lerp(self, other: $glam_ty, t: $scalar) -> $glam_ty {
                <$glam_ty>::lerp(self, other, t)
            }
        }
    }
}

macro_rules! impl_abs {
    ($glam_ty:ty) => {
        impl Abs for $glam_ty {
            #[inline]
            fn abs(self) -> Self {
                <$glam_ty>::abs(self)
            }

            #[inline]
            fn signum(self) -> Self {
                <$glam_ty>::signum(self)
            }
        }
    };
}

impl_base!(f32 => glam::Vec2 { x, y });
impl_base!(f32 => glam::Vec3 { x, y, z});
impl_base!(f32 => glam::Vec4 { x, y, z, w });
impl_base!(f64 => glam::DVec2 { x, y });
impl_base!(f64 => glam::DVec3 { x, y, z});
impl_base!(f64 => glam::DVec4 { x, y, z, w });
impl_base!(i32 => glam::IVec2 { x, y });
impl_base!(i32 => glam::IVec3 { x, y, z});
impl_base!(i32 => glam::IVec4 { x, y, z, w });
impl_base!(u32 => glam::UVec2 { x, y });
impl_base!(u32 => glam::UVec3 { x, y, z});
impl_base!(u32 => glam::UVec4 { x, y, z, w });

impl_base_float!(f32 => glam::Vec2 { x, y });
impl_base_float!(f32 => glam::Vec3 { x, y, z});
impl_base_float!(f32 => glam::Vec4 { x, y, z, w });
impl_base_float!(f64 => glam::DVec2 { x, y });
impl_base_float!(f64 => glam::DVec3 { x, y, z});
impl_base_float!(f64 => glam::DVec4 { x, y, z, w });

impl_abs!(glam::Vec2);
impl_abs!(glam::Vec3);
impl_abs!(glam::Vec4);
impl_abs!(glam::DVec2);
impl_abs!(glam::DVec3);
impl_abs!(glam::DVec4);
impl_abs!(glam::IVec2);
impl_abs!(glam::IVec3);
impl_abs!(glam::IVec4);

use super::*;

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

    #[must_use]
    fn zero() -> Self;
    #[must_use]
    fn one() -> Self;
    #[must_use]
    fn splat(scalar: Self::Scalar) -> Self;
    #[must_use]
    fn get(&self, index: usize) -> Self::Scalar {
        self.as_ref()[index]
    }
    fn set(&mut self, index: usize, value: Self::Scalar) {
        self.as_mut()[index] = value;
    }
    #[inline]
    #[must_use]
    fn const_get<const N: usize>(&self) -> Self::Scalar {
        self.get(N)
    }
    #[inline]
    fn const_set<const N: usize>(&mut self, value: Self::Scalar) {
        self.set(N, value);
    }

    #[must_use]
    fn cmpeq(self, other: Self) -> Self::Mask;
    #[must_use]
    fn cmpne(self, other: Self) -> Self::Mask;
    #[must_use]
    fn cmpge(self, other: Self) -> Self::Mask;
    #[must_use]
    fn cmpgt(self, other: Self) -> Self::Mask;
    #[must_use]
    fn cmple(self, other: Self) -> Self::Mask;
    #[must_use]
    fn cmplt(self, other: Self) -> Self::Mask;
    #[must_use]
    fn clamp(self, min: Self, max: Self) -> Self;
    #[must_use]
    fn min(self, other: Self) -> Self;
    #[must_use]
    fn max(self, other: Self) -> Self;
    #[must_use]
    fn min_element(self) -> Self::Scalar;
    #[must_use]
    fn max_element(self) -> Self::Scalar;

    #[must_use]
    fn select(mask: Self::Mask, if_true: Self, if_false: Self) -> Self;

    #[must_use]
    fn dot(self, other: Self) -> Self::Scalar;
}

/// Implemented for `glam` vectors with floating-point scalars.
#[allow(missing_docs)]
pub trait SimdVecFloat<const D: usize>: SimdVec<D> {
    #[must_use]
    fn nan() -> Self;
    #[must_use]
    fn is_finite(self) -> bool;
    #[must_use]
    fn is_nan(self) -> bool;
    #[must_use]
    fn is_nan_mask(self) -> Self::Mask;
    #[must_use]
    fn ceil(self) -> Self;
    #[must_use]
    fn floor(self) -> Self;
    #[must_use]
    fn round(self) -> Self;
    #[must_use]
    fn normalize(self) -> Self;
    #[must_use]
    fn length(self) -> Self::Scalar;
    #[must_use]
    fn exp(self) -> Self;
    #[must_use]
    fn powf(self, n: Self::Scalar) -> Self;
    #[must_use]
    fn recip(self) -> Self;
    #[must_use]
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
            glam::BVec3A
        );
    };
    ($scalar:ty => $glam_ty:ty { $x:ident, $y:ident, $z:ident, $w:ident }) => {
        impl_base!(
            @impl
            $scalar
            [4]
            => $glam_ty
            { $x, $y, $z, $w }
            glam::BVec4A
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
                <$glam_ty>::cmpeq(self, other).into()
            }
            fn cmpne(self, other: Self) -> Self::Mask {
                <$glam_ty>::cmpne(self, other).into()
            }
            fn cmpge(self, other: Self) -> Self::Mask {
                <$glam_ty>::cmpge(self, other).into()
            }
            fn cmpgt(self, other: Self) -> Self::Mask {
                <$glam_ty>::cmpgt(self, other).into()
            }
            fn cmple(self, other: Self) -> Self::Mask {
                <$glam_ty>::cmple(self, other).into()
            }
            fn cmplt(self, other: Self) -> Self::Mask {
                <$glam_ty>::cmplt(self, other).into()
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
                <$glam_ty>::is_nan_mask(self).into()
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
        impl crate::traits::Abs for $glam_ty {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn get_set() {
        let v = glam::Vec4::new(0.0, 1.0, 2.0, 3.0);
        assert_eq!(v.get(0), 0.0);
        assert_eq!(v.get(1), 1.0);
        assert_eq!(v.get(2), 2.0);
        assert_eq!(v.get(3), 3.0);

        let mut v = v;
        v.set(0, 3.0);
        v.set(1, 2.0);
        v.set(2, 1.0);
        v.set(3, 0.0);
        assert_eq!(v.get(0), 3.0);
        assert_eq!(v.get(1), 2.0);
        assert_eq!(v.get(2), 1.0);
        assert_eq!(v.get(3), 0.0);
    }

    #[test]
    fn const_get_set() {
        let v = glam::Vec4::new(0.0, 1.0, 2.0, 3.0);
        assert_eq!(v.const_get::<0>(), 0.0);
        assert_eq!(v.const_get::<1>(), 1.0);
        assert_eq!(v.const_get::<2>(), 2.0);
        assert_eq!(v.const_get::<3>(), 3.0);

        let mut v = v;
        v.const_set::<0>(3.0);
        v.const_set::<1>(2.0);
        v.const_set::<2>(1.0);
        v.const_set::<3>(0.0);
        assert_eq!(v.const_get::<0>(), 3.0);
        assert_eq!(v.const_get::<1>(), 2.0);
        assert_eq!(v.const_get::<2>(), 1.0);
        assert_eq!(v.const_get::<3>(), 0.0);
    }

    #[test]
    fn nan() {
        let a = glam::Vec4::nan();
        let b = glam::Vec4::NAN;

        assert!(SimdVecFloat::is_nan(a));
        assert!(SimdVecFloat::is_nan(b));

        let a: glam::UVec4 = bytemuck::cast(a);
        let b: glam::UVec4 = bytemuck::cast(b);
        assert_eq!(a, b);
    }

    #[test]
    fn is_finite() {
        let a = glam::Vec4::NAN;
        assert!(!SimdVecFloat::<4>::is_finite(a));

        let a = glam::DVec4::NAN;
        assert!(!SimdVecFloat::<4>::is_finite(a));
    }
}

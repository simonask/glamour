use super::*;

/// Basic `glam` types used to back strongly typed vectors.
#[allow(missing_docs)]
pub trait Vector<const D: usize>:
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
    fn splat(scalar: Self::Scalar) -> Self;
    #[must_use]
    fn get(&self, index: usize) -> Self::Scalar;
    fn set(&mut self, index: usize, value: Self::Scalar);
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
pub trait VectorFloat<const D: usize>: Vector<D> {
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
    fn normalize_or_zero(self) -> Self;
    #[must_use]
    fn is_normalized(self) -> bool;
    #[must_use]
    fn length(self) -> Self::Scalar;
    #[must_use]
    fn length_squared(self) -> Self::Scalar;
    #[must_use]
    fn exp(self) -> Self;
    #[must_use]
    fn powf(self, n: Self::Scalar) -> Self;
    #[must_use]
    fn recip(self) -> Self;
    #[must_use]
    fn mul_add(self, a: Self, b: Self) -> Self;
}

#[allow(missing_docs)]
pub trait VectorFloat2: VectorFloat<2> {
    #[must_use]
    fn from_angle(angle: Self::Scalar) -> Self;
    #[must_use]
    fn rotate(self, other: Self) -> Self;
}

macro_rules! impl_base {
    ($scalar:ty [$dimensions:literal] => $glam_ty:ty, $mask:ty) => {
        impl Vector<$dimensions> for $glam_ty {
            type Scalar = $scalar;
            type Mask = $mask;

            forward_impl!($glam_ty => fn splat(scalar: $scalar) -> Self);

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

            forward_impl!($glam_ty => fn cmpeq(self, other: Self) -> Self::Mask);
            forward_impl!($glam_ty => fn cmpne(self, other: Self) -> Self::Mask);
            forward_impl!($glam_ty => fn cmplt(self, other: Self) -> Self::Mask);
            forward_impl!($glam_ty => fn cmple(self, other: Self) -> Self::Mask);
            forward_impl!($glam_ty => fn cmpgt(self, other: Self) -> Self::Mask);
            forward_impl!($glam_ty => fn cmpge(self, other: Self) -> Self::Mask);

            forward_impl!($glam_ty => fn clamp(self, min: Self, max: Self) -> Self);
            forward_impl!($glam_ty => fn min(self, other: Self) -> Self);
            forward_impl!($glam_ty => fn max(self, other: Self) -> Self);
            forward_impl!($glam_ty => fn min_element(self) -> $scalar);
            forward_impl!($glam_ty => fn max_element(self) -> $scalar);

            forward_impl!($glam_ty => fn select(mask: $mask, if_true: Self, if_false: Self) -> Self);
            forward_impl!($glam_ty => fn dot(self, other: Self) -> $scalar);
        }
    };
}

macro_rules! impl_base_float {
    ($dimensions:literal => $glam_ty:ty) => {
        impl VectorFloat<$dimensions> for $glam_ty {
            fn nan() -> Self {
                <$glam_ty>::NAN
            }

            forward_impl!($glam_ty => fn is_finite(self) -> bool);
            forward_impl!($glam_ty => fn is_nan(self) -> bool);
            forward_impl!($glam_ty => fn is_nan_mask(self) -> Self::Mask);
            forward_impl!($glam_ty => fn ceil(self) -> Self);
            forward_impl!($glam_ty => fn floor(self) -> Self);
            forward_impl!($glam_ty => fn round(self) -> Self);
            forward_impl!($glam_ty => fn normalize(self) -> Self);
            forward_impl!($glam_ty => fn normalize_or_zero(self) -> Self);
            forward_impl!($glam_ty => fn is_normalized(self) -> bool);
            forward_impl!($glam_ty => fn length(self) -> Self::Scalar);
            forward_impl!($glam_ty => fn length_squared(self) -> Self::Scalar);
            forward_impl!($glam_ty => fn exp(self) -> Self);
            forward_impl!($glam_ty => fn powf(self, n: Self::Scalar) -> Self);
            forward_impl!($glam_ty => fn recip(self) -> Self);
            forward_impl!($glam_ty => fn mul_add(self, a: Self, b: Self) -> Self);
        }

        impl crate::traits::Lerp<<$glam_ty as Vector<$dimensions>>::Scalar> for $glam_ty {
            forward_impl!($glam_ty => fn lerp(self, other: $glam_ty, t: <Self as Vector<$dimensions>>::Scalar) -> Self);
        }
    }
}

macro_rules! impl_abs {
    ($glam_ty:ty) => {
        impl crate::traits::Abs for $glam_ty {
            forward_impl!($glam_ty => fn abs(self) -> Self);
            forward_impl!($glam_ty => fn signum(self) -> Self);
        }
    };
}

impl_base!(f32[2] => glam::Vec2, glam::BVec2);
impl_base!(f32[3] => glam::Vec3, glam::BVec3A);
impl_base!(f32[4] => glam::Vec4, glam::BVec4A);
impl_base!(f64[2] => glam::DVec2, glam::BVec2);
impl_base!(f64[3] => glam::DVec3, glam::BVec3A);
impl_base!(f64[4] => glam::DVec4, glam::BVec4A);
impl_base!(i32[2] => glam::IVec2, glam::BVec2);
impl_base!(i32[3] => glam::IVec3, glam::BVec3A);
impl_base!(i32[4] => glam::IVec4, glam::BVec4A);
impl_base!(u32[2] => glam::UVec2, glam::BVec2);
impl_base!(u32[3] => glam::UVec3, glam::BVec3A);
impl_base!(u32[4] => glam::UVec4, glam::BVec4A);

impl_base_float!(2 => glam::Vec2);
impl_base_float!(3 => glam::Vec3);
impl_base_float!(4 => glam::Vec4);
impl_base_float!(2 => glam::DVec2);
impl_base_float!(3 => glam::DVec3);
impl_base_float!(4 => glam::DVec4);

impl VectorFloat2 for glam::Vec2 {
    fn from_angle(angle: Self::Scalar) -> Self {
        <glam::Vec2>::from_angle(angle)
    }

    fn rotate(self, other: Self) -> Self {
        <glam::Vec2>::rotate(self, other)
    }
}

impl VectorFloat2 for glam::DVec2 {
    fn from_angle(angle: Self::Scalar) -> Self {
        <glam::DVec2>::from_angle(angle)
    }

    fn rotate(self, other: Self) -> Self {
        <glam::DVec2>::rotate(self, other)
    }
}

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

        assert!(VectorFloat::is_nan(a));
        assert!(VectorFloat::is_nan(b));

        let a: glam::UVec4 = bytemuck::cast(a);
        let b: glam::UVec4 = bytemuck::cast(b);
        assert_eq!(a, b);
    }

    #[test]
    fn is_finite() {
        let a = glam::Vec4::NAN;
        assert!(!VectorFloat::<4>::is_finite(a));

        let a = glam::DVec4::NAN;
        assert!(!VectorFloat::<4>::is_finite(a));
    }
}

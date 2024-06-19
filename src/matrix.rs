//! Matrix types.
//!
//! Matrices do not have a [`Unit`](crate::Unit), because their values do not
//! necessarily have a clear logical meaning in the context of any particular
//! unit. Instead, they just use `Scalar`.

use core::ops::{Div, DivAssign, Mul, MulAssign};

use crate::{
    bindings::{Matrix, Matrix2 as SimdMatrix2, Matrix3 as SimdMatrix3, Matrix4 as SimdMatrix4},
    peel, peel_ref,
    prelude::*,
    scalar::FloatScalar,
    unit::FloatUnit,
    wrap, Point2, Scalar, Transparent, Unit, Vector2, Vector3, Vector4,
};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use bytemuck::{Pod, Zeroable};

/// 2x2 column-major matrix.
///
/// Bitwise compatible with [`glam::Mat2`] / [`glam::DMat2`].
///
/// Alignment: Always 16-byte aligned.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Matrix2<T: Scalar>(Vector4<T>);

unsafe impl<T: Scalar> Zeroable for Matrix2<T> {}
unsafe impl<T: Scalar> Pod for Matrix2<T> {}
// SAFETY: This is the fundamental guarantee of this crate.
unsafe impl<T: FloatScalar> Transparent for Matrix2<T> {
    type Wrapped = T::Mat2;
}

/// 3x3 column-major matrix.
///
/// Bitwise compatible with [`glam::Mat3`] / [`glam::DMat3`].
///
/// Alignment: Same as `T`.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct Matrix3<T: Scalar> {
    pub x_axis: Vector3<T>,
    pub y_axis: Vector3<T>,
    pub z_axis: Vector3<T>,
}

unsafe impl<T: Scalar> Zeroable for Matrix3<T> {}
unsafe impl<T: Scalar> Pod for Matrix3<T> {}
// SAFETY: This is the fundamental guarantee of this crate.
unsafe impl<T: FloatScalar> Transparent for Matrix3<T> {
    type Wrapped = T::Mat3;
}

/// 4x4 column-major matrix.
///
/// Bitwise compatible with [`glam::Mat4`] / [`glam::DMat4`].
///
/// Alignment: Always 16-byte aligned.
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct Matrix4<T: Scalar> {
    pub x_axis: Vector4<T>,
    pub y_axis: Vector4<T>,
    pub z_axis: Vector4<T>,
    pub w_axis: Vector4<T>,
}

unsafe impl<T: Scalar> Zeroable for Matrix4<T> {}
unsafe impl<T: Scalar> Pod for Matrix4<T> {}
// SAFETY: This is the fundamental guarantee of this crate.
unsafe impl<T: FloatScalar> Transparent for Matrix4<T> {
    type Wrapped = T::Mat4;
}

macro_rules! impl_matrix {
    ($base_type_name:ident < $dimensions:literal > => $mat_name:ident [ $axis_vector_ty:ident ]) => {
        impl<T> Default for $base_type_name<T>
        where
            T: FloatScalar,
        {
            #[inline]
            fn default() -> Self {
                Self::IDENTITY
            }
        }

        impl<T> core::fmt::Debug for $base_type_name<T>
        where
            T: FloatScalar,
        {
            fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
                let mut list = fmt.debug_list();
                for i in 0..$dimensions {
                    list.entry(&self.row(i).to_tuple());
                }
                list.finish()
            }
        }

        impl<T: Scalar> AsRef<[[T; $dimensions]; $dimensions]> for $base_type_name<T> {
            fn as_ref(&self) -> &[[T; $dimensions]; $dimensions] {
                bytemuck::cast_ref(self)
            }
        }

        impl<T: Scalar> AsMut<[[T; $dimensions]; $dimensions]> for $base_type_name<T> {
            fn as_mut(&mut self) -> &mut [[T; $dimensions]; $dimensions] {
                bytemuck::cast_mut(self)
            }
        }

        impl<T: Scalar> From<[[T; $dimensions]; $dimensions]> for $base_type_name<T> {
            fn from(value: [[T; $dimensions]; $dimensions]) -> Self {
                bytemuck::cast(value)
            }
        }
    };
}

impl<T> Matrix2<T>
where
    T: Scalar,
{
    /// All zeroes.
    pub const ZERO: Self = Self(Vector4::ZERO);

    /// Identity matrix
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix2::<f32>::IDENTITY;
    /// assert_eq!(matrix.row(0), vec2!(1.0, 0.0));
    /// assert_eq!(matrix.row(1), vec2!(0.0, 1.0));
    /// ```
    pub const IDENTITY: Self = Self(vec4!(T::ONE, T::ZERO, T::ZERO, T::ONE));
}

crate::impl_matrixlike::matrixlike!(Matrix2, 2);

impl<T> Matrix2<T>
where
    T: FloatScalar,
{
    /// All NaNs.
    pub const NAN: Self = Self(Vector4::NAN);
}

impl<T> Matrix3<T>
where
    T: FloatScalar,
{
    /// All zeroes.
    pub const ZERO: Self = Self {
        x_axis: Vector3::ZERO,
        y_axis: Vector3::ZERO,
        z_axis: Vector3::ZERO,
    };
    /// All NaNs.
    pub const NAN: Self = Self {
        x_axis: Vector3::NAN,
        y_axis: Vector3::NAN,
        z_axis: Vector3::NAN,
    };
    /// Identity matrix
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix3::<f32>::IDENTITY;
    /// assert_eq!(matrix.row(0), vec3!(1.0, 0.0, 0.0));
    /// assert_eq!(matrix.row(1), vec3!(0.0, 1.0, 0.0));
    /// assert_eq!(matrix.row(2), vec3!(0.0, 0.0, 1.0));
    /// ```
    pub const IDENTITY: Self = Self {
        x_axis: Vector3::X,
        y_axis: Vector3::Y,
        z_axis: Vector3::Z,
    };

    /// Transform 2D point.
    ///
    /// See [`glam::Mat3::transform_point2()`] or
    /// [`glam::DMat3::transform_point2()`] (depending on the scalar).
    ///
    /// #### Example
    ///
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix3::<f32>::from_angle(Angle::<f32>::from_degrees(90.0));
    /// let point = Point2::<f32> { x: 1.0, y: 0.0 };
    /// let rotated = matrix.transform_point(point);
    /// approx::assert_abs_diff_eq!(rotated, point!(0.0, 1.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_point<U: Unit<Scalar = T>>(&self, point: Point2<U>) -> Point2<U> {
        wrap(peel(*self).transform_point2(peel(point)))
    }

    /// Transform 2D vector.
    ///
    /// See [`glam::Mat3::transform_vector2()`] or
    /// [`glam::DMat3::transform_vector2()`] (depending on the scalar).
    #[inline]
    #[must_use]
    pub fn transform_vector<U: Unit<Scalar = T>>(&self, vector: Vector2<U>) -> Vector2<U> {
        wrap(peel(*self).transform_vector2(peel(vector)))
    }
}

crate::impl_matrixlike::matrixlike!(Matrix3, 3);

impl From<glam::Mat2> for Matrix2<f32> {
    fn from(mat: glam::Mat2) -> Self {
        wrap(mat)
    }
}

impl From<Matrix2<f32>> for glam::Mat2 {
    fn from(mat: Matrix2<f32>) -> Self {
        peel(mat)
    }
}

impl<T: Scalar> From<[T; 4]> for Matrix2<T> {
    fn from(value: [T; 4]) -> Matrix2<T> {
        Matrix2(value.into())
    }
}

impl<T: Scalar> From<Matrix2<T>> for [T; 4] {
    fn from(value: Matrix2<T>) -> [T; 4] {
        value.0.into()
    }
}

impl From<glam::DMat2> for Matrix2<f64> {
    fn from(mat: glam::DMat2) -> Self {
        wrap(mat)
    }
}

impl From<Matrix2<f64>> for glam::DMat2 {
    fn from(mat: Matrix2<f64>) -> Self {
        peel(mat)
    }
}

impl From<glam::Mat3A> for Matrix3<f32> {
    fn from(mat: glam::Mat3A) -> Self {
        wrap(mat.into())
    }
}

impl From<Matrix3<f32>> for glam::Mat3A {
    fn from(mat: Matrix3<f32>) -> Self {
        peel(mat).into()
    }
}

impl From<glam::Mat3> for Matrix3<f32> {
    fn from(mat: glam::Mat3) -> Self {
        wrap(mat)
    }
}

impl From<Matrix3<f32>> for glam::Mat3 {
    fn from(mat: Matrix3<f32>) -> Self {
        peel(mat)
    }
}

impl<T: FloatScalar> From<[T; 9]> for Matrix3<T> {
    fn from(value: [T; 9]) -> Matrix3<T> {
        Matrix3::from_cols_array(&value)
    }
}

impl<T: FloatScalar> From<Matrix3<T>> for [T; 9] {
    fn from(value: Matrix3<T>) -> [T; 9] {
        value.to_cols_array()
    }
}

impl From<glam::DMat3> for Matrix3<f64> {
    fn from(mat: glam::DMat3) -> Self {
        wrap(mat)
    }
}

impl From<Matrix3<f64>> for glam::DMat3 {
    fn from(mat: Matrix3<f64>) -> Self {
        peel(mat)
    }
}

impl From<glam::Mat4> for Matrix4<f32> {
    fn from(mat: glam::Mat4) -> Self {
        wrap(mat)
    }
}

impl From<Matrix4<f32>> for glam::Mat4 {
    fn from(mat: Matrix4<f32>) -> Self {
        peel(mat)
    }
}

impl<T: FloatScalar> From<[T; 16]> for Matrix4<T> {
    fn from(value: [T; 16]) -> Matrix4<T> {
        Matrix4::from_cols_array(&value)
    }
}

impl<T: FloatScalar> From<Matrix4<T>> for [T; 16] {
    fn from(value: Matrix4<T>) -> [T; 16] {
        value.to_cols_array()
    }
}

impl From<glam::DMat4> for Matrix4<f64> {
    fn from(mat: glam::DMat4) -> Self {
        wrap(mat)
    }
}

impl From<Matrix4<f64>> for glam::DMat4 {
    fn from(mat: Matrix4<f64>) -> Self {
        peel(mat)
    }
}

impl<T> Mul for Matrix2<T>
where
    T: FloatScalar,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        wrap(peel(self) * peel(rhs))
    }
}

impl<T> Mul<T> for Matrix2<T>
where
    T: FloatScalar,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: T) -> Self::Output {
        wrap(peel(self) * rhs)
    }
}

impl<T> MulAssign<T> for Matrix2<T>
where
    T: FloatScalar,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        *self = wrap(peel(*self) * rhs);
    }
}

impl<T> Mul<Vector2<T>> for Matrix2<T::Scalar>
where
    T: FloatScalar,
{
    type Output = Vector2<T>;

    #[inline(always)]
    fn mul(self, rhs: Vector2<T>) -> Vector2<T> {
        wrap(peel(self).mul_vec2(peel(rhs)))
    }
}

impl<T> Div<T> for Matrix2<T>
where
    T: FloatScalar,
{
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: T) -> Self::Output {
        wrap(peel(self) / rhs)
    }
}

impl<T> DivAssign<T> for Matrix2<T>
where
    T: FloatScalar,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: T) {
        *self = wrap(peel(*self) / rhs);
    }
}

impl<T> Mul for Matrix3<T>
where
    T: FloatScalar,
{
    type Output = Matrix3<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Self) -> Self::Output {
        wrap(peel(self) * peel(rhs))
    }
}

impl<T> Mul<T> for Matrix3<T>
where
    T: FloatScalar,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: T) -> Self::Output {
        wrap(peel(self) * rhs)
    }
}

impl<T> MulAssign<T> for Matrix3<T>
where
    T: FloatScalar,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T> Mul<Vector3<T>> for Matrix3<T::Scalar>
where
    T: FloatUnit,
{
    type Output = Vector3<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        wrap(peel(self).mul_vec3(peel(rhs)))
    }
}

impl<T> Div<T> for Matrix3<T>
where
    T: FloatScalar,
{
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: T) -> Self::Output {
        wrap(peel(self) / rhs)
    }
}

impl<T> DivAssign<T> for Matrix3<T>
where
    T: FloatScalar,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl<T> Mul<T> for Matrix4<T>
where
    T: FloatScalar,
{
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: T) -> Self::Output {
        wrap(peel(self) * rhs)
    }
}

impl<T> MulAssign<T> for Matrix4<T>
where
    T: FloatScalar,
{
    #[inline(always)]
    fn mul_assign(&mut self, rhs: T) {
        *self = *self * rhs;
    }
}

impl<T> Div<T> for Matrix4<T>
where
    T: FloatScalar,
{
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: T) -> Self::Output {
        wrap(peel(self) / rhs)
    }
}

impl<T> DivAssign<T> for Matrix4<T>
where
    T: FloatScalar,
{
    #[inline(always)]
    fn div_assign(&mut self, rhs: T) {
        *self = *self / rhs;
    }
}

impl<T> Mul<Vector4<T>> for Matrix4<T::Scalar>
where
    T: FloatUnit,
{
    type Output = Vector4<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Vector4<T>) -> Self::Output {
        wrap(peel(self).mul_vec4(peel(rhs)))
    }
}

impl<T> Matrix4<T>
where
    T: FloatScalar,
{
    /// All zeroes.
    pub const ZERO: Self = Self {
        x_axis: Vector4::ZERO,
        y_axis: Vector4::ZERO,
        z_axis: Vector4::ZERO,
        w_axis: Vector4::ZERO,
    };
    /// All NaNs.
    pub const NAN: Self = Self {
        x_axis: Vector4::NAN,
        y_axis: Vector4::NAN,
        z_axis: Vector4::NAN,
        w_axis: Vector4::NAN,
    };
    /// Identity matrix
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix4::<f32>::IDENTITY;
    /// assert_eq!(matrix.row(0), vec4!(1.0, 0.0, 0.0, 0.0));
    /// assert_eq!(matrix.row(1), vec4!(0.0, 1.0, 0.0, 0.0));
    /// assert_eq!(matrix.row(2), vec4!(0.0, 0.0, 1.0, 0.0));
    /// assert_eq!(matrix.row(3), vec4!(0.0, 0.0, 0.0, 1.0));
    /// ```
    pub const IDENTITY: Self = Self {
        x_axis: Vector4::X,
        y_axis: Vector4::Y,
        z_axis: Vector4::Z,
        w_axis: Vector4::W,
    };
}

crate::impl_matrixlike::matrixlike!(Matrix4, 4);

impl<T> Mul for Matrix4<T>
where
    T: FloatScalar,
{
    type Output = Matrix4<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Self) -> Self::Output {
        wrap(peel(self) * peel(rhs))
    }
}

impl_matrix!(Matrix2 <2> => Mat2 [Vector2]);
impl_matrix!(Matrix3 <3> => Mat3 [Vector3]);
impl_matrix!(Matrix4 <4> => Mat4 [Vector4]);

impl<T> AbsDiffEq for Matrix2<T>
where
    T: FloatScalar,
    T::Epsilon: Clone,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        peel_ref(self).abs_diff_eq(peel_ref(other), epsilon)
    }

    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        peel_ref(self).abs_diff_ne(peel_ref(other), epsilon)
    }
}

impl<T> RelativeEq for Matrix2<T>
where
    T: FloatScalar,
    T::Epsilon: Clone,
{
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        peel_ref(self).relative_eq(peel_ref(other), epsilon, max_relative)
    }

    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        peel_ref(self).relative_ne(peel_ref(other), epsilon, max_relative)
    }
}

impl<T: FloatScalar> UlpsEq for Matrix2<T> {
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        peel_ref(self).ulps_eq(peel_ref(other), epsilon, max_ulps)
    }

    fn ulps_ne(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        peel_ref(self).ulps_ne(peel_ref(other), epsilon, max_ulps)
    }
}

impl<T> AbsDiffEq for Matrix3<T>
where
    T: FloatScalar,
    T::Epsilon: Clone,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        peel(*self).abs_diff_eq(&peel(*other), epsilon)
    }

    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        peel(*self).abs_diff_ne(&peel(*other), epsilon)
    }
}

impl<T> RelativeEq for Matrix3<T>
where
    T: FloatScalar,
    T::Epsilon: Clone,
{
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        peel_ref(self).relative_eq(peel_ref(other), epsilon, max_relative)
    }

    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        peel_ref(self).relative_ne(peel_ref(other), epsilon, max_relative)
    }
}

impl<T: FloatScalar> UlpsEq for Matrix3<T> {
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        peel_ref(self).ulps_eq(peel_ref(other), epsilon, max_ulps)
    }

    fn ulps_ne(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        peel_ref(self).ulps_ne(peel_ref(other), epsilon, max_ulps)
    }
}

impl<T> AbsDiffEq for Matrix4<T>
where
    T: FloatScalar,
    T::Epsilon: Clone,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        peel_ref(self).abs_diff_eq(peel_ref(other), epsilon)
    }

    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        peel_ref(self).abs_diff_ne(peel_ref(other), epsilon)
    }
}

impl<T: FloatScalar> RelativeEq for Matrix4<T> {
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        peel_ref(self).relative_eq(peel_ref(other), epsilon, max_relative)
    }

    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        peel_ref(self).relative_ne(peel_ref(other), epsilon, max_relative)
    }
}

impl<T: FloatScalar> UlpsEq for Matrix4<T>
where
    T: FloatScalar,
    T::Epsilon: Clone,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        peel_ref(self).ulps_eq(peel_ref(other), epsilon, max_ulps)
    }

    fn ulps_ne(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        peel_ref(self).ulps_ne(peel_ref(other), epsilon, max_ulps)
    }
}

#[cfg(test)]
mod tests {
    use approx::{
        assert_abs_diff_eq, assert_abs_diff_ne, assert_relative_eq, assert_relative_ne,
        assert_ulps_eq, assert_ulps_ne,
    };

    use crate::{point3, vec2, vec3};

    use super::*;

    type Mat2 = Matrix2<f32>;
    type Mat3 = Matrix3<f32>;
    type Mat4 = Matrix4<f32>;
    type DMat2 = Matrix2<f64>;
    type DMat3 = Matrix3<f64>;
    type DMat4 = Matrix4<f64>;

    type Vec2 = Vector2<f32>;
    type Vec3 = Vector3<f32>;
    type Vec4 = Vector4<f32>;
    type DVec2 = Vector2<f64>;
    type DVec3 = Vector3<f64>;
    type DVec4 = Vector4<f64>;

    #[test]
    fn from_rows() {
        let from_rows = Mat2::from_rows(vec2!(1.0, 2.0), vec2!(3.0, 4.0));
        assert_eq!(from_rows, Mat2::from_cols(vec2!(1.0, 3.0), vec2!(2.0, 4.0)));
    }

    #[test]
    fn from_scale() {
        let m3 = Mat3::from_scale(Vec2::new(2.0, 3.0));
        let m4 = Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0));

        assert_eq!(
            m3,
            Mat3::from_rows(
                (2.0, 0.0, 0.0).into(),
                (0.0, 3.0, 0.0).into(),
                (0.0, 0.0, 1.0).into()
            )
        );
        assert_eq!(
            m4,
            Mat4::from_rows(
                (2.0, 0.0, 0.0, 0.0).into(),
                (0.0, 3.0, 0.0, 0.0).into(),
                (0.0, 0.0, 4.0, 0.0).into(),
                (0.0, 0.0, 0.0, 1.0).into()
            )
        );

        assert_eq!(m3 * Vec3::ONE, vec3!(2.0, 3.0, 1.0));

        let m3 = DMat3::from_scale(DVec2::new(2.0, 3.0));
        let m4 = DMat4::from_scale(DVec3::new(2.0, 3.0, 4.0));

        assert_eq!(
            m3,
            DMat3::from_rows(
                (2.0, 0.0, 0.0).into(),
                (0.0, 3.0, 0.0).into(),
                (0.0, 0.0, 1.0).into()
            )
        );
        assert_eq!(
            m4,
            DMat4::from_rows(
                (2.0, 0.0, 0.0, 0.0).into(),
                (0.0, 3.0, 0.0, 0.0).into(),
                (0.0, 0.0, 4.0, 0.0).into(),
                (0.0, 0.0, 0.0, 1.0).into()
            )
        );

        assert_eq!(m3 * DVec3::ONE, vec3!(2.0, 3.0, 1.0));
    }

    #[test]
    fn from_angle() {
        let m3 = Mat3::from_angle(Angle::from_degrees(90.0));
        let m4 = Mat4::from_axis_angle(Vec3::Z, Angle::from_degrees(90.0));

        assert_abs_diff_eq!(
            m3,
            Mat3::from_rows(
                (0.0, -1.0, 0.0).into(),
                (1.0, 0.0, 0.0).into(),
                (0.0, 0.0, 1.0).into()
            )
        );
        assert_abs_diff_eq!(
            m4,
            Mat4::from_rows(
                (0.0, -1.0, 0.0, 0.0).into(),
                (1.0, 0.0, 0.0, 0.0).into(),
                (0.0, 0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 0.0, 1.0).into()
            )
        );

        let m3 = DMat3::from_angle(Angle::from_degrees(90.0));
        let m4 = DMat4::from_axis_angle(Vector3::Z, Angle::from_degrees(90.0));

        assert_abs_diff_eq!(
            m3,
            DMat3::from_rows(
                (0.0, -1.0, 0.0).into(),
                (1.0, 0.0, 0.0).into(),
                (0.0, 0.0, 1.0).into()
            )
        );
        assert_abs_diff_eq!(
            m4,
            DMat4::from_rows(
                (0.0, -1.0, 0.0, 0.0).into(),
                (1.0, 0.0, 0.0, 0.0).into(),
                (0.0, 0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 0.0, 1.0).into()
            )
        );
    }

    #[test]
    fn from_translation() {
        let m3 = Mat3::from_translation(Vector2::new(2.0, 3.0));
        let m4 = Mat4::from_translation(Vector3::new(2.0, 3.0, 4.0));

        assert_eq!(
            m3,
            Mat3::from_rows(
                (1.0, 0.0, 2.0).into(),
                (0.0, 1.0, 3.0).into(),
                (0.0, 0.0, 1.0).into()
            )
        );
        assert_eq!(
            m4,
            Mat4::from_rows(
                (1.0, 0.0, 0.0, 2.0).into(),
                (0.0, 1.0, 0.0, 3.0).into(),
                (0.0, 0.0, 1.0, 4.0).into(),
                (0.0, 0.0, 0.0, 1.0).into(),
            )
        );

        let m3 = DMat3::from_translation(Vector2::new(2.0, 3.0));
        let m4 = DMat4::from_translation(Vector3::new(2.0, 3.0, 4.0));

        assert_eq!(
            m3,
            DMat3::from_rows(
                (1.0, 0.0, 2.0).into(),
                (0.0, 1.0, 3.0).into(),
                (0.0, 0.0, 1.0).into()
            )
        );
        assert_eq!(
            m4,
            DMat4::from_rows(
                (1.0, 0.0, 0.0, 2.0).into(),
                (0.0, 1.0, 0.0, 3.0).into(),
                (0.0, 0.0, 1.0, 4.0).into(),
                (0.0, 0.0, 0.0, 1.0).into(),
            )
        );
    }

    #[test]
    fn from_scale_angle_translation() {
        {
            let scale = Vec2::new(2.0, 3.0);
            let angle = Angle::from_degrees(90.0);
            let translation = Vec2::new(4.0, 5.0);

            assert_abs_diff_eq!(
                Mat3::from_scale_angle_translation(scale, angle, translation),
                Mat3::from_translation(translation)
                    * Mat3::from_angle(angle)
                    * Mat3::from_scale(scale),
                epsilon = 0.0001
            );
        }

        {
            let scale = DVec2::new(2.0, 3.0);
            let angle = Angle::from_degrees(90.0);
            let translation = DVec2::new(4.0, 5.0);

            assert_abs_diff_eq!(
                DMat3::from_scale_angle_translation(scale, angle, translation),
                DMat3::from_translation(translation)
                    * DMat3::from_angle(angle)
                    * DMat3::from_scale(scale),
                epsilon = 0.0001
            );
        }

        {
            let scale = Vec3::new(2.0, 3.0, 4.0);
            let axis = Vec3::Z;
            let angle = Angle::from_degrees(90.0);
            let translation = Vec3::new(5.0, 6.0, 7.0);

            assert_abs_diff_eq!(
                Mat4::from_scale_rotation_translation(
                    scale,
                    glam::Quat::from_axis_angle(peel(axis), peel(angle)),
                    translation
                ),
                Mat4::from_translation(translation)
                    * Mat4::from_axis_angle(axis, angle)
                    * Mat4::from_scale(scale),
                epsilon = 0.0001
            );
        }

        {
            let scale = DVec3::new(2.0, 3.0, 4.0);
            let axis = DVec3::Z;
            let angle = Angle::from_degrees(90.0);
            let translation = DVec3::new(5.0, 6.0, 7.0);

            assert_abs_diff_eq!(
                DMat4::from_scale_rotation_translation(
                    scale,
                    glam::DQuat::from_axis_angle(peel(axis), peel(angle)),
                    translation
                ),
                DMat4::from_translation(translation)
                    * DMat4::from_axis_angle(axis, angle)
                    * DMat4::from_scale(scale),
                epsilon = 0.0001
            );
        }
    }

    #[test]
    fn to_cols() {
        assert_eq!(Mat2::IDENTITY.to_cols_array(), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            Mat3::IDENTITY.to_cols_array(),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            Mat4::IDENTITY.to_cols_array_2d(),
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        );

        assert_eq!(DMat2::IDENTITY.to_cols_array(), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            DMat3::IDENTITY.to_cols_array(),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            DMat4::IDENTITY.to_cols_array_2d(),
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        );
    }

    #[test]
    fn to_rows() {
        assert_eq!(Mat2::IDENTITY.to_rows_array(), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            Mat3::IDENTITY.to_rows_array_2d(),
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        );
        assert_eq!(
            Mat4::IDENTITY.to_rows_array_2d(),
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        );

        assert_eq!(DMat2::IDENTITY.to_rows_array(), [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(
            DMat3::IDENTITY.to_rows_array(),
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            DMat4::IDENTITY.to_rows_array(),
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        );
    }

    #[test]
    fn from_cols() {
        assert_eq!(
            Mat2::from_cols((1.0, 0.0).into(), (0.0, 1.0).into()),
            Mat2::IDENTITY
        );
        assert_eq!(
            Mat3::from_cols(
                (1.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 1.0).into()
            ),
            Mat3::IDENTITY
        );
        assert_eq!(
            Mat4::from_cols(
                (1.0, 0.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0, 0.0).into(),
                (0.0, 0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 0.0, 1.0).into()
            ),
            Mat4::IDENTITY
        );

        assert_eq!(
            DMat2::from_cols((1.0, 0.0).into(), (0.0, 1.0).into()),
            DMat2::IDENTITY
        );
        assert_eq!(
            DMat3::from_cols(
                (1.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 1.0).into()
            ),
            DMat3::IDENTITY
        );
        assert_eq!(
            DMat4::from_cols(
                (1.0, 0.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0, 0.0).into(),
                (0.0, 0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 0.0, 1.0).into()
            ),
            DMat4::IDENTITY
        );
    }

    #[test]
    fn equality() {
        let m2 = Mat2::IDENTITY;
        assert_eq!(m2, m2);
        assert_abs_diff_eq!(m2, m2);
        assert_relative_eq!(m2, m2);
        assert_ulps_eq!(m2, m2);
        assert_ne!(m2, Mat2::ZERO);
        assert_abs_diff_ne!(m2, Mat2::ZERO);
        assert_relative_ne!(m2, Mat2::ZERO);
        assert_ulps_ne!(m2, Mat2::ZERO);

        let m3 = Mat3::IDENTITY;
        assert_eq!(m3, m3);
        assert_abs_diff_eq!(m3, m3);
        assert_relative_eq!(m3, m3);
        assert_ulps_eq!(m3, m3);
        assert_ne!(m3, Mat3::ZERO);
        assert_abs_diff_ne!(m3, Mat3::ZERO);
        assert_relative_ne!(m3, Mat3::ZERO);
        assert_ulps_ne!(m3, Mat3::ZERO);

        let m4 = Mat4::IDENTITY;
        assert_eq!(m4, m4);
        assert_abs_diff_eq!(m4, m4);
        assert_relative_eq!(m4, m4);
        assert_ulps_eq!(m4, m4);
        assert_ne!(m4, Mat4::ZERO);
        assert_abs_diff_ne!(m4, Mat4::ZERO);
        assert_relative_ne!(m4, Mat4::ZERO);
        assert_ulps_ne!(m4, Mat4::ZERO);
    }

    #[test]
    fn transform() {
        let v3 = Vec3::new(1.0, 2.0, 3.0);
        let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);

        let mat3 = Mat3::from_scale((2.0, 2.0).into());
        let mat4 = Mat4::from_scale((2.0, 2.0, 2.0).into());

        assert_eq!(mat3 * v3, Vec3::new(2.0, 4.0, 3.0));
        assert_eq!(mat4 * v4, Vec4::new(2.0, 4.0, 6.0, 4.0));
    }

    #[test]
    fn transform_f64() {
        let v2 = DVec2::new(1.0, 2.0);
        let v3 = DVec3::new(1.0, 2.0, 3.0);
        let v4 = DVec4::new(1.0, 2.0, 3.0, 4.0);

        let mat3 = DMat3::from_scale((2.0, 2.0).into());
        let mat4 = DMat4::from_scale((2.0, 2.0, 2.0).into());

        assert_eq!(mat3 * v3, mat3.transform_vector(v2).extend(3.0));
        assert_eq!(mat4 * v4, DVec4::new(2.0, 4.0, 6.0, 4.0));
    }

    #[test]
    fn determinant() {
        assert_eq!(Mat2::IDENTITY.determinant(), 1.0);
        assert_eq!(Mat3::IDENTITY.determinant(), 1.0);
        assert_eq!(Mat4::IDENTITY.determinant(), 1.0);
        assert_eq!(DMat2::IDENTITY.determinant(), 1.0);
        assert_eq!(DMat3::IDENTITY.determinant(), 1.0);
        assert_eq!(DMat4::IDENTITY.determinant(), 1.0);
    }

    #[test]
    fn nan() {
        let m2 = Mat2::NAN;
        assert!(m2.is_nan());
        assert!(m2.col(0).is_nan());
        assert!(m2.col(1).is_nan());

        let m3 = Mat3::NAN;
        assert!(m2.is_nan());
        assert!(m3.col(0).is_nan());
        assert!(m3.col(1).is_nan());
        assert!(m3.col(2).is_nan());

        let m4 = Mat4::NAN;
        assert!(m4.is_nan());
        assert!(m4.col(0).is_nan());
        assert!(m4.col(1).is_nan());
        assert!(m4.col(2).is_nan());
        assert!(m4.col(3).is_nan());

        let m2 = DMat2::NAN;
        assert!(m2.is_nan());
        assert!(m2.col(0).is_nan());
        assert!(m2.col(1).is_nan());

        let m3 = DMat3::NAN;
        assert!(m2.is_nan());
        assert!(m3.col(0).is_nan());
        assert!(m3.col(1).is_nan());
        assert!(m3.col(2).is_nan());

        let m4 = DMat4::NAN;
        assert!(m4.is_nan());
        assert!(m4.col(0).is_nan());
        assert!(m4.col(1).is_nan());
        assert!(m4.col(2).is_nan());
        assert!(m4.col(3).is_nan());
    }

    #[test]
    fn is_finite() {
        assert!(Mat2::IDENTITY.is_finite());
        assert!(Mat3::IDENTITY.is_finite());
        assert!(Mat4::IDENTITY.is_finite());
        assert!(DMat2::IDENTITY.is_finite());
        assert!(DMat3::IDENTITY.is_finite());
        assert!(DMat4::IDENTITY.is_finite());

        assert!(!Mat2::NAN.is_finite());
        assert!(!Mat3::NAN.is_finite());
        assert!(!Mat4::NAN.is_finite());
        assert!(!DMat2::NAN.is_finite());
        assert!(!DMat3::NAN.is_finite());
        assert!(!DMat4::NAN.is_finite());

        assert!(!DMat4::from_rows(
            (1.0, 1.0, 1.0, 1.0).into(),
            (1.0, 1.0, 1.0, 1.0).into(),
            (1.0, 1.0, 1.0, 1.0).into(),
            (1.0, 1.0, 1.0, f64::NAN).into(),
        )
        .is_finite());

        assert!(!DMat4::from_rows(
            (1.0, 1.0, 1.0, 1.0).into(),
            (1.0, 1.0, 1.0, 1.0).into(),
            (1.0, 1.0, 1.0, 1.0).into(),
            (1.0, 1.0, 1.0, f64::INFINITY).into(),
        )
        .is_finite());

        assert!(!DMat4::from_rows(
            (1.0, 1.0, 1.0, 1.0).into(),
            (1.0, 1.0, 1.0, 1.0).into(),
            (1.0, 1.0, 1.0, 1.0).into(),
            (1.0, 1.0, 1.0, f64::NEG_INFINITY).into(),
        )
        .is_finite());
    }

    #[test]
    fn is_invertible() {
        assert!(Mat2::IDENTITY.is_invertible());
        assert!(Mat3::IDENTITY.is_invertible());
        assert!(Mat4::IDENTITY.is_invertible());
        assert!(!Mat2::ZERO.is_invertible());
        assert!(!Mat3::ZERO.is_invertible());
        assert!(!Mat4::ZERO.is_invertible());
        assert!(!Mat2::NAN.is_invertible());
        assert!(!Mat3::NAN.is_invertible());
        assert!(!Mat4::NAN.is_invertible());
        assert_eq!(Mat2::IDENTITY.try_inverse(), Some(Mat2::IDENTITY));
        assert_eq!(Mat3::IDENTITY.try_inverse(), Some(Mat3::IDENTITY));
        assert_eq!(Mat4::IDENTITY.try_inverse(), Some(Mat4::IDENTITY));

        assert!(DMat2::IDENTITY.is_invertible());
        assert!(DMat3::IDENTITY.is_invertible());
        assert!(DMat4::IDENTITY.is_invertible());
        assert!(!DMat2::ZERO.is_invertible());
        assert!(!DMat3::ZERO.is_invertible());
        assert!(!DMat4::ZERO.is_invertible());
        assert!(!DMat2::NAN.is_invertible());
        assert!(!DMat3::NAN.is_invertible());
        assert!(!DMat4::NAN.is_invertible());
        assert_eq!(DMat2::IDENTITY.try_inverse(), Some(DMat2::IDENTITY));
        assert_eq!(DMat3::IDENTITY.try_inverse(), Some(DMat3::IDENTITY));
        assert_eq!(DMat4::IDENTITY.try_inverse(), Some(DMat4::IDENTITY));

        {
            assert!(!Mat2::zeroed().is_invertible());
            assert!(!Mat2::from_cols(Vec2::ZERO, Vec2::ONE).is_invertible());
            assert!(!Mat2::from_cols(Vec2::ONE, Vec2::ZERO).is_invertible());
            assert!(Mat2::IDENTITY.is_invertible());
        }
        {
            assert!(!Mat3::zeroed().is_invertible());
            assert!(!Mat3::from_cols(Vec3::ZERO, Vec3::ONE, Vec3::ONE).is_invertible());
            assert!(!Mat3::from_cols(Vec3::ONE, Vec3::ZERO, Vec3::ONE).is_invertible());
            assert!(!Mat3::from_cols(Vec3::ONE, Vec3::ONE, Vec3::ZERO).is_invertible());
            assert!(Mat3::IDENTITY.is_invertible());
        }
        {
            assert!(!Mat4::zeroed().is_invertible());
            assert!(!Mat4::from_cols(Vec4::ZERO, Vec4::ONE, Vec4::ONE, Vec4::ONE).is_invertible());
            assert!(!Mat4::from_cols(Vec4::ONE, Vec4::ZERO, Vec4::ONE, Vec4::ONE).is_invertible());
            assert!(!Mat4::from_cols(Vec4::ONE, Vec4::ONE, Vec4::ZERO, Vec4::ONE).is_invertible());
            assert!(!Mat4::from_cols(Vec4::ONE, Vec4::ONE, Vec4::ONE, Vec4::ZERO).is_invertible());
            assert!(Mat4::IDENTITY.is_invertible());
        }

        assert!(Mat2::ZERO.try_inverse().is_none());
        assert!(Mat3::ZERO.try_inverse().is_none());
        assert!(Mat4::ZERO.try_inverse().is_none());
    }

    #[test]
    fn mat3a() {
        let mat3 = Mat3::from_cols(
            (1.0, 2.0, 3.0).into(),
            (4.0, 5.0, 6.0).into(),
            (7.0, 8.0, 9.0).into(),
        );
        let mat3a: glam::Mat3A = mat3.into();
        assert_eq!(
            mat3a,
            glam::Mat3A::from_cols(
                (1.0, 2.0, 3.0).into(),
                (4.0, 5.0, 6.0).into(),
                (7.0, 8.0, 9.0).into(),
            )
        );
        let mat3_2: Matrix3<f32> = mat3a.into();
        assert_eq!(mat3_2, mat3);
    }

    #[test]
    fn mat4_constructors() {
        assert_eq!(
            peel(Matrix4::<f32>::look_at_lh(
                point3!(1.0, 2.0, 3.0),
                point3!(4.0, 5.0, 6.0),
                vec3!(1.0, 0.0, 0.0)
            )),
            glam::Mat4::look_at_lh(
                (1.0, 2.0, 3.0).into(),
                (4.0, 5.0, 6.0).into(),
                (1.0, 0.0, 0.0).into()
            )
        );
        assert_eq!(
            peel(Matrix4::<f32>::look_at_rh(
                point3!(1.0, 2.0, 3.0),
                point3!(4.0, 5.0, 6.0),
                vec3!(1.0, 0.0, 0.0)
            )),
            glam::Mat4::look_at_rh(
                (1.0, 2.0, 3.0).into(),
                (4.0, 5.0, 6.0).into(),
                (1.0, 0.0, 0.0).into()
            )
        );
        assert_eq!(
            Matrix4::<f32>::perspective_rh_gl(Angle::new(1.0), 2.0, 3.0, 4.0),
            wrap(glam::Mat4::perspective_rh_gl(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            Matrix4::<f32>::perspective_lh(Angle::new(1.0), 2.0, 3.0, 4.0),
            wrap(glam::Mat4::perspective_lh(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            Matrix4::<f32>::perspective_rh(Angle::new(1.0), 2.0, 3.0, 4.0),
            wrap(glam::Mat4::perspective_rh(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            Matrix4::<f32>::perspective_infinite_lh(Angle::new(1.0), 2.0, 3.0),
            wrap(glam::Mat4::perspective_infinite_lh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f32>::perspective_infinite_reverse_lh(Angle::new(1.0), 2.0, 3.0),
            wrap(glam::Mat4::perspective_infinite_reverse_lh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f32>::perspective_infinite_rh(Angle::new(1.0), 2.0, 3.0),
            wrap(glam::Mat4::perspective_infinite_rh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f32>::perspective_infinite_reverse_rh(Angle::new(1.0), 2.0, 3.0),
            wrap(glam::Mat4::perspective_infinite_reverse_rh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f32>::orthographic_rh_gl(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            wrap(glam::Mat4::orthographic_rh_gl(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        );
        assert_eq!(
            Matrix4::<f32>::orthographic_lh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            wrap(glam::Mat4::orthographic_lh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        );
        assert_eq!(
            Matrix4::<f32>::orthographic_rh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            wrap(glam::Mat4::orthographic_rh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        );
    }

    #[test]
    fn dmat4_constructors() {
        assert_eq!(
            peel(Matrix4::<f64>::look_at_lh(
                point3!(1.0, 2.0, 3.0),
                point3!(4.0, 5.0, 6.0),
                vec3!(1.0, 0.0, 0.0)
            )),
            glam::DMat4::look_at_lh(
                (1.0, 2.0, 3.0).into(),
                (4.0, 5.0, 6.0).into(),
                (1.0, 0.0, 0.0).into()
            )
        );
        assert_eq!(
            peel(Matrix4::<f64>::look_at_rh(
                point3!(1.0, 2.0, 3.0),
                point3!(4.0, 5.0, 6.0),
                vec3!(1.0, 0.0, 0.0)
            )),
            glam::DMat4::look_at_rh(
                (1.0, 2.0, 3.0).into(),
                (4.0, 5.0, 6.0).into(),
                (1.0, 0.0, 0.0).into()
            )
        );
        assert_eq!(
            Matrix4::<f64>::perspective_rh_gl(Angle::new(1.0), 2.0, 3.0, 4.0),
            wrap(glam::DMat4::perspective_rh_gl(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            Matrix4::<f64>::perspective_lh(Angle::new(1.0), 2.0, 3.0, 4.0),
            wrap(glam::DMat4::perspective_lh(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            Matrix4::<f64>::perspective_rh(Angle::new(1.0), 2.0, 3.0, 4.0),
            wrap(glam::DMat4::perspective_rh(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            Matrix4::<f64>::perspective_infinite_lh(Angle::new(1.0), 2.0, 3.0),
            wrap(glam::DMat4::perspective_infinite_lh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f64>::perspective_infinite_reverse_lh(Angle::new(1.0), 2.0, 3.0),
            wrap(glam::DMat4::perspective_infinite_reverse_lh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f64>::perspective_infinite_rh(Angle::new(1.0), 2.0, 3.0),
            wrap(glam::DMat4::perspective_infinite_rh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f64>::perspective_infinite_reverse_rh(Angle::new(1.0), 2.0, 3.0),
            wrap(glam::DMat4::perspective_infinite_reverse_rh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f64>::orthographic_rh_gl(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            wrap(glam::DMat4::orthographic_rh_gl(
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0
            ))
        );
        assert_eq!(
            Matrix4::<f64>::orthographic_lh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            wrap(glam::DMat4::orthographic_lh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        );
        assert_eq!(
            Matrix4::<f64>::orthographic_rh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            wrap(glam::DMat4::orthographic_rh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn debug_print() {
        extern crate alloc;

        let m4 = Mat4::IDENTITY;

        let s = alloc::format!("{m4:?}");
        assert_eq!(s, "[(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)]");
    }

    #[test]
    fn from_into() {
        assert_eq!(
            Matrix2::from(glam::Mat2::from(Matrix2::IDENTITY)),
            Matrix2::IDENTITY
        );
        assert_eq!(
            Matrix2::from(glam::DMat2::from(Matrix2::IDENTITY)),
            Matrix2::IDENTITY
        );
        assert_eq!(
            Matrix3::from(glam::Mat3::from(Matrix3::IDENTITY)),
            Matrix3::IDENTITY
        );
        assert_eq!(
            Matrix3::from(glam::Mat3A::from(Matrix3::IDENTITY)),
            Matrix3::IDENTITY
        );
        assert_eq!(
            Matrix3::from(glam::DMat3::from(Matrix3::IDENTITY)),
            Matrix3::IDENTITY
        );
        assert_eq!(
            Matrix4::from(glam::Mat4::from(Matrix4::IDENTITY)),
            Matrix4::IDENTITY
        );
        assert_eq!(
            Matrix4::from(glam::DMat4::from(Matrix4::IDENTITY)),
            Matrix4::IDENTITY
        );
    }

    #[test]
    fn as_ref_column_major() {
        let mut m2 = Matrix2::from_cols(vec2!(1.0, 2.0), vec2!(3.0, 4.0));
        assert_eq!(m2, Matrix2::from([[1.0, 2.0], [3.0, 4.0]]));
        let r: &[[f32; 2]; 2] = m2.as_ref();
        assert_eq!(*r, [[1.0, 2.0], [3.0, 4.0]]);
        let m: &mut [[f32; 2]; 2] = m2.as_mut();
        m[1][1] = 5.0;
        assert_eq!(m2, Matrix2::from_cols(vec2!(1.0, 2.0), vec2!(3.0, 5.0)));

        let mut m3 = Matrix3::from_cols(
            vec3!(1.0, 2.0, 3.0),
            vec3!(4.0, 5.0, 6.0),
            vec3!(7.0, 8.0, 9.0),
        );
        assert_eq!(
            m3,
            Matrix3::from([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        );
        let r: &[[f32; 3]; 3] = m3.as_ref();
        assert_eq!(*r, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let m: &mut [[f32; 3]; 3] = m3.as_mut();
        m[2][2] = 10.0;
        assert_eq!(
            m3,
            Matrix3::from_cols(
                vec3!(1.0, 2.0, 3.0),
                vec3!(4.0, 5.0, 6.0),
                vec3!(7.0, 8.0, 10.0)
            )
        );
    }

    #[test]
    fn mul() {
        let m2 = Matrix2::from_cols(vec2!(1.0, 2.0), vec2!(3.0, 4.0));
        let m3 = Matrix3::from_cols(
            vec3!(1.0, 2.0, 3.0),
            vec3!(4.0, 5.0, 6.0),
            vec3!(7.0, 8.0, 9.0),
        );
        let m4 = Matrix4::from_cols(
            vec4!(1.0, 2.0, 3.0, 4.0),
            vec4!(5.0, 6.0, 7.0, 8.0),
            vec4!(9.0, 10.0, 11.0, 12.0),
            vec4!(13.0, 14.0, 15.0, 16.0),
        );

        assert_eq!(
            m2 * 2.0,
            Matrix2::from_cols(vec2!(2.0, 4.0), vec2!(6.0, 8.0))
        );
        assert_eq!(
            m3 * 2.0,
            Matrix3::from_cols(
                vec3!(2.0, 4.0, 6.0),
                vec3!(8.0, 10.0, 12.0),
                vec3!(14.0, 16.0, 18.0)
            )
        );
        assert_eq!(
            m4 * 2.0,
            Matrix4::from_cols(
                vec4!(2.0, 4.0, 6.0, 8.0),
                vec4!(10.0, 12.0, 14.0, 16.0),
                vec4!(18.0, 20.0, 22.0, 24.0),
                vec4!(26.0, 28.0, 30.0, 32.0)
            )
        );

        let mut m2_copy = m2;
        let mut m3_copy = m3;
        let mut m4_copy = m4;

        m2_copy *= 2.0;
        m3_copy *= 2.0;
        m4_copy *= 2.0;

        assert_eq!(m2_copy, m2 * 2.0);
        assert_eq!(m3_copy, m3 * 2.0);
        assert_eq!(m4_copy, m4 * 2.0);

        assert_eq!(crate::peel(m2 * m2), crate::peel(m2) * crate::peel(m2));
        assert_eq!(crate::peel(m3 * m3), crate::peel(m3) * crate::peel(m3));
        assert_eq!(crate::peel(m4 * m4), crate::peel(m4) * crate::peel(m4));

        assert_eq!(
            crate::peel(m2 * Vector2::<f32>::new(2.0, 2.0)),
            crate::peel(m2) * crate::peel(Vector2::<f32>::new(2.0, 2.0))
        );
    }

    #[test]
    fn div() {
        let m2 = Matrix2::from_cols(vec2!(2.0, 4.0), vec2!(6.0, 8.0));
        let m3 = Matrix3::from_cols(
            vec3!(2.0, 4.0, 6.0),
            vec3!(8.0, 10.0, 12.0),
            vec3!(14.0, 16.0, 18.0),
        );
        let m4 = Matrix4::from_cols(
            vec4!(2.0, 4.0, 6.0, 8.0),
            vec4!(10.0, 12.0, 14.0, 16.0),
            vec4!(18.0, 20.0, 22.0, 24.0),
            vec4!(26.0, 28.0, 30.0, 32.0),
        );

        assert_eq!(
            m2 / 2.0,
            Matrix2::from_cols(vec2!(1.0, 2.0), vec2!(3.0, 4.0))
        );
        assert_eq!(
            m3 / 2.0,
            Matrix3::from_cols(
                vec3!(1.0, 2.0, 3.0),
                vec3!(4.0, 5.0, 6.0),
                vec3!(7.0, 8.0, 9.0)
            )
        );
        assert_eq!(
            m4 / 2.0,
            Matrix4::from_cols(
                vec4!(1.0, 2.0, 3.0, 4.0),
                vec4!(5.0, 6.0, 7.0, 8.0),
                vec4!(9.0, 10.0, 11.0, 12.0),
                vec4!(13.0, 14.0, 15.0, 16.0)
            )
        );

        let mut m2_copy = m2;
        let mut m3_copy = m3;
        let mut m4_copy = m4;

        m2_copy /= 2.0;
        m3_copy /= 2.0;
        m4_copy /= 2.0;

        assert_eq!(m2_copy, m2 / 2.0);
        assert_eq!(m3_copy, m3 / 2.0);
        assert_eq!(m4_copy, m4 / 2.0);
    }

    #[test]
    fn abs() {
        let m2 = Mat2::from_diagonal(vec2!(1.0, -1.0)).abs();
        assert_eq!(m2, Mat2::IDENTITY);
        let m3 = Mat3::from_diagonal(vec3!(1.0, 1.0, -1.0)).abs();
        assert_eq!(m3, Mat3::IDENTITY);
        let m4 = Mat4::from_diagonal(vec4!(1.0, 1.0, -1.0, -1.0)).abs();
        assert_eq!(m4, Mat4::IDENTITY);
    }
}

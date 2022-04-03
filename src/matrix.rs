//! Matrix types.
//!
//! Matrices do not have a [`Unit`](crate::traits::Unit), because their values
//! do not necessarily have a clear logical meaning in the context of any
//! particular unit.
//!
//! Instead, they are based on fundamental floating-point
//! [primitive](crate::traits::Primitive) scalars (`f32` or `f64`).

use core::ops::Mul;

use crate::{
    traits::{PrimitiveMatrices, SimdMatrix, SimdMatrix2, SimdMatrix3, SimdMatrix4},
    Angle, Point2, Point3, Point4, Vector2, Vector3, Vector4,
};
use approx::{AbsDiffEq, RelativeEq};
use bytemuck::{cast, cast_mut, cast_ref, Pod, Zeroable};

/// 2x2 column-major matrix.
///
/// Bitwise compatible with [`glam::Mat2`] / [`glam::DMat2`].
///
/// Alignment: Always 16-byte aligned.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct Matrix2<T> {
    pub m11: T,
    pub m12: T,
    pub m21: T,
    pub m22: T,
}

unsafe impl<T: Zeroable> Zeroable for Matrix2<T> {}
unsafe impl<T: Pod> Pod for Matrix2<T> {}

/// 3x3 column-major matrix.
///
/// Bitwise compatible with [`glam::Mat3`] / [`glam::DMat3`].
///
/// Alignment: Same as `T`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct Matrix3<T> {
    pub m11: T,
    pub m12: T,
    pub m13: T,
    pub m21: T,
    pub m22: T,
    pub m23: T,
    pub m31: T,
    pub m32: T,
    pub m33: T,
}

unsafe impl<T: Zeroable> Zeroable for Matrix3<T> {}
unsafe impl<T: Pod> Pod for Matrix3<T> {}

/// 4x4 column-major matrix.
///
/// Bitwise compatible with [`glam::Mat4`] / [`glam::DMat4`].
///
/// Alignment: Always 16-byte aligned.
#[repr(C, align(16))]
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct Matrix4<T> {
    pub m11: T,
    pub m12: T,
    pub m13: T,
    pub m14: T,
    pub m21: T,
    pub m22: T,
    pub m23: T,
    pub m24: T,
    pub m31: T,
    pub m32: T,
    pub m33: T,
    pub m34: T,
    pub m41: T,
    pub m42: T,
    pub m43: T,
    pub m44: T,
}

unsafe impl<T: Zeroable> Zeroable for Matrix4<T> {}
unsafe impl<T: Pod> Pod for Matrix4<T> {}

macro_rules! impl_matrix {
    ($base_type_name:ident < $dimensions:literal > => $mat_name:ident [ $axis_vector_ty:ident, $transform_vector_ty:ident ]) => {
        impl<T> $base_type_name<T>
        where
            T: PrimitiveMatrices,
        {
            #[doc = "Get the underlying `glam` matrix."]
            #[inline(always)]
            #[must_use]
            pub fn to_raw(self) -> T::$mat_name {
                cast(self)
            }

            #[doc = "Create from underlying `glam` matrix."]
            #[inline(always)]
            #[must_use]
            pub fn from_raw(raw: T::$mat_name) -> Self {
                cast(raw)
            }

            #[doc = "Cast to `glam` matrix."]
            #[inline(always)]
            #[must_use]
            pub fn as_raw(&self) -> &T::$mat_name {
                cast_ref(self)
            }

            #[doc = "Cast to `glam` matrix."]
            #[inline(always)]
            #[must_use]
            pub fn as_raw_mut(&mut self) -> &mut T::$mat_name {
                cast_mut(self)
            }

            #[doc = "Get column vector at `index`."]
            #[inline]
            #[must_use]
            pub fn col(&self, index: usize) -> $axis_vector_ty<T> {
                $axis_vector_ty::from_raw(self.as_raw().col(index))
            }

            #[doc = "Get row vector at `index`."]
            #[inline]
            #[must_use]
            pub fn row(&self, index: usize) -> $axis_vector_ty<T> {
                $axis_vector_ty::from_raw(self.as_raw().row(index))
            }

            #[doc = "Get column vectors."]
            #[inline]
            #[must_use]
            pub fn cols(&self) -> [$axis_vector_ty<T>; $dimensions] {
                bytemuck::cast(self.as_raw().to_cols())
            }

            #[doc = "Get row vectors."]
            #[inline]
            #[must_use]
            pub fn rows(&self) -> [$axis_vector_ty<T>; $dimensions] {
                bytemuck::cast(self.as_raw().to_rows())
            }
        }

        impl<T> Default for $base_type_name<T>
        where
            T: PrimitiveMatrices,
        {
            #[inline(always)]
            fn default() -> Self {
                Self::identity()
            }
        }
    };
}

impl<T> Matrix2<T>
where
    T: PrimitiveMatrices,
{
    /// Identity matrix
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix2::<f32>::identity();
    /// assert_eq!(matrix.row(0), (1.0, 0.0));
    /// assert_eq!(matrix.row(1), (0.0, 1.0));
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn identity() -> Self {
        Self::from_raw(T::Mat2::identity())
    }

    /// Scaling matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix2::<f32>::from_scale((2.0, 3.0).into());
    /// assert_eq!(matrix.row(0), (2.0, 0.0));
    /// assert_eq!(matrix.row(1), (0.0, 3.0));
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn from_scale(scale: Vector2<T>) -> Self {
        Self::from_raw(T::Mat2::from_scale(scale.to_raw()))
    }

    /// Rotation matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    /// let theta = Angle::from_degrees(90.0);
    /// let matrix = Matrix2::<f32>::from_angle(theta);
    /// assert_abs_diff_eq!(matrix.row(0), (0.0, -1.0));
    /// assert_abs_diff_eq!(matrix.row(1), (1.0,  0.0));
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn from_angle(angle: Angle<T>) -> Self {
        Self::from_raw(T::Mat2::from_angle(angle))
    }

    /// Transform 2D point.
    ///
    /// See [`glam::Mat3::transform_point2()`] or
    /// [`glam::DMat3::transform_point2()`] (depending on the scalar).
    ///
    /// #### Example
    ///
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix2::<f32>::from_angle(Angle::<f32>::from_degrees(90.0));
    /// let point = Point2::<f32> { x: 1.0, y: 0.0 };
    /// let rotated = matrix.transform_point(point);
    /// approx::assert_abs_diff_eq!(rotated, (0.0, 1.0));
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn transform_point(&self, point: Point2<T>) -> Point2<T> {
        Point2::from_raw(self.as_raw().transform_point(point.to_raw()))
    }

    /// Transform 2D vector.
    #[inline(always)]
    #[must_use]
    pub fn transform_vector(&self, vector: Vector2<T>) -> Vector2<T> {
        Vector2::from_raw(self.as_raw().transform_vector(vector.to_raw()))
    }

    /// Invert the matrix.
    ///
    /// See [`glam::Mat2::inverse()`] and [`glam::DMat2::inverse()`].
    #[inline(always)]
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self::from_raw(self.as_raw().inverse())
    }
}

impl<T> Matrix3<T>
where
    T: PrimitiveMatrices,
{
    /// Identity matrix
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix3::<f32>::identity();
    /// assert_eq!(matrix.row(0), (1.0, 0.0, 0.0));
    /// assert_eq!(matrix.row(1), (0.0, 1.0, 0.0));
    /// assert_eq!(matrix.row(2), (0.0, 0.0, 1.0));
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn identity() -> Self {
        Self::from_raw(T::Mat3::identity())
    }

    /// Scaling matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix3::<f32>::from_scale((2.0, 3.0).into());
    /// assert_eq!(matrix.row(0), (2.0, 0.0, 0.0));
    /// assert_eq!(matrix.row(1), (0.0, 3.0, 0.0));
    /// assert_eq!(matrix.row(2), (0.0, 0.0, 1.0));
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn from_scale(scale: Vector2<T>) -> Self {
        Self::from_raw(T::Mat3::from_scale(scale.to_raw()))
    }

    /// Rotation matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    /// let theta = Angle::from_degrees(90.0);
    /// let matrix = Matrix3::<f32>::from_angle(theta);
    /// assert_abs_diff_eq!(matrix.row(0), (0.0, -1.0, 0.0));
    /// assert_abs_diff_eq!(matrix.row(1), (1.0,  0.0, 0.0));
    /// assert_abs_diff_eq!(matrix.row(2), (0.0,  0.0, 1.0));
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn from_angle(angle: Angle<T>) -> Self {
        Self::from_raw(T::Mat3::from_angle(angle))
    }

    /// Translation matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    /// let matrix = Matrix3::<f32>::from_translation((10.0, 20.0).into());
    /// assert_abs_diff_eq!(matrix.row(0), (1.0, 0.0, 10.0));
    /// assert_abs_diff_eq!(matrix.row(1), (0.0, 1.0, 20.0));
    /// assert_abs_diff_eq!(matrix.row(2), (0.0, 0.0,  1.0));
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn from_translation(translation: Vector2<T>) -> Self {
        Self::from_raw(T::Mat3::from_translation(translation.to_raw()))
    }

    /// Scaling, rotation, and translation matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    /// let matrix = Matrix3::<f64>::from_scale_angle_translation(
    ///     (2.0, 3.0).into(),
    ///     Angle::from_degrees(90.0),
    ///     (10.0, 20.0).into(),
    /// );
    /// assert_abs_diff_eq!(matrix.row(0), (0.0, -3.0, 10.0));
    /// assert_abs_diff_eq!(matrix.row(1), (2.0,  0.0, 20.0));
    /// assert_abs_diff_eq!(matrix.row(2), (0.0,  0.0,  1.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_scale_angle_translation(
        scale: Vector2<T>,
        angle: Angle<T>,
        translation: Vector2<T>,
    ) -> Self {
        Self::from_raw(T::Mat3::from_scale_angle_translation(
            scale.to_raw(),
            angle,
            translation.to_raw(),
        ))
    }

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
    /// approx::assert_abs_diff_eq!(rotated, (0.0, 1.0));
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn transform_point(&self, point: Point2<T>) -> Point2<T> {
        Point2::from_raw(self.as_raw().transform_point(point.to_raw()))
    }

    /// Transform 2D vector.
    ///
    /// See [`glam::Mat3::transform_vector2()`] or
    /// [`glam::DMat3::transform_vector2()`] (depending on the scalar).
    #[inline(always)]
    #[must_use]
    pub fn transform_vector(&self, vector: Vector2<T>) -> Vector2<T> {
        Vector2::from_raw(self.as_raw().transform_vector(vector.to_raw()))
    }

    /// Invert the matrix.
    ///
    /// See [`glam::Mat3::inverse()`] and [`glam::DMat3::inverse()`].
    #[inline(always)]
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self::from_raw(self.as_raw().inverse())
    }
}

impl<T> Mul for Matrix2<T>
where
    T: PrimitiveMatrices,
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl<T> Mul<Vector2<T>> for Matrix2<T>
where
    T: PrimitiveMatrices,
{
    type Output = Vector2<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Vector2<T>) -> Self::Output {
        Vector2::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl<T> Mul<Point2<T>> for Matrix2<T>
where
    T: PrimitiveMatrices,
{
    type Output = Point2<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Point2<T>) -> Self::Output {
        Point2::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl<T> Mul for Matrix3<T>
where
    T: PrimitiveMatrices,
{
    type Output = Matrix3<T>;

    #[inline(always)]
    #[must_use]
    fn mul(self, rhs: Self) -> Self::Output {
        Matrix3::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl<T> Mul<Vector3<T>> for Matrix3<T>
where
    T: PrimitiveMatrices,
{
    type Output = Vector3<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        Vector3::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl<T> Mul<Point3<T>> for Matrix3<T>
where
    T: PrimitiveMatrices,
{
    type Output = Point3<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Point3<T>) -> Self::Output {
        Point3::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl<T> Mul<Vector2<T>> for Matrix3<T>
where
    T: PrimitiveMatrices,
{
    type Output = Vector2<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Vector2<T>) -> Self::Output {
        self.transform_vector(rhs)
    }
}

impl<T> Mul<Point2<T>> for Matrix3<T>
where
    T: PrimitiveMatrices,
{
    type Output = Point2<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Point2<T>) -> Self::Output {
        self.transform_point(rhs)
    }
}

impl<T> Mul<Vector4<T>> for Matrix4<T>
where
    T: PrimitiveMatrices,
{
    type Output = Vector4<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Vector4<T>) -> Self::Output {
        Vector4::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl<T> Mul<Point4<T>> for Matrix4<T>
where
    T: PrimitiveMatrices,
{
    type Output = Point4<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Point4<T>) -> Self::Output {
        Point4::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl<T> Mul<Vector3<T>> for Matrix4<T>
where
    T: PrimitiveMatrices,
{
    type Output = Vector3<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        self.transform_vector(rhs)
    }
}

impl<T> Mul<Point3<T>> for Matrix4<T>
where
    T: PrimitiveMatrices,
{
    type Output = Point3<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Point3<T>) -> Self::Output {
        self.project_point(rhs)
    }
}

impl<T> Matrix4<T>
where
    T: PrimitiveMatrices,
{
    /// Identity matrix
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix4::<f32>::identity();
    /// assert_eq!(matrix.row(0), (1.0, 0.0, 0.0, 0.0));
    /// assert_eq!(matrix.row(1), (0.0, 1.0, 0.0, 0.0));
    /// assert_eq!(matrix.row(2), (0.0, 0.0, 1.0, 0.0));
    /// assert_eq!(matrix.row(3), (0.0, 0.0, 0.0, 1.0));
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn identity() -> Self {
        Self::from_raw(T::Mat4::identity())
    }

    /// Scaling matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix4::<f32>::from_scale((2.0, 3.0, 4.0).into());
    /// assert_eq!(matrix.row(0), (2.0, 0.0, 0.0, 0.0));
    /// assert_eq!(matrix.row(1), (0.0, 3.0, 0.0, 0.0));
    /// assert_eq!(matrix.row(2), (0.0, 0.0, 4.0, 0.0));
    /// assert_eq!(matrix.row(3), (0.0, 0.0, 0.0, 1.0));
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn from_scale(scale: Vector3<T>) -> Self {
        Self::from_raw(T::Mat4::from_scale(scale.to_raw()))
    }

    /// Rotation matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    /// let theta = Angle::from_degrees(90.0);
    /// let matrix = Matrix4::<f32>::from_axis_angle(Vector3::unit_z(), theta);
    /// assert_abs_diff_eq!(matrix.row(0), (0.0, -1.0, 0.0, 0.0));
    /// assert_abs_diff_eq!(matrix.row(1), (1.0,  0.0, 0.0, 0.0));
    /// assert_abs_diff_eq!(matrix.row(2), (0.0,  0.0, 1.0, 0.0));
    /// assert_abs_diff_eq!(matrix.row(3), (0.0,  0.0, 0.0, 1.0));
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn from_axis_angle(axis: Vector3<T>, angle: Angle<T>) -> Self {
        Self::from_raw(T::Mat4::from_axis_angle(axis.to_raw(), angle))
    }

    /// Translation matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    /// let matrix = Matrix4::<f32>::from_translation((10.0, 20.0, 30.0).into());
    /// assert_abs_diff_eq!(matrix.row(0), (1.0, 0.0, 0.0, 10.0));
    /// assert_abs_diff_eq!(matrix.row(1), (0.0, 1.0, 0.0, 20.0));
    /// assert_abs_diff_eq!(matrix.row(2), (0.0, 0.0, 1.0, 30.0));
    /// assert_abs_diff_eq!(matrix.row(3), (0.0, 0.0, 0.0,  1.0));
    /// ```
    #[inline(always)]
    #[must_use]
    pub fn from_translation(translation: Vector3<T>) -> Self {
        Self::from_raw(T::Mat4::from_translation(translation.to_raw()))
    }

    /// Scaling, rotation, and translation matrix.
    ///
    /// Note: This internally converts `axis` and `angle` to a quaternion and
    /// calls [`glam::Mat4::from_scale_rotation_translation()`].
    #[inline]
    #[must_use]
    pub fn from_scale_rotation_translation(
        scale: Vector3<T>,
        axis: Vector3<T>,
        angle: Angle<T>,
        translation: Vector3<T>,
    ) -> Self {
        Self::from_raw(T::Mat4::from_scale_rotation_translation(
            scale.to_raw(),
            axis.to_raw(),
            angle,
            translation.to_raw(),
        ))
    }

    /// Transform 3D point.
    ///
    /// This assumes that the matrix is a valid affine matrix, and does not
    /// perform perspective correction.
    ///
    /// See [`glam::Mat4::transform_point3()`] or
    /// [`glam::DMat4::transform_point3()`] (depending on the scalar).
    #[inline(always)]
    #[must_use]
    pub fn transform_point(&self, point: Point3<T>) -> Point3<T> {
        Point3::from_raw(self.as_raw().transform_point(point.to_raw()))
    }

    /// Transform 3D vector.
    ///
    /// See [`glam::Mat4::transform_vector3()`] or
    /// [`glam::DMat4::transform_vector3()`] (depending on the scalar).
    #[inline(always)]
    #[must_use]
    pub fn transform_vector(&self, vector: Vector3<T>) -> Vector3<T> {
        Vector3::from_raw(self.as_raw().transform_vector(vector.to_raw()))
    }

    /// Project 3D point.
    ///
    /// Transform the point, including perspective correction.
    ///
    /// See [`glam::Mat4::project_point3()`] or
    /// [`glam::DMat4::project_point3()`] (depending on the scalar).
    #[inline(always)]
    #[must_use]
    pub fn project_point(&self, point: Point3<T>) -> Point3<T> {
        Point3::from_raw(self.as_raw().project_point(point.to_raw()))
    }

    /// Invert the matrix.
    ///
    /// See [`glam::Mat4::inverse()`] and [`glam::DMat4::inverse()`].
    #[inline(always)]
    #[must_use]
    pub fn inverse(&self) -> Self {
        Self::from_raw(self.as_raw().inverse())
    }
}

impl<T> Mul for Matrix4<T>
where
    T: PrimitiveMatrices,
{
    type Output = Matrix4<T>;

    #[inline(always)]
    #[must_use]
    fn mul(self, rhs: Self) -> Self::Output {
        Matrix4::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl_matrix!(Matrix2 <2> => Mat2 [Vector2, Vector2]);
impl_matrix!(Matrix3 <3> => Mat3 [Vector3, Vector2]);
impl_matrix!(Matrix4 <4> => Mat4 [Vector4, Vector3]);

impl<T: AbsDiffEq> AbsDiffEq for Matrix2<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.m11.abs_diff_eq(&other.m11, epsilon)
            && self.m12.abs_diff_eq(&other.m12, epsilon)
            && self.m21.abs_diff_eq(&other.m21, epsilon)
            && self.m22.abs_diff_eq(&other.m22, epsilon)
    }

    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.m11.abs_diff_ne(&other.m11, epsilon)
            || self.m12.abs_diff_ne(&other.m12, epsilon)
            || self.m21.abs_diff_ne(&other.m21, epsilon)
            || self.m22.abs_diff_ne(&other.m22, epsilon)
    }
}

impl<T: RelativeEq> RelativeEq for Matrix2<T>
where
    T::Epsilon: Copy,
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
        self.m11.relative_eq(&other.m11, epsilon, max_relative)
            && self.m12.relative_eq(&other.m12, epsilon, max_relative)
            && self.m21.relative_eq(&other.m21, epsilon, max_relative)
            && self.m22.relative_eq(&other.m22, epsilon, max_relative)
    }

    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.m11.relative_ne(&other.m11, epsilon, max_relative)
            || self.m12.relative_ne(&other.m12, epsilon, max_relative)
            || self.m21.relative_ne(&other.m21, epsilon, max_relative)
            || self.m22.relative_ne(&other.m22, epsilon, max_relative)
    }
}

impl<T: AbsDiffEq> AbsDiffEq for Matrix3<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.m11.abs_diff_eq(&other.m11, epsilon)
            && self.m12.abs_diff_eq(&other.m12, epsilon)
            && self.m13.abs_diff_eq(&other.m13, epsilon)
            && self.m21.abs_diff_eq(&other.m21, epsilon)
            && self.m22.abs_diff_eq(&other.m22, epsilon)
            && self.m23.abs_diff_eq(&other.m23, epsilon)
            && self.m31.abs_diff_eq(&other.m31, epsilon)
            && self.m32.abs_diff_eq(&other.m32, epsilon)
            && self.m33.abs_diff_eq(&other.m33, epsilon)
    }

    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.m11.abs_diff_ne(&other.m11, epsilon)
            || self.m12.abs_diff_ne(&other.m12, epsilon)
            || self.m13.abs_diff_ne(&other.m13, epsilon)
            || self.m21.abs_diff_ne(&other.m21, epsilon)
            || self.m22.abs_diff_ne(&other.m22, epsilon)
            || self.m23.abs_diff_ne(&other.m23, epsilon)
            || self.m31.abs_diff_ne(&other.m31, epsilon)
            || self.m32.abs_diff_ne(&other.m32, epsilon)
            || self.m33.abs_diff_ne(&other.m33, epsilon)
    }
}

impl<T: RelativeEq> RelativeEq for Matrix3<T>
where
    T::Epsilon: Copy,
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
        self.m11.relative_eq(&other.m11, epsilon, max_relative)
            && self.m12.relative_eq(&other.m12, epsilon, max_relative)
            && self.m13.relative_eq(&other.m13, epsilon, max_relative)
            && self.m21.relative_eq(&other.m21, epsilon, max_relative)
            && self.m22.relative_eq(&other.m22, epsilon, max_relative)
            && self.m23.relative_eq(&other.m23, epsilon, max_relative)
            && self.m31.relative_eq(&other.m31, epsilon, max_relative)
            && self.m32.relative_eq(&other.m32, epsilon, max_relative)
            && self.m33.relative_eq(&other.m33, epsilon, max_relative)
    }

    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.m11.relative_ne(&other.m11, epsilon, max_relative)
            || self.m12.relative_ne(&other.m12, epsilon, max_relative)
            || self.m13.relative_ne(&other.m13, epsilon, max_relative)
            || self.m21.relative_ne(&other.m21, epsilon, max_relative)
            || self.m22.relative_ne(&other.m22, epsilon, max_relative)
            || self.m23.relative_ne(&other.m23, epsilon, max_relative)
            || self.m31.relative_ne(&other.m31, epsilon, max_relative)
            || self.m32.relative_ne(&other.m32, epsilon, max_relative)
            || self.m33.relative_ne(&other.m33, epsilon, max_relative)
    }
}

impl<T: AbsDiffEq> AbsDiffEq for Matrix4<T>
where
    T::Epsilon: Copy,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.m11.abs_diff_eq(&other.m11, epsilon)
            && self.m12.abs_diff_eq(&other.m12, epsilon)
            && self.m13.abs_diff_eq(&other.m13, epsilon)
            && self.m14.abs_diff_eq(&other.m14, epsilon)
            && self.m21.abs_diff_eq(&other.m21, epsilon)
            && self.m22.abs_diff_eq(&other.m22, epsilon)
            && self.m23.abs_diff_eq(&other.m23, epsilon)
            && self.m24.abs_diff_eq(&other.m24, epsilon)
            && self.m31.abs_diff_eq(&other.m31, epsilon)
            && self.m32.abs_diff_eq(&other.m32, epsilon)
            && self.m33.abs_diff_eq(&other.m33, epsilon)
            && self.m34.abs_diff_eq(&other.m34, epsilon)
            && self.m41.abs_diff_eq(&other.m41, epsilon)
            && self.m42.abs_diff_eq(&other.m42, epsilon)
            && self.m43.abs_diff_eq(&other.m43, epsilon)
            && self.m44.abs_diff_eq(&other.m44, epsilon)
    }

    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.m11.abs_diff_ne(&other.m11, epsilon)
            || self.m12.abs_diff_ne(&other.m12, epsilon)
            || self.m13.abs_diff_ne(&other.m13, epsilon)
            || self.m14.abs_diff_ne(&other.m14, epsilon)
            || self.m21.abs_diff_ne(&other.m21, epsilon)
            || self.m22.abs_diff_ne(&other.m22, epsilon)
            || self.m23.abs_diff_ne(&other.m23, epsilon)
            || self.m24.abs_diff_ne(&other.m24, epsilon)
            || self.m31.abs_diff_ne(&other.m31, epsilon)
            || self.m32.abs_diff_ne(&other.m32, epsilon)
            || self.m33.abs_diff_ne(&other.m33, epsilon)
            || self.m34.abs_diff_ne(&other.m34, epsilon)
            || self.m41.abs_diff_ne(&other.m41, epsilon)
            || self.m42.abs_diff_ne(&other.m42, epsilon)
            || self.m43.abs_diff_ne(&other.m43, epsilon)
            || self.m44.abs_diff_ne(&other.m44, epsilon)
    }
}

impl<T: RelativeEq> RelativeEq for Matrix4<T>
where
    T::Epsilon: Copy,
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
        self.m11.relative_eq(&other.m11, epsilon, max_relative)
            && self.m12.relative_eq(&other.m12, epsilon, max_relative)
            && self.m13.relative_eq(&other.m13, epsilon, max_relative)
            && self.m14.relative_eq(&other.m14, epsilon, max_relative)
            && self.m21.relative_eq(&other.m21, epsilon, max_relative)
            && self.m22.relative_eq(&other.m22, epsilon, max_relative)
            && self.m23.relative_eq(&other.m23, epsilon, max_relative)
            && self.m24.relative_eq(&other.m24, epsilon, max_relative)
            && self.m31.relative_eq(&other.m31, epsilon, max_relative)
            && self.m32.relative_eq(&other.m32, epsilon, max_relative)
            && self.m33.relative_eq(&other.m33, epsilon, max_relative)
            && self.m34.relative_eq(&other.m34, epsilon, max_relative)
            && self.m41.relative_eq(&other.m41, epsilon, max_relative)
            && self.m42.relative_eq(&other.m42, epsilon, max_relative)
            && self.m43.relative_eq(&other.m43, epsilon, max_relative)
            && self.m44.relative_eq(&other.m44, epsilon, max_relative)
    }

    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.m11.relative_ne(&other.m11, epsilon, max_relative)
            || self.m12.relative_ne(&other.m12, epsilon, max_relative)
            || self.m13.relative_ne(&other.m13, epsilon, max_relative)
            || self.m14.relative_ne(&other.m14, epsilon, max_relative)
            || self.m21.relative_ne(&other.m21, epsilon, max_relative)
            || self.m22.relative_ne(&other.m22, epsilon, max_relative)
            || self.m23.relative_ne(&other.m23, epsilon, max_relative)
            || self.m24.relative_ne(&other.m24, epsilon, max_relative)
            || self.m31.relative_ne(&other.m31, epsilon, max_relative)
            || self.m32.relative_ne(&other.m32, epsilon, max_relative)
            || self.m33.relative_ne(&other.m33, epsilon, max_relative)
            || self.m34.relative_ne(&other.m34, epsilon, max_relative)
            || self.m41.relative_ne(&other.m41, epsilon, max_relative)
            || self.m42.relative_ne(&other.m42, epsilon, max_relative)
            || self.m43.relative_ne(&other.m43, epsilon, max_relative)
            || self.m44.relative_ne(&other.m44, epsilon, max_relative)
    }
}

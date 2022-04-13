//! Matrix types.
//!
//! Matrices do not have a [`Unit`](crate::Unit), because their values
//! do not necessarily have a clear logical meaning in the context of any
//! particular unit.
//!
//! Instead, they are based on fundamental floating-point
//! [primitive](crate::bindings::Primitive) scalars (`f32` or `f64`).

use core::ops::Mul;

use crate::{
    bindings::{Matrix, PrimitiveMatrices, SimdMatrix2, SimdMatrix3, SimdMatrix4},
    Angle, Point2, Point3, Point4, Vector2, Vector3, Vector4,
};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use bytemuck::{cast, cast_mut, cast_ref, Pod, Zeroable};

/// 2x2 column-major matrix.
///
/// Bitwise compatible with [`glam::Mat2`] / [`glam::DMat2`].
///
/// Alignment: Always 16-byte aligned.
#[repr(C, align(16))]
#[derive(Clone, Copy, PartialEq)]
#[allow(missing_docs)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
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
#[derive(Clone, Copy, PartialEq)]
#[allow(missing_docs)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
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
#[derive(Clone, Copy, PartialEq)]
#[allow(missing_docs)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
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
            #[doc = "Create from rows with implicit conversion."]
            #[inline]
            #[must_use]
            pub fn with_rows<U>(rows: [U; $dimensions]) -> Self
            where
                U: Into<$axis_vector_ty<T>>,
            {
                Self::from_rows(rows.map(Into::into))
            }

            #[doc = "Create from columns with implicit conversion."]
            #[inline]
            #[must_use]
            pub fn with_cols<U>(rows: [U; $dimensions]) -> Self
            where
                U: Into<$axis_vector_ty<T>>,
            {
                Self::from_cols(rows.map(Into::into))
            }

            #[doc = "Create zero matrix."]
            #[inline]
            #[must_use]
            pub fn zero() -> Self {
                Self::from_raw(T::$mat_name::zero())
            }

            #[doc = "Create a matrix with all NaNs."]
            #[inline]
            #[must_use]
            pub fn nan() -> Self {
                Self::from_raw(T::$mat_name::nan())
            }

            #[doc = "Create from column vectors."]
            #[inline]
            #[must_use]
            pub fn from_cols(cols: [$axis_vector_ty<T>; $dimensions]) -> Self {
                Self::from_raw(T::$mat_name::from_cols(bytemuck::cast(cols)))
            }

            #[doc = "Create from row vectors."]
            #[inline]
            #[must_use]
            pub fn from_rows(cols: [$axis_vector_ty<T>; $dimensions]) -> Self {
                Self::from_raw(T::$mat_name::from_cols(bytemuck::cast(cols)).transpose())
            }

            #[doc = "Get the underlying `glam` matrix."]
            #[inline]
            #[must_use]
            pub fn to_raw(self) -> T::$mat_name {
                cast(self)
            }

            #[doc = "Create from underlying `glam` matrix."]
            #[inline]
            #[must_use]
            pub fn from_raw(raw: T::$mat_name) -> Self {
                cast(raw)
            }

            #[doc = "Cast to `glam` matrix."]
            #[inline]
            #[must_use]
            pub fn as_raw(&self) -> &T::$mat_name {
                cast_ref(self)
            }

            #[doc = "Cast to `glam` matrix."]
            #[inline]
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

            #[doc = "Get mutable reference to column vector at `index`."]
            #[inline]
            #[must_use]
            pub fn col_mut(&mut self, index: usize) -> &mut $axis_vector_ty<T> {
                &mut self.as_cols_mut()[index]
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
            pub fn to_cols(&self) -> [$axis_vector_ty<T>; $dimensions] {
                bytemuck::cast(self.as_raw().to_cols())
            }

            #[doc = "Get row vectors."]
            #[inline]
            #[must_use]
            pub fn to_rows(&self) -> [$axis_vector_ty<T>; $dimensions] {
                bytemuck::cast(self.as_raw().to_rows())
            }

            #[doc = "Get column vectors as slice."]
            #[inline]
            #[must_use]
            pub fn as_cols(&self) -> &[$axis_vector_ty<T>; $dimensions] {
                bytemuck::cast_ref(self)
            }

            #[doc = "Get column vectors as slice."]
            #[inline]
            #[must_use]
            pub fn as_cols_mut(&mut self) -> &mut [$axis_vector_ty<T>; $dimensions] {
                bytemuck::cast_mut(self)
            }

            #[doc = "Matrix determinant."]
            #[inline]
            #[must_use]
            pub fn determinant(&self) -> T {
                self.as_raw().determinant()
            }

            #[doc = "True if matrix is invertible."]
            #[doc = ""]
            #[doc = "This is equivalent to checking if the determinant is finite and non-zero."]
            #[inline]
            #[must_use]
            pub fn is_invertible(&self) -> bool {
                let d = self.determinant();
                d != T::zero() && crate::Scalar::is_finite(d)
            }

            #[doc = "Return the inverse matrix."]
            #[doc = ""]
            #[doc = "If the matrix is not invertible, this returns an invalid matrix."]
            #[doc = ""]
            #[doc = "See (e.g.) [`glam::Mat3::inverse()`]."]
            #[inline]
            #[must_use]
            pub fn inverse_unchecked(&self) -> Self {
                Self::from_raw(self.as_raw().inverse())
            }

            #[doc = "Return the inverse matrix, if invertible."]
            #[doc = ""]
            #[doc = "If the matrix is not invertible, this returns `None`."]
            #[inline]
            #[must_use]
            pub fn inverse(&self) -> Option<Self> {
                if self.is_invertible() {
                    Some(self.inverse_unchecked())
                } else {
                    None
                }
            }

            #[doc = "True if any element in the matrix is NaN."]
            #[inline]
            #[must_use]
            pub fn is_nan(&self) -> bool {
                self.as_raw().is_nan()
            }

            #[doc = "True if all elements in the matrix are finite (non-infinite, non-NaN)."]
            #[inline]
            #[must_use]
            pub fn is_finite(&self) -> bool {
                self.as_raw().is_finite()
            }
        }

        impl<T> Default for $base_type_name<T>
        where
            T: PrimitiveMatrices,
        {
            #[inline]
            fn default() -> Self {
                Self::identity()
            }
        }

        impl<T> core::fmt::Debug for $base_type_name<T>
        where
            T: PrimitiveMatrices,
        {
            fn fmt(&self, fmt: &mut core::fmt::Formatter) -> core::fmt::Result {
                let mut list = fmt.debug_list();
                for i in 0..$dimensions {
                    list.entry(&self.row(i).to_tuple());
                }
                list.finish()
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
    #[inline]
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
    #[inline]
    #[must_use]
    pub fn from_scale(scale: Vector2<T>) -> Self {
        Self::from_raw(T::Mat2::from_scale_angle(scale.to_raw(), T::zero()))
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
    #[inline]
    #[must_use]
    pub fn from_angle(angle: Angle<T>) -> Self {
        Self::from_raw(T::Mat2::from_angle(angle.radians))
    }

    /// Transform 2D point.
    ///
    /// See [`glam::Mat2::mul_vec2()`] or
    /// [`glam::DMat2::mul_vec2()`] (depending on the scalar).
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
    #[inline]
    #[must_use]
    pub fn transform_point(&self, point: Point2<T>) -> Point2<T> {
        Point2::from_raw(self.as_raw().transform_point(point.to_raw()))
    }

    /// Transform 2D vector.
    ///
    /// See [`glam::Mat2::mul_vec2()`] or
    /// [`glam::DMat2::mul_vec2()`] (depending on the scalar).
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix2::<f32>::from_angle(Angle::<f32>::from_degrees(90.0));
    /// let vector = Vector2::<f32> { x: 1.0, y: 0.0 };
    /// let rotated = matrix.transform_vector(vector);
    /// approx::assert_abs_diff_eq!(rotated, (0.0, 1.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, vector: Vector2<T>) -> Vector2<T> {
        Vector2::from_raw(self.as_raw().transform_vector(vector.to_raw()))
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
    #[inline]
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
    #[inline]
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
    #[inline]
    #[must_use]
    pub fn from_angle(angle: Angle<T>) -> Self {
        Self::from_raw(T::Mat3::from_angle(angle.radians))
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
    #[inline]
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
            angle.radians,
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
    #[inline]
    #[must_use]
    pub fn transform_point(&self, point: Point2<T>) -> Point2<T> {
        Point2::from_raw(self.as_raw().transform_point(point.to_raw()))
    }

    /// Transform 2D vector.
    ///
    /// See [`glam::Mat3::transform_vector2()`] or
    /// [`glam::DMat3::transform_vector2()`] (depending on the scalar).
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, vector: Vector2<T>) -> Vector2<T> {
        Vector2::from_raw(self.as_raw().transform_vector(vector.to_raw()))
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

    #[inline]
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
    #[inline]
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
    #[inline]
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
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: Vector3<T>, angle: Angle<T>) -> Self {
        Self::from_raw(T::Mat4::from_axis_angle(axis.to_raw(), angle.radians))
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
    #[inline]
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
        use crate::bindings::Quat;
        let quat = <T as PrimitiveMatrices>::Quat::from_axis_angle(axis.to_raw(), angle.radians);
        Self::from_raw(T::Mat4::from_scale_rotation_translation(
            scale.to_raw(),
            quat,
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
    #[inline]
    #[must_use]
    pub fn transform_point(&self, point: Point3<T>) -> Point3<T> {
        Point3::from_raw(self.as_raw().transform_point(point.to_raw()))
    }

    /// Transform 3D vector.
    ///
    /// See [`glam::Mat4::transform_vector3()`] or
    /// [`glam::DMat4::transform_vector3()`] (depending on the scalar).
    #[inline]
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
    #[inline]
    #[must_use]
    pub fn project_point(&self, point: Point3<T>) -> Point3<T> {
        Point3::from_raw(self.as_raw().project_point(point.to_raw()))
    }
}

impl<T> Mul for Matrix4<T>
where
    T: PrimitiveMatrices,
{
    type Output = Matrix4<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Self) -> Self::Output {
        Matrix4::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl_matrix!(Matrix2 <2> => Mat2 [Vector2, Vector2]);
impl_matrix!(Matrix3 <3> => Mat3 [Vector3, Vector2]);
impl_matrix!(Matrix4 <4> => Mat4 [Vector4, Vector3]);

impl<T> AbsDiffEq for Matrix2<T>
where
    T: PrimitiveMatrices + AbsDiffEq,
    T::Epsilon: Clone,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_raw().abs_diff_eq(other.as_raw(), epsilon)
    }

    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_raw().abs_diff_ne(other.as_raw(), epsilon)
    }
}

impl<T> RelativeEq for Matrix2<T>
where
    T: PrimitiveMatrices + RelativeEq,
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
        self.to_cols()
            .relative_eq(&other.to_cols(), epsilon, max_relative)
    }

    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.to_cols()
            .relative_ne(&other.to_cols(), epsilon, max_relative)
    }
}

impl<T> UlpsEq for Matrix2<T>
where
    T: PrimitiveMatrices + UlpsEq,
    T::Epsilon: Clone,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.to_cols().ulps_eq(&other.to_cols(), epsilon, max_ulps)
    }

    fn ulps_ne(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.to_cols().ulps_ne(&other.to_cols(), epsilon, max_ulps)
    }
}

impl<T> AbsDiffEq for Matrix3<T>
where
    T: PrimitiveMatrices + AbsDiffEq,
    T::Epsilon: Clone,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_raw().abs_diff_eq(other.as_raw(), epsilon)
    }

    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_raw().abs_diff_ne(other.as_raw(), epsilon)
    }
}

impl<T> RelativeEq for Matrix3<T>
where
    T: PrimitiveMatrices + RelativeEq,
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
        self.to_cols()
            .relative_eq(&other.to_cols(), epsilon, max_relative)
    }

    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.to_cols()
            .relative_ne(&other.to_cols(), epsilon, max_relative)
    }
}

impl<T> UlpsEq for Matrix3<T>
where
    T: PrimitiveMatrices + UlpsEq,
    T::Epsilon: Clone,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.to_cols().ulps_eq(&other.to_cols(), epsilon, max_ulps)
    }

    fn ulps_ne(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.to_cols().ulps_ne(&other.to_cols(), epsilon, max_ulps)
    }
}

impl<T> AbsDiffEq for Matrix4<T>
where
    T: AbsDiffEq + PrimitiveMatrices,
    T::Epsilon: Clone,
{
    type Epsilon = T::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_raw().abs_diff_eq(other.as_raw(), epsilon)
    }

    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.as_raw().abs_diff_ne(other.as_raw(), epsilon)
    }
}

impl<T> RelativeEq for Matrix4<T>
where
    T: RelativeEq + PrimitiveMatrices,
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
        self.to_cols()
            .relative_eq(&other.to_cols(), epsilon, max_relative)
    }

    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.to_cols()
            .relative_ne(&other.to_cols(), epsilon, max_relative)
    }
}

impl<T> UlpsEq for Matrix4<T>
where
    T: PrimitiveMatrices + UlpsEq,
    T::Epsilon: Clone,
{
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.to_cols().ulps_eq(&other.to_cols(), epsilon, max_ulps)
    }

    fn ulps_ne(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.to_cols().ulps_ne(&other.to_cols(), epsilon, max_ulps)
    }
}

#[cfg(test)]
mod tests {
    use approx::{
        assert_abs_diff_eq, assert_abs_diff_ne, assert_relative_eq, assert_relative_ne,
        assert_ulps_eq, assert_ulps_ne,
    };

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

    type Point2 = super::Point2<f32>;
    type Point3 = super::Point3<f32>;
    type Point4 = super::Point4<f32>;
    type DPoint2 = super::Point2<f64>;
    type DPoint3 = super::Point3<f64>;
    type DPoint4 = super::Point4<f64>;

    #[test]
    fn from_scale() {
        let m2 = Mat2::from_scale(Vec2::new(2.0, 3.0));
        let m3 = Mat3::from_scale(Vec2::new(2.0, 3.0));
        let m4 = Mat4::from_scale(Vec3::new(2.0, 3.0, 4.0));

        assert_eq!(m2, Mat2::with_rows([(2.0, 0.0), (0.0, 3.0)]));
        assert_eq!(
            m3,
            Mat3::with_rows([(2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 1.0)])
        );
        assert_eq!(
            m4,
            Mat4::with_rows([
                (2.0, 0.0, 0.0, 0.0),
                (0.0, 3.0, 0.0, 0.0),
                (0.0, 0.0, 4.0, 0.0),
                (0.0, 0.0, 0.0, 1.0)
            ])
        );

        let m2 = DMat2::from_scale(DVec2::new(2.0, 3.0));
        let m3 = DMat3::from_scale(DVec2::new(2.0, 3.0));
        let m4 = DMat4::from_scale(DVec3::new(2.0, 3.0, 4.0));

        assert_eq!(m2, DMat2::with_rows([(2.0, 0.0), (0.0, 3.0)]));
        assert_eq!(
            m3,
            DMat3::with_rows([(2.0, 0.0, 0.0), (0.0, 3.0, 0.0), (0.0, 0.0, 1.0)])
        );
        assert_eq!(
            m4,
            DMat4::with_rows([
                (2.0, 0.0, 0.0, 0.0),
                (0.0, 3.0, 0.0, 0.0),
                (0.0, 0.0, 4.0, 0.0),
                (0.0, 0.0, 0.0, 1.0)
            ])
        );
    }

    #[test]
    fn from_angle() {
        let m2 = Mat2::from_angle(Angle::from_degrees(90.0));
        let m3 = Mat3::from_angle(Angle::from_degrees(90.0));
        let m4 = Mat4::from_axis_angle(Vec3::unit_z(), Angle::from_degrees(90.0));

        assert_abs_diff_eq!(m2, Mat2::with_rows([(0.0, -1.0), (1.0, 0.0)]));
        assert_abs_diff_eq!(
            m3,
            Mat3::with_rows([(0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
        );
        assert_abs_diff_eq!(
            m4,
            Mat4::with_rows([
                (0.0, -1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0)
            ])
        );

        let m2 = DMat2::from_angle(Angle::from_degrees(90.0));
        let m3 = DMat3::from_angle(Angle::from_degrees(90.0));
        let m4 = DMat4::from_axis_angle(Vector3::unit_z(), Angle::from_degrees(90.0));

        assert_abs_diff_eq!(m2, DMat2::with_rows([(0.0, -1.0), (1.0, 0.0)]));
        assert_abs_diff_eq!(
            m3,
            DMat3::with_rows([(0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)])
        );
        assert_abs_diff_eq!(
            m4,
            DMat4::with_rows([
                (0.0, -1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0)
            ])
        );
    }

    #[test]
    fn from_translation() {
        let m3 = Mat3::from_translation(Vector2::new(2.0, 3.0));
        let m4 = Mat4::from_translation(Vector3::new(2.0, 3.0, 4.0));

        assert_eq!(
            m3,
            Mat3::with_rows([(1.0, 0.0, 2.0), (0.0, 1.0, 3.0), (0.0, 0.0, 1.0),])
        );
        assert_eq!(
            m4,
            Mat4::with_rows([
                (1.0, 0.0, 0.0, 2.0),
                (0.0, 1.0, 0.0, 3.0),
                (0.0, 0.0, 1.0, 4.0),
                (0.0, 0.0, 0.0, 1.0),
            ])
        );

        let m3 = DMat3::from_translation(Vector2::new(2.0, 3.0));
        let m4 = DMat4::from_translation(Vector3::new(2.0, 3.0, 4.0));

        assert_eq!(
            m3,
            DMat3::with_rows([(1.0, 0.0, 2.0), (0.0, 1.0, 3.0), (0.0, 0.0, 1.0),])
        );
        assert_eq!(
            m4,
            DMat4::with_rows([
                (1.0, 0.0, 0.0, 2.0),
                (0.0, 1.0, 0.0, 3.0),
                (0.0, 0.0, 1.0, 4.0),
                (0.0, 0.0, 0.0, 1.0),
            ])
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
            let axis = Vec3::unit_z();
            let angle = Angle::from_degrees(90.0);
            let translation = Vec3::new(5.0, 6.0, 7.0);

            assert_abs_diff_eq!(
                Mat4::from_scale_rotation_translation(scale, axis, angle, translation),
                Mat4::from_translation(translation)
                    * Mat4::from_axis_angle(axis, angle)
                    * Mat4::from_scale(scale),
                epsilon = 0.0001
            );
        }

        {
            let scale = DVec3::new(2.0, 3.0, 4.0);
            let axis = DVec3::unit_z();
            let angle = Angle::from_degrees(90.0);
            let translation = DVec3::new(5.0, 6.0, 7.0);

            assert_abs_diff_eq!(
                DMat4::from_scale_rotation_translation(scale, axis, angle, translation),
                DMat4::from_translation(translation)
                    * DMat4::from_axis_angle(axis, angle)
                    * DMat4::from_scale(scale),
                epsilon = 0.0001
            );
        }
    }

    #[test]
    fn to_cols() {
        assert_eq!(Mat2::identity().to_cols(), [(1.0, 0.0), (0.0, 1.0)]);
        assert_eq!(
            Mat3::identity().to_cols(),
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        );
        assert_eq!(
            Mat4::identity().to_cols(),
            [
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0)
            ]
        );

        assert_eq!(DMat2::identity().to_cols(), [(1.0, 0.0), (0.0, 1.0)]);
        assert_eq!(
            DMat3::identity().to_cols(),
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        );
        assert_eq!(
            DMat4::identity().to_cols(),
            [
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0)
            ]
        );
    }

    #[test]
    fn to_rows() {
        assert_eq!(Mat2::identity().to_rows(), [(1.0, 0.0), (0.0, 1.0)]);
        assert_eq!(
            Mat3::identity().to_rows(),
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        );
        assert_eq!(
            Mat4::identity().to_rows(),
            [
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0)
            ]
        );

        assert_eq!(DMat2::identity().to_rows(), [(1.0, 0.0), (0.0, 1.0)]);
        assert_eq!(
            DMat3::identity().to_rows(),
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        );
        assert_eq!(
            DMat4::identity().to_rows(),
            [
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0)
            ]
        );
    }

    #[test]
    fn from_cols() {
        assert_eq!(
            Mat2::from_cols([(1.0, 0.0).into(), (0.0, 1.0).into()]),
            Mat2::identity()
        );
        assert_eq!(
            Mat3::from_cols([
                (1.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 1.0).into()
            ]),
            Mat3::identity()
        );
        assert_eq!(
            Mat4::from_cols([
                (1.0, 0.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0, 0.0).into(),
                (0.0, 0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 0.0, 1.0).into()
            ]),
            Mat4::identity()
        );

        assert_eq!(
            DMat2::from_cols([(1.0, 0.0).into(), (0.0, 1.0).into()]),
            DMat2::identity()
        );
        assert_eq!(
            DMat3::from_cols([
                (1.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 1.0).into()
            ]),
            DMat3::identity()
        );
        assert_eq!(
            DMat4::from_cols([
                (1.0, 0.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0, 0.0).into(),
                (0.0, 0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 0.0, 1.0).into()
            ]),
            DMat4::identity()
        );
    }

    #[test]
    fn from_rows() {
        assert_eq!(
            Mat2::from_rows([(1.0, 0.0).into(), (0.0, 1.0).into()]),
            Mat2::identity()
        );
        assert_eq!(
            Mat3::from_rows([
                (1.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 1.0).into()
            ]),
            Mat3::identity()
        );
        assert_eq!(
            Mat4::from_rows([
                (1.0, 0.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0, 0.0).into(),
                (0.0, 0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 0.0, 1.0).into()
            ]),
            Mat4::identity()
        );

        assert_eq!(
            DMat2::from_rows([(1.0, 0.0).into(), (0.0, 1.0).into()]),
            DMat2::identity()
        );
        assert_eq!(
            DMat3::from_rows([
                (1.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 1.0).into()
            ]),
            DMat3::identity()
        );
        assert_eq!(
            DMat4::from_rows([
                (1.0, 0.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0, 0.0).into(),
                (0.0, 0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 0.0, 1.0).into()
            ]),
            DMat4::identity()
        );
    }

    #[test]
    fn col_mut() {
        let mut m2 = Mat2::identity();
        let mut m3 = Mat3::identity();
        let mut m4 = Mat4::identity();
        let mut dm2 = DMat2::identity();
        let mut dm3 = DMat3::identity();
        let mut dm4 = DMat4::identity();

        let _: &[Vec2; 2] = m2.as_cols();
        let _: &[Vec3; 3] = m3.as_cols();
        let _: &[Vec4; 4] = m4.as_cols();
        let _: &[DVec2; 2] = dm2.as_cols();
        let _: &[DVec3; 3] = dm3.as_cols();
        let _: &[DVec4; 4] = dm4.as_cols();

        m2.col_mut(0).set(1, 2.0);
        m3.col_mut(0).set(1, 2.0);
        m4.col_mut(0).set(1, 2.0);
        dm2.col_mut(0).set(1, 2.0);
        dm3.col_mut(0).set(1, 2.0);
        dm4.col_mut(0).set(1, 2.0);

        assert_eq!(m2.col(0), (1.0, 2.0));
        assert_eq!(m3.col(0), (1.0, 2.0, 0.0));
        assert_eq!(m4.col(0), (1.0, 2.0, 0.0, 0.0));
        assert_eq!(dm2.col(0), (1.0, 2.0));
        assert_eq!(dm3.col(0), (1.0, 2.0, 0.0));
        assert_eq!(dm4.col(0), (1.0, 2.0, 0.0, 0.0));
    }

    #[test]
    fn equality() {
        let m2 = Mat2::identity();
        assert_eq!(m2, m2);
        assert_abs_diff_eq!(m2, m2);
        assert_relative_eq!(m2, m2);
        assert_ulps_eq!(m2, m2);
        assert_ne!(m2, Mat2::zero());
        assert_abs_diff_ne!(m2, Mat2::zero());
        assert_relative_ne!(m2, Mat2::zero());
        assert_ulps_ne!(m2, Mat2::zero());

        let m3 = Mat3::identity();
        assert_eq!(m3, m3);
        assert_abs_diff_eq!(m3, m3);
        assert_relative_eq!(m3, m3);
        assert_ulps_eq!(m3, m3);
        assert_ne!(m3, Mat3::zero());
        assert_abs_diff_ne!(m3, Mat3::zero());
        assert_relative_ne!(m3, Mat3::zero());
        assert_ulps_ne!(m3, Mat3::zero());

        let m4 = Mat4::identity();
        assert_eq!(m4, m4);
        assert_abs_diff_eq!(m4, m4);
        assert_relative_eq!(m4, m4);
        assert_ulps_eq!(m4, m4);
        assert_ne!(m4, Mat4::zero());
        assert_abs_diff_ne!(m4, Mat4::zero());
        assert_relative_ne!(m4, Mat4::zero());
        assert_ulps_ne!(m4, Mat4::zero());
    }

    #[test]
    fn transform() {
        let v2 = Vec2::new(1.0, 2.0);
        let v3 = Vec3::new(1.0, 2.0, 3.0);
        let v4 = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let p2 = Point2::new(1.0, 2.0);
        let p3 = Point3::new(1.0, 2.0, 3.0);
        let p4 = Point4::new(1.0, 2.0, 3.0, 4.0);

        let mat2 = Mat2::from_scale((2.0, 2.0).into());
        let mat3 = Mat3::from_scale((2.0, 2.0).into());
        let mat4 = Mat4::from_scale((2.0, 2.0, 2.0).into());

        assert_eq!(mat2 * v2, mat2.transform_vector(v2));
        assert_eq!(mat3 * v2, mat3.transform_vector(v2));
        assert_eq!(mat4 * v3, mat4.transform_vector(v3));
        assert_eq!(mat4 * v4, Vec4::new(2.0, 4.0, 6.0, 4.0));

        assert_eq!(mat2 * p2, mat2.transform_point(p2));
        assert_eq!(mat3 * p2, mat3.transform_point(p2));
        assert_eq!(mat4 * p3, mat4.transform_point(p3));
        assert_eq!(mat4 * p4, Point4::new(2.0, 4.0, 6.0, 4.0));
    }

    #[test]
    fn transform_f64() {
        let v2 = DVec2::new(1.0, 2.0);
        let v3 = DVec3::new(1.0, 2.0, 3.0);
        let v4 = DVec4::new(1.0, 2.0, 3.0, 4.0);
        let p2 = DPoint2::new(1.0, 2.0);
        let p3 = DPoint3::new(1.0, 2.0, 3.0);
        let p4 = DPoint4::new(1.0, 2.0, 3.0, 4.0);

        let mat2 = DMat2::from_scale((2.0, 2.0).into());
        let mat3 = DMat3::from_scale((2.0, 2.0).into());
        let mat4 = DMat4::from_scale((2.0, 2.0, 2.0).into());

        assert_eq!(mat2 * v2, mat2.transform_vector(v2));
        assert_eq!(mat3 * v2, mat3.transform_vector(v2));
        assert_eq!(mat4 * v3, mat4.transform_vector(v3));
        assert_eq!(mat4 * v4, DVec4::new(2.0, 4.0, 6.0, 4.0));

        assert_eq!(mat2 * p2, mat2.transform_point(p2));
        assert_eq!(mat3 * p2, mat3.transform_point(p2));
        assert_eq!(mat4 * p3, mat4.transform_point(p3));
        assert_eq!(mat4 * p4, DPoint4::new(2.0, 4.0, 6.0, 4.0));
    }

    #[test]
    fn determinant() {
        assert_eq!(Mat2::identity().determinant(), 1.0);
        assert_eq!(Mat3::identity().determinant(), 1.0);
        assert_eq!(Mat4::identity().determinant(), 1.0);
        assert_eq!(DMat2::identity().determinant(), 1.0);
        assert_eq!(DMat3::identity().determinant(), 1.0);
        assert_eq!(DMat4::identity().determinant(), 1.0);
    }

    #[test]
    fn nan() {
        let m2 = Mat2::nan();
        assert!(m2.col(0).is_nan());
        assert!(m2.col(1).is_nan());

        let m3 = Mat3::nan();
        assert!(m3.col(0).is_nan());
        assert!(m3.col(1).is_nan());
        assert!(m3.col(2).is_nan());

        let m4 = Mat4::nan();
        assert!(m4.col(0).is_nan());
        assert!(m4.col(1).is_nan());
        assert!(m4.col(2).is_nan());
        assert!(m4.col(3).is_nan());

        let m2 = DMat2::nan();
        assert!(m2.col(0).is_nan());
        assert!(m2.col(1).is_nan());

        let m3 = DMat3::nan();
        assert!(m3.col(0).is_nan());
        assert!(m3.col(1).is_nan());
        assert!(m3.col(2).is_nan());

        let m4 = DMat4::nan();
        assert!(m4.col(0).is_nan());
        assert!(m4.col(1).is_nan());
        assert!(m4.col(2).is_nan());
        assert!(m4.col(3).is_nan());
    }

    #[test]
    fn is_finite() {
        assert!(Mat2::identity().is_finite());
        assert!(Mat3::identity().is_finite());
        assert!(Mat4::identity().is_finite());
        assert!(DMat2::identity().is_finite());
        assert!(DMat3::identity().is_finite());
        assert!(DMat4::identity().is_finite());

        assert!(!Mat2::nan().is_finite());
        assert!(!Mat3::nan().is_finite());
        assert!(!Mat4::nan().is_finite());
        assert!(!DMat2::nan().is_finite());
        assert!(!DMat3::nan().is_finite());
        assert!(!DMat4::nan().is_finite());

        assert!(!DMat4::with_rows([
            (1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, f64::NAN),
        ])
        .is_finite());

        assert!(!DMat4::with_rows([
            (1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, f64::INFINITY),
        ])
        .is_finite());

        assert!(!DMat4::with_rows([
            (1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, 1.0),
            (1.0, 1.0, 1.0, f64::NEG_INFINITY),
        ])
        .is_finite());
    }

    #[test]
    fn is_invertible() {
        assert!(Mat2::identity().is_invertible());
        assert!(Mat3::identity().is_invertible());
        assert!(Mat4::identity().is_invertible());
        assert!(!Mat2::zero().is_invertible());
        assert!(!Mat3::zero().is_invertible());
        assert!(!Mat4::zero().is_invertible());
        assert!(!Mat2::nan().is_invertible());
        assert!(!Mat3::nan().is_invertible());
        assert!(!Mat4::nan().is_invertible());
        assert_eq!(Mat2::identity().inverse(), Some(Mat2::identity()));
        assert_eq!(Mat3::identity().inverse(), Some(Mat3::identity()));
        assert_eq!(Mat4::identity().inverse(), Some(Mat4::identity()));

        assert!(DMat2::identity().is_invertible());
        assert!(DMat3::identity().is_invertible());
        assert!(DMat4::identity().is_invertible());
        assert!(!DMat2::zero().is_invertible());
        assert!(!DMat3::zero().is_invertible());
        assert!(!DMat4::zero().is_invertible());
        assert!(!DMat2::nan().is_invertible());
        assert!(!DMat3::nan().is_invertible());
        assert!(!DMat4::nan().is_invertible());
        assert_eq!(DMat2::identity().inverse(), Some(DMat2::identity()));
        assert_eq!(DMat3::identity().inverse(), Some(DMat3::identity()));
        assert_eq!(DMat4::identity().inverse(), Some(DMat4::identity()));

        {
            assert!(!Mat2::zeroed().is_invertible());
            assert!(!Mat2::from_cols([Vec2::zero(), Vec2::one()]).is_invertible());
            assert!(!Mat2::from_cols([Vec2::one(), Vec2::zero()]).is_invertible());
            assert!(!Mat2::from_rows([Vec2::zero(), Vec2::one()]).is_invertible());
            assert!(!Mat2::from_rows([Vec2::one(), Vec2::zero()]).is_invertible());
            assert!(Mat2::identity().is_invertible());
        }
        {
            assert!(!Mat3::zeroed().is_invertible());
            assert!(!Mat3::from_cols([Vec3::zero(), Vec3::one(), Vec3::one()]).is_invertible());
            assert!(!Mat3::from_cols([Vec3::one(), Vec3::zero(), Vec3::one()]).is_invertible());
            assert!(!Mat3::from_cols([Vec3::one(), Vec3::one(), Vec3::zero()]).is_invertible());
            assert!(!Mat3::from_rows([Vec3::zero(), Vec3::one(), Vec3::one()]).is_invertible());
            assert!(!Mat3::from_rows([Vec3::one(), Vec3::zero(), Vec3::one()]).is_invertible());
            assert!(!Mat3::from_rows([Vec3::one(), Vec3::one(), Vec3::zero()]).is_invertible());
            assert!(Mat3::identity().is_invertible());
        }
        {
            assert!(!Mat4::zeroed().is_invertible());
            assert!(
                !Mat4::from_cols([Vec4::zero(), Vec4::one(), Vec4::one(), Vec4::one()])
                    .is_invertible()
            );
            assert!(
                !Mat4::from_cols([Vec4::one(), Vec4::zero(), Vec4::one(), Vec4::one()])
                    .is_invertible()
            );
            assert!(
                !Mat4::from_cols([Vec4::one(), Vec4::one(), Vec4::zero(), Vec4::one()])
                    .is_invertible()
            );
            assert!(
                !Mat4::from_cols([Vec4::one(), Vec4::one(), Vec4::one(), Vec4::zero()])
                    .is_invertible()
            );
            assert!(
                !Mat4::from_rows([Vec4::zero(), Vec4::one(), Vec4::one(), Vec4::one()])
                    .is_invertible()
            );
            assert!(
                !Mat4::from_rows([Vec4::one(), Vec4::zero(), Vec4::one(), Vec4::one()])
                    .is_invertible()
            );
            assert!(
                !Mat4::from_rows([Vec4::one(), Vec4::one(), Vec4::zero(), Vec4::one()])
                    .is_invertible()
            );
            assert!(
                !Mat4::from_rows([Vec4::one(), Vec4::one(), Vec4::one(), Vec4::zero()])
                    .is_invertible()
            );
            assert!(Mat4::identity().is_invertible());
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn debug_print() {
        extern crate alloc;

        let m4 = Mat4::identity();

        let s = alloc::format!("{:?}", m4);
        assert_eq!(s, "[(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)]");
    }
}

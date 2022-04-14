//! Matrix types.
//!
//! Matrices do not have a [`Unit`](crate::Unit), because their values do not
//! necessarily have a clear logical meaning in the context of any particular
//! unit.
//!
//! Similarly, matrices do not use [`Scalar`](crate::Unit). A matrix of "angle"
//! components is not really meaningful.
//!
//! Instead, they are based on fundamental floating-point
//! [primitive](crate::bindings::Primitive) scalars (`f32` or `f64`).

use core::ops::Mul;

use crate::{
    bindings::{
        Matrix, Matrix2 as SimdMatrix2, Matrix3 as SimdMatrix3, Matrix4 as SimdMatrix4,
        PrimitiveMatrices,
    },
    unit::UnitMatrices,
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
                Self::with_cols(rows).transpose()
            }

            #[doc = "Create from columns with implicit conversion."]
            #[inline]
            #[must_use]
            pub fn with_cols<U>(rows: [U; $dimensions]) -> Self
            where
                U: Into<$axis_vector_ty<T>>,
            {
                bytemuck::cast(rows.map(Into::into))
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
                bytemuck::cast(*self)
            }

            #[doc = "Get row vectors."]
            #[inline]
            #[must_use]
            pub fn to_rows(&self) -> [$axis_vector_ty<T>; $dimensions] {
                self.transpose().to_cols()
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
                d != T::ZERO && crate::Scalar::is_finite(d)
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

            #[doc = "Return the transposed matrix."]
            #[inline]
            #[must_use]
            pub fn transpose(&self) -> Self {
                Self::from_raw(self.as_raw().transpose())
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
                Self::IDENTITY
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
    /// All zeroes.
    pub const ZERO: Self = Self {
        m11: T::ZERO,
        m12: T::ZERO,
        m21: T::ZERO,
        m22: T::ZERO,
    };
    /// All NaNs.
    pub const NAN: Self = Self {
        m11: T::NAN,
        m12: T::NAN,
        m21: T::NAN,
        m22: T::NAN,
    };
    /// Identity matrix
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix2::<f32>::IDENTITY;
    /// assert_eq!(matrix.row(0), vec2!(1.0, 0.0));
    /// assert_eq!(matrix.row(1), vec2!(0.0, 1.0));
    /// ```
    pub const IDENTITY: Self = Self {
        m11: T::ONE,
        m12: T::ZERO,
        m21: T::ZERO,
        m22: T::ONE,
    };

    /// Create from diagonal.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// assert_eq!(Matrix2::from_diagonal(Vector2::<f32>::ONE), Matrix2::IDENTITY);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_diagonal(Vector2 { x, y }: Vector2<T>) -> Self {
        Matrix2 {
            m11: x,
            m22: y,
            ..Self::ZERO
        }
    }

    /// Create from column vectors.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    ///
    /// let m = Matrix2::<f32>::from_cols(vec2!(0.0, 1.0), vec2!(2.0, 3.0));
    /// assert_eq!(m.col(0), (0.0, 1.0));
    /// assert_eq!(m.col(1), (2.0, 3.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_cols(x_axis: Vector2<T>, y_axis: Vector2<T>) -> Self {
        Self::from_raw(T::Mat2::from_cols(x_axis.to_raw(), y_axis.to_raw()))
    }

    /// Create from row vectors.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    ///
    /// let m = Matrix2::<f32>::from_rows(vec2!(0.0, 1.0), vec2!(2.0, 3.0));
    /// assert_eq!(m.col(0), (0.0, 2.0));
    /// assert_eq!(m.col(1), (1.0, 3.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_rows(x_axis: Vector2<T>, y_axis: Vector2<T>) -> Self {
        Self::from_cols(x_axis, y_axis).transpose()
    }

    /// Scaling matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix2::<f32>::from_scale((2.0, 3.0).into());
    /// assert_eq!(matrix.row(0), vec2!(2.0, 0.0));
    /// assert_eq!(matrix.row(1), vec2!(0.0, 3.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_scale(scale: Vector2<T>) -> Self {
        Self::from_raw(T::Mat2::from_scale_angle(scale.to_raw(), T::ZERO))
    }

    /// Rotation matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    /// let theta = Angle::from_degrees(90.0);
    /// let matrix = Matrix2::<f32>::from_angle(theta);
    /// assert_abs_diff_eq!(matrix.row(0), vec2!(0.0, -1.0));
    /// assert_abs_diff_eq!(matrix.row(1), vec2!(1.0,  0.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_angle(angle: Angle<T>) -> Self {
        Self::from_raw(T::Mat2::from_angle(angle.radians))
    }

    /// Create from 3x3 matrix, discarding the third row and column.
    ///
    /// ### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let m3 = Matrix3::<f32>::from_cols(
    ///     vec3!(1.0, 2.0, 3.0),
    ///     vec3!(4.0, 5.0, 6.0),
    ///     vec3!(7.0, 8.0, 9.0),
    /// );
    /// let m2 = Matrix2::<f32>::from_cols(
    ///     vec2!(1.0, 2.0),
    ///     vec2!(4.0, 5.0),
    /// );
    /// assert_eq!(Matrix2::from_mat3(m3), m2);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_mat3(mat3: Matrix3<T>) -> Matrix2<T> {
        Matrix2 {
            m11: mat3.m11,
            m12: mat3.m12,
            m21: mat3.m21,
            m22: mat3.m22,
        }
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
    /// approx::assert_abs_diff_eq!(rotated, point!(0.0, 1.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_point(&self, point: Point2<T>) -> Point2<T> {
        Point2::from_raw(self.as_raw().mul_vec2(point.to_raw()))
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
    /// approx::assert_abs_diff_eq!(rotated, vec2!(0.0, 1.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_vector(&self, vector: Vector2<T>) -> Vector2<T> {
        Vector2::from_raw(self.as_raw().mul_vec2(vector.to_raw()))
    }
}

impl<T> Matrix3<T>
where
    T: PrimitiveMatrices,
{
    /// All zeroes.
    pub const ZERO: Self = Self {
        m11: T::ZERO,
        m12: T::ZERO,
        m13: T::ZERO,
        m21: T::ZERO,
        m22: T::ZERO,
        m23: T::ZERO,
        m31: T::ZERO,
        m32: T::ZERO,
        m33: T::ZERO,
    };
    /// All NaNs.
    pub const NAN: Self = Self {
        m11: T::NAN,
        m12: T::NAN,
        m13: T::NAN,
        m21: T::NAN,
        m23: T::NAN,
        m22: T::NAN,
        m31: T::NAN,
        m32: T::NAN,
        m33: T::NAN,
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
        m11: T::ONE,
        m12: T::ZERO,
        m13: T::ZERO,
        m21: T::ZERO,
        m22: T::ONE,
        m23: T::ZERO,
        m31: T::ZERO,
        m32: T::ZERO,
        m33: T::ONE,
    };

    /// Create from diagonal.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// assert_eq!(
    ///     Matrix3::from_diagonal(Vector3::<f32>::ONE),
    ///     Matrix3::IDENTITY
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn from_diagonal(Vector3 { x, y, z }: Vector3<T>) -> Self {
        Matrix3 {
            m11: x,
            m22: y,
            m33: z,
            ..Self::ZERO
        }
    }

    /// Create from column vectors.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    ///
    /// let m = Matrix3::<f32>::from_cols(
    ///     vec3!(0.0, 1.0, 2.0),
    ///     vec3!(3.0, 4.0, 5.0),
    ///     vec3!(6.0, 7.0, 8.0));
    /// assert_eq!(m.col(0), (0.0, 1.0, 2.0));
    /// assert_eq!(m.col(1), (3.0, 4.0, 5.0));
    /// assert_eq!(m.col(2), (6.0, 7.0, 8.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_cols(x_axis: Vector3<T>, y_axis: Vector3<T>, z_axis: Vector3<T>) -> Self {
        Self::from_raw(T::Mat3::from_cols(
            x_axis.to_raw(),
            y_axis.to_raw(),
            z_axis.to_raw(),
        ))
    }

    /// Create from row vectors.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    ///
    /// let m = Matrix3::<f32>::from_rows(
    ///     vec3!(0.0, 1.0, 2.0),
    ///     vec3!(3.0, 4.0, 5.0),
    ///     vec3!(6.0, 7.0, 8.0));
    /// assert_eq!(m.col(0), (0.0, 3.0, 6.0));
    /// assert_eq!(m.col(1), (1.0, 4.0, 7.0));
    /// assert_eq!(m.col(2), (2.0, 5.0, 8.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_rows(x_axis: Vector3<T>, y_axis: Vector3<T>, z_axis: Vector3<T>) -> Self {
        Self::from_cols(x_axis, y_axis, z_axis).transpose()
    }

    /// Scaling matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix3::<f32>::from_scale((2.0, 3.0).into());
    /// assert_eq!(matrix.row(0), vec3!(2.0, 0.0, 0.0));
    /// assert_eq!(matrix.row(1), vec3!(0.0, 3.0, 0.0));
    /// assert_eq!(matrix.row(2), vec3!(0.0, 0.0, 1.0));
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
    /// assert_abs_diff_eq!(matrix.row(0), vec3!(0.0, -1.0, 0.0));
    /// assert_abs_diff_eq!(matrix.row(1), vec3!(1.0,  0.0, 0.0));
    /// assert_abs_diff_eq!(matrix.row(2), vec3!(0.0,  0.0, 1.0));
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
    /// assert_abs_diff_eq!(matrix.row(0), vec3!(1.0, 0.0, 10.0));
    /// assert_abs_diff_eq!(matrix.row(1), vec3!(0.0, 1.0, 20.0));
    /// assert_abs_diff_eq!(matrix.row(2), vec3!(0.0, 0.0,  1.0));
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
    /// assert_abs_diff_eq!(matrix.row(0), vec3!(0.0, -3.0, 10.0));
    /// assert_abs_diff_eq!(matrix.row(1), vec3!(2.0,  0.0, 20.0));
    /// assert_abs_diff_eq!(matrix.row(2), vec3!(0.0,  0.0,  1.0));
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

    /// Create affine transformation matrix from the given 2x2 matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let m2 = Matrix2::<f32>::from_cols(
    ///     vec2!(1.0, 2.0),
    ///     vec2!(3.0, 4.0),
    /// );
    /// let m3 = Matrix3::<f32>::from_cols(
    ///     vec3!(1.0, 2.0, 0.0),
    ///     vec3!(3.0, 4.0, 0.0),
    ///     vec3!(0.0, 0.0, 1.0),
    /// );
    /// assert_eq!(Matrix3::from_mat2(m2), m3);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_mat2(mat2: Matrix2<T>) -> Self {
        Matrix3 {
            m11: mat2.m11,
            m12: mat2.m12,
            m21: mat2.m21,
            m22: mat2.m22,
            ..Self::IDENTITY
        }
    }

    /// Create from 4x4 matrix, discarding the fourth row and column.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let m4 = Matrix4::<f32>::from_cols(
    ///     vec4!( 1.0,  2.0,  3.0,  4.0),
    ///     vec4!( 5.0,  6.0,  7.0,  8.0),
    ///     vec4!( 9.0, 10.0, 11.0, 12.0),
    ///     vec4!(13.0, 14.0, 15.0, 16.0),
    /// );
    /// let m3 = Matrix3::<f32>::from_cols(
    ///     vec3!( 1.0,  2.0,  3.0),
    ///     vec3!( 5.0,  6.0,  7.0),
    ///     vec3!( 9.0, 10.0, 11.0),
    /// );
    /// assert_eq!(Matrix3::from_mat4(m4), m3);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_mat4(mat4: Matrix4<T>) -> Self {
        Matrix3 {
            m11: mat4.m11,
            m12: mat4.m12,
            m13: mat4.m13,
            m21: mat4.m21,
            m22: mat4.m22,
            m23: mat4.m23,
            m31: mat4.m31,
            m32: mat4.m32,
            m33: mat4.m33,
        }
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
    /// approx::assert_abs_diff_eq!(rotated, point!(0.0, 1.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn transform_point<U: UnitMatrices<Vec2 = T::Vec2>>(&self, point: Point2<U>) -> Point2<U> {
        Point2::from_raw(self.as_raw().transform_point2(point.to_raw()))
    }

    /// Transform 2D vector.
    ///
    /// See [`glam::Mat3::transform_vector2()`] or
    /// [`glam::DMat3::transform_vector2()`] (depending on the scalar).
    #[inline]
    #[must_use]
    pub fn transform_vector<U: UnitMatrices<Vec2 = T::Vec2>>(
        &self,
        vector: Vector2<U>,
    ) -> Vector2<U> {
        Vector2::from_raw(self.as_raw().transform_vector2(vector.to_raw()))
    }
}

impl From<glam::Mat3A> for Matrix3<f32> {
    fn from(mat: glam::Mat3A) -> Self {
        Self::from_raw(mat.into())
    }
}

impl From<Matrix3<f32>> for glam::Mat3A {
    fn from(mat: Matrix3<f32>) -> Self {
        mat.to_raw().into()
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

impl<T> Mul<Vector2<T>> for Matrix2<T::Primitive>
where
    T: UnitMatrices,
{
    type Output = Vector2<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Vector2<T>) -> Self::Output {
        Vector2::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl<T> Mul<Point2<T>> for Matrix2<T::Primitive>
where
    T: UnitMatrices,
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

impl<T> Mul<Vector3<T>> for Matrix3<T::Primitive>
where
    T: UnitMatrices,
{
    type Output = Vector3<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        Vector3::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl<T> Mul<Point3<T>> for Matrix3<T::Primitive>
where
    T: UnitMatrices,
{
    type Output = Point3<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Point3<T>) -> Self::Output {
        Point3::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl<T> Mul<Vector2<T>> for Matrix3<T::Primitive>
where
    T: UnitMatrices,
{
    type Output = Vector2<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Vector2<T>) -> Self::Output {
        self.transform_vector(rhs)
    }
}

impl<T> Mul<Point2<T>> for Matrix3<T::Primitive>
where
    T: UnitMatrices,
{
    type Output = Point2<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Point2<T>) -> Self::Output {
        self.transform_point(rhs)
    }
}

impl<T> Mul<Vector4<T>> for Matrix4<T::Primitive>
where
    T: UnitMatrices,
{
    type Output = Vector4<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Vector4<T>) -> Self::Output {
        Vector4::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl<T> Mul<Point4<T>> for Matrix4<T::Primitive>
where
    T: UnitMatrices,
{
    type Output = Point4<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Point4<T>) -> Self::Output {
        Point4::from_raw(self.to_raw() * rhs.to_raw())
    }
}

impl<T> Mul<Vector3<T>> for Matrix4<T::Primitive>
where
    T: UnitMatrices,
{
    type Output = Vector3<T>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        self.transform_vector(rhs)
    }
}

impl<T> Mul<Point3<T>> for Matrix4<T::Primitive>
where
    T: UnitMatrices,
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
    /// All zeroes.
    pub const ZERO: Self = Self {
        m11: T::ZERO,
        m12: T::ZERO,
        m13: T::ZERO,
        m14: T::ZERO,
        m21: T::ZERO,
        m22: T::ZERO,
        m23: T::ZERO,
        m24: T::ZERO,
        m31: T::ZERO,
        m32: T::ZERO,
        m33: T::ZERO,
        m34: T::ZERO,
        m41: T::ZERO,
        m42: T::ZERO,
        m43: T::ZERO,
        m44: T::ZERO,
    };
    /// All NaNs.
    pub const NAN: Self = Self {
        m11: T::NAN,
        m12: T::NAN,
        m13: T::NAN,
        m14: T::NAN,
        m21: T::NAN,
        m22: T::NAN,
        m23: T::NAN,
        m24: T::NAN,
        m31: T::NAN,
        m32: T::NAN,
        m33: T::NAN,
        m34: T::NAN,
        m41: T::NAN,
        m42: T::NAN,
        m43: T::NAN,
        m44: T::NAN,
    };
    /// Identity matrix
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let matrix = Matrix4::<f32>::IDENTITY;
    /// assert_eq!(matrix.row(0), (1.0, 0.0, 0.0, 0.0));
    /// assert_eq!(matrix.row(1), (0.0, 1.0, 0.0, 0.0));
    /// assert_eq!(matrix.row(2), (0.0, 0.0, 1.0, 0.0));
    /// assert_eq!(matrix.row(3), (0.0, 0.0, 0.0, 1.0));
    /// ```
    pub const IDENTITY: Self = Self {
        m11: T::ONE,
        m12: T::ZERO,
        m13: T::ZERO,
        m14: T::ZERO,
        m21: T::ZERO,
        m22: T::ONE,
        m23: T::ZERO,
        m24: T::ZERO,
        m31: T::ZERO,
        m32: T::ZERO,
        m33: T::ONE,
        m34: T::ZERO,
        m41: T::ZERO,
        m42: T::ZERO,
        m43: T::ZERO,
        m44: T::ONE,
    };

    /// Create from diagonal.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// assert_eq!(
    ///     Matrix4::from_diagonal(Vector4::<f32>::ONE),
    ///     Matrix4::IDENTITY
    /// );
    /// ```
    #[inline]
    #[must_use]
    pub fn from_diagonal(Vector4 { x, y, z, w }: Vector4<T>) -> Self {
        Matrix4 {
            m11: x,
            m22: y,
            m33: z,
            m44: w,
            ..Self::ZERO
        }
    }

    /// Create from column vectors.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    ///
    /// let _m = Matrix4::<f32>::from_cols(
    ///     vec4!( 0.0,  1.0,  2.0,  3.0),
    ///     vec4!( 4.0,  5.0,  6.0,  7.0),
    ///     vec4!( 8.0,  9.0, 10.0, 11.0),
    ///     vec4!(12.0, 13.0, 14.0, 15.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_cols(
        x_axis: Vector4<T>,
        y_axis: Vector4<T>,
        z_axis: Vector4<T>,
        w_axis: Vector4<T>,
    ) -> Self {
        Self::from_raw(T::Mat4::from_cols(
            x_axis.to_raw(),
            y_axis.to_raw(),
            z_axis.to_raw(),
            w_axis.to_raw(),
        ))
    }

    /// Create from row vectors.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    ///
    /// let m = Matrix4::<f32>::from_rows(
    ///     vec4!( 0.0,  1.0,  2.0,  3.0),
    ///     vec4!( 4.0,  5.0,  6.0,  7.0),
    ///     vec4!( 8.0,  9.0, 10.0, 11.0),
    ///     vec4!(12.0, 13.0, 14.0, 15.0));
    /// assert_eq!(m.col(0), (0.0, 4.0,  8.0, 12.0));
    /// assert_eq!(m.col(1), (1.0, 5.0,  9.0, 13.0));
    /// assert_eq!(m.col(2), (2.0, 6.0, 10.0, 14.0));
    /// assert_eq!(m.col(3), (3.0, 7.0, 11.0, 15.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_rows(
        x_axis: Vector4<T>,
        y_axis: Vector4<T>,
        z_axis: Vector4<T>,
        w_axis: Vector4<T>,
    ) -> Self {
        Self::from_cols(x_axis, y_axis, z_axis, w_axis).transpose()
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
    /// let matrix = Matrix4::<f32>::from_axis_angle(Vector3::Z, theta);
    /// assert_abs_diff_eq!(matrix.row(0), vec4!(0.0, -1.0, 0.0, 0.0));
    /// assert_abs_diff_eq!(matrix.row(1), vec4!(1.0,  0.0, 0.0, 0.0));
    /// assert_abs_diff_eq!(matrix.row(2), vec4!(0.0,  0.0, 1.0, 0.0));
    /// assert_abs_diff_eq!(matrix.row(3), vec4!(0.0,  0.0, 0.0, 1.0));
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
    /// assert_abs_diff_eq!(matrix.row(0), vec4!(1.0, 0.0, 0.0, 10.0));
    /// assert_abs_diff_eq!(matrix.row(1), vec4!(0.0, 1.0, 0.0, 20.0));
    /// assert_abs_diff_eq!(matrix.row(2), vec4!(0.0, 0.0, 1.0, 30.0));
    /// assert_abs_diff_eq!(matrix.row(3), vec4!(0.0, 0.0, 0.0,  1.0));
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

    /// Creates an affine transformation matrix from the given 3x3 linear transformation matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let m3 = Matrix3::<f32>::from_cols(
    ///     vec3!(1.0, 2.0, 3.0),
    ///     vec3!(4.0, 5.0, 6.0),
    ///     vec3!(7.0, 8.0, 9.0),
    /// );
    /// let m4 = Matrix4::<f32>::from_cols(
    ///     vec4!(1.0, 2.0, 3.0, 0.0),
    ///     vec4!(4.0, 5.0, 6.0, 0.0),
    ///     vec4!(7.0, 8.0, 9.0, 0.0),
    ///     vec4!(0.0, 0.0, 0.0, 1.0),
    /// );
    /// assert_eq!(Matrix4::from_mat3(m3), m4);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_mat3(mat3: Matrix3<T>) -> Self {
        Matrix4 {
            m11: mat3.m11,
            m12: mat3.m12,
            m13: mat3.m13,
            m21: mat3.m21,
            m22: mat3.m22,
            m23: mat3.m23,
            m31: mat3.m31,
            m32: mat3.m32,
            m33: mat3.m33,
            ..Self::IDENTITY
        }
    }

    /// See [`glam::Mat4::look_at_lh()`] or [`glam::DMat4::look_at_lh()`].
    pub fn look_at_lh(eye: Point3<T>, center: Point3<T>, up: Vector3<T>) -> Self {
        Self::from_raw(T::Mat4::look_at_lh(
            eye.to_raw(),
            center.to_raw(),
            up.to_raw(),
        ))
    }

    /// See [`glam::Mat4::look_at_rh()`] or [`glam::DMat4::look_at_rh()`].
    pub fn look_at_rh(eye: Point3<T>, center: Point3<T>, up: Vector3<T>) -> Self {
        Self::from_raw(T::Mat4::look_at_rh(
            eye.to_raw(),
            center.to_raw(),
            up.to_raw(),
        ))
    }

    /// See [`glam::Mat4::perspective_rh_gl()`] or
    /// [`glam::DMat4::perspective_rh_gl()`].
    pub fn perspective_rh_gl(fov_y_radians: T, aspect_ratio: T, z_near: T, z_far: T) -> Self {
        Self::from_raw(T::Mat4::perspective_rh_gl(
            fov_y_radians,
            aspect_ratio,
            z_near,
            z_far,
        ))
    }

    /// See [`glam::Mat4::perspective_lh()`] or
    /// [`glam::DMat4::perspective_lh()`].
    pub fn perspective_lh(fov_y_radians: T, aspect_ratio: T, z_near: T, z_far: T) -> Self {
        Self::from_raw(T::Mat4::perspective_lh(
            fov_y_radians,
            aspect_ratio,
            z_near,
            z_far,
        ))
    }

    /// See [`glam::Mat4::perspective_rh()`] or
    /// [`glam::DMat4::perspective_rh()`].
    pub fn perspective_rh(fov_y_radians: T, aspect_ratio: T, z_near: T, z_far: T) -> Self {
        Self::from_raw(T::Mat4::perspective_rh(
            fov_y_radians,
            aspect_ratio,
            z_near,
            z_far,
        ))
    }

    /// See [`glam::Mat4::perspective_infinite_lh()`] or
    /// [`glam::DMat4::perspective_infinite_lh()`].
    pub fn perspective_infinite_lh(fov_y_radians: T, aspect_ratio: T, z_near: T) -> Self {
        Self::from_raw(T::Mat4::perspective_infinite_lh(
            fov_y_radians,
            aspect_ratio,
            z_near,
        ))
    }

    /// See [`glam::Mat4::perspective_infinite_reverse_lh()`] or
    /// [`glam::DMat4::perspective_infinite_reverse_lh()`].
    pub fn perspective_infinite_reverse_lh(fov_y_radians: T, aspect_ratio: T, z_near: T) -> Self {
        Self::from_raw(T::Mat4::perspective_infinite_reverse_lh(
            fov_y_radians,
            aspect_ratio,
            z_near,
        ))
    }

    /// See [`glam::Mat4::perspective_infinite_rh()`] or
    /// [`glam::DMat4::perspective_infinite_rh()`].
    pub fn perspective_infinite_rh(fov_y_radians: T, aspect_ratio: T, z_near: T) -> Self {
        Self::from_raw(T::Mat4::perspective_infinite_rh(
            fov_y_radians,
            aspect_ratio,
            z_near,
        ))
    }

    /// See [`glam::Mat4::perspective_infinite_reverse_rh()`] or
    /// [`glam::DMat4::perspective_infinite_reverse_rh()`].
    pub fn perspective_infinite_reverse_rh(fov_y_radians: T, aspect_ratio: T, z_near: T) -> Self {
        Self::from_raw(T::Mat4::perspective_infinite_reverse_rh(
            fov_y_radians,
            aspect_ratio,
            z_near,
        ))
    }

    /// See [`glam::Mat4::orthographic_rh_gl()`] or
    /// [`glam::DMat4::orthographic_rh_gl()`].
    pub fn orthographic_rh_gl(left: T, right: T, bottom: T, top: T, near: T, far: T) -> Self {
        Self::from_raw(T::Mat4::orthographic_rh_gl(
            left, right, bottom, top, near, far,
        ))
    }

    /// See [`glam::Mat4::orthographic_lh()`] or
    /// [`glam::DMat4::orthographic_lh()`].
    pub fn orthographic_lh(left: T, right: T, bottom: T, top: T, near: T, far: T) -> Self {
        Self::from_raw(T::Mat4::orthographic_lh(
            left, right, bottom, top, near, far,
        ))
    }

    /// See [`glam::Mat4::orthographic_rh()`] or
    /// [`glam::DMat4::orthographic_rh()`].
    pub fn orthographic_rh(left: T, right: T, bottom: T, top: T, near: T, far: T) -> Self {
        Self::from_raw(T::Mat4::orthographic_rh(
            left, right, bottom, top, near, far,
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
        Point3::from_raw(self.as_raw().transform_point3(point.to_raw()))
    }

    /// Transform 3D vector.
    ///
    /// See [`glam::Mat4::transform_vector3()`] or
    /// [`glam::DMat4::transform_vector3()`] (depending on the scalar).
    #[inline]
    #[must_use]
    pub fn transform_vector<U: UnitMatrices<Vec3 = T::Vec3>>(
        &self,
        vector: Vector3<U>,
    ) -> Vector3<U> {
        Vector3::from_raw(self.as_raw().transform_vector3(vector.to_raw()))
    }

    /// Project 3D point.
    ///
    /// Transform the point, including perspective correction.
    ///
    /// See [`glam::Mat4::project_point3()`] or
    /// [`glam::DMat4::project_point3()`] (depending on the scalar).
    #[inline]
    #[must_use]
    pub fn project_point<U: UnitMatrices<Vec3 = T::Vec3>>(&self, point: Point3<U>) -> Point3<U> {
        Point3::from_raw(self.as_raw().project_point3(point.to_raw()))
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
    T: PrimitiveMatrices,
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
    T: PrimitiveMatrices,
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
    T: PrimitiveMatrices,
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
    T: PrimitiveMatrices,
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
    T: PrimitiveMatrices,
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
    T: PrimitiveMatrices,
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
    T: PrimitiveMatrices,
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
    T: PrimitiveMatrices,
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
    T: PrimitiveMatrices,
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

    use crate::{point2, point3, vec2, vec3};

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

        assert_eq!(m2 * Vec2::ONE, vec2!(2.0, 3.0));
        assert_eq!(m2 * Point2::ONE, point2!(2.0, 3.0));
        assert_eq!(m3 * Vec2::ONE, vec2!(2.0, 3.0));
        assert_eq!(m3 * Point2::ONE, point2!(2.0, 3.0));

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

        assert_eq!(m2 * DVec2::ONE, vec2!(2.0, 3.0));
        assert_eq!(m2 * DPoint2::ONE, point2!(2.0, 3.0));
        assert_eq!(m3 * DVec2::ONE, vec2!(2.0, 3.0));
        assert_eq!(m3 * DPoint2::ONE, point2!(2.0, 3.0));
    }

    #[test]
    fn from_angle() {
        let m2 = Mat2::from_angle(Angle::from_degrees(90.0));
        let m3 = Mat3::from_angle(Angle::from_degrees(90.0));
        let m4 = Mat4::from_axis_angle(Vec3::Z, Angle::from_degrees(90.0));

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
        let m4 = DMat4::from_axis_angle(Vector3::Z, Angle::from_degrees(90.0));

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
            let axis = Vec3::Z;
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
            let axis = DVec3::Z;
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
        assert_eq!(Mat2::IDENTITY.to_cols(), [(1.0, 0.0), (0.0, 1.0)]);
        assert_eq!(
            Mat3::IDENTITY.to_cols(),
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        );
        assert_eq!(
            Mat4::IDENTITY.to_cols(),
            [
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0)
            ]
        );

        assert_eq!(DMat2::IDENTITY.to_cols(), [(1.0, 0.0), (0.0, 1.0)]);
        assert_eq!(
            DMat3::IDENTITY.to_cols(),
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        );
        assert_eq!(
            DMat4::IDENTITY.to_cols(),
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
        assert_eq!(Mat2::IDENTITY.to_rows(), [(1.0, 0.0), (0.0, 1.0)]);
        assert_eq!(
            Mat3::IDENTITY.to_rows(),
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        );
        assert_eq!(
            Mat4::IDENTITY.to_rows(),
            [
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0)
            ]
        );

        assert_eq!(DMat2::IDENTITY.to_rows(), [(1.0, 0.0), (0.0, 1.0)]);
        assert_eq!(
            DMat3::IDENTITY.to_rows(),
            [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
        );
        assert_eq!(
            DMat4::IDENTITY.to_rows(),
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
    fn from_rows() {
        assert_eq!(
            Mat2::from_rows((1.0, 0.0).into(), (0.0, 1.0).into()),
            Mat2::IDENTITY
        );
        assert_eq!(
            Mat3::from_rows(
                (1.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 1.0).into()
            ),
            Mat3::IDENTITY
        );
        assert_eq!(
            Mat4::from_rows(
                (1.0, 0.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0, 0.0).into(),
                (0.0, 0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 0.0, 1.0).into()
            ),
            Mat4::IDENTITY
        );

        assert_eq!(
            DMat2::from_rows((1.0, 0.0).into(), (0.0, 1.0).into()),
            DMat2::IDENTITY
        );
        assert_eq!(
            DMat3::from_rows(
                (1.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 1.0).into()
            ),
            DMat3::IDENTITY
        );
        assert_eq!(
            DMat4::from_rows(
                (1.0, 0.0, 0.0, 0.0).into(),
                (0.0, 1.0, 0.0, 0.0).into(),
                (0.0, 0.0, 1.0, 0.0).into(),
                (0.0, 0.0, 0.0, 1.0).into()
            ),
            DMat4::IDENTITY
        );
    }

    #[test]
    fn col_mut() {
        let mut m2 = Mat2::IDENTITY;
        let mut m3 = Mat3::IDENTITY;
        let mut m4 = Mat4::IDENTITY;
        let mut dm2 = DMat2::IDENTITY;
        let mut dm3 = DMat3::IDENTITY;
        let mut dm4 = DMat4::IDENTITY;

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
        assert!(m2.col(0).is_nan());
        assert!(m2.col(1).is_nan());

        let m3 = Mat3::NAN;
        assert!(m3.col(0).is_nan());
        assert!(m3.col(1).is_nan());
        assert!(m3.col(2).is_nan());

        let m4 = Mat4::NAN;
        assert!(m4.col(0).is_nan());
        assert!(m4.col(1).is_nan());
        assert!(m4.col(2).is_nan());
        assert!(m4.col(3).is_nan());

        let m2 = DMat2::NAN;
        assert!(m2.col(0).is_nan());
        assert!(m2.col(1).is_nan());

        let m3 = DMat3::NAN;
        assert!(m3.col(0).is_nan());
        assert!(m3.col(1).is_nan());
        assert!(m3.col(2).is_nan());

        let m4 = DMat4::NAN;
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
        assert!(Mat2::IDENTITY.is_invertible());
        assert!(Mat3::IDENTITY.is_invertible());
        assert!(Mat4::IDENTITY.is_invertible());
        assert!(!Mat2::ZERO.is_invertible());
        assert!(!Mat3::ZERO.is_invertible());
        assert!(!Mat4::ZERO.is_invertible());
        assert!(!Mat2::NAN.is_invertible());
        assert!(!Mat3::NAN.is_invertible());
        assert!(!Mat4::NAN.is_invertible());
        assert_eq!(Mat2::IDENTITY.inverse(), Some(Mat2::IDENTITY));
        assert_eq!(Mat3::IDENTITY.inverse(), Some(Mat3::IDENTITY));
        assert_eq!(Mat4::IDENTITY.inverse(), Some(Mat4::IDENTITY));

        assert!(DMat2::IDENTITY.is_invertible());
        assert!(DMat3::IDENTITY.is_invertible());
        assert!(DMat4::IDENTITY.is_invertible());
        assert!(!DMat2::ZERO.is_invertible());
        assert!(!DMat3::ZERO.is_invertible());
        assert!(!DMat4::ZERO.is_invertible());
        assert!(!DMat2::NAN.is_invertible());
        assert!(!DMat3::NAN.is_invertible());
        assert!(!DMat4::NAN.is_invertible());
        assert_eq!(DMat2::IDENTITY.inverse(), Some(DMat2::IDENTITY));
        assert_eq!(DMat3::IDENTITY.inverse(), Some(DMat3::IDENTITY));
        assert_eq!(DMat4::IDENTITY.inverse(), Some(DMat4::IDENTITY));

        {
            assert!(!Mat2::zeroed().is_invertible());
            assert!(!Mat2::from_cols(Vec2::ZERO, Vec2::ONE).is_invertible());
            assert!(!Mat2::from_cols(Vec2::ONE, Vec2::ZERO).is_invertible());
            assert!(!Mat2::from_rows(Vec2::ZERO, Vec2::ONE).is_invertible());
            assert!(!Mat2::from_rows(Vec2::ONE, Vec2::ZERO).is_invertible());
            assert!(Mat2::IDENTITY.is_invertible());
        }
        {
            assert!(!Mat3::zeroed().is_invertible());
            assert!(!Mat3::from_cols(Vec3::ZERO, Vec3::ONE, Vec3::ONE).is_invertible());
            assert!(!Mat3::from_cols(Vec3::ONE, Vec3::ZERO, Vec3::ONE).is_invertible());
            assert!(!Mat3::from_cols(Vec3::ONE, Vec3::ONE, Vec3::ZERO).is_invertible());
            assert!(!Mat3::from_rows(Vec3::ZERO, Vec3::ONE, Vec3::ONE).is_invertible());
            assert!(!Mat3::from_rows(Vec3::ONE, Vec3::ZERO, Vec3::ONE).is_invertible());
            assert!(!Mat3::from_rows(Vec3::ONE, Vec3::ONE, Vec3::ZERO).is_invertible());
            assert!(Mat3::IDENTITY.is_invertible());
        }
        {
            assert!(!Mat4::zeroed().is_invertible());
            assert!(!Mat4::from_cols(Vec4::ZERO, Vec4::ONE, Vec4::ONE, Vec4::ONE).is_invertible());
            assert!(!Mat4::from_cols(Vec4::ONE, Vec4::ZERO, Vec4::ONE, Vec4::ONE).is_invertible());
            assert!(!Mat4::from_cols(Vec4::ONE, Vec4::ONE, Vec4::ZERO, Vec4::ONE).is_invertible());
            assert!(!Mat4::from_cols(Vec4::ONE, Vec4::ONE, Vec4::ONE, Vec4::ZERO).is_invertible());
            assert!(!Mat4::from_rows(Vec4::ZERO, Vec4::ONE, Vec4::ONE, Vec4::ONE).is_invertible());
            assert!(!Mat4::from_rows(Vec4::ONE, Vec4::ZERO, Vec4::ONE, Vec4::ONE).is_invertible());
            assert!(!Mat4::from_rows(Vec4::ONE, Vec4::ONE, Vec4::ZERO, Vec4::ONE).is_invertible());
            assert!(!Mat4::from_rows(Vec4::ONE, Vec4::ONE, Vec4::ONE, Vec4::ZERO).is_invertible());
            assert!(Mat4::IDENTITY.is_invertible());
        }
    }

    #[test]
    fn mat3a() {
        let mat3 = Mat3::with_cols([(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]);
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
            Matrix4::<f32>::look_at_lh(
                point3!(1.0, 2.0, 3.0),
                point3!(4.0, 5.0, 6.0),
                vec3!(1.0, 0.0, 0.0)
            )
            .to_raw(),
            glam::Mat4::look_at_lh(
                (1.0, 2.0, 3.0).into(),
                (4.0, 5.0, 6.0).into(),
                (1.0, 0.0, 0.0).into()
            )
        );
        assert_eq!(
            Matrix4::<f32>::look_at_rh(
                point3!(1.0, 2.0, 3.0),
                point3!(4.0, 5.0, 6.0),
                vec3!(1.0, 0.0, 0.0)
            )
            .to_raw(),
            glam::Mat4::look_at_rh(
                (1.0, 2.0, 3.0).into(),
                (4.0, 5.0, 6.0).into(),
                (1.0, 0.0, 0.0).into()
            )
        );
        assert_eq!(
            Matrix4::<f32>::perspective_rh_gl(1.0, 2.0, 3.0, 4.0),
            Matrix4::from_raw(glam::Mat4::perspective_rh_gl(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            Matrix4::<f32>::perspective_lh(1.0, 2.0, 3.0, 4.0),
            Matrix4::from_raw(glam::Mat4::perspective_lh(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            Matrix4::<f32>::perspective_rh(1.0, 2.0, 3.0, 4.0),
            Matrix4::from_raw(glam::Mat4::perspective_rh(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            Matrix4::<f32>::perspective_infinite_lh(1.0, 2.0, 3.0),
            Matrix4::from_raw(glam::Mat4::perspective_infinite_lh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f32>::perspective_infinite_reverse_lh(1.0, 2.0, 3.0),
            Matrix4::from_raw(glam::Mat4::perspective_infinite_reverse_lh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f32>::perspective_infinite_rh(1.0, 2.0, 3.0),
            Matrix4::from_raw(glam::Mat4::perspective_infinite_rh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f32>::perspective_infinite_reverse_rh(1.0, 2.0, 3.0),
            Matrix4::from_raw(glam::Mat4::perspective_infinite_reverse_rh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f32>::orthographic_rh_gl(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Matrix4::from_raw(glam::Mat4::orthographic_rh_gl(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        );
        assert_eq!(
            Matrix4::<f32>::orthographic_lh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Matrix4::from_raw(glam::Mat4::orthographic_lh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        );
        assert_eq!(
            Matrix4::<f32>::orthographic_rh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Matrix4::from_raw(glam::Mat4::orthographic_rh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        );
    }

    #[test]
    fn dmat4_constructors() {
        assert_eq!(
            Matrix4::<f64>::look_at_lh(
                point3!(1.0, 2.0, 3.0),
                point3!(4.0, 5.0, 6.0),
                vec3!(1.0, 0.0, 0.0)
            )
            .to_raw(),
            glam::DMat4::look_at_lh(
                (1.0, 2.0, 3.0).into(),
                (4.0, 5.0, 6.0).into(),
                (1.0, 0.0, 0.0).into()
            )
        );
        assert_eq!(
            Matrix4::<f64>::look_at_rh(
                point3!(1.0, 2.0, 3.0),
                point3!(4.0, 5.0, 6.0),
                vec3!(1.0, 0.0, 0.0)
            )
            .to_raw(),
            glam::DMat4::look_at_rh(
                (1.0, 2.0, 3.0).into(),
                (4.0, 5.0, 6.0).into(),
                (1.0, 0.0, 0.0).into()
            )
        );
        assert_eq!(
            Matrix4::<f64>::perspective_rh_gl(1.0, 2.0, 3.0, 4.0),
            Matrix4::from_raw(glam::DMat4::perspective_rh_gl(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            Matrix4::<f64>::perspective_lh(1.0, 2.0, 3.0, 4.0),
            Matrix4::from_raw(glam::DMat4::perspective_lh(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            Matrix4::<f64>::perspective_rh(1.0, 2.0, 3.0, 4.0),
            Matrix4::from_raw(glam::DMat4::perspective_rh(1.0, 2.0, 3.0, 4.0))
        );
        assert_eq!(
            Matrix4::<f64>::perspective_infinite_lh(1.0, 2.0, 3.0),
            Matrix4::from_raw(glam::DMat4::perspective_infinite_lh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f64>::perspective_infinite_reverse_lh(1.0, 2.0, 3.0),
            Matrix4::from_raw(glam::DMat4::perspective_infinite_reverse_lh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f64>::perspective_infinite_rh(1.0, 2.0, 3.0),
            Matrix4::from_raw(glam::DMat4::perspective_infinite_rh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f64>::perspective_infinite_reverse_rh(1.0, 2.0, 3.0),
            Matrix4::from_raw(glam::DMat4::perspective_infinite_reverse_rh(1.0, 2.0, 3.0))
        );
        assert_eq!(
            Matrix4::<f64>::orthographic_rh_gl(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Matrix4::from_raw(glam::DMat4::orthographic_rh_gl(
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0
            ))
        );
        assert_eq!(
            Matrix4::<f64>::orthographic_lh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Matrix4::from_raw(glam::DMat4::orthographic_lh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        );
        assert_eq!(
            Matrix4::<f64>::orthographic_rh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            Matrix4::from_raw(glam::DMat4::orthographic_rh(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn debug_print() {
        extern crate alloc;

        let m4 = Mat4::IDENTITY;

        let s = alloc::format!("{:?}", m4);
        assert_eq!(s, "[(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)]");
    }
}

use approx::AbsDiffEq;
use num_traits::Float;

use crate::angle::Angle;

use super::{marker::ValueSemantics, Primitive, PrimitiveMatrices};

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

/// Trait describing a glam N x N matrix type.
///
/// Note: All glam matrices are square.
pub trait SimdMatrix:
    ValueSemantics
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Mul<Self::Scalar, Output = Self>
    + MulAssign<Self::Scalar>
    + Neg<Output = Self>
    + AbsDiffEq<Epsilon = <Self::Scalar as AbsDiffEq>::Epsilon>
{
    /// The component type of the `glam` matrix. Either `f32` or `f64`.
    type Scalar: PrimitiveMatrices + Primitive + Float + AbsDiffEq;

    /// Matrix with all elements set to zero.
    #[must_use]
    fn zero() -> Self;

    /// Identity matrix.
    #[must_use]
    fn identity() -> Self;

    /// Matrix with all NaNs.
    #[must_use]
    fn nan() -> Self;

    /// True if any element is NaN.
    #[must_use]
    fn is_nan(&self) -> bool;

    /// True if all elements are finite and non NaN.
    #[must_use]
    fn is_finite(&self) -> bool;

    /// Transpose the matrix.
    #[must_use]
    fn transpose(&self) -> Self;

    /// Invert the matrix.
    ///
    /// Note: If the matrix is not invertible, this returns an invalid matrix.
    /// See (e.g.) [`glam::Mat4::inverse()`].
    #[must_use]
    fn inverse(&self) -> Self;

    /// Matrix determinant.
    ///
    /// If the determinant is non-zero, the matrix is invertible.
    #[must_use]
    fn determinant(&self) -> Self::Scalar;

    /// Check if the matrix is invertible. Default implementation returns
    /// `self.determinant() != 0.0`.
    #[must_use]
    fn is_invertible(&self) -> bool;
}

/// Primitive 2x2 matrix.
///
/// Implemented for [`glam::Mat2`] and [`glam::DMat2`].
pub trait SimdMatrix2:
    SimdMatrix + Mul<<Self::Scalar as Primitive>::Vec2, Output = <Self::Scalar as Primitive>::Vec2>
{
    /// Transform point.
    ///
    /// See [`glam::Mat2::mul_vec2()`] or
    /// [`glam::DMat2::mul_vec2()`].
    fn transform_point(
        &self,
        point: <Self::Scalar as Primitive>::Vec2,
    ) -> <Self::Scalar as Primitive>::Vec2;

    /// Transform vector.
    ///
    /// See [`glam::Mat2::mul_vec2()`] or
    /// [`glam::DMat2::mul_vec2()`].
    fn transform_vector(
        &self,
        vector: <Self::Scalar as Primitive>::Vec2,
    ) -> <Self::Scalar as Primitive>::Vec2;

    /// Create from column vectors.
    fn from_cols(cols: [<Self::Scalar as Primitive>::Vec2; 2]) -> Self;

    /// Convert to column vectors.
    fn to_cols(self) -> [<Self::Scalar as Primitive>::Vec2; 2];

    /// Convert to row vectors.
    fn to_rows(self) -> [<Self::Scalar as Primitive>::Vec2; 2];

    /// Get column at `index`.
    fn col(&self, index: usize) -> <Self::Scalar as Primitive>::Vec2;
    /// Get a mutable reference to column at `index`.
    fn col_mut(&mut self, index: usize) -> &mut <Self::Scalar as Primitive>::Vec2;
    /// Get row at `index`.
    fn row(&self, index: usize) -> <Self::Scalar as Primitive>::Vec2;

    /// 2D scaling matrix.
    fn from_scale(vector: <Self::Scalar as Primitive>::Vec2) -> Self;

    /// 2D rotation matrix.
    fn from_angle(angle: Angle<Self::Scalar>) -> Self;
}

/// Primitive 3x3 matrix.
///
/// Implemented for [`glam::Mat3`] and [`glam::DMat3`].
pub trait SimdMatrix3:
    SimdMatrix + Mul<<Self::Scalar as Primitive>::Vec3, Output = <Self::Scalar as Primitive>::Vec3>
{
    /// Transform point.
    ///
    /// See [`glam::Mat3::transform_point2()`] or
    /// [`glam::DMat3::transform_point2()`].
    fn transform_point(
        &self,
        point: <Self::Scalar as Primitive>::Vec2,
    ) -> <Self::Scalar as Primitive>::Vec2;

    /// Create from column vectors.
    fn from_cols(cols: [<Self::Scalar as Primitive>::Vec3; 3]) -> Self;

    /// Convert to column vectors.
    fn to_cols(self) -> [<Self::Scalar as Primitive>::Vec3; 3];

    /// Convert to row vectors.
    fn to_rows(self) -> [<Self::Scalar as Primitive>::Vec3; 3];

    /// Get column at `index`.
    fn col(&self, index: usize) -> <Self::Scalar as Primitive>::Vec3;
    /// Get a mutable reference to column at `index`.
    fn col_mut(&mut self, index: usize) -> &mut <Self::Scalar as Primitive>::Vec3;
    /// Get row at `index`.
    fn row(&self, index: usize) -> <Self::Scalar as Primitive>::Vec3;

    /// Transform vector.
    ///
    /// See [`glam::Mat3::transform_vector2()`] or
    /// [`glam::DMat3::transform_vector2()`].
    fn transform_vector(
        &self,
        vector: <Self::Scalar as Primitive>::Vec2,
    ) -> <Self::Scalar as Primitive>::Vec2;

    /// Scaling matrix
    fn from_scale(vector: <Self::Scalar as Primitive>::Vec2) -> Self;

    /// Rotation matrix
    fn from_angle(angle: Angle<Self::Scalar>) -> Self;

    /// 2D translation matrix.
    fn from_translation(translation: <Self::Scalar as Primitive>::Vec2) -> Self;

    /// 2D transform.
    fn from_scale_angle_translation(
        scale: <Self::Scalar as Primitive>::Vec2,
        angle: Angle<Self::Scalar>,
        translation: <Self::Scalar as Primitive>::Vec2,
    ) -> Self;
}

/// Primitive 4x4 matrix.
///
/// Implemented for [`glam::Mat4`] and [`glam::DMat4`].
pub trait SimdMatrix4:
    SimdMatrix + Mul<<Self::Scalar as Primitive>::Vec4, Output = <Self::Scalar as Primitive>::Vec4>
{
    /// Transform point.
    ///
    /// See [`glam::Mat4::transform_point3()`] or
    /// [`glam::DMat4::transform_point3()`].
    fn transform_point(
        &self,
        point: <Self::Scalar as Primitive>::Vec3,
    ) -> <Self::Scalar as Primitive>::Vec3;

    /// Transform vector.
    ///
    /// See [`glam::Mat4::transform_vector3()`] or
    /// [`glam::DMat4::transform_vector3()`].
    fn transform_vector(
        &self,
        vector: <Self::Scalar as Primitive>::Vec3,
    ) -> <Self::Scalar as Primitive>::Vec3;

    /// Project point.
    ///
    /// See [`glam::Mat4::project_point3()`] or
    /// [`glam::DMat4::project_point3()`].
    fn project_point(
        &self,
        vector: <Self::Scalar as Primitive>::Vec3,
    ) -> <Self::Scalar as Primitive>::Vec3;

    /// Create from column vectors.
    fn from_cols(cols: [<Self::Scalar as Primitive>::Vec4; 4]) -> Self;

    /// Convert to column vectors.
    fn to_cols(self) -> [<Self::Scalar as Primitive>::Vec4; 4];

    /// Convert to row vectors.
    fn to_rows(self) -> [<Self::Scalar as Primitive>::Vec4; 4];

    /// Get column at `index`.
    fn col(&self, index: usize) -> <Self::Scalar as Primitive>::Vec4;
    /// Get a mutable reference to column at `index`.
    fn col_mut(&mut self, index: usize) -> &mut <Self::Scalar as Primitive>::Vec4;
    /// Get row at `index`.
    fn row(&self, index: usize) -> <Self::Scalar as Primitive>::Vec4;

    /// Scaling matrix
    fn from_scale(vector: <Self::Scalar as Primitive>::Vec3) -> Self;

    /// Rotation matrix
    fn from_axis_angle(axis: <Self::Scalar as Primitive>::Vec3, angle: Angle<Self::Scalar>)
        -> Self;

    /// 3D translation matrix.
    fn from_translation(translation: <Self::Scalar as Primitive>::Vec3) -> Self;

    /// Scale, rotation, translation.
    fn from_scale_rotation_translation(
        scale: <Self::Scalar as Primitive>::Vec3,
        axis: <Self::Scalar as Primitive>::Vec3,
        angle: Angle<Self::Scalar>,
        translation: <Self::Scalar as Primitive>::Vec3,
    ) -> Self;
}

macro_rules! impl_matrix {
    ($scalar:ty, $n:literal, $glam_ty:ty {
        $($axes:ident),*
    }
    ) => {
        impl SimdMatrix for $glam_ty {
            type Scalar = $scalar;

            #[inline]
            fn zero() -> Self {
                <$glam_ty>::ZERO
            }

            #[inline]
            fn identity() -> Self {
                <$glam_ty>::IDENTITY
            }

            #[inline]
            fn nan() -> Self {
                <$glam_ty>::NAN
            }

            #[inline]
            fn is_nan(&self) -> bool {
                <$glam_ty>::is_nan(self)
            }

            #[inline]
            fn is_finite(&self) -> bool {
                <$glam_ty>::is_finite(self)
            }

            #[inline]
            fn determinant(&self) -> $scalar {
                <$glam_ty>::determinant(self)
            }

            #[inline]
            fn is_invertible(&self) -> bool {
                let d = <$glam_ty>::determinant(self);
                d.is_finite() && d != 0.0
            }

            #[inline]
            fn transpose(&self) -> Self {
                <$glam_ty>::transpose(self)
            }

            #[inline]
            fn inverse(&self) -> Self {
                <$glam_ty>::inverse(self)
            }
        }
    };
}

impl_matrix!(f32, 2, glam::Mat2 { x, y });
impl_matrix!(f32, 3, glam::Mat3 { x, y, z });
impl_matrix!(f32, 4, glam::Mat4 { x, y, z, w });
impl_matrix!(f64, 2, glam::DMat2 { x, y });
impl_matrix!(f64, 3, glam::DMat3 { x, y, z });
impl_matrix!(f64, 4, glam::DMat4 { x, y, z, w });

impl SimdMatrix2 for glam::Mat2 {
    #[inline]
    fn transform_point(&self, point: glam::Vec2) -> glam::Vec2 {
        self.mul_vec2(point)
    }

    #[inline]
    fn transform_vector(&self, vector: glam::Vec2) -> glam::Vec2 {
        self.mul_vec2(vector)
    }

    #[inline]
    fn from_cols([x_axis, y_axis]: [glam::Vec2; 2]) -> Self {
        <glam::Mat2>::from_cols(x_axis, y_axis)
    }

    #[inline]
    fn to_cols(self) -> [glam::Vec2; 2] {
        bytemuck::cast(self)
    }

    #[inline]
    fn to_rows(self) -> [glam::Vec2; 2] {
        [self.row(0), self.row(1)]
    }

    #[inline]
    fn col(&self, index: usize) -> glam::Vec2 {
        <glam::Mat2>::col(self, index)
    }

    #[inline]
    fn col_mut(&mut self, index: usize) -> &mut glam::Vec2 {
        <glam::Mat2>::col_mut(self, index)
    }

    #[inline]
    fn row(&self, index: usize) -> glam::Vec2 {
        <glam::Mat2>::row(self, index)
    }

    #[inline]
    fn from_scale(vector: glam::Vec2) -> Self {
        <glam::Mat2>::from_scale_angle(vector, 0.0)
    }

    #[inline]
    fn from_angle(angle: Angle<f32>) -> Self {
        <glam::Mat2>::from_angle(angle.radians)
    }
}

impl SimdMatrix2 for glam::DMat2 {
    #[inline]
    fn transform_point(&self, point: glam::DVec2) -> glam::DVec2 {
        self.mul_vec2(point)
    }

    #[inline]
    fn transform_vector(&self, vector: glam::DVec2) -> glam::DVec2 {
        self.mul_vec2(vector)
    }

    #[inline]
    fn from_cols([x_axis, y_axis]: [glam::DVec2; 2]) -> Self {
        <glam::DMat2>::from_cols(x_axis, y_axis)
    }

    #[inline]
    fn to_cols(self) -> [glam::DVec2; 2] {
        bytemuck::cast(self)
    }

    #[inline]
    fn to_rows(self) -> [glam::DVec2; 2] {
        [self.row(0), self.row(1)]
    }

    #[inline]
    fn col(&self, index: usize) -> glam::DVec2 {
        <glam::DMat2>::col(self, index)
    }

    #[inline]
    fn col_mut(&mut self, index: usize) -> &mut glam::DVec2 {
        <glam::DMat2>::col_mut(self, index)
    }

    #[inline]
    fn row(&self, index: usize) -> glam::DVec2 {
        <glam::DMat2>::row(self, index)
    }

    #[inline]
    fn from_scale(vector: glam::DVec2) -> Self {
        <glam::DMat2>::from_scale_angle(vector, 0.0)
    }

    #[inline]
    fn from_angle(angle: Angle<f64>) -> Self {
        <glam::DMat2>::from_angle(angle.radians)
    }
}

impl SimdMatrix3 for glam::Mat3 {
    #[inline]
    fn transform_point(&self, point: glam::Vec2) -> glam::Vec2 {
        self.transform_point2(point)
    }

    #[inline]
    fn from_cols([x_axis, y_axis, z_axis]: [glam::Vec3; 3]) -> Self {
        <glam::Mat3>::from_cols(x_axis, y_axis, z_axis)
    }

    #[inline]
    fn to_cols(self) -> [glam::Vec3; 3] {
        bytemuck::cast(self)
    }

    #[inline]
    fn to_rows(self) -> [glam::Vec3; 3] {
        [self.row(0), self.row(1), self.row(2)]
    }

    #[inline]
    fn col(&self, index: usize) -> glam::Vec3 {
        <glam::Mat3>::col(self, index)
    }

    #[inline]
    fn col_mut(&mut self, index: usize) -> &mut glam::Vec3 {
        <glam::Mat3>::col_mut(self, index)
    }

    #[inline]
    fn row(&self, index: usize) -> glam::Vec3 {
        <glam::Mat3>::row(self, index)
    }

    #[inline]
    fn transform_vector(&self, vector: glam::Vec2) -> glam::Vec2 {
        self.transform_vector2(vector)
    }

    #[inline]
    fn from_scale(vector: glam::Vec2) -> Self {
        <glam::Mat3>::from_scale(vector)
    }

    #[inline]
    fn from_angle(angle: Angle<f32>) -> Self {
        <glam::Mat3>::from_angle(angle.radians)
    }

    #[inline]
    fn from_translation(translation: glam::Vec2) -> Self {
        <glam::Mat3>::from_translation(translation)
    }

    #[inline]
    fn from_scale_angle_translation(
        scale: glam::Vec2,
        angle: Angle<f32>,
        translation: glam::Vec2,
    ) -> Self {
        <glam::Mat3>::from_scale_angle_translation(scale, angle.radians, translation)
    }
}

impl SimdMatrix3 for glam::DMat3 {
    #[inline]
    fn transform_point(&self, point: glam::DVec2) -> glam::DVec2 {
        self.transform_point2(point)
    }

    #[inline]
    fn from_cols([x_axis, y_axis, z_axis]: [glam::DVec3; 3]) -> Self {
        <glam::DMat3>::from_cols(x_axis, y_axis, z_axis)
    }

    #[inline]
    fn to_cols(self) -> [glam::DVec3; 3] {
        bytemuck::cast(self)
    }

    #[inline]
    fn to_rows(self) -> [glam::DVec3; 3] {
        [self.row(0), self.row(1), self.row(2)]
    }

    #[inline]
    fn col(&self, index: usize) -> glam::DVec3 {
        <glam::DMat3>::col(self, index)
    }

    #[inline]
    fn col_mut(&mut self, index: usize) -> &mut glam::DVec3 {
        <glam::DMat3>::col_mut(self, index)
    }

    #[inline]
    fn row(&self, index: usize) -> glam::DVec3 {
        <glam::DMat3>::row(self, index)
    }

    #[inline]
    fn transform_vector(&self, vector: glam::DVec2) -> glam::DVec2 {
        self.transform_vector2(vector)
    }

    #[inline]
    fn from_scale(vector: glam::DVec2) -> Self {
        <glam::DMat3>::from_scale(vector)
    }

    #[inline]
    fn from_angle(angle: Angle<f64>) -> Self {
        <glam::DMat3>::from_angle(angle.radians)
    }

    #[inline]
    fn from_translation(translation: glam::DVec2) -> Self {
        <glam::DMat3>::from_translation(translation)
    }

    #[inline]
    fn from_scale_angle_translation(
        scale: glam::DVec2,
        angle: Angle<f64>,
        translation: glam::DVec2,
    ) -> Self {
        <glam::DMat3>::from_scale_angle_translation(scale, angle.radians, translation)
    }
}

impl SimdMatrix4 for glam::Mat4 {
    #[inline]
    fn transform_point(&self, point: glam::Vec3) -> glam::Vec3 {
        self.transform_point3(point)
    }

    #[inline]
    fn transform_vector(&self, vector: glam::Vec3) -> glam::Vec3 {
        self.transform_vector3(vector)
    }

    #[inline]
    fn project_point(&self, vector: glam::Vec3) -> glam::Vec3 {
        <glam::Mat4>::project_point3(self, vector)
    }

    #[inline]
    fn from_cols([x_axis, y_axis, z_axis, w_axis]: [glam::Vec4; 4]) -> Self {
        <glam::Mat4>::from_cols(x_axis, y_axis, z_axis, w_axis)
    }

    #[inline]
    fn to_cols(self) -> [glam::Vec4; 4] {
        bytemuck::cast(self)
    }

    #[inline]
    fn to_rows(self) -> [glam::Vec4; 4] {
        [self.row(0), self.row(1), self.row(2), self.row(3)]
    }

    #[inline]
    fn col(&self, index: usize) -> glam::Vec4 {
        <glam::Mat4>::col(self, index)
    }

    #[inline]
    fn col_mut(&mut self, index: usize) -> &mut glam::Vec4 {
        <glam::Mat4>::col_mut(self, index)
    }

    #[inline]
    fn row(&self, index: usize) -> glam::Vec4 {
        <glam::Mat4>::row(self, index)
    }

    #[inline]
    fn from_scale(vector: glam::Vec3) -> Self {
        <glam::Mat4>::from_scale(vector)
    }

    #[inline]
    fn from_axis_angle(axis: glam::Vec3, angle: Angle<f32>) -> Self {
        glam::Mat4::from_axis_angle(axis, angle.radians)
    }

    #[inline]
    fn from_translation(translation: glam::Vec3) -> Self {
        <glam::Mat4>::from_translation(translation)
    }

    fn from_scale_rotation_translation(
        scale: glam::Vec3,
        axis: glam::Vec3,
        angle: Angle<f32>,
        translation: glam::Vec3,
    ) -> Self {
        let quat = glam::Quat::from_axis_angle(axis, angle.radians);
        <glam::Mat4>::from_scale_rotation_translation(scale, quat, translation)
    }
}

impl SimdMatrix4 for glam::DMat4 {
    #[inline]
    fn transform_point(&self, point: glam::DVec3) -> glam::DVec3 {
        self.transform_point3(point)
    }

    #[inline]
    fn transform_vector(&self, vector: glam::DVec3) -> glam::DVec3 {
        self.transform_vector3(vector)
    }

    #[inline]
    fn project_point(&self, vector: glam::DVec3) -> glam::DVec3 {
        <glam::DMat4>::project_point3(self, vector)
    }

    #[inline]
    fn from_cols([x_axis, y_axis, z_axis, w_axis]: [glam::DVec4; 4]) -> Self {
        <glam::DMat4>::from_cols(x_axis, y_axis, z_axis, w_axis)
    }

    #[inline]
    fn to_cols(self) -> [glam::DVec4; 4] {
        bytemuck::cast(self)
    }

    #[inline]
    fn to_rows(self) -> [glam::DVec4; 4] {
        [self.row(0), self.row(1), self.row(2), self.row(3)]
    }

    #[inline]
    fn col(&self, index: usize) -> glam::DVec4 {
        <glam::DMat4>::col(self, index)
    }

    #[inline]
    fn col_mut(&mut self, index: usize) -> &mut glam::DVec4 {
        <glam::DMat4>::col_mut(self, index)
    }

    #[inline]
    fn row(&self, index: usize) -> glam::DVec4 {
        <glam::DMat4>::row(self, index)
    }

    #[inline]
    fn from_scale(vector: glam::DVec3) -> Self {
        <glam::DMat4>::from_scale(vector)
    }

    #[inline]
    fn from_axis_angle(axis: glam::DVec3, angle: Angle<f64>) -> Self {
        <glam::DMat4>::from_axis_angle(axis, angle.radians)
    }

    #[inline]
    fn from_translation(translation: glam::DVec3) -> Self {
        <glam::DMat4>::from_translation(translation)
    }

    #[inline]
    fn from_scale_rotation_translation(
        scale: glam::DVec3,
        axis: glam::DVec3,
        angle: Angle<f64>,
        translation: glam::DVec3,
    ) -> Self {
        let quat = glam::DQuat::from_axis_angle(axis, angle.radians);
        <glam::DMat4>::from_scale_rotation_translation(scale, quat, translation)
    }
}

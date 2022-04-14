use super::*;

/// Trait describing a glam N x N matrix type.
///
/// Note: All glam matrices are square.
pub trait Matrix:
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
}

/// Primitive 2x2 matrix.
///
/// Implemented for [`glam::Mat2`] and [`glam::DMat2`].
pub trait Matrix2:
    Matrix + Mul<<Self::Scalar as Primitive>::Vec2, Output = <Self::Scalar as Primitive>::Vec2>
{
    /// Transform vector or point.
    ///
    /// See [`glam::Mat2::mul_vec2()`] or
    /// [`glam::DMat2::mul_vec2()`].
    fn mul_vec2(&self, vec: <Self::Scalar as Primitive>::Vec2)
        -> <Self::Scalar as Primitive>::Vec2;

    /// Create from column vectors.
    fn from_cols(
        x_axis: <Self::Scalar as Primitive>::Vec2,
        y_axis: <Self::Scalar as Primitive>::Vec2,
    ) -> Self;

    /// Get column at `index`.
    fn col(&self, index: usize) -> <Self::Scalar as Primitive>::Vec2;
    /// Get row at `index`.
    fn row(&self, index: usize) -> <Self::Scalar as Primitive>::Vec2;

    /// 2D scaling matrix.
    fn from_scale_angle(vector: <Self::Scalar as Primitive>::Vec2, angle: Self::Scalar) -> Self;

    /// 2D rotation matrix.
    fn from_angle(angle: Self::Scalar) -> Self;
}

/// Primitive 3x3 matrix.
///
/// Implemented for [`glam::Mat3`] and [`glam::DMat3`].
pub trait Matrix3:
    Matrix + Mul<<Self::Scalar as Primitive>::Vec3, Output = <Self::Scalar as Primitive>::Vec3>
{
    /// Create from column vectors.
    fn from_cols(
        x_axis: <Self::Scalar as Primitive>::Vec3,
        y_axis: <Self::Scalar as Primitive>::Vec3,
        z_axis: <Self::Scalar as Primitive>::Vec3,
    ) -> Self;

    /// Get column at `index`.
    fn col(&self, index: usize) -> <Self::Scalar as Primitive>::Vec3;
    /// Get row at `index`.
    fn row(&self, index: usize) -> <Self::Scalar as Primitive>::Vec3;

    /// Transform vector.
    ///
    /// See [`glam::Mat3::transform_vector2()`] or
    /// [`glam::DMat3::transform_vector2()`].
    fn transform_vector2(
        &self,
        vector: <Self::Scalar as Primitive>::Vec2,
    ) -> <Self::Scalar as Primitive>::Vec2;

    /// Transform point.
    ///
    /// See [`glam::Mat3::transform_point2()`] or
    /// [`glam::DMat3::transform_point2()`].
    fn transform_point2(
        &self,
        point: <Self::Scalar as Primitive>::Vec2,
    ) -> <Self::Scalar as Primitive>::Vec2;

    /// Scaling matrix
    fn from_scale(vector: <Self::Scalar as Primitive>::Vec2) -> Self;

    /// Rotation matrix
    fn from_angle(angle: Self::Scalar) -> Self;

    /// 2D translation matrix.
    fn from_translation(translation: <Self::Scalar as Primitive>::Vec2) -> Self;

    /// 2D transform.
    fn from_scale_angle_translation(
        scale: <Self::Scalar as Primitive>::Vec2,
        angle: Self::Scalar,
        translation: <Self::Scalar as Primitive>::Vec2,
    ) -> Self;
}

/// Primitive 4x4 matrix.
///
/// Implemented for [`glam::Mat4`] and [`glam::DMat4`].
pub trait Matrix4:
    Matrix + Mul<<Self::Scalar as Primitive>::Vec4, Output = <Self::Scalar as Primitive>::Vec4>
{
    /// Transform point.
    ///
    /// See [`glam::Mat4::transform_point3()`] or
    /// [`glam::DMat4::transform_point3()`].
    fn transform_point3(
        &self,
        point: <Self::Scalar as Primitive>::Vec3,
    ) -> <Self::Scalar as Primitive>::Vec3;

    /// Transform vector.
    ///
    /// See [`glam::Mat4::transform_vector3()`] or
    /// [`glam::DMat4::transform_vector3()`].
    fn transform_vector3(
        &self,
        vector: <Self::Scalar as Primitive>::Vec3,
    ) -> <Self::Scalar as Primitive>::Vec3;

    /// Project point.
    ///
    /// See [`glam::Mat4::project_point3()`] or
    /// [`glam::DMat4::project_point3()`].
    fn project_point3(
        &self,
        vector: <Self::Scalar as Primitive>::Vec3,
    ) -> <Self::Scalar as Primitive>::Vec3;

    /// Create from column vectors.
    fn from_cols(
        x_axis: <Self::Scalar as Primitive>::Vec4,
        y_axis: <Self::Scalar as Primitive>::Vec4,
        z_axis: <Self::Scalar as Primitive>::Vec4,
        w_axis: <Self::Scalar as Primitive>::Vec4,
    ) -> Self;

    /// Get column at `index`.
    fn col(&self, index: usize) -> <Self::Scalar as Primitive>::Vec4;

    /// Get row at `index`.
    fn row(&self, index: usize) -> <Self::Scalar as Primitive>::Vec4;

    /// Scaling matrix
    fn from_scale(vector: <Self::Scalar as Primitive>::Vec3) -> Self;

    /// Rotation matrix
    fn from_axis_angle(axis: <Self::Scalar as Primitive>::Vec3, angle: Self::Scalar) -> Self;

    /// 3D translation matrix.
    fn from_translation(translation: <Self::Scalar as Primitive>::Vec3) -> Self;

    /// Scale, rotation, translation.
    fn from_scale_rotation_translation(
        scale: <Self::Scalar as Primitive>::Vec3,
        axis: <Self::Scalar as PrimitiveMatrices>::Quat,
        translation: <Self::Scalar as Primitive>::Vec3,
    ) -> Self;
}

macro_rules! impl_matrix {
    ($scalar:ty, $glam_ty:ty) => {
        impl Matrix for $glam_ty {
            type Scalar = $scalar;

            forward_impl!($glam_ty => fn is_nan(&self) -> bool);
            forward_impl!($glam_ty => fn is_finite(&self) -> bool);
            forward_impl!($glam_ty => fn determinant(&self) -> $scalar);
            forward_impl!($glam_ty => fn transpose(&self) -> Self);
            forward_impl!($glam_ty => fn inverse(&self) -> Self);
        }
    };
}

impl_matrix!(f32, glam::Mat2);
impl_matrix!(f32, glam::Mat3);
impl_matrix!(f32, glam::Mat4);
impl_matrix!(f64, glam::DMat2);
impl_matrix!(f64, glam::DMat3);
impl_matrix!(f64, glam::DMat4);

impl Matrix2 for glam::Mat2 {
    forward_impl!(glam::Mat2 => fn from_cols(x: glam::Vec2, y: glam::Vec2) -> Self);
    forward_impl!(glam::Mat2 => fn mul_vec2(&self, vec: glam::Vec2) -> glam::Vec2);
    forward_impl!(glam::Mat2 => fn col(&self, index: usize) -> glam::Vec2);
    forward_impl!(glam::Mat2 => fn row(&self, index: usize) -> glam::Vec2);
    forward_impl!(glam::Mat2 => fn from_scale_angle(vector: glam::Vec2, angle: f32) -> Self);
    forward_impl!(glam::Mat2 => fn from_angle(angle: f32) -> Self);
}

impl Matrix2 for glam::DMat2 {
    forward_impl!(glam::DMat2 => fn from_cols(x: glam::DVec2, y: glam::DVec2) -> Self);
    forward_impl!(glam::DMat2 => fn mul_vec2(&self, vec: glam::DVec2) -> glam::DVec2);
    forward_impl!(glam::DMat2 => fn col(&self, index: usize) -> glam::DVec2);
    forward_impl!(glam::DMat2 => fn row(&self, index: usize) -> glam::DVec2);
    forward_impl!(glam::DMat2 => fn from_scale_angle(vector: glam::DVec2, angle: f64) -> Self);
    forward_impl!(glam::DMat2 => fn from_angle(angle: f64) -> Self);
}

impl Matrix3 for glam::Mat3 {
    forward_impl!(glam::Mat3 => fn transform_point2(&self, point: glam::Vec2) -> glam::Vec2);
    forward_impl!(glam::Mat3 => fn transform_vector2(&self, point: glam::Vec2) -> glam::Vec2);
    forward_impl!(glam::Mat3 => fn from_cols(x: glam::Vec3, y: glam::Vec3, z: glam::Vec3) -> Self);
    forward_impl!(glam::Mat3 => fn col(&self, index: usize) -> glam::Vec3);
    forward_impl!(glam::Mat3 => fn row(&self, index: usize) -> glam::Vec3);
    forward_impl!(glam::Mat3 => fn from_scale(vector: glam::Vec2) -> Self);
    forward_impl!(glam::Mat3 => fn from_angle(angle: f32) -> Self);
    forward_impl!(glam::Mat3 => fn from_translation(translation: glam::Vec2) -> Self);

    forward_impl!(glam::Mat3 => fn from_scale_angle_translation(
        scale: glam::Vec2,
        angle: f32,
        translation: glam::Vec2
    ) -> Self);
}

impl Matrix3 for glam::DMat3 {
    forward_impl!(glam::DMat3 => fn from_cols(x: glam::DVec3, y: glam::DVec3, z: glam::DVec3) -> Self);
    forward_impl!(glam::DMat3 => fn transform_point2(&self, point: glam::DVec2) -> glam::DVec2);
    forward_impl!(glam::DMat3 => fn transform_vector2(&self, point: glam::DVec2) -> glam::DVec2);
    forward_impl!(glam::DMat3 => fn col(&self, index: usize) -> glam::DVec3);
    forward_impl!(glam::DMat3 => fn row(&self, index: usize) -> glam::DVec3);
    forward_impl!(glam::DMat3 => fn from_scale(vector: glam::DVec2) -> Self);
    forward_impl!(glam::DMat3 => fn from_angle(angle: f64) -> Self);
    forward_impl!(glam::DMat3 => fn from_translation(translation: glam::DVec2) -> Self);

    forward_impl!(glam::DMat3 => fn from_scale_angle_translation(
        scale: glam::DVec2,
        angle: f64,
        translation: glam::DVec2
    ) -> Self);
}

impl Matrix4 for glam::Mat4 {
    forward_impl!(glam::Mat4 => fn transform_point3(&self, point: glam::Vec3) -> glam::Vec3);
    forward_impl!(glam::Mat4 => fn transform_vector3(&self, point: glam::Vec3) -> glam::Vec3);
    forward_impl!(glam::Mat4 => fn project_point3(&self, point: glam::Vec3) -> glam::Vec3);
    forward_impl!(glam::Mat4 => fn from_cols(x: glam::Vec4, y: glam::Vec4, z: glam::Vec4, w: glam::Vec4) -> Self);
    forward_impl!(glam::Mat4 => fn col(&self, index: usize) -> glam::Vec4);
    forward_impl!(glam::Mat4 => fn row(&self, index: usize) -> glam::Vec4);
    forward_impl!(glam::Mat4 => fn from_scale(vector: glam::Vec3) -> Self);
    forward_impl!(glam::Mat4 => fn from_axis_angle(axis: glam::Vec3, angle: f32) -> Self);
    forward_impl!(glam::Mat4 => fn from_translation(translation: glam::Vec3) -> Self);

    forward_impl!(glam::Mat4 => fn from_scale_rotation_translation(
        scale: glam::Vec3,
        rotation: glam::Quat,
        translation: glam::Vec3
    ) -> Self);
}

impl Matrix4 for glam::DMat4 {
    forward_impl!(glam::DMat4 => fn transform_point3(&self, point: glam::DVec3) -> glam::DVec3);
    forward_impl!(glam::DMat4 => fn transform_vector3(&self, point: glam::DVec3) -> glam::DVec3);
    forward_impl!(glam::DMat4 => fn project_point3(&self, point: glam::DVec3) -> glam::DVec3);
    forward_impl!(glam::DMat4 => fn from_cols(x: glam::DVec4, y: glam::DVec4, z: glam::DVec4, w: glam::DVec4) -> Self);
    forward_impl!(glam::DMat4 => fn col(&self, index: usize) -> glam::DVec4);
    forward_impl!(glam::DMat4 => fn row(&self, index: usize) -> glam::DVec4);
    forward_impl!(glam::DMat4 => fn from_scale(vector: glam::DVec3) -> Self);
    forward_impl!(glam::DMat4 => fn from_axis_angle(axis: glam::DVec3, angle: f64) -> Self);
    forward_impl!(glam::DMat4 => fn from_translation(translation: glam::DVec3) -> Self);

    forward_impl!(glam::DMat4 => fn from_scale_rotation_translation(
        scale: glam::DVec3,
        rotation: glam::DQuat,
        translation: glam::DVec3
    ) -> Self);
}

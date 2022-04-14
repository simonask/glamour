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
    type Scalar: PrimitiveMatrices<Vec2 = Self::Vec2, Vec3 = Self::Vec3, Vec4 = Self::Vec4>
        + Float
        + AbsDiffEq;
    /// Shorthand for `glam::Vec2` or `glam::DVec2".
    type Vec2: Vector<2, Scalar = Self::Scalar>;
    /// Shorthand for `glam::Vec3` or `glam::DVec3".
    type Vec3: Vector<3, Scalar = Self::Scalar>;
    /// Shorthand for `glam::Vec4` or `glam::DVec4".
    type Vec4: Vector<4, Scalar = Self::Scalar>;

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
pub trait Matrix2: Matrix + Mul<Self::Vec2, Output = Self::Vec2> {
    /// Transform vector or point.
    ///
    /// See [`glam::Mat2::mul_vec2()`] or
    /// [`glam::DMat2::mul_vec2()`].
    fn mul_vec2(&self, vec: Self::Vec2) -> Self::Vec2;

    /// Create from column vectors.
    fn from_cols(x_axis: Self::Vec2, y_axis: Self::Vec2) -> Self;

    /// Get column at `index`.
    fn col(&self, index: usize) -> Self::Vec2;
    /// Get row at `index`.
    fn row(&self, index: usize) -> Self::Vec2;

    /// 2D scaling matrix.
    fn from_scale_angle(vector: Self::Vec2, angle: Self::Scalar) -> Self;

    /// 2D rotation matrix.
    fn from_angle(angle: Self::Scalar) -> Self;
}

/// Primitive 3x3 matrix.
///
/// Implemented for [`glam::Mat3`] and [`glam::DMat3`].
pub trait Matrix3: Matrix + Mul<Self::Vec3, Output = Self::Vec3> {
    /// Create from column vectors.
    fn from_cols(x_axis: Self::Vec3, y_axis: Self::Vec3, z_axis: Self::Vec3) -> Self;

    /// Get column at `index`.
    fn col(&self, index: usize) -> Self::Vec3;
    /// Get row at `index`.
    fn row(&self, index: usize) -> Self::Vec3;

    /// Transform vector.
    ///
    /// See [`glam::Mat3::transform_vector2()`] or
    /// [`glam::DMat3::transform_vector2()`].
    fn transform_vector2(&self, vector: Self::Vec2) -> Self::Vec2;

    /// Transform point.
    ///
    /// See [`glam::Mat3::transform_point2()`] or
    /// [`glam::DMat3::transform_point2()`].
    fn transform_point2(&self, point: Self::Vec2) -> Self::Vec2;

    /// Scaling matrix
    fn from_scale(vector: Self::Vec2) -> Self;

    /// Rotation matrix
    fn from_angle(angle: Self::Scalar) -> Self;

    /// 2D translation matrix.
    fn from_translation(translation: Self::Vec2) -> Self;

    /// 2D transform.
    fn from_scale_angle_translation(
        scale: Self::Vec2,
        angle: Self::Scalar,
        translation: Self::Vec2,
    ) -> Self;
}

/// Primitive 4x4 matrix.
///
/// Implemented for [`glam::Mat4`] and [`glam::DMat4`].
#[allow(missing_docs)]
pub trait Matrix4: Matrix + Mul<Self::Vec4, Output = Self::Vec4> {
    /// Transform point.
    ///
    /// See [`glam::Mat4::transform_point3()`] or
    /// [`glam::DMat4::transform_point3()`].
    fn transform_point3(&self, point: Self::Vec3) -> Self::Vec3;

    /// Transform vector.
    ///
    /// See [`glam::Mat4::transform_vector3()`] or
    /// [`glam::DMat4::transform_vector3()`].
    fn transform_vector3(&self, vector: Self::Vec3) -> Self::Vec3;

    /// Project point.
    ///
    /// See [`glam::Mat4::project_point3()`] or
    /// [`glam::DMat4::project_point3()`].
    fn project_point3(&self, vector: Self::Vec3) -> Self::Vec3;

    /// Create from column vectors.
    fn from_cols(
        x_axis: Self::Vec4,
        y_axis: Self::Vec4,
        z_axis: Self::Vec4,
        w_axis: Self::Vec4,
    ) -> Self;

    /// Get column at `index`.
    fn col(&self, index: usize) -> Self::Vec4;

    /// Get row at `index`.
    fn row(&self, index: usize) -> Self::Vec4;

    /// Scaling matrix
    fn from_scale(vector: Self::Vec3) -> Self;

    /// Rotation matrix
    fn from_axis_angle(axis: Self::Vec3, angle: Self::Scalar) -> Self;

    /// 3D translation matrix.
    fn from_translation(translation: Self::Vec3) -> Self;

    /// Scale, rotation, translation.
    fn from_scale_rotation_translation(
        scale: Self::Vec3,
        axis: <Self::Scalar as PrimitiveMatrices>::Quat,
        translation: Self::Vec3,
    ) -> Self;

    fn look_at_lh(eye: Self::Vec3, center: Self::Vec3, up: Self::Vec3) -> Self;
    fn look_at_rh(eye: Self::Vec3, center: Self::Vec3, up: Self::Vec3) -> Self;
    fn perspective_rh_gl(
        fov_y_radians: Self::Scalar,
        aspect_ratio: Self::Scalar,
        z_near: Self::Scalar,
        z_far: Self::Scalar,
    ) -> Self;
    fn perspective_lh(
        fov_y_radians: Self::Scalar,
        aspect_ratio: Self::Scalar,
        z_near: Self::Scalar,
        z_far: Self::Scalar,
    ) -> Self;
    fn perspective_rh(
        fov_y_radians: Self::Scalar,
        aspect_ratio: Self::Scalar,
        z_near: Self::Scalar,
        z_far: Self::Scalar,
    ) -> Self;
    fn perspective_infinite_lh(
        fov_y_radians: Self::Scalar,
        aspect_ratio: Self::Scalar,
        z_near: Self::Scalar,
    ) -> Self;
    fn perspective_infinite_reverse_lh(
        fov_y_radians: Self::Scalar,
        aspect_ratio: Self::Scalar,
        z_near: Self::Scalar,
    ) -> Self;
    fn perspective_infinite_rh(
        fov_y_radians: Self::Scalar,
        aspect_ratio: Self::Scalar,
        z_near: Self::Scalar,
    ) -> Self;
    fn perspective_infinite_reverse_rh(
        fov_y_radians: Self::Scalar,
        aspect_ratio: Self::Scalar,
        z_near: Self::Scalar,
    ) -> Self;
    fn orthographic_rh_gl(
        left: Self::Scalar,
        right: Self::Scalar,
        bottom: Self::Scalar,
        top: Self::Scalar,
        near: Self::Scalar,
        far: Self::Scalar,
    ) -> Self;
    fn orthographic_lh(
        left: Self::Scalar,
        right: Self::Scalar,
        bottom: Self::Scalar,
        top: Self::Scalar,
        near: Self::Scalar,
        far: Self::Scalar,
    ) -> Self;
    fn orthographic_rh(
        left: Self::Scalar,
        right: Self::Scalar,
        bottom: Self::Scalar,
        top: Self::Scalar,
        near: Self::Scalar,
        far: Self::Scalar,
    ) -> Self;
}

macro_rules! impl_matrix {
    ($scalar:ty, $glam_ty:ty) => {
        impl Matrix for $glam_ty {
            type Scalar = $scalar;
            type Vec2 = <$scalar as Primitive>::Vec2;
            type Vec3 = <$scalar as Primitive>::Vec3;
            type Vec4 = <$scalar as Primitive>::Vec4;

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

    forward_impl!(glam::Mat4 => fn look_at_lh(eye: glam::Vec3, center: glam::Vec3, up: glam::Vec3) -> Self);
    forward_impl!(glam::Mat4 => fn look_at_rh(eye: glam::Vec3, center: glam::Vec3, up: glam::Vec3) -> Self);
    forward_impl!(glam::Mat4 => fn perspective_rh_gl(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self);
    forward_impl!(glam::Mat4 => fn perspective_lh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self);
    forward_impl!(glam::Mat4 => fn perspective_rh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32, z_far: f32) -> Self);
    forward_impl!(glam::Mat4 => fn perspective_infinite_lh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32) -> Self);
    forward_impl!(glam::Mat4 => fn perspective_infinite_reverse_lh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32) -> Self);
    forward_impl!(glam::Mat4 => fn perspective_infinite_rh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32) -> Self);
    forward_impl!(glam::Mat4 => fn perspective_infinite_reverse_rh(fov_y_radians: f32, aspect_ratio: f32, z_near: f32) -> Self);
    forward_impl!(glam::Mat4 => fn orthographic_rh_gl(
        left: f32,
        right: f32,
        bottom: f32,
        top: f32,
        near: f32,
        far: f32
    ) -> Self);
    forward_impl!(glam::Mat4 => fn orthographic_lh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self);
    forward_impl!(glam::Mat4 => fn orthographic_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Self);
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

    forward_impl!(glam::DMat4 => fn look_at_lh(eye: glam::DVec3, center: glam::DVec3, up: glam::DVec3) -> Self);
    forward_impl!(glam::DMat4 => fn look_at_rh(eye: glam::DVec3, center: glam::DVec3, up: glam::DVec3) -> Self);
    forward_impl!(glam::DMat4 => fn perspective_rh_gl(fov_y_radians: f64, aspect_ratio: f64, z_near: f64, z_far: f64) -> Self);
    forward_impl!(glam::DMat4 => fn perspective_lh(fov_y_radians: f64, aspect_ratio: f64, z_near: f64, z_far: f64) -> Self);
    forward_impl!(glam::DMat4 => fn perspective_rh(fov_y_radians: f64, aspect_ratio: f64, z_near: f64, z_far: f64) -> Self);
    forward_impl!(glam::DMat4 => fn perspective_infinite_lh(fov_y_radians: f64, aspect_ratio: f64, z_near: f64) -> Self);
    forward_impl!(glam::DMat4 => fn perspective_infinite_reverse_lh(fov_y_radians: f64, aspect_ratio: f64, z_near: f64) -> Self);
    forward_impl!(glam::DMat4 => fn perspective_infinite_rh(fov_y_radians: f64, aspect_ratio: f64, z_near: f64) -> Self);
    forward_impl!(glam::DMat4 => fn perspective_infinite_reverse_rh(fov_y_radians: f64, aspect_ratio: f64, z_near: f64) -> Self);
    forward_impl!(glam::DMat4 => fn orthographic_rh_gl(
        left: f64,
        right: f64,
        bottom: f64,
        top: f64,
        near: f64,
        far: f64
    ) -> Self);
    forward_impl!(glam::DMat4 => fn orthographic_lh(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Self);
    forward_impl!(glam::DMat4 => fn orthographic_rh(left: f64, right: f64, bottom: f64, top: f64, near: f64, far: f64) -> Self);
}

#![allow(missing_docs, clippy::return_self_not_must_use)]

use crate::scalar::FloatScalar;

use super::*;

/// Trait describing a glam N x N matrix type.
///
/// Note: All glam matrices are square.
#[allow(missing_docs)]
pub trait Matrix<T: FloatScalar>:
    PodValue
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Mul<T, Output = Self>
    + MulAssign<T>
    + Div<T, Output = Self>
    + DivAssign<T>
    + Neg<Output = Self>
    + AbsDiffEq<Epsilon = T::Epsilon>
{
    #[must_use]
    fn is_nan(&self) -> bool;
    #[must_use]
    fn is_finite(&self) -> bool;
    #[must_use]
    fn transpose(&self) -> Self;
    #[must_use]
    fn inverse(&self) -> Self;
    #[must_use]
    fn determinant(&self) -> T;
    #[must_use]
    fn abs(&self) -> Self;
}

/// Primitive 2x2 matrix.
///
/// Implemented for [`glam::Mat2`] and [`glam::DMat2`].
#[allow(missing_docs)]
pub trait Matrix2<T: FloatScalar<Mat2 = Self>>: Matrix<T> {
    fn mul_vec2(&self, vec: T::Vec2) -> T::Vec2;
    fn from_cols(x_axis: T::Vec2, y_axis: T::Vec2) -> Self;
    fn col(&self, index: usize) -> T::Vec2;
    fn row(&self, index: usize) -> T::Vec2;
    fn from_scale_angle(vector: T::Vec2, angle: T) -> Self;
    fn from_angle(angle: T) -> Self;
    fn from_mat3(mat3: T::Mat3) -> Self;
    fn from_diagonal(diagonal: T::Vec2) -> Self;
    fn mul_mat2(&self, other: &Self) -> Self;
    fn add_mat2(&self, other: &Self) -> Self;
    fn sub_mat2(&self, other: &Self) -> Self;
    fn to_cols_array(&self) -> [T; 4];
    fn from_cols_array(array: &[T; 4]) -> Self;
}

/// Primitive 3x3 matrix.
///
/// Implemented for [`glam::Mat3`] and [`glam::DMat3`].
#[allow(missing_docs)]
pub trait Matrix3<T: FloatScalar<Mat3 = Self>>: Matrix<T> {
    fn mul_vec3(&self, vec: T::Vec3) -> T::Vec3;
    fn from_cols(x_axis: T::Vec3, y_axis: T::Vec3, z_axis: T::Vec3) -> Self;
    fn col(&self, index: usize) -> T::Vec3;
    fn row(&self, index: usize) -> T::Vec3;
    fn transform_vector2(&self, vector: T::Vec2) -> T::Vec2;
    fn transform_point2(&self, point: T::Vec2) -> T::Vec2;
    fn from_scale(vector: T::Vec2) -> Self;
    fn from_angle(angle: T) -> Self;
    fn from_translation(translation: T::Vec2) -> Self;
    fn from_scale_angle_translation(scale: T::Vec2, angle: T, translation: T::Vec2) -> Self;
    fn from_diagonal(diagonal: T::Vec3) -> Self;
    fn from_mat2(mat2: T::Mat2) -> Self;
    fn from_mat4(mat4: T::Mat4) -> Self;
    fn mul_mat3(&self, other: &Self) -> Self;
    fn add_mat3(&self, other: &Self) -> Self;
    fn sub_mat3(&self, other: &Self) -> Self;
    fn to_cols_array(&self) -> [T; 9];
    fn from_cols_array(array: &[T; 9]) -> Self;
}

/// Primitive 4x4 matrix.
///
/// Implemented for [`glam::Mat4`] and [`glam::DMat4`].
#[allow(missing_docs)]
pub trait Matrix4<T: FloatScalar<Mat4 = Self>>: Matrix<T> {
    fn mul_vec4(&self, vec: T::Vec4) -> T::Vec4;
    fn transform_point3(&self, point: T::Vec3) -> T::Vec3;
    fn transform_vector3(&self, vector: T::Vec3) -> T::Vec3;
    fn project_point3(&self, vector: T::Vec3) -> T::Vec3;

    fn from_cols(x_axis: T::Vec4, y_axis: T::Vec4, z_axis: T::Vec4, w_axis: T::Vec4) -> Self;
    fn col(&self, index: usize) -> T::Vec4;
    fn row(&self, index: usize) -> T::Vec4;
    fn from_scale(vector: T::Vec3) -> Self;
    fn from_axis_angle(axis: T::Vec3, angle: T) -> Self;
    fn from_translation(translation: T::Vec3) -> Self;
    fn from_scale_rotation_translation(scale: T::Vec3, axis: T::Quat, translation: T::Vec3)
        -> Self;
    fn look_at_lh(eye: T::Vec3, center: T::Vec3, up: T::Vec3) -> Self;
    fn look_at_rh(eye: T::Vec3, center: T::Vec3, up: T::Vec3) -> Self;
    fn perspective_rh_gl(fov_y_radians: T, aspect_ratio: T, z_near: T, z_far: T) -> Self;
    fn perspective_lh(fov_y_radians: T, aspect_ratio: T, z_near: T, z_far: T) -> Self;
    fn perspective_rh(fov_y_radians: T, aspect_ratio: T, z_near: T, z_far: T) -> Self;
    fn perspective_infinite_lh(fov_y_radians: T, aspect_ratio: T, z_near: T) -> Self;
    fn perspective_infinite_reverse_lh(fov_y_radians: T, aspect_ratio: T, z_near: T) -> Self;
    fn perspective_infinite_rh(fov_y_radians: T, aspect_ratio: T, z_near: T) -> Self;
    fn perspective_infinite_reverse_rh(fov_y_radians: T, aspect_ratio: T, z_near: T) -> Self;
    fn orthographic_rh_gl(left: T, right: T, bottom: T, top: T, near: T, far: T) -> Self;
    fn orthographic_lh(left: T, right: T, bottom: T, top: T, near: T, far: T) -> Self;
    fn orthographic_rh(left: T, right: T, bottom: T, top: T, near: T, far: T) -> Self;

    fn from_diagonal(diagonal: T::Vec4) -> Self;
    fn from_rotation_translation(rotation: T::Quat, translation: T::Vec3) -> Self;
    fn from_quat(quat: T::Quat) -> Self;
    fn from_mat3(mat3: T::Mat3) -> Self;
    fn mul_mat4(&self, other: &Self) -> Self;
    fn add_mat4(&self, other: &Self) -> Self;
    fn sub_mat4(&self, other: &Self) -> Self;
    fn to_cols_array(&self) -> [T; 16];
    fn from_cols_array(array: &[T; 16]) -> Self;
}

macro_rules! impl_matrix {
    ($scalar:ty, $glam_ty:ty) => {
        impl Matrix<$scalar> for $glam_ty {
            forward_impl!($glam_ty => fn is_nan(&self) -> bool);
            forward_impl!($glam_ty => fn is_finite(&self) -> bool);
            forward_impl!($glam_ty => fn determinant(&self) -> $scalar);
            forward_impl!($glam_ty => fn transpose(&self) -> Self);
            forward_impl!($glam_ty => fn inverse(&self) -> Self);
            forward_impl!($glam_ty => fn abs(&self) -> Self);
        }
    };
}

impl_matrix!(f32, glam::Mat2);
impl_matrix!(f32, glam::Mat3);
impl_matrix!(f32, glam::Mat4);
impl_matrix!(f64, glam::DMat2);
impl_matrix!(f64, glam::DMat3);
impl_matrix!(f64, glam::DMat4);

impl Matrix2<f32> for glam::Mat2 {
    forward_impl!(glam::Mat2 => fn from_cols(x: glam::Vec2, y: glam::Vec2) -> Self);
    forward_impl!(glam::Mat2 => fn mul_vec2(&self, vec: glam::Vec2) -> glam::Vec2);
    forward_impl!(glam::Mat2 => fn col(&self, index: usize) -> glam::Vec2);
    forward_impl!(glam::Mat2 => fn row(&self, index: usize) -> glam::Vec2);
    forward_impl!(glam::Mat2 => fn from_mat3(mat3: glam::Mat3) -> Self);
    forward_impl!(glam::Mat2 => fn from_scale_angle(vector: glam::Vec2, angle: f32) -> Self);
    forward_impl!(glam::Mat2 => fn from_angle(angle: f32) -> Self);
    forward_impl!(glam::Mat2 => fn from_diagonal(diagonal: glam::Vec2) -> Self);
    forward_impl!(glam::Mat2 => fn mul_mat2(&self, other: &Self) -> Self);
    forward_impl!(glam::Mat2 => fn add_mat2(&self, other: &Self) -> Self);
    forward_impl!(glam::Mat2 => fn sub_mat2(&self, other: &Self) -> Self);
    forward_impl!(glam::Mat2 => fn to_cols_array(&self) -> [f32; 4]);
    forward_impl!(glam::Mat2 => fn from_cols_array(array: &[f32; 4]) -> Self);
}

impl Matrix2<f64> for glam::DMat2 {
    forward_impl!(glam::DMat2 => fn from_cols(x: glam::DVec2, y: glam::DVec2) -> Self);
    forward_impl!(glam::DMat2 => fn mul_vec2(&self, vec: glam::DVec2) -> glam::DVec2);
    forward_impl!(glam::DMat2 => fn col(&self, index: usize) -> glam::DVec2);
    forward_impl!(glam::DMat2 => fn row(&self, index: usize) -> glam::DVec2);
    forward_impl!(glam::DMat2 => fn from_mat3(mat3: glam::DMat3) -> Self);
    forward_impl!(glam::DMat2 => fn from_scale_angle(vector: glam::DVec2, angle: f64) -> Self);
    forward_impl!(glam::DMat2 => fn from_angle(angle: f64) -> Self);
    forward_impl!(glam::DMat2 => fn from_diagonal(diagonal: glam::DVec2) -> Self);
    forward_impl!(glam::DMat2 => fn mul_mat2(&self, other: &Self) -> Self);
    forward_impl!(glam::DMat2 => fn add_mat2(&self, other: &Self) -> Self);
    forward_impl!(glam::DMat2 => fn sub_mat2(&self, other: &Self) -> Self);
    forward_impl!(glam::DMat2 => fn to_cols_array(&self) -> [f64; 4]);
    forward_impl!(glam::DMat2 => fn from_cols_array(array: &[f64; 4]) -> Self);
}

impl Matrix3<f32> for glam::Mat3 {
    forward_impl!(glam::Mat3 => fn transform_point2(&self, point: glam::Vec2) -> glam::Vec2);
    forward_impl!(glam::Mat3 => fn transform_vector2(&self, point: glam::Vec2) -> glam::Vec2);
    forward_impl!(glam::Mat3 => fn from_cols(x: glam::Vec3, y: glam::Vec3, z: glam::Vec3) -> Self);
    forward_impl!(glam::Mat3 => fn mul_vec3(&self, vec: glam::Vec3) -> glam::Vec3);
    forward_impl!(glam::Mat3 => fn col(&self, index: usize) -> glam::Vec3);
    forward_impl!(glam::Mat3 => fn row(&self, index: usize) -> glam::Vec3);
    forward_impl!(glam::Mat3 => fn from_scale(vector: glam::Vec2) -> Self);
    forward_impl!(glam::Mat3 => fn from_angle(angle: f32) -> Self);
    forward_impl!(glam::Mat3 => fn from_translation(translation: glam::Vec2) -> Self);
    forward_impl!(glam::Mat3 => fn from_diagonal(diagonal: glam::Vec3) -> Self);
    forward_impl!(glam::Mat3 => fn from_mat2(mat2: glam::Mat2) -> Self);
    forward_impl!(glam::Mat3 => fn from_mat4(mat4: glam::Mat4) -> Self);
    forward_impl!(glam::Mat3 => fn mul_mat3(&self, other: &Self) -> Self);
    forward_impl!(glam::Mat3 => fn add_mat3(&self, other: &Self) -> Self);
    forward_impl!(glam::Mat3 => fn sub_mat3(&self, other: &Self) -> Self);
    forward_impl!(glam::Mat3 => fn to_cols_array(&self) -> [f32; 9]);
    forward_impl!(glam::Mat3 => fn from_cols_array(array: &[f32; 9]) -> Self);

    forward_impl!(glam::Mat3 => fn from_scale_angle_translation(
        scale: glam::Vec2,
        angle: f32,
        translation: glam::Vec2
    ) -> Self);
}

impl Matrix3<f64> for glam::DMat3 {
    forward_impl!(glam::DMat3 => fn from_cols(x: glam::DVec3, y: glam::DVec3, z: glam::DVec3) -> Self);
    forward_impl!(glam::DMat3 => fn mul_vec3(&self, vec: glam::DVec3) -> glam::DVec3);
    forward_impl!(glam::DMat3 => fn transform_point2(&self, point: glam::DVec2) -> glam::DVec2);
    forward_impl!(glam::DMat3 => fn transform_vector2(&self, point: glam::DVec2) -> glam::DVec2);
    forward_impl!(glam::DMat3 => fn col(&self, index: usize) -> glam::DVec3);
    forward_impl!(glam::DMat3 => fn row(&self, index: usize) -> glam::DVec3);
    forward_impl!(glam::DMat3 => fn from_scale(vector: glam::DVec2) -> Self);
    forward_impl!(glam::DMat3 => fn from_angle(angle: f64) -> Self);
    forward_impl!(glam::DMat3 => fn from_translation(translation: glam::DVec2) -> Self);
    forward_impl!(glam::DMat3 => fn from_diagonal(diagonal: glam::DVec3) -> Self);
    forward_impl!(glam::DMat3 => fn from_mat2(mat2: glam::DMat2) -> Self);
    forward_impl!(glam::DMat3 => fn from_mat4(mat4: glam::DMat4) -> Self);
    forward_impl!(glam::DMat3 => fn mul_mat3(&self, other: &Self) -> Self);
    forward_impl!(glam::DMat3 => fn add_mat3(&self, other: &Self) -> Self);
    forward_impl!(glam::DMat3 => fn sub_mat3(&self, other: &Self) -> Self);
    forward_impl!(glam::DMat3 => fn to_cols_array(&self) -> [f64; 9]);
    forward_impl!(glam::DMat3 => fn from_cols_array(array: &[f64; 9]) -> Self);

    forward_impl!(glam::DMat3 => fn from_scale_angle_translation(
        scale: glam::DVec2,
        angle: f64,
        translation: glam::DVec2
    ) -> Self);
}

impl Matrix4<f32> for glam::Mat4 {
    forward_impl!(glam::Mat4 => fn transform_point3(&self, point: glam::Vec3) -> glam::Vec3);
    forward_impl!(glam::Mat4 => fn transform_vector3(&self, point: glam::Vec3) -> glam::Vec3);
    forward_impl!(glam::Mat4 => fn project_point3(&self, point: glam::Vec3) -> glam::Vec3);
    forward_impl!(glam::Mat4 => fn from_cols(x: glam::Vec4, y: glam::Vec4, z: glam::Vec4, w: glam::Vec4) -> Self);
    forward_impl!(glam::Mat4 => fn mul_vec4(&self, vec: glam::Vec4) -> glam::Vec4);
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

    forward_impl!(glam::Mat4 => fn from_diagonal(diagonal: glam::Vec4) -> Self);
    forward_impl!(glam::Mat4 => fn from_rotation_translation(
        rotation: glam::Quat,
        translation: glam::Vec3
    ) -> Self);
    forward_impl!(glam::Mat4 => fn from_quat(quat: glam::Quat) -> Self);
    forward_impl!(glam::Mat4 => fn from_mat3(mat3: glam::Mat3) -> Self);
    forward_impl!(glam::Mat4 => fn mul_mat4(&self, other: &Self) -> Self);
    forward_impl!(glam::Mat4 => fn add_mat4(&self, other: &Self) -> Self);
    forward_impl!(glam::Mat4 => fn sub_mat4(&self, other: &Self) -> Self);
    forward_impl!(glam::Mat4 => fn to_cols_array(&self) -> [f32; 16]);
    forward_impl!(glam::Mat4 => fn from_cols_array(array: &[f32; 16]) -> Self);
}

impl Matrix4<f64> for glam::DMat4 {
    forward_impl!(glam::DMat4 => fn transform_point3(&self, point: glam::DVec3) -> glam::DVec3);
    forward_impl!(glam::DMat4 => fn transform_vector3(&self, point: glam::DVec3) -> glam::DVec3);
    forward_impl!(glam::DMat4 => fn project_point3(&self, point: glam::DVec3) -> glam::DVec3);
    forward_impl!(glam::DMat4 => fn from_cols(x: glam::DVec4, y: glam::DVec4, z: glam::DVec4, w: glam::DVec4) -> Self);
    forward_impl!(glam::DMat4 => fn mul_vec4(&self, vec: glam::DVec4) -> glam::DVec4);
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

    forward_impl!(glam::DMat4 => fn from_diagonal(diagonal: glam::DVec4) -> Self);
    forward_impl!(glam::DMat4 => fn from_rotation_translation(
        rotation: glam::DQuat,
        translation: glam::DVec3
    ) -> Self);
    forward_impl!(glam::DMat4 => fn from_quat(quat: glam::DQuat) -> Self);
    forward_impl!(glam::DMat4 => fn from_mat3(mat3: glam::DMat3) -> Self);
    forward_impl!(glam::DMat4 => fn mul_mat4(&self, other: &Self) -> Self);
    forward_impl!(glam::DMat4 => fn add_mat4(&self, other: &Self) -> Self);
    forward_impl!(glam::DMat4 => fn sub_mat4(&self, other: &Self) -> Self);
    forward_impl!(glam::DMat4 => fn to_cols_array(&self) -> [f64; 16]);
    forward_impl!(glam::DMat4 => fn from_cols_array(array: &[f64; 16]) -> Self);
}

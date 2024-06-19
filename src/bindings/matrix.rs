#![allow(missing_docs, clippy::return_self_not_must_use)]

use crate::scalar::FloatScalar;

use super::*;

/// Trait describing a glam N x N matrix type.
///
/// Note: All glam matrices are square.
#[allow(missing_docs)]
pub trait Matrix:
    PodValue
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Mul<Self::Scalar, Output = Self>
    + MulAssign<Self::Scalar>
    + Div<Self::Scalar, Output = Self>
    + DivAssign<Self::Scalar>
    + Neg<Output = Self>
    + AbsDiffEq<Epsilon = Self::Scalar>
    + approx::RelativeEq<Epsilon = Self::Scalar>
    + approx::UlpsEq<Epsilon = Self::Scalar>
{
    type Scalar: FloatScalar;
    crate::interfaces::matrix_base_interface!(trait_decl);
}

/// Primitive 2x2 matrix.
///
/// Implemented for [`glam::Mat2`] and [`glam::DMat2`].
#[allow(missing_docs)]
pub trait Matrix2: Matrix<Scalar: FloatScalar<Mat2 = Self>> {
    crate::interfaces::matrix2_base_interface!(trait_decl);
}

/// Primitive 3x3 matrix.
///
/// Implemented for [`glam::Mat3`] and [`glam::DMat3`].
#[allow(missing_docs)]
pub trait Matrix3: Matrix<Scalar: FloatScalar<Mat3 = Self>> {
    crate::interfaces::matrix3_base_interface!(trait_decl);
}

/// Primitive 4x4 matrix.
///
/// Implemented for [`glam::Mat4`] and [`glam::DMat4`].
#[allow(missing_docs)]
pub trait Matrix4: Matrix<Scalar: FloatScalar<Mat4 = Self>> {
    crate::interfaces::matrix4_base_interface!(trait_decl);
}

macro_rules! impl_matrix {
    ($scalar:ty, $glam_ty:ty) => {
        impl Matrix for $glam_ty {
            type Scalar = $scalar;
            crate::interfaces::matrix_base_interface!(trait_impl);
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
    crate::interfaces::matrix2_base_interface!(trait_impl);
}

impl Matrix2 for glam::DMat2 {
    crate::interfaces::matrix2_base_interface!(trait_impl);
}

impl Matrix3 for glam::Mat3 {
    crate::interfaces::matrix3_base_interface!(trait_impl);
}

impl Matrix3 for glam::DMat3 {
    crate::interfaces::matrix3_base_interface!(trait_impl);
}

impl Matrix4 for glam::Mat4 {
    crate::interfaces::matrix4_base_interface!(trait_impl);
}

impl Matrix4 for glam::DMat4 {
    crate::interfaces::matrix4_base_interface!(trait_impl);
}

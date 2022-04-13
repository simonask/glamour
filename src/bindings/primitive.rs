use approx::{RelativeEq, UlpsEq};

use crate::AngleConsts;

use super::*;

/// Mapping from primitive scalar type to `glam` vector types.
///
/// Depending on the base type of the scalar, vectors of that scalar are mapped
/// to `glam` vectors in the following way:
///
/// | Primitive  | 2D                     | 3D                     | 4D                     |
/// | ---------- | ---------------------- | ---------------------- | ---------------------- |
/// | `f32`      | [`Vec2`](glam::Vec2)   | [`Vec3`](glam::Vec3)   | [`Vec4`](glam::Vec4)   |
/// | `f64`      | [`DVec2`](glam::DVec2) | [`DVec3`](glam::DVec3) | [`DVec4`](glam::DVec4) |
/// | `i32`      | [`IVec2`](glam::IVec2) | [`IVec3`](glam::IVec3) | [`IVec4`](glam::IVec4) |
/// | `u32`      | [`UVec2`](glam::UVec2) | [`UVec3`](glam::UVec3) | [`UVec4`](glam::UVec4) |
///
/// See also [the documentation module](crate::docs#how).
pub trait Primitive:
    crate::Scalar<Primitive = Self>
    + crate::Unit<Scalar = Self>
    + num_traits::NumCast
    + num_traits::Num
    + Debug
    + Display
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + Rem<Self, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + RemAssign<Self>
    + AbsDiffEq<Epsilon = Self>
    + Sized
    + Send
    + Sync
    + 'static
{
    /// 2D vector type
    type Vec2: Vector<2, Scalar = Self, Mask = glam::BVec2>;
    /// 3D vector type
    type Vec3: Vector<3, Scalar = Self, Mask = glam::BVec3A>;
    /// 4D vector type
    type Vec4: Vector<4, Scalar = Self, Mask = glam::BVec4A>;

    /// True if the value is finite (not infinity, not NaN).
    ///
    /// This is always true for integers.
    #[must_use]
    fn is_finite(self) -> bool;
}

impl Primitive for f32 {
    type Vec2 = glam::Vec2;
    type Vec3 = glam::Vec3;
    type Vec4 = glam::Vec4;

    fn is_finite(self) -> bool {
        <f32>::is_finite(self)
    }
}

impl Primitive for f64 {
    type Vec2 = glam::DVec2;
    type Vec3 = glam::DVec3;
    type Vec4 = glam::DVec4;

    fn is_finite(self) -> bool {
        <f64>::is_finite(self)
    }
}

impl Primitive for i32 {
    type Vec2 = glam::IVec2;
    type Vec3 = glam::IVec3;
    type Vec4 = glam::IVec4;

    fn is_finite(self) -> bool {
        true
    }
}

impl Primitive for u32 {
    type Vec2 = glam::UVec2;
    type Vec3 = glam::UVec3;
    type Vec4 = glam::UVec4;

    fn is_finite(self) -> bool {
        true
    }
}

/// Mapping from primitive scalar type to `glam` matrix types.
///
/// Depending on the base type of the scalar, matrices of that scalar are mapped
/// to `glam` matrices in the following way:
///
/// | Primitive  | 2D                     | 3D                     | 4D                     |
/// | ---------- | ---------------------- | ---------------------- | ---------------------- |
/// | `f32`      | [`Mat2`](glam::Mat2)   | [`Mat3`](glam::Mat3)   | [`Mat4`](glam::Mat4)   |
/// | `f64`      | [`DMat2`](glam::DMat2) | [`DMat3`](glam::DMat3) | [`DMat4`](glam::DMat4) |
///
/// Note that `glam` does not support integer matrices.
///
/// See also [the documentation module](crate::docs#how).
pub trait PrimitiveMatrices:
    Primitive + Float + AngleConsts + AbsDiffEq + RelativeEq + UlpsEq
{
    /// [`glam::Mat2`] or [`glam::DMat2`].
    type Mat2: Matrix2<Scalar = Self>;
    /// [`glam::Mat3`] or [`glam::DMat3`].
    type Mat3: Matrix3<Scalar = Self>;
    /// [`glam::Mat4`] or [`glam::DMat4`].
    type Mat4: Matrix4<Scalar = Self>;
    /// [`glam::Quat`] or [`glam::DQuat`].
    type Quat: Quat<Scalar = Self, Vec3 = Self::Vec3>;
}

impl PrimitiveMatrices for f32 {
    type Mat2 = glam::Mat2;
    type Mat3 = glam::Mat3;
    type Mat4 = glam::Mat4;
    type Quat = glam::Quat;
}

impl PrimitiveMatrices for f64 {
    type Mat2 = glam::DMat2;
    type Mat3 = glam::DMat3;
    type Mat4 = glam::DMat4;
    type Quat = glam::DQuat;
}

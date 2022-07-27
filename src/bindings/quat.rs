use crate::scalar::FloatScalar;

use super::*;

/// Common trait for [`glam::Quat`] and [`glam::DQuat`].
#[allow(missing_docs)]
pub trait Quat:
    PodValue
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Div<Self::Scalar, Output = Self>
    + Neg<Output = Self>
    + AbsDiffEq
{
    type Scalar: FloatScalar<Quat = Self, Vec3 = Self::Vec3, Vec4 = Self::Vec4> + AbsDiffEq;
    type Vec3: FloatVector3<Scalar = Self::Scalar>;
    type Vec4: FloatVector4<Scalar = Self::Scalar>;

    fn from_axis_angle(axis: Self::Vec3, angle: Self::Scalar) -> Self;
}

macro_rules! impl_quat {
    ($base:ty, $scalar:ty, $vec3:ty, $vec4:ty) => {
        impl Quat for $base {
            type Scalar = $scalar;
            type Vec3 = $vec3;
            type Vec4 = $vec4;

            forward_impl!($base => fn from_axis_angle(axis: $vec3, angle: $scalar) -> Self);
        }
    };
}

impl_quat!(glam::Quat, f32, glam::Vec3, glam::Vec4);
impl_quat!(glam::DQuat, f64, glam::DVec3, glam::DVec4);

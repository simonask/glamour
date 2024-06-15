use crate::scalar::{FloatScalar, Scalar};

use super::*;

/// Common trait for [`glam::Quat`] and [`glam::DQuat`].
#[allow(missing_docs)]
pub trait Quat<T: FloatScalar>:
    PodValue
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Div<T, Output = Self>
    + Neg<Output = Self>
    + AbsDiffEq
{
    fn from_axis_angle(axis: T::Vec3, angle: T) -> Self;
}

macro_rules! impl_quat {
    ($base:ty, $scalar:ty) => {
        impl Quat<$scalar> for $base {
            forward_impl!($base => fn from_axis_angle(axis: <$scalar as Scalar>::Vec3, angle: $scalar) -> Self);
        }
    };
}

impl_quat!(glam::Quat, f32);
impl_quat!(glam::DQuat, f64);

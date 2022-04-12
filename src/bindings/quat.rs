use super::*;

/// Common trait for [`glam::Quat`] and [`glam::DQuat`].
#[allow(missing_docs)]
pub trait Quat:
    ValueSemantics
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Div<Self::Scalar, Output = Self>
    + Neg<Output = Self>
    + AbsDiffEq
{
    type Scalar: PrimitiveMatrices<Vec3 = Self::Vec3, Quat = Self> + Float + AbsDiffEq;
    type Vec3: Vector<3, Scalar = Self::Scalar>;
    type Vec4: Vector<4, Scalar = Self::Scalar>;

    fn from_axis_angle(axis: Self::Vec3, angle: Self::Scalar) -> Self;
}

macro_rules! forward_impl {
    ($base:ty => fn $name:ident ( &self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty) => {
        fn $name(&self $(, $arg:$arg_ty)*) -> $ret {
            <$base>::$name(self $(, $arg)*)
        }
    };
    ($base:ty => fn $name:ident ( &mut self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty) => {
        fn $name(&mut self $(, $arg:$arg_ty)*) -> $ret {
            <$base>::$name(self $(, $arg)*)
        }
    };
    ($base:ty => fn $name:ident ( self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty) => {
        fn $name(self $(, $arg:$arg_ty)*) -> $ret {
            <$base>::$name(self $(, $arg)*)
        }
    };
    ($base:ty => fn $name:ident ( $($arg:ident : $arg_ty:ty),*) -> $ret:ty) => {
        fn $name($($arg:$arg_ty),*) -> $ret {
            <$base>::$name($($arg),*)
        }
    };
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

//! Generic angles.

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use bytemuck::{Pod, Zeroable};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use num_traits::{Float, NumAssignOps};

use crate::{
    bindings::{Primitive, PrimitiveMatrices, Quat},
    traits::marker::ValueSemantics,
    Scalar, Unit, Vector2, Vector3, Vector4,
};

/// Angle in radians.
///
/// Note that Angle implements both [Scalar] and [Unit], so it is compatible
/// with vector types: `Vector2<Angle<f32>>` is a vector containing two angles.
#[repr(transparent)]
#[derive(Clone, Copy, Default, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
pub struct Angle<T = f32> {
    /// Angle in radians.
    pub radians: T,
}

/// Strongly typed angle constants.
///
/// These allow common constants used as angles to be strongly typed (so it's
/// possible to write `Angle::<f32>::PI` instead of
/// `Angle::radians(core::f32::PI)`).
///
/// Note that this is also the most convenient way to get these constants in a
/// `const` context. Since [`Angle`][crate::Angle] is a generic type, the
/// [`Angle::from_radians()`](crate::Angle::from_radians) constructor cannot be
/// invoked in a const context. This is also why [`num_traits::FloatConst`] is
/// not good enough.
pub trait AngleConsts {
    /// π
    const PI: Self;

    /// τ = 2π
    const TAU: Self;

    /// 1.0 / π
    const FRAG_1_PI: Self;
    /// 2.0 / π
    const FRAG_2_PI: Self;

    /// π / 2.0
    const FRAG_PI_2: Self;
    /// π / 3.0
    const FRAG_PI_3: Self;
    /// π / 4.0
    const FRAG_PI_4: Self;
    /// π / 6.0
    const FRAG_PI_6: Self;
    /// π / 8.0
    const FRAG_PI_8: Self;
}

macro_rules! impl_float_angle_consts {
    ($float:ident) => {
        impl AngleConsts for $float {
            const PI: Self = core::$float::consts::PI;
            const TAU: Self = Self::PI + Self::PI;

            const FRAG_1_PI: Self = 1.0 / Self::PI;
            const FRAG_2_PI: Self = 2.0 / Self::PI;
            const FRAG_PI_2: Self = Self::PI / 2.0;
            const FRAG_PI_3: Self = Self::PI / 3.0;
            const FRAG_PI_4: Self = Self::PI / 4.0;
            const FRAG_PI_6: Self = Self::PI / 6.0;
            const FRAG_PI_8: Self = Self::PI / 8.0;
        }
    };
}

impl_float_angle_consts!(f32);
impl_float_angle_consts!(f64);

macro_rules! forward_to_float_as_angle {
    (
        $(#[$attr:meta])?
        fn $f:ident(self $(, $args:ident: $args_ty:ty)*) -> Angle<Self>) => {
        $(#[$attr])?
        #[must_use]
        #[inline]
        fn $f(self, $($args: $args_ty),*) -> Angle<Self> {
            Angle::from_radians(<Self as num_traits::Float>::$f(self $($args),*))
        }
    };
}

/// Convenience trait to create strongly typed angles from floats.
pub trait FloatAngleExt: num_traits::Float + AngleConsts + ValueSemantics {
    forward_to_float_as_angle!(
        #[doc = "asin()"]
        fn asin(self) -> Angle<Self>
    );
    forward_to_float_as_angle!(
        #[doc = "acos()"]
        fn acos(self) -> Angle<Self>);
    forward_to_float_as_angle!(
        #[doc = "atan()"]
        fn atan(self) -> Angle<Self>
    );
}

impl FloatAngleExt for f32 {}
impl FloatAngleExt for f64 {}

impl<T: Primitive + AngleConsts> core::fmt::Debug for Angle<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use core::fmt::Write;
        f.write_str("Angle(")?;
        let r = self.radians;
        if r == T::PI {
            f.write_char('π')?;
        } else if r == T::TAU {
            f.write_str("2π")?;
        } else if r == T::FRAG_1_PI {
            f.write_str("1/π")?;
        } else if r == T::FRAG_2_PI {
            f.write_str("2/π")?;
        } else if r == T::FRAG_PI_2 {
            f.write_str("π/2")?;
        } else if r == T::FRAG_PI_3 {
            f.write_str("π/3")?;
        } else if r == T::FRAG_PI_4 {
            f.write_str("π/4")?;
        } else if r == T::FRAG_PI_6 {
            f.write_str("π/6")?;
        } else if r == T::FRAG_PI_8 {
            f.write_str("π/8")?;
        } else {
            write!(f, "{:0.5}", r)?;
        }
        f.write_char(')')
    }
}

unsafe impl<T: Zeroable> Zeroable for Angle<T> {}
unsafe impl<T: Pod> Pod for Angle<T> {}

impl<T> Scalar for Angle<T>
where
    T: Primitive + AngleConsts + Float,
{
    type Primitive = T;
}

impl<T> Unit for Angle<T>
where
    T: Primitive + AngleConsts + Float,
{
    type Scalar = Angle<T>;

    fn name() -> Option<&'static str> {
        Some("Angle")
    }
}

impl<T> AbsDiffEq for Angle<T>
where
    T: AbsDiffEq,
{
    type Epsilon = T::Epsilon;

    #[inline]
    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    #[inline]
    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.radians.abs_diff_eq(&other.radians, epsilon)
    }

    #[inline]
    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.radians.abs_diff_ne(&other.radians, epsilon)
    }
}

impl<T> UlpsEq for Angle<T>
where
    T: UlpsEq,
{
    #[inline]
    fn default_max_ulps() -> u32 {
        T::default_max_ulps()
    }

    #[inline]
    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.radians.ulps_eq(&other.radians, epsilon, max_ulps)
    }

    #[inline]
    fn ulps_ne(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.radians.ulps_ne(&other.radians, epsilon, max_ulps)
    }
}

impl<T> RelativeEq for Angle<T>
where
    T: RelativeEq,
{
    #[inline]
    fn default_max_relative() -> Self::Epsilon {
        T::default_max_relative()
    }

    #[inline]
    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.radians
            .relative_eq(&other.radians, epsilon, max_relative)
    }

    #[inline]
    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.radians
            .relative_ne(&other.radians, epsilon, max_relative)
    }
}

impl<T: Primitive + Float> PartialEq<T> for Angle<T> {
    fn eq(&self, other: &T) -> bool {
        self.radians == *other
    }
}

macro_rules! forward_to_unit {
    (
        $(#[$attr:meta])?
        fn $f:ident(self $(, $args:ident: $args_ty:ty)*) -> $ret:ty) => {
        $(#[$attr])*
        #[must_use]
        #[inline]
        pub fn $f(self, $($args: $args_ty),*) -> $ret {
            self.radians.$f($($args),*)
        }
    };
}

macro_rules! forward_to_unit_as_self {
    (
        $(#[$attr:meta])?
        fn $f:ident(self $(, $args:ident: $args_ty:ty)*) -> Self) => {
        $(#[$attr])*
        #[must_use]
        #[inline]
        pub fn $f(self,  $($args: $args_ty),*) -> Angle<T> {
            Angle::from_radians(self.radians.$f($($args.radians),*))
        }
    };
}

impl<T: Float> Angle<T> {
    /// Angle from radians.
    #[inline]
    #[must_use]
    pub fn from_radians(radians: T) -> Self {
        Angle { radians }
    }

    /// Angle from degrees.
    ///
    /// Will be converted to radians internally.
    #[inline]
    #[must_use]
    pub fn from_degrees(degrees: T) -> Self {
        Self::from_radians(degrees.to_radians())
    }

    /// Get this angle in radians.
    #[inline]
    #[must_use]
    pub fn to_radians(self) -> T {
        self.radians
    }

    forward_to_unit!(
        #[doc = "See [num_traits::Float::to_degrees()]"]
        fn to_degrees(self) -> T);
    forward_to_unit!(
        #[doc = "See [num_traits::Float::sin()]."]
        fn sin(self) -> T);
    forward_to_unit!(
        #[doc = "See [num_traits::Float::cos()]."]
        fn cos(self) -> T);
    forward_to_unit!(
        #[doc = "See [num_traits::Float::tan()]."]
        fn tan(self) -> T);
    forward_to_unit!(
        #[doc = "See [num_traits::Float::sin_cos()]."]
        fn sin_cos(self) -> (T, T));

    forward_to_unit_as_self!(
        #[doc = "See [num_traits::Float::min()]."]
        fn min(self, other: Self) -> Self);
    forward_to_unit_as_self!(
        #[doc = "See [num_traits::Float::max()]."]
        fn max(self, other: Self) -> Self);
}

impl<T: Float + AngleConsts> Angle<T> {
    /// 2π
    pub const CIRCLE: Self = Self::TAU;
    /// π
    pub const HALF_CIRCLE: Self = Self::PI;
    /// π/2
    pub const QUARTER_CIRCLE: Self = Self::FRAG_PI_2;
}

impl<T> From<T> for Angle<T> {
    fn from(radians: T) -> Self {
        Angle { radians }
    }
}

macro_rules! forward_op_scalar {
    ($trait_name:ident, $op:ident) => {
        impl<T: Float> $trait_name<T> for Angle<T> {
            type Output = Self;

            fn $op(self, rhs: T) -> Self {
                Angle {
                    radians: self.radians.$op(rhs),
                }
            }
        }

        forward_op_scalar_vector!($trait_name, $op, Vector2);
        forward_op_scalar_vector!($trait_name, $op, Vector3);
        forward_op_scalar_vector!($trait_name, $op, Vector4);
    };
}

macro_rules! forward_op_scalar_vector {
    ($trait_name:ident, $op:ident, $vector:ident) => {
        impl<T: Float + Primitive + AngleConsts> $trait_name<$vector<T>> for $vector<Angle<T>> {
            type Output = $vector<Angle<T>>;

            fn $op(self, rhs: $vector<T>) -> Self {
                Self::from_untyped(self.to_untyped().$op(rhs.to_untyped()))
            }
        }
    };
}

macro_rules! forward_op_scalar_assign {
    ($trait_name:ident, $op:ident) => {
        impl<T: Float + NumAssignOps> $trait_name<T> for Angle<T> {
            fn $op(&mut self, rhs: T) {
                self.radians.$op(rhs);
            }
        }

        forward_op_scalar_assign_vector!($trait_name, $op, Vector2);
        forward_op_scalar_assign_vector!($trait_name, $op, Vector3);
        forward_op_scalar_assign_vector!($trait_name, $op, Vector4);
    };
}

macro_rules! forward_op_scalar_assign_vector {
    ($trait_name:ident, $op:ident, $vector:ident) => {
        impl<T: Float + Primitive + AngleConsts> $trait_name<$vector<T>> for $vector<Angle<T>> {
            fn $op(&mut self, rhs: $vector<T>) {
                self.as_raw_mut().$op(rhs.to_raw());
            }
        }
    };
}

macro_rules! forward_op_self {
    ($trait_name:ident, $op:ident) => {
        impl<T: Float> $trait_name for Angle<T> {
            type Output = Self;

            fn $op(self, rhs: Self) -> Self {
                Angle {
                    radians: self.radians.$op(rhs.radians),
                }
            }
        }
    };
}

macro_rules! forward_op_self_assign {
    ($trait_name:ident, $op:ident) => {
        impl<T: Float + NumAssignOps> $trait_name for Angle<T> {
            fn $op(&mut self, rhs: Self) {
                self.radians.$op(rhs.radians);
            }
        }
    };
}

forward_op_scalar!(Mul, mul);
forward_op_scalar!(Div, div);
forward_op_scalar!(Rem, rem);
forward_op_scalar_assign!(MulAssign, mul_assign);
forward_op_scalar_assign!(DivAssign, div_assign);
forward_op_scalar_assign!(RemAssign, rem_assign);

forward_op_self!(Add, add);
forward_op_self!(Sub, sub);
forward_op_self!(Rem, rem);
forward_op_self_assign!(AddAssign, add_assign);
forward_op_self_assign!(SubAssign, sub_assign);
forward_op_self_assign!(RemAssign, rem_assign);

impl<T: Primitive + Div> Div<Angle<T>> for Angle<T> {
    type Output = T;

    fn div(self, rhs: Angle<T>) -> Self::Output {
        self.radians / rhs.radians
    }
}

impl<T: AngleConsts> AngleConsts for Angle<T> {
    const PI: Self = Angle { radians: T::PI };
    const TAU: Self = Angle { radians: T::TAU };

    const FRAG_1_PI: Self = Angle {
        radians: T::FRAG_1_PI,
    };
    const FRAG_2_PI: Self = Angle {
        radians: T::FRAG_2_PI,
    };
    const FRAG_PI_2: Self = Angle {
        radians: T::FRAG_PI_2,
    };
    const FRAG_PI_3: Self = Angle {
        radians: T::FRAG_PI_3,
    };
    const FRAG_PI_4: Self = Angle {
        radians: T::FRAG_PI_4,
    };
    const FRAG_PI_6: Self = Angle {
        radians: T::FRAG_PI_6,
    };
    const FRAG_PI_8: Self = Angle {
        radians: T::FRAG_PI_8,
    };
}

impl<T> Angle<T>
where
    T: PrimitiveMatrices,
{
    /// Create quaternion from this angle and an axis vector.
    #[inline]
    #[must_use]
    pub fn to_rotation(self, axis: Vector3<T>) -> T::Quat {
        <T::Quat as Quat>::from_axis_angle(axis.to_raw(), self.radians)
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq, assert_ulps_eq};

    use super::*;

    type Angle = super::Angle<f32>;
    type AngleVec = Vector4<Angle>;
    type Vec4 = Vector4<f32>;

    #[test]
    fn forward_to_float() {
        use super::FloatAngleExt;
        let s: Angle = FloatAngleExt::asin(1.0);
        let c: Angle = FloatAngleExt::acos(1.0);
        let t: Angle = FloatAngleExt::atan(1.0);
        assert_eq!(s.radians, f32::asin(1.0));
        assert_eq!(c.radians, f32::acos(1.0));
        assert_eq!(t.radians, f32::atan(1.0));
    }

    #[test]
    fn angle_arithmetic() {
        let a = Angle::from_radians(1.0);
        let b = Angle::from_radians(-0.5);
        assert_abs_diff_eq!(a + b, Angle::from_radians(0.5));
        let x: Angle = a * 2.0f32;
        assert_abs_diff_eq!(x, Angle::from_radians(2.0));
        let y: Angle = a / 2.0f32;
        assert_abs_diff_eq!(y, Angle::from_radians(0.5));
        let z: f32 = a / b;
        assert_abs_diff_eq!(z, -2.0);
    }

    #[test]
    fn angle_degrees() {
        let circle = Angle::CIRCLE;
        assert_abs_diff_eq!(circle, Angle::from_degrees(360.0));
        assert_ulps_eq!(circle, Angle::from_degrees(360.0));
        assert_relative_eq!(circle, Angle::from_degrees(360.0));

        let quarter_circle = Angle::FRAG_PI_2;
        assert_abs_diff_eq!(quarter_circle, Angle::from_degrees(90.0));
        assert_ulps_eq!(quarter_circle, Angle::from_degrees(90.0));
        assert_relative_eq!(quarter_circle, Angle::from_degrees(90.0));
    }

    #[test]
    fn angle_cast() {
        let mut a = Angle::CIRCLE;
        let _: &f32 = a.as_raw();
        let _: &mut f32 = a.as_raw_mut();
        let _: f32 = a.to_raw();
        let _: f32 = a.to_radians();
        let _: Angle = 1.0.into();
    }

    #[test]
    fn angle_vec() {
        let mut vec = AngleVec::splat(Angle::PI);
        vec *= Vec4::splat(2.0);
        assert_eq!(vec, (Angle::TAU, Angle::TAU, Angle::TAU, Angle::TAU));
    }

    #[test]
    fn angle_vector_cast() {
        let mut vec = AngleVec::splat(Angle::PI);
        let _: &glam::Vec4 = vec.as_raw();
        let _: &mut glam::Vec4 = vec.as_raw_mut();
        let _: glam::Vec4 = vec.to_raw();
        let _: &Vec4 = vec.as_untyped();
        let _: &mut Vec4 = vec.as_untyped_mut();
        let _: Vec4 = vec.to_untyped();
        let _: &[Angle] = vec.as_ref();
        let _: &mut [Angle] = vec.as_mut();
        let _: &[Angle; 4] = vec.as_ref();
        let _: &mut [Angle; 4] = vec.as_mut();
        let _: Angle = vec.const_get::<0>();
        let _: AngleVec = vec.swizzle::<3, 2, 1, 0>();
        let _: AngleVec = -vec;
        let _: AngleVec = vec % vec;
        let _: AngleVec = vec * 2.0;

        assert_eq!(vec[0], Angle::PI);
    }

    #[test]
    fn to_rotation() {
        let angle = Angle::HALF_CIRCLE;
        let quat = angle.to_rotation(Vector3::unit_z());
        assert_abs_diff_eq!(quat, glam::Quat::from_axis_angle(glam::Vec3::Z, f32::PI));
        let v = quat * Vector3::<f32>::unit_x();
        assert_abs_diff_eq!(v, -Vector3::unit_x());
    }

    struct BufWriter<'a> {
        buffer: &'a mut [u8],
        pos: usize,
    }

    impl<'a> core::fmt::Write for BufWriter<'a> {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let bytes = s.as_bytes();
            if self.buffer.len() < bytes.len() + self.pos {
                Err(core::fmt::Error)
            } else {
                (&mut self.buffer[self.pos..self.pos + bytes.len()]).copy_from_slice(bytes);
                self.pos += bytes.len();
                Ok(())
            }
        }
    }

    fn write_buf<'a>(buffer: &'a mut [u8], fmt: core::fmt::Arguments) -> &'a str {
        use core::fmt::Write;
        let mut writer = BufWriter { buffer, pos: 0 };
        writer.write_fmt(fmt).unwrap();
        core::str::from_utf8(&writer.buffer[0..writer.pos]).unwrap()
    }

    #[test]
    fn angle_debug_print() {
        let mut buffer = [0; 128];

        assert_eq!(
            write_buf(&mut buffer, format_args!("{:?}", Angle::HALF_CIRCLE)),
            "Angle(π)"
        );
        assert_eq!(
            write_buf(&mut buffer, format_args!("{:?}", Angle::CIRCLE)),
            "Angle(2π)"
        );
        assert_eq!(
            write_buf(&mut buffer, format_args!("{:?}", Angle::PI)),
            "Angle(π)"
        );
        assert_eq!(
            write_buf(&mut buffer, format_args!("{:?}", Angle::TAU)),
            "Angle(2π)"
        );

        assert_eq!(
            write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_1_PI)),
            "Angle(1/π)"
        );
        assert_eq!(
            write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_2_PI)),
            "Angle(2/π)"
        );
        assert_eq!(
            write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_PI_2)),
            "Angle(π/2)"
        );
        assert_eq!(
            write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_PI_3)),
            "Angle(π/3)"
        );
        assert_eq!(
            write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_PI_4)),
            "Angle(π/4)"
        );
        assert_eq!(
            write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_PI_6)),
            "Angle(π/6)"
        );
        assert_eq!(
            write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_PI_8)),
            "Angle(π/8)"
        );

        assert_eq!(
            write_buf(&mut buffer, format_args!("{:?}", Angle::from_radians(1.0))),
            "Angle(1.00000)"
        );
    }
}

//! Generic angles.

use approx::{AbsDiffEq, RelativeEq};
use bytemuck::{Pod, Zeroable};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use num_traits::{Float, NumAssignOps};

use crate::{traits::marker::ValueSemantics, Scalar, Unit, Vector2, Vector3, Vector4};

/// Angle in radians.
///
/// Note that Angle implements both [Scalar] and [Unit], so it is compatible
/// with vector types: `Vector2<Angle<f32>>` is a vector containing two angles.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, PartialOrd)]
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

unsafe impl<T: Zeroable> Zeroable for Angle<T> {}
unsafe impl<T: Pod> Pod for Angle<T> {}

impl<T> Scalar for Angle<T>
where
    T: Scalar,
{
    type Primitive = T::Primitive;
}

impl<T> Unit for Angle<T>
where
    T: Scalar + Unit<Scalar = T> + Float,
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

    fn default_epsilon() -> Self::Epsilon {
        T::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.radians.abs_diff_eq(&other.radians, epsilon)
    }

    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.radians.abs_diff_ne(&other.radians, epsilon)
    }
}

impl<T> RelativeEq for Angle<T>
where
    T: RelativeEq,
{
    fn default_max_relative() -> Self::Epsilon {
        todo!()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.radians
            .relative_eq(&other.radians, epsilon, max_relative)
    }

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

macro_rules! forward_to_unit {
    (
        $(#[$attr:meta])?
        fn $f:ident(self $(, $args:ident: $args_ty:ty)*) -> $ret:ty) => {
        $(#[$attr])*
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
        pub fn $f(self,  $($args: $args_ty),*) -> Angle<T> {
            Angle::from_radians(self.radians.$f($($args.radians),*))
        }
    };
}

impl<T: Float> Angle<T> {
    /// Angle from radians.
    pub fn from_radians(radians: T) -> Self {
        Angle { radians }
    }

    /// Angle from degrees.
    ///
    /// Will be converted to radians internally.
    pub fn from_degrees(degrees: T) -> Self {
        Self::from_radians(degrees.to_radians())
    }

    /// Get this angle in radians.
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
        impl<T: Float + Scalar + Unit<Scalar = T> + AngleConsts> $trait_name<$vector<T>>
            for $vector<Angle<T>>
        {
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
        impl<T: Float + Scalar + Unit<Scalar = T> + AngleConsts> $trait_name<$vector<T>>
            for $vector<Angle<T>>
        {
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

#[doc(hidden)]
impl<T: Float> Div for Angle<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        Angle::from_radians(self.radians / rhs.radians)
    }
}

#[doc(hidden)]
impl<T: Float> Mul for Angle<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Angle::from_radians(self.radians * rhs.radians)
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

#[cfg(test)]
mod tests {
    use super::*;

    type Angle = super::Angle<f32>;
    type AngleVec = Vector4<Angle>;
    type Vec4 = Vector4<f32>;

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
}

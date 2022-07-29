#![doc = include_str!("../README.md")]
#![no_std]
#![deny(missing_docs, clippy::useless_conversion, clippy::useless_asref)]
#![warn(clippy::pedantic)]
#![allow(
    clippy::inline_always,
    clippy::module_name_repetitions,
    clippy::wildcard_imports
)]
#![cfg_attr(coverage, feature(no_coverage))]

#[cfg(doc)]
#[cfg_attr(coverage, no_coverage)]
pub mod docs;

mod angle;
#[doc(hidden)]
pub mod bindings;
mod r#box;
mod matrix;
mod point;
mod rect;
mod scalar;
mod size;
pub mod traits;

mod compat;
mod raw;
mod transform;
mod unit;
mod vector;

pub use angle::{Angle, AngleConsts, FloatAngleExt};
pub use matrix::{Matrix2, Matrix3, Matrix4};
pub use point::{Point2, Point3, Point4};
pub use r#box::{Box2, Box3};
pub use raw::*;
pub use rect::Rect;
pub use scalar::Scalar;
pub use size::{Size2, Size3};
pub use transform::{Transform2, Transform3, TransformMap};
pub use unit::Unit;
pub use vector::{Vector2, Vector3, Vector4};

mod macros;
use macros::*;

#[doc(no_inline)]
pub use traits::{Contains, Intersection, Union};

/// Convenience glob import.
///
/// All traits are imported anonymously, except [`Unit`].
pub mod prelude {
    #[doc(no_inline)]
    pub use super::{
        Angle, AngleConsts as _, AsRaw, Box2, Box3, FloatAngleExt as _, FromRawRef, Matrix2,
        Matrix3, Matrix4, Point2, Point3, Point4, Rect, Scalar as _, Size2, Size3, ToRaw, Unit,
        Vector2, Vector3, Vector4,
    };
    #[doc(no_inline)]
    pub use super::{Transform2, Transform3, TransformMap as _};

    #[doc(no_inline)]
    pub use super::traits::{Contains as _, Intersection as _, Union as _};

    #[doc(no_inline)]
    pub use super::{point, point2, point3, point4, size, size2, size3, vec2, vec3, vec4, vector};
}

/// Construct a [`Vector2`]. Usable in const contexts.
#[macro_export]
macro_rules! vec2 {
    ($x:expr, $y:expr) => {
        $crate::Vector2 { x: $x, y: $y }
    };
    [$x:expr, $y:expr] => {
        $crate::Vector2 { x: $x, y: $y }
    };
    ($splat:expr) => {{
        let splat = $splat;
        $crate::Vector2 {
            x: splat,
            y: splat,
        }
    }};
}

/// Construct a [`Vector3`]. Usable in const contexts.
#[macro_export]
macro_rules! vec3 {
    ($x:expr, $y:expr, $z:expr) => {
        $crate::Vector3 { x: $x, y: $y, z: $z }
    };
    [$x:expr, $y:expr, $z:expr] => {
        $crate::Vector3 { x: $x, y: $y, z: $z }
    };
    ($splat:expr) => {{
        let splat = $splat;
        $crate::Vector3 {
            x: splat,
            y: splat,
            z: splat,
        }
    }};
}

/// Construct a [`Vector4`]. Usable in const contexts.
#[macro_export]
macro_rules! vec4 {
    ($x:expr, $y:expr, $z:expr, $w:expr) => {
        $crate::Vector4 { x: $x, y: $y, z: $z, w: $w }
    };
    [$x:expr, $y:expr, $z:expr, $w:expr] => {
        $crate::Vector4 { x: $x, y: $y, z: $z, w: $w }
    };
    ($splat:expr) => {{
        let splat = $splat;
        $crate::Vector4 {
            x: splat,
            y: splat,
            z: splat,
            w: splat,
        }
    }};
}

/// Construct a [`Vector2`], [`Vector3`], or [`Vector4`] depending on the
/// number of arguments. Usable in const contexts.
#[macro_export]
macro_rules! vector {
    ($x:expr, $y:expr) => {
        $crate::vec2!($x, $y)
    };
    ($x:expr, $y:expr, $z:expr) => {
        $crate::vec3!($x, $y, $z)
    };
    ($x:expr, $y:expr, $z:expr, $w:expr) => {
        $crate::vec4!($x, $y, $z, $w)
    };

    [$x:expr, $y:expr] => {
        $crate::vec2!($x, $y)
    };
    [$x:expr, $y:expr, $z:expr] => {
        $crate::vec3!($x, $y, $z)
    };
    [$x:expr, $y:expr, $z:expr, $w:expr] => {
        $crate::vec4!($x, $y, $z, $w)
    };

    ([$splat:expr; 2]) => {
        $crate::vec2!($splat)
    };
    ([$splat:expr; 3]) => {
        $crate::vec3!($splat)
    };
    ([$splat:expr; 4]) => {
        $crate::vec4!($splat)
    };
}

/// Construct a [`Point2`]. Usable in const contexts.
#[macro_export]
macro_rules! point2 {
    ($x:expr, $y:expr) => {
        $crate::Point2 { x: $x, y: $y }
    };
    [$x:expr, $y:expr] => {
        $crate::Point2 { x: $x, y: $y }
    };
    ($splat:expr) => {{
        let splat = $splat;
        $crate::Point2 {
            x: splat,
            y: splat,
        }
    }};
}

/// Construct a [`Point3`]. Usable in const contexts.
#[macro_export]
macro_rules! point3 {
    ($x:expr, $y:expr, $z:expr) => {
        $crate::Point3 { x: $x, y: $y, z: $z }
    };
    [$x:expr, $y:expr, $z:expr] => {
        $crate::Point3 { x: $x, y: $y, z: $z }
    };
    ($splat:expr) => {{
        let splat = $splat;
        $crate::Point3 {
            x: splat,
            y: splat,
            z: splat,
        }
    }};
}

/// Construct a [`Point4`]. Usable in const contexts.
#[macro_export]
macro_rules! point4 {
    ($x:expr, $y:expr, $z:expr, $w:expr) => {
        $crate::Point4 { x: $x, y: $y, z: $z, w: $w }
    };
    [$x:expr, $y:expr, $z:expr, $w:expr] => {
        $crate::Point4 { x: $x, y: $y, z: $z, w: $w }
    };
    ($splat:expr) => {{
        let splat = $splat;
        $crate::Point4 {
            x: splat,
            y: splat,
            z: splat,
            w: splat,
        }
    }};
}

/// Construct a [`Point2`], [`Point3`], or [`Point4`] depending on the
/// number of arguments. Usable in const contexts.
#[macro_export]
macro_rules! point {
    ($x:expr, $y:expr) => {
        $crate::point2!($x, $y)
    };
    ($x:expr, $y:expr, $z:expr) => {
        $crate::point3!($x, $y, $z)
    };
    ($x:expr, $y:expr, $z:expr, $w:expr) => {
        $crate::point4!($x, $y, $z, $w)
    };

    [$x:expr, $y:expr] => {
        $crate::point2!($x, $y)
    };
    [$x:expr, $y:expr, $z:expr] => {
        $crate::point3!($x, $y, $z)
    };
    [$x:expr, $y:expr, $z:expr, $w:expr] => {
        $crate::point4!($x, $y, $z, $w)
    };

    ([$splat:expr; 2]) => {
        $crate::point2!($splat)
    };
    ([$splat:expr; 3]) => {
        $crate::point3!($splat)
    };
    ([$splat:expr; 4]) => {
        $crate::point4!($splat)
    };
}

/// Construct a [`Size2`]. Usable in const contexts.
#[macro_export]
macro_rules! size2 {
    ($width:expr, $height:expr) => {
        $crate::Size2 { width: $width, height: $height }
    };
    [$width:expr, $height:expr] => {
        $crate::Size2 { width: $width, height: $height }
    };
    ($splat:expr) => {{
        let splat = $splat;
        $crate::Size2 {
            width: splat,
            height: splat,
        }
    }};
}

/// Construct a [`Size3`]. Usable in const contexts.
#[macro_export]
macro_rules! size3 {

    ($width:expr, $height:expr, $depth:expr) => {
        $crate::Size3 { width: $width, height: $height, depth: $depth }
    };
    [$width:expr, $height:expr, $depth:expr] => {
        $crate::Size3 { width: $width, height: $height, depth: $depth }
    };
    ($splat:expr) => {{
        let splat = $splat;
        $crate::Size3 {
            width: splat,
            height: splat,
            depth: splat,
        }
    }};
}

/// Construct a [`Size2`] or [`Size3`] depending on the number of arguments.
/// Usable in const contexts.
#[macro_export]
macro_rules! size {
    ($width:expr, $height:expr) => {
        $crate::size2!($width, $height)
    };
    ($width:expr, $height:expr, $depth:expr) => {
        $crate::size3!($width, $height, $depth)
    };

    [$width:expr, $height:expr] => {
        $crate::size2!($width, $height)
    };
    [$width:expr, $height:expr, $depth:expr] => {
        $crate::size3!($width, $height, $depth)
    };

    ([$splat:expr; 2]) => {
        $crate::size2!($splat)
    };

    ([$splat:expr; 3]) => {
        $crate::size3!($splat)
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    //
    // Some units for testing:
    //

    struct UnitF32;
    impl Unit for UnitF32 {
        type Scalar = f32;
    }
    struct UnitF64;
    impl Unit for UnitF64 {
        type Scalar = f64;
    }
    struct UnitI32;
    impl Unit for UnitI32 {
        type Scalar = i32;
    }
    struct UnitU32;
    impl Unit for UnitU32 {
        type Scalar = u32;
    }

    #[test]
    fn basic() {
        let v1 = <Vector2>::new(123.0, 456.0);
        let v2 = Vector2 { x: 2.0, y: 3.0 };
        let v3 = v1 + v2;
        assert_eq!(v3, (125.0, 459.0));
    }

    #[test]
    fn type_alias() {
        type V = Vector2<UnitF32>;
        let _ = V { x: 1.0, y: 2.0 };
    }

    #[test]
    fn try_cast() {
        let v = Vector4::<f32>::new(core::f32::MAX, 0.0, 1.0, 2.0);
        let t: Option<Vector4<i32>> = v.try_cast();
        assert!(t.is_none());

        let v = Vector4::<f32>::new(3.0, 0.0, 1.0, 2.0);
        let t: Option<Vector4<i32>> = v.try_cast();
        assert_eq!(t, Some(vec4!(3, 0, 1, 2)));
    }
}

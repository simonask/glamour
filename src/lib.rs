#![doc = include_str!("../README.md")]
#![no_std]
#![deny(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::inline_always, clippy::module_name_repetitions)]

#[cfg(feature = "mint")]
extern crate mint_crate as mint;

#[cfg(doc)]
pub mod docs;

mod angle;
pub mod bindings;
mod r#box;
mod matrix;
mod point;
mod rect;
mod scalar;
mod size;
pub mod traits;

mod transform;
mod unit;
mod vector;

pub use angle::{Angle, AngleConsts, FloatAngleExt};
pub use matrix::{Matrix2, Matrix3, Matrix4};
pub use point::{Point2, Point3, Point4};
pub use r#box::{Box2, Box3};
pub use rect::Rect;
pub use scalar::Scalar;
pub use size::{Size2, Size3};
pub use transform::{Transform2, Transform3, TransformMap};
pub use unit::{Unit, UnitTypes};
pub use vector::{Vector2, Vector3, Vector4};

mod macros;
use macros::*;

#[doc(no_inline)]
pub use traits::{Contains, Intersection, Lerp, Union};

/// Convenience glob import.
///
/// All traits are imported anonymously, except [`Unit`] and [`Scalar`].
pub mod prelude {
    #[doc(no_inline)]
    pub use super::{
        Angle, AngleConsts as _, Box2, Box3, FloatAngleExt as _, Matrix2, Matrix3, Matrix4, Point2,
        Point3, Point4, Rect, Scalar, Size2, Size3, Unit, Vector2, Vector3, Vector4,
    };
    #[doc(no_inline)]
    pub use super::{Transform2, Transform3, TransformMap as _};

    #[doc(no_inline)]
    pub use super::traits::{Contains as _, Intersection as _, Lerp as _, Union as _};

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
    use approx::{AbsDiffEq, RelativeEq, UlpsEq};

    use crate::traits::{marker::ValueSemantics, Lerp};

    use super::*;
    use core::ops::{
        Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign,
    };

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

    //
    // A whole bunch of traits that exist to check if a type implements its
    // supertraits.
    //
    // This is so we can statically check that the dark trait magic actually
    // results in the correct traits being implemented for our types.
    //

    trait AssertValueLike: ValueSemantics {}

    trait AssertSigned: AssertValueLike + core::ops::Neg<Output = Self> {}

    trait AssertScalable<Unit>:
        AssertValueLike + MulAssign<Unit> + DivAssign<Unit> + Mul<Unit> + Div<Unit>
    {
    }

    trait AssertVectorLike<Unit>:
        AssertValueLike
        + AssertScalable<Unit>
        + Add<Self, Output = Self>
        + Sub<Self, Output = Self>
        + Mul<Self, Output = Self>
        + Div<Self, Output = Self>
        + Rem<Self, Output = Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + DivAssign
        + RemAssign
        + AbsDiffEq
    {
    }

    trait AssertEq: Eq {}

    trait AssertFloatVectorLike<T>: AssertVectorLike<T> + Lerp<T> + RelativeEq + UlpsEq {}

    trait AssertPointLike<Unit, DiffType>:
        Sub<Self, Output = DiffType> + Add<DiffType, Output = Self> + Sized
    where
        DiffType: AssertVectorLike<Unit>,
    {
    }

    trait AssertSize2DLike<Unit>:
        AssertScalable<Unit> + Sub<Self, Output = Self> + Add<Self, Output = Self>
    {
    }

    trait AssertSize3DLike<Unit>:
        AssertScalable<Unit> + Sub<Self, Output = Self> + Add<Self, Output = Self>
    {
    }

    trait AssertMatrixLike<Vector>:
        AssertValueLike + Mul<Self, Output = Self> + Mul<Vector, Output = Vector>
    {
    }

    trait AssertRefConversion<T>: Into<T> + From<T> + AsRef<T> + AsMut<T> {}

    trait AssertConversion<T>: Into<T> + From<T> {}

    trait AssertTuple2<T>: Into<(T, T)> + From<(T, T)> {}
    trait AssertTuple3<T>: Into<(T, T, T)> + From<(T, T, T)> {}
    trait AssertTuple4<T>: Into<(T, T, T, T)> + From<(T, T, T, T)> {}

    const _CHECK_TYPES: () = {
        impl AssertValueLike for Matrix2<f32> {}
        impl AssertValueLike for Matrix2<f64> {}
        impl AssertValueLike for Matrix3<f32> {}
        impl AssertValueLike for Matrix3<f64> {}
        impl AssertValueLike for Matrix4<f32> {}
        impl AssertValueLike for Matrix4<f64> {}
        impl AssertValueLike for Point2<UnitF32> {}
        impl AssertValueLike for Point2<UnitF64> {}
        impl AssertValueLike for Point2<UnitI32> {}
        impl AssertValueLike for Point2<UnitU32> {}
        impl AssertValueLike for Point3<UnitF32> {}
        impl AssertValueLike for Point3<UnitF64> {}
        impl AssertValueLike for Point3<UnitI32> {}
        impl AssertValueLike for Point3<UnitU32> {}
        impl AssertValueLike for Point4<UnitF32> {}
        impl AssertValueLike for Point4<UnitF64> {}
        impl AssertValueLike for Point4<UnitI32> {}
        impl AssertValueLike for Point4<UnitU32> {}
        impl AssertValueLike for Rect<UnitF32> {}
        impl AssertValueLike for Rect<UnitF64> {}
        impl AssertValueLike for Rect<UnitI32> {}
        impl AssertValueLike for Rect<UnitU32> {}
        impl AssertValueLike for Size2<UnitF32> {}
        impl AssertValueLike for Size2<UnitF64> {}
        impl AssertValueLike for Size2<UnitI32> {}
        impl AssertValueLike for Size2<UnitU32> {}
        impl AssertValueLike for Size3<UnitF32> {}
        impl AssertValueLike for Size3<UnitF64> {}
        impl AssertValueLike for Size3<UnitI32> {}
        impl AssertValueLike for Size3<UnitU32> {}
        impl AssertValueLike for Transform2<UnitF32, UnitF32> {}
        impl AssertValueLike for Transform2<UnitF64, UnitF64> {}
        impl AssertValueLike for Transform3<UnitF32, UnitF32> {}
        impl AssertValueLike for Transform3<UnitF64, UnitF64> {}
        impl AssertValueLike for Vector2<UnitF32> {}
        impl AssertValueLike for Vector2<UnitF64> {}
        impl AssertValueLike for Vector2<UnitI32> {}
        impl AssertValueLike for Vector2<UnitU32> {}
        impl AssertValueLike for Vector3<UnitF32> {}
        impl AssertValueLike for Vector3<UnitF64> {}
        impl AssertValueLike for Vector3<UnitI32> {}
        impl AssertValueLike for Vector3<UnitU32> {}
        impl AssertValueLike for Vector4<UnitF32> {}
        impl AssertValueLike for Vector4<UnitF64> {}
        impl AssertValueLike for Vector4<UnitI32> {}
        impl AssertValueLike for Vector4<UnitU32> {}

        impl AssertSigned for Point2<UnitF32> {}
        impl AssertSigned for Point2<UnitF64> {}
        impl AssertSigned for Point2<UnitI32> {}
        impl AssertSigned for Point3<UnitF32> {}
        impl AssertSigned for Point3<UnitF64> {}
        impl AssertSigned for Point3<UnitI32> {}
        impl AssertSigned for Point4<UnitF32> {}
        impl AssertSigned for Point4<UnitF64> {}
        impl AssertSigned for Point4<UnitI32> {}
        impl AssertSigned for Vector2<UnitF32> {}
        impl AssertSigned for Vector2<UnitF64> {}
        impl AssertSigned for Vector2<UnitI32> {}
        impl AssertSigned for Vector3<UnitF32> {}
        impl AssertSigned for Vector3<UnitF64> {}
        impl AssertSigned for Vector3<UnitI32> {}
        impl AssertSigned for Vector4<UnitF32> {}
        impl AssertSigned for Vector4<UnitF64> {}
        impl AssertSigned for Vector4<UnitI32> {}

        impl AssertEq for Box2<UnitI32> {}
        impl AssertEq for Box2<UnitU32> {}
        impl AssertEq for Box3<UnitI32> {}
        impl AssertEq for Box3<UnitU32> {}
        impl AssertEq for Point2<UnitI32> {}
        impl AssertEq for Point3<UnitI32> {}
        impl AssertEq for Point4<UnitI32> {}
        impl AssertEq for Rect<UnitI32> {}
        impl AssertEq for Rect<UnitU32> {}
        impl AssertEq for Size2<UnitI32> {}
        impl AssertEq for Size2<UnitU32> {}
        impl AssertEq for Size3<UnitI32> {}
        impl AssertEq for Size3<UnitU32> {}
        impl AssertEq for Vector2<UnitI32> {}
        impl AssertEq for Vector3<UnitI32> {}
        impl AssertEq for Vector4<UnitI32> {}

        impl AssertFloatVectorLike<f32> for Vector2<UnitF32> {}
        impl AssertFloatVectorLike<f32> for Vector3<UnitF32> {}
        impl AssertFloatVectorLike<f32> for Vector4<UnitF32> {}
        impl AssertFloatVectorLike<f64> for Vector2<UnitF64> {}
        impl AssertFloatVectorLike<f64> for Vector3<UnitF64> {}
        impl AssertFloatVectorLike<f64> for Vector4<UnitF64> {}
        impl AssertVectorLike<f32> for Vector2<UnitF32> {}
        impl AssertVectorLike<f32> for Vector3<UnitF32> {}
        impl AssertVectorLike<f32> for Vector4<UnitF32> {}
        impl AssertVectorLike<f64> for Vector2<UnitF64> {}
        impl AssertVectorLike<f64> for Vector3<UnitF64> {}
        impl AssertVectorLike<f64> for Vector4<UnitF64> {}
        impl AssertVectorLike<i32> for Vector2<UnitI32> {}
        impl AssertVectorLike<i32> for Vector3<UnitI32> {}
        impl AssertVectorLike<i32> for Vector4<UnitI32> {}
        impl AssertVectorLike<u32> for Vector2<UnitU32> {}
        impl AssertVectorLike<u32> for Vector3<UnitU32> {}
        impl AssertVectorLike<u32> for Vector4<UnitU32> {}

        impl AssertPointLike<f32, Vector2<UnitF32>> for Point2<UnitF32> {}
        impl AssertPointLike<f32, Vector3<UnitF32>> for Point3<UnitF32> {}
        impl AssertPointLike<f32, Vector4<UnitF32>> for Point4<UnitF32> {}
        impl AssertPointLike<f64, Vector2<UnitF64>> for Point2<UnitF64> {}
        impl AssertPointLike<f64, Vector3<UnitF64>> for Point3<UnitF64> {}
        impl AssertPointLike<f64, Vector4<UnitF64>> for Point4<UnitF64> {}

        impl AssertSize2DLike<f32> for Size2<UnitF32> {}
        impl AssertSize2DLike<f64> for Size2<UnitF64> {}
        impl AssertSize2DLike<i32> for Size2<UnitI32> {}
        impl AssertSize2DLike<u32> for Size2<UnitU32> {}

        impl AssertScalable<f32> for Vector2<UnitF32> {}
        impl AssertScalable<f32> for Vector3<UnitF32> {}
        impl AssertScalable<f32> for Vector4<UnitF32> {}
        impl AssertScalable<f32> for Size2<UnitF32> {}
        impl AssertScalable<f32> for Size3<UnitF32> {}
        impl AssertScalable<f64> for Vector2<UnitF64> {}
        impl AssertScalable<f64> for Vector3<UnitF64> {}
        impl AssertScalable<f64> for Vector4<UnitF64> {}
        impl AssertScalable<f64> for Size2<UnitF64> {}
        impl AssertScalable<f64> for Size3<UnitF64> {}
        impl AssertScalable<i32> for Vector2<UnitI32> {}
        impl AssertScalable<i32> for Vector3<UnitI32> {}
        impl AssertScalable<i32> for Vector4<UnitI32> {}
        impl AssertScalable<i32> for Size2<UnitI32> {}
        impl AssertScalable<i32> for Size3<UnitI32> {}
        impl AssertScalable<u32> for Vector2<UnitU32> {}
        impl AssertScalable<u32> for Vector3<UnitU32> {}
        impl AssertScalable<u32> for Vector4<UnitU32> {}
        impl AssertScalable<u32> for Size2<UnitU32> {}
        impl AssertScalable<u32> for Size3<UnitU32> {}

        impl AssertRefConversion<glam::DVec2> for Point2<UnitF64> {}
        impl AssertRefConversion<glam::DVec2> for Size2<UnitF64> {}
        impl AssertRefConversion<glam::DVec2> for Vector2<UnitF64> {}
        impl AssertRefConversion<glam::DVec3> for Point3<UnitF64> {}
        impl AssertRefConversion<glam::DVec3> for Size3<UnitF64> {}
        impl AssertRefConversion<glam::DVec3> for Vector3<UnitF64> {}
        impl AssertRefConversion<glam::DVec4> for Point4<UnitF64> {}
        impl AssertRefConversion<glam::DVec4> for Vector4<UnitF64> {}
        impl AssertRefConversion<glam::IVec2> for Point2<UnitI32> {}
        impl AssertRefConversion<glam::IVec2> for Vector2<UnitI32> {}
        impl AssertRefConversion<glam::IVec3> for Point3<UnitI32> {}
        impl AssertRefConversion<glam::IVec3> for Vector3<UnitI32> {}
        impl AssertRefConversion<glam::IVec4> for Point4<UnitI32> {}
        impl AssertRefConversion<glam::IVec4> for Vector4<UnitI32> {}
        impl AssertRefConversion<glam::UVec2> for Point2<UnitU32> {}
        impl AssertRefConversion<glam::UVec2> for Size2<UnitU32> {}
        impl AssertRefConversion<glam::UVec2> for Vector2<UnitU32> {}
        impl AssertRefConversion<glam::UVec3> for Point3<UnitU32> {}
        impl AssertRefConversion<glam::UVec3> for Size3<UnitU32> {}
        impl AssertRefConversion<glam::UVec3> for Vector3<UnitU32> {}
        impl AssertRefConversion<glam::UVec4> for Point4<UnitU32> {}
        impl AssertRefConversion<glam::UVec4> for Vector4<UnitU32> {}
        impl AssertRefConversion<glam::Vec2> for Point2<UnitF32> {}
        impl AssertRefConversion<glam::Vec2> for Size2<UnitF32> {}
        impl AssertRefConversion<glam::Vec2> for Vector2<UnitF32> {}
        impl AssertRefConversion<glam::Vec3> for Point3<UnitF32> {}
        impl AssertRefConversion<glam::Vec3> for Size3<UnitF32> {}
        impl AssertRefConversion<glam::Vec3> for Vector3<UnitF32> {}
        impl AssertRefConversion<glam::Vec4> for Point4<UnitF32> {}
        impl AssertRefConversion<glam::Vec4> for Vector4<UnitF32> {}

        impl AssertRefConversion<[f64; 2]> for Point2<UnitF64> {}
        impl AssertRefConversion<[f64; 2]> for Size2<UnitF64> {}
        impl AssertRefConversion<[f64; 2]> for Vector2<UnitF64> {}
        impl AssertRefConversion<[f64; 3]> for Point3<UnitF64> {}
        impl AssertRefConversion<[f64; 3]> for Size3<UnitF64> {}
        impl AssertRefConversion<[f64; 3]> for Vector3<UnitF64> {}
        impl AssertRefConversion<[f64; 4]> for Point4<UnitF64> {}
        impl AssertRefConversion<[f64; 4]> for Vector4<UnitF64> {}
        impl AssertRefConversion<[i32; 2]> for Point2<UnitI32> {}
        impl AssertRefConversion<[i32; 2]> for Vector2<UnitI32> {}
        impl AssertRefConversion<[i32; 3]> for Point3<UnitI32> {}
        impl AssertRefConversion<[i32; 3]> for Vector3<UnitI32> {}
        impl AssertRefConversion<[i32; 4]> for Point4<UnitI32> {}
        impl AssertRefConversion<[i32; 4]> for Vector4<UnitI32> {}
        impl AssertRefConversion<[u32; 2]> for Point2<UnitU32> {}
        impl AssertRefConversion<[u32; 2]> for Size2<UnitU32> {}
        impl AssertRefConversion<[u32; 2]> for Vector2<UnitU32> {}
        impl AssertRefConversion<[u32; 3]> for Point3<UnitU32> {}
        impl AssertRefConversion<[u32; 3]> for Size3<UnitU32> {}
        impl AssertRefConversion<[u32; 3]> for Vector3<UnitU32> {}
        impl AssertRefConversion<[u32; 4]> for Point4<UnitU32> {}
        impl AssertRefConversion<[u32; 4]> for Vector4<UnitU32> {}
        impl AssertRefConversion<[f32; 2]> for Point2<UnitF32> {}
        impl AssertRefConversion<[f32; 2]> for Size2<UnitF32> {}
        impl AssertRefConversion<[f32; 2]> for Vector2<UnitF32> {}
        impl AssertRefConversion<[f32; 3]> for Point3<UnitF32> {}
        impl AssertRefConversion<[f32; 3]> for Size3<UnitF32> {}
        impl AssertRefConversion<[f32; 3]> for Vector3<UnitF32> {}
        impl AssertRefConversion<[f32; 4]> for Point4<UnitF32> {}
        impl AssertRefConversion<[f32; 4]> for Vector4<UnitF32> {}

        impl AssertTuple2<f32> for Point2<UnitF32> {}
        impl AssertTuple2<f32> for Size2<UnitF32> {}
        impl AssertTuple2<f32> for Vector2<UnitF32> {}
        impl AssertTuple2<f64> for Point2<UnitF64> {}
        impl AssertTuple2<f64> for Size2<UnitF64> {}
        impl AssertTuple2<f64> for Vector2<UnitF64> {}
        impl AssertTuple2<i32> for Point2<UnitI32> {}
        impl AssertTuple2<i32> for Vector2<UnitI32> {}
        impl AssertTuple2<u32> for Point2<UnitU32> {}
        impl AssertTuple2<u32> for Size2<UnitU32> {}
        impl AssertTuple2<u32> for Vector2<UnitU32> {}
        impl AssertTuple3<f32> for Point3<UnitF32> {}
        impl AssertTuple3<f32> for Size3<UnitF32> {}
        impl AssertTuple3<f32> for Vector3<UnitF32> {}
        impl AssertTuple3<f64> for Point3<UnitF64> {}
        impl AssertTuple3<f64> for Size3<UnitF64> {}
        impl AssertTuple3<f64> for Vector3<UnitF64> {}
        impl AssertTuple3<i32> for Point3<UnitI32> {}
        impl AssertTuple3<i32> for Vector3<UnitI32> {}
        impl AssertTuple3<u32> for Point3<UnitU32> {}
        impl AssertTuple3<u32> for Size3<UnitU32> {}
        impl AssertTuple3<u32> for Vector3<UnitU32> {}
        impl AssertTuple4<f32> for Point4<UnitF32> {}
        impl AssertTuple4<f32> for Vector4<UnitF32> {}
        impl AssertTuple4<f64> for Point4<UnitF64> {}
        impl AssertTuple4<f64> for Vector4<UnitF64> {}
        impl AssertTuple4<i32> for Point4<UnitI32> {}
        impl AssertTuple4<i32> for Vector4<UnitI32> {}
        impl AssertTuple4<u32> for Point4<UnitU32> {}
        impl AssertTuple4<u32> for Vector4<UnitU32> {}
        impl AssertTuple2<Point2<UnitF32>> for Box2<UnitF32> {}
        impl AssertTuple2<Point2<UnitF64>> for Box2<UnitF64> {}
        impl AssertTuple2<Point2<UnitI32>> for Box2<UnitI32> {}
        impl AssertTuple2<Point2<UnitU32>> for Box2<UnitU32> {}
        impl AssertTuple2<Point3<UnitF32>> for Box3<UnitF32> {}
        impl AssertTuple2<Point3<UnitF64>> for Box3<UnitF64> {}
        impl AssertTuple2<Point3<UnitI32>> for Box3<UnitI32> {}
        impl AssertTuple2<Point3<UnitU32>> for Box3<UnitU32> {}

        impl AssertConversion<(Point2<UnitF32>, Size2<UnitF32>)> for Rect<UnitF32> {}
        impl AssertConversion<(Point2<UnitF64>, Size2<UnitF64>)> for Rect<UnitF64> {}
        impl AssertConversion<(Point2<UnitI32>, Size2<UnitI32>)> for Rect<UnitI32> {}
        impl AssertConversion<(Point2<UnitU32>, Size2<UnitU32>)> for Rect<UnitU32> {}

        impl AssertMatrixLike<Vector2<f32>> for Matrix2<f32> {}
        impl AssertMatrixLike<Point2<f32>> for Matrix2<f32> {}
        impl AssertMatrixLike<Vector2<f32>> for Matrix3<f32> {}
        impl AssertMatrixLike<Point2<f32>> for Matrix3<f32> {}
        impl AssertMatrixLike<Vector3<f32>> for Matrix4<f32> {}
        impl AssertMatrixLike<Point3<f32>> for Matrix4<f32> {}
        impl AssertMatrixLike<Vector2<f64>> for Matrix2<f64> {}
        impl AssertMatrixLike<Point2<f64>> for Matrix2<f64> {}
        impl AssertMatrixLike<Vector2<f64>> for Matrix3<f64> {}
        impl AssertMatrixLike<Point2<f64>> for Matrix3<f64> {}
        impl AssertMatrixLike<Vector3<f64>> for Matrix4<f64> {}
        impl AssertMatrixLike<Point3<f64>> for Matrix4<f64> {}
    };

    #[test]
    fn basic() {
        let v1 = <Vector2>::new(123.0, 456.0);
        let v2 = Vector2 { x: 2.0, y: 3.0 };
        let v3 = v1 + v2;
        assert_eq!(v3, (125.0, 459.0));
    }

    #[test]
    fn swizzle() {
        let v: Vector4<f32> = [1.0, 2.0, 3.0, 4.0].into();

        let v4 = v.swizzle::<3, 1, 0, 2>();
        assert_eq!(v4, [4.0, 2.0, 1.0, 3.0]);

        let v2 = v.swizzle2::<0, 1>();
        assert_eq!(v2, [1.0, 2.0]);

        let v3 = v.swizzle3::<0, 1, 2>();
        assert_eq!(v3, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn type_alias() {
        type V = Vector2<UnitF32>;
        let _ = V { x: 1.0, y: 2.0 };
    }

    #[derive(
        Clone,
        Copy,
        Debug,
        Default,
        PartialEq,
        Eq,
        PartialOrd,
        Ord,
        derive_more::Add,
        derive_more::AddAssign,
        derive_more::Div,
        derive_more::DivAssign,
        derive_more::Mul,
        derive_more::MulAssign,
        derive_more::Rem,
        derive_more::RemAssign,
        derive_more::Sub,
        derive_more::SubAssign,
        bytemuck::Pod,
        bytemuck::Zeroable,
    )]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[repr(C)]
    struct MyInt(i32);

    impl Scalar for MyInt {
        type Primitive = i32;
    }

    impl Unit for MyInt {
        type Scalar = MyInt;
    }

    #[test]
    fn custom_scalars() {
        let my_vec: Vector4<MyInt> = (MyInt(1), MyInt(2), MyInt(3), MyInt(4)).into();
        assert_eq!(my_vec[0], MyInt(1));
    }

    #[test]
    fn alignment() {
        use core::mem::align_of;

        assert_eq!(align_of::<glam::Vec2>(), align_of::<Vector2::<f32>>());
        assert_eq!(align_of::<glam::Vec3>(), align_of::<Vector3::<f32>>());
        assert_eq!(align_of::<glam::Vec4>(), align_of::<Vector4::<f32>>());
        assert_eq!(align_of::<glam::DVec2>(), align_of::<Vector2::<f64>>());
        assert_eq!(align_of::<glam::DVec3>(), align_of::<Vector3::<f64>>());
        assert!(align_of::<glam::DVec4>() <= align_of::<Vector4::<f64>>());
        assert_eq!(align_of::<glam::IVec2>(), align_of::<Vector2::<i32>>());
        assert_eq!(align_of::<glam::IVec3>(), align_of::<Vector3::<i32>>());
        assert!(align_of::<glam::IVec4>() <= align_of::<Vector4::<i32>>());
        assert_eq!(align_of::<glam::UVec2>(), align_of::<Vector2::<u32>>());
        assert_eq!(align_of::<glam::UVec3>(), align_of::<Vector3::<u32>>());
        assert!(align_of::<glam::UVec4>() <= align_of::<Vector4::<u32>>());

        assert_eq!(align_of::<glam::Vec2>(), align_of::<Point2::<f32>>());
        assert_eq!(align_of::<glam::Vec3>(), align_of::<Point3::<f32>>());
        assert_eq!(align_of::<glam::Vec4>(), align_of::<Point4::<f32>>());
        assert_eq!(align_of::<glam::DVec2>(), align_of::<Point2::<f64>>());
        assert_eq!(align_of::<glam::DVec3>(), align_of::<Point3::<f64>>());
        assert!(align_of::<glam::DVec4>() <= align_of::<Point4::<f64>>());
        assert_eq!(align_of::<glam::IVec2>(), align_of::<Point2::<i32>>());
        assert_eq!(align_of::<glam::IVec3>(), align_of::<Point3::<i32>>());
        assert!(align_of::<glam::IVec4>() <= align_of::<Point4::<i32>>());
        assert_eq!(align_of::<glam::UVec2>(), align_of::<Point2::<u32>>());
        assert_eq!(align_of::<glam::UVec3>(), align_of::<Point3::<u32>>());
        assert!(align_of::<glam::UVec4>() <= align_of::<Point4::<u32>>());

        assert_eq!(align_of::<glam::Vec2>(), align_of::<Size2::<f32>>());
        assert_eq!(align_of::<glam::Vec3>(), align_of::<Size3::<f32>>());
        assert_eq!(align_of::<glam::DVec2>(), align_of::<Size2::<f64>>());
        assert_eq!(align_of::<glam::DVec3>(), align_of::<Size3::<f64>>());
        assert_eq!(align_of::<glam::IVec2>(), align_of::<Size2::<i32>>());
        assert_eq!(align_of::<glam::IVec3>(), align_of::<Size3::<i32>>());
        assert_eq!(align_of::<glam::UVec2>(), align_of::<Size2::<u32>>());
        assert_eq!(align_of::<glam::UVec3>(), align_of::<Size3::<u32>>());

        assert_eq!(align_of::<glam::Mat2>(), align_of::<Matrix2<f32>>());
        assert_eq!(align_of::<glam::Mat3>(), align_of::<Matrix3<f32>>());
        assert_eq!(align_of::<glam::Mat4>(), align_of::<Matrix4<f32>>());
        assert!(align_of::<glam::DMat2>() <= align_of::<Matrix2<f64>>());
        assert_eq!(align_of::<glam::DMat3>(), align_of::<Matrix3<f64>>());
        assert!(align_of::<glam::DMat4>() <= align_of::<Matrix4<f64>>());

        assert_eq!(align_of::<glam::Mat3>(), align_of::<Transform2<f32, f32>>());
        assert_eq!(align_of::<glam::Mat4>(), align_of::<Transform3<f32, f32>>());
        assert_eq!(
            align_of::<glam::DMat3>(),
            align_of::<Transform2<f64, f64>>()
        );
        assert!(align_of::<glam::DMat4>() <= align_of::<Transform3<f64, f64>>());
    }

    #[test]
    fn size() {
        use core::mem::size_of;

        assert_eq!(size_of::<glam::Vec2>(), size_of::<Vector2::<f32>>());
        assert_eq!(size_of::<glam::Vec3>(), size_of::<Vector3::<f32>>());
        assert_eq!(size_of::<glam::Vec4>(), size_of::<Vector4::<f32>>());
        assert_eq!(size_of::<glam::DVec2>(), size_of::<Vector2::<f64>>());
        assert_eq!(size_of::<glam::DVec3>(), size_of::<Vector3::<f64>>());
        assert_eq!(size_of::<glam::DVec4>(), size_of::<Vector4::<f64>>());
        assert_eq!(size_of::<glam::IVec2>(), size_of::<Vector2::<i32>>());
        assert_eq!(size_of::<glam::IVec3>(), size_of::<Vector3::<i32>>());
        assert_eq!(size_of::<glam::IVec4>(), size_of::<Vector4::<i32>>());
        assert_eq!(size_of::<glam::UVec2>(), size_of::<Vector2::<u32>>());
        assert_eq!(size_of::<glam::UVec3>(), size_of::<Vector3::<u32>>());
        assert_eq!(size_of::<glam::UVec4>(), size_of::<Vector4::<u32>>());

        assert_eq!(size_of::<glam::Vec2>(), size_of::<Point2::<f32>>());
        assert_eq!(size_of::<glam::Vec3>(), size_of::<Point3::<f32>>());
        assert_eq!(size_of::<glam::Vec4>(), size_of::<Point4::<f32>>());
        assert_eq!(size_of::<glam::DVec2>(), size_of::<Point2::<f64>>());
        assert_eq!(size_of::<glam::DVec3>(), size_of::<Point3::<f64>>());
        assert_eq!(size_of::<glam::DVec4>(), size_of::<Point4::<f64>>());
        assert_eq!(size_of::<glam::IVec2>(), size_of::<Point2::<i32>>());
        assert_eq!(size_of::<glam::IVec3>(), size_of::<Point3::<i32>>());
        assert_eq!(size_of::<glam::IVec4>(), size_of::<Point4::<i32>>());
        assert_eq!(size_of::<glam::UVec2>(), size_of::<Point2::<u32>>());
        assert_eq!(size_of::<glam::UVec3>(), size_of::<Point3::<u32>>());
        assert_eq!(size_of::<glam::UVec4>(), size_of::<Point4::<u32>>());

        assert_eq!(size_of::<glam::Vec2>(), size_of::<Size2::<f32>>());
        assert_eq!(size_of::<glam::Vec3>(), size_of::<Size3::<f32>>());
        assert_eq!(size_of::<glam::DVec2>(), size_of::<Size2::<f64>>());
        assert_eq!(size_of::<glam::DVec3>(), size_of::<Size3::<f64>>());
        assert_eq!(size_of::<glam::IVec2>(), size_of::<Size2::<i32>>());
        assert_eq!(size_of::<glam::IVec3>(), size_of::<Size3::<i32>>());
        assert_eq!(size_of::<glam::UVec2>(), size_of::<Size2::<u32>>());
        assert_eq!(size_of::<glam::UVec3>(), size_of::<Size3::<u32>>());

        assert_eq!(size_of::<glam::Mat2>(), size_of::<Matrix2<f32>>());
        assert_eq!(size_of::<glam::Mat3>(), size_of::<Matrix3<f32>>());
        assert_eq!(size_of::<glam::Mat4>(), size_of::<Matrix4<f32>>());
        assert_eq!(size_of::<glam::DMat2>(), size_of::<Matrix2<f64>>());
        assert_eq!(size_of::<glam::DMat3>(), size_of::<Matrix3<f64>>());
        assert_eq!(size_of::<glam::DMat4>(), size_of::<Matrix4<f64>>());

        assert_eq!(size_of::<glam::Mat3>(), size_of::<Transform2<f32, f32>>());
        assert_eq!(size_of::<glam::Mat4>(), size_of::<Transform3<f32, f32>>());
        assert_eq!(size_of::<glam::DMat3>(), size_of::<Transform2<f64, f64>>());
        assert_eq!(size_of::<glam::DMat4>(), size_of::<Transform3<f64, f64>>());
    }

    #[test]
    fn cast_to_glam_by_reference() {
        let vec2 = Vector2::<f32>::new(1.0, 2.0);
        let vec3 = Vector3::<f32>::new(1.0, 2.0, 3.0);
        let vec4 = Vector4::<f32>::new(1.0, 2.0, 3.0, 4.0);
        let dvec2 = Vector2::<f64>::new(1.0, 2.0);
        let dvec3 = Vector3::<f64>::new(1.0, 2.0, 3.0);
        let dvec4 = Vector4::<f64>::new(1.0, 2.0, 3.0, 4.0);
        let ivec2 = Vector2::<i32>::new(1, 2);
        let ivec3 = Vector3::<i32>::new(1, 2, 3);
        let ivec4 = Vector4::<i32>::new(1, 2, 3, 4);
        let uvec2 = Vector2::<u32>::new(1, 2);
        let uvec3 = Vector3::<u32>::new(1, 2, 3);
        let uvec4 = Vector4::<u32>::new(1, 2, 3, 4);

        let point2 = Point2::<f32>::new(1.0, 2.0);
        let point3 = Point3::<f32>::new(1.0, 2.0, 3.0);
        let point4 = Point4::<f32>::new(1.0, 2.0, 3.0, 4.0);
        let dpoint2 = Point2::<f64>::new(1.0, 2.0);
        let dpoint3 = Point3::<f64>::new(1.0, 2.0, 3.0);
        let dpoint4 = Point4::<f64>::new(1.0, 2.0, 3.0, 4.0);
        let ipoint2 = Point2::<i32>::new(1, 2);
        let ipoint3 = Point3::<i32>::new(1, 2, 3);
        let ipoint4 = Point4::<i32>::new(1, 2, 3, 4);
        let upoint2 = Point2::<u32>::new(1, 2);
        let upoint3 = Point3::<u32>::new(1, 2, 3);
        let upoint4 = Point4::<u32>::new(1, 2, 3, 4);

        let vec2_raw: &glam::Vec2 = vec2.as_raw();
        let vec3_raw: &glam::Vec3 = vec3.as_raw();
        let vec4_raw: &glam::Vec4 = vec4.as_raw();
        let dvec2_raw: &glam::DVec2 = dvec2.as_raw();
        let dvec3_raw: &glam::DVec3 = dvec3.as_raw();
        let dvec4_raw: &glam::DVec4 = dvec4.as_raw();
        let ivec2_raw: &glam::IVec2 = ivec2.as_raw();
        let ivec3_raw: &glam::IVec3 = ivec3.as_raw();
        let ivec4_raw: &glam::IVec4 = ivec4.as_raw();
        let uvec2_raw: &glam::UVec2 = uvec2.as_raw();
        let uvec3_raw: &glam::UVec3 = uvec3.as_raw();
        let uvec4_raw: &glam::UVec4 = uvec4.as_raw();

        let point2_raw: &glam::Vec2 = point2.as_raw();
        let point3_raw: &glam::Vec3 = point3.as_raw();
        let point4_raw: &glam::Vec4 = point4.as_raw();
        let dpoint2_raw: &glam::DVec2 = dpoint2.as_raw();
        let dpoint3_raw: &glam::DVec3 = dpoint3.as_raw();
        let dpoint4_raw: &glam::DVec4 = dpoint4.as_raw();
        let ipoint2_raw: &glam::IVec2 = ipoint2.as_raw();
        let ipoint3_raw: &glam::IVec3 = ipoint3.as_raw();
        let ipoint4_raw: &glam::IVec4 = ipoint4.as_raw();
        let upoint2_raw: &glam::UVec2 = upoint2.as_raw();
        let upoint3_raw: &glam::UVec3 = upoint3.as_raw();
        let upoint4_raw: &glam::UVec4 = upoint4.as_raw();

        assert_eq!(vec2_raw.x, vec2.x);
        assert_eq!(vec2_raw.y, vec2.y);
        assert_eq!(vec3_raw.x, vec3.x);
        assert_eq!(vec3_raw.y, vec3.y);
        assert_eq!(vec3_raw.z, vec3.z);
        assert_eq!(vec4_raw.x, vec4.x);
        assert_eq!(vec4_raw.y, vec4.y);
        assert_eq!(vec4_raw.z, vec4.z);
        assert_eq!(vec4_raw.w, vec4.w);
        assert_eq!(dvec2_raw.x, dvec2.x);
        assert_eq!(dvec2_raw.y, dvec2.y);
        assert_eq!(dvec3_raw.z, dvec3.z);
        assert_eq!(dvec3_raw.x, dvec3.x);
        assert_eq!(dvec3_raw.y, dvec3.y);
        assert_eq!(dvec4_raw.x, dvec4.x);
        assert_eq!(dvec4_raw.y, dvec4.y);
        assert_eq!(dvec4_raw.z, dvec4.z);
        assert_eq!(dvec4_raw.w, dvec4.w);
        assert_eq!(ivec2_raw.x, ivec2.x);
        assert_eq!(ivec2_raw.y, ivec2.y);
        assert_eq!(ivec3_raw.x, ivec3.x);
        assert_eq!(ivec3_raw.y, ivec3.y);
        assert_eq!(ivec3_raw.z, ivec3.z);
        assert_eq!(ivec4_raw.x, ivec4.x);
        assert_eq!(ivec4_raw.y, ivec4.y);
        assert_eq!(ivec4_raw.z, ivec4.z);
        assert_eq!(ivec4_raw.w, ivec4.w);
        assert_eq!(uvec2_raw.x, uvec2.x);
        assert_eq!(uvec2_raw.y, uvec2.y);
        assert_eq!(uvec3_raw.x, uvec3.x);
        assert_eq!(uvec3_raw.y, uvec3.y);
        assert_eq!(uvec3_raw.z, uvec3.z);
        assert_eq!(uvec4_raw.x, uvec4.x);
        assert_eq!(uvec4_raw.y, uvec4.y);
        assert_eq!(uvec4_raw.z, uvec4.z);
        assert_eq!(uvec4_raw.w, uvec4.w);
        assert_eq!(point2_raw.x, point2.x);
        assert_eq!(point2_raw.y, point2.y);
        assert_eq!(point3_raw.x, point3.x);
        assert_eq!(point3_raw.y, point3.y);
        assert_eq!(point3_raw.z, point3.z);
        assert_eq!(point4_raw.x, point4.x);
        assert_eq!(point4_raw.y, point4.y);
        assert_eq!(point4_raw.z, point4.z);
        assert_eq!(point4_raw.w, point4.w);
        assert_eq!(dpoint2_raw.x, dpoint2.x);
        assert_eq!(dpoint2_raw.y, dpoint2.y);
        assert_eq!(dpoint3_raw.z, dpoint3.z);
        assert_eq!(dpoint3_raw.x, dpoint3.x);
        assert_eq!(dpoint3_raw.y, dpoint3.y);
        assert_eq!(dpoint4_raw.x, dpoint4.x);
        assert_eq!(dpoint4_raw.y, dpoint4.y);
        assert_eq!(dpoint4_raw.z, dpoint4.z);
        assert_eq!(dpoint4_raw.w, dpoint4.w);
        assert_eq!(ipoint2_raw.x, ipoint2.x);
        assert_eq!(ipoint2_raw.y, ipoint2.y);
        assert_eq!(ipoint3_raw.x, ipoint3.x);
        assert_eq!(ipoint3_raw.y, ipoint3.y);
        assert_eq!(ipoint3_raw.z, ipoint3.z);
        assert_eq!(ipoint4_raw.x, ipoint4.x);
        assert_eq!(ipoint4_raw.y, ipoint4.y);
        assert_eq!(ipoint4_raw.z, ipoint4.z);
        assert_eq!(ipoint4_raw.w, ipoint4.w);
        assert_eq!(upoint2_raw.x, upoint2.x);
        assert_eq!(upoint2_raw.y, upoint2.y);
        assert_eq!(upoint3_raw.x, upoint3.x);
        assert_eq!(upoint3_raw.y, upoint3.y);
        assert_eq!(upoint3_raw.z, upoint3.z);
        assert_eq!(upoint4_raw.x, upoint4.x);
        assert_eq!(upoint4_raw.y, upoint4.y);
        assert_eq!(upoint4_raw.z, upoint4.z);
        assert_eq!(upoint4_raw.w, upoint4.w);
    }

    #[test]
    fn try_cast() {
        let v = Vector4::<f32>::new(core::f32::MAX, 0.0, 1.0, 2.0);
        let t: Option<Vector4<i32>> = v.try_cast();
        assert!(t.is_none());
    }

    #[test]
    fn vec_macros() {
        let a: Vector2<f32> = vec2!(1.0, 2.0);
        let b: Vector2<f32> = vec2![1.0, 2.0];
        let c: Vector2<f32> = vec2!(2.0);
        assert_eq!(a.x, 1.0);
        assert_eq!(a.y, 2.0);
        assert_eq!(b.x, 1.0);
        assert_eq!(b.y, 2.0);
        assert_eq!(c.x, 2.0);
        assert_eq!(c.y, 2.0);

        let a: Vector3<f32> = vec3!(1.0, 2.0, 3.0);
        let b: Vector3<f32> = vec3![1.0, 2.0, 3.0];
        let c: Vector3<f32> = vec3!(2.0);
        assert_eq!(a.x, 1.0);
        assert_eq!(a.y, 2.0);
        assert_eq!(a.z, 3.0);
        assert_eq!(b.x, 1.0);
        assert_eq!(b.y, 2.0);
        assert_eq!(b.z, 3.0);
        assert_eq!(c.x, 2.0);
        assert_eq!(c.y, 2.0);
        assert_eq!(c.z, 2.0);

        let a: Vector4<f32> = vec4!(1.0, 2.0, 3.0, 4.0);
        let b: Vector4<f32> = vec4![1.0, 2.0, 3.0, 4.0];
        let c: Vector4<f32> = vec4!(2.0);
        assert_eq!(a.x, 1.0);
        assert_eq!(a.y, 2.0);
        assert_eq!(a.z, 3.0);
        assert_eq!(a.w, 4.0);
        assert_eq!(b.x, 1.0);
        assert_eq!(b.y, 2.0);
        assert_eq!(b.z, 3.0);
        assert_eq!(b.w, 4.0);
        assert_eq!(c.x, 2.0);
        assert_eq!(c.y, 2.0);
        assert_eq!(c.z, 2.0);
        assert_eq!(c.w, 2.0);

        let a: Vector2<f32> = vector!(1.0, 2.0);
        let b: Vector3<f32> = vector!(1.0, 2.0, 3.0);
        let c: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        assert_eq!(a.x, 1.0);
        assert_eq!(a.y, 2.0);
        assert_eq!(b.x, 1.0);
        assert_eq!(b.y, 2.0);
        assert_eq!(b.z, 3.0);
        assert_eq!(c.x, 1.0);
        assert_eq!(c.y, 2.0);
        assert_eq!(c.z, 3.0);
        assert_eq!(c.w, 4.0);

        let a: Vector2<f32> = vector!([2.0; 2]);
        let b: Vector3<f32> = vector!([3.0; 3]);
        let c: Vector4<f32> = vector!([4.0; 4]);
        assert_eq!(a.x, 2.0);
        assert_eq!(a.y, 2.0);
        assert_eq!(b.x, 3.0);
        assert_eq!(b.y, 3.0);
        assert_eq!(b.z, 3.0);
        assert_eq!(c.x, 4.0);
        assert_eq!(c.y, 4.0);
        assert_eq!(c.z, 4.0);
        assert_eq!(c.w, 4.0);
    }

    #[test]
    fn point_macros() {
        let a: Point2<f32> = point2!(1.0, 2.0);
        let b: Point2<f32> = point2![1.0, 2.0];
        let c: Point2<f32> = point2!(2.0);
        assert_eq!(a.x, 1.0);
        assert_eq!(a.y, 2.0);
        assert_eq!(b.x, 1.0);
        assert_eq!(b.y, 2.0);
        assert_eq!(c.x, 2.0);
        assert_eq!(c.y, 2.0);

        let a: Point3<f32> = point3!(1.0, 2.0, 3.0);
        let b: Point3<f32> = point3![1.0, 2.0, 3.0];
        let c: Point3<f32> = point3!(2.0);
        assert_eq!(a.x, 1.0);
        assert_eq!(a.y, 2.0);
        assert_eq!(a.z, 3.0);
        assert_eq!(b.x, 1.0);
        assert_eq!(b.y, 2.0);
        assert_eq!(b.z, 3.0);
        assert_eq!(c.x, 2.0);
        assert_eq!(c.y, 2.0);
        assert_eq!(c.z, 2.0);

        let a: Point4<f32> = point4!(1.0, 2.0, 3.0, 4.0);
        let b: Point4<f32> = point4![1.0, 2.0, 3.0, 4.0];
        let c: Point4<f32> = point4!(2.0);
        assert_eq!(a.x, 1.0);
        assert_eq!(a.y, 2.0);
        assert_eq!(a.z, 3.0);
        assert_eq!(a.w, 4.0);
        assert_eq!(b.x, 1.0);
        assert_eq!(b.y, 2.0);
        assert_eq!(b.z, 3.0);
        assert_eq!(b.w, 4.0);
        assert_eq!(c.x, 2.0);
        assert_eq!(c.y, 2.0);
        assert_eq!(c.z, 2.0);
        assert_eq!(c.w, 2.0);

        let a: Point2<f32> = point!(1.0, 2.0);
        let b: Point3<f32> = point!(1.0, 2.0, 3.0);
        let c: Point4<f32> = point!(1.0, 2.0, 3.0, 4.0);
        assert_eq!(a.x, 1.0);
        assert_eq!(a.y, 2.0);
        assert_eq!(b.x, 1.0);
        assert_eq!(b.y, 2.0);
        assert_eq!(b.z, 3.0);
        assert_eq!(c.x, 1.0);
        assert_eq!(c.y, 2.0);
        assert_eq!(c.z, 3.0);
        assert_eq!(c.w, 4.0);

        let a: Point2<f32> = point!([2.0; 2]);
        let b: Point3<f32> = point!([3.0; 3]);
        let c: Point4<f32> = point!([4.0; 4]);
        assert_eq!(a.x, 2.0);
        assert_eq!(a.y, 2.0);
        assert_eq!(b.x, 3.0);
        assert_eq!(b.y, 3.0);
        assert_eq!(b.z, 3.0);
        assert_eq!(c.x, 4.0);
        assert_eq!(c.y, 4.0);
        assert_eq!(c.z, 4.0);
        assert_eq!(c.w, 4.0);
    }

    #[test]
    fn size_macros() {
        let a: Size2<f32> = size2!(1.0, 2.0);
        let b: Size2<f32> = size2![1.0, 2.0];
        let c: Size2<f32> = size2!(2.0);
        assert_eq!(a.width, 1.0);
        assert_eq!(a.height, 2.0);
        assert_eq!(b.width, 1.0);
        assert_eq!(b.height, 2.0);
        assert_eq!(c.width, 2.0);
        assert_eq!(c.height, 2.0);

        let a: Size3<f32> = size3!(1.0, 2.0, 3.0);
        let b: Size3<f32> = size3![1.0, 2.0, 3.0];
        let c: Size3<f32> = size3!(2.0);
        assert_eq!(a.width, 1.0);
        assert_eq!(a.height, 2.0);
        assert_eq!(a.depth, 3.0);
        assert_eq!(b.width, 1.0);
        assert_eq!(b.height, 2.0);
        assert_eq!(b.depth, 3.0);
        assert_eq!(c.width, 2.0);
        assert_eq!(c.height, 2.0);
        assert_eq!(c.depth, 2.0);

        let a: Size2<f32> = size!(1.0, 2.0);
        let b: Size3<f32> = size!(1.0, 2.0, 3.0);
        assert_eq!(a.width, 1.0);
        assert_eq!(a.height, 2.0);
        assert_eq!(b.width, 1.0);
        assert_eq!(b.height, 2.0);
        assert_eq!(b.depth, 3.0);

        let a: Size2<f32> = size!([2.0; 2]);
        let b: Size3<f32> = size!([3.0; 3]);
        assert_eq!(a.width, 2.0);
        assert_eq!(a.height, 2.0);
        assert_eq!(b.width, 3.0);
        assert_eq!(b.height, 3.0);
        assert_eq!(b.depth, 3.0);
    }

    // Verify that the macros can be used in a const context.
    const _CONST_TEST_VEC: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
    const _CONST_TEST_POINT: Point4<f32> = point!(1.0, 2.0, 3.0, 4.0);
    const _CONST_TEST_SIZE: Size3<f32> = size!(1.0, 2.0, 3.0);
}

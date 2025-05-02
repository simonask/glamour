#![allow(unused)]

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use glamour::{Unit, prelude::*, traits::marker::PodValue};

//
// A whole bunch of traits that exist to check if a type implements its
// supertraits.
//
// This is so we can statically check that the dark trait magic actually
// results in the correct traits being implemented for our types.
//

trait AssertValueLike: PodValue {}

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
{
}

trait AssertEq: Eq {}

trait AssertFloatVectorLike<T>: AssertVectorLike<T> + AbsDiffEq + RelativeEq + UlpsEq {}

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

impl AssertMatrixLike<Vector2<f32>> for Matrix2<f32> {}
impl AssertMatrixLike<Vector3<f32>> for Matrix3<f32> {}
impl AssertMatrixLike<Vector4<f32>> for Matrix4<f32> {}
impl AssertMatrixLike<Vector2<f64>> for Matrix2<f64> {}
impl AssertMatrixLike<Vector3<f64>> for Matrix3<f64> {}
impl AssertMatrixLike<Vector4<f64>> for Matrix4<f64> {}

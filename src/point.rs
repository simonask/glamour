//! Point types.
//!
//! Points are vectors with different and more limited semantics.
//!
//! For example, transforming a 3D point through a 4 x 4 matrix is different
//! from transforming a 3D vector through the same matrix.
//!
//! Arithmetic operators with points are also typed differently: Subtracting two
//! points yields a vector, while subtracting a vector from a point yields
//! another point.

use bytemuck::{Pod, Zeroable};
use num_traits::{ConstOne, ConstZero};

use crate::{
    bindings::prelude::*, peel, rewrap, scalar::FloatScalar, unit::FloatUnit, wrap, Scalar,
    Transparent, Unit, Vector2, Vector3, Vector4,
};
use core::ops::Mul;

/// 2D point.
///
/// Alignment: Same as the scalar.
#[repr(C)]
pub struct Point2<T: Unit = f32> {
    /// X coordinate
    pub x: T::Scalar,
    /// Y coordinate
    pub y: T::Scalar,
}

// SAFETY: `T::Scalar` is `Zeroable`, and `Point2` is `#[repr(C)]`.
unsafe impl<T: Unit> Zeroable for Point2<T> {}
// SAFETY: `T::Scalar` is `Pod`.
unsafe impl<T: Unit> Pod for Point2<T> {}
// SAFETY: This is the fundamental guarantee of this crate.
unsafe impl<T: Unit> Transparent for Point2<T> {
    type Wrapped = <T::Scalar as Scalar>::Vec2;
}

/// 3D point.
///
/// Alignment: Same as the scalar (so not 16 bytes). If you really need 16-byte
/// alignment, use [`Point4`].
#[repr(C)]
pub struct Point3<T: Unit = f32> {
    /// X coordinate
    pub x: T::Scalar,
    /// Y coordinate
    pub y: T::Scalar,
    /// Z coordinate
    pub z: T::Scalar,
}

/// SAFETY: `T::Scalar` is `Zeroable`, and `Point3` is `#[repr(C)]`.
unsafe impl<T: Unit> Zeroable for Point3<T> {}
/// SAFETY: `T::Scalar` is `Pod`.
unsafe impl<T: Unit> Pod for Point3<T> {}
unsafe impl<T: Unit> Transparent for Point3<T> {
    type Wrapped = <T::Scalar as Scalar>::Vec3;
}

/// 4D point.
///
/// Alignment: This is always 16-byte aligned. [`glam::DVec4`] is only 8-byte
/// aligned (for some reason), and integer vectors are only 4-byte aligned,
/// which means that reference-casting from those glam types to `Point4` type
/// will fail (but not the other way around).
#[cfg_attr(
    any(
        not(any(feature = "scalar-math", target_arch = "spirv")),
        feature = "cuda"
    ),
    repr(C, align(16))
)]
pub struct Point4<T: Unit = f32> {
    /// X coordinate
    pub x: T::Scalar,
    /// Y coordinate
    pub y: T::Scalar,
    /// Z coordinate
    pub z: T::Scalar,
    /// W coordinate
    pub w: T::Scalar,
}

/// SAFETY: `T::Scalar` is `Zeroable`, and `Point4` is `#[repr(C)]`.
unsafe impl<T: Unit> Zeroable for Point4<T> {}
/// SAFETY: `T::Scalar` is `Pod`.
unsafe impl<T: Unit> Pod for Point4<T> {}
unsafe impl<T: Unit> Transparent for Point4<T> {
    type Wrapped = <T::Scalar as Scalar>::Vec4;
}

macro_rules! point_interface {
    ($base_type_name:ident, $vector_type:ident) => {
        /// Convert a vector to a point as-is.
        #[inline]
        pub fn from_vector(vec: $vector_type<T>) -> Self {
            vec.into()
        }

        /// Cast this point as-is to a vector.
        #[inline]
        pub fn to_vector(self) -> $vector_type<T> {
            self.into()
        }

        #[doc = "Reinterpret as vector."]
        #[inline]
        pub fn as_vector(&self) -> &$vector_type<T> {
            bytemuck::cast_ref(self)
        }

        #[doc = "Reinterpret as vector."]
        #[inline]
        pub fn as_vector_mut(&mut self) -> &mut $vector_type<T> {
            bytemuck::cast_mut(self)
        }

        /// Translate this point by vector.
        ///
        /// Equivalent to `self + by`.
        #[inline]
        #[must_use]
        pub fn translate(self, by: $vector_type<T>) -> Self {
            self + by
        }
    };
}

crate::impl_ops::point_ops!(Point2, Vector2);
crate::impl_ops::point_ops!(Point3, Vector3);
crate::impl_ops::point_ops!(Point4, Vector4);

impl<T: Unit> Point2<T> {
    /// New point.
    pub const fn new(x: T::Scalar, y: T::Scalar) -> Self {
        Point2 { x, y }
    }

    point_interface!(Point2, Vector2);
}

crate::impl_vectorlike::pointlike!(Point2, 2);

impl<T: Unit> Point3<T> {
    /// New point.
    pub const fn new(x: T::Scalar, y: T::Scalar, z: T::Scalar) -> Self {
        Point3 { x, y, z }
    }

    point_interface!(Point3, Vector3);
}

crate::impl_vectorlike::pointlike!(Point3, 3);

impl<T: Unit> Point4<T> {
    /// New point.
    pub const fn new(x: T::Scalar, y: T::Scalar, z: T::Scalar, w: T::Scalar) -> Self {
        Point4 { x, y, z, w }
    }

    point_interface!(Point4, Vector4);
}

crate::impl_vectorlike::pointlike!(Point4, 4);

impl<T> Point3<T>
where
    T: Unit<Scalar = f32>,
{
    /// Create from SIMD-aligned [`glam::Vec3A`].
    ///
    /// See [the design limitations](crate::docs::design#vector-overalignment)
    /// for why this is needed.
    #[inline]
    #[must_use]
    pub fn from_vec3a(vec: glam::Vec3A) -> Self {
        vec.into()
    }

    /// Convert to SIMD-aligned [`glam::Vec3A`].
    ///
    /// See [the design limitations](crate::docs::design#vector-overalignment)
    /// for why this is needed.
    #[inline]
    #[must_use]
    pub fn to_vec3a(self) -> glam::Vec3A {
        self.into()
    }
}

impl<T> Mul<Point3<T>> for glam::Quat
where
    T: Unit<Scalar = f32>,
{
    type Output = Point3<T>;

    fn mul(self, rhs: Point3<T>) -> Self::Output {
        wrap(self * peel(rhs))
    }
}

impl<T> Mul<Point3<T>> for glam::DQuat
where
    T: Unit<Scalar = f64>,
{
    type Output = Point3<T>;

    fn mul(self, rhs: Point3<T>) -> Self::Output {
        wrap(self * peel(rhs))
    }
}

impl<T: Unit> From<Vector2<T>> for Point2<T> {
    fn from(vec: Vector2<T>) -> Point2<T> {
        rewrap(vec)
    }
}

impl<T: Unit> From<Vector3<T>> for Point3<T> {
    #[inline]
    fn from(vec: Vector3<T>) -> Point3<T> {
        rewrap(vec)
    }
}

impl<T: Unit> From<Vector4<T>> for Point4<T> {
    #[inline]
    fn from(vec: Vector4<T>) -> Point4<T> {
        rewrap(vec)
    }
}

impl<T: Unit> From<Point2<T>> for Vector2<T> {
    #[inline]
    fn from(point: Point2<T>) -> Vector2<T> {
        rewrap(point)
    }
}

impl<T: Unit> From<Point3<T>> for Vector3<T> {
    #[inline]
    fn from(point: Point3<T>) -> Vector3<T> {
        rewrap(point)
    }
}

impl<T: Unit> From<Point4<T>> for Vector4<T> {
    #[inline]
    fn from(point: Point4<T>) -> Vector4<T> {
        rewrap(point)
    }
}

impl<T: Unit> core::fmt::Debug for Point2<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("Point2")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}
impl<T: Unit> core::fmt::Debug for Point3<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("Point3")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .finish()
    }
}
impl<T: Unit> core::fmt::Debug for Point4<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("Point4")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .field("w", &self.w)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::point;

    use super::*;

    type Point = super::Point3<f32>;

    #[test]
    fn subtraction_yields_vector() {
        let p = Point::ONE;
        let q = Point::ONE;
        let v: Vector3<f32> = q - p;
        assert_eq!(v, Vector3::<f32>::ZERO);
    }

    #[test]
    fn not_scalable() {
        let p = Point::default();

        // This should not compile:
        // let q = p * 2.0;

        let _ = p;
    }

    #[test]
    fn vec3a() {
        let a: glam::Vec3A = Point::new(0.0, 1.0, 2.0).to_vec3a();
        assert_eq!(a, glam::Vec3A::new(0.0, 1.0, 2.0));
        let b = Point::from_vec3a(a);
        assert_eq!(b, Point::new(0.0, 1.0, 2.0));
    }

    #[test]
    fn rotate() {
        use crate::Angle;
        use approx::assert_abs_diff_eq;

        let v = Point3::<f32>::new(1.0, 0.0, 0.0);
        let quat = Angle::from_degrees(180.0f32).to_rotation(Vector3::Z);
        assert_abs_diff_eq!(quat * v, -v);

        let v = Point3::<f64>::new(1.0, 0.0, 0.0);
        let quat = Angle::from_degrees(180.0f64).to_rotation(Vector3::Z);
        assert_abs_diff_eq!(quat * v, -v);
    }

    #[test]
    fn from_into_vector2() {
        let mut p: Point2<f32> = point!(1.0, 2.0);
        let mut v: Vector2<f32> = p.to_vector();
        let q: Point2<f32> = Point2::from_vector(v);
        assert_eq!(p, q);
        assert_eq!(Vector2::from_point(p), v);

        let _: &Vector2<_> = p.as_vector();
        let _: &mut Vector2<_> = p.as_vector_mut();
        let _: &Point2<_> = v.as_point();
        let _: &mut Point2<_> = v.as_point_mut();
    }

    #[test]
    fn from_into_vector3() {
        let mut p: Point3<f32> = point!(1.0, 2.0, 3.0);
        let mut v: Vector3<f32> = p.to_vector();
        let q: Point3<f32> = Point3::from_vector(v);
        assert_eq!(p, q);
        assert_eq!(Vector3::from_point(p), v);

        let _: &Vector3<_> = p.as_vector();
        let _: &mut Vector3<_> = p.as_vector_mut();
        let _: &Point3<_> = v.as_point();
        let _: &mut Point3<_> = v.as_point_mut();
    }

    #[test]
    fn from_into_vector4() {
        let mut p: Point4<f32> = point!(1.0, 2.0, 3.0, 4.0);
        let mut v: Vector4<f32> = p.to_vector();
        let q: Point4<f32> = Point4::from_vector(v);
        assert_eq!(p, q);
        assert_eq!(Vector4::from_point(p), v);

        let _: &Vector4<_> = p.as_vector();
        let _: &mut Vector4<_> = p.as_vector_mut();
        let _: &Point4<_> = v.as_point();
        let _: &mut Point4<_> = v.as_point_mut();
    }

    #[test]
    fn extend() {
        let p2: Point2<f32> = point!(1.0, 2.0);
        let p3: Point3<f32> = p2.extend(3.0);
        assert_eq!(p3, point!(1.0, 2.0, 3.0));
        let p4: Point4<f32> = p3.extend(4.0);
        assert_eq!(p4, point!(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn gaslight_coverage() {
        extern crate alloc;
        _ = alloc::format!("{:?}", Point2::<f32>::default());
        _ = alloc::format!("{:?}", Point3::<f32>::default());
        _ = alloc::format!("{:?}", Point4::<f32>::default());
    }
}

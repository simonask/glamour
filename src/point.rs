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

use crate::{traits::Lerp, Scalar, Unit, UnitTypes, Vector2, Vector3, Vector4};
use core::ops::{Add, AddAssign, Mul, Sub, SubAssign};

/// 2D point.
///
/// Alignment: Same as the scalar.
#[repr(C)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
pub struct Point2<T: Unit = f32> {
    /// X coordinate
    pub x: T::Scalar,
    /// Y coordinate
    pub y: T::Scalar,
}

/// 3D point.
///
/// Alignment: Same as the scalar (so not 16 bytes). If you really need 16-byte
/// alignment, use [`Point4`].
#[repr(C)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
pub struct Point3<T: Unit = f32> {
    /// X coordinate
    pub x: T::Scalar,
    /// Y coordinate
    pub y: T::Scalar,
    /// Z coordinate
    pub z: T::Scalar,
}

/// 4D point.
///
/// Alignment: This is always 16-byte aligned. [`glam::DVec4`] is only 8-byte
/// aligned (for some reason), and integer vectors are only 4-byte aligned,
/// which means that reference-casting from those glam types to `Point4` type
/// will fail (but not the other way around).
#[repr(C, align(16))]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
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

crate::impl_common!(Point2 {
    x: T::Scalar,
    y: T::Scalar
});
crate::impl_common!(Point3 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar
});
crate::impl_common!(Point4 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar,
    w: T::Scalar
});

crate::impl_simd_common!(Point2 [2] => Vec2, glam::BVec2 { x, y });
crate::impl_simd_common!(Point3 [3] => Vec3, glam::BVec3 { x, y, z });
crate::impl_simd_common!(Point4 [4] => Vec4, glam::BVec4 { x, y, z, w });

crate::impl_as_tuple!(Point2 {
    x: T::Scalar,
    y: T::Scalar
});
crate::impl_as_tuple!(Point3 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar
});
crate::impl_as_tuple!(Point4 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar,
    w: T::Scalar
});

crate::impl_glam_conversion!(Point2, 2 [f32 => glam::Vec2, f64 => glam::DVec2, i32 => glam::IVec2, u32 => glam::UVec2]);
crate::impl_glam_conversion!(Point3, 3 [f32 => glam::Vec3, f64 => glam::DVec3, i32 => glam::IVec3, u32 => glam::UVec3]);
crate::impl_glam_conversion!(Point4, 4 [f32 => glam::Vec4, f64 => glam::DVec4, i32 => glam::IVec4, u32 => glam::UVec4]);

macro_rules! impl_point {
    ($base_type_name:ident [$dimensions:literal] => $vec_ty:ident, $vector_type:ident) => {
        impl<T: Unit> Sub for $base_type_name<T> {
            type Output = $vector_type<T>;

            #[inline]
            fn sub(self, other: Self) -> Self::Output {
                $vector_type::<T>::from_raw(self.to_raw() - other.to_raw())
            }
        }

        impl<T: Unit> Add<$vector_type<T>> for $base_type_name<T> {
            type Output = Self;

            #[inline]
            fn add(self, other: $vector_type<T>) -> Self {
                Self::from_raw(self.to_raw() + other.to_raw())
            }
        }

        impl<T: Unit> Sub<$vector_type<T>> for $base_type_name<T> {
            type Output = Self;

            #[inline]
            fn sub(self, other: $vector_type<T>) -> Self {
                Self::from_raw(self.to_raw() - other.to_raw())
            }
        }

        impl<T: Unit> AddAssign<$vector_type<T>> for $base_type_name<T> {
            #[inline]
            fn add_assign(&mut self, vector: $vector_type<T>) {
                *self.as_raw_mut() += vector.to_raw();
            }
        }

        impl<T: Unit> SubAssign<$vector_type<T>> for $base_type_name<T> {
            #[inline]
            fn sub_assign(&mut self, vector: $vector_type<T>) {
                *self.as_raw_mut() -= vector.to_raw();
            }
        }

        impl<T: Unit> From<$vector_type<T>> for $base_type_name<T> {
            #[inline]
            fn from(vec: $vector_type<T>) -> Self {
                Self::from_raw(vec.to_raw())
            }
        }

        impl<T: Unit> From<$base_type_name<T>> for $vector_type<T> {
            #[inline]
            fn from(point: $base_type_name<T>) -> Self {
                Self::from_raw(point.to_raw())
            }
        }

        impl<T: Unit> $base_type_name<T> {
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
        }

        impl<T> Lerp<T::Primitive> for $base_type_name<T>
        where
            T: crate::UnitTypes,
            T::$vec_ty: Lerp<T::Primitive>,
        {
            #[inline]
            fn lerp(self, end: Self, t: T::Primitive) -> Self {
                Self::from_raw(self.to_raw().lerp(end.to_raw(), t))
            }
        }
    };
}

impl_point!(Point2 [2] => Vec2, Vector2);
impl_point!(Point3 [3] => Vec3, Vector3);
impl_point!(Point4 [4] => Vec4, Vector4);

crate::impl_mint!(Point2, 2, Point2);
crate::impl_mint!(Point3, 3, Point3);

impl<T> Point3<T>
where
    T: Unit,
    T::Scalar: Scalar<Primitive = f32>,
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

impl<T> From<glam::Vec3A> for Point3<T>
where
    T: Unit,
    T::Scalar: Scalar<Primitive = f32>,
{
    fn from(v: glam::Vec3A) -> Self {
        Self::from_raw(v.into())
    }
}

impl<T> From<Point3<T>> for glam::Vec3A
where
    T: Unit,
    T::Scalar: Scalar<Primitive = f32>,
{
    fn from(v: Point3<T>) -> Self {
        v.to_raw().into()
    }
}

impl<T: UnitTypes<Vec3 = glam::Vec3>> Mul<Point3<T>> for glam::Quat {
    type Output = Point3<T>;

    fn mul(self, rhs: Point3<T>) -> Self::Output {
        Point3::from_raw(self * rhs.to_raw())
    }
}

impl<T: UnitTypes<Vec3 = glam::DVec3>> Mul<Point3<T>> for glam::DQuat {
    type Output = Point3<T>;

    fn mul(self, rhs: Point3<T>) -> Self::Output {
        Point3::from_raw(self * rhs.to_raw())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Point = super::Point3<f32>;

    #[test]
    fn subtraction_yields_vector() {
        let p = Point::one();
        let q = Point::one();
        let v: Vector3<f32> = q - p;
        assert_eq!(v, Vector3::zero());
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
        let quat = Angle::from_degrees(180.0f32).to_rotation(Vector3::unit_z());
        assert_abs_diff_eq!(quat * v, -v);

        let v = Point3::<f64>::new(1.0, 0.0, 0.0);
        let quat = Angle::from_degrees(180.0f64).to_rotation(Vector3::unit_z());
        assert_abs_diff_eq!(quat * v, -v);
    }
}

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

use crate::{
    bindings::prelude::*, scalar::FloatScalar, AsRaw, FromRawRef, Scalar, ToRaw, Unit, Vector2,
    Vector3, Vector4,
};
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
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
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

impl<T: Unit> ToRaw for Point2<T> {
    type Raw = <T::Scalar as Scalar>::Vec2;

    fn to_raw(self) -> Self::Raw {
        bytemuck::cast(self)
    }

    fn from_raw(raw: Self::Raw) -> Self {
        bytemuck::cast(raw)
    }
}

impl<T: Unit> AsRaw for Point2<T> {
    fn as_raw(&self) -> &Self::Raw {
        bytemuck::cast_ref(self)
    }

    fn as_raw_mut(&mut self) -> &mut Self::Raw {
        bytemuck::cast_mut(self)
    }
}

impl<T: Unit> FromRawRef for Point2<T> {
    /// By-ref conversion from `Self::Raw`.
    fn from_raw_ref(raw: &Self::Raw) -> &Self {
        bytemuck::cast_ref(raw)
    }

    /// By-ref mutable conversion from `Self::Raw`.
    fn from_raw_mut(raw: &mut Self::Raw) -> &mut Self {
        bytemuck::cast_mut(raw)
    }
}

impl<T: Unit> ToRaw for Point3<T> {
    type Raw = <T::Scalar as Scalar>::Vec3;

    fn to_raw(self) -> Self::Raw {
        bytemuck::cast(self)
    }

    fn from_raw(raw: Self::Raw) -> Self {
        bytemuck::cast(raw)
    }
}

impl<T: Unit> AsRaw for Point3<T> {
    fn as_raw(&self) -> &Self::Raw {
        bytemuck::cast_ref(self)
    }

    fn as_raw_mut(&mut self) -> &mut Self::Raw {
        bytemuck::cast_mut(self)
    }
}

impl<T: Unit> FromRawRef for Point3<T> {
    /// By-ref conversion from `Self::Raw`.
    fn from_raw_ref(raw: &Self::Raw) -> &Self {
        bytemuck::cast_ref(raw)
    }

    /// By-ref mutable conversion from `Self::Raw`.
    fn from_raw_mut(raw: &mut Self::Raw) -> &mut Self {
        bytemuck::cast_mut(raw)
    }
}

impl<T: Unit> ToRaw for Point4<T> {
    type Raw = <T::Scalar as Scalar>::Vec4;

    fn to_raw(self) -> Self::Raw {
        bytemuck::cast(self)
    }

    fn from_raw(raw: Self::Raw) -> Self {
        bytemuck::cast(raw)
    }
}

impl<T: Unit> AsRaw for Point4<T> {
    fn as_raw(&self) -> &Self::Raw {
        bytemuck::cast_ref(self)
    }

    fn as_raw_mut(&mut self) -> &mut Self::Raw {
        bytemuck::cast_mut(self)
    }
}

impl<T: Unit> FromRawRef for Point4<T> {
    /// By-ref conversion from `Self::Raw`.
    fn from_raw_ref(raw: &Self::Raw) -> &Self {
        bytemuck::cast_ref(raw)
    }

    /// By-ref mutable conversion from `Self::Raw`.
    fn from_raw_mut(raw: &mut Self::Raw) -> &mut Self {
        bytemuck::cast_mut(raw)
    }
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

crate::impl_vector_common!(Point2 [2] => Vec2 { x, y });
crate::impl_vector_common!(Point3 [3] => Vec3 { x, y, z });
crate::impl_vector_common!(Point4 [4] => Vec4 { x, y, z, w });

impl<T: Unit> Point2<T> {
    /// All zeroes.
    pub const ZERO: Self = Self {
        x: T::Scalar::ZERO,
        y: T::Scalar::ZERO,
    };

    /// All ones.
    pub const ONE: Self = Self {
        x: T::Scalar::ONE,
        y: T::Scalar::ONE,
    };

    /// New point.
    pub const fn new(x: T::Scalar, y: T::Scalar) -> Self {
        Point2 { x, y }
    }

    crate::forward_all_to_raw!(
        #[doc = "Instantiate from array."]
        pub fn from_array(array: [T::Scalar; 2]) -> Self;
        #[doc = "Convert to array."]
        pub fn to_array(self) -> [T::Scalar; 2];
        #[doc = "Instance with all components set to `scalar`."]
        pub fn splat(scalar: T::Scalar) -> Self;
        #[doc = "Return a mask with the result of a component-wise equals comparison."]
        pub fn cmpeq(self, other: Self) -> glam::BVec2;
        #[doc = "Return a mask with the result of a component-wise not-equal comparison."]
        pub fn cmpne(self, other: Self) -> glam::BVec2;
        #[doc = "Return a mask with the result of a component-wise greater-than-or-equal comparison."]
        pub fn cmpge(self, other: Self) -> glam::BVec2;
        #[doc = "Return a mask with the result of a component-wise greater-than comparison."]
        pub fn cmpgt(self, other: Self) -> glam::BVec2;
        #[doc = "Return a mask with the result of a component-wise less-than-or-equal comparison."]
        pub fn cmple(self, other: Self) -> glam::BVec2;
        #[doc = "Return a mask with the result of a component-wise less-than comparison."]
        pub fn cmplt(self, other: Self) -> glam::BVec2;
        #[doc = "Minimum by component."]
        pub fn min(self, other: Self) -> Self;
        #[doc = "Maximum by component."]
        pub fn max(self, other: Self) -> Self;
        #[doc = "Horizontal minimum (smallest component)."]
        pub fn min_element(self) -> T::Scalar;
        #[doc = "Horizontal maximum (largest component)."]
        pub fn max_element(self) -> T::Scalar;
        #[doc = "Component-wise clamp."]
        pub fn clamp(self, min: Self, max: Self) -> Self;
    );
}

impl<T> Point2<T>
where
    T: Unit,
    T::Scalar: FloatScalar,
{
    crate::forward_all_to_raw!(
        #[doc = "True if all components are non-infinity and non-NaN."]
        pub fn is_finite(&self) -> bool;
        #[doc = "True if any component is NaN."]
        pub fn is_nan(&self) -> bool;
        #[doc = "Return a mask where each bit is set if the corresponding component is NaN."]
        pub fn is_nan_mask(&self) -> glam::BVec2;
        #[doc = "Round all components up."]
        pub fn ceil(self) -> Self;
        #[doc = "Round all components down."]
        pub fn floor(self) -> Self;
        #[doc = "Round all components."]
        pub fn round(self) -> Self;
        #[doc = "Linear interpolation."]
        pub fn lerp(self, other: Self, t: T::Scalar) -> Self;
    );
}

impl<T: Unit> Point3<T> {
    /// All zeroes.
    pub const ZERO: Self = Self {
        x: T::Scalar::ZERO,
        y: T::Scalar::ZERO,
        z: T::Scalar::ZERO,
    };

    /// All ones.
    pub const ONE: Self = Self {
        x: T::Scalar::ONE,
        y: T::Scalar::ONE,
        z: T::Scalar::ONE,
    };

    /// New point.
    pub fn new(x: T::Scalar, y: T::Scalar, z: T::Scalar) -> Self {
        Point3 { x, y, z }
    }

    crate::forward_all_to_raw!(
        #[doc = "Instantiate from array."]
        pub fn from_array(array: [T::Scalar; 3]) -> Self;
        #[doc = "Convert to array."]
        pub fn to_array(self) -> [T::Scalar; 3];
        #[doc = "Instance with all components set to `scalar`."]
        pub fn splat(scalar: T::Scalar) -> Self;
        #[doc = "Return a mask with the result of a component-wise equals comparison."]
        pub fn cmpeq(self, other: Self) -> glam::BVec3;
        #[doc = "Return a mask with the result of a component-wise not-equal comparison."]
        pub fn cmpne(self, other: Self) -> glam::BVec3;
        #[doc = "Return a mask with the result of a component-wise greater-than-or-equal comparison."]
        pub fn cmpge(self, other: Self) -> glam::BVec3;
        #[doc = "Return a mask with the result of a component-wise greater-than comparison."]
        pub fn cmpgt(self, other: Self) -> glam::BVec3;
        #[doc = "Return a mask with the result of a component-wise less-than-or-equal comparison."]
        pub fn cmple(self, other: Self) -> glam::BVec3;
        #[doc = "Return a mask with the result of a component-wise less-than comparison."]
        pub fn cmplt(self, other: Self) -> glam::BVec3;
        #[doc = "Minimum by component."]
        pub fn min(self, other: Self) -> Self;
        #[doc = "Maximum by component."]
        pub fn max(self, other: Self) -> Self;
        #[doc = "Horizontal minimum (smallest component)."]
        pub fn min_element(self) -> T::Scalar;
        #[doc = "Horizontal maximum (largest component)."]
        pub fn max_element(self) -> T::Scalar;
        #[doc = "Component-wise clamp."]
        pub fn clamp(self, min: Self, max: Self) -> Self;
    );
}

impl<T> Point3<T>
where
    T: Unit,
    T::Scalar: FloatScalar,
{
    crate::forward_all_to_raw!(
        #[doc = "True if all components are non-infinity and non-NaN."]
        pub fn is_finite(&self) -> bool;
        #[doc = "True if any component is NaN."]
        pub fn is_nan(&self) -> bool;
        #[doc = "Return a mask where each bit is set if the corresponding component is NaN."]
        pub fn is_nan_mask(&self) -> glam::BVec3;
        #[doc = "Round all components up."]
        pub fn ceil(self) -> Self;
        #[doc = "Round all components down."]
        pub fn floor(self) -> Self;
        #[doc = "Round all components."]
        pub fn round(self) -> Self;
        #[doc = "Linear interpolation."]
        pub fn lerp(self, other: Self, t: T::Scalar) -> Self;
    );
}

impl<T: Unit> Point4<T> {
    /// All zeroes.
    pub const ZERO: Self = Self {
        x: T::Scalar::ZERO,
        y: T::Scalar::ZERO,
        z: T::Scalar::ZERO,
        w: T::Scalar::ZERO,
    };

    /// All ones.
    pub const ONE: Self = Self {
        x: T::Scalar::ONE,
        y: T::Scalar::ONE,
        z: T::Scalar::ONE,
        w: T::Scalar::ONE,
    };

    /// New point.
    pub fn new(x: T::Scalar, y: T::Scalar, z: T::Scalar, w: T::Scalar) -> Self {
        Point4 { x, y, z, w }
    }

    crate::forward_all_to_raw!(
        #[doc = "Instantiate from array."]
        pub fn from_array(array: [T::Scalar; 4]) -> Self;
        #[doc = "Convert to array."]
        pub fn to_array(self) -> [T::Scalar; 4];
        #[doc = "Instance with all components set to `scalar`."]
        pub fn splat(scalar: T::Scalar) -> Self;
        #[doc = "Return a mask with the result of a component-wise equals comparison."]
        pub fn cmpeq(self, other: Self) -> glam::BVec4;
        #[doc = "Return a mask with the result of a component-wise not-equal comparison."]
        pub fn cmpne(self, other: Self) -> glam::BVec4;
        #[doc = "Return a mask with the result of a component-wise greater-than-or-equal comparison."]
        pub fn cmpge(self, other: Self) -> glam::BVec4;
        #[doc = "Return a mask with the result of a component-wise greater-than comparison."]
        pub fn cmpgt(self, other: Self) -> glam::BVec4;
        #[doc = "Return a mask with the result of a component-wise less-than-or-equal comparison."]
        pub fn cmple(self, other: Self) -> glam::BVec4;
        #[doc = "Return a mask with the result of a component-wise less-than comparison."]
        pub fn cmplt(self, other: Self) -> glam::BVec4;
        #[doc = "Minimum by component."]
        pub fn min(self, other: Self) -> Self;
        #[doc = "Maximum by component."]
        pub fn max(self, other: Self) -> Self;
        #[doc = "Horizontal minimum (smallest component)."]
        pub fn min_element(self) -> T::Scalar;
        #[doc = "Horizontal maximum (largest component)."]
        pub fn max_element(self) -> T::Scalar;
        #[doc = "Component-wise clamp."]
        pub fn clamp(self, min: Self, max: Self) -> Self;
    );
}

impl<T> Point4<T>
where
    T: Unit,
    T::Scalar: FloatScalar,
{
    crate::forward_all_to_raw!(
        #[doc = "True if all components are non-infinity and non-NaN."]
        pub fn is_finite(&self) -> bool;
        #[doc = "True if any component is NaN."]
        pub fn is_nan(&self) -> bool;
        #[doc = "Return a mask where each bit is set if the corresponding component is NaN."]
        pub fn is_nan_mask(&self) -> glam::BVec4;
        #[doc = "Round all components up."]
        pub fn ceil(self) -> Self;
        #[doc = "Round all components down."]
        pub fn floor(self) -> Self;
        #[doc = "Round all components."]
        pub fn round(self) -> Self;
        #[doc = "Linear interpolation."]
        pub fn lerp(self, other: Self, t: T::Scalar) -> Self;
    );
}

crate::impl_glam_conversion!(Point2 [f32 => glam::Vec2, f64 => glam::DVec2, i32 => glam::IVec2, u32 => glam::UVec2]);
crate::impl_glam_conversion!(Point3 [f32 => glam::Vec3, f64 => glam::DVec3, i32 => glam::IVec3, u32 => glam::UVec3]);
crate::impl_glam_conversion!(Point4 [f32 => glam::Vec4, f64 => glam::DVec4, i32 => glam::IVec4, u32 => glam::UVec4]);

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
    };
}

impl_point!(Point2 [2] => Vec2, Vector2);
impl_point!(Point3 [3] => Vec3, Vector3);
impl_point!(Point4 [4] => Vec4, Vector4);

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

impl<T> From<glam::Vec3A> for Point3<T>
where
    T: Unit<Scalar = f32>,
{
    fn from(v: glam::Vec3A) -> Self {
        Self::from_raw(v.into())
    }
}

impl<T> From<Point3<T>> for glam::Vec3A
where
    T: Unit<Scalar = f32>,
{
    fn from(v: Point3<T>) -> Self {
        v.to_raw().into()
    }
}

impl<T> Mul<Point3<T>> for glam::Quat
where
    T: Unit<Scalar = f32>,
    T::Scalar: FloatScalar<Vec3f = glam::Vec3>,
{
    type Output = Point3<T>;

    fn mul(self, rhs: Point3<T>) -> Self::Output {
        Point3::from_raw(self * rhs.to_raw())
    }
}

impl<T> Mul<Point3<T>> for glam::DQuat
where
    T: Unit<Scalar = f64>,
    T::Scalar: FloatScalar<Vec3f = glam::DVec3>,
{
    type Output = Point3<T>;

    fn mul(self, rhs: Point3<T>) -> Self::Output {
        Point3::from_raw(self * rhs.to_raw())
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
        assert_eq!(v, Vector3::ZERO);
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
    fn from_into_vector() {
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
}

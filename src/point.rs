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

use bytemuck::{Pod, TransparentWrapper, Zeroable};

use crate::{
    bindings::prelude::*,
    scalar::{FloatScalar, IntScalar, SignedScalar},
    AsRaw, FromRaw, Scalar, ToRaw, Unit, Vector2, Vector3, Vector4,
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

/// SAFETY: `T::Scalar` is `Zeroable`, and `Point2` is `#[repr(C)]`.
unsafe impl<T: Unit> Zeroable for Point2<T> {}
/// SAFETY: `T::Scalar` is `Pod`.
unsafe impl<T: Unit> Pod for Point2<T> {}
unsafe impl<T: Unit> TransparentWrapper<<T::Scalar as Scalar>::Vec2> for Point2<T> {}

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
unsafe impl<T: Unit> TransparentWrapper<<T::Scalar as Scalar>::Vec3> for Point3<T> {}

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
unsafe impl<T: Unit> TransparentWrapper<<T::Scalar as Scalar>::Vec4> for Point4<T> {}

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

macro_rules! float_point_interface {
    ($see_also_doc_ty:ty) => {
        crate::forward_to_raw!(
            $see_also_doc_ty =>
            #[doc = "Distance"]
            pub fn distance(self, other: Self) -> T::Scalar;
            #[doc = "Distance squared"]
            pub fn distance_squared(self, other: Self) -> T::Scalar;
            #[doc = "Midpoint between two points"]
            pub fn midpoint(self, rhs: Self) -> Self;
            #[doc = "Moves towards rhs based on the value d."]
            pub fn move_towards(self, rhs: Self, d: T::Scalar) -> Self;
        );
    };
}

macro_rules! int_point_interface {
    ($vector:ty, $see_also_doc_ty:ty) => {
        crate::forward_to_raw!(
            $see_also_doc_ty =>
            #[doc = "Returns a vector containing the saturating addition of self and rhs."]
            pub fn saturating_add(self, rhs: $vector) -> Self;
            #[doc = "Returns a vector containing the saturating subtraction of self and rhs."]
            pub fn saturating_sub(self, rhs: $vector) -> Self;
            #[doc = "Returns a vector containing the saturating multiplication of self and rhs."]
            pub fn saturating_mul(self, rhs: $vector) -> Self;
            #[doc = "Returns a vector containing the saturating division of self and rhs."]
            pub fn saturating_div(self, rhs: $vector) -> Self;
            #[doc = "Returns a vector containing the wrapping addition of self and rhs."]
            pub fn wrapping_add(self, rhs: $vector) -> Self;
            #[doc = "Returns a vector containing the wrapping subtraction of self and rhs."]
            pub fn wrapping_sub(self, rhs: $vector) -> Self;
            #[doc = "Returns a vector containing the wrapping multiplication of self and rhs."]
            pub fn wrapping_mul(self, rhs: $vector) -> Self;
            #[doc = "Returns a vector containing the wrapping division of self and rhs."]
            pub fn wrapping_div(self, rhs: $vector) -> Self;
        );
    };
}

crate::forward_op_to_raw!(Point2, Add<Vector2<T>>::add -> Self);
crate::forward_op_to_raw!(Point3, Add<Vector3<T>>::add -> Self);
crate::forward_op_to_raw!(Point4, Add<Vector4<T>>::add -> Self);
crate::forward_op_to_raw!(Point2, Sub<Vector2<T>>::sub -> Self);
crate::forward_op_to_raw!(Point3, Sub<Vector3<T>>::sub -> Self);
crate::forward_op_to_raw!(Point4, Sub<Vector4<T>>::sub -> Self);
crate::forward_op_to_raw!(Point2, Sub<Self>::sub -> Vector2<T>);
crate::forward_op_to_raw!(Point3, Sub<Self>::sub -> Vector3<T>);
crate::forward_op_to_raw!(Point4, Sub<Self>::sub -> Vector4<T>);

crate::forward_neg_to_raw!(Point2);
crate::forward_neg_to_raw!(Point3);
crate::forward_neg_to_raw!(Point4);

crate::forward_op_assign_to_raw!(Point2, AddAssign<Vector2<T>>::add_assign);
crate::forward_op_assign_to_raw!(Point3, AddAssign<Vector3<T>>::add_assign);
crate::forward_op_assign_to_raw!(Point4, AddAssign<Vector4<T>>::add_assign);
crate::forward_op_assign_to_raw!(Point2, SubAssign<Vector2<T>>::sub_assign);
crate::forward_op_assign_to_raw!(Point3, SubAssign<Vector3<T>>::sub_assign);
crate::forward_op_assign_to_raw!(Point4, SubAssign<Vector4<T>>::sub_assign);

crate::derive_standard_traits!(Point2 {
    x: T::Scalar,
    y: T::Scalar
});
crate::derive_standard_traits!(Point3 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar
});
crate::derive_standard_traits!(Point4 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar,
    w: T::Scalar
});

crate::derive_array_conversion_traits!(Point2, 2);
crate::derive_array_conversion_traits!(Point3, 3);
crate::derive_array_conversion_traits!(Point4, 4);

crate::derive_tuple_conversion_traits!(Point2 {
    x: T::Scalar,
    y: T::Scalar
});
crate::derive_tuple_conversion_traits!(Point3 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar
});
crate::derive_tuple_conversion_traits!(Point4 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar,
    w: T::Scalar
});

crate::derive_glam_conversion_traits!(Point2 {
    x: T::Scalar,
    y: T::Scalar
});
crate::derive_glam_conversion_traits!(Point3 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar
});
crate::derive_glam_conversion_traits!(Point4 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar,
    w: T::Scalar
});

impl<T: Unit> ToRaw for Point2<T> {
    type Raw = <T::Scalar as Scalar>::Vec2;

    fn to_raw(self) -> Self::Raw {
        bytemuck::cast(self)
    }
}

impl<T: Unit> FromRaw for Point2<T> {
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

impl<T: Unit> ToRaw for Point3<T> {
    type Raw = <T::Scalar as Scalar>::Vec3;

    fn to_raw(self) -> Self::Raw {
        bytemuck::cast(self)
    }
}

impl<T: Unit> FromRaw for Point3<T> {
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

impl<T: Unit> ToRaw for Point4<T> {
    type Raw = <T::Scalar as Scalar>::Vec4;

    fn to_raw(self) -> Self::Raw {
        bytemuck::cast(self)
    }
}

impl<T: Unit> FromRaw for Point4<T> {
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

    crate::forward_constructors!(2, glam::Vec2);
    crate::forward_comparison!(glam::BVec2, glam::Vec2);
    crate::casting_interface!(Point2 {
        x: T::Scalar,
        y: T::Scalar
    });
    crate::tuple_interface!(Point2 {
        x: T::Scalar,
        y: T::Scalar
    });
    crate::array_interface!(2);

    crate::forward_to_raw!(
        glam::Vec2 =>
        #[doc = "Extend with z-component to [`Point3`]."]
        pub fn extend(self, z: T::Scalar) -> Point3<T>;
        #[doc = "Replace the x-component with a new value."]
        pub fn with_x(self, x: T::Scalar) -> Self;
        #[doc = "Replace the y-component with a new value."]
        pub fn with_y(self, y: T::Scalar) -> Self;
    );

    point_interface!(Point2, Vector2);
}

impl<T> Point2<T>
where
    T: Unit,
    T::Scalar: FloatScalar,
{
    /// All NaN.
    pub const NAN: Self = Self {
        x: <T::Scalar as FloatScalar>::NAN,
        y: <T::Scalar as FloatScalar>::NAN,
    };
    /// All positive infinity.
    pub const INFINITY: Self = Self {
        x: <T::Scalar as FloatScalar>::INFINITY,
        y: <T::Scalar as FloatScalar>::INFINITY,
    };
    /// All negative infinity.
    pub const NEG_INFINITY: Self = Self {
        x: <T::Scalar as FloatScalar>::NEG_INFINITY,
        y: <T::Scalar as FloatScalar>::NEG_INFINITY,
    };

    crate::forward_float_ops!(glam::BVec2, glam::Vec2);
    float_point_interface!(glam::Vec2);
}

impl<T> Point2<T>
where
    T: Unit,
    T::Scalar: IntScalar,
{
    int_point_interface!(Vector2<T>, glam::IVec2);
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

    crate::forward_constructors!(3, glam::Vec3);
    crate::forward_comparison!(glam::BVec3, glam::Vec3);
    crate::casting_interface!(Point3 {
        x: T::Scalar,
        y: T::Scalar,
        z: T::Scalar
    });
    crate::tuple_interface!(Point3 {
        x: T::Scalar,
        y: T::Scalar,
        z: T::Scalar
    });
    crate::array_interface!(3);

    crate::forward_to_raw!(
        glam::Vec3 =>
        #[doc = "Extend with w-component to [`Point4`]."]
        pub fn extend(self, w: T::Scalar) -> Point4<T>;
        #[doc = "Truncate to [`Point2`]."]
        pub fn truncate(self) -> Point2<T>;
        #[doc = "Replace the x-component with a new value."]
        pub fn with_x(self, x: T::Scalar) -> Self;
        #[doc = "Replace the y-component with a new value."]
        pub fn with_y(self, y: T::Scalar) -> Self;
        #[doc = "Replace the z-component with a new value."]
        pub fn with_z(self, z: T::Scalar) -> Self;
    );

    point_interface!(Point3, Vector3);
}

impl<T> Point3<T>
where
    T: Unit,
    T::Scalar: FloatScalar,
{
    /// All NaN.
    pub const NAN: Self = Self {
        x: <T::Scalar as FloatScalar>::NAN,
        y: <T::Scalar as FloatScalar>::NAN,
        z: <T::Scalar as FloatScalar>::NAN,
    };
    /// All positive infinity.
    pub const INFINITY: Self = Self {
        x: <T::Scalar as FloatScalar>::INFINITY,
        y: <T::Scalar as FloatScalar>::INFINITY,
        z: <T::Scalar as FloatScalar>::INFINITY,
    };
    /// All negative infinity.
    pub const NEG_INFINITY: Self = Self {
        x: <T::Scalar as FloatScalar>::NEG_INFINITY,
        y: <T::Scalar as FloatScalar>::NEG_INFINITY,
        z: <T::Scalar as FloatScalar>::NEG_INFINITY,
    };

    crate::forward_float_ops!(glam::BVec3, glam::Vec3);
    float_point_interface!(glam::Vec3);
}

impl<T> Point3<T>
where
    T: Unit,
    T::Scalar: IntScalar,
{
    int_point_interface!(Vector3<T>, glam::IVec3);
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

    crate::forward_constructors!(4, glam::Vec4);
    crate::forward_comparison!(glam::BVec4, glam::Vec4);
    crate::casting_interface!(Point4 {
        x: T::Scalar,
        y: T::Scalar,
        z: T::Scalar,
        w: T::Scalar
    });
    crate::tuple_interface!(Point4 {
        x: T::Scalar,
        y: T::Scalar,
        z: T::Scalar,
        w: T::Scalar
    });
    crate::array_interface!(4);

    crate::forward_to_raw!(
        glam::Vec4 =>
        #[doc = "Truncate to [`Point3`]."]
        pub fn truncate(self) -> Point3<T>;
        #[doc = "Replace the x-component with a new value."]
        pub fn with_x(self, x: T::Scalar) -> Self;
        #[doc = "Replace the y-component with a new value."]
        pub fn with_y(self, y: T::Scalar) -> Self;
        #[doc = "Replace the z-component with a new value."]
        pub fn with_z(self, z: T::Scalar) -> Self;
        #[doc = "Replace the w-component with a new value."]
        pub fn with_w(self, w: T::Scalar) -> Self;
    );

    point_interface!(Point4, Vector4);
}

impl<T> Point4<T>
where
    T: Unit,
    T::Scalar: FloatScalar,
{
    /// All NaN.
    pub const NAN: Self = Self {
        x: <T::Scalar as FloatScalar>::NAN,
        y: <T::Scalar as FloatScalar>::NAN,
        z: <T::Scalar as FloatScalar>::NAN,
        w: <T::Scalar as FloatScalar>::NAN,
    };
    /// All positive infinity.
    pub const INFINITY: Self = Self {
        x: <T::Scalar as FloatScalar>::INFINITY,
        y: <T::Scalar as FloatScalar>::INFINITY,
        z: <T::Scalar as FloatScalar>::INFINITY,
        w: <T::Scalar as FloatScalar>::INFINITY,
    };
    /// All negative infinity.
    pub const NEG_INFINITY: Self = Self {
        x: <T::Scalar as FloatScalar>::NEG_INFINITY,
        y: <T::Scalar as FloatScalar>::NEG_INFINITY,
        z: <T::Scalar as FloatScalar>::NEG_INFINITY,
        w: <T::Scalar as FloatScalar>::NEG_INFINITY,
    };

    crate::forward_float_ops!(glam::BVec4, glam::Vec4);
    float_point_interface!(glam::Vec4);
}

impl<T> Point4<T>
where
    T: Unit,
    T::Scalar: IntScalar,
{
    int_point_interface!(Vector4<T>, glam::IVec4);
}

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

impl<T: Unit> From<Vector2<T>> for Point2<T> {
    fn from(vec: Vector2<T>) -> Point2<T> {
        Self::from_raw(vec.to_raw())
    }
}

impl<T: Unit> From<Vector3<T>> for Point3<T> {
    #[inline]
    fn from(vec: Vector3<T>) -> Point3<T> {
        Self::from_raw(vec.to_raw())
    }
}

impl<T: Unit> From<Vector4<T>> for Point4<T> {
    #[inline]
    fn from(vec: Vector4<T>) -> Point4<T> {
        Self::from_raw(vec.to_raw())
    }
}

impl<T: Unit> From<Point2<T>> for Vector2<T> {
    #[inline]
    fn from(point: Point2<T>) -> Vector2<T> {
        Self::from_raw(point.to_raw())
    }
}

impl<T: Unit> From<Point3<T>> for Vector3<T> {
    #[inline]
    fn from(point: Point3<T>) -> Vector3<T> {
        Self::from_raw(point.to_raw())
    }
}

impl<T: Unit> From<Point4<T>> for Vector4<T> {
    #[inline]
    fn from(point: Point4<T>) -> Vector4<T> {
        Self::from_raw(point.to_raw())
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
}

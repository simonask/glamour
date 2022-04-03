//! Vector types.
//!
//! These are first and foremost geometric vectors in 2D, 3D, or 4D space, but
//! they may also be used as the "generic" SIMD type when there is a temporary
//! need to interpret a size or a point as a vector.
//!
//! For example, swizzling and normalization are not implemented for sizes and
//! points, but by temporarily converting to a vector, it can still be done
//! transparently.

use core::iter::Sum;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};

use crate::{
    traits::{Lerp, SimdVec, SimdVecFloat},
    Point2, Point3, Point4, Scalar, Size2, Size3, Unit,
};

/// 2D vector.
///
/// Bitwise compatible with [`glam::Vec2`] / [`glam::DVec2`] / [`glam::IVec2`]
///
/// Alignment: Same as the scalar.
#[repr(C)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
pub struct Vector2<T: Unit = f32> {
    /// X coordinate
    pub x: T::Scalar,
    /// Y coordinate
    pub y: T::Scalar,
}

/// 3D vector.
///
/// Alignment: Same as the scalar (so not 16 bytes). If you really need 16-byte
/// alignment, use [`Vector4`].
#[repr(C)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
pub struct Vector3<T: Unit = f32> {
    /// X coordinate
    pub x: T::Scalar,
    /// Y coordinate
    pub y: T::Scalar,
    /// Z coordinate
    pub z: T::Scalar,
}

/// 4D vector.
///
/// Alignment: This is always 16-byte aligned. [`glam::DVec4`] is only 8-byte
/// aligned (for some reason), and integer vectors are only 4-byte aligned,
/// which means that reference-casting from those glam types to `Vector4` type
/// will fail (but not the other way around - see [`Vector4::as_raw()`]).
#[repr(C, align(16))]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
pub struct Vector4<T: Unit = f32> {
    /// X coordinate
    pub x: T::Scalar,
    /// Y coordinate
    pub y: T::Scalar,
    /// Z coordinate
    pub z: T::Scalar,
    /// W coordinate
    pub w: T::Scalar,
}

crate::impl_common!(Vector2 {
    x: T::Scalar,
    y: T::Scalar
});
crate::impl_common!(Vector3 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar
});
crate::impl_common!(Vector4 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar,
    w: T::Scalar
});

crate::impl_simd_common!(Vector2 [2] => Vec2, glam::BVec2 { x, y });
crate::impl_simd_common!(Vector3 [3] => Vec3, glam::BVec3 { x, y, z });
crate::impl_simd_common!(Vector4 [4] => Vec4, glam::BVec4 { x, y, z, w });

crate::impl_as_tuple!(Vector2 {
    x: T::Scalar,
    y: T::Scalar
});
crate::impl_as_tuple!(Vector3 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar
});
crate::impl_as_tuple!(Vector4 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar,
    w: T::Scalar
});

crate::impl_glam_conversion!(Vector2, 2 [f32 => glam::Vec2, f64 => glam::DVec2, i32 => glam::IVec2, u32 => glam::UVec2]);
crate::impl_glam_conversion!(Vector3, 3 [f32 => glam::Vec3, f64 => glam::DVec3, i32 => glam::IVec3, u32 => glam::UVec3]);
crate::impl_glam_conversion!(Vector4, 4 [f32 => glam::Vec4, f64 => glam::DVec4, i32 => glam::IVec4, u32 => glam::UVec4]);

crate::impl_scaling!(Vector2, 2 [f32, f64, i32, u32]);
crate::impl_scaling!(Vector3, 3 [f32, f64, i32, u32]);
crate::impl_scaling!(Vector4, 4 [f32, f64, i32, u32]);

macro_rules! impl_vector {
    ($base_type_name:ident [ $dimensions:literal ] => $vec_ty:ident, $point_ty:ident $(, $size_ty:ident)?) => {
        impl<T: Unit> $base_type_name<T> {
            #[doc = "Instantiate vector with all components set to 2."]
            #[doc = ""]
            #[doc = "This is a shorthand for `Self::one() + Self::one()`."]
            #[inline(always)]
            #[must_use]
            pub fn two() -> Self {
                Self::one() + Self::one()
            }

            #[doc = "Multiply all components of this vector with a scalar value."]
            #[doc = ""]
            #[doc = "This exists as a method because `Mul<T::Scalar>` cannot be implemented"]
            #[doc = "for all `Vector<T>` (but it is implemented for all `Vector<T>` where `T:"]
            #[doc = "Unit<Scalar = f32>` etc.)."]
            #[doc = ""]
            #[doc = "This is equivalent to `self * Self::splat(scalar)`."]
            #[inline(always)]
            #[must_use]
            pub fn mul_scalar(self, scalar: T::Scalar) -> Self {
                self * Self::splat(scalar)
            }

            #[doc = "Same notes as [`mul_scalar()`](Self::mul_scalar()), except division."]
            #[inline(always)]
            #[must_use]
            pub fn div_scalar(self, scalar: T::Scalar) -> Self {
                self / Self::splat(scalar)
            }

            #[doc = "Same notes as [`mul_scalar()`](Self::mul_scalar()), except remainder."]
            #[inline(always)]
            #[must_use]
            pub fn rem_scalar(self, scalar: T::Scalar) -> Self {
                self % Self::splat(scalar)
            }

            #[doc = "Dot product."]
            #[inline(always)]
            #[must_use]
            pub fn dot(self, other: Self) -> T::Scalar {
                T::Scalar::from_raw(self.to_raw().dot(other.to_raw()))
            }

            #[doc = "Instantiate from point."]
            #[inline(always)]
            #[must_use]
            pub fn from_point(point: $point_ty<T>) -> Self {
                bytemuck::cast(point)
            }

            #[doc = "Convert to point."]
            #[inline(always)]
            #[must_use]
            pub fn to_point(self) -> $point_ty<T> {
                bytemuck::cast(self)
            }

            #[doc = "Reinterpret as point."]
            #[inline(always)]
            #[must_use]
            pub fn as_point(&self) -> &$point_ty<T> {
                bytemuck::cast_ref(self)
            }

            #[doc = "Reinterpret as point."]
            #[inline(always)]
            #[must_use]
            pub fn as_point_mut(&mut self) -> &mut $point_ty<T> {
                bytemuck::cast_mut(self)
            }

            $(
                #[doc = "Instantiate from size."]
                #[inline(always)]
                #[must_use]
                pub fn from_size(size: $size_ty<T>) -> Self {
                    bytemuck::cast(size)
                }

                #[doc = "Convert to size."]
                #[inline(always)]
                #[must_use]
                pub fn to_size(self) -> $size_ty<T> {
                    bytemuck::cast(self)
                }

                #[doc = "Reinterpret as size."]
                #[inline(always)]
                #[must_use]
                pub fn as_size(&self) -> &$size_ty<T> {
                    bytemuck::cast_ref(self)
                }

                #[doc = "Reinterpret as size."]
                #[inline(always)]
                #[must_use]
                pub fn as_size_mut(&mut self) -> &mut $size_ty<T> {
                    bytemuck::cast_mut(self)
                }
            )*

            #[doc = "Select two components from this vector and return a 2D vector made from"]
            #[doc = "those components."]
            #[inline(always)]
            #[must_use]
            pub fn swizzle2<const X: usize, const Y: usize>(&self) -> Vector2<T> {
                [self.const_get::<X>(), self.const_get::<Y>()].into()
            }

            #[doc = "Select three components from this vector and return a 3D vector made from"]
            #[doc = "those components."]
            #[inline(always)]
            #[must_use]
            pub fn swizzle3<const X: usize, const Y: usize, const Z: usize>(&self) -> Vector3<T> {
                [
                    self.const_get::<X>(),
                    self.const_get::<Y>(),
                    self.const_get::<Z>(),
                ]
                .into()
            }

            #[doc = "Select four components from this vector and return a 4D vector made from"]
            #[doc = "those components."]
            #[inline(always)]
            #[must_use]
            pub fn swizzle4<const X: usize, const Y: usize, const Z: usize, const W: usize>(
                &self,
            ) -> Vector4<T> {
                [
                    self.const_get::<X>(),
                    self.const_get::<Y>(),
                    self.const_get::<Z>(),
                    self.const_get::<W>(),
                ]
                .into()
            }
        }

        impl<T> $base_type_name<T>
        where
            T: crate::traits::UnitTypes,
            T::$vec_ty: SimdVecFloat<$dimensions, Scalar = T::Primitive>,
        {
            #[doc = "Normalize the vector."]
            #[doc = ""]
            #[doc = "See (e.g.) [`glam::Vec4::normalize()`]."]
            #[must_use]
            #[inline]
            pub fn normalize(&self) -> Self {
                Self::from_raw(self.to_raw().normalize())
            }

            #[doc = "Get the length of the vector"]
            #[doc = ""]
            #[doc = "See (e.g.) [`glam::Vec3::length()]."]
            #[must_use]
            #[inline]
            pub fn length(&self) -> T::Primitive {
                T::Primitive::from_raw(self.as_raw().length())
            }

            #[doc = "Returns a vector containing `e^self` (the exponential function) for each element of `self`."]
            #[inline]
            #[must_use]
            pub fn exp(self) -> Self {
                Self::from_raw(self.to_raw().exp())
            }
            #[doc = "Returns a vector containing each element of `self` raised to the power of `n`."]
            #[inline]
            #[must_use]
            pub fn powf(self, n: T::Scalar) -> Self {
                Self::from_raw(self.to_raw().powf(n.to_raw()))
            }
            #[doc = "Returns a vector containing the reciprocal `1.0/n` of each element of `self`."]
            #[inline]
            #[must_use]
            pub fn recip(self) -> Self {
                Self::from_raw(self.to_raw().recip())
            }
            #[doc = "Fused multiply-add. Computes `(self * a) + b` element-wise with only one rounding error, yielding a more accurate result than an unfused multiply-add."]
            #[inline]
            #[must_use]
            pub fn mul_add(self, a: Self, b: Self) -> Self {
                Self::from_raw(self.to_raw().mul_add(a.to_raw(), b.to_raw()))
            }
        }

        impl<T: Unit> Add<Self> for $base_type_name<T> {
            type Output = Self;

            #[must_use]
            fn add(self, rhs: Self) -> Self::Output {
                Self::from_raw(self.to_raw() + rhs.to_raw())
            }
        }
        impl<T: Unit> Sub<Self> for $base_type_name<T> {
            type Output = Self;

            #[must_use]
            fn sub(self, rhs: Self) -> Self::Output {
                Self::from_raw(self.to_raw() - rhs.to_raw())
            }
        }
        impl<T: Unit> Mul<Self> for $base_type_name<T> {
            type Output = Self;

            #[must_use]
            fn mul(self, rhs: Self) -> Self::Output {
                Self::from_raw(self.to_raw() * rhs.to_raw())
            }
        }
        impl<T: Unit> Div<Self> for $base_type_name<T> {
            type Output = Self;

            #[must_use]
            fn div(self, rhs: Self) -> Self::Output {
                Self::from_raw(self.to_raw() / rhs.to_raw())
            }
        }

        impl<T: Unit> AddAssign for $base_type_name<T> {
            fn add_assign(&mut self, rhs: Self) {
                *self.as_raw_mut() += rhs.to_raw();
            }
        }

        impl<T: Unit> SubAssign for $base_type_name<T> {
            fn sub_assign(&mut self, rhs: Self) {
                *self.as_raw_mut() -= rhs.to_raw();
            }
        }

        impl<T: Unit> MulAssign for $base_type_name<T> {
            fn mul_assign(&mut self, rhs: Self) {
                *self.as_raw_mut() *= rhs.to_raw();
            }
        }

        impl<T: Unit> DivAssign for $base_type_name<T> {
            fn div_assign(&mut self, rhs: Self) {
                *self.as_raw_mut() /= rhs.to_raw();
            }
        }

        impl<T: Unit> Rem for $base_type_name<T> {
            type Output = Self;

            #[inline(always)]
            #[must_use]
            fn rem(self, rhs: Self) -> Self::Output {
                Self::from_raw(self.to_raw() % rhs.to_raw())
            }
        }

        impl<T: Unit> RemAssign for $base_type_name<T> {
            #[inline(always)]
            fn rem_assign(&mut self, rhs: Self) {
                *self.as_raw_mut() %= rhs.to_raw();
            }
        }

        impl<T: Unit> Sum for $base_type_name<T> {
            #[must_use]
            fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
                iter.fold(Self::zero(), Add::add)
            }
        }

        impl<T> Lerp<T::Primitive> for $base_type_name<T>
        where
            T: crate::traits::UnitTypes,
            T::$vec_ty: Lerp<T::Primitive>,
        {
            #[inline(always)]
            #[must_use]
            fn lerp(self, end: Self, t: T::Primitive) -> Self {
                Self::from_raw(self.to_raw().lerp(end.to_raw(), t.to_raw()))
            }
        }
    };
}

impl_vector!(Vector2 [2] => Vec2, Point2, Size2);
impl_vector!(Vector3 [3] => Vec3, Point3, Size3);
impl_vector!(Vector4 [4] => Vec4, Point4);

impl<T: Unit> Vector2<T> {
    /// Unit vector in the direction of the X axis.
    #[inline(always)]
    #[must_use]
    pub fn unit_x() -> Self {
        Self {
            x: T::Scalar::one(),
            y: T::Scalar::zero(),
        }
    }

    /// Unit vector in the direction of the Y axis.
    #[inline(always)]
    #[must_use]
    pub fn unit_y() -> Self {
        Self {
            x: T::Scalar::zero(),
            y: T::Scalar::one(),
        }
    }

    /// Select components of this vector and return a new vector containing
    /// those components.
    #[inline(always)]
    #[must_use]
    pub fn swizzle<const X: usize, const Y: usize>(&self) -> Self {
        self.swizzle2::<X, Y>()
    }

    /// Convert this `Vector2` to a [`Vector3`](Vector3) with `z` component.
    #[must_use]
    pub fn to_3d(self, z: T::Scalar) -> Vector3<T> {
        Vector3 {
            x: self.x,
            y: self.y,
            z,
        }
    }
}

impl<T: Unit> Vector3<T> {
    /// Unit vector in the direction of the X axis.
    #[inline(always)]
    #[must_use]
    pub fn unit_x() -> Self {
        Self {
            x: T::Scalar::one(),
            y: T::Scalar::zero(),
            z: T::Scalar::zero(),
        }
    }

    /// Unit vector in the direction of the Y axis.
    #[inline(always)]
    #[must_use]
    pub fn unit_y() -> Self {
        Self {
            x: T::Scalar::zero(),
            y: T::Scalar::one(),
            z: T::Scalar::zero(),
        }
    }

    /// Unit vector in the direction of the Z axis.
    #[inline(always)]
    #[must_use]
    pub fn unit_z() -> Self {
        Self {
            x: T::Scalar::zero(),
            y: T::Scalar::zero(),
            z: T::Scalar::one(),
        }
    }

    /// Select components of this vector and return a new vector containing
    /// those components.
    #[inline(always)]
    #[must_use]
    pub fn swizzle<const X: usize, const Y: usize, const Z: usize>(&self) -> Self {
        self.swizzle3::<X, Y, Z>()
    }

    /// Convert to [`Vector4`] with `w` component.
    #[must_use]
    pub fn to_4d(self, w: T::Scalar) -> Vector4<T> {
        Vector4 {
            x: self.x,
            y: self.y,
            z: self.z,
            w,
        }
    }
}

impl<T> Vector3<T>
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

impl<T> From<glam::Vec3A> for Vector3<T>
where
    T: Unit,
    T::Scalar: Scalar<Primitive = f32>,
{
    fn from(v: glam::Vec3A) -> Self {
        Self::from_raw(v.into())
    }
}

impl<T> From<Vector3<T>> for glam::Vec3A
where
    T: Unit,
    T::Scalar: Scalar<Primitive = f32>,
{
    fn from(v: Vector3<T>) -> Self {
        v.to_raw().into()
    }
}

impl<T: Unit> Vector4<T> {
    /// Unit vector in the direction of the X axis.
    #[inline(always)]
    #[must_use]
    pub fn unit_x() -> Self {
        Self {
            x: T::Scalar::one(),
            y: T::Scalar::zero(),
            z: T::Scalar::zero(),
            w: T::Scalar::zero(),
        }
    }

    /// Unit vector in the direction of the Y axis.
    #[inline(always)]
    #[must_use]
    pub fn unit_y() -> Self {
        Self {
            x: T::Scalar::zero(),
            y: T::Scalar::one(),
            z: T::Scalar::zero(),
            w: T::Scalar::zero(),
        }
    }

    /// Unit vector in the direction of the Z axis.
    #[inline(always)]
    #[must_use]
    pub fn unit_z() -> Self {
        Self {
            x: T::Scalar::zero(),
            y: T::Scalar::zero(),
            z: T::Scalar::one(),
            w: T::Scalar::zero(),
        }
    }

    /// Unit vector in the direction of the W axis.
    #[inline(always)]
    #[must_use]
    pub fn unit_w() -> Self {
        Self {
            x: T::Scalar::zero(),
            y: T::Scalar::zero(),
            z: T::Scalar::zero(),
            w: T::Scalar::one(),
        }
    }

    /// Select components of this vector and return a new vector containing
    /// those components.
    #[inline(always)]
    #[must_use]
    pub fn swizzle<const X: usize, const Y: usize, const Z: usize, const W: usize>(&self) -> Self {
        self.swizzle4::<X, Y, Z, W>()
    }
}

crate::impl_mint!(Vector2, 2, Vector2);
crate::impl_mint!(Vector3, 3, Vector3);
crate::impl_mint!(Vector4, 4, Vector4);

#[cfg(test)]
mod tests {
    use super::*;

    struct F32;
    impl Unit for F32 {
        type Scalar = f32;
    }
    struct F64;
    impl Unit for F64 {
        type Scalar = f64;
    }
    struct I32;
    impl Unit for I32 {
        type Scalar = i32;
    }
    struct U32;
    impl Unit for U32 {
        type Scalar = u32;
    }

    // Note: Not using the "untyped" scalar units.
    type Vec2 = Vector2<F32>;
    type Vec3 = Vector3<F32>;
    type Vec4 = Vector4<F32>;
    type DVec2 = Vector2<F64>;
    type DVec3 = Vector3<F64>;
    type DVec4 = Vector4<F64>;
    type IVec2 = Vector2<I32>;
    type IVec3 = Vector3<I32>;
    type IVec4 = Vector4<I32>;
    type UVec2 = Vector2<U32>;
    type UVec3 = Vector3<U32>;
    type UVec4 = Vector4<U32>;

    macro_rules! check_splat {
        ($value:expr, $op:ident, $expected:expr) => {
            assert_eq!(Vec2::splat($value).$op(), Vec2::splat($expected));
            assert_eq!(Vec3::splat($value).$op(), Vec3::splat($expected));
            assert_eq!(Vec4::splat($value).$op(), Vec4::splat($expected));
            assert_eq!(DVec2::splat($value).$op(), DVec2::splat($expected));
            assert_eq!(DVec3::splat($value).$op(), DVec3::splat($expected));
            assert_eq!(DVec4::splat($value).$op(), DVec4::splat($expected));
        };
        ($value:expr, $op:ident ( $arg:expr ), $expected:expr) => {
            assert_eq!(
                Vec2::splat($value).$op(Vec2::splat($arg)),
                Vec2::splat($expected)
            );
            assert_eq!(
                Vec3::splat($value).$op(Vec3::splat($arg)),
                Vec3::splat($expected)
            );
            assert_eq!(
                Vec4::splat($value).$op(Vec4::splat($arg)),
                Vec4::splat($expected)
            );
            assert_eq!(
                DVec2::splat($value).$op(DVec2::splat($arg)),
                DVec2::splat($expected)
            );
            assert_eq!(
                DVec3::splat($value).$op(DVec3::splat($arg)),
                DVec3::splat($expected)
            );
            assert_eq!(
                DVec4::splat($value).$op(DVec4::splat($arg)),
                DVec4::splat($expected)
            );
        };
    }

    macro_rules! check_splat_ints {
        ($value:expr, $op:ident, $expected:expr) => {
            check_splat!($value as _, $op, $expected as _);
            assert_eq!(IVec2::splat($value).$op(), IVec2::splat($expected));
            assert_eq!(IVec3::splat($value).$op(), IVec3::splat($expected));
            assert_eq!(IVec4::splat($value).$op(), IVec4::splat($expected));
            assert_eq!(UVec2::splat($value).$op(), UVec2::splat($expected));
            assert_eq!(UVec3::splat($value).$op(), UVec3::splat($expected));
            assert_eq!(UVec4::splat($value).$op(), UVec4::splat($expected));
        };
        ($value:expr, $op:ident ( $arg:expr ), $expected:expr) => {
            assert_eq!(
                IVec2::splat($value).$op(IVec2::splat($arg)),
                IVec2::splat($expected)
            );
            assert_eq!(
                IVec3::splat($value).$op(IVec3::splat($arg)),
                IVec3::splat($expected)
            );
            assert_eq!(
                IVec4::splat($value).$op(IVec4::splat($arg)),
                IVec4::splat($expected)
            );
            assert_eq!(
                UVec2::splat($value).$op(UVec2::splat($arg)),
                UVec2::splat($expected)
            );
            assert_eq!(
                UVec3::splat($value).$op(UVec3::splat($arg)),
                UVec3::splat($expected)
            );
            assert_eq!(
                UVec4::splat($value).$op(UVec4::splat($arg)),
                UVec4::splat($expected)
            );
        };
    }

    #[test]
    fn round() {
        check_splat!(1.4, round, 1.0);
        check_splat!(1.6, round, 2.0);
        check_splat!(1.5, round, 2.0);
        check_splat!(1.49999, round, 1.0);
    }

    #[test]
    fn ceil() {
        check_splat!(1.4, ceil, 2.0);
        check_splat!(1.6, ceil, 2.0);
        check_splat!(1.5, ceil, 2.0);
    }

    #[test]
    fn floor() {
        check_splat!(1.4, floor, 1.0);
        check_splat!(1.6, floor, 1.0);
        check_splat!(1.5, floor, 1.0);
    }

    #[test]
    fn arithmetic() {
        check_splat_ints!(2, div(1), 2);
        check_splat_ints!(10, div(2), 5);
        check_splat!(2.0, mul(2.5), 2.0 * 2.5);
        check_splat_ints!(10, mul(2), 20);
        check_splat_ints!(2, add(1), 3);
        check_splat_ints!(10, add(2), 12);
        check_splat_ints!(2, sub(1), 1);
        check_splat_ints!(10, sub(2), 8);
    }

    #[test]
    fn lerp() {
        let a = Vec2 { x: 1.0, y: 2.0 };
        let b = Vec2 { x: 2.0, y: 3.0 };

        assert_eq!(a.lerp(b, 0.0), a);
        assert_eq!(a.lerp(b, 0.5), (1.5, 2.5));
        assert_eq!(a.lerp(b, 1.0), b);
    }

    #[test]
    fn const_construction() {
        const CONST_VEC: Vec2 = Vector2 { x: 1.0, y: 2.0 };
        static STATIC_VEC: Vec2 = Vector2 { x: 2.0, y: 3.0 };

        assert_eq!(CONST_VEC, (1.0, 2.0));
        assert_eq!(STATIC_VEC, (2.0, 3.0));
    }

    #[test]
    fn scaling_by_scalar() {
        // Test that vector types can be multiplied/divided by their
        // (unsplatted) scalar. This doesn't work in generic code, but it should
        // work when the concrete vector type is known to the compiler.

        {
            let x: Vec4 = (1.0, 2.0, 3.0, 4.0).into();

            let a = x * 2.0;
            let b = x / 2.0;

            assert_eq!(a, (2.0, 4.0, 6.0, 8.0));
            assert_eq!(b, (0.5, 1.0, 1.5, 2.0));
        }
        {
            let x: DVec4 = (1.0, 2.0, 3.0, 4.0).into();

            let a = x * 2.0;
            let b = x / 2.0;

            assert_eq!(a, (2.0, 4.0, 6.0, 8.0));
            assert_eq!(b, (0.5, 1.0, 1.5, 2.0));
        }
        {
            let x: IVec4 = (1, 2, 3, 4).into();

            let a = x * 2;
            let b = x / 2;

            assert_eq!(a, (2, 4, 6, 8));
            assert_eq!(b, (0, 1, 1, 2));
        }
        {
            let x: UVec4 = (1, 2, 3, 4).into();

            let a = x * 2;
            let b = x / 2;

            assert_eq!(a, (2, 4, 6, 8));
            assert_eq!(b, (0, 1, 1, 2));
        }
    }
}

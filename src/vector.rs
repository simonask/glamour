//! Vector types.
//!
//! These are first and foremost geometric vectors in 2D, 3D, or 4D space, but
//! they may also be used as the "generic" SIMD type when there is a temporary
//! need to interpret a size or a point as a vector.
//!
//! For example, swizzling and normalization are not implemented for sizes and
//! points, but by temporarily converting to a vector, it can still be done
//! transparently.

use core::iter::{Product, Sum};
use core::ops::Mul;

use bytemuck::{Pod, Zeroable};
use num_traits::identities::{ConstOne, ConstZero};

use crate::scalar::FloatScalar;
use crate::unit::FloatUnit;
use crate::{bindings::prelude::*, Point2, Point3, Point4, Scalar, Size2, Size3, Unit};
use crate::{peel, peel_ref, wrap, Transparent};

/// Vector swizzling by const generics.
///
/// For GLSL-like swizzling, see [`glam::Vec2Swizzles`], [`glam::Vec3Swizzles`],
/// or [`glam::Vec4Swizzles`].
pub trait Swizzle<T: Unit> {
    #[doc = "Select two components from this vector and return a 2D vector made from"]
    #[doc = "those components."]
    #[must_use]
    fn swizzle2<const X: usize, const Y: usize>(&self) -> Vector2<T>;

    #[doc = "Select three components from this vector and return a 3D vector made from"]
    #[doc = "those components."]
    #[must_use]
    fn swizzle3<const X: usize, const Y: usize, const Z: usize>(&self) -> Vector3<T>;

    #[doc = "Select four components from this vector and return a 4D vector made from"]
    #[doc = "those components."]
    #[must_use]
    fn swizzle4<const X: usize, const Y: usize, const Z: usize, const W: usize>(
        &self,
    ) -> Vector4<T>;
}

/// 2D vector.
///
/// Bitwise compatible with [`glam::Vec2`] / [`glam::DVec2`] / [`glam::IVec2`]
///
/// Alignment: Same as the scalar.
#[repr(C)]
#[cfg_attr(
    all(not(target_arch = "wasm32"), feature = "wasmtime"),
    derive(
        wasmtime::component::ComponentType,
        wasmtime::component::Lower,
        wasmtime::component::Lift
    )
)]
#[cfg_attr(
    all(not(target_arch = "wasm32"), feature = "wasmtime"),
    component(record)
)]
pub struct Vector2<U: Unit = f32> {
    /// X coordinate
    pub x: U::Scalar,
    /// Y coordinate
    pub y: U::Scalar,
}

/// SAFETY: `T::Scalar` is `Zeroable`, and `Vector2` is `#[repr(C)]`.
unsafe impl<T: Unit> Zeroable for Vector2<T> {}
/// SAFETY: `T::Scalar` is `Pod`.
unsafe impl<T: Unit> Pod for Vector2<T> {}
/// SAFETY: These are guaranteed to have the same representation.
unsafe impl<T: Unit> Transparent for Vector2<T> {
    type Wrapped = <T::Scalar as Scalar>::Vec2;
}

/// 3D vector.
///
/// Alignment: Same as the scalar (so not 16 bytes). If you really need 16-byte
/// alignment, use [`Vector4`].
#[cfg_attr(
    all(not(target_arch = "wasm32"), feature = "wasmtime"),
    derive(
        wasmtime::component::ComponentType,
        wasmtime::component::Lower,
        wasmtime::component::Lift
    )
)]
#[cfg_attr(
    all(not(target_arch = "wasm32"), feature = "wasmtime"),
    component(record)
)]
#[repr(C)]
pub struct Vector3<U: Unit = f32> {
    /// X coordinate
    pub x: U::Scalar,
    /// Y coordinate
    pub y: U::Scalar,
    /// Z coordinate
    pub z: U::Scalar,
}

/// SAFETY: `T::Scalar` is `Zeroable`, and `Vector3` is `#[repr(C)]`.
unsafe impl<T: Unit> Zeroable for Vector3<T> {}
/// SAFETY: `T::Scalar` is `Pod`.
unsafe impl<T: Unit> Pod for Vector3<T> {}
/// SAFETY: These are guaranteed to have the same representation.
unsafe impl<T: Unit> Transparent for Vector3<T> {
    type Wrapped = <T::Scalar as Scalar>::Vec3;
}

/// 4D vector.
///
/// # Alignment
///
/// This is always 16-byte aligned. [`glam::DVec4`] is only 8-byte aligned (for some reason), and integer vectors are
/// only 4-byte aligned, which means that reference-casting from those glam types to `Vector4` type will fail (but not
/// the other way around - see [`Vector4::as_raw()`]).
///
/// This also means that smaller integer types (i16 etc.) will be over-aligned, consuming much more memory.
#[cfg_attr(
    not(any(feature = "scalar-math", target_arch = "spirv")),
    repr(C, align(16))
)]
#[cfg_attr(
    all(not(target_arch = "wasm32"), feature = "wasmtime"),
    derive(
        wasmtime::component::ComponentType,
        wasmtime::component::Lower,
        wasmtime::component::Lift
    )
)]
#[cfg_attr(
    all(not(target_arch = "wasm32"), feature = "wasmtime"),
    component(record)
)]
pub struct Vector4<U: Unit = f32> {
    /// X coordinate
    pub x: U::Scalar,
    /// Y coordinate
    pub y: U::Scalar,
    /// Z coordinate
    pub z: U::Scalar,
    /// W coordinate
    pub w: U::Scalar,
}

/// SAFETY: `T::Scalar` is `Zeroable`, and `Vector4` is `#[repr(C)]`.
unsafe impl<T: Unit> Zeroable for Vector4<T> {}
/// SAFETY: `T::Scalar` is `Pod`.
unsafe impl<T: Unit> Pod for Vector4<T> {}
/// SAFETY: These are guaranteed to have the same representation.
unsafe impl<T: Unit> Transparent for Vector4<T> {
    type Wrapped = <T::Scalar as Scalar>::Vec4;
}

macro_rules! vector_conversion_to_other_units {
    ($point_ty:ident $(, $size_ty:ident)?) => {
        #[doc = "Instantiate from point."]
        #[inline]
        #[must_use]
        pub fn from_point(point: $point_ty<T>) -> Self {
            crate::rewrap(point)
        }

        #[doc = "Convert to point."]
        #[inline]
        #[must_use]
        pub fn to_point(self) -> $point_ty<T> {
            crate::rewrap(self)
        }

        #[doc = "Reinterpret as point."]
        #[inline]
        #[must_use]
        pub fn as_point(&self) -> &$point_ty<T> {
            bytemuck::cast_ref(self)
        }

        #[doc = "Reinterpret as point."]
        #[inline]
        #[must_use]
        pub fn as_point_mut(&mut self) -> &mut $point_ty<T> {
            bytemuck::cast_mut(self)
        }

        $(
            #[doc = "Instantiate from size."]
            #[inline]
            #[must_use]
            pub fn from_size(size: $size_ty<T>) -> Self {
                crate::rewrap(size)
            }

            #[doc = "Convert to size."]
            #[inline]
            #[must_use]
            pub fn to_size(self) -> $size_ty<T> {
                crate::rewrap(self)
            }

            #[doc = "Reinterpret as size."]
            #[inline]
            #[must_use]
            pub fn as_size(&self) -> &$size_ty<T> {
                bytemuck::cast_ref(self)
            }

            #[doc = "Reinterpret as size."]
            #[inline]
            #[must_use]
            pub fn as_size_mut(&mut self) -> &mut $size_ty<T> {
                bytemuck::cast_mut(self)
            }
        )*
    };
}

crate::impl_ops::vector_ops!(Vector2);
crate::impl_ops::vector_ops!(Vector3);
crate::impl_ops::vector_ops!(Vector4);

impl<T: Unit> Vector2<T> {
    /// New vector.
    pub const fn new(x: T::Scalar, y: T::Scalar) -> Self {
        Self { x, y }
    }

    vector_conversion_to_other_units!(Point2, Size2);
}

crate::impl_vectorlike::vectorlike!(Vector2, 2);

impl<T: Unit> Vector3<T> {
    /// New vector.
    pub const fn new(x: T::Scalar, y: T::Scalar, z: T::Scalar) -> Self {
        Self { x, y, z }
    }

    vector_conversion_to_other_units!(Point3, Size3);
}
crate::impl_vectorlike::vectorlike!(Vector3, 3);

impl<T: Unit<Scalar = f32>> Vector3<T> {
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

impl<T: Unit> Vector4<T> {
    /// New vector.
    pub const fn new(x: T::Scalar, y: T::Scalar, z: T::Scalar, w: T::Scalar) -> Self {
        Self { x, y, z, w }
    }

    vector_conversion_to_other_units!(Point4);
}

crate::impl_vectorlike::vectorlike!(Vector4, 4);

impl<T: Unit> From<glam::BVec2> for Vector2<T> {
    #[inline(always)]
    fn from(v: glam::BVec2) -> Self {
        wrap(<<T::Scalar as Scalar>::Vec2>::from(v))
    }
}

impl<T: Unit> From<glam::BVec3> for Vector3<T> {
    #[inline(always)]
    fn from(v: glam::BVec3) -> Self {
        wrap(<<T::Scalar as Scalar>::Vec3>::from(v))
    }
}

impl<T: Unit> From<glam::BVec4> for Vector4<T> {
    #[inline(always)]
    fn from(v: glam::BVec4) -> Self {
        wrap(<<T::Scalar as Scalar>::Vec4>::from(v))
    }
}

impl<T> Mul<Vector3<T>> for glam::Quat
where
    T: Unit<Scalar = f32>,
{
    type Output = Vector3<T>;

    #[inline]
    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        wrap(self * peel(rhs))
    }
}

impl<T> Mul<Vector3<T>> for glam::DQuat
where
    T: Unit<Scalar = f64>,
{
    type Output = Vector3<T>;

    #[inline]
    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        wrap(self * peel(rhs))
    }
}

impl<'a, T: Unit> Sum<&'a Vector2<T>> for Vector2<T> {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        wrap(iter.map(peel_ref).sum())
    }
}

impl<'a, T: Unit> Sum<&'a Vector3<T>> for Vector3<T> {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        wrap(iter.map(peel_ref).sum())
    }
}

impl<'a, T: Unit> Sum<&'a Vector4<T>> for Vector4<T> {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        wrap(iter.map(peel_ref).sum())
    }
}

impl<'a, T: Unit> Product<&'a Vector2<T>> for Vector2<T> {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        wrap(iter.map(peel_ref).product())
    }
}

impl<'a, T: Unit> Product<&'a Vector3<T>> for Vector3<T> {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        wrap(iter.map(peel_ref).product())
    }
}

impl<'a, T: Unit> Product<&'a Vector4<T>> for Vector4<T> {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        wrap(iter.map(peel_ref).product())
    }
}

impl<T: Unit> core::fmt::Debug for Vector2<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("Vector2")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}
impl<T: Unit> core::fmt::Debug for Vector3<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("Vector3")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .finish()
    }
}
impl<T: Unit> core::fmt::Debug for Vector4<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("Vector4")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .field("w", &self.w)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, RelativeEq, UlpsEq};

    use crate::{vec2, vec3, vec4, vector, Angle, AngleConsts};

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
        use core::ops::{Add, Div, Sub};
        check_splat_ints!(2, div(1), 2);
        check_splat_ints!(10, div(2), 5);
        check_splat!(2.0, mul(2.5), 2.0 * 2.5);
        check_splat_ints!(10, mul(2), 20);
        check_splat_ints!(2, add(1), 3);
        check_splat_ints!(10, add(2), 12);
        check_splat_ints!(2, sub(1), 1);
        check_splat_ints!(10, sub(2), 8);

        let mut v: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        v *= Vector4::splat(2.0);
        assert_eq!(v, vec4!(2.0, 4.0, 6.0, 8.0));
        v /= Vector4::splat(2.0);
        assert_eq!(v, vec4!(1.0, 2.0, 3.0, 4.0));
        v += Vector4::splat(1.0);
        assert_eq!(v, vec4!(2.0, 3.0, 4.0, 5.0));
        v -= Vector4::splat(2.0);
        assert_eq!(v, vec4!(0.0, 1.0, 2.0, 3.0));
        v %= Vector4::splat(2.0);
        assert_eq!(v, vec4!(0.0, 1.0, 0.0, 1.0));
    }

    #[test]
    fn sum2() {
        let a: Vector2<f32> = vector!(1.0, 2.0);
        let b: Vector2<f32> = vector!(1.0, 2.0);
        let c: Vector2<f32> = vector!(1.0, 2.0);
        let d: Vector2<f32> = vector!(1.0, 2.0);
        let sum: Vector2<f32> = [a, b, c, d].iter().sum();
        assert_eq!(sum, (4.0, 8.0));
    }

    #[test]
    fn sum3() {
        let a: Vector3<f32> = vector!(1.0, 2.0, 3.0);
        let b: Vector3<f32> = vector!(1.0, 2.0, 3.0);
        let c: Vector3<f32> = vector!(1.0, 2.0, 3.0);
        let d: Vector3<f32> = vector!(1.0, 2.0, 3.0);
        let sum: Vector3<f32> = [a, b, c, d].iter().sum();
        assert_eq!(sum, vec3!(4.0, 8.0, 12.0));
    }

    #[test]
    fn sum4() {
        let a: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let b: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let c: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let d: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let sum: Vector4<f32> = [a, b, c, d].iter().sum();
        assert_eq!(sum, vec4!(4.0, 8.0, 12.0, 16.0));
    }

    #[test]
    fn product2() {
        let a: Vector2<f32> = vector!(1.0, 2.0);
        let b: Vector2<f32> = vector!(1.0, 2.0);
        let c: Vector2<f32> = vector!(1.0, 2.0);
        let d: Vector2<f32> = vector!(1.0, 2.0);
        let product: Vector2<f32> = [a, b, c, d].iter().product();
        assert_eq!(product, vec2!(1.0, 16.0));
    }

    #[test]
    fn product3() {
        let a: Vector3<f32> = vector!(1.0, 2.0, 3.0);
        let b: Vector3<f32> = vector!(1.0, 2.0, 3.0);
        let c: Vector3<f32> = vector!(1.0, 2.0, 3.0);
        let d: Vector3<f32> = vector!(1.0, 2.0, 3.0);
        let product: Vector3<f32> = [a, b, c, d].iter().product();
        assert_eq!(product, vec3!(1.0, 16.0, 81.0));
    }

    #[test]
    fn product4() {
        let a: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let b: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let c: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let d: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let product: Vector4<f32> = [a, b, c, d].iter().product();
        assert_eq!(product, vec4!(1.0, 16.0, 81.0, 256.0));
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
    fn units() {
        use crate::vector;
        assert_abs_diff_eq!(Vec2::X, vector!(1.0, 0.0));
        assert_abs_diff_eq!(Vec2::Y, vector!(0.0, 1.0));
        assert_abs_diff_eq!(Vec3::X, vector!(1.0, 0.0, 0.0));
        assert_abs_diff_eq!(Vec3::Y, vector!(0.0, 1.0, 0.0));
        assert_abs_diff_eq!(Vec3::Z, vector!(0.0, 0.0, 1.0));
        assert_abs_diff_eq!(Vec4::X, vector!(1.0, 0.0, 0.0, 0.0));
        assert_abs_diff_eq!(Vec4::Y, vector!(0.0, 1.0, 0.0, 0.0));
        assert_abs_diff_eq!(Vec4::Z, vector!(0.0, 0.0, 1.0, 0.0));
        assert_abs_diff_eq!(Vec4::W, vector!(0.0, 0.0, 0.0, 1.0));
    }

    #[test]
    fn nan() {
        let v2 = Vec2::new(1.0, f32::NAN);
        assert!(v2.is_nan());
        assert!(!v2.is_finite());
        assert_eq!(v2.is_nan_mask(), glam::BVec2::new(false, true));
        assert_eq!(v2.is_finite_mask(), glam::BVec2::new(true, false));

        let v3 = Vec3::new(1.0, f32::NAN, f32::INFINITY);
        assert!(v3.is_nan());
        assert!(!v3.is_finite());
        assert_eq!(v3.is_nan_mask(), glam::BVec3::new(false, true, false));
        assert_eq!(v3.is_finite_mask(), glam::BVec3::new(true, false, false));

        let v4 = Vec4::new(1.0, 2.0, f32::NAN, f32::INFINITY);
        assert!(v4.is_nan());
        assert!(!v4.is_finite());
        assert_eq!(
            v4.is_nan_mask(),
            glam::BVec4::new(false, false, true, false)
        );
        assert_eq!(
            v4.is_finite_mask(),
            glam::BVec4::new(true, true, false, false)
        );

        assert!(Vec2::NAN.is_nan());
        assert!(Vec3::NAN.is_nan());
        assert!(Vec4::NAN.is_nan());

        // Replace NaNs with zeroes.
        let v = Vec4::select(v4.is_nan_mask(), Vec4::ZERO, v4);
        assert_eq!(v, vec4!(1.0, 2.0, 0.0, f32::INFINITY));
    }

    #[test]
    fn swizzle2() {
        assert_eq!(Vec2::X.swizzle::<1, 0>(), (0.0, 1.0));
        assert_eq!(
            Vec3::new(1.0, 2.0, 3.0).swizzle2::<1, 0>(),
            Vec2::new(2.0, 1.0)
        );
    }

    #[test]
    fn swizzle3() {
        assert_eq!(
            Vec3::new(1.0, 2.0, 3.0).swizzle::<2, 1, 0>(),
            vec3!(3.0, 2.0, 1.0)
        );
        assert_eq!(Vec2::X.swizzle3::<1, 0, 1>(), Vec3::new(0.0, 1.0, 0.0));
        assert_eq!(
            Vec4::new(1.0, 2.0, 3.0, 4.0).swizzle3::<3, 2, 1>(),
            Vec3::new(4.0, 3.0, 2.0)
        );
    }

    #[test]
    fn swizzle4() {
        assert_eq!(
            Vec4::new(0.0, 1.0, 2.0, 3.0).swizzle::<3, 2, 1, 0>(),
            vec4!(3.0, 2.0, 1.0, 0.0)
        );

        assert_eq!(
            Vec2::X.swizzle4::<1, 0, 1, 0>(),
            Vec4::new(0.0, 1.0, 0.0, 1.0)
        );
        assert_eq!(
            Vec3::new(0.0, 1.0, 2.0).swizzle4::<2, 1, 0, 2>(),
            Vec4::new(2.0, 1.0, 0.0, 2.0)
        );
    }

    #[test]
    fn to_3d() {
        let v = Vec2 { x: 3.0, y: 4.0 };
        assert_eq!(v.extend(2.0), crate::vec3!(3.0, 4.0, 2.0));
    }

    #[test]
    fn to_4d() {
        assert_eq!(Vec3::Z.extend(2.0), Vec4::new(0.0, 0.0, 1.0, 2.0));
    }

    #[test]
    fn vec3a() {
        let a: glam::Vec3A = Vec3::new(0.0, 1.0, 2.0).to_vec3a();
        assert_eq!(a, glam::Vec3A::new(0.0, 1.0, 2.0));
        let b = Vec3::from_vec3a(a);
        assert_eq!(b, Vec3::new(0.0, 1.0, 2.0));
    }

    #[test]
    fn scaling_by_scalar() {
        // Test that vector types can be multiplied/divided by their
        // (unsplatted) scalar. This doesn't work in generic code, but it should
        // work when the concrete vector type is known to the compiler.

        {
            let x: Vec4 = vec4!(1.0, 2.0, 3.0, 4.0);

            let mut a = x;
            let mut b = x;
            let mut c = x;
            a *= 2.0;
            b /= 2.0;
            c %= 2.0;
            assert_eq!(a, vec4!(2.0, 4.0, 6.0, 8.0));
            assert_eq!(b, vec4!(0.5, 1.0, 1.5, 2.0));
            assert_eq!(c, vec4!(1.0, 0.0, 1.0, 0.0));
        }
        {
            let x: DVec4 = vec4!(1.0, 2.0, 3.0, 4.0);

            let a = x * 2.0;
            let b = x / 2.0;

            assert_eq!(a, vec4!(2.0, 4.0, 6.0, 8.0));
            assert_eq!(b, vec4!(0.5, 1.0, 1.5, 2.0));
        }
        {
            let x: IVec4 = vec4!(1, 2, 3, 4);

            let a = x * 2;
            let b = x / 2;

            assert_eq!(a, vec4!(2, 4, 6, 8));
            assert_eq!(b, vec4!(0, 1, 1, 2));
        }
        {
            let x: UVec4 = vec4!(1, 2, 3, 4);

            let a = x * 2;
            let b = x / 2;

            assert_eq!(a, vec4!(2, 4, 6, 8));
            assert_eq!(b, vec4!(0, 1, 1, 2));
        }
    }

    #[test]
    fn ops_by_vector_ref() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let added = a + b;
        let subtracted = a - b;
        let multiplied = a * b;
        let divided = a / b;

        assert_eq!(a + &b, added);
        assert_eq!(&a + b, added);
        assert_eq!(&a + &b, added);
        assert_eq!(a - &b, subtracted);
        assert_eq!(&a - b, subtracted);
        assert_eq!(&a - &b, subtracted);
        assert_eq!(a * &b, multiplied);
        assert_eq!(&a * b, multiplied);
        assert_eq!(&a * &b, multiplied);
        assert_eq!(a / &b, divided);
        assert_eq!(&a / b, divided);
        assert_eq!(&a / &b, divided);

        let mut a2 = a;
        a2 += &b;
        assert_eq!(a2, added);
        let mut a2 = a;
        a2 -= &b;
        assert_eq!(a2, subtracted);
        let mut a2 = a;
        a2 *= &b;
        assert_eq!(a2, multiplied);
        let mut a2 = a;
        a2 /= &b;
        assert_eq!(a2, divided);
    }

    #[test]
    fn ops_by_scalar_ref() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = 2.0;
        let added = a + b;
        let subtracted = a - b;
        let multiplied = a * b;
        let divided = a / b;

        assert_eq!(a + &b, added);
        assert_eq!(&a + b, added);
        assert_eq!(&a + &b, added);
        assert_eq!(a - &b, subtracted);
        assert_eq!(&a - b, subtracted);
        assert_eq!(&a - &b, subtracted);
        assert_eq!(a * &b, multiplied);
        assert_eq!(&a * b, multiplied);
        assert_eq!(&a * &b, multiplied);
        assert_eq!(a / &b, divided);
        assert_eq!(&a / b, divided);
        assert_eq!(&a / &b, divided);

        let mut a2 = a;
        a2 += &b;
        assert_eq!(a2, added);
        let mut a2 = a;
        a2 -= &b;
        assert_eq!(a2, subtracted);
        let mut a2 = a;
        a2 *= &b;
        assert_eq!(a2, multiplied);
        let mut a2 = a;
        a2 /= &b;
        assert_eq!(a2, divided);
    }

    #[test]
    fn map() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = a.map(|x| x * 2.0);
        assert_eq!(b, vec4![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn cmp() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        let b = Vec4::new(4.0, 2.0, 1.0, 3.0);

        let eq = a.cmpeq(b);
        let ne = a.cmpne(b);
        let lt = a.cmplt(b);
        let le = a.cmple(b);
        let gt = a.cmpgt(b);
        let ge = a.cmpge(b);

        assert_eq!(eq, glam::BVec4::new(false, true, false, false));
        assert_eq!(ne, glam::BVec4::new(true, false, true, true));
        assert_eq!(lt, glam::BVec4::new(true, false, false, false));
        assert_eq!(le, glam::BVec4::new(true, true, false, false));
        assert_eq!(gt, glam::BVec4::new(false, false, true, true));
        assert_eq!(ge, glam::BVec4::new(false, true, true, true));

        assert_eq!(a.min(b), vec4![1.0, 2.0, 1.0, 3.0]);
        assert_eq!(a.max(b), vec4![4.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.min_element(), 1.0);
        assert_eq!(a.max_element(), 4.0);
    }

    #[test]
    fn cmp_f64() {
        let a = Vector4::<f64>::new(1.0, 2.0, 3.0, 4.0);
        let b = Vector4::<f64>::new(4.0, 2.0, 1.0, 3.0);

        let eq = a.cmpeq(b);
        let ne = a.cmpne(b);
        let lt = a.cmplt(b);
        let le = a.cmple(b);
        let gt = a.cmpgt(b);
        let ge = a.cmpge(b);

        assert_eq!(eq, glam::BVec4::new(false, true, false, false));
        assert_eq!(ne, glam::BVec4::new(true, false, true, true));
        assert_eq!(lt, glam::BVec4::new(true, false, false, false));
        assert_eq!(le, glam::BVec4::new(true, true, false, false));
        assert_eq!(gt, glam::BVec4::new(false, false, true, true));
        assert_eq!(ge, glam::BVec4::new(false, true, true, true));

        assert_eq!(a.min(b), vec4![1.0, 2.0, 1.0, 3.0]);
        assert_eq!(a.max(b), vec4![4.0, 2.0, 3.0, 4.0]);
        assert_eq!(a.min_element(), 1.0);
        assert_eq!(a.max_element(), 4.0);
    }

    #[test]
    fn clamp() {
        let a = Vec4::new(1.0, 2.0, 3.0, 4.0);
        assert_eq!(
            a.clamp(Vec4::splat(2.0), Vec4::splat(3.5)),
            Vec4::new(2.0, 2.0, 3.0, 3.5)
        );

        let b = IVec4::new(1, 2, 3, 5);
        assert_eq!(b.clamp(IVec4::splat(2), IVec4::splat(4)), vec4!(2, 2, 3, 4));
    }

    #[test]
    fn dot() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(2.0, 3.0, 4.0);
        assert_abs_diff_eq!(a.dot(b), 20.0);
    }

    #[test]
    fn normalize() {
        let a = Vec3::new(1.0, 1.0, 2.0);
        assert!(!a.is_normalized());
        let d = a.dot(a).sqrt();
        let b = a / Vec3::splat(d);
        assert_abs_diff_eq!(a.normalize(), b);

        assert!(a.normalize().is_normalized());

        let z = Vec3::ZERO;
        assert_eq!(z.normalize_or_zero(), Vec3::ZERO);
    }

    #[test]
    fn length() {
        let v = Vec3::new(1.0, 2.0, 3.0);
        let u = glam::Vec3::new(1.0, 2.0, 3.0);
        assert_eq!(v.length(), u.length());

        assert_abs_diff_eq!(
            v.length_squared(),
            u.length() * u.length(),
            epsilon = 0.00001
        );
    }

    #[test]
    fn exp() {
        let a = Vec3::splat(1.0);
        assert_eq!(a.exp(), Vec3::splat(1.0f32.exp()));
    }

    #[test]
    fn powf() {
        let a = Vec3::splat(2.0);
        assert_eq!(a.powf(2.0), Vec3::splat(4.0));
    }

    #[test]
    fn recip() {
        let a = Vec3::splat(2.0);
        assert_eq!(a.recip(), Vec3::splat(0.5));
    }

    #[test]
    fn mul_add() {
        let a = Vec3::new(1.0, 2.0, 3.0);
        let b = Vec3::new(2.0, 3.0, 4.0);
        let c = Vec3::new(3.0, 4.0, 5.0);
        assert_eq!(a.mul_add(b, c), a * b + c);
    }

    #[test]
    fn abs() {
        let a = Vec3::new(1.0, -2.0, -3.0);
        assert_eq!(a.abs(), vec3!(1.0, 2.0, 3.0));
    }

    #[test]
    fn signum() {
        let a = Vec3::new(1.0, -2.0, -3.0);
        assert_eq!(a.signum(), vec3!(1.0, -1.0, -1.0));
    }

    #[test]
    fn rotate() {
        use crate::Angle;
        use approx::assert_abs_diff_eq;

        let v = Vector3::<f32>::X;
        let quat = Angle::from_degrees(180.0f32).to_rotation(Vector3::Z);
        assert_abs_diff_eq!(quat * v, -v);

        let v = Vector3::<f64>::X;
        let quat = Angle::from_degrees(180.0f64).to_rotation(Vector3::Z);
        assert_abs_diff_eq!(quat * v, -v);
    }

    #[test]
    fn rotate2() {
        let a = Vector2::<f32>::X;
        let rotate_by = Angle::<f32>::FRAG_PI_2;
        let b = Vector2::<f32>::from_angle(rotate_by);
        let rotated = a.rotate(b);
        assert_abs_diff_eq!(rotated, Vector2::<f32>::Y);

        let x = glam::Vec2::X;
        let rotate_by = glam::Vec2::from_angle(f32::FRAG_PI_2);
        let y = x.rotate(rotate_by);
        assert_abs_diff_eq!(peel(rotated), y);
    }

    #[test]
    fn matrix_mul_custom_unit() {
        use crate::{vec3, Matrix3};
        let mat = Matrix3::<f32>::IDENTITY;
        let a: Vector3<F32> = vec3!(20.0, 30.0, 1.0);
        let b: Vector3<F32> = mat * a;
        assert_eq!(b, vec3!(20.0, 30.0, 1.0));
    }

    #[test]
    fn array_interface() {
        let mut vec = Vector4::<i64>::new(1, 2, 3, 4);
        assert_eq!(vec.get(0), 1);
        assert_eq!(vec.get(1), 2);
        assert_eq!(vec.get(2), 3);
        assert_eq!(vec.get(3), 4);

        let array_ref: &[i64; 4] = vec.as_ref();
        assert_eq!(array_ref, &[1i64, 2, 3, 4]);

        let array_mut: &mut [i64; 4] = vec.as_mut();
        assert_eq!(array_mut, &[1i64, 2, 3, 4]);

        let mut array: [i64; 4] = vec.into();
        assert_eq!(array, [1i64, 2, 3, 4]);
        vec.set(3, 5);
        vec.write_to_slice(&mut array);
        assert_eq!(array, [1i64, 2, 3, 5]);
    }

    #[test]
    fn untyped_mut() {
        let mut vec = Vector4::<I32>::new(1, 2, 3, 4);
        let vec2: &mut Vector4<i32> = vec.as_untyped_mut();
        vec2.y = 100;
        assert_eq!(vec, vec4![1, 100, 3, 4]);

        let vec3: &mut Vector4<i32> = vec.cast_mut();
        vec3.z = 100;
        assert_eq!(vec, vec4![1, 100, 100, 4]);
    }

    #[test]
    fn from_bools() {
        assert_eq!(
            Vector2::<f32>::from(glam::BVec2::new(true, false)),
            vec2!(1.0, 0.0)
        );
        assert_eq!(
            Vector3::<f32>::from(glam::BVec3::new(true, false, true)),
            vec3!(1.0, 0.0, 1.0)
        );
        assert_eq!(
            Vector4::<f32>::from(glam::BVec4::new(true, false, true, false)),
            vec4!(1.0, 0.0, 1.0, 0.0)
        );
        assert_eq!(
            Vector4::<i32>::from(glam::BVec4::new(true, false, true, false)),
            vec4!(1, 0, 1, 0)
        );
    }

    #[test]
    fn glam_reference_conversion() {
        use core::borrow::{Borrow, BorrowMut};

        let mut vec = Vector4::<I32>::new(1, 2, 3, 4);

        let vec1: &glam::IVec4 = vec.as_ref();
        assert_eq!(*vec1, glam::IVec4::new(1, 2, 3, 4));

        let vec2: &mut glam::IVec4 = vec.as_mut();
        vec2.y = 100;
        assert_eq!(vec, vec4![1, 100, 3, 4]);

        let vec3: &glam::IVec4 = vec.borrow();
        assert_eq!(*vec3, glam::IVec4::new(1, 100, 3, 4));

        let vec4: &mut glam::IVec4 = vec.borrow_mut();
        vec4.z = 100;
        assert_eq!(*vec4, glam::IVec4::new(1, 100, 100, 4));
    }

    fn hash_one<T: core::hash::Hash, H: core::hash::BuildHasher>(
        build_hasher: &H,
        value: T,
    ) -> u64 {
        use core::hash::Hasher;
        let mut h = build_hasher.build_hasher();
        value.hash(&mut h);
        h.finish()
    }

    #[derive(Default)]
    struct PoorHasher(u64);

    impl core::hash::Hasher for PoorHasher {
        fn finish(&self) -> u64 {
            self.0
        }

        fn write(&mut self, bytes: &[u8]) {
            for byte in bytes {
                self.0 = self.0.rotate_left(4) ^ u64::from(*byte);
            }
        }
    }

    #[test]
    fn hash_equality() {
        let hasher = core::hash::BuildHasherDefault::<PoorHasher>::default();
        let h1 = hash_one(&hasher, Vector2::<i32> { x: 123, y: 456 });
        let h2 = hash_one(&hasher, glam::IVec2::new(123, 456));
        assert_eq!(h1, h2);
    }

    #[test]
    fn gaslight_coverage() {
        extern crate alloc;

        let v: Vector2<f32> = vec2![1.0, 2.0];
        assert_eq!(v.to_point(), crate::point2!(1.0, 2.0));

        _ = alloc::format!("{:?}", Vector2::<f32>::default());
        _ = alloc::format!("{:?}", Vector3::<f32>::default());
        _ = alloc::format!("{:?}", Vector4::<f32>::default());

        let _: (f32, f32) = Vector2::<f32>::ZERO.into();
        let _: (f32, f32) = Vector2::<f32>::ZERO.to_tuple();
        let _: (f32, f32, f32) = Vector3::<f32>::ZERO.into();
        let _: (f32, f32, f32) = Vector3::<f32>::ZERO.to_tuple();
        let _: (f32, f32, f32, f32) = Vector4::<f32>::ZERO.into();
        let _: (f32, f32, f32, f32) = Vector4::<f32>::ZERO.to_tuple();

        _ = Vector2::<f32>::from((0.0, 0.0));
        _ = Vector3::<f32>::from((0.0, 0.0, 0.0));
        _ = Vector4::<f32>::from((0.0, 0.0, 0.0, 0.0));
        _ = Vector2::<f32>::from_tuple((0.0, 0.0));
        _ = Vector3::<f32>::from_tuple((0.0, 0.0, 0.0));
        _ = Vector4::<f32>::from_tuple((0.0, 0.0, 0.0, 0.0));

        assert_eq!(
            Vector2::<f32>::default_max_relative(),
            f32::default_max_relative()
        );
        assert_eq!(Vector2::<f32>::default_max_ulps(), f32::default_max_ulps());
    }
}

#[cfg(all(test, feature = "serde"))]
mod serde_tests {
    use super::*;

    #[test]
    fn serde_vector() {
        let vec = Vector4::<f32>::new(10.0, 20.0, 30.0, 40.0);
        let serialized = serde_json::to_string(&vec).unwrap();
        assert_eq!(serialized, r#"{"x":10.0,"y":20.0,"z":30.0,"w":40.0}"#);
        let deserialized: Vector4<f32> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(vec, deserialized);
    }
}

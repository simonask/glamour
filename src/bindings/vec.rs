#![allow(missing_docs, clippy::return_self_not_must_use)]

use core::ops::Neg;

use crate::Scalar;

use super::*;

/// Basic `glam` types used to back strongly typed vectors.
#[allow(missing_docs)]
pub trait Vector:
    PodValue
    + Mul<Self::Scalar, Output = Self>
    + MulAssign<Self::Scalar>
    + Div<Self::Scalar, Output = Self>
    + DivAssign<Self::Scalar>
    + Rem<Self::Scalar, Output = Self>
    + RemAssign<Self::Scalar>
    + Add<Self, Output = Self>
    + AddAssign<Self>
    + Sub<Self, Output = Self>
    + SubAssign<Self>
    + Mul<Self, Output = Self>
    + MulAssign<Self>
    + Div<Self, Output = Self>
    + DivAssign<Self>
    + Rem<Self, Output = Self>
    + RemAssign<Self>
    + for<'a> core::iter::Sum<&'a Self>
    + for<'a> core::iter::Product<&'a Self>
{
    /// The component type of this `glam` vector.
    type Scalar: Scalar;
    crate::interfaces::vector_base_interface!(trait_decl);
}

macro_rules! impl_vector {
    ($glam_ty:ty, $scalar:ty, $mask:ident) => {
        impl Vector for $glam_ty {
            type Scalar = $scalar;
            crate::interfaces::vector_base_interface!(trait_impl);
        }
    };
}

pub trait SignedVector: Vector + Neg<Output = Self> {
    crate::interfaces::vector_signed_interface!(trait_decl);
}

macro_rules! impl_signed_vector {
    ($glam_ty:ty) => {
        impl SignedVector for $glam_ty {
            crate::interfaces::vector_signed_interface!(trait_impl);
        }
    };
}

pub trait IntegerVector: Vector + core::hash::Hash {
    crate::interfaces::vector_integer_interface!(trait_decl);
}

macro_rules! impl_integer_vector {
    ($glam_ty:ty) => {
        impl IntegerVector for $glam_ty {
            crate::interfaces::vector_integer_interface!(trait_impl);
        }
    };
}

pub trait FloatVector:
    SignedVector
    + approx::AbsDiffEq<Epsilon = Self::Scalar>
    + approx::UlpsEq
    + approx::RelativeEq<Epsilon = Self::Scalar>
{
    crate::interfaces::vector_float_base_interface!(trait_decl);
}

macro_rules! impl_float_vector {
    ($glam_ty:ty, $scalar:ty) => {
        impl FloatVector for $glam_ty {
            crate::interfaces::vector_float_base_interface!(trait_impl);
        }
    };
}

pub trait Vector2: Vector + From<glam::BVec2> {
    crate::interfaces::vector2_base_interface!(trait_decl);
}

macro_rules! impl_vector2 {
    ($glam_ty:ty, $vec3_ty:ty) => {
        impl Vector2 for $glam_ty {
            crate::interfaces::vector2_base_interface!(trait_impl);
        }
    };
}

pub trait Vector3: Vector + From<glam::BVec3> {
    crate::interfaces::vector3_base_interface!(trait_decl);
}

macro_rules! impl_vector3 {
    ($glam_ty:ty, $vec2_ty:ty, $vec4_ty:ty) => {
        impl Vector3 for $glam_ty {
            crate::interfaces::vector3_base_interface!(trait_impl);
        }
    };
}

pub trait Vector4: Vector + From<glam::BVec4> {
    crate::interfaces::vector4_base_interface!(trait_decl);
}

macro_rules! impl_vector4 {
    ($glam_ty:ty, $vec3_ty:ty) => {
        impl Vector4 for $glam_ty {
            crate::interfaces::vector4_base_interface!(trait_impl);
        }
    };
}

pub trait SignedVector2: SignedVector + Vector2 {
    crate::interfaces::vector2_signed_interface!(trait_decl);
}

pub trait FloatVector2: SignedVector2 + FloatVector {
    crate::interfaces::vector2_float_interface!(trait_decl);
}

pub trait FloatVector3: Vector3 + FloatVector {
    crate::interfaces::vector3_float_interface!(trait_decl);
}

pub trait FloatVector4: Vector4 + FloatVector {
    crate::interfaces::vector4_float_interface!(trait_decl);
}

impl_vector!(glam::Vec2, f32, BVec2);
impl_vector!(glam::Vec3, f32, BVec3);
impl_vector!(glam::Vec4, f32, BVec4);
impl_vector!(glam::DVec2, f64, BVec2);
impl_vector!(glam::DVec3, f64, BVec3);
impl_vector!(glam::DVec4, f64, BVec4);
impl_vector!(glam::I16Vec2, i16, BVec2);
impl_vector!(glam::I16Vec3, i16, BVec3);
impl_vector!(glam::I16Vec4, i16, BVec4);
impl_vector!(glam::U16Vec2, u16, BVec2);
impl_vector!(glam::U16Vec3, u16, BVec3);
impl_vector!(glam::U16Vec4, u16, BVec4);
impl_vector!(glam::IVec2, i32, BVec2);
impl_vector!(glam::IVec3, i32, BVec3);
impl_vector!(glam::IVec4, i32, BVec4);
impl_vector!(glam::UVec2, u32, BVec2);
impl_vector!(glam::UVec3, u32, BVec3);
impl_vector!(glam::UVec4, u32, BVec4);
impl_vector!(glam::I64Vec2, i64, BVec2);
impl_vector!(glam::I64Vec3, i64, BVec3);
impl_vector!(glam::I64Vec4, i64, BVec4);
impl_vector!(glam::U64Vec2, u64, BVec2);
impl_vector!(glam::U64Vec3, u64, BVec3);
impl_vector!(glam::U64Vec4, u64, BVec4);

impl_vector2!(glam::Vec2, glam::Vec3);
impl_vector2!(glam::DVec2, glam::DVec3);
impl_vector2!(glam::I16Vec2, glam::I16Vec3);
impl_vector2!(glam::U16Vec2, glam::U16Vec3);
impl_vector2!(glam::IVec2, glam::IVec3);
impl_vector2!(glam::UVec2, glam::UVec3);
impl_vector2!(glam::I64Vec2, glam::I64Vec3);
impl_vector2!(glam::U64Vec2, glam::U64Vec3);

impl_vector3!(glam::Vec3, glam::Vec2, glam::Vec4);
impl_vector3!(glam::DVec3, glam::DVec2, glam::DVec4);
impl_vector3!(glam::I16Vec3, glam::I16Vec2, glam::I16Vec4);
impl_vector3!(glam::U16Vec3, glam::U16Vec2, glam::U16Vec4);
impl_vector3!(glam::IVec3, glam::IVec2, glam::IVec4);
impl_vector3!(glam::UVec3, glam::UVec2, glam::UVec4);
impl_vector3!(glam::I64Vec3, glam::I64Vec2, glam::I64Vec4);
impl_vector3!(glam::U64Vec3, glam::U64Vec2, glam::U64Vec4);

impl_vector4!(glam::DVec4, glam::DVec3);
impl_vector4!(glam::I16Vec4, glam::I16Vec3);
impl_vector4!(glam::U16Vec4, glam::U16Vec3);
impl_vector4!(glam::IVec4, glam::IVec3);
impl_vector4!(glam::UVec4, glam::UVec3);
impl_vector4!(glam::I64Vec4, glam::I64Vec3);
impl_vector4!(glam::U64Vec4, glam::U64Vec3);

impl_signed_vector!(glam::Vec2);
impl_signed_vector!(glam::Vec3);
impl_signed_vector!(glam::Vec4);
impl_signed_vector!(glam::DVec2);
impl_signed_vector!(glam::DVec3);
impl_signed_vector!(glam::DVec4);
impl_signed_vector!(glam::I16Vec2);
impl_signed_vector!(glam::I16Vec3);
impl_signed_vector!(glam::I16Vec4);
impl_signed_vector!(glam::IVec2);
impl_signed_vector!(glam::IVec3);
impl_signed_vector!(glam::IVec4);
impl_signed_vector!(glam::I64Vec2);
impl_signed_vector!(glam::I64Vec3);
impl_signed_vector!(glam::I64Vec4);

impl_integer_vector!(glam::U16Vec2);
impl_integer_vector!(glam::U16Vec3);
impl_integer_vector!(glam::U16Vec4);
impl_integer_vector!(glam::UVec2);
impl_integer_vector!(glam::UVec3);
impl_integer_vector!(glam::UVec4);
impl_integer_vector!(glam::U64Vec2);
impl_integer_vector!(glam::U64Vec3);
impl_integer_vector!(glam::U64Vec4);
impl_integer_vector!(glam::I16Vec2);
impl_integer_vector!(glam::I16Vec3);
impl_integer_vector!(glam::I16Vec4);
impl_integer_vector!(glam::IVec2);
impl_integer_vector!(glam::IVec3);
impl_integer_vector!(glam::IVec4);
impl_integer_vector!(glam::I64Vec2);
impl_integer_vector!(glam::I64Vec3);
impl_integer_vector!(glam::I64Vec4);

impl_float_vector!(glam::Vec2, f32);
impl_float_vector!(glam::Vec3, f32);
impl_float_vector!(glam::Vec4, f32);
impl_float_vector!(glam::DVec2, f64);
impl_float_vector!(glam::DVec3, f64);
impl_float_vector!(glam::DVec4, f64);

impl Vector4 for glam::Vec4 {
    // Note: Manual impl because of the `BVec4A` discrepancy.
    crate::forward_fn!(trait_impl => fn from_array(array: [scalar; 4]) -> Self);
    crate::forward_fn!(trait_impl => fn to_array(&self) -> [scalar; 4]);
    crate::forward_fn!(trait_impl => fn truncate(self) -> vec3);
    crate::forward_fn!(trait_impl => fn with_z(self, z: scalar) -> Self);
    crate::forward_fn!(trait_impl => fn with_w(self, w: scalar) -> Self);

    fn cmpeq(self, other: Self) -> glam::BVec4 {
        bvec4a_to_bvec4(glam::Vec4::cmpeq(self, other))
    }
    fn cmpne(self, other: Self) -> glam::BVec4 {
        bvec4a_to_bvec4(glam::Vec4::cmpne(self, other))
    }
    fn cmpge(self, other: Self) -> glam::BVec4 {
        bvec4a_to_bvec4(glam::Vec4::cmpge(self, other))
    }
    fn cmpgt(self, other: Self) -> glam::BVec4 {
        bvec4a_to_bvec4(glam::Vec4::cmpgt(self, other))
    }
    fn cmple(self, other: Self) -> glam::BVec4 {
        bvec4a_to_bvec4(glam::Vec4::cmple(self, other))
    }
    fn cmplt(self, other: Self) -> glam::BVec4 {
        bvec4a_to_bvec4(glam::Vec4::cmplt(self, other))
    }
    fn select(mask: glam::BVec4, if_true: Self, if_false: Self) -> Self {
        glam::Vec4::select(bvec4_to_bvec4a(mask), if_true, if_false)
    }
}

impl SignedVector2 for glam::Vec2 {
    crate::interfaces::vector2_signed_interface!(trait_impl);
}
impl SignedVector2 for glam::DVec2 {
    crate::interfaces::vector2_signed_interface!(trait_impl);
}
impl SignedVector2 for glam::I16Vec2 {
    crate::interfaces::vector2_signed_interface!(trait_impl);
}
impl SignedVector2 for glam::IVec2 {
    crate::interfaces::vector2_signed_interface!(trait_impl);
}
impl SignedVector2 for glam::I64Vec2 {
    crate::interfaces::vector2_signed_interface!(trait_impl);
}

impl FloatVector2 for glam::Vec2 {
    crate::interfaces::vector2_float_interface!(trait_impl);
}

impl FloatVector2 for glam::DVec2 {
    crate::interfaces::vector2_float_interface!(trait_impl);
}

impl FloatVector3 for glam::Vec3 {
    crate::interfaces::vector3_float_interface!(trait_impl);
}
impl FloatVector3 for glam::DVec3 {
    crate::interfaces::vector3_float_interface!(trait_impl);
}
impl FloatVector4 for glam::Vec4 {
    fn is_nan_mask(self) -> glam::BVec4 {
        bvec4a_to_bvec4(glam::Vec4::is_nan_mask(self))
    }
}
impl FloatVector4 for glam::DVec4 {
    crate::interfaces::vector4_float_interface!(trait_impl);
}

#[cfg(not(feature = "scalar-math"))]
mod bvec_compat {
    #[inline(always)]
    pub fn bvec4_to_bvec4a(bvec: glam::BVec4) -> glam::BVec4A {
        glam::BVec4A::new(bvec.x, bvec.y, bvec.z, bvec.w)
    }

    #[inline(always)]
    pub fn bvec4a_to_bvec4(bvec: glam::BVec4A) -> glam::BVec4 {
        let bitmask = bvec.bitmask();
        let (x, y, z, w) = (
            bitmask & 0b0001 != 0,
            bitmask & 0b0010 != 0,
            bitmask & 0b0100 != 0,
            bitmask & 0b1000 != 0,
        );
        glam::BVec4 { x, y, z, w }
    }
}

#[cfg(feature = "scalar-math")]
mod bvec_compat {
    #[inline(always)]
    pub fn bvec4_to_bvec4a(bvec: glam::BVec4) -> glam::BVec4 {
        bvec
    }

    #[inline(always)]
    pub fn bvec4a_to_bvec4(bvec: glam::BVec4) -> glam::BVec4 {
        bvec
    }
}

use bvec_compat::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_finite() {
        let a = glam::Vec4::NAN;
        assert!(!FloatVector::is_finite(a));

        let a = glam::DVec4::NAN;
        assert!(!FloatVector::is_finite(a));
    }
}

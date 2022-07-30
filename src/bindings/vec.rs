#![allow(missing_docs, clippy::return_self_not_must_use)]

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
{
    /// The component type of this `glam` vector.
    type Scalar: Scalar;

    fn splat(scalar: Self::Scalar) -> Self;
    fn clamp(self, min: Self, max: Self) -> Self;
    fn min(self, other: Self) -> Self;
    fn max(self, other: Self) -> Self;
    fn min_element(self) -> Self::Scalar;
    fn max_element(self) -> Self::Scalar;
    fn dot(self, other: Self) -> Self::Scalar;
    fn write_to_slice(self, slice: &mut [Self::Scalar]);
}

macro_rules! impl_vector {
    ($glam_ty:ty, $scalar:ty) => {
        impl Vector for $glam_ty {
            type Scalar = $scalar;
            forward_impl!($glam_ty => fn splat(scalar: Self::Scalar) -> Self);
            forward_impl!($glam_ty => fn clamp(self, min: Self, max: Self) -> Self);
            forward_impl!($glam_ty => fn min(self, other: Self) -> Self);
            forward_impl!($glam_ty => fn max(self, other: Self) -> Self);
            forward_impl!($glam_ty => fn min_element(self) -> Self::Scalar);
            forward_impl!($glam_ty => fn max_element(self) -> Self::Scalar);
            forward_impl!($glam_ty => fn dot(self, other: Self) -> Self::Scalar);
            forward_impl!($glam_ty => fn write_to_slice(self, slice: &mut [Self::Scalar]) -> ());
        }
    };
}

pub trait SignedVector: Vector + core::ops::Neg<Output = Self> {
    fn signum(self) -> Self;
    fn abs(self) -> Self;
}

macro_rules! impl_signed_vector {
    ($glam_ty:ty) => {
        impl SignedVector for $glam_ty {
            forward_impl!($glam_ty => fn signum(self) -> Self);
            forward_impl!($glam_ty => fn abs(self) -> Self);
        }
    };
}

pub trait FloatVector: SignedVector {
    fn ceil(self) -> Self;
    fn clamp_length_max(self, min: Self::Scalar) -> Self;
    fn clamp_length_min(self, min: Self::Scalar) -> Self;
    fn clamp_length(self, min: Self::Scalar, max: Self::Scalar) -> Self;
    fn distance_squared(self, other: Self) -> Self::Scalar;
    fn distance(self, other: Self) -> Self::Scalar;
    fn exp(self) -> Self;
    fn floor(self) -> Self;
    fn fract(self) -> Self;
    fn is_finite(self) -> bool;
    fn is_nan(self) -> bool;
    fn is_normalized(self) -> bool;
    fn length_recip(self) -> Self::Scalar;
    fn length_squared(self) -> Self::Scalar;
    fn length(self) -> Self::Scalar;
    fn lerp(self, rhs: Self, s: Self::Scalar) -> Self;
    fn mul_add(self, a: Self, b: Self) -> Self;
    fn normalize_or_zero(self) -> Self;
    fn normalize(self) -> Self;
    fn powf(self, n: Self::Scalar) -> Self;
    fn project_onto_normalized(self, other: Self) -> Self;
    fn project_onto(self, other: Self) -> Self;
    fn recip(self) -> Self;
    fn reject_from_normalized(self, other: Self) -> Self;
    fn reject_from(self, other: Self) -> Self;
    fn round(self) -> Self;
    fn try_normalize(self) -> Option<Self>;
}

macro_rules! impl_float_vector {
    ($glam_ty:ty, $scalar:ty) => {
        impl FloatVector for $glam_ty {
            forward_impl!($glam_ty => fn ceil(self) -> Self);
            forward_impl!($glam_ty => fn clamp_length_max(self, min: Self::Scalar) -> Self);
            forward_impl!($glam_ty => fn clamp_length_min(self, min: Self::Scalar) -> Self);
            forward_impl!($glam_ty => fn clamp_length(self, min: Self::Scalar, max: Self::Scalar) -> Self);
            forward_impl!($glam_ty => fn distance_squared(self, other: Self) -> Self::Scalar);
            forward_impl!($glam_ty => fn distance(self, other: Self) -> Self::Scalar);
            forward_impl!($glam_ty => fn exp(self) -> Self);
            forward_impl!($glam_ty => fn floor(self) -> Self);
            forward_impl!($glam_ty => fn fract(self) -> Self);
            forward_impl!($glam_ty => fn is_finite(self) -> bool);
            forward_impl!($glam_ty => fn is_nan(self) -> bool);
            forward_impl!($glam_ty => fn is_normalized(self) -> bool);
            forward_impl!($glam_ty => fn length_recip(self) -> Self::Scalar);
            forward_impl!($glam_ty => fn length_squared(self) -> Self::Scalar);
            forward_impl!($glam_ty => fn length(self) -> Self::Scalar);
            forward_impl!($glam_ty => fn lerp(self, rhs: Self, s: Self::Scalar) -> Self);
            forward_impl!($glam_ty => fn mul_add(self, a: Self, b: Self) -> Self);
            forward_impl!($glam_ty => fn normalize_or_zero(self) -> Self);
            forward_impl!($glam_ty => fn normalize(self) -> Self);
            forward_impl!($glam_ty => fn powf(self, n: Self::Scalar) -> Self);
            forward_impl!($glam_ty => fn project_onto_normalized(self, other: Self) -> Self);
            forward_impl!($glam_ty => fn project_onto(self, other: Self) -> Self);
            forward_impl!($glam_ty => fn recip(self) -> Self);
            forward_impl!($glam_ty => fn reject_from_normalized(self, other: Self) -> Self);
            forward_impl!($glam_ty => fn reject_from(self, other: Self) -> Self);
            forward_impl!($glam_ty => fn round(self) -> Self);
            forward_impl!($glam_ty => fn try_normalize(self) -> Option<Self>);
        }
    };
}

pub trait Vector2: Vector {
    fn from_array(array: [Self::Scalar; 2]) -> Self;
    fn to_array(&self) -> [Self::Scalar; 2];
    fn cmpeq(self, other: Self) -> glam::BVec2;
    fn cmpne(self, other: Self) -> glam::BVec2;
    fn cmpge(self, other: Self) -> glam::BVec2;
    fn cmpgt(self, other: Self) -> glam::BVec2;
    fn cmple(self, other: Self) -> glam::BVec2;
    fn cmplt(self, other: Self) -> glam::BVec2;
    fn select(mask: glam::BVec2, if_true: Self, if_false: Self) -> Self;
    fn extend(self, z: Self::Scalar) -> <Self::Scalar as Scalar>::Vec3;
}

macro_rules! impl_vector2 {
    ($glam_ty:ty, $vec3_ty:ty) => {
        impl Vector2 for $glam_ty {
            forward_impl!($glam_ty => fn from_array(array: [Self::Scalar; 2]) -> Self);
            forward_impl!($glam_ty => fn to_array(&self) -> [Self::Scalar; 2]);
            forward_impl!($glam_ty => fn cmpeq(self, other: Self) -> glam::BVec2);
            forward_impl!($glam_ty => fn cmpne(self, other: Self) -> glam::BVec2);
            forward_impl!($glam_ty => fn cmpge(self, other: Self) -> glam::BVec2);
            forward_impl!($glam_ty => fn cmpgt(self, other: Self) -> glam::BVec2);
            forward_impl!($glam_ty => fn cmple(self, other: Self) -> glam::BVec2);
            forward_impl!($glam_ty => fn cmplt(self, other: Self) -> glam::BVec2);
            forward_impl!($glam_ty => fn select(mask: glam::BVec2, if_true: Self, if_false: Self) -> Self);
            forward_impl!($glam_ty => fn extend(self, z: Self::Scalar) -> $vec3_ty);
        }
    };
}

pub trait Vector3: Vector {
    fn from_array(array: [Self::Scalar; 3]) -> Self;
    fn to_array(&self) -> [Self::Scalar; 3];
    fn cmpeq(self, other: Self) -> glam::BVec3;
    fn cmpne(self, other: Self) -> glam::BVec3;
    fn cmpge(self, other: Self) -> glam::BVec3;
    fn cmpgt(self, other: Self) -> glam::BVec3;
    fn cmple(self, other: Self) -> glam::BVec3;
    fn cmplt(self, other: Self) -> glam::BVec3;
    fn select(mask: glam::BVec3, if_true: Self, if_false: Self) -> Self;
    fn extend(self, w: Self::Scalar) -> <Self::Scalar as Scalar>::Vec4;
    fn truncate(self) -> <Self::Scalar as Scalar>::Vec2;
    fn cross(self, other: Self) -> Self;
}

macro_rules! impl_vector3 {
    ($glam_ty:ty, $vec2_ty:ty, $vec4_ty:ty) => {
        impl Vector3 for $glam_ty {
            forward_impl!($glam_ty => fn from_array(array: [Self::Scalar; 3]) -> Self);
            forward_impl!($glam_ty => fn to_array(&self) -> [Self::Scalar; 3]);
            forward_impl!($glam_ty => fn cmpeq(self, other: Self) -> glam::BVec3);
            forward_impl!($glam_ty => fn cmpne(self, other: Self) -> glam::BVec3);
            forward_impl!($glam_ty => fn cmpge(self, other: Self) -> glam::BVec3);
            forward_impl!($glam_ty => fn cmpgt(self, other: Self) -> glam::BVec3);
            forward_impl!($glam_ty => fn cmple(self, other: Self) -> glam::BVec3);
            forward_impl!($glam_ty => fn cmplt(self, other: Self) -> glam::BVec3);
            forward_impl!($glam_ty => fn select(mask: glam::BVec3, if_true: Self, if_false: Self) -> Self);
            forward_impl!($glam_ty => fn cross(self, other: Self) -> Self);
            forward_impl!($glam_ty => fn extend(self, w: Self::Scalar) -> $vec4_ty);
            forward_impl!($glam_ty => fn truncate(self) -> $vec2_ty);
        }
    };
}

pub trait Vector4: Vector {
    fn from_array(array: [Self::Scalar; 4]) -> Self;
    fn to_array(&self) -> [Self::Scalar; 4];
    fn cmpeq(self, other: Self) -> glam::BVec4;
    fn cmpne(self, other: Self) -> glam::BVec4;
    fn cmpge(self, other: Self) -> glam::BVec4;
    fn cmpgt(self, other: Self) -> glam::BVec4;
    fn cmple(self, other: Self) -> glam::BVec4;
    fn cmplt(self, other: Self) -> glam::BVec4;
    fn select(mask: glam::BVec4, if_true: Self, if_false: Self) -> Self;
    fn truncate(self) -> <Self::Scalar as Scalar>::Vec3;
}

macro_rules! impl_vector4 {
    ($glam_ty:ty, $vec3_ty:ty) => {
        impl Vector4 for $glam_ty {
            forward_impl!($glam_ty => fn from_array(array: [Self::Scalar; 4]) -> Self);
            forward_impl!($glam_ty => fn to_array(&self) -> [Self::Scalar; 4]);
            forward_impl!($glam_ty => fn cmpeq(self, other: Self) -> glam::BVec4);
            forward_impl!($glam_ty => fn cmpne(self, other: Self) -> glam::BVec4);
            forward_impl!($glam_ty => fn cmpge(self, other: Self) -> glam::BVec4);
            forward_impl!($glam_ty => fn cmpgt(self, other: Self) -> glam::BVec4);
            forward_impl!($glam_ty => fn cmple(self, other: Self) -> glam::BVec4);
            forward_impl!($glam_ty => fn cmplt(self, other: Self) -> glam::BVec4);
            forward_impl!($glam_ty => fn select(mask: glam::BVec4, if_true: Self, if_false: Self) -> Self);
            forward_impl!($glam_ty => fn truncate(self) -> $vec3_ty);
        }
    };
}

pub trait SignedVector2: SignedVector + Vector2 {
    fn perp(self) -> Self;
    fn perp_dot(self, other: Self) -> Self::Scalar;
}

pub trait FloatVector2: SignedVector2 + FloatVector {
    fn is_nan_mask(self) -> glam::BVec2;
    fn from_angle(angle: Self::Scalar) -> Self;
    fn angle_between(self, rhs: Self) -> Self::Scalar;
    fn rotate(self, other: Self) -> Self;
}

pub trait FloatVector3: Vector3 + FloatVector {
    fn is_nan_mask(self) -> glam::BVec3;
    fn angle_between(self, rhs: Self) -> Self::Scalar;
    fn any_orthogonal_vector(&self) -> Self;
    fn any_orthonormal_vector(&self) -> Self;
    fn any_orthonormal_pair(&self) -> (Self, Self);
}

pub trait FloatVector4: Vector4 + FloatVector {
    fn is_nan_mask(self) -> glam::BVec4;
}

impl_vector!(glam::Vec2, f32);
impl_vector!(glam::Vec3, f32);
impl_vector!(glam::Vec4, f32);
impl_vector!(glam::DVec2, f64);
impl_vector!(glam::DVec3, f64);
impl_vector!(glam::DVec4, f64);
impl_vector!(glam::IVec2, i32);
impl_vector!(glam::IVec3, i32);
impl_vector!(glam::IVec4, i32);
impl_vector!(glam::UVec2, u32);
impl_vector!(glam::UVec3, u32);
impl_vector!(glam::UVec4, u32);

impl_vector2!(glam::Vec2, glam::Vec3);
impl_vector2!(glam::DVec2, glam::DVec3);
impl_vector2!(glam::IVec2, glam::IVec3);
impl_vector2!(glam::UVec2, glam::UVec3);

impl_vector3!(glam::Vec3, glam::Vec2, glam::Vec4);
impl_vector3!(glam::DVec3, glam::DVec2, glam::DVec4);
impl_vector3!(glam::IVec3, glam::IVec2, glam::IVec4);
impl_vector3!(glam::UVec3, glam::UVec2, glam::UVec4);

// Note: glam::Vec4 uses BVec4A instead of BVec4, so we provide a manual impl.
//
// See https://github.com/bitshifter/glam-rs/issues/306.
impl_vector4!(glam::DVec4, glam::DVec3);
impl_vector4!(glam::IVec4, glam::IVec3);
impl_vector4!(glam::UVec4, glam::UVec3);

impl_signed_vector!(glam::Vec2);
impl_signed_vector!(glam::Vec3);
impl_signed_vector!(glam::Vec4);
impl_signed_vector!(glam::DVec2);
impl_signed_vector!(glam::DVec3);
impl_signed_vector!(glam::DVec4);
impl_signed_vector!(glam::IVec2);
impl_signed_vector!(glam::IVec3);
impl_signed_vector!(glam::IVec4);

impl_float_vector!(glam::Vec2, f32);
impl_float_vector!(glam::Vec3, f32);
impl_float_vector!(glam::Vec4, f32);
impl_float_vector!(glam::DVec2, f64);
impl_float_vector!(glam::DVec3, f64);
impl_float_vector!(glam::DVec4, f64);

#[inline]
fn bvec4a_to_bvec4(bvec: glam::BVec4A) -> glam::BVec4 {
    let bitmask = bvec.bitmask();
    let x = bitmask & 1 != 0;
    let y = bitmask & 2 != 0;
    let z = bitmask & 4 != 0;
    let w = bitmask & 8 != 0;
    glam::BVec4::new(x, y, z, w)
}

#[inline]
fn bvec4_to_bvec4a(bvec: glam::BVec4) -> glam::BVec4A {
    glam::BVec4A::new(bvec.x, bvec.y, bvec.z, bvec.w)
}

impl Vector4 for glam::Vec4 {
    forward_impl!(glam::Vec4 => fn from_array(array: [Self::Scalar; 4]) -> Self);
    forward_impl!(glam::Vec4 => fn to_array(&self) -> [Self::Scalar; 4]);

    fn cmpeq(self, other: Self) -> glam::BVec4 {
        bvec4a_to_bvec4(<glam::Vec4>::cmpeq(self, other))
    }
    fn cmpne(self, other: Self) -> glam::BVec4 {
        bvec4a_to_bvec4(<glam::Vec4>::cmpne(self, other))
    }
    fn cmpge(self, other: Self) -> glam::BVec4 {
        bvec4a_to_bvec4(<glam::Vec4>::cmpge(self, other))
    }
    fn cmpgt(self, other: Self) -> glam::BVec4 {
        bvec4a_to_bvec4(<glam::Vec4>::cmpgt(self, other))
    }
    fn cmple(self, other: Self) -> glam::BVec4 {
        bvec4a_to_bvec4(<glam::Vec4>::cmple(self, other))
    }
    fn cmplt(self, other: Self) -> glam::BVec4 {
        bvec4a_to_bvec4(<glam::Vec4>::cmplt(self, other))
    }
    fn select(mask: glam::BVec4, if_true: Self, if_false: Self) -> Self {
        <glam::Vec4>::select(bvec4_to_bvec4a(mask), if_true, if_false)
    }

    forward_impl!(glam::Vec4 => fn truncate(self) -> glam::Vec3);
}

impl SignedVector2 for glam::Vec2 {
    forward_impl!(glam::Vec2 => fn perp(self) -> Self);
    forward_impl!(glam::Vec2 => fn perp_dot(self, other: Self) -> Self::Scalar);
}
impl SignedVector2 for glam::DVec2 {
    forward_impl!(glam::DVec2 => fn perp(self) -> Self);
    forward_impl!(glam::DVec2 => fn perp_dot(self, other: Self) -> Self::Scalar);
}
impl SignedVector2 for glam::IVec2 {
    forward_impl!(glam::IVec2 => fn perp(self) -> Self);
    forward_impl!(glam::IVec2 => fn perp_dot(self, other: Self) -> Self::Scalar);
}

impl FloatVector2 for glam::Vec2 {
    forward_impl!(glam::Vec2 => fn is_nan_mask(self) -> glam::BVec2);
    forward_impl!(glam::Vec2 => fn from_angle(angle: Self::Scalar) -> Self);
    forward_impl!(glam::Vec2 => fn angle_between(self, other: Self) -> f32);
    forward_impl!(glam::Vec2 => fn rotate(self, other: Self) -> Self);
}

impl FloatVector2 for glam::DVec2 {
    forward_impl!(glam::DVec2 => fn is_nan_mask(self) -> glam::BVec2);
    forward_impl!(glam::DVec2 => fn from_angle(angle: Self::Scalar) -> Self);
    forward_impl!(glam::DVec2 => fn angle_between(self, other: Self) -> f64);
    forward_impl!(glam::DVec2 => fn rotate(self, other: Self) -> Self);
}

impl FloatVector3 for glam::Vec3 {
    forward_impl!(glam::Vec3 => fn is_nan_mask(self) -> glam::BVec3);
    forward_impl!(glam::Vec3 => fn angle_between(self, other: Self) -> f32);
    forward_impl!(glam::Vec3 => fn any_orthogonal_vector(&self) -> Self);
    forward_impl!(glam::Vec3 => fn any_orthonormal_vector(&self) -> Self);
    forward_impl!(glam::Vec3 => fn any_orthonormal_pair(&self) -> (Self, Self));
}
impl FloatVector3 for glam::DVec3 {
    forward_impl!(glam::DVec3 => fn is_nan_mask(self) -> glam::BVec3);
    forward_impl!(glam::DVec3 => fn angle_between(self, other: Self) -> f64);
    forward_impl!(glam::DVec3 => fn any_orthogonal_vector(&self) -> Self);
    forward_impl!(glam::DVec3 => fn any_orthonormal_vector(&self) -> Self);
    forward_impl!(glam::DVec3 => fn any_orthonormal_pair(&self) -> (Self, Self));
}
impl FloatVector4 for glam::Vec4 {
    #[inline]
    fn is_nan_mask(self) -> glam::BVec4 {
        bvec4a_to_bvec4(<glam::Vec4>::is_nan_mask(self))
    }
}
impl FloatVector4 for glam::DVec4 {
    forward_impl!(glam::DVec4 => fn is_nan_mask(self) -> glam::BVec4);
}

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

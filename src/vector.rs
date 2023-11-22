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

use bytemuck::{Pod, TransparentWrapper, Zeroable};

use crate::scalar::SignedScalar;
use crate::{
    bindings::prelude::*, scalar::FloatScalar, Point2, Point3, Point4, Scalar, Size2, Size3, Unit,
};
use crate::{Angle, AsRaw, FromRaw, ToRaw};

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
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
pub struct Vector2<T: Unit = f32> {
    /// X coordinate
    pub x: T::Scalar,
    /// Y coordinate
    pub y: T::Scalar,
}

/// SAFETY: `T::Scalar` is `Zeroable`, and `Vector2` is `#[repr(C)]`.
unsafe impl<T: Unit> Zeroable for Vector2<T> {}
/// SAFETY: `T::Scalar` is `Pod`.
unsafe impl<T: Unit> Pod for Vector2<T> {}
/// SAFETY: These are guaranteed to have the same representation.
unsafe impl<T: Unit> TransparentWrapper<<T::Scalar as Scalar>::Vec2> for Vector2<T> {}

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

/// SAFETY: `T::Scalar` is `Zeroable`, and `Vector3` is `#[repr(C)]`.
unsafe impl<T: Unit> Zeroable for Vector3<T> {}
/// SAFETY: `T::Scalar` is `Pod`.
unsafe impl<T: Unit> Pod for Vector3<T> {}
/// SAFETY: These are guaranteed to have the same representation.
unsafe impl<T: Unit> TransparentWrapper<<T::Scalar as Scalar>::Vec3> for Vector3<T> {}

/// 4D vector.
///
/// Alignment: This is always 16-byte aligned. [`glam::DVec4`] is only 8-byte
/// aligned (for some reason), and integer vectors are only 4-byte aligned,
/// which means that reference-casting from those glam types to `Vector4` type
/// will fail (but not the other way around - see [`Vector4::as_raw()`]).
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
#[cfg_attr(
    any(
        not(any(feature = "scalar-math", target_arch = "spirv")),
        feature = "cuda"
    ),
    repr(C, align(16))
)]
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

/// SAFETY: `T::Scalar` is `Zeroable`, and `Vector4` is `#[repr(C)]`.
unsafe impl<T: Unit> Zeroable for Vector4<T> {}
/// SAFETY: `T::Scalar` is `Pod`.
unsafe impl<T: Unit> Pod for Vector4<T> {}
/// SAFETY: These are guaranteed to have the same representation.
unsafe impl<T: Unit> TransparentWrapper<<T::Scalar as Scalar>::Vec4> for Vector4<T> {}

macro_rules! vector_interface {
    ($point_ty:ident $(, $size_ty:ident)?) => {
        #[doc = "Instantiate from point."]
        #[inline]
        #[must_use]
        pub fn from_point(point: $point_ty<T>) -> Self {
            bytemuck::cast(point)
        }

        #[doc = "Convert to point."]
        #[inline]
        #[must_use]
        pub fn to_point(self) -> $point_ty<T> {
            bytemuck::cast(self)
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
                bytemuck::cast(size)
            }

            #[doc = "Convert to size."]
            #[inline]
            #[must_use]
            pub fn to_size(self) -> $size_ty<T> {
                bytemuck::cast(self)
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

macro_rules! implement_swizzle {
    ($base_type_name:ident) => {
        impl<T: Unit> Swizzle<T> for $base_type_name<T> {
            #[inline]
            fn swizzle2<const X: usize, const Y: usize>(&self) -> Vector2<T> {
                [self.const_get::<X>(), self.const_get::<Y>()].into()
            }

            #[inline]
            fn swizzle3<const X: usize, const Y: usize, const Z: usize>(&self) -> Vector3<T> {
                [
                    self.const_get::<X>(),
                    self.const_get::<Y>(),
                    self.const_get::<Z>(),
                ]
                .into()
            }

            #[inline]
            fn swizzle4<const X: usize, const Y: usize, const Z: usize, const W: usize>(
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
    };
}

crate::forward_op_to_raw!(Vector2, Add<Self>::add -> Self);
crate::forward_op_to_raw!(Vector3, Add<Self>::add -> Self);
crate::forward_op_to_raw!(Vector4, Add<Self>::add -> Self);
crate::forward_op_to_raw!(Vector2, Sub<Self>::sub -> Self);
crate::forward_op_to_raw!(Vector3, Sub<Self>::sub -> Self);
crate::forward_op_to_raw!(Vector4, Sub<Self>::sub -> Self);
crate::forward_op_to_raw!(Vector2, Mul<Self>::mul -> Self);
crate::forward_op_to_raw!(Vector3, Mul<Self>::mul -> Self);
crate::forward_op_to_raw!(Vector4, Mul<Self>::mul -> Self);
crate::forward_op_to_raw!(Vector2, Div<Self>::div -> Self);
crate::forward_op_to_raw!(Vector3, Div<Self>::div -> Self);
crate::forward_op_to_raw!(Vector4, Div<Self>::div -> Self);
crate::forward_op_to_raw!(Vector2, Rem<Self>::rem -> Self);
crate::forward_op_to_raw!(Vector3, Rem<Self>::rem -> Self);
crate::forward_op_to_raw!(Vector4, Rem<Self>::rem -> Self);

crate::forward_neg_to_raw!(Vector2);
crate::forward_neg_to_raw!(Vector3);
crate::forward_neg_to_raw!(Vector4);

crate::forward_op_to_raw!(Vector2, Mul<[f32, f64, i32, u32]>::mul -> Self);
crate::forward_op_to_raw!(Vector3, Mul<[f32, f64, i32, u32]>::mul -> Self);
crate::forward_op_to_raw!(Vector4, Mul<[f32, f64, i32, u32]>::mul -> Self);
crate::forward_op_to_raw!(Vector2, Div<[f32, f64, i32, u32]>::div -> Self);
crate::forward_op_to_raw!(Vector3, Div<[f32, f64, i32, u32]>::div -> Self);
crate::forward_op_to_raw!(Vector4, Div<[f32, f64, i32, u32]>::div -> Self);
crate::forward_op_to_raw!(Vector2, Rem<[f32, f64, i32, u32]>::rem -> Self);
crate::forward_op_to_raw!(Vector3, Rem<[f32, f64, i32, u32]>::rem -> Self);
crate::forward_op_to_raw!(Vector4, Rem<[f32, f64, i32, u32]>::rem -> Self);

crate::forward_op_assign_to_raw!(Vector2, AddAssign<Self>::add_assign);
crate::forward_op_assign_to_raw!(Vector3, AddAssign<Self>::add_assign);
crate::forward_op_assign_to_raw!(Vector4, AddAssign<Self>::add_assign);
crate::forward_op_assign_to_raw!(Vector2, SubAssign<Self>::sub_assign);
crate::forward_op_assign_to_raw!(Vector3, SubAssign<Self>::sub_assign);
crate::forward_op_assign_to_raw!(Vector4, SubAssign<Self>::sub_assign);
crate::forward_op_assign_to_raw!(Vector2, MulAssign<Self>::mul_assign);
crate::forward_op_assign_to_raw!(Vector3, MulAssign<Self>::mul_assign);
crate::forward_op_assign_to_raw!(Vector4, MulAssign<Self>::mul_assign);
crate::forward_op_assign_to_raw!(Vector2, DivAssign<Self>::div_assign);
crate::forward_op_assign_to_raw!(Vector3, DivAssign<Self>::div_assign);
crate::forward_op_assign_to_raw!(Vector4, DivAssign<Self>::div_assign);
crate::forward_op_assign_to_raw!(Vector2, RemAssign<Self>::rem_assign);
crate::forward_op_assign_to_raw!(Vector3, RemAssign<Self>::rem_assign);
crate::forward_op_assign_to_raw!(Vector4, RemAssign<Self>::rem_assign);

crate::forward_op_assign_to_raw!(Vector2, AddAssign<[f32, f64, i32, u32]>::add_assign);
crate::forward_op_assign_to_raw!(Vector3, AddAssign<[f32, f64, i32, u32]>::add_assign);
crate::forward_op_assign_to_raw!(Vector4, AddAssign<[f32, f64, i32, u32]>::add_assign);
crate::forward_op_assign_to_raw!(Vector2, SubAssign<[f32, f64, i32, u32]>::sub_assign);
crate::forward_op_assign_to_raw!(Vector3, SubAssign<[f32, f64, i32, u32]>::sub_assign);
crate::forward_op_assign_to_raw!(Vector4, SubAssign<[f32, f64, i32, u32]>::sub_assign);
crate::forward_op_assign_to_raw!(Vector2, MulAssign<[f32, f64, i32, u32]>::mul_assign);
crate::forward_op_assign_to_raw!(Vector3, MulAssign<[f32, f64, i32, u32]>::mul_assign);
crate::forward_op_assign_to_raw!(Vector4, MulAssign<[f32, f64, i32, u32]>::mul_assign);
crate::forward_op_assign_to_raw!(Vector2, DivAssign<[f32, f64, i32, u32]>::div_assign);
crate::forward_op_assign_to_raw!(Vector3, DivAssign<[f32, f64, i32, u32]>::div_assign);
crate::forward_op_assign_to_raw!(Vector4, DivAssign<[f32, f64, i32, u32]>::div_assign);
crate::forward_op_assign_to_raw!(Vector2, RemAssign<[f32, f64, i32, u32]>::rem_assign);
crate::forward_op_assign_to_raw!(Vector3, RemAssign<[f32, f64, i32, u32]>::rem_assign);
crate::forward_op_assign_to_raw!(Vector4, RemAssign<[f32, f64, i32, u32]>::rem_assign);

crate::derive_standard_traits!(Vector2 {
    x: T::Scalar,
    y: T::Scalar
});
crate::derive_standard_traits!(Vector3 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar
});
crate::derive_standard_traits!(Vector4 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar,
    w: T::Scalar
});

crate::derive_array_conversion_traits!(Vector2, 2);
crate::derive_array_conversion_traits!(Vector3, 3);
crate::derive_array_conversion_traits!(Vector4, 4);

crate::derive_tuple_conversion_traits!(Vector2 {
    x: T::Scalar,
    y: T::Scalar
});
crate::derive_tuple_conversion_traits!(Vector3 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar
});
crate::derive_tuple_conversion_traits!(Vector4 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar,
    w: T::Scalar
});

crate::derive_glam_conversion_traits!(Vector2 {
    x: T::Scalar,
    y: T::Scalar
});
crate::derive_glam_conversion_traits!(Vector3 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar
});
crate::derive_glam_conversion_traits!(Vector4 {
    x: T::Scalar,
    y: T::Scalar,
    z: T::Scalar,
    w: T::Scalar
});

implement_swizzle!(Vector2);
implement_swizzle!(Vector3);
implement_swizzle!(Vector4);

impl<T: Unit> Vector2<T> {
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

    /// Unit vector in the direction of the X axis.
    pub const X: Self = Vector2 {
        x: T::Scalar::ONE,
        y: T::Scalar::ZERO,
    };

    /// Unit vector in the direction of the Y axis.
    pub const Y: Self = Vector2 {
        x: T::Scalar::ZERO,
        y: T::Scalar::ONE,
    };

    /// The unit axes.
    pub const AXES: [Self; 2] = [Self::X, Self::Y];

    /// New vector.
    pub const fn new(x: T::Scalar, y: T::Scalar) -> Self {
        Self { x, y }
    }

    crate::forward_constructors!(2, glam::Vec2);
    crate::forward_comparison!(glam::BVec2, glam::Vec2);

    crate::casting_interface!(Vector2 {
        x: T::Scalar,
        y: T::Scalar
    });
    crate::tuple_interface!(Vector2 {
        x: T::Scalar,
        y: T::Scalar
    });
    crate::array_interface!(2);

    crate::forward_to_raw!(
        glam::Vec2 =>
        #[doc = "Dot product"]
        pub fn dot(self, other: Self) -> T::Scalar;
        #[doc = "Extend with z-component to [`Vector3`]."]
        pub fn extend(self, z: T::Scalar) -> Vector3<T>;
    );

    /// Select components of this vector and return a new vector containing
    /// those components.
    #[inline]
    #[must_use]
    pub fn swizzle<const X: usize, const Y: usize>(&self) -> Self {
        self.swizzle2::<X, Y>()
    }

    vector_interface!(Point2, Size2);
}

impl<T> Vector2<T>
where
    T: Unit,
    T::Scalar: FloatScalar,
{
    /// All NaN.
    pub const NAN: Self = Vector2 {
        x: <T::Scalar as FloatScalar>::NAN,
        y: <T::Scalar as FloatScalar>::NAN,
    };
    /// All positive infinity.
    pub const INFINITY: Self = Vector2 {
        x: <T::Scalar as FloatScalar>::INFINITY,
        y: <T::Scalar as FloatScalar>::INFINITY,
    };
    /// All negative infinity.
    pub const NEG_INFINITY: Self = Vector2 {
        x: <T::Scalar as FloatScalar>::NEG_INFINITY,
        y: <T::Scalar as FloatScalar>::NEG_INFINITY,
    };

    crate::forward_float_ops!(glam::BVec2, glam::Vec2);
    crate::forward_float_vector_ops!(glam::Vec2);

    crate::forward_to_raw!(
        glam::Vec2 =>
        #[doc = "Return `(sin(angle), cos(angle)`."]
        pub fn from_angle(angle: Angle<T::Scalar>) -> Vector2<T::Scalar>;
        #[doc = "Rotate by a vector containing `(sin(angle), cos(angle))`"]
        pub fn rotate(self, rotation: Vector2<T::Scalar>) -> Self;
        #[doc = "Angle between this and another vector."]
        pub fn angle_between(self, other: Self) -> Angle<T::Scalar>;
    );
}

impl<T> Vector2<T>
where
    T: Unit,
    T::Scalar: SignedScalar,
{
    /// All negative one.
    pub const NEG_ONE: Self = Vector2 {
        x: T::Scalar::NEG_ONE,
        y: T::Scalar::NEG_ONE,
    };

    /// (-1, 0)
    pub const NEG_X: Self = Vector2 {
        x: T::Scalar::NEG_ONE,
        y: T::Scalar::ZERO,
    };
    /// (0, -1)
    pub const NEG_Y: Self = Vector2 {
        x: T::Scalar::ZERO,
        y: T::Scalar::NEG_ONE,
    };

    crate::forward_to_raw!(
        glam::Vec2 =>
        #[doc = "Turn all components positive."]
        pub fn abs(self) -> Self;
        #[doc = "Return a vector where each component is 1 or -1 depending on the sign of the input."]
        pub fn signum(self) -> Self;
        #[doc = "Get the perpendicular vector."]
        pub fn perp(self) -> Self;
        #[doc(alias = "wedge")]
        #[doc(alias = "cross")]
        #[doc(alias = "determinant")]
        #[doc = "Perpendicular dot product"]
        pub fn perp_dot(self, other: Self) -> T::Scalar;
    );
}

impl<T: Unit> Vector3<T> {
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

    /// Unit vector in the direction of the X axis.
    pub const X: Self = Self {
        x: T::Scalar::ONE,
        y: T::Scalar::ZERO,
        z: T::Scalar::ZERO,
    };
    /// Unit vector in the direction of the Y axis.
    pub const Y: Self = Self {
        x: T::Scalar::ZERO,
        y: T::Scalar::ONE,
        z: T::Scalar::ZERO,
    };
    /// Unit vector in the direction of the Z axis.
    pub const Z: Self = Self {
        x: T::Scalar::ZERO,
        y: T::Scalar::ZERO,
        z: T::Scalar::ONE,
    };

    /// The unit axes.
    pub const AXES: [Self; 3] = [Self::X, Self::Y, Self::Z];

    /// New vector.
    pub const fn new(x: T::Scalar, y: T::Scalar, z: T::Scalar) -> Self {
        Self { x, y, z }
    }

    crate::forward_constructors!(3, glam::Vec3);
    crate::forward_comparison!(glam::BVec3, glam::Vec3);

    crate::casting_interface!(Vector3 {
        x: T::Scalar,
        y: T::Scalar,
        z: T::Scalar
    });
    crate::tuple_interface!(Vector3 {
        x: T::Scalar,
        y: T::Scalar,
        z: T::Scalar
    });
    crate::array_interface!(3);

    crate::forward_to_raw!(
        glam::Vec3 =>
        #[doc = "Dot product"]
        pub fn dot(self, other: Self) -> T::Scalar;
        #[doc = "Extend with w-component to [`Vector4`]."]
        pub fn extend(self, w: T::Scalar) -> Vector4<T>;
        #[doc = "Truncate to [`Vector2`]."]
        pub fn truncate(self) -> Vector2<T>;
    );

    /// Select components of this vector and return a new vector containing
    /// those components.
    #[inline]
    #[must_use]
    pub fn swizzle<const X: usize, const Y: usize, const Z: usize>(&self) -> Self {
        self.swizzle3::<X, Y, Z>()
    }

    vector_interface!(Point3, Size3);
}

impl<T> Vector3<T>
where
    T: Unit,
    T::Scalar: FloatScalar,
{
    /// All NaN.
    pub const NAN: Self = Vector3 {
        x: <T::Scalar as FloatScalar>::NAN,
        y: <T::Scalar as FloatScalar>::NAN,
        z: <T::Scalar as FloatScalar>::NAN,
    };
    /// All positive infinity.
    pub const INFINITY: Self = Vector3 {
        x: <T::Scalar as FloatScalar>::INFINITY,
        y: <T::Scalar as FloatScalar>::INFINITY,
        z: <T::Scalar as FloatScalar>::INFINITY,
    };
    /// All negative infinity.
    pub const NEG_INFINITY: Self = Vector3 {
        x: <T::Scalar as FloatScalar>::NEG_INFINITY,
        y: <T::Scalar as FloatScalar>::NEG_INFINITY,
        z: <T::Scalar as FloatScalar>::NEG_INFINITY,
    };

    crate::forward_float_ops!(glam::BVec3, glam::Vec3);
    crate::forward_float_vector_ops!(glam::Vec3);

    crate::forward_to_raw!(
        glam::Vec3 =>
        #[doc = "Angle between this and another vector."]
        pub fn angle_between(self, other: Self) -> Angle<T::Scalar>;
        #[doc = "See (e.g.) [`glam::Vec3::any_orthogonal_vector()`]."]
        pub fn any_orthogonal_vector(&self) -> Self;
        #[doc = "See (e.g.) [`glam::Vec3::any_orthonormal_vector()`]."]
        pub fn any_orthonormal_vector(&self) -> Self;
        #[doc = "See (e.g.) [`glam::Vec3::any_orthonormal_pair()`]."]
        pub fn any_orthonormal_pair(&self) -> (Self, Self);
        #[doc = "Cross product"]
        pub fn cross(self, other: Self) -> Self;
    );
}

impl<T> Vector3<T>
where
    T: Unit,
    T::Scalar: SignedScalar,
{
    /// All negative one.
    pub const NEG_ONE: Self = Vector3 {
        x: T::Scalar::NEG_ONE,
        y: T::Scalar::NEG_ONE,
        z: T::Scalar::NEG_ONE,
    };

    /// (-1, 0, 0)
    pub const NEG_X: Self = Vector3 {
        x: T::Scalar::NEG_ONE,
        y: T::Scalar::ZERO,
        z: T::Scalar::ZERO,
    };
    /// (0, -1, 0)
    pub const NEG_Y: Self = Vector3 {
        x: T::Scalar::ZERO,
        y: T::Scalar::NEG_ONE,
        z: T::Scalar::ZERO,
    };
    /// (0, 0, -1)
    pub const NEG_Z: Self = Vector3 {
        x: T::Scalar::ZERO,
        y: T::Scalar::ZERO,
        z: T::Scalar::NEG_ONE,
    };

    crate::forward_to_raw!(
        glam::Vec3 =>
        #[doc = "Turn all components positive."]
        pub fn abs(self) -> Self;
        #[doc = "Return a vector where each component is 1 or -1 depending on the sign of the input."]
        pub fn signum(self) -> Self;
    );
}

impl<T> Vector3<T>
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

impl<T: Unit> Vector4<T> {
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

    /// Unit vector in the direction of the X axis.
    pub const X: Self = Self {
        x: T::Scalar::ONE,
        y: T::Scalar::ZERO,
        z: T::Scalar::ZERO,
        w: T::Scalar::ZERO,
    };
    /// Unit vector in the direction of the Y axis.
    pub const Y: Self = Self {
        x: T::Scalar::ZERO,
        y: T::Scalar::ONE,
        z: T::Scalar::ZERO,
        w: T::Scalar::ZERO,
    };
    /// Unit vector in the direction of the Z axis.
    pub const Z: Self = Self {
        x: T::Scalar::ZERO,
        y: T::Scalar::ZERO,
        z: T::Scalar::ONE,
        w: T::Scalar::ZERO,
    };
    /// Unit vector in the direction of the W axis.
    pub const W: Self = Self {
        x: T::Scalar::ZERO,
        y: T::Scalar::ZERO,
        z: T::Scalar::ZERO,
        w: T::Scalar::ONE,
    };

    /// The unit axes.
    pub const AXES: [Self; 4] = [Self::X, Self::Y, Self::Z, Self::W];

    /// New vector.
    pub const fn new(x: T::Scalar, y: T::Scalar, z: T::Scalar, w: T::Scalar) -> Self {
        Self { x, y, z, w }
    }

    crate::forward_constructors!(4, glam::Vec4);
    crate::forward_comparison!(glam::BVec4, glam::Vec4);

    crate::casting_interface!(Vector4 {
        x: T::Scalar,
        y: T::Scalar,
        z: T::Scalar,
        w: T::Scalar
    });
    crate::tuple_interface!(Vector4 {
        x: T::Scalar,
        y: T::Scalar,
        z: T::Scalar,
        w: T::Scalar
    });
    crate::array_interface!(4);

    crate::forward_to_raw!(
        glam::Vec4 =>
        #[doc = "Dot product"]
        pub fn dot(self, other: Self) -> T::Scalar;
        #[doc = "Truncate to [`Vector3`]."]
        pub fn truncate(self) -> Vector3<T>;
    );

    /// Select components of this vector and return a new vector containing
    /// those components.
    #[inline]
    #[must_use]
    pub fn swizzle<const X: usize, const Y: usize, const Z: usize, const W: usize>(&self) -> Self {
        self.swizzle4::<X, Y, Z, W>()
    }

    vector_interface!(Point4);
}

impl<T> Vector4<T>
where
    T: Unit,
    T::Scalar: FloatScalar,
{
    /// All NaN.
    pub const NAN: Self = Vector4 {
        x: <T::Scalar as FloatScalar>::NAN,
        y: <T::Scalar as FloatScalar>::NAN,
        z: <T::Scalar as FloatScalar>::NAN,
        w: <T::Scalar as FloatScalar>::NAN,
    };
    /// All positive infinity.
    pub const INFINITY: Self = Vector4 {
        x: <T::Scalar as FloatScalar>::INFINITY,
        y: <T::Scalar as FloatScalar>::INFINITY,
        z: <T::Scalar as FloatScalar>::INFINITY,
        w: <T::Scalar as FloatScalar>::INFINITY,
    };
    /// All negative infinity.
    pub const NEG_INFINITY: Self = Vector4 {
        x: <T::Scalar as FloatScalar>::NEG_INFINITY,
        y: <T::Scalar as FloatScalar>::NEG_INFINITY,
        z: <T::Scalar as FloatScalar>::NEG_INFINITY,
        w: <T::Scalar as FloatScalar>::NEG_INFINITY,
    };

    crate::forward_float_ops!(glam::BVec4, glam::Vec4);
    crate::forward_float_vector_ops!(glam::Vec4);
}

impl<T> Vector4<T>
where
    T: Unit,
    T::Scalar: SignedScalar,
{
    /// All negative one.
    pub const NEG_ONE: Self = Vector4 {
        x: T::Scalar::NEG_ONE,
        y: T::Scalar::NEG_ONE,
        z: T::Scalar::NEG_ONE,
        w: T::Scalar::NEG_ONE,
    };

    /// (-1, 0, 0, 0)
    pub const NEG_X: Self = Vector4 {
        x: T::Scalar::NEG_ONE,
        y: T::Scalar::ZERO,
        z: T::Scalar::ZERO,
        w: T::Scalar::ZERO,
    };
    /// (0, -1, 0, 0)
    pub const NEG_Y: Self = Vector4 {
        x: T::Scalar::ZERO,
        y: T::Scalar::NEG_ONE,
        z: T::Scalar::ZERO,
        w: T::Scalar::ZERO,
    };
    /// (0, 0, -1, 0)
    pub const NEG_Z: Self = Vector4 {
        x: T::Scalar::ZERO,
        y: T::Scalar::ZERO,
        z: T::Scalar::NEG_ONE,
        w: T::Scalar::ZERO,
    };
    /// (0, 0, 0, -1)
    pub const NEG_W: Self = Vector4 {
        x: T::Scalar::ZERO,
        y: T::Scalar::ZERO,
        z: T::Scalar::ZERO,
        w: T::Scalar::NEG_ONE,
    };

    crate::forward_to_raw!(
        glam::Vec4 =>
        #[doc = "Turn all components positive."]
        pub fn abs(self) -> Self;
        #[doc = "Return a vector where each component is 1 or -1 depending on the sign of the input."]
        pub fn signum(self) -> Self;
    );
}

impl<T: Unit> ToRaw for Vector2<T> {
    type Raw = <T::Scalar as Scalar>::Vec2;

    #[inline]
    fn to_raw(self) -> Self::Raw {
        bytemuck::cast(self)
    }
}

impl<T: Unit> FromRaw for Vector2<T> {
    #[inline]
    fn from_raw(raw: Self::Raw) -> Self {
        bytemuck::cast(raw)
    }
}

impl<T: Unit> AsRaw for Vector2<T> {
    #[inline]
    fn as_raw(&self) -> &Self::Raw {
        bytemuck::cast_ref(self)
    }

    #[inline]
    fn as_raw_mut(&mut self) -> &mut Self::Raw {
        bytemuck::cast_mut(self)
    }
}

impl<T: Unit> ToRaw for Vector3<T> {
    type Raw = <T::Scalar as Scalar>::Vec3;

    #[inline]
    fn to_raw(self) -> Self::Raw {
        bytemuck::cast(self)
    }
}

impl<T: Unit> FromRaw for Vector3<T> {
    #[inline]
    fn from_raw(raw: Self::Raw) -> Self {
        bytemuck::cast(raw)
    }
}

impl<T: Unit> AsRaw for Vector3<T> {
    #[inline]
    fn as_raw(&self) -> &Self::Raw {
        bytemuck::cast_ref(self)
    }

    #[inline]
    fn as_raw_mut(&mut self) -> &mut Self::Raw {
        bytemuck::cast_mut(self)
    }
}

impl<T: Unit> ToRaw for Vector4<T> {
    type Raw = <T::Scalar as Scalar>::Vec4;

    #[inline]
    fn to_raw(self) -> Self::Raw {
        bytemuck::cast(self)
    }
}

impl<T: Unit> FromRaw for Vector4<T> {
    #[inline]
    fn from_raw(raw: Self::Raw) -> Self {
        bytemuck::cast(raw)
    }
}

impl<T: Unit> AsRaw for Vector4<T> {
    #[inline]
    fn as_raw(&self) -> &Self::Raw {
        bytemuck::cast_ref(self)
    }

    #[inline]
    fn as_raw_mut(&mut self) -> &mut Self::Raw {
        bytemuck::cast_mut(self)
    }
}

impl<T> From<glam::Vec3A> for Vector3<T>
where
    T: Unit<Scalar = f32>,
{
    #[inline]
    fn from(v: glam::Vec3A) -> Self {
        Self::from_raw(v.into())
    }
}

impl<T> From<Vector3<T>> for glam::Vec3A
where
    T: Unit<Scalar = f32>,
{
    #[inline]
    fn from(v: Vector3<T>) -> Self {
        v.to_raw().into()
    }
}

impl<T> Mul<Vector3<T>> for glam::Quat
where
    T: Unit<Scalar = f32>,
    T::Scalar: FloatScalar<Vec3f = glam::Vec3>,
{
    type Output = Vector3<T>;

    #[inline]
    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        Vector3::from_raw(self * rhs.to_raw())
    }
}

impl<T> Mul<Vector3<T>> for glam::DQuat
where
    T: Unit<Scalar = f64>,
    T::Scalar: FloatScalar<Vec3f = glam::DVec3>,
{
    type Output = Vector3<T>;

    #[inline]
    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        Vector3::from_raw(self * rhs.to_raw())
    }
}

impl<'a, T: Unit> Sum<&'a Vector2<T>> for Vector2<T> {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        Self::from_raw(iter.map(AsRaw::as_raw).sum())
    }
}

impl<'a, T: Unit> Sum<&'a Vector3<T>> for Vector3<T> {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        Self::from_raw(iter.map(AsRaw::as_raw).sum())
    }
}

impl<'a, T: Unit> Sum<&'a Vector4<T>> for Vector4<T> {
    #[inline]
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        Self::from_raw(iter.map(AsRaw::as_raw).sum())
    }
}

impl<'a, T: Unit> Product<&'a Vector2<T>> for Vector2<T> {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        Self::from_raw(iter.map(AsRaw::as_raw).product())
    }
}

impl<'a, T: Unit> Product<&'a Vector3<T>> for Vector3<T> {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        Self::from_raw(iter.map(AsRaw::as_raw).product())
    }
}

impl<'a, T: Unit> Product<&'a Vector4<T>> for Vector4<T> {
    #[inline]
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = &'a Self>,
    {
        Self::from_raw(iter.map(AsRaw::as_raw).product())
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::{vector, AngleConsts};

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
        assert_eq!(v, (2.0, 4.0, 6.0, 8.0));
        v /= Vector4::splat(2.0);
        assert_eq!(v, (1.0, 2.0, 3.0, 4.0));
        v += Vector4::splat(1.0);
        assert_eq!(v, (2.0, 3.0, 4.0, 5.0));
        v -= Vector4::splat(2.0);
        assert_eq!(v, (0.0, 1.0, 2.0, 3.0));
        v %= Vector4::splat(2.0);
        assert_eq!(v, (0.0, 1.0, 0.0, 1.0));
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
        assert_eq!(sum, (4.0, 8.0, 12.0));
    }

    #[test]
    fn sum4() {
        let a: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let b: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let c: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let d: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let sum: Vector4<f32> = [a, b, c, d].iter().sum();
        assert_eq!(sum, (4.0, 8.0, 12.0, 16.0));
    }

    #[test]
    fn product2() {
        let a: Vector2<f32> = vector!(1.0, 2.0);
        let b: Vector2<f32> = vector!(1.0, 2.0);
        let c: Vector2<f32> = vector!(1.0, 2.0);
        let d: Vector2<f32> = vector!(1.0, 2.0);
        let product: Vector2<f32> = [a, b, c, d].iter().product();
        assert_eq!(product, (1.0, 16.0));
    }

    #[test]
    fn product3() {
        let a: Vector3<f32> = vector!(1.0, 2.0, 3.0);
        let b: Vector3<f32> = vector!(1.0, 2.0, 3.0);
        let c: Vector3<f32> = vector!(1.0, 2.0, 3.0);
        let d: Vector3<f32> = vector!(1.0, 2.0, 3.0);
        let product: Vector3<f32> = [a, b, c, d].iter().product();
        assert_eq!(product, (1.0, 16.0, 81.0));
    }

    #[test]
    fn product4() {
        let a: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let b: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let c: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let d: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
        let product: Vector4<f32> = [a, b, c, d].iter().product();
        assert_eq!(product, (1.0, 16.0, 81.0, 256.0));
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
        let v2 = Vec2::new(1.0, core::f32::NAN);
        assert!(v2.is_nan());
        assert!(!v2.is_finite());
        assert_eq!(v2.is_nan_mask(), glam::BVec2::new(false, true));

        let v3 = Vec3::new(1.0, core::f32::NAN, 3.0);
        assert!(v3.is_nan());
        assert!(!v3.is_finite());
        assert_eq!(v3.is_nan_mask(), glam::BVec3::new(false, true, false));

        let v4 = Vec4::new(1.0, 2.0, core::f32::NAN, 4.0);
        assert!(v4.is_nan());
        assert!(!v4.is_finite());
        assert_eq!(
            v4.is_nan_mask(),
            glam::BVec4::new(false, false, true, false)
        );

        assert!(Vec2::NAN.is_nan());
        assert!(Vec3::NAN.is_nan());
        assert!(Vec4::NAN.is_nan());

        // Replace NaNs with zeroes.
        let v = Vec4::select(v4.is_nan_mask(), Vec4::ZERO, v4);
        assert_eq!(v, (1.0, 2.0, 0.0, 4.0));
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
            (3.0, 2.0, 1.0)
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
            (3.0, 2.0, 1.0, 0.0)
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
            let x: Vec4 = (1.0, 2.0, 3.0, 4.0).into();

            let mut a = x;
            let mut b = x;
            let mut c = x;
            a *= 2.0;
            b /= 2.0;
            c %= 2.0;
            assert_eq!(a, (2.0, 4.0, 6.0, 8.0));
            assert_eq!(b, (0.5, 1.0, 1.5, 2.0));
            assert_eq!(c, (1.0, 0.0, 1.0, 0.0));
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

        assert_eq!(a.min(b), [1.0, 2.0, 1.0, 3.0]);
        assert_eq!(a.max(b), [4.0, 2.0, 3.0, 4.0]);
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

        assert_eq!(a.min(b), [1.0, 2.0, 1.0, 3.0]);
        assert_eq!(a.max(b), [4.0, 2.0, 3.0, 4.0]);
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
        assert_eq!(b.clamp(IVec4::splat(2), IVec4::splat(4)), (2, 2, 3, 4));
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
        assert_eq!(a.abs(), (1.0, 2.0, 3.0));
    }

    #[test]
    fn signum() {
        let a = Vec3::new(1.0, -2.0, -3.0);
        assert_eq!(a.signum(), (1.0, -1.0, -1.0));
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
        assert_abs_diff_eq!(rotated.to_raw(), y);
    }

    #[test]
    fn matrix_mul_custom_unit() {
        use crate::{vec3, Matrix3};
        let mat = Matrix3::<f32>::IDENTITY;
        let a: Vector3<F32> = vec3!(20.0, 30.0, 1.0);
        let b: Vector3<F32> = mat * a;
        assert_eq!(b, (20.0, 30.0, 1.0));
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
        assert_eq!(vec, [1, 100, 3, 4]);

        let vec3: &mut Vector4<i32> = vec.cast_mut();
        vec3.z = 100;
        assert_eq!(vec, [1, 100, 100, 4]);
    }

    #[test]
    fn glam_reference_conversion() {
        use core::borrow::{Borrow, BorrowMut};

        let mut vec = Vector4::<I32>::new(1, 2, 3, 4);

        let vec1: &glam::IVec4 = vec.as_ref();
        assert_eq!(*vec1, glam::IVec4::new(1, 2, 3, 4));

        let vec2: &mut glam::IVec4 = vec.as_mut();
        vec2.y = 100;
        assert_eq!(vec, [1, 100, 3, 4]);

        let vec3: &glam::IVec4 = vec.borrow();
        assert_eq!(*vec3, glam::IVec4::new(1, 100, 3, 4));

        let vec3a: &Vector4<I32> = vec3.borrow();
        assert_eq!(*vec3a, [1, 100, 3, 4]);

        let vec4: &mut glam::IVec4 = vec.borrow_mut();
        vec4.z = 100;
        assert_eq!(*vec4, glam::IVec4::new(1, 100, 100, 4));

        let vec4a: &mut Vector4<I32> = vec4.borrow_mut();
        assert_eq!(*vec4a, [1, 100, 100, 4]);
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
                self.0 = self.0.rotate_left(4) ^ *byte as u64;
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
    fn hash_map() {
        let mut map = hashbrown::HashMap::<glam::IVec4, &'static str>::default();
        map.insert(glam::IVec4::new(1, 2, 3, 4), "hello");
        assert_eq!(map.get(&Vector4::<I32>::new(1, 2, 3, 4)), Some(&"hello"));
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

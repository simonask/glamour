//! Size vectors

use crate::{traits::Lerp, Scalar, Unit, Vector2, Vector3};

use core::ops::{Add, AddAssign, Sub, SubAssign};

/// 2D size.
#[repr(C)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
pub struct Size2<T: Unit = f32> {
    /// Width
    pub width: T::Scalar,
    /// Height
    pub height: T::Scalar,
}

/// 3D size.
#[repr(C)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
pub struct Size3<T: Unit = f32> {
    /// Width
    pub width: T::Scalar,
    /// Height
    pub height: T::Scalar,
    /// Depth
    pub depth: T::Scalar,
}

crate::impl_common!(Size2 {
    width: T::Scalar,
    height: T::Scalar
});
crate::impl_common!(Size3 {
    width: T::Scalar,
    height: T::Scalar,
    depth: T::Scalar
});

crate::impl_simd_common!(Size2 [2] => Vec2, glam::BVec2 { width, height });

crate::impl_simd_common!(Size3 [3] => Vec3, glam::BVec3 {
    width,
    height,
    depth
});

crate::impl_glam_conversion!(Size2, 2 [f32 => glam::Vec2, f64 => glam::DVec2, i32 => glam::IVec2, u32 => glam::UVec2]);
crate::impl_glam_conversion!(Size3, 3 [f32 => glam::Vec3, f64 => glam::DVec3, i32 => glam::IVec3, u32 => glam::UVec3]);

crate::impl_scaling!(Size2, 2 [f32, f64, i32, u32]);
crate::impl_scaling!(Size3, 3 [f32, f64, i32, u32]);

crate::impl_as_tuple!(Size2 {
    width: T::Scalar,
    height: T::Scalar
});
crate::impl_as_tuple!(Size3 {
    width: T::Scalar,
    height: T::Scalar,
    depth: T::Scalar
});

macro_rules! impl_size {
    ($base_type_name:ident [ $dimensions:literal ] => $vec_ty:ident, $vector_type:ident) => {
        impl<T: Unit> Add for $base_type_name<T> {
            type Output = Self;

            fn add(self, other: Self) -> Self {
                Self::from_raw(self.to_raw() + other.to_raw())
            }
        }

        impl<T: Unit> Sub for $base_type_name<T> {
            type Output = Self;

            fn sub(self, other: Self) -> Self {
                Self::from_raw(self.to_raw() - other.to_raw())
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

        impl<T: Unit> From<$vector_type<T>> for $base_type_name<T> {
            #[inline(always)]
            fn from(vec: $vector_type<T>) -> Self {
                Self::from_raw(vec.to_raw())
            }
        }

        impl<T: Unit> From<$base_type_name<T>> for $vector_type<T> {
            #[inline(always)]
            fn from(point: $base_type_name<T>) -> Self {
                Self::from_raw(point.to_raw())
            }
        }

        impl<T: Unit> $base_type_name<T> {
            #[doc = "Interpret `vec` as size."]
            #[inline(always)]
            pub fn from_vector(vec: $vector_type<T>) -> Self {
                vec.into()
            }

            #[doc = "Convert to vector."]
            #[inline(always)]
            pub fn to_vector(self) -> $vector_type<T> {
                self.into()
            }

            #[doc = "Reinterpret as vector."]
            #[inline(always)]
            pub fn as_vector(&self) -> &$vector_type<T> {
                bytemuck::cast_ref(self)
            }

            #[doc = "Reinterpret as vector."]
            #[inline(always)]
            pub fn as_vector_mut(&mut self) -> &mut $vector_type<T> {
                bytemuck::cast_mut(self)
            }
        }

        impl<T> Lerp<T::Primitive> for $base_type_name<T>
        where
            T: crate::traits::UnitTypes,
            T::$vec_ty: Lerp<T::Primitive>,
        {
            #[inline(always)]
            fn lerp(self, end: Self, t: T::Primitive) -> Self {
                Self::from_raw(self.to_raw().lerp(end.to_raw(), t))
            }
        }
    };
}

impl_size!(Size2 [2] => Vec2, Vector2);
impl_size!(Size3 [3] => Vec3, Vector3);

impl<T: crate::traits::UnitTypes> Size2<T> {
    /// Calculate the area.
    pub fn area(&self) -> T::Scalar {
        T::Scalar::from_raw(self.width.to_raw() * self.height.to_raw())
    }
}

impl<T: crate::traits::UnitTypes> Size3<T> {
    /// Calculate the volume.
    pub fn volume(&self) -> T::Scalar {
        T::Scalar::from_raw(self.width.to_raw() * self.height.to_raw() * self.depth.to_raw())
    }
}

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

crate::impl_simd_common!(Size2 [2] => Vec2 { width, height });

crate::impl_simd_common!(Size3 [3] => Vec3 {
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
            #[doc = "Interpret `vec` as size."]
            #[inline]
            pub fn from_vector(vec: $vector_type<T>) -> Self {
                vec.into()
            }

            #[doc = "Convert to vector."]
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

impl_size!(Size2 [2] => Vec2, Vector2);
impl_size!(Size3 [3] => Vec3, Vector3);

impl<T: crate::UnitTypes> Size2<T> {
    /// Calculate the area.
    pub fn area(&self) -> T::Scalar {
        T::Scalar::from_raw(self.width.to_raw() * self.height.to_raw())
    }
}

impl<T: crate::UnitTypes> Size3<T> {
    /// Calculate the volume.
    pub fn volume(&self) -> T::Scalar {
        T::Scalar::from_raw(self.width.to_raw() * self.height.to_raw() * self.depth.to_raw())
    }
}

#[cfg(test)]
mod tests {
    use crate::size;

    use super::*;

    #[test]
    fn arithmetic() {
        let mut a: Size2<f32> = size!(100.0, 200.0);
        let b: Size2<f32> = a - size!(50.0, 100.0);
        assert_eq!(b, (50.0, 100.0));
        a -= size!(50.0, 100.0);
        assert_eq!(a, (50.0, 100.0));
        assert_eq!(a + size!(2.0, 3.0), (52.0, 103.0));
        a += size!(100.0, 200.0);
        assert_eq!(a, size!(150.0, 300.0));
    }

    #[test]
    fn as_vector() {
        let mut a: Size2<f32> = size!(100.0, 200.0);
        let _: &Vector2<f32> = a.as_vector();
        let _: &mut Vector2<f32> = a.as_vector_mut();
    }

    #[test]
    fn area() {
        let a: Size2<f32> = Size2 {
            width: 2.0,
            height: 3.0,
        };
        let b: Size2<i32> = Size2 {
            width: 5,
            height: 6,
        };
        let c: Size2<f64> = Size2 {
            width: 100.0,
            height: 300.0,
        };

        assert_eq!(a.area(), 6.0);
        assert_eq!(b.area(), 30);
        assert_eq!(c.area(), 30000.0);
    }

    #[test]
    fn volume() {
        let a: Size3<f32> = Size3 {
            width: 2.0,
            height: 3.0,
            depth: 2.0,
        };
        let b: Size3<i32> = Size3 {
            width: 5,
            height: 6,
            depth: 2,
        };
        let c: Size3<f64> = Size3 {
            width: 100.0,
            height: 300.0,
            depth: 2.0,
        };

        assert_eq!(a.volume(), 12.0);
        assert_eq!(b.volume(), 60);
        assert_eq!(c.volume(), 60000.0);
    }

    #[test]
    fn from_into_vector() {
        let mut p: Size2<f32> = size!(1.0, 2.0);
        let mut v: Vector2<f32> = p.to_vector();
        let q: Size2<f32> = Size2::from_vector(v);
        assert_eq!(p, q);
        assert_eq!(Vector2::from_size(p), v);

        let _: &Vector2<_> = p.as_vector();
        let _: &mut Vector2<_> = p.as_vector_mut();
        let _: &Size2<_> = v.as_size();
        let _: &mut Size2<_> = v.as_size_mut();
    }
}

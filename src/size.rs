//! Size vectors

use crate::{
    bindings::prelude::*, scalar::FloatScalar, AsRaw, FromRawRef, Scalar, ToRaw, Unit, Vector2,
    Vector3,
};

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

impl<T: Unit> ToRaw for Size2<T> {
    type Raw = <T::Scalar as Scalar>::Vec2;

    fn to_raw(self) -> Self::Raw {
        bytemuck::cast(self)
    }

    fn from_raw(raw: Self::Raw) -> Self {
        bytemuck::cast(raw)
    }
}

impl<T: Unit> AsRaw for Size2<T> {
    fn as_raw(&self) -> &Self::Raw {
        bytemuck::cast_ref(self)
    }

    fn as_raw_mut(&mut self) -> &mut Self::Raw {
        bytemuck::cast_mut(self)
    }
}

impl<T: Unit> FromRawRef for Size2<T> {
    /// By-ref conversion from `Self::Raw`.
    fn from_raw_ref(raw: &Self::Raw) -> &Self {
        bytemuck::cast_ref(raw)
    }

    /// By-ref mutable conversion from `Self::Raw`.
    fn from_raw_mut(raw: &mut Self::Raw) -> &mut Self {
        bytemuck::cast_mut(raw)
    }
}

impl<T: Unit> ToRaw for Size3<T> {
    type Raw = <T::Scalar as Scalar>::Vec3;

    fn to_raw(self) -> Self::Raw {
        bytemuck::cast(self)
    }

    fn from_raw(raw: Self::Raw) -> Self {
        bytemuck::cast(raw)
    }
}

impl<T: Unit> AsRaw for Size3<T> {
    fn as_raw(&self) -> &Self::Raw {
        bytemuck::cast_ref(self)
    }

    fn as_raw_mut(&mut self) -> &mut Self::Raw {
        bytemuck::cast_mut(self)
    }
}

impl<T: Unit> FromRawRef for Size3<T> {
    fn from_raw_ref(raw: &Self::Raw) -> &Self {
        bytemuck::cast_ref(raw)
    }

    fn from_raw_mut(raw: &mut Self::Raw) -> &mut Self {
        bytemuck::cast_mut(raw)
    }
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

crate::impl_vector_common!(Size2 [2] => Vec2 { width, height });

crate::impl_vector_common!(Size3 [3] => Vec3 {
    width,
    height,
    depth
});

macro_rules! impl_size {
    ($base_type_name:ident [ $dimensions:literal ] => $vec_ty:ident, $vector_type:ident) => {
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
    };
}

impl_size!(Size2 [2] => Vec2, Vector2);
impl_size!(Size3 [3] => Vec3, Vector3);

crate::impl_glam_conversion!(Size2 [f32 => glam::Vec2, f64 => glam::DVec2, i32 => glam::IVec2, u32 => glam::UVec2]);
crate::impl_glam_conversion!(Size3 [f32 => glam::Vec3, f64 => glam::DVec3, i32 => glam::IVec3, u32 => glam::UVec3]);

crate::forward_op_to_raw!(Size2, Add<Self>::add -> Self);
crate::forward_op_to_raw!(Size3, Add<Self>::add -> Self);
crate::forward_op_to_raw!(Size2, Sub<Self>::sub -> Self);
crate::forward_op_to_raw!(Size3, Sub<Self>::sub -> Self);
crate::forward_op_to_raw!(Size2, Mul<T::Scalar>::mul -> Self);
crate::forward_op_to_raw!(Size3, Mul<T::Scalar>::mul -> Self);
crate::forward_op_to_raw!(Size2, Div<T::Scalar>::div -> Self);
crate::forward_op_to_raw!(Size3, Div<T::Scalar>::div -> Self);
crate::forward_op_to_raw!(Size2, Rem<T::Scalar>::rem -> Self);
crate::forward_op_to_raw!(Size3, Rem<T::Scalar>::rem -> Self);

crate::forward_op_assign_to_raw!(Size2, AddAssign<Self>::add_assign);
crate::forward_op_assign_to_raw!(Size3, AddAssign<Self>::add_assign);
crate::forward_op_assign_to_raw!(Size2, SubAssign<Self>::sub_assign);
crate::forward_op_assign_to_raw!(Size3, SubAssign<Self>::sub_assign);
crate::forward_op_assign_to_raw!(Size2, MulAssign<T::Scalar>::mul_assign);
crate::forward_op_assign_to_raw!(Size3, MulAssign<T::Scalar>::mul_assign);
crate::forward_op_assign_to_raw!(Size2, DivAssign<T::Scalar>::div_assign);
crate::forward_op_assign_to_raw!(Size3, DivAssign<T::Scalar>::div_assign);
crate::forward_op_assign_to_raw!(Size2, RemAssign<T::Scalar>::rem_assign);
crate::forward_op_assign_to_raw!(Size3, RemAssign<T::Scalar>::rem_assign);

impl<T: Unit> Size2<T> {
    /// All zeroes.
    pub const ZERO: Self = Self {
        width: T::Scalar::ZERO,
        height: T::Scalar::ZERO,
    };

    /// All ones.
    pub const ONE: Self = Self {
        width: T::Scalar::ONE,
        height: T::Scalar::ONE,
    };

    /// New size.
    pub fn new(width: T::Scalar, height: T::Scalar) -> Self {
        Self { width, height }
    }

    /// Calculate the area.
    pub fn area(&self) -> T::Scalar {
        self.width * self.height
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

impl<T> Size2<T>
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

impl<T: Unit> Size3<T> {
    /// All zeroes.
    pub const ZERO: Self = Self {
        width: T::Scalar::ZERO,
        height: T::Scalar::ZERO,
        depth: T::Scalar::ZERO,
    };

    /// All ones.
    pub const ONE: Self = Self {
        width: T::Scalar::ONE,
        height: T::Scalar::ONE,
        depth: T::Scalar::ONE,
    };

    /// New size.
    pub fn new(width: T::Scalar, height: T::Scalar, depth: T::Scalar) -> Self {
        Self {
            width,
            height,
            depth,
        }
    }

    /// Calculate the volume.
    pub fn volume(&self) -> T::Scalar {
        self.width * self.height * self.depth
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

impl<T> Size3<T>
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

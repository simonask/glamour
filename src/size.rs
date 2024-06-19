//! Size vectors

use crate::{
    bindings::prelude::*, rewrap, scalar::FloatScalar, unit::FloatUnit, Scalar, Transparent, Unit,
    Vector2, Vector3,
};

use bytemuck::{Pod, Zeroable};
use num_traits::{ConstOne, ConstZero};

/// 2D size.
#[repr(C)]
pub struct Size2<T: Unit = f32> {
    /// Width
    pub width: T::Scalar,
    /// Height
    pub height: T::Scalar,
}

/// SAFETY: All members are `Pod`, and we are `#[repr(C)]`.
unsafe impl<T: Unit> Pod for Size2<T> {}
/// SAFETY: All members are `Pod`, and we are `#[repr(C)]`.
unsafe impl<T: Unit> Zeroable for Size2<T> {}
unsafe impl<T: Unit> Transparent for Size2<T> {
    type Wrapped = <T::Scalar as Scalar>::Vec2;
}

/// 3D size.
#[repr(C)]
pub struct Size3<T: Unit = f32> {
    /// Width
    pub width: T::Scalar,
    /// Height
    pub height: T::Scalar,
    /// Depth
    pub depth: T::Scalar,
}

/// SAFETY: All members are `Pod`, and we are `#[repr(C)]`.
unsafe impl<T: Unit> Pod for Size3<T> {}
/// SAFETY: All members are `Pod`, and we are `#[repr(C)]`.
unsafe impl<T: Unit> Zeroable for Size3<T> {}
// SAFETY: This is the guarantee of this crate.
unsafe impl<T: Unit> Transparent for Size3<T> {
    type Wrapped = <T::Scalar as Scalar>::Vec3;
}

macro_rules! impl_size {
    ($base_type_name:ident [ $dimensions:tt ] => $vec_ty:ident, $vector_type:ident) => {
        impl<T: Unit> From<$vector_type<T>> for $base_type_name<T> {
            #[inline]
            fn from(vec: $vector_type<T>) -> Self {
                rewrap(vec)
            }
        }

        impl<T: Unit> From<$base_type_name<T>> for $vector_type<T> {
            #[inline]
            fn from(point: $base_type_name<T>) -> Self {
                rewrap(point)
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

            #[doc = "True if any component is zero, or negative, or NaN."]
            pub fn is_empty(self) -> bool {
                let dim: [T::Scalar; $dimensions] = self.into();
                !dim.into_iter().all(|d| d > T::Scalar::ZERO)
            }

            #[doc = "Replace the width component with a new value."]
            #[inline]
            #[must_use]
            pub fn with_width(self, width: T::Scalar) -> Self {
                Self { width, ..self }
            }

            #[doc = "Replace the width component with a new value."]
            #[inline]
            #[must_use]
            pub fn with_height(self, height: T::Scalar) -> Self {
                Self { height, ..self }
            }
        }
    };
}

impl_size!(Size2 [2] => Vec2, Vector2);
impl_size!(Size3 [3] => Vec3, Vector3);

crate::impl_vectorlike::vectorlike!(Size2, 2);
crate::impl_vectorlike::vectorlike!(Size3, 3);

crate::macros::forward_op_to_raw!(Size2, Add<Self>::add -> Self);
crate::macros::forward_op_to_raw!(Size3, Add<Self>::add -> Self);
crate::macros::forward_op_to_raw!(Size2, Sub<Self>::sub -> Self);
crate::macros::forward_op_to_raw!(Size3, Sub<Self>::sub -> Self);
crate::macros::forward_op_to_raw!(Size2, Mul<@T::Scalar>::mul -> Self);
crate::macros::forward_op_to_raw!(Size3, Mul<@T::Scalar>::mul -> Self);
crate::macros::forward_op_to_raw!(Size2, Div<@T::Scalar>::div -> Self);
crate::macros::forward_op_to_raw!(Size3, Div<@T::Scalar>::div -> Self);
crate::macros::forward_op_to_raw!(Size2, Rem<@T::Scalar>::rem -> Self);
crate::macros::forward_op_to_raw!(Size3, Rem<@T::Scalar>::rem -> Self);

crate::macros::forward_op_assign_to_raw!(Size2, AddAssign<Self>::add_assign);
crate::macros::forward_op_assign_to_raw!(Size3, AddAssign<Self>::add_assign);
crate::macros::forward_op_assign_to_raw!(Size2, SubAssign<Self>::sub_assign);
crate::macros::forward_op_assign_to_raw!(Size3, SubAssign<Self>::sub_assign);
crate::macros::forward_op_assign_to_raw!(Size2, MulAssign<@T::Scalar>::mul_assign);
crate::macros::forward_op_assign_to_raw!(Size3, MulAssign<@T::Scalar>::mul_assign);
crate::macros::forward_op_assign_to_raw!(Size2, DivAssign<@T::Scalar>::div_assign);
crate::macros::forward_op_assign_to_raw!(Size3, DivAssign<@T::Scalar>::div_assign);
crate::macros::forward_op_assign_to_raw!(Size2, RemAssign<@T::Scalar>::rem_assign);
crate::macros::forward_op_assign_to_raw!(Size3, RemAssign<@T::Scalar>::rem_assign);

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
    #[inline]
    #[must_use]
    pub fn area(&self) -> T::Scalar {
        self.to_vector().element_product()
    }
}

impl<T: FloatUnit> Size2<T> {
    /// All NaN.
    pub const NAN: Self = Self {
        width: T::Scalar::NAN,
        height: T::Scalar::NAN,
    };
    /// All positive infinity.
    pub const INFINITY: Self = Self {
        width: T::Scalar::INFINITY,
        height: T::Scalar::INFINITY,
    };
    /// All negative infinity.
    pub const NEG_INFINITY: Self = Self {
        width: T::Scalar::NEG_INFINITY,
        height: T::Scalar::NEG_INFINITY,
    };
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
    #[inline]
    #[must_use]
    pub fn volume(&self) -> T::Scalar {
        self.to_vector().element_product()
    }

    /// Replace the depth component with a new value.
    #[inline]
    #[must_use]
    pub fn with_depth(self, depth: T::Scalar) -> Self {
        Self { depth, ..self }
    }
}

impl<T: FloatUnit> Size3<T> {
    /// All NaN.
    pub const NAN: Self = Self {
        width: T::Scalar::NAN,
        height: T::Scalar::NAN,
        depth: T::Scalar::NAN,
    };
    /// All positive infinity.
    pub const INFINITY: Self = Self {
        width: T::Scalar::INFINITY,
        height: T::Scalar::INFINITY,
        depth: T::Scalar::INFINITY,
    };
    /// All negative infinity.
    pub const NEG_INFINITY: Self = Self {
        width: T::Scalar::NEG_INFINITY,
        height: T::Scalar::NEG_INFINITY,
        depth: T::Scalar::NEG_INFINITY,
    };
}

impl<T: Unit> core::fmt::Debug for Size2<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("Size2")
            .field("width", &self.width)
            .field("height", &self.height)
            .finish()
    }
}
impl<T: Unit> core::fmt::Debug for Size3<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        f.debug_struct("Size3")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("depth", &self.depth)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use crate::{peel, size};

    use super::*;

    #[test]
    fn arithmetic() {
        let mut a: Size2<f32> = size!(100.0, 200.0);
        let b: Size2<f32> = a - size!(50.0, 100.0);
        assert_eq!(b, size!(50.0, 100.0));
        a -= size!(50.0, 100.0);
        assert_eq!(a, size!(50.0, 100.0));
        assert_eq!(a + size!(2.0, 3.0), size!(52.0, 103.0));
        a += size!(100.0, 200.0);
        assert_eq!(a, size!(150.0, 300.0));
    }

    #[test]
    fn as_vector() {
        let mut a: Size2<f32> = size!(100.0, 200.0);
        let _: &Vector2<f32> = a.as_vector();
        let _: &mut Vector2<f32> = a.as_vector_mut();

        let mut b: Size3<f32> = size!(100.0, 200.0, 300.0);
        let _: &Vector3<f32> = b.as_vector();
        let _: &mut Vector3<f32> = b.as_vector_mut();
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
    fn is_empty() {
        assert!(Size2::<f32>::ZERO.is_empty());
        assert!(Size3::<f32>::ZERO.is_empty());
        assert!(!Size2::<f32>::ONE.is_empty());
        assert!(!Size3::<f32>::ONE.is_empty());

        assert!(Size2::<f32> {
            width: 0.0,
            height: 1.0
        }
        .is_empty());
        assert!(Size3::<f32> {
            width: 0.0,
            height: 1.0,
            depth: 1.0
        }
        .is_empty());
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

    #[test]
    fn to_raw() {
        let s = Size2::<f32> {
            width: 1.0,
            height: 2.0,
        };
        assert_eq!(peel(s), glam::Vec2 { x: 1.0, y: 2.0 });
        let s = Size3::<f32> {
            width: 1.0,
            height: 2.0,
            depth: 3.0,
        };
        assert_eq!(
            peel(s),
            glam::Vec3 {
                x: 1.0,
                y: 2.0,
                z: 3.0
            }
        );
        assert_eq!(
            Size3::<f32>::wrap(glam::Vec3::new(1.0, 2.0, 3.0)),
            Size3 {
                width: 1.0,
                height: 2.0,
                depth: 3.0
            }
        );
    }

    #[test]
    fn with_width_height_depth() {
        let s = Size2::<f32> {
            width: 1.0,
            height: 2.0,
        };
        assert_eq!(
            s.with_width(3.0),
            Size2::<f32> {
                width: 3.0,
                height: 2.0
            }
        );
        assert_eq!(
            s.with_height(3.0),
            Size2::<f32> {
                width: 1.0,
                height: 3.0
            }
        );

        let s = Size3::<f32> {
            width: 1.0,
            height: 2.0,
            depth: 3.0,
        };
        assert_eq!(
            s.with_width(4.0),
            Size3::<f32> {
                width: 4.0,
                height: 2.0,
                depth: 3.0
            }
        );
        assert_eq!(
            s.with_height(4.0),
            Size3::<f32> {
                width: 1.0,
                height: 4.0,
                depth: 3.0
            }
        );
        assert_eq!(
            s.with_depth(4.0),
            Size3::<f32> {
                width: 1.0,
                height: 2.0,
                depth: 4.0
            }
        );
    }
}

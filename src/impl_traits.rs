macro_rules! impl_basic_traits {
    ($base_type_name:ident, $n:tt) => {
        impl<T: Unit> Clone for $base_type_name<T> {
            #[inline]
            #[must_use]
            #[cfg_attr(coverage, coverage(off))]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<T: Unit> Copy for $base_type_name<T> {}

        impl<T: Unit> PartialEq for $base_type_name<T> {
            fn eq(&self, other: &Self) -> bool {
                *crate::peel_ref(self) == *crate::peel_ref(other)
            }
        }

        impl<T: Unit<Scalar: Eq>> Eq for $base_type_name<T> {}

        crate::impl_traits::impl_basic_traits!(@for_size $base_type_name, $n);
    };
    (@for_size $base_type_name:ident, 2) => {
        impl<T: Unit> Default for $base_type_name<T> {
            #[inline]
            #[must_use]
            fn default() -> Self {
                Self::new(Default::default(), Default::default())
            }
        }
    };
    (@for_size $base_type_name:ident, 3) => {
        impl<T: Unit> Default for $base_type_name<T> {
            #[inline]
            #[must_use]
            fn default() -> Self {
                Self::new(Default::default(), Default::default(), Default::default())
            }
        }
    };
    (@for_size $base_type_name:ident, 4) => {
        impl<T: Unit> Default for $base_type_name<T> {
            #[inline]
            #[must_use]
            fn default() -> Self {
                Self::new(Default::default(), Default::default(), Default::default(), Default::default())
            }
        }
    };
}
pub(crate) use impl_basic_traits;

/// Implements `Clone`, `Copy`, `PartialEq`, etc. for a vectorlike type.
macro_rules! impl_basic_traits_vectorlike {
    ($base_type_name:ident, $n:tt) => {
        crate::impl_traits::impl_basic_traits!($base_type_name, $n);

        impl<T: Unit> AsRef<[T::Scalar; $n]> for $base_type_name<T> {
            #[inline]
            #[must_use]
            fn as_ref(&self) -> &[T::Scalar; $n] {
                self.as_array()
            }
        }
        impl<T: Unit> AsMut<[T::Scalar; $n]> for $base_type_name<T> {
            #[inline]
            #[must_use]
            fn as_mut(&mut self) -> &mut [T::Scalar; $n] {
                self.as_array_mut()
            }
        }

        impl<T: crate::IntUnit> core::hash::Hash for $base_type_name<T> {
            fn hash<H>(&self, state: &mut H)
            where
                H: core::hash::Hasher,
            {
                core::hash::Hash::hash(crate::peel_ref(self), state)
            }
        }

        impl<T: Unit> core::ops::Index<usize> for $base_type_name<T> {
            type Output = T::Scalar;

            #[inline]
            #[must_use]
            fn index(&self, index: usize) -> &Self::Output {
                &self.as_array()[index]
            }
        }

        impl<T: Unit> core::ops::IndexMut<usize> for $base_type_name<T> {
            #[inline]
            #[must_use]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_array_mut()[index]
            }
        }

        impl<T: Unit> From<[T::Scalar; $n]> for $base_type_name<T> {
            #[inline]
            fn from(value: [T::Scalar; $n]) -> Self {
                Self::from_array(value)
            }
        }

        impl<T: Unit> From<$base_type_name<T>> for [T::Scalar; $n] {
            #[inline]
            fn from(value: $base_type_name<T>) -> [T::Scalar; $n] {
                value.to_array()
            }
        }

        impl<T: Unit> IntoIterator for $base_type_name<T> {
            type Item = T::Scalar;
            type IntoIter = <[T::Scalar; $n] as IntoIterator>::IntoIter;

            #[must_use]
            fn into_iter(self) -> Self::IntoIter {
                self.to_array().into_iter()
            }
        }

        impl<T: FloatUnit> approx::AbsDiffEq<Self> for $base_type_name<T> {
            type Epsilon = <T::Scalar as approx::AbsDiffEq>::Epsilon;

            #[must_use]
            fn default_epsilon() -> Self::Epsilon {
                T::Scalar::default_epsilon()
            }

            #[must_use]
            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                crate::peel_ref(self).abs_diff_eq(crate::peel_ref(other), epsilon)
            }

            #[must_use]
            fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                crate::peel_ref(self).abs_diff_ne(crate::peel_ref(other), epsilon)
            }
        }

        impl<T: crate::FloatUnit> approx::RelativeEq<Self> for $base_type_name<T> {
            #[must_use]
            fn default_max_relative() -> Self::Epsilon {
                T::Scalar::default_max_relative()
            }

            #[must_use]
            fn relative_eq(
                &self,
                other: &Self,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                crate::peel_ref(self).relative_eq(crate::peel_ref(other), epsilon, max_relative)
            }

            #[must_use]
            fn relative_ne(
                &self,
                other: &Self,
                epsilon: Self::Epsilon,
                max_relative: Self::Epsilon,
            ) -> bool {
                crate::peel_ref(self).relative_ne(crate::peel_ref(other), epsilon, max_relative)
            }
        }

        impl<T: crate::FloatUnit> approx::UlpsEq<Self> for $base_type_name<T> {
            #[must_use]
            fn default_max_ulps() -> u32 {
                T::Scalar::default_max_ulps()
            }

            #[must_use]
            fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                crate::peel_ref(self).ulps_eq(crate::peel_ref(other), epsilon, max_ulps)
            }

            #[must_use]
            fn ulps_ne(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                crate::peel_ref(self).ulps_ne(crate::peel_ref(other), epsilon, max_ulps)
            }
        }

        crate::impl_traits::impl_basic_traits_vectorlike!(@for_size $base_type_name, $n);
    };
    (@for_size $base_type_name:ident, 2) => {
        crate::impl_traits::impl_basic_traits_vectorlike!(@for_size_and_scalars $base_type_name, 2, {
            f32: glam::Vec2,
            f64: glam::DVec2,
            i32: glam::IVec2,
            u32: glam::UVec2,
            i64: glam::I64Vec2,
            u64: glam::U64Vec2,
            i16: glam::I16Vec2,
            u16: glam::U16Vec2,
        });
        impl<T: Unit> From<(T::Scalar, T::Scalar)> for $base_type_name<T> {
            fn from((x, y): (T::Scalar, T::Scalar)) -> Self {
                Self::new(x, y)
            }
        }
        impl<T: Unit> From<$base_type_name<T>> for (T::Scalar, T::Scalar) {
            fn from(value: $base_type_name<T>) -> Self {
                value.to_tuple()
            }
        }
        impl<T: Unit> PartialEq<(T::Scalar, T::Scalar)> for $base_type_name<T> {
            fn eq(&self, other: &(T::Scalar, T::Scalar)) -> bool {
                self.to_tuple() == *other
            }
        }
    };
    (@for_size $base_type_name:ident, 3) => {
        crate::impl_traits::impl_basic_traits_vectorlike!(@for_size_and_scalars $base_type_name, 3, {
            f32: glam::Vec3,
            f64: glam::DVec3,
            i32: glam::IVec3,
            u32: glam::UVec3,
            i64: glam::I64Vec3,
            u64: glam::U64Vec3,
            i16: glam::I16Vec3,
            u16: glam::U16Vec3,
        });
        impl<T: Unit> From<(T::Scalar, T::Scalar, T::Scalar)> for $base_type_name<T> {
            fn from((x, y, z): (T::Scalar, T::Scalar, T::Scalar)) -> Self {
                Self::new(x, y, z)
            }
        }
        impl<T: Unit> From<$base_type_name<T>> for (T::Scalar, T::Scalar, T::Scalar) {
            fn from(value: $base_type_name<T>) -> Self {
                value.to_tuple()
            }
        }

        impl<T: Unit<Scalar = f32>> From<glam::Vec3A> for $base_type_name<T> {
            fn from(value: glam::Vec3A) -> Self {
                crate::wrap(value.into())
            }
        }
        impl<T: Unit<Scalar = f32>> From<$base_type_name<T>> for glam::Vec3A {
            fn from(value: $base_type_name<T>) -> Self {
                crate::peel(value).into()
            }
        }
    };
    (@for_size $base_type_name:ident, 4) => {
        crate::impl_traits::impl_basic_traits_vectorlike!(@for_size_and_scalars $base_type_name, 4, {
            f32: glam::Vec4,
            f64: glam::DVec4,
            i32: glam::IVec4,
            u32: glam::UVec4,
            i64: glam::I64Vec4,
            u64: glam::U64Vec4,
            i16: glam::I16Vec4,
            u16: glam::U16Vec4,
        });
        impl<T: Unit> From<(T::Scalar, T::Scalar, T::Scalar, T::Scalar)> for $base_type_name<T> {
            fn from((x, y, z, w): (T::Scalar, T::Scalar, T::Scalar, T::Scalar)) -> Self {
                Self::new(x, y, z, w)
            }
        }
        impl<T: Unit> From<$base_type_name<T>> for (T::Scalar, T::Scalar, T::Scalar, T::Scalar) {
            fn from(value: $base_type_name<T>) -> Self {
                value.to_tuple()
            }
        }
    };
    (@for_size_and_scalars $base_type_name:ident, $n:tt, {
        $($scalar:ty: $glam_ty:ty,)*
    }) => {
        $(
            impl<T: Unit<Scalar = $scalar>> From<$glam_ty> for $base_type_name<T> {
                fn from(value: $glam_ty) -> Self {
                    crate::wrap(value)
                }
            }
            impl<T: Unit<Scalar = $scalar>> From<$base_type_name<T>> for $glam_ty {
                fn from(value: $base_type_name<T>) -> Self {
                    crate::peel(value)
                }
            }
            impl<T: Unit<Scalar = $scalar>> core::borrow::Borrow<$glam_ty> for $base_type_name<T>
            {
                fn borrow(&self) -> &$glam_ty {
                    crate::peel_ref(self)
                }
            }
            impl<T: Unit<Scalar = $scalar>> core::borrow::BorrowMut<$glam_ty>
                for $base_type_name<T>
            {
                fn borrow_mut(&mut self) -> &mut $glam_ty {
                    crate::peel_mut(self)
                }
            }
            impl<T: Unit<Scalar = $scalar>> AsRef<$glam_ty> for $base_type_name<T> {
                fn as_ref(&self) -> &$glam_ty {
                    crate::peel_ref(self)
                }
            }
            impl<T: Unit<Scalar = $scalar>> AsMut<$glam_ty> for $base_type_name<T> {
                fn as_mut(&mut self) -> &mut $glam_ty {
                    crate::peel_mut(self)
                }
            }
        )*
    };
}
pub(crate) use impl_basic_traits_vectorlike;

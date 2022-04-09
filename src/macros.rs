#[macro_export]
#[doc(hidden)]
macro_rules! impl_common {
    ($base_type_name:ident {
        $($fields:ident: $fields_ty:ty),*
    }) => {
        impl<T: crate::traits::UnitTypes> $base_type_name<T> {
            #[doc = "Instantiate with field values."]
            #[inline(always)]
            #[must_use]
            pub fn new($($fields: $fields_ty),*) -> $base_type_name<T> {
                $base_type_name {
                    $($fields: $fields.into()),*
                }
            }

            #[doc = "Bitcast an untyped instance to self."]
            #[inline(always)]
            #[must_use]
            pub fn from_untyped(untyped: $base_type_name<T::Primitive>) -> $base_type_name<T> {
                untyped.cast()
            }

            #[doc = "Bitcast self to an untyped instance."]
            #[inline(always)]
            #[must_use]
            pub fn to_untyped(self) -> $base_type_name<T::Primitive> {
                self.cast()
            }

            #[doc = "Reinterpret cast self as the untyped variant."]
            #[inline(always)]
            #[must_use]
            pub fn as_untyped(&self) -> &$base_type_name<T::Primitive> {
                self.cast_ref()
            }

            #[doc = "Reinterpret cast self as the untyped variant."]
            #[inline(always)]
            #[must_use]
            pub fn as_untyped_mut(&mut self) -> &mut $base_type_name<T::Primitive> {
                self.cast_mut()
            }

            #[doc = "Cast to a different coordinate space with the same underlying scalar type."]
            #[inline(always)]
            #[must_use]
            pub fn cast<T2>(self) -> $base_type_name<T2>
            where
                T2: crate::traits::UnitTypes<Primitive = T::Primitive>,
            {
                bytemuck::cast(self)
            }

            #[doc = "Cast to a different coordinate space with the same underlying scalar type."]
            #[inline(always)]
            #[must_use]
            pub fn cast_ref<T2>(&self) -> &$base_type_name<T2>
            where
                T2: crate::traits::UnitTypes<Primitive = T::Primitive>,
            {
                bytemuck::cast_ref(self)
            }

            #[doc = "Cast to a different coordinate space with the same underlying scalar type."]
            #[inline(always)]
            #[must_use]
            pub fn cast_mut<T2>(&mut self) -> &mut $base_type_name<T2>
            where
                T2: crate::traits::UnitTypes<Primitive = T::Primitive>,
            {
                bytemuck::cast_mut(self)
            }

            #[doc = "Cast to a different coordinate space with scalar type conversion. Returns `None` if any component could not be converted to the target scalar type."]
            #[must_use]
            pub fn try_cast<T2>(self) -> Option<$base_type_name<T2>>
            where
                T2: Unit,
            {
                $(
                    let $fields = self.$fields.try_cast()?;
                )*
                Some($base_type_name { $($fields),* })
            }
        }

        impl<T: Unit> Clone for $base_type_name<T> {
            #[inline(always)]
            #[must_use]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<T: Unit> Copy for $base_type_name<T> {}

        impl<T: Unit> Default for $base_type_name<T> {
            #[inline(always)]
            #[must_use]
            fn default() -> Self {
                $base_type_name {
                    $($fields: Default::default()),*
                }
            }
        }

        impl<T: Unit> core::fmt::Debug for $base_type_name<T> {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                f.write_str(stringify!($base_type_name))?;
                if let Some(unit_name) = T::name() {
                    use core::fmt::Write;
                    f.write_char('<')?;
                    f.write_str(unit_name)?;
                    f.write_char('>')?
                }
                let mut d = f.debug_struct("");
                $(
                    d.field(stringify!($fields), &self.$fields);
                )*
                d.finish()
            }
        }

        /// SAFETY: Vector types only contain members of type `T::Scalar`, which is required to be [`Zeroable`](bytemuck::Zeroable).
        unsafe impl<T: crate::traits::Unit> bytemuck::Zeroable for $base_type_name<T> where $($fields_ty: bytemuck::Zeroable),* {}
        /// SAFETY: Vector types only contain members of type `T::Scalar`, which is required to be [`Pod`](bytemuck::Pod).
        unsafe impl<T: crate::traits::Unit> bytemuck::Pod for $base_type_name<T> where $($fields_ty: bytemuck::Pod),* {}

        impl<T: Unit> PartialEq for $base_type_name<T> {
            #[inline(always)]
            #[must_use]
            fn eq(&self, other: &Self) -> bool {
                ($(self.$fields),*) == ($(other.$fields),*)
            }
        }

        impl<T> Eq for $base_type_name<T>
        where
            T: Unit,
            T::Scalar: Eq
        {}

        impl<T: Unit> approx::AbsDiffEq<Self> for $base_type_name<T>
        where
            T::Scalar: approx::AbsDiffEq,
            <T::Scalar as approx::AbsDiffEq>::Epsilon: Copy
        {
            type Epsilon = <T::Scalar as approx::AbsDiffEq>::Epsilon;

            #[must_use]
            fn default_epsilon() -> Self::Epsilon {
                T::Scalar::default_epsilon()
            }

            #[must_use]
            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                $(self.$fields.abs_diff_eq(&other.$fields, epsilon) && )* true
            }

            #[must_use]
            fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                $(self.$fields.abs_diff_ne(&other.$fields, epsilon) || )* false
            }
        }

        impl<T: Unit> approx::RelativeEq<Self> for $base_type_name<T>
        where
            T::Scalar: approx::RelativeEq,
            <T::Scalar as approx::AbsDiffEq>::Epsilon: Copy
        {
            #[must_use]
            fn default_max_relative() -> Self::Epsilon {
                T::Scalar::default_max_relative()
            }

            #[must_use]
            fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
                $(self.$fields.relative_eq(&other.$fields, epsilon, max_relative) && )* true
            }

            #[must_use]
            fn relative_ne(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
                $(self.$fields.relative_ne(&other.$fields, epsilon, max_relative) || )* false
            }
        }

        impl<T: Unit> approx::UlpsEq<Self> for $base_type_name<T>
        where
            T::Scalar: approx::UlpsEq,
            <T::Scalar as approx::AbsDiffEq>::Epsilon: Copy
        {
            #[must_use]
            fn default_max_ulps() -> u32 {
                T::Scalar::default_max_ulps()
            }

            #[must_use]
            fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                $(self.$fields.ulps_eq(&other.$fields, epsilon, max_ulps) && )* true
            }

            #[must_use]
            fn ulps_ne(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                $(self.$fields.ulps_ne(&other.$fields, epsilon, max_ulps) || )* false
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_as_tuple {
    ($base_type_name:ident {
        $($fields:ident: $fields_ty:ty),*
    }) => {
        impl<T: Unit> $base_type_name<T> {
            #[doc = "Instantiate from tuple."]
            #[allow(unused_parens)]
            #[inline]
            #[must_use]
            pub fn from_tuple(($($fields),*): ($($fields_ty),*)) -> Self {
                $base_type_name { $($fields),* }
            }

            #[doc = "Convert to tuple."]
            #[inline]
            #[must_use]
            pub fn to_tuple(self) -> ($($fields_ty),*) {
                ($(self.$fields),*)
            }
        }

        impl<T: Unit> From<($($fields_ty),*)> for $base_type_name<T> {
            #[inline]
            #[allow(unused_parens)]
            #[must_use]
            fn from(($($fields),*): ($($fields_ty),*)) -> $base_type_name<T> {
                $base_type_name { $($fields),* }
            }
        }

        impl<T: Unit> From<$base_type_name<T>> for ($($fields_ty),*) {
            #[inline]
            #[must_use]
            fn from(value: $base_type_name<T>) -> ($($fields_ty),*) {
                let $base_type_name {
                    $($fields),*
                } = value;
                ($($fields),*)
            }
        }

        impl<T: Unit> PartialEq<($($fields_ty),*)> for $base_type_name<T> {
            #[inline(always)]
            #[must_use]
            fn eq(&self, ($($fields),*): &($($fields_ty),*)) -> bool {
                Self {
                    $(
                        $fields: *$fields
                    ),*
                } == *self
            }
        }

        impl<T> approx::AbsDiffEq<($($fields_ty),*)> for $base_type_name<T>
        where
            T: Unit,
            T::Scalar: approx::AbsDiffEq,
            <T::Scalar as approx::AbsDiffEq>::Epsilon: Copy
        {
            type Epsilon = <T::Scalar as approx::AbsDiffEq>::Epsilon;

            #[must_use]
            fn default_epsilon() -> Self::Epsilon {
                T::Scalar::default_epsilon()
            }

            #[must_use]
            fn abs_diff_eq(&self, other: &($($fields_ty),*), epsilon: Self::Epsilon) -> bool {
                self.abs_diff_eq(&Self::from_tuple(*other), epsilon)
            }

            #[must_use]
            fn abs_diff_ne(&self, other: &($($fields_ty),*), epsilon: Self::Epsilon) -> bool {
                self.abs_diff_ne(&Self::from_tuple(*other), epsilon)
            }
        }

        impl<T> approx::RelativeEq<($($fields_ty),*)> for $base_type_name<T>
        where
            T: Unit,
            T::Scalar: approx::RelativeEq,
            <T::Scalar as approx::AbsDiffEq>::Epsilon: Copy
        {
            #[must_use]
            fn default_max_relative() -> Self::Epsilon {
                T::Scalar::default_max_relative()
            }

            #[must_use]
            fn relative_eq(&self, other: &($($fields_ty),*), epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
                self.relative_eq(&Self::from_tuple(*other), epsilon, max_relative)
            }

            #[must_use]
            fn relative_ne(&self, other: &($($fields_ty),*), epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
                self.relative_ne(&Self::from_tuple(*other), epsilon, max_relative)
            }
        }

        impl<T: Unit> approx::UlpsEq<($($fields_ty),*)> for $base_type_name<T>
        where
            T::Scalar: approx::UlpsEq,
            <T::Scalar as approx::AbsDiffEq>::Epsilon: Copy
        {
            #[must_use]
            fn default_max_ulps() -> u32 {
                T::Scalar::default_max_ulps()
            }

            #[must_use]
            fn ulps_eq(&self, other: &($($fields_ty),*), epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                self.ulps_eq(&Self::from_tuple(*other), epsilon, max_ulps)
            }

            #[must_use]
            fn ulps_ne(&self, other: &($($fields_ty),*), epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                self.ulps_ne(&Self::from_tuple(*other), epsilon, max_ulps)
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_simd_common {
    ($base_type_name:ident [$dimensions:literal] => $vec_ty:ident, $mask_ty:ty {
        $($fields:ident),*
    }) => {
        impl<T: crate::traits::UnitTypes> $base_type_name<T> {
            #[doc = "Get the underlying `glam` vector."]
            #[inline(always)]
            #[must_use]
            pub fn to_raw(self) -> T::$vec_ty {
                bytemuck::cast(self)
            }

            #[doc = "Create new instance from `glam` vector."]
            #[inline(always)]
            #[must_use]
            pub fn from_raw(raw: T::$vec_ty) -> Self {
                bytemuck::cast(raw)
            }

            #[doc = "Reinterpret as the underlying `glam` vector."]
            #[doc = ""]
            #[doc = "Note: The corresponding \"from_raw_ref\" method does not exist"]
            #[doc = "because some `glam` vector types have smaller alignment than"]
            #[doc = "these vector types, so the cast may fail. If you are sure this"]
            #[doc = "isn't the case, you can use [`bytemuck::cast_ref()`] directly."]
            #[inline(always)]
            #[must_use]
            pub fn as_raw(&self) -> &T::$vec_ty {
                bytemuck::cast_ref(self)
            }

            #[doc = "Reinterpret as the underlying `glam` vector."]
            #[inline(always)]
            #[must_use]
            pub fn as_raw_mut(&mut self) -> &mut T::$vec_ty {
                bytemuck::cast_mut(self)
            }

            #[doc = "Instantiate from array."]
            #[inline(always)]
            #[must_use]
            pub fn from_array([$($fields),*]: [T::Scalar; $dimensions]) -> Self {
                Self { $($fields),* }
            }

            #[doc = "Convert to array."]
            #[inline(always)]
            #[must_use]
            pub fn to_array(self) -> [T::Scalar; $dimensions] {
                let Self { $($fields),* } = self;
                [$($fields),*]
            }

            #[doc = "Reinterpret as array."]
            #[inline(always)]
            #[must_use]
            pub fn as_array(&self) -> &[T::Scalar; $dimensions] {
                bytemuck::cast_ref(self)
            }

            #[doc = "Reinterpret as mutable array."]
            #[inline(always)]
            #[must_use]
            pub fn as_array_mut(&mut self) -> &mut [T::Scalar; $dimensions] {
                bytemuck::cast_mut(self)
            }

            #[doc = "Instance with all zeroes."]
            #[inline(always)]
            #[must_use]
            pub fn zero() -> Self {
                use crate::traits::SimdVec;
                Self::from_raw(T::$vec_ty::zero())
            }

            #[doc = "Instance with all ones."]
            #[inline(always)]
            #[must_use]
            pub fn one() -> Self {
                use crate::traits::SimdVec;
                Self::from_raw(T::$vec_ty::one())
            }

            #[doc = "Instance with all components set to `scalar`."]
            #[inline]
            #[must_use]
            pub fn splat(scalar: T::Scalar) -> Self {
                $(
                    let $fields = scalar;
                )*
                Self { $($fields),* }
            }

            #[doc = "Component-wise clamp."]
            #[inline]
            #[must_use]
            pub fn clamp(self, min: Self, max: Self) -> Self {
                use crate::traits::SimdVec;
                Self::from_raw(self.to_raw().clamp(min.to_raw(), max.to_raw()))
            }

            #[doc = "Get component at `index`."]
            #[inline]
            #[must_use]
            pub fn get(&self, index: usize) -> T::Scalar {
                self.as_array()[index]
            }

            #[doc = "Set component at `index`."]
            #[inline]
            pub fn set(&mut self, index: usize, value: T::Scalar) {
                self.as_array_mut()[index] = value;
            }

            #[doc = "Get component at index `N`."]
            #[inline(always)]
            #[must_use]
            pub fn const_get<const N: usize>(&self) -> T::Scalar {
                self.as_array()[N]
            }

            #[doc = "Set component at index `N`."]
            #[inline(always)]
            pub fn const_set<const N: usize>(&mut self, value: T::Scalar) {
                self.as_array_mut()[N] = value;
            }

            #[doc = "Return a mask with the result of a component-wise equals comparison."]
            #[inline(always)]
            #[must_use]
            pub fn cmpeq(self, other: Self) -> $mask_ty {
                use crate::traits::SimdVec;
                self.to_raw().cmpeq(other.to_raw()).into()
            }

            #[doc = "Return a mask with the result of a component-wise not-equal comparison."]
            #[inline(always)]
            #[must_use]
            pub fn cmpne(self, other: Self) -> $mask_ty {
                use crate::traits::SimdVec;
                self.to_raw().cmpne(other.to_raw()).into()
            }

            #[doc = "Return a mask with the result of a component-wise greater-than-or-equal comparison."]
            #[inline(always)]
            #[must_use]
            pub fn cmpge(self, other: Self) -> $mask_ty {
                use crate::traits::SimdVec;
                self.to_raw().cmpge(other.to_raw()).into()
            }

            #[doc = "Return a mask with the result of a component-wise greater-than comparison."]
            #[inline(always)]
            #[must_use]
            pub fn cmpgt(self, other: Self) -> $mask_ty {
                use crate::traits::SimdVec;
                self.to_raw().cmpgt(other.to_raw()).into()
            }

            #[doc = "Return a mask with the result of a component-wise less-than-or-equal comparison."]
            #[inline(always)]
            #[must_use]
            pub fn cmple(self, other: Self) -> $mask_ty {
                use crate::traits::SimdVec;
                self.to_raw().cmple(other.to_raw()).into()
            }

            #[doc = "Return a mask with the result of a component-wise less-than comparison."]
            #[inline(always)]
            #[must_use]
            pub fn cmplt(self, other: Self) -> $mask_ty {
                use crate::traits::SimdVec;
                self.to_raw().cmplt(other.to_raw()).into()
            }

            #[doc = "Minimum by component."]
            #[inline(always)]
            #[must_use]
            pub fn min(self, other: Self) -> Self {
                use crate::traits::SimdVec;
                Self::from_raw(self.to_raw().min(other.to_raw()))
            }

            #[doc = "Maximum by component."]
            #[inline(always)]
            #[must_use]
            pub fn max(self, other: Self) -> Self {
                use crate::traits::SimdVec;
                Self::from_raw(self.to_raw().max(other.to_raw()))
            }

            #[doc = "Horizontal minimum (smallest component)."]
            #[inline]
            #[must_use]
            pub fn min_element(self) -> T::Scalar {
                use crate::traits::SimdVec;
                T::Scalar::from_raw(self.to_raw().min_element())
            }

            #[doc = "Horizontal maximum (largest component)."]
            #[inline]
            #[must_use]
            pub fn max_element(self) -> T::Scalar {
                use crate::traits::SimdVec;
                T::Scalar::from_raw(self.to_raw().max_element())
            }

            #[doc = "Select components from two instances based on a mask."]
            #[inline(always)]
            #[must_use]
            pub fn select(mask: $mask_ty, if_true: Self, if_false: Self) -> Self {
                use crate::traits::SimdVec;
                Self::from_raw(T::$vec_ty::select(mask.into(), if_true.to_raw(), if_false.to_raw()))
            }
        }

        impl<T> $base_type_name<T>
        where
            T: crate::traits::UnitTypes,
            T::$vec_ty: crate::traits::SimdVecFloat<$dimensions>,
        {
            #[doc = "Return an instance where all components are NaN."]
            #[inline(always)]
            #[must_use]
            pub fn nan() -> Self {
                use crate::traits::SimdVecFloat;
                Self::from_raw(T::$vec_ty::nan())
            }
            #[doc = "True if all components are non-infinity and non-NaN."]
            #[inline(always)]
            #[must_use]
            pub fn is_finite(&self) -> bool {
                use crate::traits::SimdVecFloat;
                self.as_raw().is_finite()
            }
            #[doc = "True if any component is NaN."]
            #[inline(always)]
            #[must_use]
            pub fn is_nan(&self) -> bool {
                use crate::traits::SimdVecFloat;
                self.as_raw().is_nan()
            }
            #[doc = "Return a mask where each bit is set if the corresponding component is NaN."]
            #[inline(always)]
            #[must_use]
            pub fn is_nan_mask(&self) -> $mask_ty {
                use crate::traits::SimdVecFloat;
                self.as_raw().is_nan_mask().into()
            }

            #[doc = "Round all components up."]
            #[inline(always)]
            #[must_use]
            pub fn ceil(self) -> Self {
                use crate::traits::SimdVecFloat;
                Self::from_raw(self.to_raw().ceil())
            }
            #[doc = "Round all components down."]
            #[inline(always)]
            #[must_use]
            pub fn floor(self) -> Self {
                use crate::traits::SimdVecFloat;
                Self::from_raw(self.to_raw().floor())
            }
            #[doc = "Round all components."]
            #[inline(always)]
            #[must_use]
            pub fn round(self) -> Self {
                use crate::traits::SimdVecFloat;
                Self::from_raw(self.to_raw().round())
            }
        }

        impl<T> $base_type_name<T>
        where
            T: crate::traits::UnitTypes,
            T::$vec_ty: crate::traits::Abs,
        {
            #[doc = "Computes the absolute value of each component."]
            #[inline]
            #[must_use]
            pub fn abs(self) -> Self {
                use crate::traits::Abs;
                Self::from_raw(self.to_raw().abs())
            }
        }

        impl<T: Unit> core::ops::Index<usize> for $base_type_name<T> {
            type Output = T::Scalar;

            #[inline(always)]
            #[must_use]
            fn index(&self, index: usize) -> &Self::Output {
                &self.as_array()[index]
            }
        }

        impl<T: Unit> core::ops::IndexMut<usize> for $base_type_name<T> {
            #[inline(always)]
            #[must_use]
            fn index_mut(&mut self, index: usize) -> &mut Self::Output {
                &mut self.as_array_mut()[index]
            }
        }

        impl<T: Unit> AsRef<[T::Scalar; $dimensions]> for $base_type_name<T>
        {
            #[inline(always)]
            #[must_use]
            fn as_ref(&self) -> &[T::Scalar; $dimensions] {
                bytemuck::cast_ref(self)
            }
        }

        impl<T: Unit> From<[T::Scalar; $dimensions]> for $base_type_name<T> {
            #[inline(always)]
            #[must_use]
            fn from(arr: [T::Scalar; $dimensions]) -> Self {
                Self::from_array(arr)
            }
        }

        impl<T: Unit> From<$base_type_name<T>> for [T::Scalar; $dimensions] {
            #[inline(always)]
            #[must_use]
            fn from(v: $base_type_name<T>) -> Self {
                v.to_array()
            }
        }

        impl<T: Unit> AsMut<[T::Scalar; $dimensions]> for $base_type_name<T>
        where
            T::Scalar: bytemuck::Pod
        {
            #[inline(always)]
            #[must_use]
            fn as_mut(&mut self) -> &mut [T::Scalar; $dimensions] {
                bytemuck::cast_mut(self)
            }
        }

        impl<T: Unit> AsRef<[T::Scalar]> for $base_type_name<T>
        where
            T::Scalar: bytemuck::Pod
        {
            #[inline(always)]
            #[must_use]
            fn as_ref(&self) -> &[T::Scalar] {
                AsRef::<[T::Scalar; $dimensions]>::as_ref(self)
            }
        }

        impl<T: Unit> AsMut<[T::Scalar]> for $base_type_name<T>
        where
            T::Scalar: bytemuck::Pod
        {
            #[inline(always)]
            #[must_use]
            fn as_mut(&mut self) -> &mut [T::Scalar] {
                AsMut::<[T::Scalar; $dimensions]>::as_mut(self)
            }
        }

        impl<T: Unit> PartialEq<[T::Scalar; $dimensions]> for $base_type_name<T> {
            #[inline(always)]
            #[must_use]
            fn eq(&self, other: &[T::Scalar; $dimensions]) -> bool {
                self.to_array() == *other
            }
        }

        impl<T: Unit> IntoIterator for $base_type_name<T> {
            type Item = T::Scalar;
            type IntoIter = <[T::Scalar; $dimensions] as IntoIterator>::IntoIter;

            #[must_use]
            fn into_iter(self) -> Self::IntoIter {
                self.to_array().into_iter()
            }
        }

        impl<T> core::ops::Neg for $base_type_name<T>
        where
            T: crate::traits::UnitTypes,
            T::$vec_ty: core::ops::Neg<Output = T::$vec_ty>,
        {
            type Output = Self;

            #[inline(always)]
            #[must_use]
            fn neg(self) -> Self::Output {
                Self::from_raw(-self.to_raw())
            }
        }
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_glam_conversion {
    ($base_type_name:ident, $dimensions:literal [
        $(
            $scalar:ty => $glam_type:ty
        ),*
    ]) => {
        $(
            impl<T> From<$base_type_name<T>> for $glam_type
            where
                T: Unit,
                T::Scalar: crate::Scalar<Primitive = $scalar>
            {
                #[inline(always)]
                #[must_use]
                fn from(vec: $base_type_name<T>) -> Self {
                    vec.to_raw()
                }
            }

            impl<T> From<$glam_type> for $base_type_name<T>
            where
                T: Unit,
                T::Scalar: crate::Scalar<Primitive = $scalar>
            {
                #[inline(always)]
                #[must_use]
                fn from(vec: $glam_type) -> Self {
                    Self::from_raw(vec)
                }
            }

            impl<T> AsRef<$glam_type> for $base_type_name<T>
            where
                T: Unit,
                T::Scalar: crate::Scalar<Primitive = $scalar>
            {
                #[inline(always)]
                #[must_use]
                fn as_ref(&self) -> &$glam_type {
                    bytemuck::cast_ref(self)
                }
            }

            impl<T> AsMut<$glam_type> for $base_type_name<T>
            where
                T: Unit,
                T::Scalar: crate::Scalar<Primitive = $scalar>
            {
                #[inline(always)]
                #[must_use]
                fn as_mut(&mut self) -> &mut $glam_type {
                    bytemuck::cast_mut(self)
                }
            }
        )*
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_scaling {
    ($base_type_name:ident, $dimensions:literal [
        $(
            $scalar:ty
        ),*
    ]) => {
        $(
            impl<T> core::ops::Mul<$scalar> for $base_type_name<T>
            where
                T: Unit,
                T::Scalar: crate::Scalar<Primitive = $scalar> + core::ops::Mul<$scalar, Output = T::Scalar>
            {
                type Output = Self;

                #[inline(always)]
                #[must_use]
                fn mul(self, rhs: $scalar) -> Self::Output {
                    Self::from_raw(self.to_raw() * rhs)
                }
            }

            impl<T> core::ops::MulAssign<$scalar> for $base_type_name<T>
            where
                T: Unit,
                T::Scalar: crate::Scalar<Primitive = $scalar> + core::ops::MulAssign<$scalar>
            {
                #[inline(always)]
                fn mul_assign(&mut self, rhs: $scalar) {
                    *self.as_raw_mut() *= rhs;
                }
            }

            impl<T> core::ops::Div<$scalar> for $base_type_name<T>
            where
                T: Unit,
                T::Scalar: crate::Scalar<Primitive = $scalar> + core::ops::Div<$scalar, Output = T::Scalar>
            {
                type Output = Self;

                #[inline(always)]
                #[must_use]
                fn div(self, rhs: $scalar) -> Self::Output {
                    Self::from_raw(self.to_raw() / rhs)
                }
            }

            impl<T> core::ops::DivAssign<$scalar> for $base_type_name<T>
            where
                T: Unit,
                T::Scalar: crate::Scalar<Primitive = $scalar> + core::ops::DivAssign<$scalar>
            {
                #[inline(always)]
                fn div_assign(&mut self, rhs: $scalar) {
                    *self.as_raw_mut() /= rhs;
                }
            }

            impl<T> core::ops::Rem<$scalar> for $base_type_name<T>
            where
                T: Unit,
                T::Scalar: crate::Scalar<Primitive = $scalar> + core::ops::Rem<$scalar, Output = T::Scalar>
            {
                type Output = Self;

                #[inline(always)]
                #[must_use]
                fn rem(self, rhs: $scalar) -> Self::Output {
                    Self::from_raw(self.to_raw() % rhs)
                }
            }

            impl<T> core::ops::RemAssign<$scalar> for $base_type_name<T>
            where
                T: Unit,
                T::Scalar: crate::Scalar<Primitive = $scalar> + core::ops::RemAssign<$scalar>
            {
                #[inline(always)]
                fn rem_assign(&mut self, rhs: $scalar) {
                    *self.as_raw_mut() %= rhs;
                }
            }
        )*
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! impl_mint {
    ($base_type_name:ident, $dimensions:literal, $mint_type:ident) => {
        #[cfg(feature = "mint")]
        const _: () = {
            impl<T: Scalar + Unit<Scalar = T>> mint::IntoMint for $base_type_name<T> {
                type MintType = mint::$mint_type<T>;
            }

            impl<T: Scalar + Unit<Scalar = T>> From<mint::$mint_type<T>> for $base_type_name<T> {
                fn from(x: mint::$mint_type<T>) -> $base_type_name<T> {
                    $base_type_name::from_array(*x.as_ref())
                }
            }

            impl<T: Scalar + Unit<Scalar = T>> From<$base_type_name<T>> for mint::$mint_type<T> {
                fn from(x: $base_type_name<T>) -> mint::$mint_type<T> {
                    x.to_array().into()
                }
            }
        };
    };
}

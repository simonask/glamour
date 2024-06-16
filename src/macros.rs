#![allow(clippy::doc_markdown)]

/// Define methods and provide implementations that forward to the method of the
/// same name on the type's `<Self as ToRaw>::Raw` type, converting all
/// arguments and the return value through `ToRaw`.
macro_rules! forward_to_raw {
    (
        $see_also_doc_ty:ty =>
        $(
            $(#[
                $($attr:meta),*
            ])*
            $visibility:vis
            fn $fn_name:ident (
                $($args:tt)*
            )
            $(-> $ret_ty:ty)?
            ;
        )*
    ) => {
        $(
            crate::forward_to_raw_impl!(
                $(#[$($attr),*])*
                #[doc = concat!("See [`", stringify!($see_also_doc_ty), "::", stringify!($fn_name), "()`].")]
                $visibility fn $fn_name($($args)*) $(-> $ret_ty)*;
            );
        )*
    };
}

macro_rules! forward_to_raw_impl {
    // self (by value)
    (
        $(#[$($attr:meta),*])*
        $visibility:vis
        fn $fn_name:ident (
            self
            $(, $arg_name:ident: $arg_ty:ty)* $(,)?
        ) -> $ret_ty:ty;
    ) => {
        #[inline]
        #[must_use]
        $(#[$($attr),*])*
        $visibility
        fn $fn_name(self $(, $arg_name: $arg_ty)*) -> $ret_ty {
            crate::wrap(crate::peel(self).$fn_name($(crate::peel($arg_name)),*))
        }
    };

    // self (by value), do not wrap return
    (
        $(#[$($attr:meta),*])*
        $visibility:vis
        fn $fn_name:ident (
            self
            $(, $arg_name:ident: $arg_ty:ty)* $(,)?
        ) -> @ $ret_ty:ty;
    ) => {
        #[inline]
        #[must_use]
        $(#[$($attr),*])*
        $visibility
        fn $fn_name(self $(, $arg_name: $arg_ty)*) -> $ret_ty {
            crate::peel(self).$fn_name($(crate::peel($arg_name)),*)
        }
    };

    // &self (by reference)
    (
        $(#[$($attr:meta),*])*
        $visibility:vis
        fn $fn_name:ident (
            &self
            $(, $arg_name:ident: $arg_ty:ty)*  $(,)?
        ) -> $ret_ty:ty;
    ) => {
        #[inline]
        #[must_use]
        $(#[$($attr),*])*
        $visibility
        fn $fn_name(&self, $($arg_name: $arg_ty),*) -> $ret_ty {
            crate::wrap(crate::peel_ref(self).$fn_name($(crate::peel($arg_name)),*))
        }
    };

    // &mut self (by mutable reference)
    (
        $(#[$($attr:meta),*])*
        $visibility:vis
        fn $fn_name:ident (
            &mut self
            $(, $arg_name:ident: $arg_ty:ty)*  $(,)?
        ) -> $ret_ty:ty;
    ) => {
        #[inline]
        #[must_use]
        $(#[$($attr),*])*
        $visibility
        fn $fn_name(&mut self, $(arg_name: $arg_ty),*) -> $ret_ty {
            <$ret_ty>::from_raw(self.as_raw_mut().$fn_name($($arg_name.to_raw()),*))
        }
    };

    // &mut self, no return
    (
        $(#[$($attr:meta),*])*
        $visibility:vis
        fn $fn_name:ident (
            &mut self
            $(, $arg_name:ident: $arg_ty:ty)*  $(,)?
        );
    ) => {
        #[inline]
        $(#[$($attr),*])*
        $visibility
        fn $fn_name(&mut self, $(arg_name: $arg_ty),*) {
            self.as_raw_mut().$fn_name($($arg_name.to_raw()),*)
        }
    };
}

/// Generate a `core::ops::*` implementation, delegating to underlying raw
/// types.
macro_rules! forward_op_to_raw {
    // Implement operation for specific scalar types.
    ($base_type_name:ident, $op_name:ident < [$($scalars:ty),+] > :: $op_fn_name:ident -> $output:ty) => {
        $(
            impl<T: Unit<Scalar = $scalars>> core::ops::$op_name<$scalars> for $base_type_name<T> {
                type Output = $output;

                #[inline]
                fn $op_fn_name(self, other: $scalars) -> $output {
                    crate::wrap(core::ops::$op_name::$op_fn_name(
                        crate::peel(self),
                        other,
                    ))
                }
            }
        )*
    };

    // Implement operation for a generic non-scalar argument.
    ($base_type_name:ident, $op_name:ident < $arg_ty:ty > :: $op_fn_name:ident -> $output:ty) => {
        impl<T: Unit> core::ops::$op_name<$arg_ty> for $base_type_name<T> {
            type Output = $output;

            #[inline]
            fn $op_fn_name(self, other: $arg_ty) -> $output {
                crate::wrap(core::ops::$op_name::$op_fn_name(
                    crate::peel(self),
                    crate::peel(other),
                ))
            }
        }
    };

    // Implement operation for a generic non-scalar argument, without peeling the rhs.
    ($base_type_name:ident, $op_name:ident < @ $arg_ty:ty > :: $op_fn_name:ident -> $output:ty) => {
        impl<T: Unit> core::ops::$op_name<$arg_ty> for $base_type_name<T> {
            type Output = $output;

            #[inline]
            fn $op_fn_name(self, other: $arg_ty) -> $output {
                crate::wrap(core::ops::$op_name::$op_fn_name(
                    crate::peel(self),
                    other,
                ))
            }
        }
    };
}

/// Similar to `forward_op_to_raw!`, but for assigning ops.
macro_rules! forward_op_assign_to_raw {
    ($base_type_name:ident, $op_name:ident < [$($scalars:ty),+] > :: $op_fn_name:ident) => {
        $(
            impl<T: Unit<Scalar = $scalars>> core::ops::$op_name<$scalars> for $base_type_name<T> {
                #[inline]
                fn $op_fn_name(&mut self, other: $scalars) {
                    core::ops::$op_name::$op_fn_name(crate::peel_mut(self), other)
                }
            }
        )*
    };

    ($base_type_name:ident, $op_name:ident < $arg_ty:ty > :: $op_fn_name:ident) => {
        impl<T: Unit> core::ops::$op_name<$arg_ty> for $base_type_name<T> {
            #[inline]
            fn $op_fn_name(&mut self, other: $arg_ty) {
                core::ops::$op_name::$op_fn_name(crate::peel_mut(self), crate::peel(other))
            }
        }
    };


    ($base_type_name:ident, $op_name:ident < @ $arg_ty:ty > :: $op_fn_name:ident) => {
        impl<T: Unit> core::ops::$op_name<$arg_ty> for $base_type_name<T> {
            #[inline]
            fn $op_fn_name(&mut self, other: $arg_ty) {
                core::ops::$op_name::$op_fn_name(crate::peel_mut(self), other)
            }
        }
    };
}

macro_rules! forward_neg_to_raw {
    ($base_type_name:ident) => {
        impl<T> core::ops::Neg for $base_type_name<T>
        where
            T: Unit<Scalar: SignedScalar>,
        {
            type Output = Self;

            fn neg(self) -> Self {
                crate::wrap(core::ops::Neg::neg(crate::peel(self)))
            }
        }
    };
}

/// from_array, splat, etc.
macro_rules! forward_constructors {
    ($dimensions:literal, $raw:ident) => {
        #[doc = "Instantiate from array."]
        pub fn from_array(array: [T::Scalar; $dimensions]) -> Self {
            crate::wrap(<<T::Scalar as Scalar>::$raw>::from_array(array))
        }
        #[doc = "Convert to array."]
        pub fn to_array(self) -> [T::Scalar; $dimensions] {
            <<T::Scalar as Scalar>::$raw>::to_array(crate::peel_ref(&self))
        }

        #[doc = "Instance with all components set to `scalar`."]
        pub fn splat(scalar: T::Scalar) -> Self {
            crate::wrap(<<T::Scalar as Scalar>::$raw>::splat(scalar))
        }

        #[doc = "Write all components to slice."]
        pub fn write_to_slice(self, slice: &mut [T::Scalar]) {
            crate::peel(self).write_to_slice(slice);
        }
    };
}

/// cmpeq etc.
macro_rules! forward_comparison {
    ($mask:ty, $t:ident) => {
        #[doc = "Return a mask with the result of a component-wise equals comparison."]
        pub fn cmpeq(self, other: Self) -> $mask {
            crate::peel(self).cmpeq(crate::peel(other))
        }
        #[doc = "Return a mask with the result of a component-wise not-equal comparison."]
        pub fn cmpne(self, other: Self) -> $mask {
            crate::peel(self).cmpne(crate::peel(other))
        }
        #[doc = "Return a mask with the result of a component-wise greater-than-or-equal comparison."]
        pub fn cmpge(self, other: Self) -> $mask {
            crate::peel(self).cmpge(crate::peel(other))
        }
        #[doc = "Return a mask with the result of a component-wise greater-than comparison."]
        pub fn cmpgt(self, other: Self) -> $mask {
            crate::peel(self).cmpgt(crate::peel(other))
        }
        #[doc = "Return a mask with the result of a component-wise less-than-or-equal comparison."]
        pub fn cmple(self, other: Self) -> $mask {
            crate::peel(self).cmple(crate::peel(other))
        }
        #[doc = "Return a mask with the result of a component-wise less-than comparison."]
        pub fn cmplt(self, other: Self) -> $mask {
            crate::peel(self).cmplt(crate::peel(other))
        }
        #[doc = "Minimum by component."]
        pub fn min(self, other: Self) -> Self {
            crate::wrap(crate::peel(self).min(crate::peel(other)))
        }
        #[doc = "Maximum by component."]
        pub fn max(self, other: Self) -> Self {
            crate::wrap(crate::peel(self).max(crate::peel(other)))
        }
        #[doc = "Horizontal minimum (smallest component)."]
        pub fn min_element(self) -> T::Scalar {
            crate::peel(self).min_element()
        }
        #[doc = "Horizontal maximum (largest component)."]
        pub fn max_element(self) -> T::Scalar {
            crate::peel(self).max_element()
        }
        #[doc = "Component-wise clamp."]
        pub fn clamp(self, min: Self, max: Self) -> Self {
            crate::wrap(crate::peel(self).clamp(crate::peel(min), crate::peel(max)))
        }
        #[doc = "Select components from two instances based on a mask."]
        pub fn select(mask: $mask, a: Self, b: Self) -> Self {
            crate::wrap(<<T::Scalar as Scalar>::$t>::select(mask, crate::peel(a), crate::peel(b)))
        }
    }
}

/// is_finite, round, ceil, etc.
macro_rules! forward_float_ops {
    ($mask:ty, $see_also_doc_ty:ty) => {
        // These are the functions that peel the arguments and re-wrap the return values.
        crate::forward_to_raw! {
            $see_also_doc_ty =>
            #[doc = "Round all components up."]
            pub fn ceil(self) -> Self;
            #[doc = "Round all components down."]
            pub fn floor(self) -> Self;
            #[doc = "Round all components."]
            pub fn round(self) -> Self;
            #[doc = "See (e.g.) [`glam::Vec2::fract()`]"]
            pub fn fract(self) -> Self;
            #[doc = "See (e.g.) [`glam::Vec2::fract_gl()`]"]
            pub fn fract_gl(self) -> Self;
        }

        // These functions have custom wrapping.
        #[doc = "True if all components are non-infinity and non-NaN."]
        pub fn is_finite(&self) -> bool {
            crate::peel_ref(self).is_finite()
        }
        #[doc = "True if any component is NaN."]
        pub fn is_nan(&self) -> bool {
            crate::peel_ref(self).is_nan()
        }
        #[doc = "Return a mask where each bit is set if the corresponding component is NaN."]
        pub fn is_nan_mask(&self) -> $mask {
            crate::peel_ref(self).is_nan_mask()
        }
        #[doc = "Linear interpolation."]
        pub fn lerp(self, other: Self, t: T::Scalar) -> Self {
            crate::wrap(crate::peel(self).lerp(crate::peel(other), t))
        }
    };
}

/// normalize, length, etc.
macro_rules! forward_float_vector_ops {
    ($see_also_doc_ty:ty) => {
        // These are the functions that peel the arguments and re-wrap the return values.
        crate::forward_to_raw! {
            $see_also_doc_ty =>
            #[doc = "Normalize the vector. Undefined results in the vector's length is (very close to) zero."]
            pub fn normalize(self) -> Self;
            #[doc = "Normalize the vector, returning zero if the length was already (very close to) zero."]
            pub fn normalize_or_zero(self) -> Self;
            #[doc = "Returns self normalized to length 1.0 if possible, else returns a fallback value."]
            pub fn normalize_or(self, fallback: Self) -> Self;
            #[doc = "e^self by component"]
            pub fn exp(self) -> Self;
            #[doc = "1.0/self by component"]
            pub fn recip(self) -> Self;
            #[doc = "self * a + b"]
            pub fn mul_add(self, a: Self, b: Self) -> Self;
            #[doc = "See (e.g.) [`glam::Vec2::project_onto()`]"]
            pub fn project_onto(self, other: Self) -> Self;
            #[doc = "See (e.g.) [`glam::Vec2::reject_from()`]"]
            pub fn reject_from(self, other: Self) -> Self;
            #[doc = "See (e.g.) [`glam::Vec2::project_onto_normalized()`]"]
            pub fn project_onto_normalized(self, other: Self) -> Self;
            #[doc = "See (e.g.) [`glam::Vec2::reject_from_normalized()`]"]
            pub fn reject_from_normalized(self, other: Self) -> Self;
        }

        // These functions have custom wrapping.
        #[doc = "Normalize the vector, returning `None` if the length was already (very close to) zero."]
        pub fn try_normalize(self) -> Option<Self> {
            crate::peel(self).try_normalize().map(crate::wrap)
        }
        #[doc = "True if the vector is normalized."]
        pub fn is_normalized(self) -> bool {
            crate::peel(self).is_normalized()
        }
        #[doc = "Squared length of the vector"]
        pub fn length_squared(self) -> T::Scalar {
            crate::peel(self).length_squared()
        }
        #[doc = "Reciprocal length of the vector"]
        pub fn length_recip(self) -> T::Scalar {
            crate::peel(self).length_recip()
        }
        #[doc = "Length of the vector"]
        pub fn length(self) -> T::Scalar {
            crate::peel(self).length()
        }
        #[doc = "self^n by component"]
        pub fn powf(self, n: T::Scalar) -> Self {
            crate::wrap(crate::peel(self).powf(n))
        }
        #[doc = "Clamp length"]
        pub fn clamp_length(self, min: T::Scalar, max: T::Scalar) -> Self {
            crate::wrap(crate::peel(self).clamp_length(min, max))
        }
        #[doc = "Clamp length"]
        pub fn clamp_length_min(self, min: T::Scalar) -> Self {
            crate::wrap(crate::peel(self).clamp_length_min(min))
        }
        #[doc = "Clamp length"]
        pub fn clamp_length_max(self, max: T::Scalar) -> Self {
            crate::wrap(crate::peel(self).clamp_length_max(max))
        }
    };
}

macro_rules! array_interface {
    ($dimensions:literal) => {
        #[doc = "Reinterpret as array."]
        #[inline]
        #[must_use]
        pub fn as_array(&self) -> &[T::Scalar; $dimensions] {
            bytemuck::cast_ref(self)
        }

        #[doc = "Reinterpret as mutable array."]
        #[inline]
        #[must_use]
        pub fn as_array_mut(&mut self) -> &mut [T::Scalar; $dimensions] {
            bytemuck::cast_mut(self)
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
        #[inline]
        #[must_use]
        pub fn const_get<const N: usize>(&self) -> T::Scalar {
            self.as_array()[N]
        }

        #[doc = "Set component at index `N`."]
        #[inline]
        pub fn const_set<const N: usize>(&mut self, value: T::Scalar) {
            self.as_array_mut()[N] = value;
        }
    };
}

macro_rules! derive_array_conversion_traits {
    ($base_type_name:ident, $dimensions:literal) => {
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

        impl<T: Unit> AsRef<[T::Scalar; $dimensions]> for $base_type_name<T> {
            #[inline]
            #[must_use]
            fn as_ref(&self) -> &[T::Scalar; $dimensions] {
                bytemuck::cast_ref(self)
            }
        }

        impl<T: Unit> From<[T::Scalar; $dimensions]> for $base_type_name<T> {
            #[inline]
            #[must_use]
            fn from(arr: [T::Scalar; $dimensions]) -> Self {
                Self::from_array(arr)
            }
        }

        impl<T: Unit> From<$base_type_name<T>> for [T::Scalar; $dimensions] {
            #[inline]
            #[must_use]
            fn from(v: $base_type_name<T>) -> Self {
                v.to_array()
            }
        }

        impl<T: Unit> AsMut<[T::Scalar; $dimensions]> for $base_type_name<T> {
            #[inline]
            #[must_use]
            fn as_mut(&mut self) -> &mut [T::Scalar; $dimensions] {
                bytemuck::cast_mut(self)
            }
        }

        impl<T: Unit> AsRef<[T::Scalar]> for $base_type_name<T> {
            #[inline]
            #[must_use]
            fn as_ref(&self) -> &[T::Scalar] {
                AsRef::<[T::Scalar; $dimensions]>::as_ref(self)
            }
        }

        impl<T: Unit> AsMut<[T::Scalar]> for $base_type_name<T> {
            #[inline]
            #[must_use]
            fn as_mut(&mut self) -> &mut [T::Scalar] {
                AsMut::<[T::Scalar; $dimensions]>::as_mut(self)
            }
        }

        impl<T: Unit> PartialEq<[T::Scalar; $dimensions]> for $base_type_name<T> {
            #[inline]
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
    };
}

macro_rules! casting_interface {
    ($base_type_name:ident { $($fields:ident : $fields_ty:ty ),* }) => {
        #[doc = "Bitcast an untyped instance to self."]
        #[inline]
        #[must_use]
        pub fn from_untyped(untyped: $base_type_name<T::Scalar>) -> $base_type_name<T> {
            untyped.cast()
        }

        #[doc = "Bitcast self to an untyped instance."]
        #[inline]
        #[must_use]
        pub fn to_untyped(self) -> $base_type_name<T::Scalar> {
            self.cast()
        }

        #[doc = "Reinterpret cast self as the untyped variant."]
        #[inline]
        #[must_use]
        pub fn as_untyped(&self) -> &$base_type_name<T::Scalar> {
            self.cast_ref()
        }

        #[doc = "Reinterpret cast self as the untyped variant."]
        #[inline]
        #[must_use]
        pub fn as_untyped_mut(&mut self) -> &mut $base_type_name<T::Scalar> {
            self.cast_mut()
        }

        #[doc = "Cast to a different coordinate space with the same underlying scalar type."]
        #[inline]
        #[must_use]
        pub fn cast<T2>(self) -> $base_type_name<T2>
        where
            T2: crate::unit::Unit<Scalar = T::Scalar>,
        {
            bytemuck::cast(self)
        }

        #[doc = "Cast to a different coordinate space with the same underlying scalar type."]
        #[inline]
        #[must_use]
        pub fn cast_ref<T2>(&self) -> &$base_type_name<T2>
        where
            T2: crate::unit::Unit<Scalar = T::Scalar>,
        {
            bytemuck::cast_ref(self)
        }

        #[doc = "Cast to a different coordinate space with the same underlying scalar type."]
        #[inline]
        #[must_use]
        pub fn cast_mut<T2>(&mut self) -> &mut $base_type_name<T2>
        where
            T2: crate::unit::Unit<Scalar = T::Scalar>,
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

        #[doc = "Cast to a different coordinate space with scalar type conversion through the `as` operator (potentially narrowing or losing precision)."]
        #[must_use]
        pub fn as_<T2>(self) -> $base_type_name<T2>
        where
            T: Unit<Scalar: num_traits::AsPrimitive<T2::Scalar>>,
            T2: Unit,
        {
            $base_type_name {
                $($fields: self.$fields.as_()),*
            }
        }
    };
}

macro_rules! derive_standard_traits {
    ($base_type_name:ident {
        $($fields:ident: $fields_ty:ty),*
    }) => {
        impl<T: Unit> Clone for $base_type_name<T> {
            #[inline]
            #[must_use]
            #[cfg_attr(coverage, coverage(off))]
            fn clone(&self) -> Self {
                *self
            }
        }

        impl<T: Unit> Copy for $base_type_name<T> {}

        impl<T: Unit> Default for $base_type_name<T> {
            #[inline]
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

        impl<T: Unit> PartialEq for $base_type_name<T> {
            #[inline]
            #[must_use]
            fn eq(&self, other: &Self) -> bool {
                ($(self.$fields),*) == ($(other.$fields),*)
            }
        }

        impl<T> Eq for $base_type_name<T>
        where
            T: Unit<Scalar: Eq>,
        {}

        impl<T> core::hash::Hash for $base_type_name<T>
        where
            T: Unit<Scalar: core::hash::Hash>,
        {
            fn hash<H>(&self, state: &mut H)
            where
                H: core::hash::Hasher
            {
                $(self.$fields.hash(state);)*
            }
        }

        impl<T: Unit> approx::AbsDiffEq<Self> for $base_type_name<T> {
            type Epsilon = <T::Scalar as approx::AbsDiffEq>::Epsilon;

            #[must_use]
            fn default_epsilon() -> Self::Epsilon {
                T::Scalar::default_epsilon()
            }

            #[must_use]
            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                $(self.$fields.abs_diff_eq(&other.$fields, epsilon.clone()) && )* true
            }

            #[must_use]
            fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                $(self.$fields.abs_diff_ne(&other.$fields, epsilon.clone()) || )* false
            }
        }

        impl<T: Unit<Scalar: approx::RelativeEq>> approx::RelativeEq<Self> for $base_type_name<T>
        {
            #[must_use]
            fn default_max_relative() -> Self::Epsilon {
                T::Scalar::default_max_relative()
            }

            #[must_use]
            fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
                $(self.$fields.relative_eq(&other.$fields, epsilon.clone(), max_relative.clone()) && )* true
            }

            #[must_use]
            fn relative_ne(&self, other: &Self, epsilon: Self::Epsilon, max_relative: Self::Epsilon) -> bool {
                $(self.$fields.relative_ne(&other.$fields, epsilon.clone(), max_relative.clone()) || )* false
            }
        }

        impl<T: Unit<Scalar: approx::UlpsEq>> approx::UlpsEq<Self> for $base_type_name<T>
        {
            #[must_use]
            fn default_max_ulps() -> u32 {
                T::Scalar::default_max_ulps()
            }

            #[must_use]
            fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                $(self.$fields.ulps_eq(&other.$fields, epsilon.clone(), max_ulps) && )* true
            }

            #[must_use]
            fn ulps_ne(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
                $(self.$fields.ulps_ne(&other.$fields, epsilon.clone(), max_ulps) || )* false
            }
        }
    };
}

macro_rules! tuple_interface {
    ($base_type_name:ident { $($fields:ident : $fields_ty:ty ),* }) => {
        #[doc = "Instantiate from tuple."]
        #[allow(unused_parens)]
        #[inline]
        #[must_use]
        pub const fn from_tuple(($($fields),*): ($($fields_ty),*)) -> Self {
            $base_type_name { $($fields),* }
        }

        #[doc = "Convert to tuple."]
        #[inline]
        #[must_use]
        pub const fn to_tuple(self) -> ($($fields_ty),*) {
            ($(self.$fields),*)
        }
    };
}

macro_rules! derive_tuple_conversion_traits {
    ($base_type_name:ident { $($fields:ident : $fields_ty:ty ),* }) => {
        impl<T: Unit> From<($($fields_ty),*)> for $base_type_name<T> {
            #[inline]
            #[allow(unused_parens)]
            #[must_use]
            fn from(tuple: ($($fields_ty),*)) -> $base_type_name<T> {
                $base_type_name::from_tuple(tuple)
            }
        }

        impl<T: Unit> From<$base_type_name<T>> for ($($fields_ty),*) {
            #[inline]
            #[must_use]
            fn from(value: $base_type_name<T>) -> ($($fields_ty),*) {
                value.to_tuple()
            }
        }

        impl<T: Unit> PartialEq<($($fields_ty,)*)> for $base_type_name<T> {
            fn eq(&self, ($($fields,)*): &($($fields_ty,)*)) -> bool {
                $(
                    self.$fields == *$fields
                )&&*
            }
        }
    };
}

macro_rules! derive_glam_conversion_traits {
    ($base_type_name:ident { $x:ident : $x_ty:ty, $y:ident : $y_ty:ty $(,)? }) => {
        crate::derive_glam_conversion_traits!(@impl $base_type_name, glam::Vec2, glam::DVec2, glam::IVec2, glam::UVec2, glam::I64Vec2, glam::U64Vec2);
    };
    ($base_type_name:ident { $x:ident : $x_ty:ty, $y:ident : $y_ty:ty, $z:ident: $z_ty:ty $(,)? }) => {
        crate::derive_glam_conversion_traits!(@impl $base_type_name, glam::Vec3, glam::DVec3, glam::IVec3, glam::UVec3, glam::I64Vec3, glam::U64Vec3);
    };
    ($base_type_name:ident { $x:ident : $x_ty:ty, $y:ident : $y_ty:ty, $z:ident: $z_ty:ty, $w:ident: $w_ty:ty $(,)? }) => {
        crate::derive_glam_conversion_traits!(@impl $base_type_name, glam::Vec4, glam::DVec4, glam::IVec4, glam::UVec4, glam::I64Vec4, glam::U64Vec4);
    };

    (@impl $base_type_name:ident, $glam_f32_ty:ty, $glam_f64_ty:ty, $glam_i32_ty:ty, $glam_u32_ty:ty, $glam_i64_ty:ty, $glam_u64_ty:ty) => {
        crate::derive_glam_conversion_traits!(@impl2 $base_type_name, $glam_f32_ty, f32);
        crate::derive_glam_conversion_traits!(@impl2 $base_type_name, $glam_f64_ty, f64);
        crate::derive_glam_conversion_traits!(@impl2 $base_type_name, $glam_i32_ty, i32);
        crate::derive_glam_conversion_traits!(@impl2 $base_type_name, $glam_u32_ty, u32);
        crate::derive_glam_conversion_traits!(@impl2 $base_type_name, $glam_i64_ty, i64);
        crate::derive_glam_conversion_traits!(@impl2 $base_type_name, $glam_u64_ty, u64);
    };
    (@impl2 $base_type_name:ident, $glam_ty:ty, $scalar_ty:ty) => {
        impl<T> From<$base_type_name<T>> for $glam_ty
        where
            T: Unit<Scalar = $scalar_ty>,
        {
            fn from(vec: $base_type_name<T>) -> $glam_ty {
                crate::peel(vec)
            }
        }

        impl<T> From<$glam_ty> for $base_type_name<T>
        where
            T: Unit<Scalar = $scalar_ty>
        {
            fn from(vec: $glam_ty) -> $base_type_name<T> {
                crate::wrap(vec)
            }
        }

        impl<T> AsRef<$glam_ty> for $base_type_name<T>
        where
            T: Unit<Scalar = $scalar_ty>
        {
            fn as_ref(&self) -> &$glam_ty {
                crate::peel_ref(self)
            }
        }

        impl<T> AsMut<$glam_ty> for $base_type_name<T>
        where
            T: Unit<Scalar = $scalar_ty>
        {
            fn as_mut(&mut self) -> &mut $glam_ty {
                crate::peel_mut(self)
            }
        }

        impl<T> core::borrow::Borrow<$glam_ty> for $base_type_name<T>
        where
            T: Unit<Scalar = $scalar_ty>
        {
            fn borrow(&self) -> &$glam_ty {
                self.as_ref()
            }
        }
        impl<T> core::borrow::BorrowMut<$glam_ty> for $base_type_name<T>
        where
            T: Unit<Scalar = $scalar_ty>
        {
            fn borrow_mut(&mut self) -> &mut $glam_ty {
                self.as_mut()
            }
        }

        impl<T> core::borrow::Borrow<$base_type_name<T>> for $glam_ty
        where
            T: Unit<Scalar = $scalar_ty>
        {
            fn borrow(&self) -> &$base_type_name<T> {
                bytemuck::cast_ref(self)
            }
        }
        impl<T> core::borrow::BorrowMut<$base_type_name<T>> for $glam_ty
        where
            T: Unit<Scalar = $scalar_ty>
        {
            fn borrow_mut(&mut self) -> &mut $base_type_name<T> {
                bytemuck::cast_mut(self)
            }
        }
    };
}

pub(crate) use forward_comparison;
pub(crate) use forward_constructors;
pub(crate) use forward_float_ops;
pub(crate) use forward_float_vector_ops;
pub(crate) use forward_neg_to_raw;
pub(crate) use forward_op_assign_to_raw;
pub(crate) use forward_op_to_raw;
pub(crate) use forward_to_raw;
pub(crate) use forward_to_raw_impl;

pub(crate) use derive_array_conversion_traits;
pub(crate) use derive_glam_conversion_traits;
pub(crate) use derive_standard_traits;
pub(crate) use derive_tuple_conversion_traits;

pub(crate) use array_interface;
pub(crate) use casting_interface;
pub(crate) use tuple_interface;

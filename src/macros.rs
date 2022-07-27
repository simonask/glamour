macro_rules! forward_all_to_raw {
    (
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
            crate::forward_to_raw!(
                $(#[$($attr),*])*
                $visibility fn $fn_name($($args)*) $(-> $ret_ty)*;
            );
        )*
    };
}

macro_rules! forward_to_raw {
    (
        $(#[$($attr:meta),*])*
        $visibility:vis
        fn $fn_name:ident (
            self
            $(, $arg_name:ident: $arg_ty:ty)*
        ) -> $ret_ty:ty;
    ) => {
        #[inline]
        #[must_use]
        $(#[$($attr),*])*
        $visibility
        fn $fn_name(self $(, $arg_name: $arg_ty)*) -> $ret_ty {
            <$ret_ty>::from_raw(self.to_raw().$fn_name($($arg_name.to_raw()),*))
        }
    };

    (
        $(#[$($attr:meta),*])*
        $visibility:vis
        fn $fn_name:ident (
            &self
            $(, $arg_name:ident: $arg_ty:ty)*
        ) -> $ret_ty:ty;
    ) => {
        #[inline]
        #[must_use]
        $(#[$($attr),*])*
        $visibility
        fn $fn_name(&self, $(arg_name: $arg_ty),*) -> $ret_ty {
            <$ret_ty>::from_raw(self.as_raw().$fn_name($($arg_name.to_raw()),*))
        }
    };

    (
        $(#[$($attr:meta),*])*
        $visibility:vis
        fn $fn_name:ident (
            &mut self
            $(, $arg_name:ident: $arg_ty:ty)*
        ) -> $ret_ty:ty;
    ) => {
        #[inline]
        #[must_use]
        $(#[$($attr),*])*
        $visibility
        fn $fn_name(&mut self, $(arg_name: $arg_ty),*) -> $ret_ty {
            self.as_raw_mut().$fn_name($($arg_name.into()),*).into()
        }
    };

    (
        $(#[$($attr:meta),*])*
        $visibility:vis
        fn $fn_name:ident (
            &mut self
            $(, $arg_name:ident: $arg_ty:ty)*
        );
    ) => {
        #[inline]
        $(#[$($attr),*])*
        $visibility
        fn $fn_name(&mut self, $(arg_name: $arg_ty),*) {
            self.as_raw_mut().$fn_name($($arg_name.into()),*)
        }
    };

    (
        $(#[$($attr:meta),*])*
        $visibility:vis
        fn $fn_name:ident(
            $($arg_name:ident: $arg_ty:ty),*
        ) -> $ret_ty:ty;
    ) => {
        #[inline]
        #[must_use]
        $(#[$($attr),*])*
        $visibility
        fn $fn_name(
            $($arg_name: $arg_ty),*
        ) -> $ret_ty {
            <$ret_ty>::from_raw(<Self as crate::ToRaw>::Raw::$fn_name($($arg_name.to_raw()),*))
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
                    <$output>::from_raw(core::ops::$op_name::$op_fn_name(
                        self.to_raw(),
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
                <$output>::from_raw(core::ops::$op_name::$op_fn_name(
                    self.to_raw(),
                    other.to_raw(),
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
                    core::ops::$op_name::$op_fn_name(self.as_raw_mut(), other)
                }
            }
        )*
    };

    ($base_type_name:ident, $op_name:ident < $arg_ty:ty > :: $op_fn_name:ident) => {
        impl<T: Unit> core::ops::$op_name<$arg_ty> for $base_type_name<T> {
            #[inline]
            fn $op_fn_name(&mut self, other: $arg_ty) {
                core::ops::$op_name::$op_fn_name(self.as_raw_mut(), other.to_raw())
            }
        }
    };
}

/// Implement common operations for all vector-like types.
macro_rules! impl_common {
    ($base_type_name:ident {
        $($fields:ident: $fields_ty:ty),*
    }) => {
        impl<T: crate::Unit> $base_type_name<T> {
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
        }

        impl<T: Unit> Clone for $base_type_name<T> {
            #[inline]
            #[must_use]
            #[cfg_attr(coverage, no_coverage)]
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

        /// SAFETY: Vector types only contain members of type `T::Scalar`, which is required to be [`Zeroable`](bytemuck::Zeroable).
        unsafe impl<T: crate::Unit> bytemuck::Zeroable for $base_type_name<T> where $($fields_ty: bytemuck::Zeroable),* {}
        /// SAFETY: Vector types only contain members of type `T::Scalar`, which is required to be [`Pod`](bytemuck::Pod).
        unsafe impl<T: crate::Unit> bytemuck::Pod for $base_type_name<T> where $($fields_ty: bytemuck::Pod),* {}

        impl<T: Unit> PartialEq for $base_type_name<T> {
            #[inline]
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
            <T::Scalar as approx::AbsDiffEq>::Epsilon: Clone
        {
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

        impl<T: Unit> approx::RelativeEq<Self> for $base_type_name<T>
        where
            T::Scalar: approx::RelativeEq,
            <T::Scalar as approx::AbsDiffEq>::Epsilon: Clone
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

        impl<T: Unit> approx::UlpsEq<Self> for $base_type_name<T>
        where
            T::Scalar: approx::UlpsEq,
            <T::Scalar as approx::AbsDiffEq>::Epsilon: Clone
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

        impl<T: Unit> $base_type_name<T> {
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
        }

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

        $crate::macros::impl_tuple_eq!($base_type_name {
            $($fields: $fields_ty,)*
        });
    };
}

macro_rules! impl_tuple_eq {
    ($base_type_name:ident {
        $x:ident: $x_ty:ty,
        $y:ident: $y_ty:ty
        $(,)?
    }) => {
        impl<T: Unit, A, B> PartialEq<(A, B)> for $base_type_name<T>
        where
            $x_ty: PartialEq<A>,
            $y_ty: PartialEq<B>,
        {
            fn eq(&self, other: &(A, B)) -> bool {
                self.$x == other.0 && self.$y == other.1
            }
        }
    };
    ($base_type_name:ident {
        $x:ident: $x_ty:ty,
        $y:ident: $y_ty:ty,
        $z:ident: $z_ty:ty
        $(,)?
    }) => {
        impl<T: Unit, A, B, C> PartialEq<(A, B, C)> for $base_type_name<T>
        where
            $x_ty: PartialEq<A>,
            $y_ty: PartialEq<B>,
            $z_ty: PartialEq<C>,
        {
            fn eq(&self, other: &(A, B, C)) -> bool {
                self.$x == other.0 && self.$y == other.1 && self.$z == other.2
            }
        }
    };
    ($base_type_name:ident {
        $x:ident: $x_ty:ty,
        $y:ident: $y_ty:ty,
        $z:ident: $z_ty:ty,
        $w:ident: $w_ty:ty
        $(,)?
    }) => {
        impl<T: Unit, A, B, C, D> PartialEq<(A, B, C, D)> for $base_type_name<T>
        where
            $x_ty: PartialEq<A>,
            $y_ty: PartialEq<B>,
            $z_ty: PartialEq<C>,
            $w_ty: PartialEq<D>,
        {
            fn eq(&self, other: &(A, B, C, D)) -> bool {
                self.$x == other.0 && self.$y == other.1 && self.$z == other.2 && self.$w == other.3
            }
        }
    };
}

macro_rules! impl_vector_common {
    ($base_type_name:ident [$dimensions:literal] => $vec_ty:ident {
        $($fields:ident),*
    }) => {
        impl<T> $base_type_name<T>
        where
            T: crate::Unit,
        {
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

        impl<T> core::ops::Neg for $base_type_name<T>
        where
            T: Unit,
            <T::Scalar as Scalar>::$vec_ty: core::ops::Neg<Output = <T::Scalar as Scalar>::$vec_ty>,
        {
            type Output = Self;

            #[inline]
            #[must_use]
            fn neg(self) -> Self::Output {
                Self::from_raw(-self.to_raw())
            }
        }
    };
}

macro_rules! impl_glam_conversion {
    ($base_type_name:ident [
        $(
            $scalar:ty => $glam_type:ty
        ),*
    ]) => {
        $(
            impl<T> From<$base_type_name<T>> for $glam_type
            where
                T: Unit<Scalar = $scalar>,
            {
                #[inline]
                #[must_use]
                fn from(vec: $base_type_name<T>) -> Self {
                    vec.to_raw()
                }
            }

            impl<T> From<$glam_type> for $base_type_name<T>
            where
                T: Unit<Scalar = $scalar>,
            {
                #[inline]
                #[must_use]
                fn from(vec: $glam_type) -> Self {
                    Self::from_raw(vec)
                }
            }

            impl<T> AsRef<$glam_type> for $base_type_name<T>
            where
                T: Unit<Scalar = $scalar>,
            {
                #[inline]
                #[must_use]
                fn as_ref(&self) -> &$glam_type {
                    bytemuck::cast_ref(self)
                }
            }

            impl<T> AsMut<$glam_type> for $base_type_name<T>
            where
                T: Unit<Scalar = $scalar>,
            {
                #[inline]
                #[must_use]
                fn as_mut(&mut self) -> &mut $glam_type {
                    bytemuck::cast_mut(self)
                }
            }
        )*
    };
}

pub(crate) use forward_all_to_raw;
pub(crate) use forward_op_assign_to_raw;
pub(crate) use forward_op_to_raw;
pub(crate) use forward_to_raw;
pub(crate) use impl_common;
pub(crate) use impl_glam_conversion;
pub(crate) use impl_tuple_eq;
pub(crate) use impl_vector_common;

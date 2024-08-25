/// This macro implements *everything* for a simdlike type.
///
/// - Bindings to methods in `glam` (everything in `crate::interfaces`).
/// - Swizzle methods
/// - Additional public APIs that do not map directly to operations in `glam`, like trait imlps.
///
/// This also selects additional methods for each simdlike type, such as "pointlike"-specific types etc.
macro_rules! simdlike {
    ($base_type_name:ident, $n:tt) => {
        impl<T: crate::Unit> $base_type_name<T> {
            crate::interfaces::simd_base_interface!(struct);
            crate::impl_vectorlike::arraylike_interface!($n);
            crate::impl_vectorlike::casting_interface!($base_type_name, $n);
            crate::impl_vectorlike::tuple_interface!($base_type_name, $n);
        }
        impl<T: crate::FloatUnit> $base_type_name<T> {
            crate::interfaces::simd_float_base_interface!(struct);
        }
        impl<T: crate::SignedUnit> $base_type_name<T> {
            crate::interfaces::simd_signed_interface!(struct);
        }
        impl<T: crate::IntUnit> $base_type_name<T> {
            crate::interfaces::vector_integer_interface!(struct);
        }
        crate::impl_vectorlike::impl_swizzle_interface!($base_type_name, $n);
        crate::impl_vectorlike::simdlike!(@for_size $base_type_name, $n);
        crate::impl_traits::impl_basic_traits_vectorlike!($base_type_name, $n);
    };
    (@for_size $base_type_name:ident, 2) => {
        impl<T: crate::Unit> $base_type_name<T> {
            /// All zeroes.
            pub const ZERO: Self = Self::new(T::Scalar::ZERO, T::Scalar::ZERO);
            /// All ones.
            pub const ONE: Self = Self::new(T::Scalar::ONE, T::Scalar::ONE);

            /// Unit vector in the direction of the X axis.
            pub const X: Self = Self::new(
                T::Scalar::ONE,
                T::Scalar::ZERO,
            );

            /// Unit vector in the direction of the Y axis.
            pub const Y: Self = Self::new(
                T::Scalar::ZERO,
                T::Scalar::ONE,
            );

            /// The unit axes.
            pub const AXES: [Self; 2] = [Self::X, Self::Y];
        }
        impl<T: crate::FloatUnit> $base_type_name<T> {
            /// All NaN.
            pub const NAN: Self = Self::new(
                <T::Scalar as FloatScalar>::NAN,
                <T::Scalar as FloatScalar>::NAN,
            );
            /// All positive infinity.
            pub const INFINITY: Self = Self::new(
                <T::Scalar as FloatScalar>::INFINITY,
                <T::Scalar as FloatScalar>::INFINITY,
            );
            /// All negative infinity.
            pub const NEG_INFINITY: Self = Self::new(
                <T::Scalar as FloatScalar>::NEG_INFINITY,
                <T::Scalar as FloatScalar>::NEG_INFINITY,
            );
            crate::interfaces::simd2_float_interface!(struct);
        }
        impl<T: crate::SignedUnit> $base_type_name<T> {
            /// All negative one.
            pub const NEG_ONE: Self = Self::new(
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
            );

            /// (-1, 0)
            pub const NEG_X: Self = Self::new(
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
                T::Scalar::ZERO,
            );

            /// (0, -1)
            pub const NEG_Y: Self = Self::new(
                T::Scalar::ZERO,
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
            );
        }
    };
    (@for_size $base_type_name:ident, 3) => {
        impl<T: crate::Unit> $base_type_name<T> {
            /// All zeroes.
            pub const ZERO: Self = Self::new(T::Scalar::ZERO, T::Scalar::ZERO, T::Scalar::ZERO);
            /// All ones.
            pub const ONE: Self = Self::new(T::Scalar::ONE, T::Scalar::ONE, T::Scalar::ONE);

            /// Unit vector in the direction of the X axis.
            pub const X: Self = Self::new(
                T::Scalar::ONE,
                T::Scalar::ZERO,
                T::Scalar::ZERO,
            );

            /// Unit vector in the direction of the Y axis.
            pub const Y: Self = Self::new(
                T::Scalar::ZERO,
                T::Scalar::ONE,
                T::Scalar::ZERO,
            );

            /// Unit vector in the direction of the Z axis.
            pub const Z: Self = Self::new(
                T::Scalar::ZERO,
                T::Scalar::ZERO,
                T::Scalar::ONE,
            );

            /// The unit axes.
            pub const AXES: [Self; 3] = [Self::X, Self::Y, Self::Z];
        }
        impl<T: crate::SignedUnit> $base_type_name<T> {
            /// All negative one.
            pub const NEG_ONE: Self = Self::new(
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
            );

            /// (-1, 0, 0)
            pub const NEG_X: Self = Self::new(
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
                T::Scalar::ZERO,
                T::Scalar::ZERO,
            );

            /// (0, -1, 0)
            pub const NEG_Y: Self = Self::new(
                T::Scalar::ZERO,
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
                T::Scalar::ZERO,
            );

            /// (0, 0, -1)
            pub const NEG_Z: Self = Self::new(
                T::Scalar::ZERO,
                T::Scalar::ZERO,
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
            );
        }
        impl<T: crate::FloatUnit> $base_type_name<T> {
            /// All NaN.
            pub const NAN: Self = Self::new(
                <T::Scalar as FloatScalar>::NAN,
                <T::Scalar as FloatScalar>::NAN,
                <T::Scalar as FloatScalar>::NAN,
            );
            /// All positive infinity.
            pub const INFINITY: Self = Self::new(
                <T::Scalar as FloatScalar>::INFINITY,
                <T::Scalar as FloatScalar>::INFINITY,
                <T::Scalar as FloatScalar>::INFINITY,
            );
            /// All negative infinity.
            pub const NEG_INFINITY: Self = Self::new(
                <T::Scalar as FloatScalar>::NEG_INFINITY,
                <T::Scalar as FloatScalar>::NEG_INFINITY,
                <T::Scalar as FloatScalar>::NEG_INFINITY,
            );
            crate::interfaces::simd3_float_interface!(struct);
        }
    };
    (@for_size $base_type_name:ident, 4) => {
        impl<T: crate::Unit> $base_type_name<T> {
            /// All zeroes.
            pub const ZERO: Self = Self::new(T::Scalar::ZERO, T::Scalar::ZERO, T::Scalar::ZERO, T::Scalar::ZERO);
            /// All ones.
            pub const ONE: Self = Self::new(T::Scalar::ONE, T::Scalar::ONE, T::Scalar::ONE, T::Scalar::ZERO);

            /// Unit vector in the direction of the X axis.
            pub const X: Self = Self::new(
                T::Scalar::ONE,
                T::Scalar::ZERO,
                T::Scalar::ZERO,
                T::Scalar::ZERO,
            );

            /// Unit vector in the direction of the Y axis.
            pub const Y: Self = Self::new(
                T::Scalar::ZERO,
                T::Scalar::ONE,
                T::Scalar::ZERO,
                T::Scalar::ZERO,
            );

            /// Unit vector in the direction of the Z axis.
            pub const Z: Self = Self::new(
                T::Scalar::ZERO,
                T::Scalar::ZERO,
                T::Scalar::ONE,
                T::Scalar::ZERO,
            );

            /// Unit vector in the direction of the W axis.
            pub const W: Self = Self::new(
                T::Scalar::ZERO,
                T::Scalar::ZERO,
                T::Scalar::ZERO,
                T::Scalar::ONE,
            );

            /// The unit axes.
            pub const AXES: [Self; 4] = [Self::X, Self::Y, Self::Z, Self::W];
        }
        impl<T: crate::SignedUnit> $base_type_name<T> {
            /// All negative one.
            pub const NEG_ONE: Self = Self::new(
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
            );

            /// (-1, 0, 0, 0)
            pub const NEG_X: Self = Self::new(
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
                T::Scalar::ZERO,
                T::Scalar::ZERO,
                T::Scalar::ZERO,
            );

            /// (0, -1, 0, 0)
            pub const NEG_Y: Self = Self::new(
                T::Scalar::ZERO,
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
                T::Scalar::ZERO,
                T::Scalar::ZERO,
            );

            /// (0, 0, -1, 0)
            pub const NEG_Z: Self = Self::new(
                T::Scalar::ZERO,
                T::Scalar::ZERO,
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
                T::Scalar::ZERO,
            );

            /// (0, 0, 0, -1)
            pub const NEG_W: Self = Self::new(
                T::Scalar::ZERO,
                T::Scalar::ZERO,
                T::Scalar::ZERO,
                <T::Scalar as crate::SignedScalar>::NEG_ONE,
            );
        }
        impl<T: crate::FloatUnit> $base_type_name<T> {
            /// All NaN.
            pub const NAN: Self = Self::new(
                <T::Scalar as FloatScalar>::NAN,
                <T::Scalar as FloatScalar>::NAN,
                <T::Scalar as FloatScalar>::NAN,
                <T::Scalar as FloatScalar>::NAN,
            );
            /// All positive infinity.
            pub const INFINITY: Self = Self::new(
                <T::Scalar as FloatScalar>::INFINITY,
                <T::Scalar as FloatScalar>::INFINITY,
                <T::Scalar as FloatScalar>::INFINITY,
                <T::Scalar as FloatScalar>::INFINITY,
            );
            /// All negative infinity.
            pub const NEG_INFINITY: Self = Self::new(
                <T::Scalar as FloatScalar>::NEG_INFINITY,
                <T::Scalar as FloatScalar>::NEG_INFINITY,
                <T::Scalar as FloatScalar>::NEG_INFINITY,
                <T::Scalar as FloatScalar>::NEG_INFINITY,
            );
            crate::interfaces::simd4_float_interface!(struct);
        }
    };
}
pub(crate) use simdlike;

macro_rules! vectorlike {
    ($base_type_name:ident, $n:tt) => {
        crate::impl_vectorlike::simdlike!($base_type_name, $n);
        crate::impl_vectorlike::vectorlike!(@for_size $base_type_name, $n);

        impl<T: crate::Unit> $base_type_name<T> {
            crate::interfaces::vector_base_interface!(struct);
        }

        impl<T: crate::FloatUnit> $base_type_name<T> {
            crate::interfaces::vector_float_interface!(struct);
        }
    };
    (@for_size $base_type_name:ident, 2) => {
        impl<T: crate::Unit> $base_type_name<T> {
            crate::interfaces::simd2_base_interface!(struct, vec3);
        }
        impl<T: crate::SignedUnit> $base_type_name<T> {
            crate::interfaces::vector2_signed_interface!(struct);
        }
        impl<T: crate::FloatUnit> $base_type_name<T> {
            crate::interfaces::vector2_float_interface!(struct);
        }
    };
    (@for_size $base_type_name:ident, 3) => {
        impl<T: crate::Unit> $base_type_name<T> {
            crate::interfaces::simd3_base_interface!(struct, vec2, vec4);
        }
        impl<T: crate::FloatUnit> $base_type_name<T> {
            crate::interfaces::vector3_float_interface!(struct);
        }
    };
    (@for_size $base_type_name:ident, 4) => {
        impl<T: crate::Unit> $base_type_name<T> {
            crate::interfaces::simd4_base_interface!(struct, vec3);
        }
    };
}
pub(crate) use vectorlike;
macro_rules! pointlike {
    ($base_type_name:ident, $n:tt) => {
        crate::impl_vectorlike::simdlike!($base_type_name, $n);
        crate::impl_vectorlike::pointlike!(@for_size $base_type_name, $n);
        impl<T: crate::FloatUnit> $base_type_name<T> {
            crate::interfaces::point_float_interface!(struct);
        }
    };
    (@for_size $base_type_name:ident, 2) => {
        impl<T: crate::Unit> $base_type_name<T> {
            crate::interfaces::simd2_base_interface!(struct, point3);
        }
    };
    (@for_size $base_type_name:ident, 3) => {
        impl<T: crate::Unit> $base_type_name<T> {
            crate::interfaces::simd3_base_interface!(struct, point2, point4);
        }
        impl<T: crate::FloatUnit> $base_type_name<T> {
            crate::interfaces::vector3_float_interface!(struct);
        }
    };
    (@for_size $base_type_name:ident, 4) => {
        impl<T: crate::Unit> $base_type_name<T> {
            crate::interfaces::simd4_base_interface!(struct, point3);
        }
    };
}
pub(crate) use pointlike;
macro_rules! sizelike {
    ($base_type_name:ident, $n:tt) => {
        crate::impl_vectorlike::simdlike!($base_type_name, $n);
        crate::impl_vectorlike::sizelike!(@for_size $base_type_name, $n);
    };
    (@for_size $base_type_name:ident, 2) => {
        impl<T: crate::Unit> $base_type_name<T> {
            crate::interfaces::simd2_base_interface!(struct, size3);
        }
    };
    (@for_size $base_type_name:ident, 3) => {
        impl<T: crate::Unit> $base_type_name<T> {
            crate::interfaces::simd3_base_interface!(struct, size2, vec4);
        }
        impl<T: crate::FloatUnit> $base_type_name<T> {
            crate::interfaces::vector3_float_interface!(struct);
        }
    };
}
pub(crate) use sizelike;

macro_rules! tuple_interface {
    (Size2, 2) => {
        /// Convert this `Size2` to a tuple.
        #[inline]
        #[must_use]
        pub fn to_tuple(self) -> (T::Scalar, T::Scalar) {
            (self.width, self.height)
        }
        /// Create from tuple.
        #[inline]
        #[must_use]
        pub fn from_tuple((width, height): (T::Scalar, T::Scalar)) -> Self {
            Self { width, height }
        }
    };
    (Size3, 3) => {
        /// Convert this `Size3` to a tuple.
        #[inline]
        #[must_use]
        pub fn to_tuple(self) -> (T::Scalar, T::Scalar, T::Scalar) {
            (self.width, self.height, self.depth)
        }
        /// Create from tuple.
        #[inline]
        #[must_use]
        pub fn from_tuple((width, height, depth): (T::Scalar, T::Scalar, T::Scalar)) -> Self {
            Self {
                width,
                height,
                depth,
            }
        }
    };
    ($base_type_name:ident, 2) => {
        /// Convert this vector or point to a tuple.
        #[inline]
        #[must_use]
        pub fn to_tuple(self) -> (T::Scalar, T::Scalar) {
            (self.x, self.y)
        }
        /// Create from tuple.
        #[inline]
        #[must_use]
        pub fn from_tuple((x, y): (T::Scalar, T::Scalar)) -> Self {
            Self { x, y }
        }
    };
    ($base_type_name:ident, 3) => {
        /// Convert this vector or point to a tuple.
        #[inline]
        #[must_use]
        pub fn to_tuple(self) -> (T::Scalar, T::Scalar, T::Scalar) {
            (self.x, self.y, self.z)
        }
        /// Create from tuple.
        #[inline]
        #[must_use]
        pub fn from_tuple((x, y, z): (T::Scalar, T::Scalar, T::Scalar)) -> Self {
            Self { x, y, z }
        }
    };
    ($base_type_name:ident, 4) => {
        /// Convert this vector or point to a tuple.
        #[inline]
        #[must_use]
        pub fn to_tuple(self) -> (T::Scalar, T::Scalar, T::Scalar, T::Scalar) {
            (self.x, self.y, self.z, self.w)
        }
        /// Create from tuple.
        #[inline]
        #[must_use]
        pub fn from_tuple((x, y, z, w): (T::Scalar, T::Scalar, T::Scalar, T::Scalar)) -> Self {
            Self { x, y, z, w }
        }
    };
}
pub(crate) use tuple_interface;

macro_rules! impl_swizzle_interface {
    ($base_type_name:ident, $n:tt) => {
        impl<T: Unit> crate::Swizzle<T> for $base_type_name<T> {
            #[inline]
            fn swizzle2<const X: usize, const Y: usize>(&self) -> Vector2<T> {
                Vector2::new(self.const_get::<X>(), self.const_get::<Y>())
            }

            #[inline]
            fn swizzle3<const X: usize, const Y: usize, const Z: usize>(&self) -> Vector3<T> {
                Vector3::new(
                    self.const_get::<X>(),
                    self.const_get::<Y>(),
                    self.const_get::<Z>(),
                )
            }

            #[inline]
            fn swizzle4<const X: usize, const Y: usize, const Z: usize, const W: usize>(
                &self,
            ) -> crate::Vector4<T> {
                crate::Vector4::new(
                    self.const_get::<X>(),
                    self.const_get::<Y>(),
                    self.const_get::<Z>(),
                    self.const_get::<W>(),
                )
            }

        }
        impl<T: Unit> $base_type_name<T> {
            crate::impl_vectorlike::impl_swizzle_interface!(@default $base_type_name, $n);
        }
    };
    (@default $base_type_name:ident, 2) => {
        /// Select components of this vector and return a new vector containing
        /// those components.
        #[inline]
        #[must_use]
        pub fn swizzle<const X: usize, const Y: usize>(&self) -> crate::Vector2<T> {
            use crate::Swizzle;
            self.swizzle2::<X, Y>()
        }
    };
    (@default $base_type_name:ident, 3) => {
        /// Select components of this vector and return a new vector containing
        /// those components.
        #[inline]
        #[must_use]
        pub fn swizzle<const X: usize, const Y: usize, const Z: usize>(&self) -> crate::Vector3<T> {
            use crate::Swizzle;
            self.swizzle3::<X, Y, Z>()
        }
    };
    (@default $base_type_name:ident, 4) => {
        /// Select components of this vector and return a new vector containing
        /// those components.
        #[inline]
        #[must_use]
        pub fn swizzle<const X: usize, const Y: usize, const Z: usize, const W: usize>(&self) -> crate::Vector4<T> {
            use crate::Swizzle;
            self.swizzle4::<X, Y, Z, W>()
        }
    };
}
pub(crate) use impl_swizzle_interface;

macro_rules! casting_interface {
    ($base_type_name:ident, $n:tt) => {
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

        #[doc = "Cast to a different coordinate space with scalar type conversion through the `as` operator (potentially narrowing or losing precision)."]
        #[must_use]
        pub fn as_<T2>(self) -> $base_type_name<T2>
        where
            T: Unit<Scalar: num_traits::AsPrimitive<T2::Scalar>>,
            T2: Unit,
        {
            $base_type_name::from_array(self.to_array().map(crate::Scalar::as_))
        }

        crate::impl_vectorlike::casting_interface!(@try_cast $base_type_name, $n);
    };

    (@try_cast $base_type_name:ident, 4) => {
        /// Cast to a different coordinate space with scalar type conversion. Returns `None` if any component could not be converted to the target scalar type.
        #[must_use]
        #[allow(clippy::question_mark)]
        pub fn try_cast<T2>(self) -> Option<$base_type_name<T2>>
        where
            T2: Unit,
        {
            let Some(x) = self.const_get::<0>().try_cast() else {
                return None;
            };
            let Some(y) = self.const_get::<1>().try_cast() else {
                return None;
            };
            let Some(z) = self.const_get::<2>().try_cast() else {
                return None;
            };
            let Some(w) = self.const_get::<3>().try_cast() else {
                return None;
            };
            Some($base_type_name::new(x, y, z, w))
        }
    };
    (@try_cast $base_type_name:ident, 3) => {
        /// Cast to a different coordinate space with scalar type conversion. Returns `None` if any component could not be converted to the target scalar type.
        #[must_use]
        #[allow(clippy::question_mark)]
        pub fn try_cast<T2>(self) -> Option<$base_type_name<T2>>
        where
            T2: Unit,
        {
            let Some(x) = self.const_get::<0>().try_cast() else {
                return None;
            };
            let Some(y) = self.const_get::<1>().try_cast() else {
                return None;
            };
            let Some(z) = self.const_get::<2>().try_cast() else {
                return None;
            };
            Some($base_type_name::new(x, y, z))
        }
    };
    (@try_cast $base_type_name:ident, 2) => {
        /// Cast to a different coordinate space with scalar type conversion. Returns `None` if any component could not be converted to the target scalar type.
        #[must_use]
        #[allow(clippy::question_mark)]
        pub fn try_cast<T2>(self) -> Option<$base_type_name<T2>>
        where
            T2: Unit,
        {
            let Some(x) = self.const_get::<0>().try_cast() else {
                return None;
            };
            let Some(y) = self.const_get::<1>().try_cast() else {
                return None;
            };
            Some($base_type_name::new(x, y))
        }
    };
}
pub(crate) use casting_interface;

macro_rules! arraylike_interface {
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

        /// Returns a vector containing each element of `self` modified by a mapping function `f`.
        pub fn map<F: Fn(T::Scalar) -> T::Scalar>(self, f: F) -> Self {
            Self::from_array(self.to_array().map(f))
        }
    };
}
pub(crate) use arraylike_interface;

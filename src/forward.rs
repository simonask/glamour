//! Macros for generating fn signatures and implementations for mapping public API types to glam.

macro_rules! forward_fn {
    (
        $mode:tt =>
        $(#[$($attr:meta),*])*
        fn $fn_name:ident (&self $($args:tt)*)
        $($rest:tt)*
    ) => {
        crate::forward_fn_self_ref!($mode =>
            $(#[$($attr),*])*
            fn $fn_name (&self $($args)*)
            $($rest)*
        );
    };

    (
        $mode:tt =>
        $(#[$($attr:meta),*])*
        fn $fn_name:ident (self $($args:tt)*)
        $($rest:tt)*
    ) => {
        crate::forward_fn_self!($mode =>
            $(#[$($attr),*])*
            fn $fn_name (self $($args)*)
            $($rest)*
        );
    };

    (
        $mode:tt =>
        $(#[$($attr:meta),*])*
        fn $fn_name:ident (
            $($arg_name:ident: $arg_ty:tt),*
        )
        $($rest:tt)*
    ) => {
        crate::forward_fn_assoc!($mode =>
            $(#[$($attr),*])*
            fn $fn_name ($($arg_name: $arg_ty),*)
            $($rest)*
        );
    };
}

macro_rules! forward_fn_self {
    (
        trait_decl =>
        $(#[
            $($attr:meta),*
        ])*
        fn $fn_name:ident (
            self $(, $arg_name:ident: $arg_ty:tt)*
        )
        $(-> $($ret:tt)*)?
    ) => {
        #[allow(missing_docs, clippy::return_self_not_must_use)]
        fn $fn_name(
            self $(, $arg_name: crate::forward_ty!(trait $arg_ty))*
        ) $(-> crate::forward_ty!(trait $($ret)*))*;
    };
    (
        trait_impl =>
        $(#[
            $($attr:meta),*
        ])*
        fn $fn_name:ident (
            self $(, $arg_name:ident: $arg_ty:tt)*
        )
        $(-> $($ret:tt)*)?
    ) => {
        #[inline(always)]
        fn $fn_name(
            self
            $(, $arg_name: crate::forward_ty!(trait $arg_ty))*
        )
            $(-> crate::forward_ty!(trait $($ret)*))*
        {
            <Self>::$fn_name(self $(, $arg_name)*)
        }
    };
    (
        struct =>
        $(#[
            $($attr:meta),*
        ])*
        fn $fn_name:ident (
            self $(, $arg_name:ident: $arg_ty:tt)*
        )
    ) => {
        $(#[$($attr),*])*
        pub fn $fn_name(
            self $(, $arg_name: crate::forward_ty!(struct $arg_ty))*
        )
        {
            crate::peel(self).$fn_name(
                $(crate::forward_arg!($arg_name: $arg_ty)),*
            )
        }
    };
    (
        struct =>
        $(#[
            $($attr:meta),*
        ])*
        fn $fn_name:ident (
            self $(, $arg_name:ident: $arg_ty:tt)*
        )
        -> $($ret:tt)*
    ) => {
        $(#[$($attr),*])*
        #[must_use]
        pub fn $fn_name(
            self $(, $arg_name: crate::forward_ty!(struct $arg_ty))*
        )
            -> crate::forward_ty!(struct $($ret)*)
        {
            crate::wrap_ret_val!(
                $($ret)* =>
                crate::peel(self).$fn_name(
                    $(crate::forward_arg!($arg_name: $arg_ty)),*
                )
            )
        }
    };
}
macro_rules! forward_fn_self_ref {
    (
        trait_decl =>
        $(#[
            $($attr:meta),*
        ])*
        fn $fn_name:ident (
            &self $(, $arg_name:ident: $arg_ty:tt)*
        )
        $(-> $($ret:tt)*)?
    ) => {
        #[allow(missing_docs, clippy::return_self_not_must_use)]
        fn $fn_name(
            &self $(, $arg_name: crate::forward_ty!(trait $arg_ty))*
        ) $(-> crate::forward_ty!(trait $($ret)*))*;
    };
    (
        trait_impl =>
        $(#[
            $($attr:meta),*
        ])*
        fn $fn_name:ident (
            &self $(, $arg_name:ident: $arg_ty:tt)*
        )
        $(-> $($ret:tt)*)?
    ) => {
        fn $fn_name(
            &self
            $(, $arg_name: crate::forward_ty!(trait $arg_ty))*
        )
            $(-> crate::forward_ty!(trait $($ret)*))*
        {
            <Self>::$fn_name(self $(, $arg_name)*)
        }
    };
    (
        struct =>
        $(#[
            $($attr:meta),*
        ])*
        fn $fn_name:ident (
            &self $(, $arg_name:ident: $arg_ty:tt)*
        )
        $(-> $($ret:tt)*)?
    ) => {
        $(#[$($attr),*])*
        #[must_use]
        #[inline]
        pub fn $fn_name(
            &self $(, $arg_name: crate::forward_ty!(struct $arg_ty))*
        )
            $(-> crate::forward_ty!(struct $($ret)*))*
        {
            crate::wrap_ret_val!(
                $($($ret)* => )*
                crate::peel_copy(self).$fn_name(
                    $(crate::forward_arg!($arg_name: $arg_ty)),*
                )
            )
        }
    };
}

macro_rules! forward_fn_assoc {
    (
        trait_decl =>
        $(#[
            $($attr:meta),*
        ])*
        fn $fn_name:ident (
            $($arg_name:ident: $arg_ty:tt),*
        )
        $(-> $ret:ident)?
    ) => {
        #[allow(missing_docs)]
        fn $fn_name($($arg_name: crate::forward_ty!(trait $arg_ty)),*) $(-> $ret)*;
    };

    (
        trait_impl =>
        $(#[
            $($attr:meta),*
        ])*
        fn $fn_name:ident (
            $($arg_name:ident: $arg_ty:tt),*
        )
        $(-> $ret:ident)?
    ) => {
        fn $fn_name(
            $($arg_name: crate::forward_ty!(trait $arg_ty)),*
        )
            $(-> $ret)*
        {
            <Self>::$fn_name($($arg_name),*)
        }
    };

    (
        struct =>
        $(#[
            $($attr:meta),*
        ])*
        fn $fn_name:ident (
            $($arg_name:ident: $arg_ty:tt),*
        )
        $(-> $ret:ident)?
    ) => {
        $(#[$($attr),*])*
        pub fn $fn_name(
            $($arg_name: crate::forward_ty!(struct $arg_ty)),*
        )
            $(-> $ret)*
        {
            crate::wrap_ret_val!(
                $($ret => )*
                <Self as crate::Transparent>::Wrapped::$fn_name($(crate::forward_arg!($arg_name: $arg_ty)),*))
        }
    };
}

/// Defines how to map types depending on context.
///
/// The `trait` context is for bindings to `glam`.
///
/// The `struct` context is the public API of `glamour`.
macro_rules! forward_ty {
    ($mode:tt Option<$thing:tt>) => {
        Option<crate::forward_ty!($mode $thing)>
    };
    (trait scalar) => {
        Self::Scalar
    };
    (struct scalar) => {
        T::Scalar
    };
    (trait uscalar) => {
        <Self::Scalar as crate::scalar::IntScalar>::Unsigned
    };
    (struct uscalar) => {
        <T::Scalar as crate::scalar::IntScalar>::Unsigned
    };
    (trait angle) => {
        Self::Scalar
    };
    (struct angle) => {
        crate::Angle<T::Scalar>
    };
    (trait vec2) => {
        <Self::Scalar as crate::Scalar>::Vec2
    };
    (struct vec2) => {
        Vector2<T>
    };
    (trait vec3) => {
        <Self::Scalar as crate::Scalar>::Vec3
    };
    (struct vec3) => {
        Vector3<T>
    };
    (trait vec4) => {
        <Self::Scalar as crate::Scalar>::Vec4
    };
    (struct vec4) => {
        crate::Vector4<T>
    };
    (trait point2) => {
        <Self::Scalar as crate::Scalar>::Vec2
    };
    (struct point2) => {
        crate::Point2<T>
    };
    (trait point3) => {
        <Self::Scalar as crate::Scalar>::Vec3
    };
    (struct point3) => {
        crate::Point3<T>
    };
    (trait point4) => {
        <Self::Scalar as crate::Scalar>::Vec4
    };
    (struct point4) => {
        crate::Point4<T>
    };
    (trait size2) => {
        <Self::Scalar as crate::Scalar>::Vec2
    };
    (struct size2) => {
        crate::Size2<T>
    };
    (trait size3) => {
        <Self::Scalar as crate::Scalar>::Vec3
    };
    (struct size3) => {
        crate::Size3<T>
    };
    (trait mat2) => {
        <Self::Scalar as crate::FloatScalar>::Mat2
    };
    (trait mat3) => {
        <Self::Scalar as crate::FloatScalar>::Mat3
    };
    (trait mat4) => {
        <Self::Scalar as crate::FloatScalar>::Mat4
    };
    (struct mat2) => {
        crate::Matrix2<T::Scalar>
    };
    (struct mat3) => {
        crate::Matrix3<T::Scalar>
    };
    (struct mat4) => {
        crate::Matrix4<T::Scalar>
    };
    (trait quat) => {
        <Self::Scalar as crate::FloatScalar>::Quat
    };
    (struct quat) => {
        <T::Scalar as crate::FloatScalar>::Quat
    };
    ($mode:tt bool) => { bool };
    ($mode:tt usize) => { usize };
    ($mode:tt bvec2) => { glam::BVec2 };
    ($mode:tt bvec3) => { glam::BVec3 };
    ($mode:tt bvec4) => { glam::BVec4 };
    ($mode:tt Self) => { Self };
    ($mode:tt ($a:tt, $b:tt)) => { (crate::forward_ty!($mode $a), crate::forward_ty!($mode $b)) };

    // These renames exist because the macros become significantly simpler when all types can be matched by a single
    // `tt`.
    ($mode:tt ref_self) => { &Self };
    ($mode:tt opt_self) => { Option<Self> };
    ($mode:tt ref_scalar_array_4) => { &[crate::forward_ty!($mode scalar); 4] };
    ($mode:tt ref_scalar_array_9) => { &[crate::forward_ty!($mode scalar); 9] };
    ($mode:tt ref_scalar_array_16) => { &[crate::forward_ty!($mode scalar); 16] };
    ($mode:tt ref_scalar_array_2_2) => { &[[crate::forward_ty!($mode scalar); 2]; 2] };
    ($mode:tt ref_scalar_array_3_3) => { &[[crate::forward_ty!($mode scalar); 3]; 3] };
    ($mode:tt ref_scalar_array_4_4) => { &[[crate::forward_ty!($mode scalar); 4]; 4] };
    ($mode:tt mut_scalar_slice) => { &mut [crate::forward_ty!($mode scalar)] };
    ($mode:tt [$t:tt; $n:literal]) => {
        [crate::forward_ty!($mode $t); $n]
    };
}

/// Convert argument from public interface to glam.
macro_rules! forward_arg {
    ($arg:ident: scalar) => {
        $arg
    };
    ($arg:ident: angle) => {
        crate::peel($arg)
    };
    ($arg:ident: vec2) => {
        crate::peel($arg)
    };
    ($arg:ident: vec3) => {
        crate::peel($arg)
    };
    ($arg:ident: vec4) => {
        crate::peel($arg)
    };
    ($arg:ident: point2) => {
        crate::peel($arg)
    };
    ($arg:ident: point3) => {
        crate::peel($arg)
    };
    ($arg:ident: point4) => {
        crate::peel($arg)
    };
    ($arg:ident: size2) => {
        crate::peel($arg)
    };
    ($arg:ident: size3) => {
        crate::peel($arg)
    };
    ($arg:ident: mat2) => {
        crate::peel($arg)
    };
    ($arg:ident: mat3) => {
        crate::peel($arg)
    };
    ($arg:ident: mat4) => {
        crate::peel($arg)
    };
    ($arg:ident: quat) => {
        $arg
    };
    ($arg:ident: bool) => {
        $arg
    };
    ($arg:ident: usize) => {
        $arg
    };
    ($arg:ident: bvec2) => {
        $arg
    };
    ($arg:ident: bvec3) => {
        $arg
    };
    ($arg:ident: bvec4) => {
        $arg
    };
    ($arg:ident: opt_self) => {
        $arg.map(crate::peel)
    };
    ($arg:ident: ($a:tt, $b:tt)) => {
        (crate::peel($a), crate::peel($b))
    };
    ($arg:ident: Self) => {
        crate::peel($arg)
    };
    ($arg:ident: ref_self) => {
        &crate::peel_copy($arg)
    };
    ($arg:ident: ref_scalar_array_4) => {
        $arg
    };
    ($arg:ident: ref_scalar_array_9) => {
        $arg
    };
    ($arg:ident: ref_scalar_array_16) => {
        $arg
    };
    ($arg:ident: ref_scalar_array_2_2) => {
        $arg
    };
    ($arg:ident: ref_scalar_array_3_3) => {
        $arg
    };
    ($arg:ident: ref_scalar_array_4_4) => {
        $arg
    };
    ($arg:ident: mut_scalar_slice) => {
        $arg
    };
    ($arg:ident: [$t:tt; $n:literal]) => {
        $arg
    };
}

macro_rules! wrap_ret_val {
    (Option<$thing:tt> => $arg:expr) => {
        $arg
    };
    ($arg:expr) => {
        $arg
    };
    ( scalar => $arg:expr) => {
        $arg
    };
    ( uscalar => $arg:expr) => {
        $arg
    };
    ( angle => $arg:expr) => {
        crate::wrap($arg)
    };
    ( vec2 => $arg:expr) => {
        crate::wrap($arg)
    };
    ( vec3 => $arg:expr) => {
        crate::wrap($arg)
    };
    ( vec4 => $arg:expr) => {
        crate::wrap($arg)
    };
    ( point2 => $arg:expr) => {
        crate::wrap($arg)
    };
    ( point3 => $arg:expr) => {
        crate::wrap($arg)
    };
    ( point4 => $arg:expr) => {
        crate::wrap($arg)
    };
    ( size2 => $arg:expr) => {
        crate::wrap($arg)
    };
    ( size3 => $arg:expr) => {
        crate::wrap($arg)
    };
    ( mat2 => $arg:expr) => {
        crate::wrap($arg)
    };
    ( mat3 => $arg:expr) => {
        crate::wrap($arg)
    };
    ( mat4 => $arg:expr) => {
        crate::wrap($arg)
    };
    ( quat => $arg:expr) => {
        $arg
    };
    ( bool => $arg:expr) => {
        $arg
    };
    ( usize => $arg:expr) => {
        $arg
    };
    ( bvec2 => $arg:expr) => {
        $arg
    };
    ( bvec3 => $arg:expr) => {
        $arg
    };
    ( bvec4 => $arg:expr) => {
        $arg
    };
    ( opt_self => $arg:expr) => {
        $arg.map(crate::wrap)
    };
    ( ($a:tt, $b:tt) => $arg:expr) => {{
        let (a, b) = $arg;
        (crate::wrap_ret_val!($a => a), crate::wrap_ret_val!($b => b))
    }};
    ( Self => $arg:expr) => {
        crate::wrap($arg)
    };
    ( ref_self => $arg:expr) => {
        crate::wrap_ref($arg)
    };
    ( ref_scalar_array_4 => $arg:expr) => {
        $arg
    };
    ( ref_scalar_array_9 => $arg:expr) => {
        $arg
    };
    ( ref_scalar_array_16 => $arg:expr) => {
        $arg
    };
    ( mut_scalar_slice => $arg:expr) => {
        $arg
    };
    ( [$t:tt; $n:literal] => $arg:expr) => {
        $arg
    };
}

macro_rules! interface {
    (
        $mode:tt =>
        $($t:tt)*
    ) => {
        crate::interface!(
            @accum
            $mode
            []
            []
            $($t)*
        );
    };

    (
        @accum
        $mode:tt
        [$($accum:tt)*]
        []
    ) => {
        $($accum)*
    };

    (
        @accum
        $mode:tt
        [$($accum:tt)*]
        [$($this_line:tt)*]
        ; $($rest:tt)*
    ) => {
        crate::interface! {
            @accum
            $mode
            [
                $($accum)*
                crate::forward_fn!($mode => $($this_line)*);
            ]
            []
            $($rest)*
        }
    };

    (
        @accum
        $mode:tt
        [ $($accumulated:tt)* ]
        [ $($this_line:tt)* ]
        $current:tt
        $($rest:tt)*
    ) => {
        crate::interface! { @accum $mode [ $($accumulated)* ] [ $($this_line)* $current ] $($rest)* }
    };
}

pub(crate) use forward_arg;
pub(crate) use forward_fn;
pub(crate) use forward_fn_assoc;
pub(crate) use forward_fn_self;
pub(crate) use forward_fn_self_ref;
pub(crate) use forward_ty;
pub(crate) use interface;
pub(crate) use wrap_ret_val;

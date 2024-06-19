#![allow(clippy::doc_markdown)]

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

pub(crate) use forward_neg_to_raw;
pub(crate) use forward_op_assign_to_raw;
pub(crate) use forward_op_to_raw;

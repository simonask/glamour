macro_rules! vector_ops {
    ($base_type_name:ident) => {
        crate::impl_ops::impl_ops! {
            Add<$base_type_name>::add for $base_type_name -> $base_type_name;
            Sub<$base_type_name>::sub for $base_type_name -> $base_type_name;
            Mul<$base_type_name>::mul for $base_type_name -> $base_type_name;
            Div<$base_type_name>::div for $base_type_name -> $base_type_name;
            Rem<$base_type_name>::rem for $base_type_name -> $base_type_name;
        }
        crate::impl_ops::impl_assign_ops! {
            AddAssign<$base_type_name>::add_assign for $base_type_name;
            SubAssign<$base_type_name>::sub_assign for $base_type_name;
            MulAssign<$base_type_name>::mul_assign for $base_type_name;
            DivAssign<$base_type_name>::div_assign for $base_type_name;
            RemAssign<$base_type_name>::rem_assign for $base_type_name;
        }

        impl<T: crate::SignedUnit> core::ops::Neg for $base_type_name<T> {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                crate::wrap(core::ops::Neg::neg(crate::peel(self)))
            }
        }

        // Need specific impls for each scalar types because the scalar is an associated type,
        // so the compiler would think that the trait impls could overlap.
        crate::impl_ops::vector_ops!(@scalar $base_type_name, [f32, f64, i32, u32, i64, u64, i16, u16]);
    };

    (@scalar $base_type_name:ident, [$($scalar:ident),*]) => {
        $(
            crate::impl_ops::impl_scalar_ops! {
                Add<$scalar>::add for $base_type_name -> $base_type_name;
                Sub<$scalar>::sub for $base_type_name -> $base_type_name;
                Mul<$scalar>::mul for $base_type_name -> $base_type_name;
                Div<$scalar>::div for $base_type_name -> $base_type_name;
                Rem<$scalar>::rem for $base_type_name -> $base_type_name;
            }
            crate::impl_ops::impl_scalar_ops_rhs! {
                Add<$scalar>::add for $base_type_name -> $base_type_name;
                Sub<$scalar>::sub for $base_type_name -> $base_type_name;
                Mul<$scalar>::mul for $base_type_name -> $base_type_name;
            }
            crate::impl_ops::impl_scalar_assign_ops! {
                AddAssign<$scalar>::add_assign for $base_type_name;
                SubAssign<$scalar>::sub_assign for $base_type_name;
                MulAssign<$scalar>::mul_assign for $base_type_name;
                DivAssign<$scalar>::div_assign for $base_type_name;
                RemAssign<$scalar>::rem_assign for $base_type_name;
            }
        )*
    }
}
pub(crate) use vector_ops;

macro_rules! point_ops {
    ($point_type_name:ident, $vector_type_name:ident) => {
        crate::impl_ops::impl_ops! {
            Add<$point_type_name>::add for $point_type_name -> $point_type_name;
            Add<$vector_type_name>::add for $point_type_name -> $point_type_name;
            Sub<$point_type_name>::sub for $point_type_name -> $vector_type_name;
            Sub<$vector_type_name>::sub for $point_type_name -> $point_type_name;
            Rem<$point_type_name>::rem for $point_type_name -> $vector_type_name;
        }
        crate::impl_ops::impl_assign_ops! {
            AddAssign<$point_type_name>::add_assign for $point_type_name;
            AddAssign<$vector_type_name>::add_assign for $point_type_name;
            SubAssign<$vector_type_name>::sub_assign for $point_type_name;
            RemAssign<$vector_type_name>::rem_assign for $point_type_name;
        }
        impl<T: crate::SignedUnit> core::ops::Neg for $point_type_name<T> {
            type Output = Self;
            #[inline(always)]
            fn neg(self) -> Self {
                crate::wrap(core::ops::Neg::neg(crate::peel(self)))
            }
        }

        // Need specific impls for each scalar types because the scalar is an associated type,
        // so the compiler would think that the trait impls could overlap.
        crate::impl_ops::point_ops!(@scalar $point_type_name, [f32, f64, i32, u32, i64, u64, i16, u16]);
    };

    (@scalar $base_type_name:ident, [$($scalar:ident),*]) => {
        $(
            crate::impl_ops::impl_scalar_ops! {
                Add<$scalar>::add for $base_type_name -> $base_type_name;
                Sub<$scalar>::sub for $base_type_name -> $base_type_name;
                Mul<$scalar>::mul for $base_type_name -> $base_type_name;
                Div<$scalar>::div for $base_type_name -> $base_type_name;
                Rem<$scalar>::rem for $base_type_name -> $base_type_name;
            }
            crate::impl_ops::impl_scalar_ops_rhs! {
                Add<$scalar>::add for $base_type_name -> $base_type_name;
                Sub<$scalar>::sub for $base_type_name -> $base_type_name;
                Mul<$scalar>::mul for $base_type_name -> $base_type_name;
            }
            crate::impl_ops::impl_scalar_assign_ops! {
                AddAssign<$scalar>::add_assign for $base_type_name;
                SubAssign<$scalar>::sub_assign for $base_type_name;
                MulAssign<$scalar>::mul_assign for $base_type_name;
                DivAssign<$scalar>::div_assign for $base_type_name;
                RemAssign<$scalar>::rem_assign for $base_type_name;
            }
        )*
    }
}
pub(crate) use point_ops;

macro_rules! size_ops {
    ($base_type_name:ident) => {
        crate::impl_ops::impl_ops! {
            Add<$base_type_name>::add for $base_type_name -> $base_type_name;
            Sub<$base_type_name>::sub for $base_type_name -> $base_type_name;
            Mul<$base_type_name>::mul for $base_type_name -> $base_type_name;
            Div<$base_type_name>::div for $base_type_name -> $base_type_name;
            Rem<$base_type_name>::rem for $base_type_name -> $base_type_name;
        }
        crate::impl_ops::impl_assign_ops! {
            AddAssign<$base_type_name>::add_assign for $base_type_name;
            SubAssign<$base_type_name>::sub_assign for $base_type_name;
            MulAssign<$base_type_name>::mul_assign for $base_type_name;
            DivAssign<$base_type_name>::div_assign for $base_type_name;
            RemAssign<$base_type_name>::rem_assign for $base_type_name;
        }
        // Need specific impls for each scalar types because the scalar is an associated type,
        // so the compiler would think that the trait impls could overlap.
        crate::impl_ops::vector_ops!(@scalar $base_type_name, [f32, f64, i32, u32, i64, u64, i16, u16]);
    };
}
pub(crate) use size_ops;

macro_rules! impl_ops {
    (
        $(
            $op_trait:ident < $rhs_ty:ident > :: $op_name:ident for $lhs_ty:ident -> $out_ty:ident
        );*
        $(;)?
    ) => {
        $(
            impl<T: crate::Unit> core::ops::$op_trait<$rhs_ty<T>> for $lhs_ty<T> {
                type Output = $out_ty<T>;
                fn $op_name(self, rhs: $rhs_ty<T>) -> $out_ty<T> {
                    crate::wrap(core::ops::$op_trait::$op_name(crate::peel(self), crate::peel(rhs)))
                }
            }
            impl<T: crate::Unit> core::ops::$op_trait<&$rhs_ty<T>> for $lhs_ty<T> {
                type Output = $out_ty<T>;
                fn $op_name(self, rhs: &$rhs_ty<T>) -> $out_ty<T> {
                    crate::wrap(core::ops::$op_trait::$op_name(crate::peel(self), crate::peel(*rhs)))
                }
            }
            impl<T: crate::Unit> core::ops::$op_trait<$rhs_ty<T>> for &$lhs_ty<T> {
                type Output = $out_ty<T>;
                fn $op_name(self, rhs: $rhs_ty<T>) -> $out_ty<T> {
                    crate::wrap(core::ops::$op_trait::$op_name(crate::peel(*self), crate::peel(rhs)))
                }
            }
            impl<T: crate::Unit> core::ops::$op_trait<&$rhs_ty<T>> for &$lhs_ty<T> {
                type Output = $out_ty<T>;
                fn $op_name(self, rhs: &$rhs_ty<T>) -> $out_ty<T> {
                    crate::wrap(core::ops::$op_trait::$op_name(crate::peel(*self), crate::peel(*rhs)))
                }
            }
        )*
    };
}
pub(crate) use impl_ops;

macro_rules! impl_scalar_ops {
    (
        $(
            $op_trait:ident < $rhs_ty:ident > :: $op_name:ident for $lhs_ty:ident -> $out_ty:ident
        );*
        $(;)?
    ) => {
        $(
            impl<T: crate::Unit<Scalar = $rhs_ty>> core::ops::$op_trait<$rhs_ty> for $lhs_ty<T> {
                type Output = $out_ty<T>;
                fn $op_name(self, rhs: $rhs_ty) -> $out_ty<T> {
                    crate::wrap(core::ops::$op_trait::$op_name(crate::peel(self), rhs))
                }
            }
            impl<T: crate::Unit<Scalar = $rhs_ty>> core::ops::$op_trait<&$rhs_ty> for $lhs_ty<T> {
                type Output = $out_ty<T>;
                fn $op_name(self, rhs: &$rhs_ty) -> $out_ty<T> {
                    crate::wrap(core::ops::$op_trait::$op_name(crate::peel(self), *rhs))
                }
            }
            impl<T: crate::Unit<Scalar = $rhs_ty>> core::ops::$op_trait<$rhs_ty> for &$lhs_ty<T> {
                type Output = $out_ty<T>;
                fn $op_name(self, rhs: $rhs_ty) -> $out_ty<T> {
                    crate::wrap(core::ops::$op_trait::$op_name(crate::peel(*self), rhs))
                }
            }
            impl<T: crate::Unit<Scalar = $rhs_ty>> core::ops::$op_trait<&$rhs_ty> for &$lhs_ty<T> {
                type Output = $out_ty<T>;
                fn $op_name(self, rhs: &$rhs_ty) -> $out_ty<T> {
                    crate::wrap(core::ops::$op_trait::$op_name(crate::peel(*self), *rhs))
                }
            }
        )*
    };
}
pub(crate) use impl_scalar_ops;

macro_rules! impl_scalar_ops_rhs {
    (
        $(
            $op_trait:ident < $rhs_ty:ident > :: $op_name:ident for $lhs_ty:ident -> $out_ty:ident
        );*
        $(;)?
    ) => {
        $(
            impl<T: crate::Unit<Scalar = $rhs_ty>> core::ops::$op_trait<$lhs_ty<T>> for $rhs_ty {
                type Output = $out_ty<T>;
                fn $op_name(self, rhs: $lhs_ty<T>) -> $out_ty<T> {
                    crate::wrap(core::ops::$op_trait::$op_name(crate::peel(rhs), self))
                }
            }
            impl<T: crate::Unit<Scalar = $rhs_ty>> core::ops::$op_trait<&$lhs_ty<T>> for $rhs_ty {
                type Output = $out_ty<T>;
                fn $op_name(self, rhs: &$lhs_ty<T>) -> $out_ty<T> {
                    crate::wrap(core::ops::$op_trait::$op_name(crate::peel(*rhs), self))
                }
            }
            impl<T: crate::Unit<Scalar = $rhs_ty>> core::ops::$op_trait<$lhs_ty<T>> for &$rhs_ty {
                type Output = $out_ty<T>;
                fn $op_name(self, rhs: $lhs_ty<T>) -> $out_ty<T> {
                    crate::wrap(core::ops::$op_trait::$op_name(crate::peel(rhs), *self))
                }
            }
            impl<T: crate::Unit<Scalar = $rhs_ty>> core::ops::$op_trait<&$lhs_ty<T>> for &$rhs_ty {
                type Output = $out_ty<T>;
                fn $op_name(self, rhs: &$lhs_ty<T>) -> $out_ty<T> {
                    crate::wrap(core::ops::$op_trait::$op_name(crate::peel(*rhs), *self))
                }
            }
        )*
    };
}
pub(crate) use impl_scalar_ops_rhs;

macro_rules! impl_assign_ops {
    (
        $(
            $op_trait:ident <$rhs_ty:ident> :: $op_name:ident for $lhs_ty:ident
        );*
        $(;)?
    ) => {
        $(
            impl<T: crate::Unit> core::ops::$op_trait<$rhs_ty<T>> for $lhs_ty<T> {
                fn $op_name(&mut self, rhs: $rhs_ty<T>) {
                    core::ops::$op_trait::$op_name(crate::peel_mut(self), crate::peel(rhs))
                }
            }
            impl<T: crate::Unit> core::ops::$op_trait<&$rhs_ty<T>> for $lhs_ty<T> {
                fn $op_name(&mut self, rhs: &$rhs_ty<T>) {
                    core::ops::$op_trait::$op_name(crate::peel_mut(self), crate::peel(*rhs))
                }
            }
        )*
    }
}
pub(crate) use impl_assign_ops;

macro_rules! impl_scalar_assign_ops {
    (
        $(
            $op_trait:ident < $rhs_ty:ident > :: $op_name:ident for $lhs_ty:ident
        );*
        $(;)?
    ) => {
        $(
            impl<T: crate::Unit<Scalar = $rhs_ty>> core::ops::$op_trait<$rhs_ty> for $lhs_ty<T> {
                fn $op_name(&mut self, rhs: $rhs_ty) {
                    core::ops::$op_trait::$op_name(crate::peel_mut(self), rhs);
                }
            }
            impl<T: crate::Unit<Scalar = $rhs_ty>> core::ops::$op_trait<&$rhs_ty> for $lhs_ty<T> {
                fn $op_name(&mut self, rhs: &$rhs_ty) {
                    core::ops::$op_trait::$op_name(crate::peel_mut(self), *rhs);
                }
            }
        )*
    };
}
pub(crate) use impl_scalar_assign_ops;

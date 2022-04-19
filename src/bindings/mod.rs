//! Binding traits to [`glam`] types.
//!
//! These traits generalize the glam types such that they can be used by generic
//! code.
//!
//! **CAUTION:** Everything in this module should be considered effectively
//! private.

use core::fmt::{Debug, Display};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

use approx::AbsDiffEq;
use num_traits::Float;

use crate::traits::marker::ValueSemantics;

mod matrix;
mod primitive;
mod quat;
mod vec;

pub use matrix::*;
pub use primitive::*;
pub use quat::*;
pub use vec::*;

macro_rules! forward_impl {
    ($base:ty => fn $name:ident ( &self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty) => {
        #[inline]
        fn $name(&self $(, $arg:$arg_ty)*) -> $ret {
            <$base>::$name(self $(, ($arg).into())*).into()
        }
    };
    ($base:ty: $base_fn:ident => fn $name:ident ( &self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty) => {
        #[inline]
        fn $name(&self $(, $arg:$arg_ty)*) -> $ret {
            <$base>::$base_fn(self $(, ($arg).into())*).into()
        }
    };
    ($base:ty => fn $name:ident ( &mut self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty) => {
        #[inline]
        fn $name(&mut self $(, $arg:$arg_ty)*) -> $ret {
            <$base>::$name(self $(, ($arg).into())*).into()
        }
    };
    ($base:ty => fn $name:ident ( self $(, $arg:ident : $arg_ty:ty)*) -> $ret:ty) => {
        #[inline]
        fn $name(self $(, $arg:$arg_ty)*) -> $ret {
            <$base>::$name(self $(, ($arg).into())*).into()
        }
    };
    ($base:ty => fn $name:ident ( $($arg:ident : $arg_ty:ty),*) -> $ret:ty) => {
        #[inline]
        fn $name($($arg:$arg_ty),*) -> $ret {
            <$base>::$name($(($arg).into()),*).into()
        }
    };
}

use forward_impl;

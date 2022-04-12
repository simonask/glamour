//! Binding traits to [`glam`] types.
//!
//! These traits generalize the glam types such that they can be used by generic
//! code.

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

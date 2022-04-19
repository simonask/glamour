use glamour::prelude::*;

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    derive_more::Add,
    derive_more::AddAssign,
    derive_more::Div,
    derive_more::DivAssign,
    derive_more::Mul,
    derive_more::MulAssign,
    derive_more::Rem,
    derive_more::RemAssign,
    derive_more::Sub,
    derive_more::SubAssign,
    bytemuck::Pod,
    bytemuck::Zeroable,
)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[repr(C)]
struct MyInt(i32);

impl Scalar for MyInt {
    type Primitive = i32;
    const ZERO: Self = MyInt(0);
    const ONE: Self = MyInt(1);
}

impl Unit for MyInt {
    type Scalar = MyInt;
}

#[test]
fn custom_scalars() {
    let my_vec: Vector4<MyInt> = (MyInt(1), MyInt(2), MyInt(3), MyInt(4)).into();
    assert_eq!(my_vec[0], MyInt(1));
}

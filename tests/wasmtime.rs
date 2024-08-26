#![cfg(feature = "wasmtime")]

use core::f32;

use ::glamour::prelude::*;

pub mod types {
    pub type Vector2 = glamour::Vector2<f32>;
    pub type Vector3 = glamour::Vector3<f32>;
    pub type Vector4 = glamour::Vector4<f32>;
    pub type Point2 = glamour::Point2<f32>;
    pub type Point3 = glamour::Point3<f32>;
    pub type Point4 = glamour::Point4<f32>;
    pub type Size2 = glamour::Size2<f32>;
    pub type Size3 = glamour::Size3<f32>;

    pub type Ivector2 = glamour::Vector2<i32>;
    pub type Ivector3 = glamour::Vector3<i32>;
    pub type Ivector4 = glamour::Vector4<i32>;
    pub type Ipoint2 = glamour::Point2<i32>;
    pub type Ipoint3 = glamour::Point3<i32>;
    pub type Ipoint4 = glamour::Point4<i32>;
    pub type Isize2 = glamour::Size2<i32>;
    pub type Isize3 = glamour::Size3<i32>;

    pub type Angle = glamour::Angle<f32>;
    pub type Rect = glamour::Rect<f32>;
    pub type Box2 = glamour::Box2<f32>;
    pub type Box3 = glamour::Box3<f32>;

    pub type Mat2 = glamour::Matrix2<f32>;
    pub type Mat3 = glamour::Matrix3<f32>;
    pub type Mat4 = glamour::Matrix4<f32>;

    pub fn add_to_linker<T, U>(
        _linker: &mut wasmtime::component::Linker<T>,
        _get: impl Fn(&mut T) -> &mut U + Send + Sync + Copy + 'static,
    ) -> wasmtime::Result<()> {
        Ok(())
    }
    pub trait Host {}
    impl<T> Host for T {}
}

wasmtime::component::bindgen!({
    path: "tests/wasmtime-guest/wit/world.wit",
    world: "default",
    with: {
        "glamour:wasmtime-guest/glamour@0.1.0": crate::types,
    }
});

// Note: Run `make -f rebuild-wasmtime-guest.make` in the project root to rebuild this.
static GUEST_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/wasmtime_guest.wasm");

struct State {}

#[test]
fn roundtrip() {
    let engine = wasmtime::Engine::default();
    let component = wasmtime::component::Component::from_file(&engine, GUEST_PATH).unwrap();
    let mut linker = wasmtime::component::Linker::<State>::new(&engine);
    Default::add_to_linker(&mut linker, |s| s).unwrap();

    let mut store = wasmtime::Store::new(&engine, State {});
    let instance = Default::instantiate(&mut store, &component, &linker).unwrap();
    let guest = instance.glamour_wasmtime_guest_client();

    let all = exports::glamour::wasmtime_guest::client::All {
        vector2: vector![1.0, 2.0],
        vector3: vector![1.0, 2.0, 3.0],
        vector4: vector![1.0, 2.0, f32::INFINITY, 4.0],
        point2: point![1.0, 2.0],
        point3: point![1.0, 2.0, 3.0],
        point4: point![1.0, 2.0, 3.0, 4.0],
        size2: size![1.0, 2.0],
        size3: size![1.0, 2.0, f32::NAN],
        ivector2: vector![1, 2],
        ivector3: vector![1, 2, 3],
        ivector4: vector![1, 2, 3, 4],
        ipoint2: point![1, 2],
        ipoint3: point![1, 2, 3],
        ipoint4: point![1, 2, 3, 4],
        isize2: size![1, 2],
        isize3: size![1, 2, 3],
        angle: Angle::HALF_CIRCLE,
        rect: Rect::new(point![1.0, 2.0], size![3.0, 4.0]),
        box2: Box2::new(point![1.0, 2.0], point![3.0, 4.0]),
        box3: Box3::new(point![1.0, 2.0, 3.0], point![4.0, 5.0, 6.0]),
        mat3: Matrix3::from_scale(vector![2.0, 3.0]),
        mat4: Matrix4::from_scale(vector![2.0, 3.0, 4.0]),
    };

    let ret = guest.call_roundtrip(&mut store, all).unwrap();

    assert_eq!(all.vector2, ret.vector2);
    assert_eq!(all.vector3, ret.vector3);
    assert_eq!(all.vector4, ret.vector4);
    assert_eq!(all.point2, ret.point2);
    assert_eq!(all.point3, ret.point3);
    assert_eq!(all.point4, ret.point4);
    assert_eq!(all.size2, ret.size2);
    assert_eq!(all.size3.width, ret.size3.width);
    assert_eq!(all.size3.height, ret.size3.height);
    assert!(ret.size3.depth.is_nan());
    assert_eq!(all.ivector2, ret.ivector2);
    assert_eq!(all.ivector3, ret.ivector3);
    assert_eq!(all.ivector4, ret.ivector4);
    assert_eq!(all.ipoint2, ret.ipoint2);
    assert_eq!(all.ipoint3, ret.ipoint3);
    assert_eq!(all.ipoint4, ret.ipoint4);
    assert_eq!(all.isize2, ret.isize2);
    assert_eq!(all.isize3, ret.isize3);
    assert_eq!(all.angle, ret.angle);
    assert_eq!(all.rect, ret.rect);
    assert_eq!(all.box2, ret.box2);
    assert_eq!(all.box3, ret.box3);
    assert_eq!(all.mat3, ret.mat3);
    assert_eq!(all.mat4, ret.mat4);
}

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
}

wit_bindgen::generate!({
    path: "wit/world.wit",
    world: "default",
    with: {
        "glamour:wasmtime-guest/glamour@0.1.0": crate::types,
        "glamour:wasmtime-guest/client@0.1.0": generate,
    }
});

export!(Guest);

struct Guest {}

impl exports::glamour::wasmtime_guest::client::Guest for Guest {
    fn roundtrip(
        all: exports::glamour::wasmtime_guest::client::All,
    ) -> exports::glamour::wasmtime_guest::client::All {
        all
    }
}

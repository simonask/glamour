use glamour::prelude::*;

#[test]
fn vec_macros() {
    let a: Vector2<f32> = vec2!(1.0, 2.0);
    let b: Vector2<f32> = vec2![1.0, 2.0];
    let c: Vector2<f32> = vec2!(2.0);
    assert_eq!(a.x, 1.0);
    assert_eq!(a.y, 2.0);
    assert_eq!(b.x, 1.0);
    assert_eq!(b.y, 2.0);
    assert_eq!(c.x, 2.0);
    assert_eq!(c.y, 2.0);

    let a: Vector3<f32> = vec3!(1.0, 2.0, 3.0);
    let b: Vector3<f32> = vec3![1.0, 2.0, 3.0];
    let c: Vector3<f32> = vec3!(2.0);
    assert_eq!(a.x, 1.0);
    assert_eq!(a.y, 2.0);
    assert_eq!(a.z, 3.0);
    assert_eq!(b.x, 1.0);
    assert_eq!(b.y, 2.0);
    assert_eq!(b.z, 3.0);
    assert_eq!(c.x, 2.0);
    assert_eq!(c.y, 2.0);
    assert_eq!(c.z, 2.0);

    let a: Vector4<f32> = vec4!(1.0, 2.0, 3.0, 4.0);
    let b: Vector4<f32> = vec4![1.0, 2.0, 3.0, 4.0];
    let c: Vector4<f32> = vec4!(2.0);
    assert_eq!(a.x, 1.0);
    assert_eq!(a.y, 2.0);
    assert_eq!(a.z, 3.0);
    assert_eq!(a.w, 4.0);
    assert_eq!(b.x, 1.0);
    assert_eq!(b.y, 2.0);
    assert_eq!(b.z, 3.0);
    assert_eq!(b.w, 4.0);
    assert_eq!(c.x, 2.0);
    assert_eq!(c.y, 2.0);
    assert_eq!(c.z, 2.0);
    assert_eq!(c.w, 2.0);

    let a: Vector2<f32> = vector!(1.0, 2.0);
    let b: Vector3<f32> = vector!(1.0, 2.0, 3.0);
    let c: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
    assert_eq!(a.x, 1.0);
    assert_eq!(a.y, 2.0);
    assert_eq!(b.x, 1.0);
    assert_eq!(b.y, 2.0);
    assert_eq!(b.z, 3.0);
    assert_eq!(c.x, 1.0);
    assert_eq!(c.y, 2.0);
    assert_eq!(c.z, 3.0);
    assert_eq!(c.w, 4.0);

    let a: Vector2<f32> = vector!([2.0; 2]);
    let b: Vector3<f32> = vector!([3.0; 3]);
    let c: Vector4<f32> = vector!([4.0; 4]);
    assert_eq!(a.x, 2.0);
    assert_eq!(a.y, 2.0);
    assert_eq!(b.x, 3.0);
    assert_eq!(b.y, 3.0);
    assert_eq!(b.z, 3.0);
    assert_eq!(c.x, 4.0);
    assert_eq!(c.y, 4.0);
    assert_eq!(c.z, 4.0);
    assert_eq!(c.w, 4.0);
}

#[test]
fn point_macros() {
    let a: Point2<f32> = point2!(1.0, 2.0);
    let b: Point2<f32> = point2![1.0, 2.0];
    let c: Point2<f32> = point2!(2.0);
    assert_eq!(a.x, 1.0);
    assert_eq!(a.y, 2.0);
    assert_eq!(b.x, 1.0);
    assert_eq!(b.y, 2.0);
    assert_eq!(c.x, 2.0);
    assert_eq!(c.y, 2.0);

    let a: Point3<f32> = point3!(1.0, 2.0, 3.0);
    let b: Point3<f32> = point3![1.0, 2.0, 3.0];
    let c: Point3<f32> = point3!(2.0);
    assert_eq!(a.x, 1.0);
    assert_eq!(a.y, 2.0);
    assert_eq!(a.z, 3.0);
    assert_eq!(b.x, 1.0);
    assert_eq!(b.y, 2.0);
    assert_eq!(b.z, 3.0);
    assert_eq!(c.x, 2.0);
    assert_eq!(c.y, 2.0);
    assert_eq!(c.z, 2.0);

    let a: Point4<f32> = point4!(1.0, 2.0, 3.0, 4.0);
    let b: Point4<f32> = point4![1.0, 2.0, 3.0, 4.0];
    let c: Point4<f32> = point4!(2.0);
    assert_eq!(a.x, 1.0);
    assert_eq!(a.y, 2.0);
    assert_eq!(a.z, 3.0);
    assert_eq!(a.w, 4.0);
    assert_eq!(b.x, 1.0);
    assert_eq!(b.y, 2.0);
    assert_eq!(b.z, 3.0);
    assert_eq!(b.w, 4.0);
    assert_eq!(c.x, 2.0);
    assert_eq!(c.y, 2.0);
    assert_eq!(c.z, 2.0);
    assert_eq!(c.w, 2.0);

    let a: Point2<f32> = point!(1.0, 2.0);
    let b: Point3<f32> = point!(1.0, 2.0, 3.0);
    let c: Point4<f32> = point!(1.0, 2.0, 3.0, 4.0);
    assert_eq!(a.x, 1.0);
    assert_eq!(a.y, 2.0);
    assert_eq!(b.x, 1.0);
    assert_eq!(b.y, 2.0);
    assert_eq!(b.z, 3.0);
    assert_eq!(c.x, 1.0);
    assert_eq!(c.y, 2.0);
    assert_eq!(c.z, 3.0);
    assert_eq!(c.w, 4.0);

    let a: Point2<f32> = point!([2.0; 2]);
    let b: Point3<f32> = point!([3.0; 3]);
    let c: Point4<f32> = point!([4.0; 4]);
    assert_eq!(a.x, 2.0);
    assert_eq!(a.y, 2.0);
    assert_eq!(b.x, 3.0);
    assert_eq!(b.y, 3.0);
    assert_eq!(b.z, 3.0);
    assert_eq!(c.x, 4.0);
    assert_eq!(c.y, 4.0);
    assert_eq!(c.z, 4.0);
    assert_eq!(c.w, 4.0);
}

#[test]
fn size_macros() {
    let a: Size2<f32> = size2!(1.0, 2.0);
    let b: Size2<f32> = size2![1.0, 2.0];
    let c: Size2<f32> = size2!(2.0);
    assert_eq!(a.width, 1.0);
    assert_eq!(a.height, 2.0);
    assert_eq!(b.width, 1.0);
    assert_eq!(b.height, 2.0);
    assert_eq!(c.width, 2.0);
    assert_eq!(c.height, 2.0);

    let a: Size3<f32> = size3!(1.0, 2.0, 3.0);
    let b: Size3<f32> = size3![1.0, 2.0, 3.0];
    let c: Size3<f32> = size3!(2.0);
    assert_eq!(a.width, 1.0);
    assert_eq!(a.height, 2.0);
    assert_eq!(a.depth, 3.0);
    assert_eq!(b.width, 1.0);
    assert_eq!(b.height, 2.0);
    assert_eq!(b.depth, 3.0);
    assert_eq!(c.width, 2.0);
    assert_eq!(c.height, 2.0);
    assert_eq!(c.depth, 2.0);

    let a: Size2<f32> = size!(1.0, 2.0);
    let b: Size3<f32> = size!(1.0, 2.0, 3.0);
    assert_eq!(a.width, 1.0);
    assert_eq!(a.height, 2.0);
    assert_eq!(b.width, 1.0);
    assert_eq!(b.height, 2.0);
    assert_eq!(b.depth, 3.0);

    let a: Size2<f32> = size!([2.0; 2]);
    let b: Size3<f32> = size!([3.0; 3]);
    assert_eq!(a.width, 2.0);
    assert_eq!(a.height, 2.0);
    assert_eq!(b.width, 3.0);
    assert_eq!(b.height, 3.0);
    assert_eq!(b.depth, 3.0);
}

// Verify that the macros can be used in a const context.
const _CONST_TEST_VEC: Vector4<f32> = vector!(1.0, 2.0, 3.0, 4.0);
const _CONST_TEST_POINT: Point4<f32> = point!(1.0, 2.0, 3.0, 4.0);
const _CONST_TEST_SIZE: Size3<f32> = size!(1.0, 2.0, 3.0);

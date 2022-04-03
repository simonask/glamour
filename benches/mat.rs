use glam::*;
use glamour::prelude::*;

use criterion::{black_box, criterion_group, criterion_main, Bencher, Criterion};

struct MyUnit;
impl Unit for MyUnit {
    type Scalar = f32;
}

fn mat4_transform_glamour(b: &mut Bencher) {
    let vector: Vector3<MyUnit> = (100.0, 200.0, 300.0).into();
    let matrix: Matrix4<f32> =
        Matrix4::from_axis_angle(Vector3::unit_y(), Angle::from_degrees(45.0));

    b.iter(|| matrix.transform_vector(black_box(vector.to_untyped())))
}

fn mat4_transform_glam(b: &mut Bencher) {
    let vector = Vec3::new(100.0, 200.0, 300.0);
    let matrix = Mat4::from_axis_angle(Vec3::Y, 45.0f32.to_radians());

    b.iter(|| matrix.transform_vector3(black_box(vector)))
}

fn mat3_transform_glamour(b: &mut Bencher) {
    let vector: Vector2<MyUnit> = (100.0, 200.0).into();
    let matrix: Matrix3<f32> = Matrix3::from_angle(Angle::from_degrees(45.0));

    b.iter(|| matrix.transform_vector(black_box(vector.to_untyped())))
}

fn mat3_transform_glam(b: &mut Bencher) {
    let vector = Vec2::new(100.0, 200.0);
    let matrix = Mat3::from_angle(45.0f32.to_radians());

    b.iter(|| matrix.transform_vector2(black_box(vector)))
}

fn mat3a_transform_glam(b: &mut Bencher) {
    let vector = Vec2::new(100.0, 200.0);
    let matrix = Mat3A::from_angle(45.0f32.to_radians());

    b.iter(|| matrix.transform_vector2(black_box(vector)))
}

fn glamour_benchmark(c: &mut Criterion) {
    c.bench_function("mat4 transform (glamour)", mat4_transform_glamour);
    c.bench_function("mat4 transform (glam)", mat4_transform_glam);
    c.bench_function("mat3 transform (glamour)", mat3_transform_glamour);
    c.bench_function("mat3 transform (glam)", mat3_transform_glam);
    c.bench_function("mat3a transform (glam)", mat3a_transform_glam);
}

criterion_group!(benches, glamour_benchmark);
criterion_main!(benches);

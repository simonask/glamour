use glamour::{prelude::*};

#[test]
fn into_iter() {
    let v: Vector4<i32> = vec4![1, 2, 3, 4];
    let mut c = 0;
    for i in v {
        c += i;
    }
    assert_eq!(c, 10);
}

#[test]
fn size() {
    use core::mem::size_of;

    assert_eq!(size_of::<glam::Vec2>(), size_of::<Vector2::<f32>>());
    assert_eq!(size_of::<glam::Vec3>(), size_of::<Vector3::<f32>>());
    assert_eq!(size_of::<glam::Vec4>(), size_of::<Vector4::<f32>>());
    assert_eq!(size_of::<glam::DVec2>(), size_of::<Vector2::<f64>>());
    assert_eq!(size_of::<glam::DVec3>(), size_of::<Vector3::<f64>>());
    assert_eq!(size_of::<glam::DVec4>(), size_of::<Vector4::<f64>>());
    assert_eq!(size_of::<glam::IVec2>(), size_of::<Vector2::<i32>>());
    assert_eq!(size_of::<glam::IVec3>(), size_of::<Vector3::<i32>>());
    assert_eq!(size_of::<glam::IVec4>(), size_of::<Vector4::<i32>>());
    assert_eq!(size_of::<glam::UVec2>(), size_of::<Vector2::<u32>>());
    assert_eq!(size_of::<glam::UVec3>(), size_of::<Vector3::<u32>>());
    assert_eq!(size_of::<glam::UVec4>(), size_of::<Vector4::<u32>>());

    assert_eq!(size_of::<glam::Vec2>(), size_of::<Point2::<f32>>());
    assert_eq!(size_of::<glam::Vec3>(), size_of::<Point3::<f32>>());
    assert_eq!(size_of::<glam::Vec4>(), size_of::<Point4::<f32>>());
    assert_eq!(size_of::<glam::DVec2>(), size_of::<Point2::<f64>>());
    assert_eq!(size_of::<glam::DVec3>(), size_of::<Point3::<f64>>());
    assert_eq!(size_of::<glam::DVec4>(), size_of::<Point4::<f64>>());
    assert_eq!(size_of::<glam::IVec2>(), size_of::<Point2::<i32>>());
    assert_eq!(size_of::<glam::IVec3>(), size_of::<Point3::<i32>>());
    assert_eq!(size_of::<glam::IVec4>(), size_of::<Point4::<i32>>());
    assert_eq!(size_of::<glam::UVec2>(), size_of::<Point2::<u32>>());
    assert_eq!(size_of::<glam::UVec3>(), size_of::<Point3::<u32>>());
    assert_eq!(size_of::<glam::UVec4>(), size_of::<Point4::<u32>>());

    assert_eq!(size_of::<glam::Vec2>(), size_of::<Size2::<f32>>());
    assert_eq!(size_of::<glam::Vec3>(), size_of::<Size3::<f32>>());
    assert_eq!(size_of::<glam::DVec2>(), size_of::<Size2::<f64>>());
    assert_eq!(size_of::<glam::DVec3>(), size_of::<Size3::<f64>>());
    assert_eq!(size_of::<glam::IVec2>(), size_of::<Size2::<i32>>());
    assert_eq!(size_of::<glam::IVec3>(), size_of::<Size3::<i32>>());
    assert_eq!(size_of::<glam::UVec2>(), size_of::<Size2::<u32>>());
    assert_eq!(size_of::<glam::UVec3>(), size_of::<Size3::<u32>>());

    assert_eq!(size_of::<glam::Mat2>(), size_of::<Matrix2<f32>>());
    assert_eq!(size_of::<glam::Mat3>(), size_of::<Matrix3<f32>>());
    assert_eq!(size_of::<glam::Mat4>(), size_of::<Matrix4<f32>>());
    assert_eq!(size_of::<glam::DMat2>(), size_of::<Matrix2<f64>>());
    assert_eq!(size_of::<glam::DMat3>(), size_of::<Matrix3<f64>>());
    assert_eq!(size_of::<glam::DMat4>(), size_of::<Matrix4<f64>>());

    assert_eq!(size_of::<glam::Mat3>(), size_of::<Transform2<f32, f32>>());
    assert_eq!(size_of::<glam::Mat4>(), size_of::<Transform3<f32, f32>>());
    assert_eq!(size_of::<glam::DMat3>(), size_of::<Transform2<f64, f64>>());
    assert_eq!(size_of::<glam::DMat4>(), size_of::<Transform3<f64, f64>>());
}


#[test]
fn from_into_glam() {
    let f: Vector4<f32> = vec4![1.0, 2.0, 3.0, 4.0];
    let d: Vector4<f64> = vec4![1.0, 2.0, 3.0, 4.0];
    let i: Vector4<i32> = vec4![1, 2, 3, 4];
    let u: Vector4<u32> = vec4![1, 2, 3, 4];

    let f: glam::Vec4 = f.into();
    let d: glam::DVec4 = d.into();
    let i: glam::IVec4 = i.into();
    let u: glam::UVec4 = u.into();

    assert_eq!(Into::<Vector4<f32>>::into(f), vec4!(1.0, 2.0, 3.0, 4.0));
    assert_eq!(Into::<Vector4<f64>>::into(d), vec4!(1.0, 2.0, 3.0, 4.0));
    assert_eq!(Into::<Vector4<i32>>::into(i), vec4!(1, 2, 3, 4));
    assert_eq!(Into::<Vector4<u32>>::into(u), vec4!(1, 2, 3, 4));
}

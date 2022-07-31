use glamour::prelude::*;

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
fn alignment() {
    use core::mem::align_of;

    assert_eq!(align_of::<glam::Vec2>(), align_of::<Vector2::<f32>>());
    assert_eq!(align_of::<glam::Vec3>(), align_of::<Vector3::<f32>>());
    assert_eq!(align_of::<glam::Vec4>(), align_of::<Vector4::<f32>>());
    assert_eq!(align_of::<glam::DVec2>(), align_of::<Vector2::<f64>>());
    assert_eq!(align_of::<glam::DVec3>(), align_of::<Vector3::<f64>>());
    assert!(align_of::<glam::DVec4>() <= align_of::<Vector4::<f64>>());
    assert_eq!(align_of::<glam::IVec2>(), align_of::<Vector2::<i32>>());
    assert_eq!(align_of::<glam::IVec3>(), align_of::<Vector3::<i32>>());
    assert!(align_of::<glam::IVec4>() <= align_of::<Vector4::<i32>>());
    assert_eq!(align_of::<glam::UVec2>(), align_of::<Vector2::<u32>>());
    assert_eq!(align_of::<glam::UVec3>(), align_of::<Vector3::<u32>>());
    assert!(align_of::<glam::UVec4>() <= align_of::<Vector4::<u32>>());

    assert_eq!(align_of::<glam::Vec2>(), align_of::<Point2::<f32>>());
    assert_eq!(align_of::<glam::Vec3>(), align_of::<Point3::<f32>>());
    assert_eq!(align_of::<glam::Vec4>(), align_of::<Point4::<f32>>());
    assert_eq!(align_of::<glam::DVec2>(), align_of::<Point2::<f64>>());
    assert_eq!(align_of::<glam::DVec3>(), align_of::<Point3::<f64>>());
    assert!(align_of::<glam::DVec4>() <= align_of::<Point4::<f64>>());
    assert_eq!(align_of::<glam::IVec2>(), align_of::<Point2::<i32>>());
    assert_eq!(align_of::<glam::IVec3>(), align_of::<Point3::<i32>>());
    assert!(align_of::<glam::IVec4>() <= align_of::<Point4::<i32>>());
    assert_eq!(align_of::<glam::UVec2>(), align_of::<Point2::<u32>>());
    assert_eq!(align_of::<glam::UVec3>(), align_of::<Point3::<u32>>());
    assert!(align_of::<glam::UVec4>() <= align_of::<Point4::<u32>>());

    assert_eq!(align_of::<glam::Vec2>(), align_of::<Size2::<f32>>());
    assert_eq!(align_of::<glam::Vec3>(), align_of::<Size3::<f32>>());
    assert_eq!(align_of::<glam::DVec2>(), align_of::<Size2::<f64>>());
    assert_eq!(align_of::<glam::DVec3>(), align_of::<Size3::<f64>>());
    assert_eq!(align_of::<glam::IVec2>(), align_of::<Size2::<i32>>());
    assert_eq!(align_of::<glam::IVec3>(), align_of::<Size3::<i32>>());
    assert_eq!(align_of::<glam::UVec2>(), align_of::<Size2::<u32>>());
    assert_eq!(align_of::<glam::UVec3>(), align_of::<Size3::<u32>>());

    assert_eq!(align_of::<glam::Mat2>(), align_of::<Matrix2<f32>>());
    assert_eq!(align_of::<glam::Mat3>(), align_of::<Matrix3<f32>>());
    assert_eq!(align_of::<glam::Mat4>(), align_of::<Matrix4<f32>>());
    assert!(align_of::<glam::DMat2>() <= align_of::<Matrix2<f64>>());
    assert_eq!(align_of::<glam::DMat3>(), align_of::<Matrix3<f64>>());
    assert!(align_of::<glam::DMat4>() <= align_of::<Matrix4<f64>>());

    assert_eq!(align_of::<glam::Mat3>(), align_of::<Transform2<f32, f32>>());
    assert_eq!(align_of::<glam::Mat4>(), align_of::<Transform3<f32, f32>>());
    assert_eq!(
        align_of::<glam::DMat3>(),
        align_of::<Transform2<f64, f64>>()
    );
    assert!(align_of::<glam::DMat4>() <= align_of::<Transform3<f64, f64>>());
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
fn cast_to_glam_by_reference() {
    let mut vec2 = Vector2::<f32>::new(1.0, 2.0);
    let mut vec3 = Vector3::<f32>::new(1.0, 2.0, 3.0);
    let mut vec4 = Vector4::<f32>::new(1.0, 2.0, 3.0, 4.0);
    let mut dvec2 = Vector2::<f64>::new(1.0, 2.0);
    let mut dvec3 = Vector3::<f64>::new(1.0, 2.0, 3.0);
    let mut dvec4 = Vector4::<f64>::new(1.0, 2.0, 3.0, 4.0);
    let mut ivec2 = Vector2::<i32>::new(1, 2);
    let mut ivec3 = Vector3::<i32>::new(1, 2, 3);
    let mut ivec4 = Vector4::<i32>::new(1, 2, 3, 4);
    let mut uvec2 = Vector2::<u32>::new(1u32, 2u32);
    let mut uvec3 = Vector3::<u32>::new(1u32, 2u32, 3u32);
    let mut uvec4 = Vector4::<u32>::new(1u32, 2u32, 3u32, 4u32);

    let mut point2 = Point2::<f32>::new(1.0, 2.0);
    let mut point3 = Point3::<f32>::new(1.0, 2.0, 3.0);
    let mut point4 = Point4::<f32>::new(1.0, 2.0, 3.0, 4.0);
    let mut dpoint2 = Point2::<f64>::new(1.0, 2.0);
    let mut dpoint3 = Point3::<f64>::new(1.0, 2.0, 3.0);
    let mut dpoint4 = Point4::<f64>::new(1.0, 2.0, 3.0, 4.0);
    let mut ipoint2 = Point2::<i32>::new(1, 2);
    let mut ipoint3 = Point3::<i32>::new(1, 2, 3);
    let mut ipoint4 = Point4::<i32>::new(1, 2, 3, 4);
    let mut upoint2 = Point2::<u32>::new(1u32, 2u32);
    let mut upoint3 = Point3::<u32>::new(1u32, 2u32, 3u32);
    let mut upoint4 = Point4::<u32>::new(1u32, 2u32, 3u32, 4u32);

    let mut size2 = Size2::<f32>::new(1.0, 2.0);
    let mut size3 = Size3::<f32>::new(1.0, 2.0, 3.0);
    let mut dsize2 = Size2::<f64>::new(1.0, 2.0);
    let mut dsize3 = Size3::<f64>::new(1.0, 2.0, 3.0);
    let mut isize2 = Size2::<i32>::new(1, 2);
    let mut isize3 = Size3::<i32>::new(1, 2, 3);
    let mut usize2 = Size2::<u32>::new(1u32, 2u32);
    let mut usize3 = Size3::<u32>::new(1u32, 2u32, 3u32);

    let mut mat2 = Matrix2::<f32>::IDENTITY;
    let mut mat3 = Matrix3::<f32>::IDENTITY;
    let mut mat4 = Matrix4::<f32>::IDENTITY;
    let mut dmat2 = Matrix2::<f64>::IDENTITY;
    let mut dmat3 = Matrix3::<f64>::IDENTITY;
    let mut dmat4 = Matrix4::<f64>::IDENTITY;

    let vec2_raw: &glam::Vec2 = vec2.as_raw();
    let vec3_raw: &glam::Vec3 = vec3.as_raw();
    let vec4_raw: &glam::Vec4 = vec4.as_raw();
    let dvec2_raw: &glam::DVec2 = dvec2.as_raw();
    let dvec3_raw: &glam::DVec3 = dvec3.as_raw();
    let dvec4_raw: &glam::DVec4 = dvec4.as_raw();
    let ivec2_raw: &glam::IVec2 = ivec2.as_raw();
    let ivec3_raw: &glam::IVec3 = ivec3.as_raw();
    let ivec4_raw: &glam::IVec4 = ivec4.as_raw();
    let uvec2_raw: &glam::UVec2 = uvec2.as_raw();
    let uvec3_raw: &glam::UVec3 = uvec3.as_raw();
    let uvec4_raw: &glam::UVec4 = uvec4.as_raw();

    let point2_raw: &glam::Vec2 = point2.as_raw();
    let point3_raw: &glam::Vec3 = point3.as_raw();
    let point4_raw: &glam::Vec4 = point4.as_raw();
    let dpoint2_raw: &glam::DVec2 = dpoint2.as_raw();
    let dpoint3_raw: &glam::DVec3 = dpoint3.as_raw();
    let dpoint4_raw: &glam::DVec4 = dpoint4.as_raw();
    let ipoint2_raw: &glam::IVec2 = ipoint2.as_raw();
    let ipoint3_raw: &glam::IVec3 = ipoint3.as_raw();
    let ipoint4_raw: &glam::IVec4 = ipoint4.as_raw();
    let upoint2_raw: &glam::UVec2 = upoint2.as_raw();
    let upoint3_raw: &glam::UVec3 = upoint3.as_raw();
    let upoint4_raw: &glam::UVec4 = upoint4.as_raw();

    let size2_raw: &glam::Vec2 = size2.as_raw();
    let size3_raw: &glam::Vec3 = size3.as_raw();
    let dsize2_raw: &glam::DVec2 = dsize2.as_raw();
    let dsize3_raw: &glam::DVec3 = dsize3.as_raw();
    let isize2_raw: &glam::IVec2 = isize2.as_raw();
    let isize3_raw: &glam::IVec3 = isize3.as_raw();
    let usize2_raw: &glam::UVec2 = usize2.as_raw();
    let usize3_raw: &glam::UVec3 = usize3.as_raw();

    let mat2_raw: &glam::Mat2 = mat2.as_raw();
    let mat3_raw: &glam::Mat3 = mat3.as_raw();
    let mat4_raw: &glam::Mat4 = mat4.as_raw();
    let dmat2_raw: &glam::DMat2 = dmat2.as_raw();
    let dmat3_raw: &glam::DMat3 = dmat3.as_raw();
    let dmat4_raw: &glam::DMat4 = dmat4.as_raw();

    assert_eq!(vec2_raw.x, vec2.x);
    assert_eq!(vec2_raw.y, vec2.y);
    assert_eq!(vec3_raw.x, vec3.x);
    assert_eq!(vec3_raw.y, vec3.y);
    assert_eq!(vec3_raw.z, vec3.z);
    assert_eq!(vec4_raw.x, vec4.x);
    assert_eq!(vec4_raw.y, vec4.y);
    assert_eq!(vec4_raw.z, vec4.z);
    assert_eq!(vec4_raw.w, vec4.w);
    assert_eq!(dvec2_raw.x, dvec2.x);
    assert_eq!(dvec2_raw.y, dvec2.y);
    assert_eq!(dvec3_raw.z, dvec3.z);
    assert_eq!(dvec3_raw.x, dvec3.x);
    assert_eq!(dvec3_raw.y, dvec3.y);
    assert_eq!(dvec4_raw.x, dvec4.x);
    assert_eq!(dvec4_raw.y, dvec4.y);
    assert_eq!(dvec4_raw.z, dvec4.z);
    assert_eq!(dvec4_raw.w, dvec4.w);
    assert_eq!(ivec2_raw.x, ivec2.x);
    assert_eq!(ivec2_raw.y, ivec2.y);
    assert_eq!(ivec3_raw.x, ivec3.x);
    assert_eq!(ivec3_raw.y, ivec3.y);
    assert_eq!(ivec3_raw.z, ivec3.z);
    assert_eq!(ivec4_raw.x, ivec4.x);
    assert_eq!(ivec4_raw.y, ivec4.y);
    assert_eq!(ivec4_raw.z, ivec4.z);
    assert_eq!(ivec4_raw.w, ivec4.w);
    assert_eq!(uvec2_raw.x, uvec2.x);
    assert_eq!(uvec2_raw.y, uvec2.y);
    assert_eq!(uvec3_raw.x, uvec3.x);
    assert_eq!(uvec3_raw.y, uvec3.y);
    assert_eq!(uvec3_raw.z, uvec3.z);
    assert_eq!(uvec4_raw.x, uvec4.x);
    assert_eq!(uvec4_raw.y, uvec4.y);
    assert_eq!(uvec4_raw.z, uvec4.z);
    assert_eq!(uvec4_raw.w, uvec4.w);
    assert_eq!(point2_raw.x, point2.x);
    assert_eq!(point2_raw.y, point2.y);
    assert_eq!(point3_raw.x, point3.x);
    assert_eq!(point3_raw.y, point3.y);
    assert_eq!(point3_raw.z, point3.z);
    assert_eq!(point4_raw.x, point4.x);
    assert_eq!(point4_raw.y, point4.y);
    assert_eq!(point4_raw.z, point4.z);
    assert_eq!(point4_raw.w, point4.w);
    assert_eq!(dpoint2_raw.x, dpoint2.x);
    assert_eq!(dpoint2_raw.y, dpoint2.y);
    assert_eq!(dpoint3_raw.z, dpoint3.z);
    assert_eq!(dpoint3_raw.x, dpoint3.x);
    assert_eq!(dpoint3_raw.y, dpoint3.y);
    assert_eq!(dpoint4_raw.x, dpoint4.x);
    assert_eq!(dpoint4_raw.y, dpoint4.y);
    assert_eq!(dpoint4_raw.z, dpoint4.z);
    assert_eq!(dpoint4_raw.w, dpoint4.w);
    assert_eq!(ipoint2_raw.x, ipoint2.x);
    assert_eq!(ipoint2_raw.y, ipoint2.y);
    assert_eq!(ipoint3_raw.x, ipoint3.x);
    assert_eq!(ipoint3_raw.y, ipoint3.y);
    assert_eq!(ipoint3_raw.z, ipoint3.z);
    assert_eq!(ipoint4_raw.x, ipoint4.x);
    assert_eq!(ipoint4_raw.y, ipoint4.y);
    assert_eq!(ipoint4_raw.z, ipoint4.z);
    assert_eq!(ipoint4_raw.w, ipoint4.w);
    assert_eq!(upoint2_raw.x, upoint2.x);
    assert_eq!(upoint2_raw.y, upoint2.y);
    assert_eq!(upoint3_raw.x, upoint3.x);
    assert_eq!(upoint3_raw.y, upoint3.y);
    assert_eq!(upoint3_raw.z, upoint3.z);
    assert_eq!(upoint4_raw.x, upoint4.x);
    assert_eq!(upoint4_raw.y, upoint4.y);
    assert_eq!(upoint4_raw.z, upoint4.z);
    assert_eq!(upoint4_raw.w, upoint4.w);

    assert_eq!(size2_raw.x, size2.width);
    assert_eq!(size2_raw.y, size2.height);
    assert_eq!(size3_raw.x, size3.width);
    assert_eq!(size3_raw.y, size3.height);
    assert_eq!(size3_raw.z, size3.depth);
    assert_eq!(dsize2_raw.x, dsize2.width);
    assert_eq!(dsize2_raw.y, dsize2.height);
    assert_eq!(dsize3_raw.z, dsize3.depth);
    assert_eq!(dsize3_raw.x, dsize3.width);
    assert_eq!(dsize3_raw.y, dsize3.height);
    assert_eq!(isize2_raw.x, isize2.width);
    assert_eq!(isize2_raw.y, isize2.height);
    assert_eq!(isize3_raw.x, isize3.width);
    assert_eq!(isize3_raw.y, isize3.height);
    assert_eq!(isize3_raw.z, isize3.depth);
    assert_eq!(usize2_raw.x, usize2.width);
    assert_eq!(usize2_raw.y, usize2.height);
    assert_eq!(usize3_raw.x, usize3.width);
    assert_eq!(usize3_raw.y, usize3.height);
    assert_eq!(usize3_raw.z, usize3.depth);

    assert_eq!(*mat2_raw, glam::Mat2::IDENTITY);
    assert_eq!(*mat3_raw, glam::Mat3::IDENTITY);
    assert_eq!(*mat4_raw, glam::Mat4::IDENTITY);
    assert_eq!(*dmat2_raw, glam::DMat2::IDENTITY);
    assert_eq!(*dmat3_raw, glam::DMat3::IDENTITY);
    assert_eq!(*dmat4_raw, glam::DMat4::IDENTITY);

    let _vec2_raw: &glam::Vec2 = vec2.as_raw();
    let _vec3_raw: &glam::Vec3 = vec3.as_raw();
    let _vec4_raw: &glam::Vec4 = vec4.as_raw();
    let _dvec2_raw: &glam::DVec2 = dvec2.as_raw();
    let _dvec3_raw: &glam::DVec3 = dvec3.as_raw();
    let _dvec4_raw: &glam::DVec4 = dvec4.as_raw();
    let _ivec2_raw: &glam::IVec2 = ivec2.as_raw();
    let _ivec3_raw: &glam::IVec3 = ivec3.as_raw();
    let _ivec4_raw: &glam::IVec4 = ivec4.as_raw();
    let _uvec2_raw: &glam::UVec2 = uvec2.as_raw();
    let _uvec3_raw: &glam::UVec3 = uvec3.as_raw();
    let _uvec4_raw: &glam::UVec4 = uvec4.as_raw();

    let _point2_raw: &glam::Vec2 = point2.as_raw();
    let _point3_raw: &glam::Vec3 = point3.as_raw();
    let _point4_raw: &glam::Vec4 = point4.as_raw();
    let _dpoint2_raw: &glam::DVec2 = dpoint2.as_raw();
    let _dpoint3_raw: &glam::DVec3 = dpoint3.as_raw();
    let _dpoint4_raw: &glam::DVec4 = dpoint4.as_raw();
    let _ipoint2_raw: &glam::IVec2 = ipoint2.as_raw();
    let _ipoint3_raw: &glam::IVec3 = ipoint3.as_raw();
    let _ipoint4_raw: &glam::IVec4 = ipoint4.as_raw();
    let _upoint2_raw: &glam::UVec2 = upoint2.as_raw();
    let _upoint3_raw: &glam::UVec3 = upoint3.as_raw();
    let _upoint4_raw: &glam::UVec4 = upoint4.as_raw();

    let _size2_raw: &glam::Vec2 = size2.as_raw();
    let _size3_raw: &glam::Vec3 = size3.as_raw();
    let _dsize2_raw: &glam::DVec2 = dsize2.as_raw();
    let _dsize3_raw: &glam::DVec3 = dsize3.as_raw();
    let _isize2_raw: &glam::IVec2 = isize2.as_raw();
    let _isize3_raw: &glam::IVec3 = isize3.as_raw();
    let _usize2_raw: &glam::UVec2 = usize2.as_raw();
    let _usize3_raw: &glam::UVec3 = usize3.as_raw();

    let _vec2_raw: &mut glam::Vec2 = vec2.as_raw_mut();
    let _vec3_raw: &mut glam::Vec3 = vec3.as_raw_mut();
    let _vec4_raw: &mut glam::Vec4 = vec4.as_raw_mut();
    let _dvec2_raw: &mut glam::DVec2 = dvec2.as_raw_mut();
    let _dvec3_raw: &mut glam::DVec3 = dvec3.as_raw_mut();
    let _dvec4_raw: &mut glam::DVec4 = dvec4.as_raw_mut();
    let _ivec2_raw: &mut glam::IVec2 = ivec2.as_raw_mut();
    let _ivec3_raw: &mut glam::IVec3 = ivec3.as_raw_mut();
    let _ivec4_raw: &mut glam::IVec4 = ivec4.as_raw_mut();
    let _uvec2_raw: &mut glam::UVec2 = uvec2.as_raw_mut();
    let _uvec3_raw: &mut glam::UVec3 = uvec3.as_raw_mut();
    let _uvec4_raw: &mut glam::UVec4 = uvec4.as_raw_mut();

    let _point2_raw: &mut glam::Vec2 = point2.as_raw_mut();
    let _point3_raw: &mut glam::Vec3 = point3.as_raw_mut();
    let _point4_raw: &mut glam::Vec4 = point4.as_raw_mut();
    let _dpoint2_raw: &mut glam::DVec2 = dpoint2.as_raw_mut();
    let _dpoint3_raw: &mut glam::DVec3 = dpoint3.as_raw_mut();
    let _dpoint4_raw: &mut glam::DVec4 = dpoint4.as_raw_mut();
    let _ipoint2_raw: &mut glam::IVec2 = ipoint2.as_raw_mut();
    let _ipoint3_raw: &mut glam::IVec3 = ipoint3.as_raw_mut();
    let _ipoint4_raw: &mut glam::IVec4 = ipoint4.as_raw_mut();
    let _upoint2_raw: &mut glam::UVec2 = upoint2.as_raw_mut();
    let _upoint3_raw: &mut glam::UVec3 = upoint3.as_raw_mut();
    let _upoint4_raw: &mut glam::UVec4 = upoint4.as_raw_mut();

    let _size2_raw: &mut glam::Vec2 = size2.as_raw_mut();
    let _size3_raw: &mut glam::Vec3 = size3.as_raw_mut();
    let _dsize2_raw: &mut glam::DVec2 = dsize2.as_raw_mut();
    let _dsize3_raw: &mut glam::DVec3 = dsize3.as_raw_mut();
    let _isize2_raw: &mut glam::IVec2 = isize2.as_raw_mut();
    let _isize3_raw: &mut glam::IVec3 = isize3.as_raw_mut();
    let _usize2_raw: &mut glam::UVec2 = usize2.as_raw_mut();
    let _usize3_raw: &mut glam::UVec3 = usize3.as_raw_mut();

    let _mat2_raw: &mut glam::Mat2 = mat2.as_raw_mut();
    let _mat3_raw: &mut glam::Mat3 = mat3.as_raw_mut();
    let _mat4_raw: &mut glam::Mat4 = mat4.as_raw_mut();
    let _dmat2_raw: &mut glam::DMat2 = dmat2.as_raw_mut();
    let _dmat3_raw: &mut glam::DMat3 = dmat3.as_raw_mut();
    let _dmat4_raw: &mut glam::DMat4 = dmat4.as_raw_mut();
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

    assert_eq!(Into::<Vector4<f32>>::into(f), (1.0, 2.0, 3.0, 4.0));
    assert_eq!(Into::<Vector4<f64>>::into(d), (1.0, 2.0, 3.0, 4.0));
    assert_eq!(Into::<Vector4<i32>>::into(i), (1, 2, 3, 4));
    assert_eq!(Into::<Vector4<u32>>::into(u), (1, 2, 3, 4));
}

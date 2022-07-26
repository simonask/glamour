use crate::{
    Matrix2, Matrix3, Matrix4, Point2, Point3, Point4, Size2, Size3, Unit, Vector2, Vector3,
    Vector4,
};

macro_rules! impl_conv {
    ($glamour_ty:ident { $($glamour_field:ident: $glam_field:ident),* }, $glam_ty:ty, $scalar:ty) => {
        impl<U: Unit<Scalar = $scalar>> From<$glam_ty> for $glamour_ty<U> {
            #[inline]
            fn from(v: $glam_ty) -> Self {
                $glamour_ty {
                    $($glamour_field: v.$glam_field),*
                }
            }
        }

        impl<U: Unit<Scalar = $scalar>> From<$glamour_ty<U>> for $glam_ty {
            #[inline]
            fn from(v: $glamour_ty<U>) -> Self {
                <$glam_ty>::new($(v.$glamour_field),*)
            }
        }
    };
}

impl_conv!(Vector2 { x: x, y: y }, glam_0_20::Vec2, f32);
impl_conv!(Point2 { x: x, y: y }, glam_0_20::Vec2, f32);
impl_conv!(
    Size2 {
        width: x,
        height: y
    },
    glam_0_20::Vec2,
    f32
);
impl_conv!(Vector2 { x: x, y: y }, glam_0_20::DVec2, f64);
impl_conv!(Point2 { x: x, y: y }, glam_0_20::DVec2, f64);
impl_conv!(
    Size2 {
        width: x,
        height: y
    },
    glam_0_20::DVec2,
    f64
);
impl_conv!(Vector2 { x: x, y: y }, glam_0_20::IVec2, i32);
impl_conv!(Point2 { x: x, y: y }, glam_0_20::IVec2, i32);
impl_conv!(
    Size2 {
        width: x,
        height: y
    },
    glam_0_20::IVec2,
    i32
);
impl_conv!(Vector2 { x: x, y: y }, glam_0_20::UVec2, u32);
impl_conv!(Point2 { x: x, y: y }, glam_0_20::UVec2, u32);
impl_conv!(
    Size2 {
        width: x,
        height: y
    },
    glam_0_20::UVec2,
    u32
);

impl_conv!(Vector3 { x: x, y: y, z: z }, glam_0_20::Vec3, f32);
impl_conv!(Vector3 { x: x, y: y, z: z }, glam_0_20::Vec3A, f32);
impl_conv!(Point3 { x: x, y: y, z: z }, glam_0_20::Vec3, f32);
impl_conv!(Point3 { x: x, y: y, z: z }, glam_0_20::Vec3A, f32);
impl_conv!(
    Size3 {
        width: x,
        height: y,
        depth: z
    },
    glam_0_20::Vec3,
    f32
);
impl_conv!(
    Size3 {
        width: x,
        height: y,
        depth: z
    },
    glam_0_20::Vec3A,
    f32
);
impl_conv!(Vector3 { x: x, y: y, z: z }, glam_0_20::DVec3, f64);
impl_conv!(Point3 { x: x, y: y, z: z }, glam_0_20::DVec3, f64);
impl_conv!(
    Size3 {
        width: x,
        height: y,
        depth: z
    },
    glam_0_20::DVec3,
    f64
);
impl_conv!(Vector3 { x: x, y: y, z: z }, glam_0_20::IVec3, i32);
impl_conv!(Point3 { x: x, y: y, z: z }, glam_0_20::IVec3, i32);
impl_conv!(
    Size3 {
        width: x,
        height: y,
        depth: z
    },
    glam_0_20::IVec3,
    i32
);
impl_conv!(Vector3 { x: x, y: y, z: z }, glam_0_20::UVec3, u32);
impl_conv!(Point3 { x: x, y: y, z: z }, glam_0_20::UVec3, u32);
impl_conv!(
    Size3 {
        width: x,
        height: y,
        depth: z
    },
    glam_0_20::UVec3,
    u32
);

impl_conv!(
    Vector4 {
        x: x,
        y: y,
        z: z,
        w: w
    },
    glam_0_20::Vec4,
    f32
);
impl_conv!(
    Point4 {
        x: x,
        y: y,
        z: z,
        w: w
    },
    glam_0_20::Vec4,
    f32
);
impl_conv!(
    Vector4 {
        x: x,
        y: y,
        z: z,
        w: w
    },
    glam_0_20::DVec4,
    f64
);
impl_conv!(
    Point4 {
        x: x,
        y: y,
        z: z,
        w: w
    },
    glam_0_20::DVec4,
    f64
);
impl_conv!(
    Vector4 {
        x: x,
        y: y,
        z: z,
        w: w
    },
    glam_0_20::IVec4,
    i32
);
impl_conv!(
    Point4 {
        x: x,
        y: y,
        z: z,
        w: w
    },
    glam_0_20::IVec4,
    i32
);
impl_conv!(
    Vector4 {
        x: x,
        y: y,
        z: z,
        w: w
    },
    glam_0_20::UVec4,
    u32
);
impl_conv!(
    Point4 {
        x: x,
        y: y,
        z: z,
        w: w
    },
    glam_0_20::UVec4,
    u32
);

macro_rules! impl_conv_mat {
    ($glamour_ty:ident { $($col:ident),* }, $glam_ty:ty, $scalar:ty) => {
        impl From<$glamour_ty<$scalar>> for $glam_ty {
            fn from(m: $glamour_ty<$scalar>) -> Self {
                let [$($col),*] = m.to_cols();
                <$glam_ty>::from_cols($($col.into()),*)
            }
        }

        impl From<$glam_ty> for $glamour_ty<$scalar> {
            fn from(m: $glam_ty) -> Self {
                let [$($col),*] = m.to_cols_array_2d();
                <$glamour_ty<$scalar>>::from_cols($($col.into()),*)
            }
        }
    };
}

impl_conv_mat!(Matrix2 { x_axis, y_axis }, glam_0_20::Mat2, f32);
impl_conv_mat!(Matrix2 { x_axis, y_axis }, glam_0_20::DMat2, f64);
impl_conv_mat!(
    Matrix3 {
        x_axis,
        y_axis,
        z_axis
    },
    glam_0_20::Mat3,
    f32
);
impl_conv_mat!(
    Matrix3 {
        x_axis,
        y_axis,
        z_axis
    },
    glam_0_20::Mat3A,
    f32
);
impl_conv_mat!(
    Matrix3 {
        x_axis,
        y_axis,
        z_axis
    },
    glam_0_20::DMat3,
    f64
);
impl_conv_mat!(
    Matrix4 {
        x_axis,
        y_axis,
        z_axis,
        w_axis
    },
    glam_0_20::Mat4,
    f32
);
impl_conv_mat!(
    Matrix4 {
        x_axis,
        y_axis,
        z_axis,
        w_axis
    },
    glam_0_20::DMat4,
    f64
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert() {
        type V2f = Vector2<f32>;
        type V3f = Vector3<f32>;
        type V4f = Vector4<f32>;
        type V2d = Vector2<f64>;
        type V3d = Vector3<f64>;
        type V4d = Vector4<f64>;
        type V2i = Vector2<i32>;
        type V3i = Vector3<i32>;
        type V4i = Vector4<i32>;
        type V2u = Vector2<u32>;
        type V3u = Vector3<u32>;
        type V4u = Vector4<u32>;

        assert_eq!(
            V2f::from(glam_0_20::Vec2::from(V2f::new(1.0, 2.0))),
            V2f::new(1.0, 2.0)
        );
        assert_eq!(
            V3f::from(glam_0_20::Vec3::from(V3f::new(1.0, 2.0, 3.0))),
            V3f::new(1.0, 2.0, 3.0)
        );
        assert_eq!(
            V3f::from(glam_0_20::Vec3A::from(V3f::new(1.0, 2.0, 3.0))),
            V3f::new(1.0, 2.0, 3.0)
        );
        assert_eq!(
            V4f::from(glam_0_20::Vec4::from(V4f::new(1.0, 2.0, 3.0, 4.0))),
            V4f::new(1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(
            V2d::from(glam_0_20::DVec2::from(V2d::new(1.0, 2.0))),
            V2d::new(1.0, 2.0)
        );
        assert_eq!(
            V3d::from(glam_0_20::DVec3::from(V3d::new(1.0, 2.0, 3.0))),
            V3d::new(1.0, 2.0, 3.0)
        );
        assert_eq!(
            V4d::from(glam_0_20::DVec4::from(V4d::new(1.0, 2.0, 3.0, 4.0))),
            V4d::new(1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(
            V2i::from(glam_0_20::IVec2::from(V2i::new(1, 2))),
            V2i::new(1, 2)
        );
        assert_eq!(
            V3i::from(glam_0_20::IVec3::from(V3i::new(1, 2, 3))),
            V3i::new(1, 2, 3)
        );
        assert_eq!(
            V4i::from(glam_0_20::IVec4::from(V4i::new(1, 2, 3, 4))),
            V4i::new(1, 2, 3, 4)
        );
        assert_eq!(
            V2u::from(glam_0_20::UVec2::from(V2u::new(1u32, 2u32))),
            V2u::new(1u32, 2u32)
        );
        assert_eq!(
            V3u::from(glam_0_20::UVec3::from(V3u::new(1u32, 2u32, 3u32))),
            V3u::new(1u32, 2u32, 3u32)
        );
        assert_eq!(
            V4u::from(glam_0_20::UVec4::from(V4u::new(1u32, 2u32, 3u32, 4u32))),
            V4u::new(1u32, 2u32, 3u32, 4u32)
        );
    }

    #[test]
    fn from_into() {
        assert_eq!(
            Matrix2::from(glam_0_20::Mat2::from(Matrix2::IDENTITY)),
            Matrix2::IDENTITY
        );
        assert_eq!(
            Matrix2::from(glam_0_20::DMat2::from(Matrix2::IDENTITY)),
            Matrix2::IDENTITY
        );
        assert_eq!(
            Matrix3::from(glam_0_20::Mat3::from(Matrix3::IDENTITY)),
            Matrix3::IDENTITY
        );
        assert_eq!(
            Matrix3::from(glam_0_20::Mat3A::from(Matrix3::IDENTITY)),
            Matrix3::IDENTITY
        );
        assert_eq!(
            Matrix3::from(glam_0_20::DMat3::from(Matrix3::IDENTITY)),
            Matrix3::IDENTITY
        );
        assert_eq!(
            Matrix4::from(glam_0_20::Mat4::from(Matrix4::IDENTITY)),
            Matrix4::IDENTITY
        );
        assert_eq!(
            Matrix4::from(glam_0_20::DMat4::from(Matrix4::IDENTITY)),
            Matrix4::IDENTITY
        );
    }
}

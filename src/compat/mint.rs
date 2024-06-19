use crate::{
    Matrix2, Matrix3, Matrix4, Point2, Point3, Size2, Size3, Unit, Vector2, Vector3, Vector4,
};

macro_rules! impl_conv {
    ($glamour_ty:ident { $($glamour_field:ident: $mint_field:ident),* }, $mint_ty:ident < $scalar:ty >) => {
        impl<U: Unit<Scalar = $scalar>> From<mint::$mint_ty<$scalar>> for $glamour_ty<U> {
            #[inline]
            fn from(v: mint::$mint_ty<$scalar>) -> Self {
                $glamour_ty {
                    $($glamour_field: v.$mint_field),*
                }
            }
        }

        impl<U: Unit<Scalar = $scalar>> From<$glamour_ty<U>> for mint::$mint_ty<$scalar> {
            #[inline]
            fn from(v: $glamour_ty<U>) -> Self {
                let $glamour_ty { $($glamour_field: $mint_field),* } = v;
                mint::$mint_ty { $($mint_field),* }
            }
        }

        impl mint::IntoMint for $glamour_ty<$scalar> {
            type MintType = mint::$mint_ty<$scalar>;
        }
    };
}

impl_conv!(Vector2 { x: x, y: y }, Vector2<f32>);
impl_conv!(Point2 { x: x, y: y }, Point2<f32>);
impl_conv!(
    Size2 {
        width: x,
        height: y
    },
    Vector2<f32>
);
impl_conv!(Vector2 { x: x, y: y }, Vector2<f64>);
impl_conv!(Point2 { x: x, y: y }, Point2<f64>);
impl_conv!(
    Size2 {
        width: x,
        height: y
    },
    Vector2<f64>
);
impl_conv!(Vector2 { x: x, y: y }, Vector2<i32>);
impl_conv!(Point2 { x: x, y: y }, Point2<i32>);
impl_conv!(
    Size2 {
        width: x,
        height: y
    },
    Vector2<i32>
);
impl_conv!(Vector2 { x: x, y: y }, Vector2<u32>);
impl_conv!(Point2 { x: x, y: y }, Point2<u32>);
impl_conv!(
    Size2 {
        width: x,
        height: y
    },
    Vector2<u32>
);

impl_conv!(Vector3 { x: x, y: y, z: z }, Vector3<f32>);
impl_conv!(Point3 { x: x, y: y, z: z }, Point3<f32>);
impl_conv!(
    Size3 {
        width: x,
        height: y,
        depth: z
    },
    Vector3<f32>
);
impl_conv!(Vector3 { x: x, y: y, z: z }, Vector3<f64>);
impl_conv!(Point3 { x: x, y: y, z: z }, Point3<f64>);
impl_conv!(
    Size3 {
        width: x,
        height: y,
        depth: z
    },
    Vector3<f64>
);
impl_conv!(Vector3 { x: x, y: y, z: z }, Vector3<i32>);
impl_conv!(Point3 { x: x, y: y, z: z }, Point3<i32>);
impl_conv!(
    Size3 {
        width: x,
        height: y,
        depth: z
    },
    Vector3<i32>
);
impl_conv!(Vector3 { x: x, y: y, z: z }, Vector3<u32>);
impl_conv!(Point3 { x: x, y: y, z: z }, Point3<u32>);
impl_conv!(
    Size3 {
        width: x,
        height: y,
        depth: z
    },
    Vector3<u32>
);

impl_conv!(
    Vector4 {
        x: x,
        y: y,
        z: z,
        w: w
    },
    Vector4<f32>
);
impl_conv!(
    Vector4 {
        x: x,
        y: y,
        z: z,
        w: w
    },
    Vector4<f64>
);
impl_conv!(
    Vector4 {
        x: x,
        y: y,
        z: z,
        w: w
    },
    Vector4<i32>
);
impl_conv!(
    Vector4 {
        x: x,
        y: y,
        z: z,
        w: w
    },
    Vector4<u32>
);

macro_rules! impl_conv_mat {
    ($glamour_ty:ident { $($col:ident),* }, $mint_ty:ident < $scalar:ty >) => {
        impl From<$glamour_ty<$scalar>> for mint::$mint_ty<$scalar> {
            fn from(m: $glamour_ty<$scalar>) -> Self {
                let [$($col),*] = m.to_cols_array_2d();
                 mint::$mint_ty { $($col: $col.into()),* }
            }
        }

        impl From<mint::$mint_ty<$scalar>> for $glamour_ty<$scalar> {
            fn from(m: mint::$mint_ty<$scalar>) -> Self {
                let mint::$mint_ty { $($col),* } = m;
                <$glamour_ty<$scalar>>::from_cols($($col.into()),*)
            }
        }
    };
}

impl_conv_mat!(Matrix2 { x, y }, ColumnMatrix2<f32>);
impl_conv_mat!(Matrix2 { x, y }, ColumnMatrix2<f64>);
impl_conv_mat!(Matrix3 { x, y, z }, ColumnMatrix3<f32>);
impl_conv_mat!(Matrix3 { x, y, z }, ColumnMatrix3<f64>);
impl_conv_mat!(Matrix4 { x, y, z, w }, ColumnMatrix4<f32>);
impl_conv_mat!(Matrix4 { x, y, z, w }, ColumnMatrix4<f64>);

#[cfg(test)]
mod tests {
    use super::*;

    type Vec2 = mint::Vector2<f32>;
    type Vec3 = mint::Vector3<f32>;
    type Vec4 = mint::Vector4<f32>;
    type DVec2 = mint::Vector2<f64>;
    type DVec3 = mint::Vector3<f64>;
    type DVec4 = mint::Vector4<f64>;
    type IVec2 = mint::Vector2<i32>;
    type IVec3 = mint::Vector3<i32>;
    type IVec4 = mint::Vector4<i32>;
    type UVec2 = mint::Vector2<u32>;
    type UVec3 = mint::Vector3<u32>;
    type UVec4 = mint::Vector4<u32>;

    type Pos2 = mint::Point2<f32>;
    type Pos3 = mint::Point3<f32>;
    type DPos2 = mint::Point2<f64>;
    type DPos3 = mint::Point3<f64>;
    type IPos2 = mint::Point2<i32>;
    type IPos3 = mint::Point3<i32>;
    type UPos2 = mint::Point2<u32>;
    type UPos3 = mint::Point3<u32>;

    type Mat2 = mint::ColumnMatrix2<f32>;
    type Mat3 = mint::ColumnMatrix3<f32>;
    type Mat4 = mint::ColumnMatrix4<f32>;
    type DMat2 = mint::ColumnMatrix2<f64>;
    type DMat3 = mint::ColumnMatrix3<f64>;
    type DMat4 = mint::ColumnMatrix4<f64>;

    #[test]
    fn convert_vec() {
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
            V2f::from(Vec2::from(V2f::new(1.0, 2.0))),
            V2f::new(1.0, 2.0)
        );
        assert_eq!(
            V3f::from(Vec3::from(V3f::new(1.0, 2.0, 3.0))),
            V3f::new(1.0, 2.0, 3.0)
        );
        assert_eq!(
            V4f::from(Vec4::from(V4f::new(1.0, 2.0, 3.0, 4.0))),
            V4f::new(1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(
            V2d::from(DVec2::from(V2d::new(1.0, 2.0))),
            V2d::new(1.0, 2.0)
        );
        assert_eq!(
            V3d::from(DVec3::from(V3d::new(1.0, 2.0, 3.0))),
            V3d::new(1.0, 2.0, 3.0)
        );
        assert_eq!(
            V4d::from(DVec4::from(V4d::new(1.0, 2.0, 3.0, 4.0))),
            V4d::new(1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(V2i::from(IVec2::from(V2i::new(1, 2))), V2i::new(1, 2));
        assert_eq!(V3i::from(IVec3::from(V3i::new(1, 2, 3))), V3i::new(1, 2, 3));
        assert_eq!(
            V4i::from(IVec4::from(V4i::new(1, 2, 3, 4))),
            V4i::new(1, 2, 3, 4)
        );
        assert_eq!(
            V2u::from(UVec2::from(V2u::new(1u32, 2u32))),
            V2u::new(1u32, 2u32)
        );
        assert_eq!(
            V3u::from(UVec3::from(V3u::new(1u32, 2u32, 3u32))),
            V3u::new(1u32, 2u32, 3u32)
        );
        assert_eq!(
            V4u::from(UVec4::from(V4u::new(1u32, 2u32, 3u32, 4u32))),
            V4u::new(1u32, 2u32, 3u32, 4u32)
        );
    }

    #[test]
    fn convert_point() {
        type V2f = Point2<f32>;
        type V3f = Point3<f32>;
        type V2d = Point2<f64>;
        type V3d = Point3<f64>;
        type V2i = Point2<i32>;
        type V3i = Point3<i32>;
        type V2u = Point2<u32>;
        type V3u = Point3<u32>;

        assert_eq!(
            V2f::from(Pos2::from(V2f::new(1.0, 2.0))),
            V2f::new(1.0, 2.0)
        );
        assert_eq!(
            V3f::from(Pos3::from(V3f::new(1.0, 2.0, 3.0))),
            V3f::new(1.0, 2.0, 3.0)
        );
        assert_eq!(
            V2d::from(DPos2::from(V2d::new(1.0, 2.0))),
            V2d::new(1.0, 2.0)
        );
        assert_eq!(
            V3d::from(DPos3::from(V3d::new(1.0, 2.0, 3.0))),
            V3d::new(1.0, 2.0, 3.0)
        );
        assert_eq!(V2i::from(IPos2::from(V2i::new(1, 2))), V2i::new(1, 2));
        assert_eq!(V3i::from(IPos3::from(V3i::new(1, 2, 3))), V3i::new(1, 2, 3));
        assert_eq!(
            V2u::from(UPos2::from(V2u::new(1u32, 2u32))),
            V2u::new(1u32, 2u32)
        );
        assert_eq!(
            V3u::from(UPos3::from(V3u::new(1u32, 2u32, 3u32))),
            V3u::new(1u32, 2u32, 3u32)
        );
    }

    #[test]
    fn from_into() {
        assert_eq!(
            Matrix2::from(Mat2::from(Matrix2::IDENTITY)),
            Matrix2::IDENTITY
        );
        assert_eq!(
            Matrix2::from(DMat2::from(Matrix2::IDENTITY)),
            Matrix2::IDENTITY
        );
        assert_eq!(
            Matrix3::from(Mat3::from(Matrix3::IDENTITY)),
            Matrix3::IDENTITY
        );
        assert_eq!(
            Matrix3::from(DMat3::from(Matrix3::IDENTITY)),
            Matrix3::IDENTITY
        );
        assert_eq!(
            Matrix4::from(Mat4::from(Matrix4::IDENTITY)),
            Matrix4::IDENTITY
        );
        assert_eq!(
            Matrix4::from(DMat4::from(Matrix4::IDENTITY)),
            Matrix4::IDENTITY
        );
    }
}

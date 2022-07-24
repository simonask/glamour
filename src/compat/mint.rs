use crate::{Point2, Point3, Size2, Size3, Unit, Vector2, Vector3, Vector4, Matrix2, Matrix3, Matrix4};

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
    };
}

impl_conv!(Vector2 { x: x, y: y }, Vector2<f32>);
impl_conv!(Point2 { x: x, y: y }, Point2<f32>);
impl_conv!(Size2 { width: x, height: y }, Vector2<f32>);
impl_conv!(Vector2 { x: x, y: y }, Vector2<f64>);
impl_conv!(Point2 { x: x, y: y }, Point2<f64>);
impl_conv!(Size2 { width: x, height: y }, Vector2<f64>);
impl_conv!(Vector2 { x: x, y: y }, Vector2<i32>);
impl_conv!(Point2 { x: x, y: y }, Point2<i32>);
impl_conv!(Size2 { width: x, height: y }, Vector2<i32>);
impl_conv!(Vector2 { x: x, y: y }, Vector2<u32>);
impl_conv!(Point2 { x: x, y: y }, Point2<u32>);
impl_conv!(Size2 { width: x, height: y }, Vector2<u32>);

impl_conv!(Vector3 { x: x, y: y, z: z }, Vector3<f32>);
impl_conv!(Point3 { x: x, y: y, z: z }, Point3<f32>);
impl_conv!(Size3 { width: x, height: y, depth: z }, Vector3<f32>);
impl_conv!(Vector3 { x: x, y: y, z: z }, Vector3<f64>);
impl_conv!(Point3 { x: x, y: y, z: z }, Point3<f64>);
impl_conv!(Size3 { width: x, height: y, depth: z }, Vector3<f64>);
impl_conv!(Vector3 { x: x, y: y, z: z }, Vector3<i32>);
impl_conv!(Point3 { x: x, y: y, z: z }, Point3<i32>);
impl_conv!(Size3 { width: x, height: y, depth: z }, Vector3<i32>);
impl_conv!(Vector3 { x: x, y: y, z: z }, Vector3<u32>);
impl_conv!(Point3 { x: x, y: y, z: z }, Point3<u32>);
impl_conv!(Size3 { width: x, height: y, depth: z }, Vector3<u32>);

impl_conv!(Vector4 { x: x, y: y, z: z, w: w }, Vector4<f32>);
impl_conv!(Vector4 { x: x, y: y, z: z, w: w }, Vector4<f64>);
impl_conv!(Vector4 { x: x, y: y, z: z, w: w }, Vector4<i32>);
impl_conv!(Vector4 { x: x, y: y, z: z, w: w }, Vector4<u32>);



macro_rules! impl_conv_mat {
    ($glamour_ty:ident { $($col:ident),* }, $mint_ty:ident < $scalar:ty >) => {
        impl From<$glamour_ty<$scalar>> for mint::$mint_ty<$scalar> {
            fn from(m: $glamour_ty<$scalar>) -> Self {
                let [$($col),*] = m.to_cols();
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
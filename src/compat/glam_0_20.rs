use crate::{Point2, Point3, Point4, Size2, Size3, Unit, Vector2, Vector3, Vector4, Matrix2, Matrix3, Matrix4};

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
impl_conv!(Size2 { width: x, height: y }, glam_0_20::Vec2, f32);
impl_conv!(Vector2 { x: x, y: y }, glam_0_20::DVec2, f64);
impl_conv!(Point2 { x: x, y: y }, glam_0_20::DVec2, f64);
impl_conv!(Size2 { width: x, height: y }, glam_0_20::DVec2, f64);
impl_conv!(Vector2 { x: x, y: y }, glam_0_20::IVec2, i32);
impl_conv!(Point2 { x: x, y: y }, glam_0_20::IVec2, i32);
impl_conv!(Size2 { width: x, height: y }, glam_0_20::IVec2, i32);
impl_conv!(Vector2 { x: x, y: y }, glam_0_20::UVec2, u32);
impl_conv!(Point2 { x: x, y: y }, glam_0_20::UVec2, u32);
impl_conv!(Size2 { width: x, height: y }, glam_0_20::UVec2, u32);

impl_conv!(Vector3 { x: x, y: y, z: z }, glam_0_20::Vec3, f32);
impl_conv!(Point3 { x: x, y: y, z: z }, glam_0_20::Vec3, f32);
impl_conv!(Size3 { width: x, height: y, depth: z }, glam_0_20::Vec3, f32);
impl_conv!(Vector3 { x: x, y: y, z: z }, glam_0_20::DVec3, f64);
impl_conv!(Point3 { x: x, y: y, z: z }, glam_0_20::DVec3, f64);
impl_conv!(Size3 { width: x, height: y, depth: z }, glam_0_20::DVec3, f64);
impl_conv!(Vector3 { x: x, y: y, z: z }, glam_0_20::IVec3, i32);
impl_conv!(Point3 { x: x, y: y, z: z }, glam_0_20::IVec3, i32);
impl_conv!(Size3 { width: x, height: y, depth: z }, glam_0_20::IVec3, i32);
impl_conv!(Vector3 { x: x, y: y, z: z }, glam_0_20::UVec3, u32);
impl_conv!(Point3 { x: x, y: y, z: z }, glam_0_20::UVec3, u32);
impl_conv!(Size3 { width: x, height: y, depth: z }, glam_0_20::UVec3, u32);

impl_conv!(Vector4 { x: x, y: y, z: z, w: w }, glam_0_20::Vec4, f32);
impl_conv!(Point4 { x: x, y: y, z: z, w: w }, glam_0_20::Vec4, f32);
impl_conv!(Vector4 { x: x, y: y, z: z, w: w }, glam_0_20::DVec4, f64);
impl_conv!(Point4 { x: x, y: y, z: z, w: w }, glam_0_20::DVec4, f64);
impl_conv!(Vector4 { x: x, y: y, z: z, w: w }, glam_0_20::IVec4, i32);
impl_conv!(Point4 { x: x, y: y, z: z, w: w }, glam_0_20::IVec4, i32);
impl_conv!(Vector4 { x: x, y: y, z: z, w: w }, glam_0_20::UVec4, u32);
impl_conv!(Point4 { x: x, y: y, z: z, w: w }, glam_0_20::UVec4, u32);


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
impl_conv_mat!(Matrix3 { x_axis, y_axis, z_axis }, glam_0_20::Mat3, f32);
impl_conv_mat!(Matrix3 { x_axis, y_axis, z_axis }, glam_0_20::DMat3, f64);
impl_conv_mat!(Matrix4 { x_axis, y_axis, z_axis, w_axis }, glam_0_20::Mat4, f32);
impl_conv_mat!(Matrix4 { x_axis, y_axis, z_axis, w_axis }, glam_0_20::DMat4, f64);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convert() {
        let glam_vec2 = glam::Vec2::new(10.0, 20.0);
        let glamour_vec2 = Vector2::<f32>::from(glam_vec2);
        let glam_0_20_vec2 = glam_0_20::Vec2::from(glamour_vec2);
        let glamour_vec2 = Vector2::<f32>::from(glam_0_20_vec2);
        assert_eq!(glamour_vec2, (10.0, 20.0));
    }
}
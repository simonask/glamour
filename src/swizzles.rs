use crate::{Point2, Point3, Point4, Size2, Size3, Unit, Vector2, Vector3, Vector4};

macro_rules! swizzle {
    (
        $vec2:ident, $vec3:ident, $vec4:ident =>
        $f:ident [
            $x:ident, $y:ident
        ]
    ) => {
        fn $f(self) -> $vec2<T> {
            [self.$x, self.$y].into()
        }
    };
    (
        $vec2:ident, $vec3:ident, $vec4:ident =>
        $f:ident [
            $x:ident, $y:ident, $z:ident
        ]
    ) => {
        fn $f(self) -> $vec3<T> {
            [self.$x, self.$y, self.$z].into()
        }
    };
    (
        $vec2:ident, $vec3:ident, $vec4:ident =>
        $f:ident [
            $x:ident, $y:ident, $z:ident, $w:ident
        ]
    ) => {
        fn $f(self) -> $vec4<T> {
            [self.$x, self.$y, self.$z, self.$w].into()
        }
    };
}

macro_rules! swizzle_with {
    (
        $vec2:ident, $vec3:ident, $vec4:ident =>
        $f:ident [$x_dst:ident = $x_src:ident, $y_dst:ident = $y_src:ident]
    ) => {
        fn $f(mut self, rhs: $vec2<T>) -> Self {
            self.$x_dst = rhs.$x_src;
            self.$y_dst = rhs.$y_src;
            self
        }
    };
    (
        $vec2:ident, $vec3:ident, $vec4:ident =>
        $f:ident [$x_dst:ident = $x_src:ident, $y_dst:ident = $y_src:ident, $z_dst:ident = $z_src:ident]
    ) => {
        fn $f(mut self, rhs: $vec3<T>) -> Self {
            self.$x_dst = rhs.$x_src;
            self.$y_dst = rhs.$y_src;
            self.$z_dst = rhs.$z_src;
            self
        }
    };
}

macro_rules! swizzles {
    ($vec2:ident, $vec3:ident, $vec4:ident => $(
        $f:ident [$($swizzle:ident),*]
    ),*) => {
        $(
            swizzle!($vec2, $vec3, $vec4 => $f [$($swizzle),*]);
        )*
    };
}

macro_rules! swizzles_with {
    ($vec2:ident, $vec3:ident, $vec4:ident => $(
        $f:ident [$($src:ident = $dst:ident),*]
    ),*) => {
        $(
            swizzle_with!($vec2, $vec3, $vec4 => $f [$($src = $dst),*]);
        )*
    };
}

macro_rules! swizzles2 {
    ($vec2:ident, $vec3:ident, $vec4:ident) => {
        swizzles2!($vec2, $vec3, $vec4, x, y);
    };
    ($vec2:ident, $vec3:ident, $vec4:ident, $x:ident, $y:ident) => {
        swizzles! {
            $vec2, $vec3, $vec4 =>
            xx[$x, $x],
            xy[$x, $y],
            yy[$y, $y],
            yx[$y, $x],
            xxx[$x, $x, $x],
            xxy[$x, $x, $y],
            xyx[$x, $y, $x],
            xyy[$x, $y, $y],
            yxx[$y, $x, $x],
            yxy[$y, $x, $y],
            yyx[$y, $y, $x],
            yyy[$y, $y, $y],
            xxxx[$x, $x, $x, $x],
            xxxy[$x, $x, $x, $y],
            xxyx[$x, $x, $y, $x],
            xxyy[$x, $x, $y, $y],
            xyxx[$x, $y, $x, $x],
            xyxy[$x, $y, $x, $y],
            xyyx[$x, $y, $y, $x],
            xyyy[$x, $y, $y, $y],
            yxxx[$y, $x, $x, $x],
            yxxy[$y, $x, $x, $y],
            yxyx[$y, $x, $y, $x],
            yxyy[$y, $x, $y, $y],
            yyxx[$y, $y, $x, $x],
            yyxy[$y, $y, $x, $y],
            yyyx[$y, $y, $y, $x],
            yyyy[$y, $y, $y, $y]
        }
    };
}

macro_rules! swizzles3 {
    ($vec2:ident, $vec3:ident, $vec4:ident) => {
        swizzles3!($vec2, $vec3, $vec4, x, y, z);
    };
    ($vec2:ident, $vec3:ident, $vec4:ident, $x:ident, $y:ident, $z:ident) => {
        swizzles_with!(
            $vec2, $vec3, $vec4 =>
            with_xy[$x = $x, $y = $y],
            with_xz[$x = $x, $z = $y],
            with_yx[$y = $x, $x = $y],
            with_yz[$y = $x, $z = $y],
            with_zx[$z = $x, $x = $y],
            with_zy[$z = $x, $y = $y]
        );

        swizzles! {
            $vec2, $vec3, $vec4 =>
            xz[$x, $z],
            yz[$y, $z],
            zx[$z, $x],
            zy[$z, $y],
            zz[$z, $z],
            xxz[$x, $x, $z],
            xyz[$x, $y, $z],
            xzx[$x, $z, $x],
            xzy[$x, $z, $y],
            xzz[$x, $z, $z],
            yxz[$y, $x, $z],
            yyz[$y, $y, $z],
            yzx[$y, $z, $x],
            yzy[$y, $z, $y],
            yzz[$y, $z, $z],
            zxx[$z, $x, $x],
            zxy[$z, $x, $y],
            zxz[$z, $x, $z],
            zyx[$z, $y, $x],
            zyy[$z, $y, $y],
            zyz[$z, $y, $z],
            zzx[$z, $z, $x],
            zzy[$z, $z, $y],
            zzz[$z, $z, $z],
            xxxz[$x, $x, $x, $z],
            xxyz[$x, $x, $y, $z],
            xxzx[$x, $x, $z, $x],
            xxzy[$x, $x, $z, $y],
            xxzz[$x, $x, $z, $z],
            xyxz[$x, $y, $x, $z],
            xyyz[$x, $y, $y, $z],
            xyzx[$x, $y, $z, $x],
            xyzy[$x, $y, $z, $y],
            xyzz[$x, $y, $z, $z],
            xzxx[$x, $z, $x, $x],
            xzxy[$x, $z, $x, $y],
            xzxz[$x, $z, $x, $z],
            xzyx[$x, $z, $y, $x],
            xzyy[$x, $z, $y, $y],
            xzyz[$x, $z, $y, $z],
            xzzx[$x, $z, $z, $x],
            xzzy[$x, $z, $z, $y],
            xzzz[$x, $z, $z, $z],
            yxxz[$y, $x, $x, $z],
            yxyz[$y, $x, $y, $z],
            yxzx[$y, $x, $z, $x],
            yxzy[$y, $x, $z, $y],
            yxzz[$y, $x, $z, $z],
            yyxz[$y, $y, $x, $z],
            yyyz[$y, $y, $y, $z],
            yyzx[$y, $y, $z, $x],
            yyzy[$y, $y, $z, $y],
            yyzz[$y, $y, $z, $z],
            yzxx[$y, $z, $x, $x],
            yzxy[$y, $z, $x, $y],
            yzxz[$y, $z, $x, $z],
            yzyx[$y, $z, $y, $x],
            yzyy[$y, $z, $y, $y],
            yzyz[$y, $z, $y, $z],
            yzzx[$y, $z, $z, $x],
            yzzy[$y, $z, $z, $y],
            yzzz[$y, $z, $z, $z],
            zxxx[$z, $x, $x, $x],
            zxxy[$z, $x, $x, $y],
            zxxz[$z, $x, $x, $z],
            zxyx[$z, $x, $y, $x],
            zxyy[$z, $x, $y, $y],
            zxyz[$z, $x, $y, $z],
            zxzx[$z, $x, $z, $x],
            zxzy[$z, $x, $z, $y],
            zxzz[$z, $x, $z, $z],
            zyxx[$z, $y, $x, $x],
            zyxy[$z, $y, $x, $y],
            zyxz[$z, $y, $x, $z],
            zyyx[$z, $y, $y, $x],
            zyyy[$z, $y, $y, $y],
            zyyz[$z, $y, $y, $z],
            zyzx[$z, $y, $z, $x],
            zyzy[$z, $y, $z, $y],
            zyzz[$z, $y, $z, $z],
            zzxx[$z, $z, $x, $x],
            zzxy[$z, $z, $x, $y],
            zzxz[$z, $z, $x, $z],
            zzyx[$z, $z, $y, $x],
            zzyy[$z, $z, $y, $y],
            zzyz[$z, $z, $y, $z],
            zzzx[$z, $z, $z, $x],
            zzzy[$z, $z, $z, $y],
            zzzz[$z, $z, $z, $z]
        }
    };
}

macro_rules! swizzles4 {
    ($vec2:ident, $vec3:ident, $vec4:ident) => {
        swizzles_with!(
            $vec2, $vec3, $vec4 =>
            with_wx[w = x, x = y],
            with_wy[w = x, y = y],
            with_wz[w = x, z = y],
            with_xw[x = x, w = y],
            // with_xy[x = x, y = y],
            // with_xz[x = x, z = y],
            with_yw[y = x, w = y],
            // with_yx[y = x, x = y],
            // with_yz[y = x, z = y],
            with_zw[z = x, w = y],
            // with_zx[z = x, x = y],
            // with_zy[z = x, y = y],

            with_wxy[w = x, x = y, y = z],
            with_wxz[w = x, x = y, z = z],
            with_wyx[w = x, y = y, x = z],
            with_wyz[w = x, y = y, z = z],
            with_wzx[w = x, z = y, x = z],
            with_wzy[w = x, z = y, y = z],

            with_xwy[x = x, w = y, y = z],
            with_xwz[x = x, w = y, z = z],
            with_xyw[x = x, y = y, w = z],
            with_xyz[x = x, y = y, z = z],
            with_xzw[x = x, z = y, w = z],
            with_xzy[x = x, z = y, y = z],

            with_ywx[y = x, w = y, x = z],
            with_yxw[y = x, x = y, w = z],
            with_yzx[y = x, z = y, x = z],
            with_yzw[y = x, z = y, w = z],
            with_yxz[y = x, x = y, z = z],
            with_ywz[y = x, w = y, z = z],

            with_zwx[z = x, w = y, x = z],
            with_zxw[z = x, x = y, w = z],
            with_zxy[z = x, x = y, y = z],
            with_zyx[z = x, y = y, x = z],
            with_zwy[z = x, w = y, y = z],
            with_zyw[z = x, y = y, w = z]
        );

        swizzles! {
            $vec2, $vec3, $vec4 =>
            xw[x, w],
            yw[y, w],
            zw[z, w],
            wx[w, x],
            wy[w, y],
            wz[w, z],
            ww[w, w],
            xxw[x, x, w],
            xyw[x, y, w],
            xzw[x, z, w],
            xwx[x, w, x],
            xwy[x, w, y],
            xwz[x, w, z],
            xww[x, w, w],
            yxw[y, x, w],
            yyw[y, y, w],
            yzw[y, z, w],
            ywx[y, w, x],
            ywy[y, w, y],
            ywz[y, w, z],
            yww[y, w, w],
            zxw[z, x, w],
            zyw[z, y, w],
            zzw[z, z, w],
            zwx[z, w, x],
            zwy[z, w, y],
            zwz[z, w, z],
            zww[z, w, w],
            wxx[w, x, x],
            wxy[w, x, y],
            wxz[w, x, z],
            wxw[w, x, w],
            wyx[w, y, x],
            wyy[w, y, y],
            wyz[w, y, z],
            wyw[w, y, w],
            wzx[w, z, x],
            wzy[w, z, y],
            wzz[w, z, z],
            wzw[w, z, w],
            wwx[w, w, x],
            wwy[w, w, y],
            wwz[w, w, z],
            www[w, w, w],
            xxxw[x, x, x, w],
            xxyw[x, x, y, w],
            xxzw[x, x, z, w],
            xxwx[x, x, w, x],
            xxwy[x, x, w, y],
            xxwz[x, x, w, z],
            xxww[x, x, w, w],
            xyxw[x, y, x, w],
            xyyw[x, y, y, w],
            xyzw[x, y, z, w],
            xywx[x, y, w, x],
            xywy[x, y, w, y],
            xywz[x, y, w, z],
            xyww[x, y, w, w],
            xzxw[x, z, x, w],
            xzyw[x, z, y, w],
            xzzw[x, z, z, w],
            xzwx[x, z, w, x],
            xzwy[x, z, w, y],
            xzwz[x, z, w, z],
            xzww[x, z, w, w],
            xwxx[x, w, x, x],
            xwxy[x, w, x, y],
            xwxz[x, w, x, z],
            xwxw[x, w, x, w],
            xwyx[x, w, y, x],
            xwyy[x, w, y, y],
            xwyz[x, w, y, z],
            xwyw[x, w, y, w],
            xwzx[x, w, z, x],
            xwzy[x, w, z, y],
            xwzz[x, w, z, z],
            xwzw[x, w, z, w],
            xwwx[x, w, w, x],
            xwwy[x, w, w, y],
            xwwz[x, w, w, z],
            xwww[x, w, w, w],
            yxxw[y, x, x, w],
            yxyw[y, x, y, w],
            yxzw[y, x, z, w],
            yxwx[y, x, w, x],
            yxwy[y, x, w, y],
            yxwz[y, x, w, z],
            yxww[y, x, w, w],
            yyxw[y, y, x, w],
            yyyw[y, y, y, w],
            yyzw[y, y, z, w],
            yywx[y, y, w, x],
            yywy[y, y, w, y],
            yywz[y, y, w, z],
            yyww[y, y, w, w],
            yzxw[y, z, x, w],
            yzyw[y, z, y, w],
            yzzw[y, z, z, w],
            yzwx[y, z, w, x],
            yzwy[y, z, w, y],
            yzwz[y, z, w, z],
            yzww[y, z, w, w],
            ywxx[y, w, x, x],
            ywxy[y, w, x, y],
            ywxz[y, w, x, z],
            ywxw[y, w, x, w],
            ywyx[y, w, y, x],
            ywyy[y, w, y, y],
            ywyz[y, w, y, z],
            ywyw[y, w, y, w],
            ywzx[y, w, z, x],
            ywzy[y, w, z, y],
            ywzz[y, w, z, z],
            ywzw[y, w, z, w],
            ywwx[y, w, w, x],
            ywwy[y, w, w, y],
            ywwz[y, w, w, z],
            ywww[y, w, w, w],
            zxxw[z, x, x, w],
            zxyw[z, x, y, w],
            zxzw[z, x, z, w],
            zxwx[z, x, w, x],
            zxwy[z, x, w, y],
            zxwz[z, x, w, z],
            zxww[z, x, w, w],
            zyxw[z, y, x, w],
            zyyw[z, y, y, w],
            zyzw[z, y, z, w],
            zywx[z, y, w, x],
            zywy[z, y, w, y],
            zywz[z, y, w, z],
            zyww[z, y, w, w],
            zzxw[z, z, x, w],
            zzyw[z, z, y, w],
            zzzw[z, z, z, w],
            zzwx[z, z, w, x],
            zzwy[z, z, w, y],
            zzwz[z, z, w, z],
            zzww[z, z, w, w],
            zwxx[z, w, x, x],
            zwxy[z, w, x, y],
            zwxz[z, w, x, z],
            zwxw[z, w, x, w],
            zwyx[z, w, y, x],
            zwyy[z, w, y, y],
            zwyz[z, w, y, z],
            zwyw[z, w, y, w],
            zwzx[z, w, z, x],
            zwzy[z, w, z, y],
            zwzz[z, w, z, z],
            zwzw[z, w, z, w],
            zwwx[z, w, w, x],
            zwwy[z, w, w, y],
            zwwz[z, w, w, z],
            zwww[z, w, w, w],
            wxxx[w, x, x, x],
            wxxy[w, x, x, y],
            wxxz[w, x, x, z],
            wxxw[w, x, x, w],
            wxyx[w, x, y, x],
            wxyy[w, x, y, y],
            wxyz[w, x, y, z],
            wxyw[w, x, y, w],
            wxzx[w, x, z, x],
            wxzy[w, x, z, y],
            wxzz[w, x, z, z],
            wxzw[w, x, z, w],
            wxwx[w, x, w, x],
            wxwy[w, x, w, y],
            wxwz[w, x, w, z],
            wxww[w, x, w, w],
            wyxx[w, y, x, x],
            wyxy[w, y, x, y],
            wyxz[w, y, x, z],
            wyxw[w, y, x, w],
            wyyx[w, y, y, x],
            wyyy[w, y, y, y],
            wyyz[w, y, y, z],
            wyyw[w, y, y, w],
            wyzx[w, y, z, x],
            wyzy[w, y, z, y],
            wyzz[w, y, z, z],
            wyzw[w, y, z, w],
            wywx[w, y, w, x],
            wywy[w, y, w, y],
            wywz[w, y, w, z],
            wyww[w, y, w, w],
            wzxx[w, z, x, x],
            wzxy[w, z, x, y],
            wzxz[w, z, x, z],
            wzxw[w, z, x, w],
            wzyx[w, z, y, x],
            wzyy[w, z, y, y],
            wzyz[w, z, y, z],
            wzyw[w, z, y, w],
            wzzx[w, z, z, x],
            wzzy[w, z, z, y],
            wzzz[w, z, z, z],
            wzzw[w, z, z, w],
            wzwx[w, z, w, x],
            wzwy[w, z, w, y],
            wzwz[w, z, w, z],
            wzww[w, z, w, w],
            wwxx[w, w, x, x],
            wwxy[w, w, x, y],
            wwxz[w, w, x, z],
            wwxw[w, w, x, w],
            wwyx[w, w, y, x],
            wwyy[w, w, y, y],
            wwyz[w, w, y, z],
            wwyw[w, w, y, w],
            wwzx[w, w, z, x],
            wwzy[w, w, z, y],
            wwzz[w, w, z, z],
            wwzw[w, w, z, w],
            wwwx[w, w, w, x],
            wwwy[w, w, w, y],
            wwwz[w, w, w, z],
            wwww[w, w, w, w]
        }
    };
}

macro_rules! impl_swizzles {
    ($vec2:ident, $vec3:ident, $vec4:ident) => {
        impl<T: Unit> glam::Vec2Swizzles for $vec2<T> {
            type Vec3 = $vec3<T>;
            type Vec4 = $vec4<T>;

            swizzles2!($vec2, $vec3, $vec4);
        }

        impl<T: Unit> glam::Vec3Swizzles for $vec3<T> {
            type Vec2 = $vec2<T>;
            type Vec4 = $vec4<T>;

            swizzles2!($vec2, $vec3, $vec4);
            swizzles3!($vec2, $vec3, $vec4);
        }

        impl<T: Unit> glam::Vec4Swizzles for $vec4<T> {
            type Vec2 = $vec2<T>;
            type Vec3 = $vec3<T>;

            swizzles2!($vec2, $vec3, $vec4);
            swizzles3!($vec2, $vec3, $vec4);
            swizzles4!($vec2, $vec3, $vec4);
        }
    };
}

impl_swizzles!(Vector2, Vector3, Vector4);
impl_swizzles!(Point2, Point3, Point4);

impl<T: Unit> glam::Vec2Swizzles for Size2<T> {
    type Vec3 = Size3<T>;
    type Vec4 = Vector4<T>;

    swizzles2!(Size2, Size3, Vector4, width, height);
}

impl<T: Unit> glam::Vec3Swizzles for Size3<T> {
    type Vec2 = Size2<T>;
    type Vec4 = Vector4<T>;

    swizzles2!(Size2, Size3, Vector4, width, height);
    swizzles3!(Size2, Size3, Vector4, width, height, depth);
}

#[cfg(test)]
mod tests {
    use crate::{size2, size3, vec2, vec3, vec4, Vec2Swizzles, Vec3Swizzles, Vec4Swizzles};

    use super::*;

    #[test]
    fn basic() {
        let v2: Vector2<f32> = vec2![1.0, 2.0];
        assert_eq!(v2.yx(), vec2![2.0, 1.0]);
        assert_eq!(v2.yxy(), vec3![2.0, 1.0, 2.0]);

        let v3: Vector3<f32> = vec3![1.0, 2.0, 3.0];
        assert_eq!(v3.zyx(), vec3![3.0, 2.0, 1.0]);
        assert_eq!(v3.zyxy(), vec4![3.0, 2.0, 1.0, 2.0]);

        let v4: Vector4<f32> = vec4![1.0, 2.0, 3.0, 4.0];
        assert_eq!(v4.wzyx(), vec4![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn size_swizzle() {
        let s2: Size2<f32> = size2!(1.0, 2.0);
        assert_eq!(s2.yx(), size2!(2.0, 1.0));
        assert_eq!(s2.yxy(), size3![2.0, 1.0, 2.0]);
        assert_eq!(s2.xyxy(), vec4![1.0, 2.0, 1.0, 2.0]);

        let s3: Size3<f32> = size3!(1.0, 2.0, 3.0);
        assert_eq!(s3.yx(), size2!(2.0, 1.0));
        assert_eq!(s3.zyx(), size3!(3.0, 2.0, 1.0));
        assert_eq!(s3.zyxy(), vec4![3.0, 2.0, 1.0, 2.0]);
    }
}

use glamour::prelude::*;

#[test]
fn const_get_set() {
    let mut v: Vector4<f32> = vec4!(1.0, 2.0, 3.0, 4.0);
    v.const_set::<3>(5.0);
    assert_eq!(v, (1.0, 2.0, 3.0, 5.0));

    let w = &mut v[3];
    *w = 6.0;
    assert_eq!(v, (1.0, 2.0, 3.0, 6.0));
}

#[test]
fn swizzle() {
    let v: Vector4<f32> = [1.0, 2.0, 3.0, 4.0].into();

    let v4 = v.swizzle::<3, 1, 0, 2>();
    assert_eq!(v4, [4.0, 2.0, 1.0, 3.0]);

    let v2 = v.swizzle2::<0, 1>();
    assert_eq!(v2, [1.0, 2.0]);

    let v3 = v.swizzle3::<0, 1, 2>();
    assert_eq!(v3, [1.0, 2.0, 3.0]);
}

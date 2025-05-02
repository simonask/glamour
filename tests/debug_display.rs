use glamour::{Unit, prelude::*};

struct BufWriter<'a> {
    buffer: &'a mut [u8],
    pos: usize,
}

impl core::fmt::Write for BufWriter<'_> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        if self.buffer.len() < bytes.len() + self.pos {
            Err(core::fmt::Error)
        } else {
            self.buffer[self.pos..self.pos + bytes.len()].copy_from_slice(bytes);
            self.pos += bytes.len();
            Ok(())
        }
    }
}

fn write_buf<'a>(buffer: &'a mut [u8], fmt: core::fmt::Arguments) -> &'a str {
    use core::fmt::Write;
    let mut writer = BufWriter { buffer, pos: 0 };
    writer.write_fmt(fmt).unwrap();
    core::str::from_utf8(&writer.buffer[0..writer.pos]).unwrap()
}

#[test]
fn vector_debug() {
    extern crate alloc;

    let untyped_f32: Vector2<f32> = Vector2 { x: 123.0, y: 456.0 };
    let untyped_f64: Vector2<f64> = Vector2 { x: 123.0, y: 456.0 };
    let untyped_u16: Vector2<u16> = Vector2 { x: 123, y: 456 };
    let untyped_i16: Vector2<i16> = Vector2 { x: 123, y: 456 };
    let untyped_u32: Vector2<u32> = Vector2 { x: 123, y: 456 };
    let untyped_i32: Vector2<i32> = Vector2 { x: 123, y: 456 };
    let untyped_u64: Vector2<u64> = Vector2 { x: 123, y: 456 };
    let untyped_i64: Vector2<i64> = Vector2 { x: 123, y: 456 };

    assert_eq!(
        alloc::format!("{:?}", untyped_f32),
        "Vector2 { x: 123.0, y: 456.0 }"
    );
    assert_eq!(
        alloc::format!("{:?}", untyped_f64),
        "Vector2 { x: 123.0, y: 456.0 }"
    );
    assert_eq!(
        alloc::format!("{:?}", untyped_u16),
        "Vector2 { x: 123, y: 456 }"
    );
    assert_eq!(
        alloc::format!("{:?}", untyped_i16),
        "Vector2 { x: 123, y: 456 }"
    );
    assert_eq!(
        alloc::format!("{:?}", untyped_u32),
        "Vector2 { x: 123, y: 456 }"
    );
    assert_eq!(
        alloc::format!("{:?}", untyped_i32),
        "Vector2 { x: 123, y: 456 }"
    );
    assert_eq!(
        alloc::format!("{:?}", untyped_u64),
        "Vector2 { x: 123, y: 456 }"
    );
    assert_eq!(
        alloc::format!("{:?}", untyped_i64),
        "Vector2 { x: 123, y: 456 }"
    );

    assert_eq!(
        alloc::format!("{:#?}", untyped_i32),
        r#"
Vector2 {
    x: 123,
    y: 456,
}"#
        .trim()
    );

    enum UnnamedUnit {}
    impl Unit for UnnamedUnit {
        type Scalar = i32;
    }

    let unnamed: Vector2<UnnamedUnit> = Vector2 { x: 123, y: 456 };
    assert_eq!(
        alloc::format!("{:?}", unnamed),
        "Vector2 { x: 123, y: 456 }"
    );
    assert_eq!(
        alloc::format!("{:#?}", unnamed),
        r#"
Vector2 {
    x: 123,
    y: 456,
}"#
        .trim()
    );

    enum CustomName {}
    impl Unit for CustomName {
        type Scalar = i32;
    }

    let custom: Vector2<CustomName> = Vector2 { x: 123, y: 456 };
    assert_eq!(alloc::format!("{:?}", custom), "Vector2 { x: 123, y: 456 }");
    assert_eq!(
        alloc::format!("{:#?}", custom),
        r#"
Vector2 {
    x: 123,
    y: 456,
}"#
        .trim()
    );
}

#[test]
fn matrix_debug() {
    extern crate alloc;

    let m4 = Matrix4::<f32>::IDENTITY;

    let s = alloc::format!("{:?}", m4);
    assert_eq!(
        s,
        "[(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0)]"
    );
}

#[test]
fn transform_debug() {
    extern crate alloc;

    struct TestSrc;
    impl Unit for TestSrc {
        type Scalar = f32;
    }

    struct TestDst;
    impl Unit for TestDst {
        type Scalar = f32;
    }

    struct TestDst2;
    impl Unit for TestDst2 {
        type Scalar = f32;
    }

    let t2 = Transform2::<TestSrc, TestDst>::IDENTITY;
    let t3 = Transform3::<TestSrc, TestDst2>::IDENTITY;

    let s2 = alloc::format!("{:?}", t2);
    let s3 = alloc::format!("{:?}", t3);
    assert_eq!(
        s2,
        alloc::format!("Transform2 {{ matrix: {:?} }}", t2.matrix)
    );
    assert_eq!(
        s3,
        alloc::format!("Transform3 {{ matrix: {:?} }}", t3.matrix)
    );
}

#[test]
fn angle_debug() {
    type Angle = glamour::Angle<f32>;

    let mut buffer = [0; 128];

    assert_eq!(
        write_buf(&mut buffer, format_args!("{:?}", Angle::HALF_CIRCLE)),
        "Angle(π)"
    );
    assert_eq!(
        write_buf(&mut buffer, format_args!("{:?}", Angle::CIRCLE)),
        "Angle(2π)"
    );
    assert_eq!(
        write_buf(&mut buffer, format_args!("{:?}", Angle::PI)),
        "Angle(π)"
    );
    assert_eq!(
        write_buf(&mut buffer, format_args!("{:?}", Angle::TAU)),
        "Angle(2π)"
    );

    assert_eq!(
        write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_1_PI)),
        "Angle(1/π)"
    );
    assert_eq!(
        write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_2_PI)),
        "Angle(2/π)"
    );
    assert_eq!(
        write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_PI_2)),
        "Angle(π/2)"
    );
    assert_eq!(
        write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_PI_3)),
        "Angle(π/3)"
    );
    assert_eq!(
        write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_PI_4)),
        "Angle(π/4)"
    );
    assert_eq!(
        write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_PI_6)),
        "Angle(π/6)"
    );
    assert_eq!(
        write_buf(&mut buffer, format_args!("{:?}", Angle::FRAG_PI_8)),
        "Angle(π/8)"
    );

    assert_eq!(
        write_buf(&mut buffer, format_args!("{:?}", Angle::from_radians(1.0))),
        "Angle(1.00000)"
    );
}

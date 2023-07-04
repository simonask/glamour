use super::Scalar;

/// The name of a coordinate space.
///
/// The unit is used to give vector types a "tag" so they can be distinguished
/// at compile time.
///
/// The unit also determines which type is used as the scalar for that
/// coordinate space. This is in contrast to a crate like
/// [euclid](https://docs.rs/euclid/latest/euclid), where the unit and the
/// scalar are separate type parameters to each vector type.
///
/// Note that primitive scalars (`f32`, `f64`, `i32`, and `u32`) also implement
/// `Unit`. This allows them to function as the "untyped" vector variants.
///
/// #### Example
/// ```rust
/// # use glamour::prelude::*;
/// struct MyTag;
///
/// impl Unit for MyTag {
///     // All types that have `MyTag` as the unit will internally be based
///     // on `f32`.
///     type Scalar = f32;
/// }
///
/// // This will be `glam::Vec2` internally, and can be constructed directly
/// // with `f32` literals.
/// let v: Vector2<MyTag> = Vector2 { x: 1.0, y: 2.0 };
/// assert_eq!(v[0], v.x);
/// ```
pub trait Unit: 'static {
    /// One of the vector component types of glam: `f32`, `f64`, `i32`, or
    /// `u32`.
    type Scalar: Scalar;

    /// Human-readable name of this coordinate space, used in debugging. `None`
    /// by default.
    #[must_use]
    fn name() -> Option<&'static str> {
        None
    }
}

impl Unit for f32 {
    type Scalar = f32;
    fn name() -> Option<&'static str> {
        Some("f32")
    }
}

impl Unit for f64 {
    type Scalar = f64;
    fn name() -> Option<&'static str> {
        Some("f64")
    }
}

impl Unit for i32 {
    type Scalar = i32;
    fn name() -> Option<&'static str> {
        Some("i32")
    }
}

impl Unit for u32 {
    type Scalar = u32;
    fn name() -> Option<&'static str> {
        Some("u32")
    }
}

impl Unit for i64 {
    type Scalar = i64;
    fn name() -> Option<&'static str> {
        Some("i64")
    }
}

impl Unit for u64 {
    type Scalar = u64;
    fn name() -> Option<&'static str> {
        Some("u64")
    }
}

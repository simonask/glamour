# Documentation

Table of contents:

- [Why](#why)
- [How](#how)
- [Design and limitations](crate::docs::design)
- [Examples](crate::docs::examples)

# Why

It's often useful to use strong types in linear algebra to represent things
like different coordinate spaces and units, because it allows the programmer
to lift some invariants into the type system. For example, distinguishing
between vectors and points is important when using homogeneous coordinates,
and subtracting two points should yield a vector rather than another point.
Another example is coordinates in different coordinate spaces - e.g., "world
space" vs. "view space".

The [Euclid][euclid] crate is a popular example of a library that implements
this, but it comes with a few limitations. In particular, it doesn't support
[`bytemuck`][bytemuck], which prevents mapping SIMD math types directly into
GPU buffers in safe code.

[`glam`][glam] is a somewhat smaller library with a strong focus on
performance. And it supports `bytemuck`. This is what allows us to implement
strong types on top of it with zero overhead using type punning, while
remaining compatible with low-level concepts like memory mapping.

[euclid]: https://docs.rs/euclid/latest/euclid/
[glam]: https://docs.rs/glam/latest/glam/

# How

In short: We define a set of linear algebra types with the exact same memory
layout as the `glam` vector types. Then we use `bytemuck` to forward all method
calls to the corresponding `glam` implementation - at zero runtime cost.

In principle, our vector types and the `glam` types are unrelated, but by
carefully crafting our types to follow the layout of `glam` types, we can
legally cast references between them, because [type punning][] is
legal under these conditions in Rust.

[type punning]: https://en.wikipedia.org/wiki/Type_punning

But to actually allow vector types to be strongly typed, not only in the
conceptual distinction between "vector", "point", "size", etc., but also in a
way that the user can customize (i.e., different types of "vectors", "points",
"sizes", etc.), all types are generic and take a [`Unit`](crate::traits::Unit)
parameter. To further complicate matters, the design goal has been to avoid
`PhantomData` members, because they hurt the ergonomics of using the types as
truly "POD-like" - in particular, destructuring and construction become much
more verbose.

To solve this, we add some extra information to user-defined `Unit`s. Any unit
is associated with a [`Scalar`](crate::traits::Scalar) type indicating the
component type of the vector, typically one of the primitive types supported by
`glam`: `f32`, `f64`, `i32`, or `u32`. The type of scalar determines which
`glam` vector type is used for arithmetic in the vector type with that unit.


| Raw scalar | 2D                    | 3D                    | 4D                    |
| ---------- | --------------------- | --------------------- | --------------------- |
| `f32`      | [`Vec2`][glam_vec2]   | [`Vec3`][glam_vec3]   | [`Vec4`][glam_vec4]   |
| `f64`      | [`DVec2`][glam_dvec2] | [`DVec3`][glam_dvec3] | [`DVec4`][glam_dvec4] |
| `i32`      | [`IVec2`][glam_ivec2] | [`IVec3`][glam_ivec3] | [`IVec4`][glam_ivec4] |
| `u32`      | [`UVec2`][glam_uvec2] | [`UVec3`][glam_uvec3] | [`UVec4`][glam_uvec4] |

The mapping to vector types also works for custom scalar values, through the
[`Scalar`](crate::traits::Scalar) trait.

[glam_vec2]: glam::Vec2
[glam_vec3]: glam::Vec3
[glam_vec4]: glam::Vec4
[glam_dvec2]: glam::DVec2
[glam_dvec3]: glam::DVec3
[glam_dvec4]: glam::DVec4
[glam_ivec2]: glam::IVec2
[glam_ivec3]: glam::IVec3
[glam_ivec4]: glam::IVec4
[glam_uvec2]: glam::UVec2
[glam_uvec3]: glam::UVec3
[glam_uvec4]: glam::UVec4

Custom implementations of [`Scalar`](crate::traits::Scalar) are also possible -
as long as those values remain 100% bitwise compatible with one of the
fundamental primitive types mentioned above.

All of this type juggling is only possible in safe code due to the particular
guarantees of the [`bytemuck::Pod`](bytemuck::Pod) trait. In _theory_ those
casts come with some overhead, because they check the alignment and size of all
casts, but the compiler should be able to optimize away all of those checks.

# Strongly typed vector math with `glam`

[![Build Status]][github-ci] [![codecov-badge]][codecov]
[![Latest Version]][crates.io] [![docs]][docs.rs]
[![Minimum Supported Rust Version]][Rust 1.85.0]

This crate uses [bytemuck][] to implement a zero-cost[^1] strongly typed
interface on top of [glam][].

The API is similar to [euclid][], but more ergonomic (although YMMV).

One of the API design goals of `glam` is to avoid complexity by not going
bananas with traits and generics. **This crate is the opposite.** But it does
allow you to easily drop down to plain `glam` when needed.

[See the `docs` module for detailed documentation.](crate::docs)

[bytemuck]: https://docs.rs/bytemuck/latest/bytemuck/
[glam]: https://docs.rs/glam/latest/glam/
[euclid]: https://docs.rs/euclid/latest/euclid/

[^1]: Zero-cost at runtime, in release builds. This crate may increase
    compile times and make debug builds slower due to increased code size.

# Step-By-Step Quickstart Guide

1. Declare your units by defining a "unit" type (can be empty, doesn't need any
   traits to be derived).
2. Implement [`Unit`] for that struct. [`Unit::Scalar`] determines the primitive
   type used in vector components.
3. The scalar must be `f32`, `f64`, `i32`, `u32`, `i64`, `u64`, `i16`, or `u16`.
4. The basic primitive scalars are also units in their own right ("untyped").

##### Example

```rust
use glamour::prelude::*;

struct MyUnit;
impl Unit for MyUnit {
    type Scalar = f32;
}

// Start using your new unit:
let vector: Vector4<MyUnit> = Vector4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
let size: Size2<MyUnit> = Size2 { width: 100.0, height: 200.0 };

// Use untyped units when needed:
let vector_untyped: &Vector4<f32> = vector.as_untyped();

// Use glam when needed:
let vector_raw: &glam::Vec4 = glamour::Transparent::peel_ref(&vector);
```

[See the documentation module for more examples.](crate::docs::examples)

# Feature gates

- `std` - enables the `glam/std` feature. Enabled by default.
- `libm` - required to compile with `no_std` (transitively enables `glam/no_std`).
- `mint` - enables conversion to/from
  [`mint`](https://docs.rs/mint/latest/mint/) types.
- `encase`: Enables implementations of [`encase::ShaderType`] for vector and matrix types, which enables them for use in
  GPU shaders.
- `scalar-math`: Don't use SIMD vector instructions, even if they are supported by the target architecture. Note that
  this flag is required to run tests under Miri, due to vector instructions not being supported. Transitively enables
  the `glam/scalar-math` feature.
- `wasmtime`: **(Experimental)** This enables implementations of `Lower`/`Lift`
  on all types, so they can be used in generated bindings for WIT components
  ([`wasmtime::component::bindgen!()`]). Glamour types can be used on both sides
  (host and guest), and can be passed "toll-free" between the two sides given a
  compatible type declaration in a WIT world, but limitations apply: Due to the
  way the `wasmtime` derive macros work, only plain scalar units can be used (so
  `Vector4<f32>` is supported, but not `Vector4<MyFloatUnit>`).
- `facet`: **(Experimental)** Adds support for
  [`facet`](https://docs.rs/facet/latest/facet/) to all types.

[`wasmtime::component::bindgen!()`]: https://docs.rs/wasmtime/latest/wasmtime/component/macro.bindgen.html

# Advantages

- Structural type construction is sometimes better because it doesn't rely on
  positional arguments. It also allows us to give more meaningful names to
  things - for example, the members of [`Size2`] are called `width` and
  `height`, rather than `x` and `y`.
- The user is able to easily drop down to plain `glam` types when needed.

##### Over plain `glam`

- Lifts some correctness checks to the type system. This can prevent certain
  common bugs, such as using a vector from one coordinate space in a context
  that logically expects a different coordinate space.
- Improves API comprehension and code readability by annotating expectations as
  part of function signatures.
- Distinguishing between points, vectors, and sizes can also prevent certain
  classes of bugs. For example, the "transform" operation in 3D is different for
  points and vectors.

##### Over `euclid`

- Type names are more concise (single generic parameter instead of two).
- Support for `bytemuck`.

# Disadvantages

- The API is heavily reliant on metaprogramming tricks. [A complex maze of
  traits](crate::traits) is required to support the goals. The trade-off can be
  summed up as: simplicity, ergonomics, type-safety - pick two. This crate picks
  ergonomics and type-safety.
- Generic struct definitions have trait bounds. This is usually considered an
  antipattern in Rust, but we need to encode two things with one type parameter
  to support structural construction of vector types, so it is unavoidable.

##### Compared to `glam`

- Due to its simplicity, `glam` is a very approachable API.
- `glam` is able to support a wide range of transformation primitives (e.g.
  [`glam::Affine3A`], [`glam::Quat`], etc.), and the user has a lot of
  flexibility to choose the most performant kind for their use case. These are
  simply unimplemented in `glamour`.

##### Compared to `euclid`

- The same unit tag cannot be used with different scalars.
- Any type cannot be used as the unit tag - it must implement [`Unit`].

# Goals

- [x] Strongly typed linear algebra primitives.
- [x] Bitwise compatibility with `glam`.
- [x] First-class [field struct expression] support in vector types.
- [x] Support direct memory mapping (e.g. upload to GPU buffers).
- [x] Support `no_std`.
- [x] Adhere to `glam` API conventions - "principle of least surprise".
- [x] Add only a few additional geometric primitives, like [rects](Rect),
  [transforms](Transform2), and [axis-aligned boxes](Box2).
- [x] Impose no runtime overhead at all (compared to using `glam` directly).
  Comprehensive benchmarks pending.
- [ ] 100% test coverage.

[field struct expression]: https://doc.rust-lang.org/reference/expressions/struct-expr.html#field-struct-expression

# Non-goals

- Complex linear algebra. Use [nalgebra][nalgebra] or [Euclid][euclid] instead.
- Vector sizes beyond 4 dimensions (the maximum supported by `glam`).
- Type parameterization on vector/matrix size.
- Non-square matrices.
- Wrapping all of the `glam` API. Instead, we make it really easy (and
  performant) to drop down to `glam` types when needed.
- Hiding the `glam` API. It's OK to use `glam` types in public APIs.
- The "`AoSoA`" pattern ("extra wide" vector types). Use [ultraviolet][uv]
  instead.

[uv]: https://docs.rs/ultraviolet/latest/ultraviolet/
[nalgebra]: https://docs.rs/nalgebra/latest/nalgebra/

# Performance

All operations should perform exactly the same as their `glam` counterparts.
There is a zero-tolerance policy for overhead in release builds.

However, debug build performance is also important in some cases. For example,
for a video game it can make the difference between being playable or not in
debug mode.

This crate should be expected to incur an overhead of about a factor 2 compared
to `glam` in debug builds. This may be alleviated in the future, but it seems
that even `glam` itself does not go out of its way to perform well in debug
builds.

[Build Status]: https://github.com/simonask/glamour/actions/workflows/ci.yml/badge.svg
[github-ci]: https://github.com/simonask/glamour/actions/workflows/ci.yml
[codecov-badge]: https://codecov.io/gh/simonask/glamour/branch/main/graph/badge.svg?token=VKK61NGSAJ
[codecov]: https://codecov.io/gh/simonask/glamour
[Latest Version]: https://img.shields.io/crates/v/glamour.svg
[crates.io]: https://crates.io/crates/glamour/
[docs]: https://docs.rs/glamour/badge.svg
[docs.rs]: https://docs.rs/glamour/
[Minimum Supported Rust Version]: https://img.shields.io/badge/Rust-1.85.0-blue?color=fc8d62&logo=rust
[Rust 1.85.0]: https://releases.rs/docs/1.85.0/

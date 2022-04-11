# Design

The idea is to provide a set of bindings to `glam` that enforce a stricter API,
without having to implement all of the algorithms.

This is achieved by defining a number of vector and matrix types, ensuring that
they are bitwise compatible with their `glam` equivalents, and then forwarding
(almost) all method calls to the underlying `glam` implementation.

This is made possible by the fact that type punning in Rust is allowed under
specific circumstances. The [`bytemuck`] crate provides a way to encode the
relevant invariants for type punning to work in the type system, and we use that
to forward calls to `glam` as if the caller type actually was the corresponding
`glam` type.

`glam` was chosen as the underlying library because of its simplicity and
popularity. It is also one of the best performing vector math libraries for
Rust.

Compatibility with `glam` is not limited to the private implementation of types,
but extends to public APIs as well, i.e., transparent conversion between
`glamour` and `glam` types. All `glamour` types can be converted to and from
their equivalent `glam` types, and references to `glamour` types can be cast to
references to `glam` types. However, [due to alignment
limitations](#vector-overalignment), references to `glam` types cannot
necessarily be cast to `glamour` types. These can always be converted by value.

Being transparent about underlying `glam` types serves several purposes:

- Enabling gradual or selective migration from `glam` to `glamour`.
- Enabling use of the `glam` API directly when `glamour` API is insufficient.
- Avoiding the need to wrap the entire `glam` API. For example, there is no real
  reason to define wrappers around boolean vector types ([`glam::BVec2`],
  [`glam::BVec3`], [`glam::BVec4`]), because they are semantically a different
  thing from value vectors, so we just expose `glam`'s boolean vectors directly
  in APIs.

### API

Whenever possible the API should conform to the expectations of a user coming
from `glam`:

- Method names should match their `glam` equivalents, except when the `glam`
  name encodes information that we can express with types. (E.g.
  [`Matrix3::transform_point()`](crate::Matrix3::transform_point()) versus
  [`glam::Mat3::transform_point2()`]).
- Vector-like things take `self` by value.
- Matrix-like things take `self` by reference.

To avoid name collisions with `glam`, this crate has chosen to use `Vector`
instead of `Vec`, `Matrix` instead of `Mat`, and so on.

In terms of ergonomics, we go a bit further than `glam`. For example, all vector
types support comparison with tuples in addition to conversion from tuples.

### Implementation

We define a set of traits that describe `glam`'s vector and matrix types. In
general, only the element type is generic - that is, we don't treat
`Vec2`/`Vec3`/`Vec4` as interchangeable in generic code, but we *do* treat
`Vec2`/`IVec2`/`UVec2`/`DVec2` (etc.) as interchangeable.

This allows us to back `Vector2<T>` (etc.) with a different `glam` vector under
the hood, while having the same name in user code.

However, `glam` avoids exposing public traits for vector types with the
rationale that a vector is not an interface, but a value. For the same reason,
we do not implement the traits used in mapping to `glam` types for the public
types in the API.

The relevant traits are:

- [`SimdVec`](crate::traits::SimdVec) - describing all `glam` vector types. Note
  that this has a generic const parameter `D` describing the dimensionality of
  the vector, but this is only for brevity and should not be relied upon in
  generic code. Specifically it exists because we require that vector types can
  be converted to constant-sized arrays of scalars, so we need `D` in
  supertraits of `SimdVec`.
- [`SimdMatrix`](crate::traits::SimdMatrix) - describing all `glam` matrix
  types.
- [`Primitive`](crate::traits::Primitive) - describes how to map from a
  primitive to a `glam` vector type of the desired size.
- [`PrimitiveMatrices`](crate::traits::PrimitiveMatrices) - describes how to map
  from a primitive to a `glam` matrix type of the desired size.
- [`Scalar`](crate::traits::Scalar) - describes which primitive to use for a
  scalar. This will only be different from `Primitive` when using a newtype -
  normally the primitive itself is the scalar.
- [`Unit`](crate::traits::Unit) - describes which scalar to use for a particular
  logical unit.

Since all of this genericity creates quite the soup of associated types and
trait bounds, a number of traits exist as shorthands for convenience:

- [`ScalarVectors`](crate::traits::ScalarVectors) - shorthands for `<<T as
  Scalar>::Primitive as Primitive>::VecN`.
- [`UnitTypes`](crate::traits::UnitTypes) - shorthands for `<T::Scalar as
  Scalar>::Primitive` and `<<T::Scalar as Scalar>::Primitive as
  Primitive>::VecN`.

Public types use the shorthand trait bounds whenever possible.

`glamour` makes extensive use of macros to forward method calls to `glam`
(similar to what `glam` does internally).

# Limitations

#### Type deduction

The type of fields in vector types are dependent on the associated type
`T::Scalar`, not `T` itself. This means that the type of `Vector<T>` cannot be
deduced from its members, so this will not work:

```rust,compile_fail
let v = Vector4 { x: 1.0, y: 2.0, z: 3.0, w: 4.0 };
// Error: Cannot deduce type parameter `T`.
```

However, type deduction can be made to work again in the following ways:

- Create type aliases for the vector types with your unit (`type MyVector = Vector4<MyUnit>;`).
- Construct vector types at API boundaries where the function signature can tell
  the compiler what the unit is.
- Explicitly specify the unit parameter when instantiating the vector type.

#### Vector overalignment

Some types in `glam` are overaligned: [`Vec3A`](glam::Vec3A) and
[`Mat3A`](glam::Mat3A). Due to the padding, [`Pod`](bytemuck::Pod) cannot be
implemented for those types, and so we cannot transparently cast references to
them. The reason they exist is that they enable SIMD optimizations for a logical
[`Vec3`](glam::Vec3) or [`Mat3`](glam::Mat3), but if we had to implement
wrappers for them, they would have to perform conversion by value, which is
likely to defeat any performance benefit of the SIMD alignment. The workaround
is to just use [`Vector4`](crate::Vector4) instead, or alternatively drop down
to [`Vec3A`](glam::Vec3A) using
[`Vector3::to_vec3a()`](crate::Vector3::to_vec3a()).

Some vector types in `glam` are not SIMD-aligned: `IVec4` and `UVec4` are 4-byte
aligned, and `DVec4` is 8-byte aligned, despite of `Vec4` being 16-byte aligned
for maximum SIMD performance.

However, we share a single struct definition for all scalar types
([`Vector4`](crate::Vector4)), and it is not possible to choose different
alignments for generic parameters in Rust without having a member of the struct
with that particular alignment.

We don't want a "dummy" field in `Vector4`, so instead the type is always
16-byte aligned. This means it is actually over-aligned for `i32`, `u32`, and
`f64` scalars, which in turn means that we can't safely reinterpret-cast
`IVec4`, `UVec4`, and `DVec4` to `Vector4<T>`.

#### Operator overloading

The type of fields in vector types are dependent on the associated type `<T as
Unit>::Scalar`. This has the consequence that implementing any trait for both
the vector type itself as well as the scalar is not possible. The compiler
correctly deduces that nothing prevents somebody from defining the associated
type in such a way that the trait implementations become overlapping.

An example of this issue:

```rust,ignore
impl<T: Unit> Mul<Vector4<T>> for Vector4<T> {
    // component-wise multiplication
}

impl<T: Unit> Mul<T::Scalar> for Vector4<T> {
    // multiply all components by scalar

    // XXX: Overlapping trait bound, because `T::Scalar` could be `Vector4<T>`.
}
```

There is no way to solve this generically at the moment. The upside is that it
is typically only a problem in generic code. `Mul<f32>` etc. can indeed be
implemented for all `Vector4<T>` where `T::Scalar: Scalar<Primitive = f32>`.
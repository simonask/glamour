# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.18.0] - 2025-05-08

### Fixes

- Fixed a soundness hole in `Vector4` when using smaller scalar types.

### Breaking changes

- Vector types and references can no longer be converted to and from `glam`
  types by reference. This fixes a soundness hole, because it relaxes the
  alignment requirement of `Vector4`, which would otherwise result in a size
  mismatch between `Vector4` and `glam` vectors for smaller scalar types. I.e.
  `glam::U8Vec4` has alignment 1. In general, it is brittle to rely on the
  specific alignment of `glam` vector types, because they are highly
  architecture dependent, and it is unlikely that it gains anything, because
  unaligned vector register loads are no longer slow (with AVX, the `vaddps`
  instruction supports unaligned loads natively).

## [0.17.0] - 2025-05-03

### Added

- Support for `u8` and `i8` scalars, mapping to `glam::U8Vec{N}` and `glam::I8Vec{N}`.
- Implement right-hand side scalar `Add`, `Sub`, and `Mul` (#69).
- Implement scalar `Point` operations (#71).
- Implement `Mul<Vector{2,3}>` and `Mul<Point{2,3}>` for `Matrix{3,4}`.
- Implemented all the methods that exist on `Box2` for `Box3` as well.
- Experimental support for [`facet`](https://docs.rs/facet/latest/facet/).

### Breaking changes

- Migrate to Edition 2024. This also bumps MSRV to Rust 1.85.0.
- Updated `wasmtime` dependency to 32.0.
- Bumped `encase` dependency to 0.11.
- `Box2::center()` is now only implemented for floating-point units. In return,
  it is now implemented in terms of `glam::Vec2::midpoint()`, which yields
  better codegen.

## [0.16.0] - 2025-02-19

### Fixes

- `Vector4::ONE.w` was accidentally zero.

### Breaking changes

- Bumped `glam` to 0.30. [Changelog](https://github.com/bitshifter/glam-rs/blob/main/CHANGELOG.md#0300---2025-02-18).
- Bumped `wasmtime` dependency to 29.x.
- Shuffled trait bounds between `IntUnit` and `IntScalar` to support mapping integers to
  their unsigned counterparts (needed for `manhattan_distance()` etc.).

## [0.15.0] - 2024-10-12

### Breaking changes

- Bumped `wasmtime` dependency to 25.0.
- Bumped `encase` dependency to 0.10.

## [0.14.1] - 2024-09-16

### Added

- (Re-)added missing glam methods:
  - `Vector3::cross()`
  - `VectorN::dot_into_vec()`
  - `VectorN::div_euclid()`
  - `copysign()`
  - `rem_euclid()`
  - `MatrixN::mul_scalar()`, `div_scalar()`.
  - `Matrix3::from_quat()`, `from_axis_angle()`, `from_rotation_x()`,
    `from_rotation_y()`, `from_rotation_z()`.
  - `Matrix4::look_to_lh()`, `look_to_rh()`.
- Re-added convenience methods `as_raw()`, `to_raw()`, `from_raw()` for SIMD types that map
  to glam `VecN` types.

### Internal changes

- Added some `#[expect(lint)]` annotations.

## [0.14.0] - 2024-08-27

### Added

- Added experimental support for WASM preview 2 components via `wasmtime`. All
  `glamour` types can be used in bindings generated for a compatible WIT
  interface by both `wasmtime::component::bindgen!()` and
  `wit_bindgen::generate!()`, using the `with` configuration key.
- Implemented `glam::Vec2Swizzles`, `glam::Vec3Swizzles`, and
  `glam::Vec4Swizzles` for all vectorlike types. 4-component swizzles of `Size2`
  and `Size3` map to `Vector4`.

## [0.13.0] - 2024-08-25

### Added

- Added `Matrix3::from_mat4_minor()` and `Matrix2::from_mat3_minor()` (glam 0.29).
- Added `is_finite_mask()` to vectorlike types (glam 0.29).
- Added `reflect()` and `refract()` for vector types (glam 0.29).
- Added `map()` to all vectorlike types (glam 0.29).
- Vector arithmetic ops are now implemented on references (mirroring glam 0.29).

### Breaking changes

- Bumped `glam` to 0.29.0.

## [0.12.0] - 2024-08-25

### Added

- Convenience traits `FloatUnit`, `IntUnit`, and `SignedUnit`, which have the
  implied trait bounds for the associated scalar type. These traits are automatically
  implemented for all `T: Unit` where `T::Scalar` is of the appropriate type.
- The `as_()` method for vectorlike types, implementing casting between units using
  the `as` operator (through `num_traits::AsPrimitive`).
- Added `rotate_towards()` for 2D float vector types.
- Added `Rect::from_origin_and_size()` which takes its arguments as `impl Into`.
- Made several vectorlike constructors `const`.

### Breaking changes

- Bumped MSRV to Rust 1.79.
- Bumped `glam` to 0.28.0.
- Many matrix methods were out of sync with the `glam` API. These interfaces now match exactly.
- `Box2::new()`, `Box3::new()`, and `Rect::new()` no longer take their arguments as `impl Into`.
- Vectorlike types can no longer be compared against raw tuples.
- Removed the `ToRaw` and `AsRaw` traits. Use a custom `Transparent` instead, similar to
  `bytemuck::TransparentWrapper`.
- Removed `ZERO` and `ONE` associated consts, in favor of `num_traits::{ConstZero, ConstOne}`.
- Added `<T: Scalar>` bound to `Angle<T>`.
- `angle_between()` was renamed to `angle_to()` in glam 0.28.0.
- Removed `Unit::name()` method, which had a doubtful use case.

### Internal changes

- Major cleanup of codegen macros, which enables exact and robust mapping to `glam` APIs.
- Removed cfg attributes for a `cuda` feature that was never really supported.
- Added `[lints]` section to Cargo.toml to prevent warnings about coverage
  attributes and `spirv` on Rust 1.80+.

### Experimental

- Experimental feature `wasmtime` derives `wasmtime::component::ComponentType`
  for all types, such that they can be used verbatim in wasmtime hosts when
  communicating with components. Both host and guest can map interface types
  directly to glamour types, as long as the WIT record matches the layout of the
  glamour types.

## [0.11.1] - 2024-03-26

### Fixed

- Deserialization of `Rect`, `Box2`, and `Box3` relied on borrowed strings for
  field names, which is incompatible with some formats.

## [0.11.0] - 2024-03-26

### Added

- `fract_gl()`, and made `fract()` available on points and sizes as well.
- `with_x()`, `with_y()`, etc., as well as `with_width()` and `with_height()`
  for size types.
- `PointN::midpoint()`
- `PointN::move_towards()`
- `saturating_add()` and friends for integer vector types.
- `wrapping_add()` and friends for integer vector types.
- `element_sum()`, `element_product()`
- `VectorN::normalize_or()`
- `From<glam::BVecN>` for vector types.
- `MatrixN::abs()`
- Implement `Mul`, `MulAssign`, `Div`, `DivAssign` by scalars for matrix types.

### Breaking changes

- Bumped dependency on `glam` to 0.27.0.
- Minimum Supported Rust Version bumped to 1.68.2 for `impl From<bool> for
  {f32,f64}` support.

## [0.10.3] - 2024-03-25

### Changes

- Serialization/deserialization are now manually implemented for all types,
  instead of relying on `#[derive(Serialize, Deserialize)]`.
- Vector types, size types, rects, and boxes can now be deserialized from
  sequence-shaped input, not only struct-shaped data.
- The `derive` feature of `serde` is now disabled.
- The dependency on `serde` now sets `default-features = false`, enabling use of
  the `serde` feature in `nostd` environments.

## [0.10.2] - 2024-02-22

### Added

- The traits `SignedScalar` and `FloatScalar` are now public.

## [0.10.1] - 2024-01-08

### Added

- `Transform2::from_matrix_unchecked()` and `Transform3::from_matrix_unchecked()` are now const fns.

## [0.10.0] - 2023-12-27

### Added

- Added support for 16-bit integer scalars, via `glam::I16Vec{N}` etc.
- Added `Vector2::to_angle()` for floating-point scalars.
- Implement `AsRef<[[T; N]; N]>` for matrix types (as well as `AsMut` and `From` counterparts).

### Breaking changes

- Bumped dependency on `glam` to 0.25.0.

### Internal changes

- No longer implement `encase` traits via `glam`. This is for maintenance purposes, because the `glam` dependency of
  `encase` is often outdated.

## [0.9.0] - 2023-11-22

### Added

- Missing `truncate()` method for `Vector3`.
- Added `Size2::is_empty()` and `Size3::is_empty()`.
- Implementation of `Product` for `Vector2`, `Vector3`, and `Vector4`.
- Implemented `bytemuck::TransparentWrapper` for applicable types.

### Breaking changes

- Changed the field structure of `Matrix2`, `Matrix3`, and `Matrix4` to match `glam`.
- Changed the implementations of `Rect::is_empty()`, `BoxN::is_empty()`, `Rect::is_negative()` to match
  the behavior of `euclid`.
- Implemented serialization for matrix types such that they are serialized to/from flat
  column arrays. Note that glam does not implement serialization for matrices.
- Added `#[serde(transparent)]` to `Transform2` and `Transform3`, such that their
  serialized representation is always identical to the inner matrix type.
- Added `#[serde(transparent)]` to `Angle`, changing its serialized representation to
  just the scalar value.
- Bumped Minimum Supported Rust Version to 1.65.0. This is to simplify CI because
  Criterion relies on a version of `regex` that only works with that version.

### Internal changes

- Simplified handling of bitmasks, making it more clear in documentation that the `glam::BVec*`
  types are indeed used directly.
- Removed the meaningless `is_finite()` trait method from `Scalar`, moved it to `FloatScalar` instead.

## [0.8.1] - 2023-10-27

### Added

- Missing `truncate()` method for `Vector4`.

### Fixes

- Fix lints caught by newer version of clippy.

## [0.8.0] - 2023-07-04

### Added

- Support for `i64` and `u64` scalar types based on `glam::I64VecN` and `glam::U64VecN`.
- Added `INFINITY` and `NEG_INFINITY` associated constants to vector types, and also added the missing `NAN` associated
  constant to `PointN` and `SizeN`.
- Implemented `Borrow<glam::VecN>` for all vector types, and `Borrow<glamour::VectorN/PointN/SizeN>` for `glam` types.
  This enables the interchangeable use of glamour/glam types in hash maps.

### Breaking changes

- Bumped `encase` dependency to 0.6.0.

## [0.7.1] - 2023-04-11

### Added

- Implement `Hash` for integer-based types.

## [0.7.0] - 2023-04-11

### Breaking Changes

- Bumped dependency on `glam` to 0.23.0.
- Bumped dependency on `encase` to 0.5.0.

## [0.6.0] - 2023-01-25

### Added

- Exposed glam's `core-simd` feature.

### Breaking Changes

- Bumped the dependency on `glam` to 0.22.0.
- Bumped the dependency on `encase` to 0.4.1.
- Removed the `glam_0_20` and `bevy_0_7` features, since newer versions of Bevy
  updated the dependency on `glam`.
- Due to breaking changes in `glam`'s `BVec4`/`BVec4A` types (which used to be
  type aliases), those changes are now reflected.

## [0.5.0] - 2022-10-23

### Added

- Almost 100% of the glam vector and matrix APIs are covered.
- Added doc links to glam equivalents for almost all methods.
- Conversion to/from glam types formalized in `ToRaw`/`FromRaw` traits.
- Feature gate `encase`, which enables implementations of [`encase`] traits for
  all types. This allows host-sharing of `glamour` types with WGSL shaders.
- `Box2`/`Box3` rounding (in, out, normal).

### Removed

- The `Primitive` trait is removed, as `Scalar` is no longer directly
  customizable. The rationale is that it introduced more complexity than it was
  worth.

### Breaking Changes

- Major trait reorganization. The `Scalar` trait is no longer customizable.
- `Angle<T>` can no longer appear as the component type of a vector.
- Replaced `Lerp` trait with regular methods.
- Removed `UnitTypes` helper traits. Method availability can be determined by
  `Unit::Scalar` bounds.

### Bugfixes

- `Rect::round_out()` now converts to `Box2` before rounding, which gives
  correct results.

### Internal changes

- Major simplification of traits and macros. It should be way easier to grok the
  code.

## [0.4.1] - 2022-07-26

### Added

- Conversion to/from glam 0.20.0 types (gated behind the `glam_0_20` feature).
- New feature: `bevy_0_7_0`. Bevy uses glam 0.20.0, but `glamour` uses 0.21.
  Enable the feature to integrate `glamour` with Bevy 0.7 types.
- Added missing `Into`/`From` implementations for matrix types.

## [0.4.0] - 2022-06-27

### Added

- Added APIs mirroring new additions in `glam` 0.21.1: `Vector2::from_angle()`,
  `Vector2::rotate()`, `NEG_*` constants for signed vector types.
- Feature: `scalar-math`, which is required if `glam` is built with its
  `scalar-math` feature. This feature affects alignment of vector types.

### Changed

- Bumped `glam` version to 0.21.1. Note that this version introduced breaking
  changes.
- Fixed alignment of 4-component types to match `glam` exactly. Previously, it
  didn't work with the `scalar-math` feature.
- Constified a couple of methods. Unfortunately, since `bytemuck::cast` and
  friends are not `const`, functions that delegate directly to `glam` cannot be
  made `const` for the time being, even if those functions are `const fn` in
  `glam`.

### Breaking Changes

- All `new` methods now receive parameters as `impl Into`.
- Minimum supported Rust version bumped to 1.61.0, because of support for
  generic `const fn` methods.
- `glam` 0.21.1 contains breaking changes:
  <https://github.com/bitshifter/glam-rs/blob/main/CHANGELOG.md#breaking-changes>.

### Internal changes

- Renamed the `ValueSemantics` marker trait to `PodValue`.
- Added `BVec` associated types to `Primitive`.
- Cleaned up implementation macros, removed the `impl_as_tuple` macro (because
  tuple conversion is now present for all types).
- Cleaned up some trait bounds for documentation clarity.

## [0.3.1] - 2022-04-25

### Added

- Added `length_squared` and `normalize_or_zero` for vectors (#16).

## [0.3.0] - 2022-04-19

### Added

- Constructor macros.
- Support for `glam::Quat` and `glam::DQuat` in `Angle<T>`, `Vector3<T>`, and
  `Point3<T>`.

### Removed

- `Vector4::max_element_w()` (no longer needed).
- Implementations of `AbsDiffEq` (et al) with tuple right-hand side.
- `Scalar::min()`, `Scalar::max()`, and `Scalar::two()`, as they are unneeded.
- `VectorN::two()`, as it was only used in two places.

## Changed

- Bumped `glam` dependency to 0.20.5, which fixes the scalar math implementation
  of `Vec4::max_element()`.
- Changed methods returning constants (`T::zero()`, `T::one()`, `T::nan()`,
  `T::identity()`) to associated constants (`T::ZERO`, `T::ONE`, `T::NAN`,
  `T::IDENTITY`), to conform with expectations from the glam API.
- Brought `Matrix` interfaces must closer to their `glam` equivalents.

## [0.2.0] - 2022-04-12

### Added

- Implemented `approx::AbsDiffEq`, `approx::RelativeEq`, and `approx::UlpsEq`
  for more types (even where `glam` doesn't implement them).
- Added `signum()` methods.
- `Matrix::with_rows()` and `Matrix::with_cols()` for more convenient matrix
  construction.
- `Matrix::zero()`, `Matrix::nan()` for constructing invalid matrices.
- Exposed `Matrix::determinant()`, `Matrix::is_invertible()`,
  `Matrix::inverse()`, `Matrix::is_nan()`, `Matrix::is_finite()`.
- `Primitive::is_finite()`, `Primitive::is_nan()`.
- Very, very many tests.

### Fixed

- `Rect::is_empty()`, `Rect::is_negative()`, `Box2::is_empty()`, and
  `Box2::is_negative()` now return true when containing non-finite components.
- `Box2::contains(point)` did not return true for coordinates exactly on the
  upper bound.
- `Matrix::is_invertible()` now returns false when the determinant is
  non-finite.
- `Transform::from_matrix()` now checks if the matrix is valid before returning
  a transform. Use `Transform::from_matrix_unchecked()` to skip this check.

### Changed

- Bumped Minimum Supported Rust Version to 1.56.0 because we want to use edition
  2021.
- Add CI workflows.
- Add CI/docs badges to README.
- `Matrix::inverse()` renamed to `Matrix::inverse_unchecked()`, and
  `Matrix::inverse_checked()` to `Matrix::inverse()`.
- Debug formatting of `Angle` uses unicode π.
- `Angle<T>` can only be used with `T: Primitive`.
- Implement `PartialEq<T>` for `Angle<T>`.
- Tidying up of `Angle` arithmetic operations.
- Changed all occurrences of `#[inline(always)]` to `#[inline]`. This helps
  create better test coverage reports.
- `Box2::translate()` is no longer in-place.
- Matrix implementations of `AbsDiffEq` comparisons now forward the comparison
  to the underlying `glam` types.
- Use `splat()` implementations from `glam`.
- Further restricted the `Primitive` supertraits to include `Debug + Display +
  Send + Sync + Sized + 'static`.
- Simplified the trait bounds for `Transform2` and `Transform3`.

## [0.1.1] - 2022-04-04

### Fixed

- Missing trait bounds on `Scalar` and `MatrixN` caused the `serde` feature to
  not compile.

## [0.1.0] - 2022-04-04

Initial release.

[0.18.0]: https://github.com/simonask/glamour/compare/v0.17.0...v0.18.0
[0.17.0]: https://github.com/simonask/glamour/compare/v0.16.0...v0.17.0
[0.16.0]: https://github.com/simonask/glamour/compare/v0.15.0...v0.16.0
[0.15.0]: https://github.com/simonask/glamour/compare/v0.14.1...v0.15.0
[0.14.1]: https://github.com/simonask/glamour/compare/v0.14.0...v0.14.1
[0.14.0]: https://github.com/simonask/glamour/compare/v0.13.0...v0.14.0
[0.13.0]: https://github.com/simonask/glamour/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/simonask/glamour/compare/v0.11.1...v0.12.0
[0.11.1]: https://github.com/simonask/glamour/compare/v0.11.0...v0.11.1
[0.11.0]: https://github.com/simonask/glamour/compare/v0.10.3...v0.11.0
[0.10.3]: https://github.com/simonask/glamour/compare/v0.10.2...v0.10.3
[0.10.2]: https://github.com/simonask/glamour/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/simonask/glamour/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/simonask/glamour/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/simonask/glamour/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/simonask/glamour/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/simonask/glamour/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/simonask/glamour/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/simonask/glamour/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/simonask/glamour/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/simonask/glamour/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/simonask/glamour/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/simonask/glamour/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/simonask/glamour/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/simonask/glamour/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/simonask/glamour/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/simonask/glamour/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/simonask/glamour/releases/tag/v0.1.0

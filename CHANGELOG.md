# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
### Added
- Support for `i64` and `u64` scalar types based on `glam::I64VecN` and `glam::U64VecN`.
- Added `INFINITY` and `NEG_INFINITY` associated constants to vector types, and also added the missing `NAN` associated
  constant to `PointN` and `SizeN`.
- Implemented `Borrow<glam::VecN>` for all vector types, and `Borrow<glamour::VectorN/PointN/SizeN>` for `glam` types.
  This enables the interchangeable use of glamour/glam types in hash maps.

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
  https://github.com/bitshifter/glam-rs/blob/main/CHANGELOG.md#breaking-changes.

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

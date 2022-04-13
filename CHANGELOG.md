# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased
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

[0.2.0]: https://github.com/simonask/glamour/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/simonask/glamour/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/simonask/glamour/releases/tag/v0.1.0

# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Bump Minimum Supported Rust Version to 1.56.0 because we want to use edition
  2021.
- Add CI workflows.
- Add CI/docs badges to README.

## [0.1.1] - 2022-04-04
### Fixed
- Missing trait bounds on `Scalar` and `MatrixN` caused the `serde` feature to
  not compile.


## [0.1.0] - 2022-04-04

Initial release.

[0.1.1]: https://github.com/simonask/glamour/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/simonask/glamour/releases/tag/v0.1.0

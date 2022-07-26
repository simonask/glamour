#!/bin/bash

set -e

# Set of features to build & test.
FEATURE_SETS=(
  # std
  "std"
  "std mint"
  "std serde"
  "std glam_0_20 mint"
  "std glam_0_20 scalar-math"
  # no_std
  "libm"
  "libm mint"
  "libm serde"
  "libm glam_0_20 mint"
  "libm glam_0_20 scalar-math"
)

rustc --version

for features in "${FEATURE_SETS[@]}"
do
   :
   cargo build --tests --no-default-features --features="$features"
   echo cargo test --no-default-features --features=\"$features\"
   cargo test --no-default-features --features="$features"
done

# pushd test_no_std && cargo check
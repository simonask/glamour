# Examples

Table of contents:

- [Transformation with units](#transformation-with-units)
- [Boolean masks](#boolean-masks)

## Transformation with units

```rust
use glamour::prelude::*;

// Define the units and their associated scalars.
// The unit types themselves do not need any trait bounds.

struct ViewSpace;
impl Unit for ViewSpace {
    type Scalar = f32;
}

struct WorldSpace;
impl Unit for WorldSpace {
    type Scalar = f32;
}

// Define type aliases to aid readability and type deduction:

type Point = Point3<ViewSpace>;
type Vector = Vector3<ViewSpace>;

type WorldPoint = Point3<WorldSpace>;
type WorldVector = Vector3<WorldSpace>;

type ViewToWorld = Transform3<ViewSpace, WorldSpace>;

// Some points:

let point = Point { x: 100.0, y: 200.0, z: 300.0 };
let vector = Vector { x: 50.0, y: 0.0, z: 0.0 };

// Dummy transform that multiplies all coordinates by 2.0
let view_to_world = ViewToWorld::from_scale(Vector3::splat(2.0));

// Transform the point and vector from view space to world space.
let world_point = view_to_world.map(point);
let world_vector = view_to_world.map(vector);

assert_eq!(world_point, (200.0, 400.0, 600.0));
assert_eq!(world_vector, (100.0, 0.0, 0.0));
```

## Boolean masks

```rust
use glamour::prelude::*;

type Vector = Vector4<f32>;

let a = Vector { x: 1.0, y: 2.0, z: 100.0, w: 1.5 };
let b = Vector { x: 2.0, y: 1.0, z: 1.0, w: 1.6};

/// Get the component-wise maximums.
let mask = a.cmpge(b);
let c = Vector::select(mask, a, b);
assert_eq!(c, (2.0, 2.0, 100.0, 1.6));
```
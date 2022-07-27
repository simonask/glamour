//! Common convenience traits.

pub mod marker;

/// General trait for the `contains()` method.
///
/// Coordinates exactly on the upper bound are considered to be contained.
///
/// See also: [`Intersection`], which is different in this regard.
pub trait Contains<T> {
    /// Returns `true` if `thing` is inside `self`.
    fn contains(&self, thing: &T) -> bool;
}

/// The `intersection()` method.
///
/// Coordinates exactly on the upper bound are _not_ considered to be
/// intersecting.
///
/// For example, each adjacent tile of a grid of [`crate::Rect`]s or
/// [`crate::Box2`]s will not be considered intersecting, while
/// [`Contains::contains()`] returns true for both tiles with coordinates that
/// fall exactly on the upper bound of one tile.
///
/// See also: [`Contains`], which is different in this regard.
pub trait Intersection<T = Self> {
    /// The type of intersection.
    ///
    /// For example, a rect/rect intersection is another rect, while a box/box
    /// intersection is a box, and a rect/box intersection is a rect, but a
    /// box/rect intersection is a box.
    type Intersection;

    /// True if `thing` intersects with `self`.
    fn intersects(&self, thing: &T) -> bool;

    /// If `thing` intersects with `self`, return the intersection. Otherwise,
    /// returns `None`.
    fn intersection(&self, thing: &T) -> Option<Self::Intersection>;
}

/// The `union()` operation.
pub trait Union<T = Self> {
    /// The type of the union.
    type Union;

    /// Compute the union of two things.
    fn union(self, other: T) -> Self::Union;
}

//! Axis-aligned boxes.

use core::ops::{Add, AddAssign, Sub, SubAssign};

use crate::{
    traits::{Contains, Intersection, Lerp, UnitTypes},
    Point2, Point3, Rect, Size2, Union, Unit, Vector2,
};

/// 2D axis-aligned box represented as "min" and "max" points.
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
pub struct Box2<T: Unit = f32> {
    /// Lower bound of the box.
    pub min: Point2<T>,
    /// Upper bound of the box.
    pub max: Point2<T>,
}

crate::impl_common!(Box2 {
    min: Point2<T>,
    max: Point2<T>
});

crate::impl_as_tuple!(Box2 {
    min: Point2<T>,
    max: Point2<T>
});

/// 3D axis-aligned box.
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
pub struct Box3<T: Unit = f32> {
    /// Lower bound of the box.
    pub min: Point3<T>,
    /// Upper bound of the box.
    pub max: Point3<T>,
}

crate::impl_common!(Box3 {
    min: Point3<T>,
    max: Point3<T>
});

crate::impl_as_tuple!(Box3 {
    min: Point3<T>,
    max: Point3<T>
});

impl<T: Unit> Box2<T> {
    /// Create from [`Rect`].
    ///
    /// Note: This may lose precision due to floating point arithmetic.
    #[inline]
    #[must_use]
    pub fn from_rect(rect: Rect<T>) -> Self {
        rect.into()
    }

    /// Create from origin and size.
    #[inline]
    #[must_use]
    pub fn from_origin_and_size(origin: Point2<T>, size: Size2<T>) -> Self {
        Box2 {
            min: origin,
            max: origin + size.to_vector(),
        }
    }

    /// Create from size at offset zero.
    #[inline]
    #[must_use]
    pub fn from_size(size: Size2<T>) -> Self {
        Box2 {
            min: Point2::zero(),
            max: size.to_vector().to_point(),
        }
    }

    /// Convert to [`Rect`].
    ///
    /// Note: This may lose precision due to floating point arithmetic.
    #[inline]
    #[must_use]
    pub fn to_rect(self) -> Rect<T> {
        self.into()
    }

    /// Create from two points.
    #[inline]
    #[must_use]
    pub fn from_array([min, max]: [Point2<T>; 2]) -> Box2<T> {
        Box2 { min, max }
    }

    /// Convert to `[min, max]`.
    #[inline]
    #[must_use]
    pub fn to_array(self) -> [Point2<T>; 2] {
        [self.min, self.max]
    }

    /// Cast to `&[min, max]`.
    #[inline]
    #[must_use]
    pub fn as_array(&self) -> &[Point2<T>; 2] {
        bytemuck::cast_ref(self)
    }

    /// Cast to `&mut [min, max]`.
    #[inline]
    #[must_use]
    pub fn as_array_mut(&mut self) -> &mut [Point2<T>; 2] {
        bytemuck::cast_mut(self)
    }

    /// Cast to `&[min.x, min.y, max.x, max.y]`.
    #[inline]
    #[must_use]
    pub fn as_scalar_array(&self) -> &[T::Scalar; 4] {
        bytemuck::cast_ref(self)
    }

    /// Cast to `&mut [min.x, min.y, max.x, max.y]`.
    #[inline]
    #[must_use]
    pub fn as_scalar_array_mut(&mut self) -> &mut [T::Scalar; 4] {
        bytemuck::cast_mut(self)
    }

    /// Convert to `[min.x, min.y, max.x, max.y]`.
    #[inline]
    #[must_use]
    pub fn to_scalar_array(self) -> [T::Scalar; 4] {
        bytemuck::cast(self)
    }

    /// Calculate the bounding box that covers all `points`.
    #[must_use]
    pub fn from_points<I>(points: I) -> Self
    where
        I: IntoIterator<Item = Point2<T>>,
    {
        let mut points = points.into_iter();
        let (mut min, mut max) = if let Some(first) = points.next() {
            (first, first)
        } else {
            return Box2::zero();
        };

        for point in points {
            if min.x > point.x {
                min.x = point.x;
            }
            if min.y > point.y {
                min.y = point.y;
            }
            if max.x < point.x {
                max.x = point.x;
            }
            if max.y < point.y {
                max.y = point.y;
            }
        }

        Box2 { min, max }
    }

    /// Corners of the box, clockwise from `min`.
    #[inline]
    #[must_use]
    pub fn corners(&self) -> [Point2<T>; 4] {
        let top_right = (self.max.x, self.min.y).into();
        let bottom_left = (self.min.x, self.max.y).into();
        [self.min, top_right, self.max, bottom_left]
    }

    /// True if the box is negative (i.e., any `min` coordinate is >= any `max`
    /// coordinate).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !(self.max.x > self.min.x && self.max.y > self.min.y)
    }

    /// Returns true when `max.x < min.x || max.y < min.y`.
    #[inline]
    #[must_use]
    pub fn is_negative(&self) -> bool {
        self.max.x < self.min.x || self.max.y < self.min.y
    }

    /// Calculate intersection, returning an invalid (negative) box when there
    /// is no intersection.
    #[inline]
    #[must_use]
    pub fn intersection_unchecked(&self, other: &Self) -> Box2<T> {
        // The intersection is the max() of the min coordinates and the min() of
        // the max coordinates. If the new max.x < min.x or max.y < min.y, there
        // is no intersection.
        let min = self.min.max(other.min);
        let max = self.max.min(other.max);
        Box2 { min, max }
    }

    /// Translate `min` and `max` by vector.
    #[inline(always)]
    pub fn translate(&mut self, by: Vector2<T>) {
        *self += by;
    }

    /// Get the center of the box.
    #[inline]
    #[must_use]
    pub fn center(&self) -> Point2<T> {
        let v = (self.max - self.min) / Vector2::two();
        self.min + v
    }

    /// Get the size of the box (`max - min`).
    #[inline]
    #[must_use]
    pub fn size(&self) -> Size2<T> {
        (self.max - self.min).into()
    }

    /// Get the area covered by the box (shorthand `self.size().area()`).
    #[inline]
    #[must_use]
    pub fn area(&self) -> T::Scalar {
        self.size().area()
    }

    /// Create a zero-sized box.
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Box2::new(Point2::zero(), Point2::zero())
    }
}

impl<T: Unit> From<Box2<T>> for Rect<T> {
    fn from(x: Box2<T>) -> Self {
        let origin = x.min;
        let size = (x.max - x.min).into();
        Rect { origin, size }
    }
}

impl<T: Unit> From<Rect<T>> for Box2<T> {
    fn from(x: Rect<T>) -> Self {
        let min = x.origin;
        let max = x.origin + x.size.to_vector();
        Box2 { min, max }
    }
}

impl<T: Unit> AddAssign<Vector2<T>> for Box2<T> {
    fn add_assign(&mut self, rhs: Vector2<T>) {
        self.min += rhs;
        self.max += rhs;
    }
}

impl<T: Unit> Add<Vector2<T>> for Box2<T> {
    type Output = Self;

    fn add(self, rhs: Vector2<T>) -> Self::Output {
        Box2 {
            min: self.min + rhs,
            max: self.max + rhs,
        }
    }
}

impl<T: Unit> SubAssign<Vector2<T>> for Box2<T> {
    fn sub_assign(&mut self, rhs: Vector2<T>) {
        self.min -= rhs;
        self.max -= rhs;
    }
}

impl<T: Unit> Sub<Vector2<T>> for Box2<T> {
    type Output = Self;

    fn sub(self, rhs: Vector2<T>) -> Self::Output {
        Box2 {
            min: self.min - rhs,
            max: self.max - rhs,
        }
    }
}

impl<T: Unit> Contains<Point2<T>> for Box2<T> {
    #[inline]
    fn contains(&self, thing: &Point2<T>) -> bool {
        thing.x >= self.min.x
            && thing.y >= self.min.y
            && thing.x < self.max.x
            && thing.y < self.max.y
    }
}

impl<T: Unit> Intersection<Point2<T>> for Box2<T> {
    type Intersection = Point2<T>;

    fn intersects(&self, other: &Point2<T>) -> bool {
        self.min.x < other.x && self.max.x > other.x && self.min.y < other.y && self.max.y > other.y
    }

    fn intersection(&self, thing: &Point2<T>) -> Option<Self::Intersection> {
        if self.intersects(thing) {
            Some(*thing)
        } else {
            None
        }
    }
}

impl<T: Unit> Intersection<Box2<T>> for Box2<T> {
    type Intersection = Box2<T>;

    /// Boxes are considered to be intersecting when a corner is entirely inside
    /// the other box. Note that this is different from the implementation of
    /// `Contains`, which returns `true` for a point that is exactly on one of
    /// the `min` coordinates.
    fn intersects(&self, other: &Box2<T>) -> bool {
        // If any of the corners is contained in the other box, there is an
        // intersection.
        self.min.x < other.max.x
            && self.max.x > other.min.x
            && self.min.y < other.max.y
            && self.max.y > other.min.y
    }

    fn intersection(&self, thing: &Box2<T>) -> Option<Self::Intersection> {
        let intersection = self.intersection_unchecked(thing);
        if intersection.is_empty() {
            None
        } else {
            Some(intersection)
        }
    }
}

impl<T: Unit> Union<Box2<T>> for Box2<T> {
    type Union = Box2<T>;

    #[inline]
    #[must_use]
    fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }

        let min = self.min.min(other.min);
        let max = self.max.max(other.max);
        Box2 { min, max }
    }
}

impl<T: UnitTypes> Lerp<T::Primitive> for Box2<T>
where
    Point2<T>: Lerp<T::Primitive>,
{
    fn lerp(self, other: Self, t: T::Primitive) -> Self {
        let min = self.min.lerp(other.min, t);
        let max = self.max.lerp(other.max, t);
        Box2 { min, max }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intersection() {
        type Box2 = super::Box2<f32>;
        let x = Box2::new((10.0, 10.0).into(), (20.0, 20.0).into());

        {
            // No intersection
            let nw = Box2::new((0.0, 0.0).into(), (10.0, 10.0).into());
            let n = Box2::new((10.0, 0.0).into(), (20.0, 10.0).into());
            let ne = Box2::new((20.0, 0.0).into(), (30.0, 10.0).into());
            let e = Box2::new((20.0, 10.0).into(), (30.0, 20.0).into());
            let se = Box2::new((20.0, 20.0).into(), (30.0, 30.0).into());
            let s = Box2::new((10.0, 20.0).into(), (20.0, 30.0).into());
            let sw = Box2::new((0.0, 20.0).into(), (10.0, 30.0).into());
            let w = Box2::new((0.0, 10.0).into(), (10.0, 20.0).into());

            assert!(!nw.intersects(&x));
            assert!(!n.intersects(&x));
            assert!(!ne.intersects(&x));
            assert!(!e.intersects(&x));
            assert!(!x.intersects(&se.min));
            assert!(!x.intersects(&se.max));
            assert!(!se.intersects(&x.min));
            assert!(!se.intersects(&x.max));
            assert!(!se.intersects(&x));
            assert!(!s.intersects(&x));
            assert!(!sw.intersects(&x));
            assert!(!w.intersects(&x));

            assert!(!x.intersects(&nw));
            assert!(!x.intersects(&n));
            assert!(!x.intersects(&ne));
            assert!(!x.intersects(&e));
            assert!(!x.intersects(&se));
            assert!(!x.intersects(&s));
            assert!(!x.intersects(&sw));
            assert!(!x.intersects(&w));

            assert_eq!(x.intersection(&nw), None);
            assert_eq!(x.intersection(&n), None);
            assert_eq!(x.intersection(&ne), None);
            assert_eq!(x.intersection(&e), None);
            assert_eq!(x.intersection(&se), None);
            assert_eq!(x.intersection(&s), None);
            assert_eq!(x.intersection(&sw), None);
            assert_eq!(x.intersection(&w), None);

            assert_eq!(nw.intersection(&x), None);
            assert_eq!(n.intersection(&x), None);
            assert_eq!(ne.intersection(&x), None);
            assert_eq!(e.intersection(&x), None);
            assert_eq!(se.intersection(&x), None);
            assert_eq!(s.intersection(&x), None);
            assert_eq!(sw.intersection(&x), None);
            assert_eq!(w.intersection(&x), None);
        }

        {
            // Intersections
            let nw = Box2::new((0.0, 0.0).into(), (11.0, 11.0).into());
            let n = Box2::new((11.0, 0.0).into(), (19.0, 11.0).into());
            let ne = Box2::new((19.0, 0.0).into(), (29.0, 11.0).into());
            let e = Box2::new((19.0, 11.0).into(), (30.0, 29.0).into());
            let se = Box2::new((19.0, 19.0).into(), (30.0, 30.0).into());
            let s = Box2::new((11.0, 19.0).into(), (19.0, 30.0).into());
            let sw = Box2::new((0.0, 19.0).into(), (11.0, 30.0).into());
            let w = Box2::new((0.0, 11.0).into(), (11.0, 19.0).into());

            assert!(nw.intersects(&x));
            assert!(n.intersects(&x));
            assert!(ne.intersects(&x));
            assert!(e.intersects(&x));
            assert!(se.intersects(&x));
            assert!(s.intersects(&x));
            assert!(sw.intersects(&x));
            assert!(w.intersects(&x));

            assert!(x.intersects(&nw));
            assert!(x.intersects(&n));
            assert!(x.intersects(&ne));
            assert!(x.intersects(&e));
            assert!(x.intersects(&se));
            assert!(x.intersects(&s));
            assert!(x.intersects(&sw));
            assert!(x.intersects(&w));

            assert_eq!(x.intersection(&nw), nw.intersection(&x));
            assert_eq!(x.intersection(&n), n.intersection(&x));
            assert_eq!(x.intersection(&ne), ne.intersection(&x));
            assert_eq!(x.intersection(&e), e.intersection(&x));
            assert_eq!(x.intersection(&se), se.intersection(&x));
            assert_eq!(x.intersection(&s), s.intersection(&x));
            assert_eq!(x.intersection(&sw), sw.intersection(&x));
            assert_eq!(x.intersection(&w), w.intersection(&x));
        }
    }
}

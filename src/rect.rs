//! 2D axis-aligned rectangles

use crate::{
    traits::{Contains, Intersection, Lerp, SimdVecFloat, Union, UnitTypes},
    Box2, Point2, Scalar, Size2, Unit, Vector2,
};

/// 2D axis-aligned rectangle represented as "origin" and "size".
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
#[repr(C)]
pub struct Rect<T: Unit = f32> {
    /// Lower bound of the rect.
    pub origin: Point2<T>,
    /// Size of the rect.
    pub size: Size2<T>,
}

crate::impl_common!(Rect {
    origin: Point2<T>,
    size: Size2<T>
});
crate::impl_as_tuple!(Rect {
    origin: Point2<T>,
    size: Size2<T>
});

impl<T: UnitTypes> Rect<T> {
    /// Zero rect (origin = 0.0, size = 0.0).
    #[inline]
    #[must_use]
    pub fn zero() -> Self {
        Rect {
            origin: Point2::zero(),
            size: Size2::zero(),
        }
    }

    /// Rect at (0.0, 0.0) with `size`.
    #[inline]
    #[must_use]
    pub fn from_size(size: Size2<T>) -> Self {
        Rect {
            origin: Point2::zero(),
            size,
        }
    }

    /// Create from [`Box2`].
    ///
    /// Note: This may lose precision due to floating point arithmetic.
    #[inline]
    #[must_use]
    pub fn from_box(b: Box2<T>) -> Self {
        Self::from_min_max(b.min, b.max)
    }

    /// Create from min/max points.
    #[inline]
    #[must_use]
    pub fn from_min_max(min: Point2<T>, max: Point2<T>) -> Self {
        let size = Size2::from_vector(max - min);
        let origin = min;
        Rect { origin, size }
    }

    /// Calculate the bounding rect that covers all `points`.
    #[must_use]
    pub fn from_points<I>(points: I) -> Self
    where
        I: IntoIterator<Item = Point2<T>>,
    {
        Box2::from_points(points).into()
    }

    /// Get rect lower bound (`origin`).
    #[inline]
    #[must_use]
    pub fn min(&self) -> Point2<T> {
        self.origin
    }

    /// Get rect upper bound (`origin + size`).
    #[inline]
    #[must_use]
    pub fn max(&self) -> Point2<T> {
        self.origin + self.size.to_vector()
    }

    /// Width of the rectangle.
    #[inline]
    #[must_use]
    pub fn width(&self) -> T::Scalar {
        self.size.width
    }

    /// Height of the rectangle.
    #[inline]
    #[must_use]
    pub fn height(&self) -> T::Scalar {
        self.size.height
    }

    /// Range of X coordinates covered by this rectangle.
    #[inline]
    #[must_use]
    pub fn x_range(&self) -> core::ops::Range<T::Scalar> {
        let (min, max) = (self.min(), self.max());
        min.x..max.x
    }

    /// Range of Y coordinates covered by this rectangle.
    #[inline]
    #[must_use]
    pub fn y_range(&self) -> core::ops::Range<T::Scalar> {
        let (min, max) = (self.min(), self.max());
        min.y..max.y
    }

    /// Corners of the rectangle, clockwise from top left.
    #[must_use]
    pub fn corners(&self) -> [Point2<T>; 4] {
        let (min, max) = (self.origin, self.max());
        let top_left = min;
        let top_right = (max.x, min.y).into();
        let bottom_right = max;
        let bottom_left = (min.x, max.y).into();
        [top_left, top_right, bottom_right, bottom_left]
    }

    /// Get the point at the center of the rect.
    #[inline(always)]
    #[must_use]
    pub fn center(&self) -> Point2<T> {
        self.origin + self.size.to_vector() / Vector2::two()
    }

    /// Translate a copy of the rect by vector.
    #[must_use]
    pub fn translate(self, by: Vector2<T>) -> Self {
        Rect {
            origin: self.origin.translate(by),
            size: self.size,
        }
    }

    /// Increase the size of the rect by moving the origin back by the size, and
    /// increasing the size of the rectangle by twice each axis.
    #[must_use]
    pub fn inflate(&self, by: Size2<T>) -> Self {
        Rect {
            origin: self.origin - by.to_vector(),
            size: self.size + by + by,
        }
    }

    /// Convert to [`Box2`].
    #[inline]
    #[must_use]
    pub fn to_box2(&self) -> Box2<T> {
        (*self).into()
    }

    /// Get the area of the rect (equivalent to `self.size.area()`).
    #[inline]
    #[must_use]
    pub fn area(&self) -> T::Scalar {
        self.size.area()
    }

    /// True if size is zero.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size == Size2::zero()
    }

    /// True if the rect only contains finite and non-NaN coordinates.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool
    where
        T::Vec2: SimdVecFloat<2, Scalar = T::Primitive>,
    {
        self.origin.is_finite() && self.size.is_finite()
    }

    /// Round all coordinates.
    #[inline]
    #[must_use]
    pub fn round(self) -> Self
    where
        T::Vec2: SimdVecFloat<2, Scalar = T::Primitive>,
    {
        Rect {
            origin: self.origin.round(),
            size: self.size.round(),
        }
    }

    /// Round all coordinates towards the center of the rect.
    #[inline]
    #[must_use]
    pub fn round_in(self) -> Self
    where
        T::Vec2: SimdVecFloat<2, Scalar = T::Primitive>,
    {
        Rect {
            origin: self.origin.ceil(),
            size: self.size.floor(),
        }
    }

    /// Round all coordinates away from the center of the rect.
    #[inline]
    #[must_use]
    pub fn round_out(self) -> Self
    where
        T::Vec2: SimdVecFloat<2, Scalar = T::Primitive>,
    {
        Rect {
            origin: self.origin.floor(),
            size: self.size.ceil(),
        }
    }
}

impl<T: Unit> Contains<Point2<T>> for Rect<T> {
    #[inline]
    #[must_use]
    fn contains(&self, point: &Point2<T>) -> bool {
        let max = self.max();
        point.x >= self.origin.x && point.y >= self.origin.y && point.x < max.x && point.y < max.y
    }
}

impl<T: Unit> Intersection<Rect<T>> for Rect<T> {
    type Intersection = Rect<T>;

    #[inline]
    #[must_use]
    fn intersects(&self, thing: &Rect<T>) -> bool {
        self.intersects(&thing.to_box2())
    }

    #[inline]
    #[must_use]
    fn intersection(&self, thing: &Rect<T>) -> Option<Self::Intersection> {
        self.intersection(&thing.to_box2())
    }
}

impl<T: Unit> Intersection<Box2<T>> for Rect<T> {
    type Intersection = Rect<T>;

    #[inline]
    #[must_use]
    fn intersects(&self, thing: &Box2<T>) -> bool {
        self.to_box2().intersects(thing)
    }

    #[inline]
    #[must_use]
    fn intersection(&self, thing: &Box2<T>) -> Option<Self::Intersection> {
        self.to_box2().intersection(thing).map(Into::into)
    }
}

impl<T: Unit> Union<Rect<T>> for Rect<T> {
    type Union = Rect<T>;

    #[inline]
    #[must_use]
    fn union(self, other: Self) -> Self {
        if self.is_empty() {
            return other;
        }
        if other.is_empty() {
            return self;
        }

        let origin = self.origin.min(other.origin);
        let max = (&self).max().max((&other).max());
        let size = (max - origin).to_size();

        Rect { origin, size }
    }
}

impl<T: Unit> Lerp<<T::Scalar as Scalar>::Primitive> for Rect<T>
where
    Point2<T>: Lerp<<T::Scalar as Scalar>::Primitive>,
    Size2<T>: Lerp<<T::Scalar as Scalar>::Primitive>,
{
    #[inline]
    #[must_use]
    fn lerp(self, end: Self, t: <T::Scalar as Scalar>::Primitive) -> Self {
        let origin = self.origin.lerp(end.origin, t);
        let size = self.size.lerp(end.size, t);
        Rect { origin, size }
    }
}

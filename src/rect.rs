//! 2D axis-aligned rectangles

use approx::AbsDiffEq;
use num_traits::ConstZero;

use crate::{
    traits::{Contains, Intersection, Union},
    unit::FloatUnit,
    Box2, Point2, Size2, Unit, Vector2,
};

/// 2D axis-aligned rectangle represented as "origin" and "size".
#[repr(C)]
pub struct Rect<T: Unit = f32> {
    /// Lower bound of the rect.
    pub origin: Point2<T>,
    /// Size of the rect.
    pub size: Size2<T>,
}

/// SAFETY: All members are `Pod`, and we are `#[repr(C)]`
unsafe impl<T: Unit> bytemuck::Pod for Rect<T> {}
/// SAFETY: All members are `Pod`, and we are `#[repr(C)]`
unsafe impl<T: Unit> bytemuck::Zeroable for Rect<T> {}

crate::derive_standard_traits!(Rect {
    origin: Point2<T>,
    size: Size2<T>
});
crate::derive_tuple_conversion_traits!(Rect {
    origin: Point2<T>,
    size: Size2<T>
});

impl<T: Unit> Rect<T> {
    /// Zero rect (origin = 0.0, size = 0.0).
    pub const ZERO: Self = Rect {
        origin: Point2::ZERO,
        size: Size2::ZERO,
    };

    /// New Rect from origin/size.
    pub fn new(origin: impl Into<Point2<T>>, size: impl Into<Size2<T>>) -> Rect<T> {
        Rect {
            origin: origin.into(),
            size: size.into(),
        }
    }

    crate::casting_interface!(Rect {
        origin: Point2<T>,
        size: Size2<T>
    });
    crate::tuple_interface!(Rect {
        origin: Point2<T>,
        size: Size2<T>
    });

    /// Rect at (0.0, 0.0) with `size`.
    #[inline]
    #[must_use]
    pub const fn from_size(size: Size2<T>) -> Self {
        Rect {
            origin: Point2::ZERO,
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
    pub const fn min(&self) -> Point2<T> {
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
    pub const fn width(&self) -> T::Scalar {
        self.size.width
    }

    /// Height of the rectangle.
    #[inline]
    #[must_use]
    pub const fn height(&self) -> T::Scalar {
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
    #[inline]
    #[must_use]
    pub fn center(&self) -> Point2<T> {
        self.origin + self.size.to_vector() / (Vector2::ONE + Vector2::ONE)
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
    /// increasing the size of the rectangle by `2 * by`.
    ///
    /// #### Example
    ///
    /// ```rust
    /// # use glamour::prelude::*;
    /// let r = Rect::<f32>::new((10.0, 10.0), (10.0, 10.0));
    /// let r = r.inflate((10.0, 10.0).into());
    /// assert_eq!(r.origin, (0.0, 0.0));
    /// assert_eq!(r.size, (30.0, 30.0));
    /// ```
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

    /// True if size is zero or negative or NaN.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.size.is_empty()
    }

    /// True if size is negative or NaN.
    #[inline]
    #[must_use]
    pub fn is_negative(&self) -> bool {
        !(self.size.width >= T::Scalar::ZERO && self.size.height >= T::Scalar::ZERO)
    }
}

impl<T> Rect<T>
where
    T: FloatUnit,
{
    /// True if the rect only contains finite and non-NaN coordinates.
    #[inline]
    #[must_use]
    pub fn is_finite(&self) -> bool {
        self.origin.is_finite() && self.size.is_finite()
    }

    /// Round all coordinates.
    ///
    /// Note: This may create an empty rect from a non-empty rect.
    ///
    /// #### Example
    ///
    /// ```rust
    /// # use glamour::prelude::*;
    /// let r = Rect::<f32>::new((0.51, 0.49), (0.51, 0.49));
    /// let r = r.round();
    /// assert_eq!(r, Rect::<f32>::new((1.0, 0.0), (1.0, 0.0)));
    /// ```
    #[inline]
    #[must_use]
    pub fn round(self) -> Self {
        Rect {
            origin: self.origin.round(),
            size: self.size.round(),
        }
    }

    /// Round all coordinates towards the center of the rect.
    ///
    /// This function needs to convert the rect to a [`Box2`] before rounding,
    /// which loses both performance and precision. Use [`Box2`] if you need to
    /// perform this operation frequently.
    ///
    /// Note: This may create an empty rect from a non-empty rect.
    ///
    /// #### Example
    ///
    /// ```rust
    /// # use glamour::prelude::*;
    /// let r = Rect::<f32>::new((0.51, 0.49), (1.51, 1.49));
    /// let r = r.round_in();
    /// assert_eq!(r, Rect::<f32>::new((1.0, 1.0), (1.0, 0.0)));
    /// ```
    #[inline]
    #[must_use]
    pub fn round_in(self) -> Self {
        self.to_box2().round_in().to_rect()
    }

    /// Round all coordinates away from the center of the rect.
    ///
    /// This function needs to convert the rect to a [`Box2`] before rounding,
    /// which loses both performance and precision. Use [`Box2`] if you need to
    /// perform this operation frequently.
    ///
    /// Note: As opposed to [`Rect::round()`] and [`Rect::round_in()`], this
    /// will not create an empty rect from a non-empty rect.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let r = Rect::<f32>::new((0.51, 0.49), (1.51, 1.49));
    /// let r = r.round_out();
    /// assert_eq!(r, Rect::new((0.0, 0.0), (3.0, 2.0)));
    /// ```
    #[inline]
    #[must_use]
    pub fn round_out(self) -> Self {
        self.to_box2().round_out().to_rect()
    }

    /// Linear interpolation between two rects.
    #[inline]
    #[must_use]
    pub fn lerp(self, other: Self, t: T::Scalar) -> Self {
        Rect {
            origin: self.origin.lerp(other.origin, t),
            size: self.size.lerp(other.size, t),
        }
    }
}

impl<T> AbsDiffEq<(Point2<T>, Size2<T>)> for Rect<T>
where
    T: Unit<Scalar: AbsDiffEq<Epsilon: Copy>>,
{
    type Epsilon = <T::Scalar as approx::AbsDiffEq>::Epsilon;

    #[must_use]
    fn default_epsilon() -> Self::Epsilon {
        T::Scalar::default_epsilon()
    }

    #[must_use]
    fn abs_diff_eq(&self, other: &(Point2<T>, Size2<T>), epsilon: Self::Epsilon) -> bool {
        self.abs_diff_eq(&Self::from_tuple(*other), epsilon)
    }

    #[must_use]
    fn abs_diff_ne(&self, other: &(Point2<T>, Size2<T>), epsilon: Self::Epsilon) -> bool {
        self.abs_diff_ne(&Self::from_tuple(*other), epsilon)
    }
}

impl<T> approx::RelativeEq<(Point2<T>, Size2<T>)> for Rect<T>
where
    T: Unit<Scalar: approx::RelativeEq<Epsilon: Copy>>,
{
    #[must_use]
    fn default_max_relative() -> Self::Epsilon {
        T::Scalar::default_max_relative()
    }

    #[must_use]
    fn relative_eq(
        &self,
        other: &(Point2<T>, Size2<T>),
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.relative_eq(&Self::from_tuple(*other), epsilon, max_relative)
    }

    #[must_use]
    fn relative_ne(
        &self,
        other: &(Point2<T>, Size2<T>),
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.relative_ne(&Self::from_tuple(*other), epsilon, max_relative)
    }
}

impl<T: Unit> approx::UlpsEq<(Point2<T>, Size2<T>)> for Rect<T>
where
    T::Scalar: approx::UlpsEq,
    <T::Scalar as approx::AbsDiffEq>::Epsilon: Copy,
{
    #[must_use]
    fn default_max_ulps() -> u32 {
        T::Scalar::default_max_ulps()
    }

    #[must_use]
    fn ulps_eq(
        &self,
        other: &(Point2<T>, Size2<T>),
        epsilon: Self::Epsilon,
        max_ulps: u32,
    ) -> bool {
        self.ulps_eq(&Self::from_tuple(*other), epsilon, max_ulps)
    }

    #[must_use]
    fn ulps_ne(
        &self,
        other: &(Point2<T>, Size2<T>),
        epsilon: Self::Epsilon,
        max_ulps: u32,
    ) -> bool {
        self.ulps_ne(&Self::from_tuple(*other), epsilon, max_ulps)
    }
}

impl<T: Unit> Contains<Point2<T>> for Rect<T> {
    #[inline]
    #[must_use]
    fn contains(&self, point: &Point2<T>) -> bool {
        // Try to avoid losing too much precision.
        let p2 = *point - self.origin;
        point.x >= self.origin.x
            && point.y >= self.origin.y
            && p2.x <= self.size.width
            && p2.y <= self.size.height
    }
}

impl<T: Unit> Intersection<Point2<T>> for Rect<T> {
    type Intersection = Point2<T>;

    fn intersects(&self, thing: &Point2<T>) -> bool {
        let diff = *thing - self.origin;
        diff.x >= T::Scalar::ZERO
            && diff.y >= T::Scalar::ZERO
            && diff.x < self.size.width
            && diff.y < self.size.height
    }

    fn intersection(&self, thing: &Point2<T>) -> Option<Self::Intersection> {
        if self.intersects(thing) {
            Some(*thing)
        } else {
            None
        }
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

impl<T> Union<Rect<T>> for Rect<T>
where
    T: Unit,
{
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
        let max = self.max().max(other.max());
        let size = (max - origin).to_size();

        Rect { origin, size }
    }
}

#[cfg(test)]
mod tests {
    use approx::{
        assert_abs_diff_eq, assert_abs_diff_ne, assert_relative_eq, assert_relative_ne,
        assert_ulps_eq, assert_ulps_ne,
    };

    type Rect = super::Rect<i32>;
    type RectF = super::Rect<f32>;
    type Size = super::Size2<i32>;
    type Box2 = super::Box2<i32>;
    type Point = super::Point2<i32>;
    type PointF = super::Point2<f32>;

    #[test]
    fn basics() {
        let zero = Rect::ZERO;
        assert_eq!(zero.origin, (0, 0));
        assert_eq!(zero.size, (0, 0));

        let from_size = Rect::from_size(Size {
            width: 100,
            height: 200,
        });
        assert_eq!(from_size.origin, (0, 0));
        assert_eq!(from_size.size, (100, 200));
        assert_eq!(from_size.min(), from_size.origin);
        assert_eq!(from_size.max(), (100, 200));
        assert_eq!(from_size.width(), 100);
        assert_eq!(from_size.height(), 200);

        let (_origin, _size) = from_size.into();
    }

    #[test]
    fn equality() {
        use crate::{Point2, Size2};
        let a = RectF::new(Point2::new(0.0, 0.0), Size2::new(1.0, 1.0));
        let b = a.translate((1.0, 1.0).into());

        assert_abs_diff_eq!(a, a);
        assert_relative_eq!(a, a);
        assert_ulps_eq!(a, a);
        assert_abs_diff_ne!(a, b);
        assert_relative_ne!(a, b);
        assert_ulps_ne!(a, b);

        assert_abs_diff_eq!(a, a.to_tuple());
        assert_relative_eq!(a, a.to_tuple());
        assert_ulps_eq!(a, a.to_tuple());
        assert_abs_diff_ne!(a, b.to_tuple());
        assert_relative_ne!(a, b.to_tuple());
        assert_ulps_ne!(a, b.to_tuple());
    }

    #[test]
    fn from_box() {
        let b = Box2::new((100, 200), (300, 400));
        let r = Rect::from_box(b);
        assert_eq!(r.origin, (100, 200));
        assert_eq!(r.size, (200, 200));

        let r2 = Rect::from_min_max((100, 200).into(), (300, 400).into());
        assert_eq!(r2, r);
    }

    #[test]
    fn from_points() {
        let points: [Point; 1] = [(100, 100)].map(Point::from);

        let r = Rect::from_points(points);
        assert_eq!(r, Rect::new((100, 100), (0, 0)));

        let points: [Point; 10] = [
            (-10, 10),
            (10, -10),
            (200, -2),
            (1, 1),
            (2, 2),
            (3, 4),
            (5, 6),
            (250, -11),
            (1, 1),
            (1, 1),
        ]
        .map(Point::from);

        let r = Rect::from_points(points);
        assert_eq!(r.origin, (-10, -11));
        assert_eq!(r.size, (260, 21));
    }

    #[test]
    fn xy_range() {
        let rect = Rect::new((10, 10), (10, 10));

        let x_range = rect.x_range().into_iter();
        assert!(x_range.eq([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]));
        let y_range = rect.y_range().into_iter();
        assert!(y_range.eq([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]));
    }

    #[test]
    fn corners() {
        let r = Rect::new((10, 11), (12, 13));
        let [nw, ne, se, sw] = r.corners();
        assert_eq!(nw, (10, 11));
        assert_eq!(ne, (22, 11));
        assert_eq!(se, (22, 24));
        assert_eq!(sw, (10, 24));
    }

    #[test]
    fn center() {
        let r = Rect::from(((10, 19).into(), (100, 200).into()));
        assert_eq!(r.center(), (60, 119));

        let r = Rect::from(((-10, -10).into(), (20, 20).into()));
        assert_eq!(r.center(), (0, 0));

        let r = Rect::from(((0, 0).into(), (-1, -1).into()));
        assert_eq!(r.center(), (0, 0));

        let r = Rect::from(((0, 0).into(), (-2, -2).into()));
        assert_eq!(r.center(), (-1, -1));
    }

    #[test]
    fn translate() {
        let r = Rect::from(((10, 20).into(), (11, 12).into()));
        let r = r.translate(crate::Vector2::new(2, 3));
        assert_eq!(r.origin, (12, 23));
        assert_eq!(r.size, (11, 12));
    }

    #[test]
    fn negative_empty() {
        let r = Rect::from_size((-1, -1).into());
        assert!(r.is_empty());
        assert!(r.is_negative());

        let r = Rect::from_size((0, 0).into());
        assert!(r.is_empty());
        assert!(!r.is_negative());

        let r = Rect::from_size((1, 1).into());
        assert!(!r.is_empty());
        assert!(!r.is_negative());

        // Negative zero.

        let r = RectF::from_size((-0.0, 0.0).into());
        assert!(r.is_empty());
        assert!(!r.is_negative());

        let r = RectF::new((1.0, 1.0), (-0.0, -0.0));
        assert!(r.is_empty());
        assert!(!r.is_negative());

        // NaN

        let r = RectF::from_size((core::f32::NAN, core::f32::NAN).into());
        assert!(r.is_empty());
        assert!(r.is_negative());

        let r = RectF::new((core::f32::NAN, 1.0), (1.0, 1.0));
        assert!(!r.is_empty());
        assert!(!r.is_negative());
    }

    #[test]
    fn contains() {
        use super::Contains;

        let r: RectF = RectF::new((10.0, 10.0), (10.0, 10.0));
        assert!(r.contains(&PointF::new(10.0, 10.0)));
        assert!(r.contains(&PointF::new(20.0, 20.0)));
        assert!(!r.contains(&PointF::new(10.0, 9.999999)));
        assert!(!r.contains(&PointF::new(9.999999, 10.0)));
    }

    #[test]
    fn intersection() {
        use super::Intersection;

        let r = RectF::new((10.0, 10.0), (10.0, 10.0));
        assert!(r.intersects(&PointF::new(10.0, 10.0)));
        assert!(!r.intersects(&PointF::new(20.0, 20.0)));
        assert!(!r.intersects(&PointF::new(10.0, 9.999999)));
        assert!(!r.intersects(&PointF::new(9.999999, 10.0)));

        assert_eq!(
            r.intersection(&PointF::new(10.0, 10.0)),
            Some(PointF::new(10.0, 10.0))
        );
        assert_eq!(r.intersection(&PointF::new(20.0, 20.0)), None);
        assert_eq!(r.intersection(&PointF::new(10.0, 9.999999)), None);
        assert_eq!(r.intersection(&PointF::new(9.999999, 10.0)), None);

        // r2 covers all of r.
        let r2 = RectF::new((5.0, 5.0), (15.0, 15.0));
        assert!(r2.intersects(&r));
        assert_eq!(
            r2.intersection(&r),
            Some(RectF {
                origin: (10.0, 10.0).into(),
                size: (10.0, 10.0).into(),
            })
        );

        // r2 covers the lower bound of r.
        let r2 = RectF::new((5.0, 5.0), (10.0, 10.0));
        assert!(r2.intersects(&r));
        assert_eq!(
            r2.intersection(&r),
            Some(RectF {
                origin: (10.0, 10.0).into(),
                size: (5.0, 5.0).into(),
            })
        )
    }

    #[test]
    fn union() {
        use super::Union;

        let r = RectF::new((10.0, 10.0), (10.0, 10.0));

        assert_eq!(r.union(RectF::ZERO), r);
        assert_eq!(RectF::ZERO.union(r), r);

        assert_eq!(r.union(r), r);
        assert_eq!(
            r.union(RectF {
                origin: (0.0, 0.0).into(),
                size: (30.0, 30.0).into(),
            }),
            RectF {
                origin: (0.0, 0.0).into(),
                size: (30.0, 30.0).into(),
            }
        );
        assert_eq!(
            r.union(RectF {
                origin: (-1.0, -1.0).into(),
                size: (2.0, 2.0).into(),
            }),
            RectF {
                origin: (-1.0, -1.0).into(),
                size: (21.0, 21.0).into(),
            }
        );
    }

    #[test]
    fn lerp() {
        let src = RectF::new((0.0, 0.0), (1.0, 1.0));
        let dst = RectF::new((1.0, 1.0), (2.0, 2.0));
        assert_eq!(
            src.lerp(dst, 0.5),
            RectF {
                origin: (0.5, 0.5).into(),
                size: (1.5, 1.5).into(),
            }
        )
    }

    #[test]
    fn area() {
        let r = RectF::new((-1.0, -1.0), (10.0, 10.0));
        assert_eq!(r.area(), 100.0);
        assert_eq!(r.size.area(), r.area());
    }

    #[test]
    fn is_finite() {
        assert!(RectF::ZERO.is_finite());
        assert!(RectF::new(super::Point2::ZERO, super::Size2::ONE).is_finite());

        let r = RectF::new((-1.0, -1.0), (10.0, f32::NAN));
        assert!(!r.is_finite());

        let r = RectF::new((-1.0, f32::NAN), (10.0, 10.0));
        assert!(!r.is_finite());
    }

    #[test]
    fn to_tuple() {
        let r = Rect::from_tuple((Point::new(10, 10), Size::new(20, 20)));
        assert_eq!(r.origin, (10, 10));
        assert_eq!(r.size, (20, 20));

        let r = Rect::new(
            crate::Point2 { x: 10, y: 10 },
            crate::Size2 {
                width: 20,
                height: 20,
            },
        );
        let (origin, size) = r.to_tuple();
        assert_eq!(origin, (10, 10));
        assert_eq!(size, (20, 20));
    }

    #[test]
    fn round_out() {
        let rect = RectF::new((0.8, 0.8), (0.1, 0.1)).round_out();
        assert_eq!(rect.origin, (0.0, 0.0));
        assert_eq!(rect.size, (1.0, 1.0));

        let rect = RectF::new((0.8, 0.8), (0.3, 0.3)).round_out();
        assert_eq!(rect.origin, (0.0, 0.0));
        assert_eq!(rect.size, (2.0, 2.0));

        let rect = RectF::new((0.1, 0.1), (0.3, 0.3)).round_out();
        assert_eq!(rect.origin, (0.0, 0.0));
        assert_eq!(rect.size, (1.0, 1.0));
    }
}

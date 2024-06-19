//! Axis-aligned boxes.

use core::ops::{Add, AddAssign, Sub, SubAssign};

use crate::{
    rewrap,
    traits::{Contains, Intersection},
    unit::FloatUnit,
    IntUnit, Point2, Point3, Rect, Size2, Union, Unit, Vector2,
};

/// 2D axis-aligned box represented as "min" and "max" points.
#[repr(C)]
pub struct Box2<T: Unit = f32> {
    /// Lower bound of the box.
    pub min: Point2<T>,
    /// Upper bound of the box.
    pub max: Point2<T>,
}

/// SAFETY: All members are `Pod`, and we are `#[repr(C)]`.
unsafe impl<T: Unit> bytemuck::Pod for Box2<T> {}
/// SAFETY: All members are `Pod`, and we are `#[repr(C)]`.
unsafe impl<T: Unit> bytemuck::Zeroable for Box2<T> {}

/// 3D axis-aligned box.
#[repr(C)]
pub struct Box3<T: Unit = f32> {
    /// Lower bound of the box.
    pub min: Point3<T>,
    /// Upper bound of the box.
    pub max: Point3<T>,
}

/// SAFETY: All members are `Pod`, and we are `#[repr(C)]`.
unsafe impl<T: Unit> bytemuck::Pod for Box3<T> {}
/// SAFETY: All members are `Pod`, and we are `#[repr(C)]`.
unsafe impl<T: Unit> bytemuck::Zeroable for Box3<T> {}

impl<T: Unit> Clone for Box2<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Unit> Clone for Box3<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: Unit> Copy for Box2<T> {}
impl<T: Unit> Copy for Box3<T> {}

impl<T: Unit> Default for Box2<T> {
    fn default() -> Self {
        Box2::ZERO
    }
}
impl<T: Unit> Default for Box3<T> {
    fn default() -> Self {
        Box3::ZERO
    }
}

impl<T: Unit> Box2<T> {
    /// Zero-sized box.
    pub const ZERO: Self = Self {
        min: Point2::ZERO,
        max: Point2::ZERO,
    };

    /// New 2D box from min/max coordinates.
    pub fn new(min: Point2<T>, max: Point2<T>) -> Self {
        Box2 { min, max }
    }

    /// New 2D box from type-inferred min and max.
    pub fn from_min_max(min: impl Into<Point2<T>>, max: impl Into<Point2<T>>) -> Self {
        Self::new(min.into(), max.into())
    }

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
            min: Point2::ZERO,
            max: rewrap(size),
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
            return Box2::ZERO;
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

    /// True if the box is zero or negative (i.e., any `min` coordinate is >= any `max`
    /// coordinate).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        !(self.max.x > self.min.x && self.max.y > self.min.y)
    }

    /// True when `max.x < min.x || max.y < min.y`.
    #[inline]
    #[must_use]
    pub fn is_negative(&self) -> bool {
        !(self.max.x >= self.min.x && self.max.y >= self.min.y)
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
    #[inline]
    #[must_use]
    pub fn translate(self, by: Vector2<T>) -> Self {
        self + by
    }

    /// Get the center of the box.
    #[inline]
    #[must_use]
    pub fn center(&self) -> Point2<T> {
        let v = (self.max - self.min) / (Vector2::ONE + Vector2::ONE);
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
}

impl<T: Unit> Box3<T> {
    /// Zero-sized box.
    pub const ZERO: Self = Self {
        min: Point3::ZERO,
        max: Point3::ZERO,
    };

    /// New 2D box from min/max coordinates.
    pub fn new(min: impl Into<Point3<T>>, max: impl Into<Point3<T>>) -> Self {
        Box3 {
            min: min.into(),
            max: max.into(),
        }
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
            && thing.x <= self.max.x
            && thing.y <= self.max.y
    }
}

impl<T: Unit> Intersection<Point2<T>> for Box2<T> {
    type Intersection = Point2<T>;

    fn intersects(&self, other: &Point2<T>) -> bool {
        other.x >= self.min.x
            && other.y >= self.min.y
            && other.x < self.max.x
            && other.y < self.max.y
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
    /// the `max` coordinates.
    fn intersects(&self, other: &Box2<T>) -> bool {
        !self.intersection_unchecked(other).is_empty()
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

impl<T: FloatUnit> Box2<T> {
    /// Round coordinates to the nearest integer.
    ///
    /// Note: This function makes no attempt to avoid creating "degenerate"
    /// boxes, where `min >= max`.
    ///
    /// ### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let b = Box2::<f32>::new(point!(0.3, 0.3), point!(2.7, 2.7));
    /// let rounded = b.round();
    /// assert_eq!(rounded.min, point!(0.0, 0.0));
    /// assert_eq!(rounded.max, point!(3.0, 3.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn round(self) -> Self {
        Box2 {
            min: self.min.round(),
            max: self.max.round(),
        }
    }

    /// Round towards from the center of the box.
    ///
    /// Note: This function makes no attempt to avoid creating "degenerate"
    /// boxes, where `min >= max`.
    ///
    /// ### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let b = Box2::<f32>::new(point!(0.3, 0.3), point!(2.7, 2.7));
    /// let rounded = b.round_in();
    /// assert_eq!(rounded.min, point!(1.0, 1.0));
    /// assert_eq!(rounded.max, point!(2.0, 2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn round_in(self) -> Self {
        Box2 {
            min: self.min.ceil(),
            max: self.max.floor(),
        }
    }

    /// Round away from the center of the box.
    ///
    /// Note: This function can only create "degenerate" boxes (where min >=
    /// max) if the box was already degenerate.
    ///
    /// ### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let b = Box2::<f32>::new(point!(0.7, 0.7), point!(1.4, 1.4));
    /// let rounded = b.round_out();
    /// assert_eq!(rounded.min, point!(0.0, 0.0));
    /// assert_eq!(rounded.max, point!(2.0, 2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn round_out(self) -> Self {
        Box2 {
            min: self.min.floor(),
            max: self.max.ceil(),
        }
    }

    /// Linear interpolation between boxes.
    #[must_use]
    pub fn lerp(self, other: Self, t: T::Scalar) -> Self {
        let min = self.min.lerp(other.min, t);
        let max = self.max.lerp(other.max, t);
        Box2 { min, max }
    }
}

impl<T: FloatUnit> Box3<T> {
    /// Round coordinates to the nearest integer.
    ///
    /// Note: This function makes no attempt to avoid creating "degenerate"
    /// boxes, where `min >= max`.
    ///
    /// ### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let b = Box3::<f32>::new(point!(0.3, 0.3, 0.3), point!(2.7, 2.7, 2.7));
    /// let rounded = b.round();
    /// assert_eq!(rounded.min, point!(0.0, 0.0, 0.0));
    /// assert_eq!(rounded.max, point!(3.0, 3.0, 3.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn round(self) -> Self {
        Box3 {
            min: self.min.round(),
            max: self.max.round(),
        }
    }

    /// Round towards from the center of the box.
    ///
    /// Note: This function makes no attempt to avoid creating "degenerate"
    /// boxes, where `min >= max`.
    ///
    /// ### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let b = Box3::<f32>::new(point!(0.3, 0.3, 0.3), point!(2.7, 2.7, 2.7));
    /// let rounded = b.round_in();
    /// assert_eq!(rounded.min, point!(1.0, 1.0, 1.0));
    /// assert_eq!(rounded.max, point!(2.0, 2.0, 2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn round_in(self) -> Self {
        Box3 {
            min: self.min.ceil(),
            max: self.max.floor(),
        }
    }

    /// Round away from the center of the box.
    ///
    /// Note: This function can only create "degenerate" boxes (where min >=
    /// max) if the box was already degenerate.
    ///
    /// ### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// let b = Box3::<f32>::new(point!(0.7, 0.7, 0.7), point!(1.4, 1.4, 1.4));
    /// let rounded = b.round_out();
    /// assert_eq!(rounded.min, point!(0.0, 0.0, 0.0));
    /// assert_eq!(rounded.max, point!(2.0, 2.0, 2.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn round_out(self) -> Self {
        Box3 {
            min: self.min.floor(),
            max: self.max.ceil(),
        }
    }

    /// Linear interpolation between boxes.
    #[must_use]
    pub fn lerp(self, other: Self, t: T::Scalar) -> Self {
        let min = self.min.lerp(other.min, t);
        let max = self.max.lerp(other.max, t);
        Box3 { min, max }
    }
}

impl<T: Unit> PartialEq for Box2<T> {
    fn eq(&self, other: &Self) -> bool {
        self.min == other.min && self.max == other.max
    }
}
impl<T: IntUnit> Eq for Box2<T> {}

impl<T: Unit> PartialEq for Box3<T> {
    fn eq(&self, other: &Self) -> bool {
        self.min == other.min && self.max == other.max
    }
}
impl<T: IntUnit> Eq for Box3<T> {}

impl<T: FloatUnit> approx::AbsDiffEq for Box2<T> {
    type Epsilon = T::Scalar;

    fn default_epsilon() -> Self::Epsilon {
        T::Scalar::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.min.abs_diff_eq(&other.min, epsilon) && self.max.abs_diff_eq(&other.max, epsilon)
    }

    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.min.abs_diff_ne(&other.min, epsilon) || self.max.abs_diff_ne(&other.max, epsilon)
    }
}
impl<T: FloatUnit> approx::RelativeEq for Box2<T> {
    fn default_max_relative() -> Self::Epsilon {
        T::Scalar::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.min.relative_eq(&other.min, epsilon, max_relative)
            && self.max.relative_eq(&other.max, epsilon, max_relative)
    }

    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.min.relative_ne(&other.min, epsilon, max_relative)
            || self.max.relative_ne(&other.max, epsilon, max_relative)
    }
}
impl<T: FloatUnit> approx::UlpsEq for Box2<T> {
    fn default_max_ulps() -> u32 {
        T::Scalar::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.min.ulps_eq(&other.min, epsilon, max_ulps)
            && self.max.ulps_eq(&other.max, epsilon, max_ulps)
    }

    fn ulps_ne(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.min.ulps_ne(&other.min, epsilon, max_ulps)
            || self.max.ulps_ne(&other.max, epsilon, max_ulps)
    }
}

impl<T: FloatUnit> approx::AbsDiffEq for Box3<T> {
    type Epsilon = T::Scalar;

    fn default_epsilon() -> Self::Epsilon {
        T::Scalar::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.min.abs_diff_eq(&other.min, epsilon) && self.max.abs_diff_eq(&other.max, epsilon)
    }

    fn abs_diff_ne(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.min.abs_diff_ne(&other.min, epsilon) || self.max.abs_diff_ne(&other.max, epsilon)
    }
}
impl<T: FloatUnit> approx::RelativeEq for Box3<T> {
    fn default_max_relative() -> Self::Epsilon {
        T::Scalar::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.min.relative_eq(&other.min, epsilon, max_relative)
            && self.max.relative_eq(&other.max, epsilon, max_relative)
    }

    fn relative_ne(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.min.relative_ne(&other.min, epsilon, max_relative)
            || self.max.relative_ne(&other.max, epsilon, max_relative)
    }
}
impl<T: FloatUnit> approx::UlpsEq for Box3<T> {
    fn default_max_ulps() -> u32 {
        T::Scalar::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.min.ulps_eq(&other.min, epsilon, max_ulps)
            && self.max.ulps_eq(&other.max, epsilon, max_ulps)
    }

    fn ulps_ne(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.min.ulps_ne(&other.min, epsilon, max_ulps)
            || self.max.ulps_ne(&other.max, epsilon, max_ulps)
    }
}

impl<T: Unit> core::fmt::Debug for Box2<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Box2")
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}
impl<T: Unit> core::fmt::Debug for Box3<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Box3")
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;

    use crate::vec2;

    use super::*;

    type Box2 = super::Box2<f32>;
    type Box3 = super::Box3<f32>;
    type IBox2 = super::Box2<i32>;
    type Rect = super::Rect<f32>;

    #[test]
    fn from_rect() {
        let r = Rect::new((10.0, 12.0).into(), (5.0, 6.0).into());
        let b = Box2::from_rect(r);
        let b2 = Box2::from_origin_and_size((10.0, 12.0).into(), (5.0, 6.0).into());
        assert_abs_diff_eq!(b, b2);
        assert_abs_diff_eq!(b.min, Point2::new(10.0, 12.0));
        assert_abs_diff_eq!(b.max, Point2::new(15.0, 18.0));

        let b3 = Box2::from_size((10.0, 12.0).into());
        assert_abs_diff_eq!(b3.min, Point2::ZERO);
        assert_abs_diff_eq!(b3.max, Point2::new(10.0, 12.0));

        let r2 = b2.to_rect();
        assert_abs_diff_eq!(r2, r);
    }

    #[test]
    fn from_array() {
        let array = [Point2::ZERO, Point2::ONE];
        let b = Box2::from_array(array);
        assert_eq!(b.min, Point2::ZERO);
        assert_eq!(b.max, Point2::ONE);
    }

    #[test]
    fn cast() {
        let mut b = Box2::new(Point2::new(0.5, 0.5), Point2::new(1.0, 1.0));
        let _: &[Point2; 2] = b.as_array();
        let _: &mut [Point2; 2] = b.as_array_mut();
        let _: [Point2; 2] = b.to_array();
        let _: &[f32; 4] = b.as_scalar_array();
        let _: &mut [f32; 4] = b.as_scalar_array_mut();
        let _: [f32; 4] = b.to_scalar_array();
    }

    #[test]
    fn from_points() {
        let empty: [Point2<_>; 0] = [];
        let b = Box2::from_points(empty);
        assert_eq!(b, Box2::ZERO);

        let b =
            Box2::from_points([(0.1, 1.0), (1.0, 2.0), (0.0, 2.0), (0.5, 0.5)].map(Point2::from));

        assert_abs_diff_eq!(
            b,
            Box2 {
                min: Point2::new(0.0, 0.5),
                max: Point2::new(1.0, 2.0),
            }
        );
    }

    #[test]
    fn corners() {
        let b = Box2 {
            min: (-1.0, -1.0).into(),
            max: (1.0, 1.0).into(),
        };
        let [nw, ne, se, sw] = b.corners();
        assert_eq!(nw, Point2::new(-1.0, -1.0));
        assert_eq!(ne, Point2::new(1.0, -1.0));
        assert_eq!(se, Point2::new(1.0, 1.0));
        assert_eq!(sw, Point2::new(-1.0, 1.0));
    }

    #[test]
    fn negative_empty() {
        let r = IBox2::from_size((-1, -1).into());
        assert!(r.is_empty());
        assert!(r.is_negative());

        let r = IBox2::from_size((0, 0).into());
        assert!(r.is_empty());
        assert!(!r.is_negative());

        let r = IBox2::from_size((1, 1).into());
        assert!(!r.is_empty());
        assert!(!r.is_negative());

        // Negative zero.

        let r = Box2::from_size((-0.0, 0.0).into());
        assert!(r.is_empty());
        assert!(!r.is_negative());

        let r = Box2::new((1.0, 1.0).into(), (-0.0, -0.0).into());
        assert!(r.is_empty());
        assert!(r.is_negative());

        // NaN

        let r = Box2::from_size((f32::NAN, f32::NAN).into());
        assert!(r.is_empty());
        assert!(r.is_negative());

        let r = Box2::new((f32::NAN, 1.0).into(), (1.0, 1.0).into());
        assert!(r.is_empty());
        assert!(r.is_negative());
    }

    #[test]
    fn translate() {
        let b = Box2 {
            min: (1.0, 2.0).into(),
            max: (2.0, 3.0).into(),
        };
        let b2 = b.translate(vec2!(1.0, 0.5));
        assert_abs_diff_eq!(
            b2,
            Box2 {
                min: (2.0, 2.5).into(),
                max: (3.0, 3.5).into(),
            }
        );

        let mut b3 = b;
        b3 += vec2![1.0, 0.5];
        assert_abs_diff_eq!(b3, b2);
        assert_abs_diff_eq!(b3 - vec2![1.0, 0.5], b);
        b3 -= vec2![1.0, 0.5];
        assert_abs_diff_eq!(b3, b);
    }

    #[test]
    fn center() {
        let b = Box2 {
            min: (-1.0, -1.0).into(),
            max: (1.0, 1.0).into(),
        };

        assert_abs_diff_eq!(b.center(), Point2 { x: 0.0, y: 0.0 });
    }

    #[test]
    fn size() {
        let b = Box2 {
            min: (-1.0, -1.0).into(),
            max: (1.0, 1.0).into(),
        };

        assert_abs_diff_eq!(
            b.size(),
            Size2 {
                width: 2.0,
                height: 2.0,
            }
        );

        assert_abs_diff_eq!(b.size().area(), b.area());
        assert_abs_diff_eq!(b.area(), 4.0);

        assert_abs_diff_eq!(Box2::ZERO.area(), 0.0);
    }

    #[test]
    fn contains() {
        let b = Box2::new((-1.0, -1.0).into(), (1.0, 1.0).into());
        assert!(b.contains(&Point2::new(-1.0, 0.0)));
        assert!(b.contains(&Point2::new(1.0, 1.0)));
    }

    #[test]
    #[allow(clippy::many_single_char_names)]
    fn intersection() {
        type Box2 = super::Box2<f32>;
        let x = Box2::new((10.0, 10.0).into(), (20.0, 20.0).into());

        {
            // No intersection
            assert!(x.intersection(&Point2::new(0.0, 0.0)).is_none());

            let nw = Box2::new((0.0, 0.0).into(), (10.0, 10.0).into());
            let n = Box2::new((10.0, 0.0).into(), (20.0, 10.0).into());
            let ne = Box2::new((20.0, 0.0).into(), (30.0, 10.0).into());
            let e = Box2::new((20.0, 10.0).into(), (30.0, 20.0).into());
            let se = Box2::new((20.0, 20.0).into(), (30.0, 30.0).into());
            let s = Box2::new((10.0, 20.0).into(), (20.0, 30.0).into());
            let sw = Box2::new((0.0, 20.0).into(), (10.0, 30.0).into());
            let w = Box2::new((0.0, 10.0).into(), (10.0, 20.0).into());

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

            assert_eq!(nw.intersection(&x), None);
            assert_eq!(n.intersection(&x), None);
            assert_eq!(ne.intersection(&x), None);
            assert_eq!(e.intersection(&x), None);
            assert_eq!(x.intersection(&se.min), None);
            assert_eq!(x.intersection(&se.max), None);
            assert_eq!(se.intersection(&x.min), None);
            assert_eq!(se.intersection(&x), None);
            assert_eq!(s.intersection(&x), None);
            assert_eq!(sw.intersection(&x), None);
            assert_eq!(w.intersection(&x), None);

            assert!(!nw.intersects(&x));
            assert!(!n.intersects(&x));
            assert!(!ne.intersects(&x));
            assert!(!e.intersects(&x));
            assert!(!x.intersects(&se.min));
            assert!(!x.intersects(&se.max));
            assert!(!se.intersects(&x.min));
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
        }

        {
            assert_eq!(
                x.intersection(&Point2::new(10.0, 10.0)),
                Some(Point2::new(10.0, 10.0))
            );

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

    #[test]
    fn intersection_contained() {
        let large = Box2::new((0.0, 0.0).into(), (100.0, 100.0).into());
        let small = Box2::new((10.0, 10.0).into(), (90.0, 90.0).into());
        assert!(large.intersects(&small));
        assert!(small.intersects(&large));
        assert_eq!(large.intersection(&small), Some(small));
        assert_eq!(small.intersection(&large), Some(small));

        let infinite = Box2::new(Point2::ZERO, Point2::INFINITY);
        assert!(infinite.intersects(&small));
        assert!(small.intersects(&infinite));
        assert_eq!(infinite.intersection(&small), Some(small));
        assert_eq!(small.intersection(&infinite), Some(small));
    }

    #[test]
    fn intersection_partial() {
        let large = Box2::new((0.0, 0.0).into(), (100.0, 100.0).into());
        let small = Box2::new((90.0, 10.0).into(), (110.0, 20.0).into());
        let expected = Box2::new((90.0, 10.0).into(), (100.0, 20.0).into());
        assert!(large.intersects(&small));
        assert!(small.intersects(&large));
        assert_eq!(large.intersection(&small), Some(expected));
        assert_eq!(small.intersection(&large), Some(expected));
    }

    #[test]
    fn union() {
        let a = Box2 {
            min: (0.0, 0.0).into(),
            max: (1.0, 1.0).into(),
        };
        let b = Box2 {
            min: (2.0, 2.0).into(),
            max: (3.0, 3.0).into(),
        };

        assert_abs_diff_eq!(
            a.union(b),
            Box2 {
                min: (0.0, 0.0).into(),
                max: (3.0, 3.0).into(),
            }
        );

        assert_abs_diff_eq!(b.union(Box2::ZERO), b);
        assert_abs_diff_eq!(Box2::ZERO.union(b), b);
    }

    #[test]
    fn lerp() {
        let a = Box2 {
            min: (0.0, 0.0).into(),
            max: (1.0, 1.0).into(),
        };
        let b = Box2 {
            min: (2.0, 2.0).into(),
            max: (3.0, 3.0).into(),
        };

        let c = a.lerp(b, 0.5);
        assert_abs_diff_eq!(
            c,
            Box2 {
                min: (1.0, 1.0).into(),
                max: (2.0, 2.0).into(),
            }
        );
    }

    #[test]
    fn lerp3() {
        let a = Box3 {
            min: (0.0, 0.0, 0.0).into(),
            max: (1.0, 1.0, 1.0).into(),
        };
        let b = Box3 {
            min: (2.0, 2.0, 2.0).into(),
            max: (3.0, 3.0, 3.0).into(),
        };

        let c = a.lerp(b, 0.5);
        assert_abs_diff_eq!(
            c,
            Box3 {
                min: (1.0, 1.0, 1.0).into(),
                max: (2.0, 2.0, 2.0).into(),
            }
        );
    }
}

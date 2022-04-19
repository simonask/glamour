//! Strongly typed 2D and 3D transforms.
//!
//! # Chaining transforms
//!
//!

use core::ops::Mul;
use core::{fmt::Debug, marker::PhantomData};

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use bytemuck::{cast, Pod, Zeroable};

use crate::{
    unit::UnitMatrices, Angle, Matrix3, Matrix4, Point2, Point3, Scalar, Unit, UnitTypes, Vector2,
    Vector3,
};

/// 2D transform represented as a 3x3 column-major matrix.
///
/// This is a strongly typed wrapper around a [`Matrix3`], where that matrix
/// describes how to map between units.
#[repr(C)]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
pub struct Transform2<Src: Unit, Dst: Unit> {
    /// Underlying matrix.
    pub matrix: Matrix3<<Src::Scalar as Scalar>::Primitive>,
    #[cfg_attr(feature = "serde", serde(skip))]
    _marker: PhantomData<Dst>,
}

/// 3D transform represented as a 4x4 column-major matrix.
///
/// This is a strongly typed wrapper around a [`Matrix4`], where that matrix
/// describes how to map between units.
#[repr(C, align(16))]
#[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(bound = ""))]
pub struct Transform3<Src: Unit, Dst: Unit> {
    /// Underlying matrix.
    pub matrix: Matrix4<<Src::Scalar as Scalar>::Primitive>,
    #[cfg_attr(feature = "serde", serde(skip))]
    _marker: PhantomData<Dst>,
}

/// The mapping operation from one unit to another through a transform.
pub trait TransformMap<T> {
    /// Result type of the transformation.
    type Output;

    /// Map `T` to `Self::Output`.
    #[must_use]
    fn map(&self, value: T) -> Self::Output;
}

impl<Src: Unit, Dst: Unit> Clone for Transform2<Src, Dst> {
    fn clone(&self) -> Self {
        Self {
            matrix: self.matrix,
            _marker: PhantomData,
        }
    }
}

impl<Src: Unit, Dst: Unit> Clone for Transform3<Src, Dst> {
    fn clone(&self) -> Self {
        Transform3 {
            matrix: self.matrix,
            _marker: PhantomData,
        }
    }
}

impl<Src: Unit, Dst: Unit> Copy for Transform2<Src, Dst> {}
impl<Src: Unit, Dst: Unit> Copy for Transform3<Src, Dst> {}

impl<Src, Dst> Default for Transform2<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit,
{
    #[inline]
    fn default() -> Self {
        Self {
            matrix: Matrix3::default(),
            _marker: PhantomData,
        }
    }
}

impl<Src, Dst> Default for Transform3<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit,
{
    #[inline]
    fn default() -> Self {
        Self {
            matrix: Matrix4::default(),
            _marker: PhantomData,
        }
    }
}

impl<Src, Dst> Debug for Transform2<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Transform2")
            .field("matrix", &self.matrix)
            .finish()
    }
}

impl<Src, Dst> Debug for Transform3<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Transform3")
            .field("matrix", &self.matrix)
            .finish()
    }
}

impl<Src, Dst> PartialEq for Transform2<Src, Dst>
where
    Src: Unit,
    Dst: Unit,
{
    fn eq(&self, other: &Self) -> bool {
        self.matrix == other.matrix
    }
}

impl<Src, Dst> AbsDiffEq for Transform2<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit,
{
    type Epsilon = <Matrix3<Src::Primitive> as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        <Matrix3<Src::Primitive> as AbsDiffEq>::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.matrix.abs_diff_eq(&other.matrix, epsilon)
    }
}

impl<Src, Dst> RelativeEq for Transform2<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit,
{
    fn default_max_relative() -> Self::Epsilon {
        <Matrix3<Src::Primitive> as RelativeEq>::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.matrix
            .relative_eq(&other.matrix, epsilon, max_relative)
    }
}

impl<Src, Dst> UlpsEq for Transform2<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit,
{
    fn default_max_ulps() -> u32 {
        <Matrix3<Src::Primitive> as UlpsEq>::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.matrix.ulps_eq(&other.matrix, epsilon, max_ulps)
    }
}

impl<Src: Unit, Dst: Unit> PartialEq for Transform3<Src, Dst> {
    fn eq(&self, other: &Self) -> bool {
        self.matrix == other.matrix
    }
}

impl<Src, Dst> AbsDiffEq for Transform3<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit,
{
    type Epsilon = <Matrix4<Src::Primitive> as AbsDiffEq>::Epsilon;

    fn default_epsilon() -> Self::Epsilon {
        <Matrix4<Src::Primitive> as AbsDiffEq>::default_epsilon()
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        self.matrix.abs_diff_eq(&other.matrix, epsilon)
    }
}

impl<Src, Dst> RelativeEq for Transform3<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit,
{
    fn default_max_relative() -> Self::Epsilon {
        <Matrix4<Src::Primitive> as RelativeEq>::default_max_relative()
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        max_relative: Self::Epsilon,
    ) -> bool {
        self.matrix
            .relative_eq(&other.matrix, epsilon, max_relative)
    }
}

impl<Src, Dst> UlpsEq for Transform3<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit,
{
    fn default_max_ulps() -> u32 {
        <Matrix4<Src::Primitive> as UlpsEq>::default_max_ulps()
    }

    fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
        self.matrix.ulps_eq(&other.matrix, epsilon, max_ulps)
    }
}

// SAFETY: These impls are safe because all members are required to be Pod by
// trait bounds in Scalar.
unsafe impl<Src: Unit, Dst: Unit> Zeroable for Transform2<Src, Dst> {}
unsafe impl<Src: Unit, Dst: Unit> Pod for Transform2<Src, Dst> {}
unsafe impl<Src: Unit, Dst: Unit> Zeroable for Transform3<Src, Dst> {}
unsafe impl<Src: Unit, Dst: Unit> Pod for Transform3<Src, Dst> {}

impl<Src, Dst> Transform2<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit<Scalar = Src::UnitMatrixScalar>,
{
    /// Identity matrix.
    pub const IDENTITY: Self = Self {
        matrix: Matrix3::IDENTITY,
        _marker: PhantomData,
    };

    /// Create from matrix.
    #[inline]
    #[must_use]
    pub fn from_matrix_unchecked(matrix: Matrix3<Src::Primitive>) -> Self {
        Transform2 {
            matrix,
            _marker: PhantomData,
        }
    }

    /// Create from matrix, checking if the matrix is invertible.
    #[inline]
    #[must_use]
    pub fn from_matrix(matrix: Matrix3<Src::Primitive>) -> Option<Self> {
        if matrix.is_invertible() {
            Some(Self::from_matrix_unchecked(matrix))
        } else {
            None
        }
    }

    /// Create rotation transform.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    /// struct A;
    /// impl Unit for A { type Scalar = f64; }
    /// struct B;
    /// impl Unit for B { type Scalar = f64; }
    ///
    /// type Transform = Transform2<A, B>;
    ///
    /// let translate = Transform::from_angle(Angle::FRAG_PI_2);
    /// let a: Vector2<A> = Vector2 { x: 10.0, y: 20.0 };
    /// let b: Vector2<B> = translate.map(a);
    /// assert_abs_diff_eq!(b, vec2!(-20.0, 10.0), epsilon = 0.000001);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_angle(angle: Angle<Src::Primitive>) -> Self {
        Self::from_matrix_unchecked(Matrix3::from_angle(Angle {
            radians: angle.to_raw(),
        }))
    }

    /// Create scaling transform.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    /// struct A;
    /// impl Unit for A { type Scalar = f32; }
    /// struct B;
    /// impl Unit for B { type Scalar = f32; }
    ///
    /// type Transform = Transform2<A, B>;
    ///
    /// let translate = Transform::from_scale(Vector2 { x: 2.0, y: 3.0 });
    /// let a: Vector2<A> = Vector2 { x: 10.0, y: 20.0 };
    /// let b: Vector2<B> = translate.map(a);
    /// assert_abs_diff_eq!(b, vec2!(20.0, 60.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_scale(scale: Vector2<Src>) -> Self {
        Self::from_matrix_unchecked(Matrix3::from_scale(scale.to_untyped()))
    }

    /// Create translation transform.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    /// struct A;
    /// impl Unit for A { type Scalar = f32; }
    /// struct B;
    /// impl Unit for B { type Scalar = f32; }
    ///
    /// type Transform = Transform2<A, B>;
    ///
    /// let translate = Transform::from_translation(Vector2 { x: 10.0, y: 20.0 });
    /// let a: Point2<A> = Point2 { x: 1.0, y: 2.0 };
    /// let b: Point2<B> = translate.map(a);
    /// assert_abs_diff_eq!(b, point!(11.0, 22.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_translation(translation: Vector2<Src>) -> Self {
        Self::from_matrix_unchecked(Matrix3::from_translation(translation.to_untyped()))
    }

    /// Create scaling, rotation, and translation matrix.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    ///
    /// type Transform = Transform2<f32, f32>;
    ///
    /// let a = Transform::from_scale_angle_translation(
    ///     (2.0, 3.0).into(),
    ///     Angle::from_degrees(90.0),
    ///     (10.0, 20.0).into(),
    /// );
    ///
    /// let b = Transform::from_scale((2.0, 3.0).into())
    ///     .then_rotate(Angle::from_degrees(90.0))
    ///     .then_translate((10.0, 20.0).into());
    ///
    /// assert_abs_diff_eq!(a, b);
    #[inline]
    #[must_use]
    pub fn from_scale_angle_translation(
        scale: Vector2<Src>,
        angle: Angle<Src::Primitive>,
        translation: Vector2<Src>,
    ) -> Self {
        Self::from_matrix_unchecked(Matrix3::from_scale_angle_translation(
            scale.to_untyped(),
            angle,
            translation.to_untyped(),
        ))
    }
}

impl<Src, Dst> TransformMap<Point2<Src>> for Transform2<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit<Scalar = Src::UnitMatrixScalar>,
{
    type Output = Point2<Dst>;

    #[must_use]
    fn map(&self, value: Point2<Src>) -> Self::Output {
        Point2::from_untyped(self.matrix.transform_point(value.to_untyped()))
    }
}

impl<Src, Dst> TransformMap<Vector2<Src>> for Transform2<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit<Scalar = Src::UnitMatrixScalar>,
{
    type Output = Vector2<Dst>;

    #[must_use]
    fn map(&self, value: Vector2<Src>) -> Self::Output {
        Vector2::from_untyped(self.matrix.transform_vector(value.to_untyped()))
    }
}

impl<Src, Dst> Transform2<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit<Scalar = Src::UnitMatrixScalar>,
{
    /// Perform matrix multiplication such that `other`'s transformation applies
    /// after `self`.
    ///
    /// This may change the target coordinate space - that is, the resulting
    /// transform takes points and vectors from `Src` to `Dst2`.
    ///
    /// This operation is equivalent to `other.matrix * self.matrix` - that is,
    /// the multiplication order is the reverse of the order of the effects of
    /// the transformation.
    #[inline]
    #[must_use]
    pub fn then<Dst2>(self, other: Transform2<Dst, Dst2>) -> Transform2<Src, Dst2>
    where
        Dst2: UnitTypes<UnitScalar = Dst::Scalar>,
    {
        Transform2::from_matrix_unchecked(other.matrix * self.matrix)
    }

    /// Shorthand for `.then(Transform2::from_angle(angle))`.
    ///
    /// This does not change the target coordinate space.
    #[inline]
    #[must_use]
    pub fn then_rotate(self, angle: Angle<Src::Primitive>) -> Self {
        self.then(Transform2::from_angle(angle))
    }

    /// Shorthand for `.then(Transform2::from_scale(scale))`.
    ///
    /// This does not change the target coordinate space.
    ///
    /// #### Example
    ///
    /// ```rust
    /// # use glamour::prelude::*;
    /// let a = Transform2::<f32, f32>::IDENTITY
    ///     .then_scale((2.0, 3.0).into());
    /// let b = Transform2::<f32, f32>::from_scale((2.0, 3.0).into());
    /// assert_eq!(a, b);
    /// ```
    #[inline]
    #[must_use]
    pub fn then_scale(self, scale: Vector2<Dst>) -> Self {
        self.then(Transform2::from_scale(scale))
    }

    /// Shorthand for `.then(Transform2::from_scale(scale))`.
    ///
    /// This does not change the target coordinate space.
    #[inline]
    #[must_use]
    pub fn then_translate(self, translation: Vector2<Dst>) -> Self {
        self.then(Transform2::from_translation(translation))
    }

    /// Invert the matrix.
    ///
    /// See [`glam::Mat3::inverse()`] and [`glam::DMat3::inverse()`].
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Transform2<Dst, Src> {
        Transform2::from_matrix_unchecked(self.matrix.inverse_unchecked())
    }
}

impl<Src, Dst, Dst2> Mul<Transform2<Dst, Dst2>> for Transform2<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit<Scalar = Src::UnitMatrixScalar>,
    Dst2: Unit<Scalar = Src::UnitMatrixScalar>,
{
    type Output = Transform2<Src, Dst2>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Transform2<Dst, Dst2>) -> Self::Output {
        Transform2::from_matrix_unchecked(rhs.matrix * self.matrix)
    }
}

impl<Src, Dst, Dst2> Mul<Transform3<Dst, Dst2>> for Transform3<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit<Scalar = Src::UnitMatrixScalar>,
    Dst2: Unit<Scalar = Src::UnitMatrixScalar>,
{
    type Output = Transform3<Src, Dst2>;

    #[inline]
    #[must_use]
    fn mul(self, rhs: Transform3<Dst, Dst2>) -> Self::Output {
        Transform3::from_matrix_unchecked(rhs.matrix * self.matrix)
    }
}

impl<Src, Dst> Transform3<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit<Scalar = Src::UnitMatrixScalar>,
{
    /// Identity matrix.
    pub const IDENTITY: Self = Self {
        matrix: Matrix4::IDENTITY,
        _marker: PhantomData,
    };

    /// Create from matrix.
    #[inline]
    #[must_use]
    pub fn from_matrix_unchecked(matrix: Matrix4<<Src::Scalar as Scalar>::Primitive>) -> Self {
        Transform3 {
            matrix,
            _marker: PhantomData,
        }
    }

    /// Create from matrix, checking if the matrix is invertible.
    #[inline]
    #[must_use]
    pub fn from_matrix(matrix: Matrix4<<Src::Scalar as Scalar>::Primitive>) -> Option<Self> {
        if matrix.is_invertible() {
            Some(Self::from_matrix_unchecked(matrix))
        } else {
            None
        }
    }

    /// Perform matrix multiplication such that `other`'s transformation applies
    /// after `self`.
    ///
    /// This may change the target coordinate space - that is, the resulting
    /// transform takes points and vectors from `Src` to `Dst2`.
    ///
    /// This operation is equivalent to `other.matrix * self.matrix` - that is,
    /// the multiplication order is the reverse of the order of the effects of
    /// the transformation.
    #[inline]
    #[must_use]
    pub fn then<Dst2: Unit<Scalar = Dst::Scalar>>(
        self,
        other: Transform3<Dst, Dst2>,
    ) -> Transform3<Src, Dst2> {
        Transform3::from_matrix_unchecked(other.matrix * self.matrix)
    }

    /// Shorthand for `.then(Transform3::from_angle(angle))`.
    ///
    /// This does not change the target coordinate space.
    #[inline]
    #[must_use]
    pub fn then_rotate(self, axis: Vector3<Dst>, angle: Angle<Src::Primitive>) -> Self {
        self.then(Transform3::from_axis_angle(axis, angle))
    }

    /// Shorthand for `.then(Transform3::from_scale(scale))`.
    ///
    /// This does not change the target coordinate space.
    ///
    /// #### Example
    ///
    /// ```rust
    /// # use glamour::prelude::*;
    /// let a = Transform3::<f32, f32>::IDENTITY
    ///     .then_scale((2.0, 3.0, 4.0).into());
    /// let b = Transform3::<f32, f32>::from_scale((2.0, 3.0, 4.0).into());
    /// assert_eq!(a, b);
    /// ```
    #[inline]
    #[must_use]
    pub fn then_scale(self, scale: Vector3<Dst>) -> Self {
        self.then(Transform3::from_scale(scale))
    }

    /// Shorthand for `.then(Transform3::from_scale(scale))`.
    ///
    /// This does not change the target coordinate space.
    #[inline]
    #[must_use]
    pub fn then_translate(self, translation: Vector3<Dst>) -> Self {
        self.then(Transform3::from_translation(translation))
    }

    /// Create rotation transform.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    /// struct A;
    /// impl Unit for A { type Scalar = f64; }
    /// struct B;
    /// impl Unit for B { type Scalar = f64; }
    ///
    /// type Transform = Transform3<A, B>;
    ///
    /// let translate = Transform::from_axis_angle(Vector3::Z, Angle::FRAG_PI_2);
    /// let a: Vector3<A> = Vector3 { x: 10.0, y: 20.0, z: 30.0 };
    /// let b: Vector3<B> = translate.map(a);
    /// assert_abs_diff_eq!(b, vec3!(-20.0, 10.0, 30.0), epsilon = 0.000001);
    /// ```
    #[inline]
    #[must_use]
    pub fn from_axis_angle(axis: Vector3<Src>, angle: Angle<Src::Primitive>) -> Self {
        Self::from_matrix_unchecked(Matrix4::from_axis_angle(
            axis.to_untyped(),
            Angle {
                radians: angle.to_raw(),
            },
        ))
    }

    /// Create scaling transform.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    /// struct A;
    /// impl Unit for A { type Scalar = f32; }
    /// struct B;
    /// impl Unit for B { type Scalar = f32; }
    ///
    /// type Transform = Transform3<A, B>;
    ///
    /// let scale = Transform::from_scale(Vector3 { x: 2.0, y: 3.0, z: 1.0 });
    /// let a: Vector3<A> = Vector3 { x: 10.0, y: 20.0, z: 30.0 };
    /// let b: Vector3<B> = scale.map(a);
    /// assert_abs_diff_eq!(b, vec3!(20.0, 60.0, 30.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_scale(scale: Vector3<Src>) -> Self {
        Self::from_matrix_unchecked(Matrix4::from_scale(scale.to_untyped()))
    }

    /// Create translation transform.
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    /// struct A;
    /// impl Unit for A { type Scalar = f32; }
    /// struct B;
    /// impl Unit for B { type Scalar = f32; }
    ///
    /// type Transform = Transform3<A, B>;
    ///
    /// let translate = Transform::from_translation(Vector3 { x: 10.0, y: 20.0, z: 30.0 });
    /// let a: Point3<A> = Point3 { x: 1.0, y: 2.0, z: 30.0 };
    /// let b: Point3<B> = translate.map(a);
    /// assert_abs_diff_eq!(b, point!(11.0, 22.0, 60.0));
    /// ```
    #[inline]
    #[must_use]
    pub fn from_translation(translation: Vector3<Src>) -> Self {
        Self::from_matrix_unchecked(Matrix4::from_translation(translation.to_untyped()))
    }

    /// Create scaling, rotation, and translation matrix.
    ///
    /// Note: This internally converts `axis` and `angle` to a quaternion and
    /// calls [`glam::Mat4::from_scale_rotation_translation()`].
    ///
    /// #### Example
    /// ```rust
    /// # use glamour::prelude::*;
    /// # use approx::*;
    ///
    /// type Transform = Transform3<f32, f32>;
    ///
    /// let a = Transform::from_scale_rotation_translation(
    ///     (2.0, 3.0, 1.0).into(),
    ///     (0.0, 0.0, 1.0).into(),
    ///     Angle::from_degrees(90.0),
    ///     (10.0, 20.0, 30.0).into(),
    /// );
    ///
    /// // While this is equivalent, it introduces a significant amount of
    /// // floating point imprecision.
    /// let b = Transform::from_scale((2.0, 3.0, 1.0).into())
    ///     .then_rotate((0.0, 0.0, 1.0).into(), Angle::from_degrees(90.0))
    ///     .then_translate((10.0, 20.0, 30.0).into());
    ///
    /// assert_abs_diff_eq!(a, b, epsilon = 0.001);
    #[inline]
    #[must_use]
    pub fn from_scale_rotation_translation(
        scale: Vector3<Src>,
        axis: Vector3<Src>,
        angle: Angle<Src::Primitive>,
        translation: Vector3<Src>,
    ) -> Self {
        Self::from_matrix_unchecked(Matrix4::from_scale_rotation_translation(
            scale.to_untyped(),
            axis.to_untyped(),
            angle,
            translation.to_untyped(),
        ))
    }

    /// Map vector from `Src` to `Dst`.
    ///
    /// See [`glam::Mat4::transform_vector3()`] and
    /// [`glam::DMat4::transform_vector3()`].
    #[inline]
    #[must_use]
    pub fn map_vector(&self, vector: Vector3<Src>) -> Vector3<Dst> {
        cast(self.matrix.transform_vector(vector.to_untyped()))
    }

    /// Map point from `Src` to `Dst`, including perspective correction.
    ///
    /// See [`glam::Mat4::project_point3()`] and
    /// [`glam::DMat4::project_point3()`].
    #[inline]
    #[must_use]
    pub fn map_point(&self, point: Point3<Src>) -> Point3<Dst> {
        cast(self.matrix.project_point(point))
    }

    /// Invert the matrix.
    ///
    /// See [`glam::Mat4::inverse()`] and [`glam::DMat4::inverse()`].
    #[inline]
    #[must_use]
    pub fn inverse(&self) -> Transform3<Dst, Src> {
        Transform3::from_matrix_unchecked(self.matrix.inverse_unchecked())
    }
}

impl<Src, Dst> TransformMap<Point3<Src>> for Transform3<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit<Scalar = Src::UnitMatrixScalar>,
{
    type Output = Point3<Dst>;

    #[must_use]
    fn map(&self, value: Point3<Src>) -> Self::Output {
        self.map_point(value)
    }
}

impl<Src, Dst> TransformMap<Vector3<Src>> for Transform3<Src, Dst>
where
    Src: UnitMatrices,
    Dst: Unit<Scalar = Src::UnitMatrixScalar>,
{
    type Output = Vector3<Dst>;

    #[must_use]
    fn map(&self, value: Vector3<Src>) -> Self::Output {
        self.map_vector(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    struct TestSrc;
    impl Unit for TestSrc {
        type Scalar = f32;
    }

    struct TestDst;
    impl Unit for TestDst {
        type Scalar = f32;
    }

    struct TestDst2;
    impl Unit for TestDst2 {
        type Scalar = f32;
    }

    macro_rules! check_2d_and_3d {
        { $($test:tt)* } => {{
            {
                type Transform = Transform2<TestSrc, TestDst>;
                type Mat = Matrix3<f32>;
                type TransformInverse = Transform2<TestDst, TestSrc>;
                type VectorSrc = Vector2<TestSrc>;
                type VectorDst = Vector2<TestDst>;
                type PointSrc = Point2<TestSrc>;
                type PointDst = Point2<TestDst>;
                let _ = core::mem::size_of::<(Transform, Mat, TransformInverse, VectorSrc, VectorDst, PointSrc, PointDst)>();
                $($test)*
            }
            {
                type Transform = Transform3<TestSrc, TestDst>;
                type Mat = Matrix4<f32>;
                type TransformInverse = Transform3<TestDst, TestSrc>;
                type VectorSrc = Vector3<TestSrc>;
                type VectorDst = Vector3<TestDst>;
                type PointSrc = Point3<TestSrc>;
                type PointDst = Point3<TestDst>;
                let _ = core::mem::size_of::<(Transform, Mat, TransformInverse, VectorSrc, VectorDst, PointSrc, PointDst)>();
                $($test)*
            }
        }};
    }

    #[test]
    fn basic() {
        check_2d_and_3d! {
            let a = Transform::IDENTITY;
            let b = a.clone();
            assert_eq!(a, b);
            assert_abs_diff_eq!(a, b);
            assert_relative_eq!(a, b);
            assert_ulps_eq!(a, b);

            let c = a.then_scale(VectorDst::splat(2.0));
            assert_ne!(a, c);
            assert_abs_diff_ne!(a, c);
            assert_relative_ne!(a, c);
            assert_ulps_ne!(a, c);
        };

        check_2d_and_3d! {
            let a = Transform::IDENTITY;
            assert_eq!(a, Transform::default());
        };
    }

    #[test]
    fn from_matrix() {
        check_2d_and_3d! {
            assert!(Transform::from_matrix(Mat::ZERO).is_none());
            assert_eq!(Transform::from_matrix(Mat::IDENTITY), Some(Transform::IDENTITY));
        };
    }

    #[test]
    fn inverse() {
        check_2d_and_3d! {
            assert!(Transform::from_matrix(Mat::zeroed()).is_none());

            let transform = Transform::from_translation(VectorSrc::splat(1.0));
            let point = PointSrc::splat(2.0);
            let point_dst: PointDst = transform.map(point);
            assert_abs_diff_eq!(point_dst, PointDst::splat(3.0));

            let inverse: TransformInverse = transform.inverse();
            let point_src = inverse.map(point_dst);
            assert_abs_diff_eq!(point_src, point);
        };
    }

    #[test]
    fn concatenation() {
        {
            let a = Transform2::<TestSrc, TestDst>::from_scale((2.0, 2.0).into());
            let b = Transform2::<TestDst, TestDst2>::from_translation((1.0, 1.0).into());
            let c: Transform2<TestSrc, TestDst2> = a * b;
            assert_eq!(
                c,
                Transform2::<TestSrc, TestDst2>::from_scale((2.0, 2.0).into())
                    .then_translate((1.0, 1.0).into())
            );
        }

        {
            let a = Transform3::<TestSrc, TestDst>::from_scale((2.0, 2.0, 2.0).into());
            let b = Transform3::<TestDst, TestDst2>::from_translation((1.0, 1.0, 1.0).into());
            let c: Transform3<TestSrc, TestDst2> = a * b;
            assert_eq!(
                c,
                Transform3::<TestSrc, TestDst2>::from_scale((2.0, 2.0, 2.0).into())
                    .then_translate((1.0, 1.0, 1.0).into())
            );
        }
    }
}

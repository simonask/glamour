/// Interface for all vectorlike things.
macro_rules! vector_base_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// Set all elements to `scalar`.
            fn splat(scalar: scalar) -> Self;
            /// Clamp all elements within `min` and `max`.
            fn clamp(self, min: Self, max: Self) -> Self;
            /// Return by-element minimum.
            fn min(self, other: Self) -> Self;
            /// Return by-element maximum.
            fn max(self, other: Self) -> Self;
            /// Min element.
            fn min_element(self) -> scalar;
            /// Max element.
            fn max_element(self) -> scalar;
            /// Dot product.
            fn dot(self, other: Self) -> scalar;
            /// Write this vector type to a slice.
            fn write_to_slice(self, slice: mut_scalar_slice);
            /// Replace x component.
            fn with_x(self, x: scalar) -> Self;
            /// Replace y component.
            fn with_y(self, x: scalar) -> Self;
            /// Sum of all elements.
            fn element_sum(self) -> scalar;
            /// Product of all elements.
            fn element_product(self) -> scalar;
        }
    };
}
pub(crate) use vector_base_interface;

/// Interface for all vectorlike things with signed components.
macro_rules! vector_signed_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// Returns a vector with elements representing the sign of `self`.
            fn signum(self) -> Self;
            /// Returns a vector containing the absolute value of each element of `self`.
            fn abs(self) -> Self;
        }
    };
}
pub(crate) use vector_signed_interface;

/// Interface for all vectorlike things with integer components.
macro_rules! vector_integer_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// Returns a vector containing the saturating addition of self and rhs.
            fn saturating_add(self, rhs: Self) -> Self;
            /// Returns a vector containing the saturating subtraction of self and rhs.
            fn saturating_sub(self, rhs: Self) -> Self;
            /// Returns a vector containing the saturating multiplication of self and rhs.
            fn saturating_mul(self, rhs: Self) -> Self;
            /// Returns a vector containing the saturating division of self and rhs.
            fn saturating_div(self, rhs: Self) -> Self;
            /// Returns a vector containing the wrapping addition of self and rhs.
            fn wrapping_add(self, rhs: Self) -> Self;
            /// Returns a vector containing the wrapping subtraction of self and rhs.
            fn wrapping_sub(self, rhs: Self) -> Self;
            /// Returns a vector containing the wrapping multiplication of self and rhs.
            fn wrapping_mul(self, rhs: Self) -> Self;
            /// Returns a vector containing the wrapping division of self and rhs.
            fn wrapping_div(self, rhs: Self) -> Self;
        }
    };
}
pub(crate) use vector_integer_interface;

/// Interface for all vectorlike things with floating-point components.
macro_rules! vector_float_base_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// Round all components up.
            fn ceil(self) -> Self;
            /// Clamp length
            fn clamp_length_max(self, min: scalar) -> Self;
            /// Clamp length
            fn clamp_length_min(self, min: scalar) -> Self;
            /// Clamp length
            fn clamp_length(self, min: scalar, max: scalar) -> Self;
            /// Compute the squared euclidean distance between two points in space.
            fn distance_squared(self, other: Self) -> scalar;
            /// Computes the Euclidean distance between two points in space.
            fn distance(self, other: Self) -> scalar;
            /// e^self by component
            fn exp(self) -> Self;
            /// Round all components down.
            fn floor(self) -> Self;
            /// See (e.g.) [`glam::Vec2::fract()`]
            fn fract(self) -> Self;
            /// See (e.g.) [`glam::Vec2::fract_gl()`]
            fn fract_gl(self) -> Self;
            /// True if all components are non-infinity and non-NaN.
            fn is_finite(self) -> bool;
            /// True if any component is NaN.
            fn is_nan(self) -> bool;
            /// True if the vector is normalized.
            fn is_normalized(self) -> bool;
            /// Reciprocal length of the vector
            fn length_recip(self) -> scalar;
        }
        // split to avoid recursion limit
        crate::interface! {
            $mode =>
            /// Squared length of the vector
            fn length_squared(self) -> scalar;
            /// Length of the vector
            fn length(self) -> scalar;
            #[doc = "Linear interpolation."]
            fn lerp(self, rhs: Self, s: scalar) -> Self;
            /// self * a + b
            fn mul_add(self, a: Self, b: Self) -> Self;
            /// Normalize the vector, returning zero if the length was already (very close to) zero.
            fn normalize_or_zero(self) -> Self;
            /// Normalize the vector. Undefined results in the vector's length is (very close to) zero.
            fn normalize(self) -> Self;
            /// Returns self normalized to length 1.0 if possible, else returns a fallback value.
            fn normalize_or(self, fallback: Self) -> Self;
            /// self^n by component
            fn powf(self, n: scalar) -> Self;
        }
        // split to avoid recursion limit
        crate::interface! {
            $mode =>
            /// See (e.g.) [`glam::Vec2::project_onto_normalized()`]
            fn project_onto_normalized(self, other: Self) -> Self;
            /// See (e.g.) [`glam::Vec2::project_onto()`]
            fn project_onto(self, other: Self) -> Self;
            /// 1.0/self by component
            fn recip(self) -> Self;
            /// See (e.g.) [`glam::Vec2::reject_from_normalized()`]
            fn reject_from_normalized(self, other: Self) -> Self;
            /// See (e.g.) [`glam::Vec2::reject_from()`]
            fn reject_from(self, other: Self) -> Self;
            #[doc = "Round all components."]
            fn round(self) -> Self;
            /// Normalize the vector, returning `None` if the length was already (very close to) zero.
            fn try_normalize(self) -> opt_self;
            /// Calculates the midpoint between self and rhs.
            ///
            /// See (e.g.) [`glam::Vec2::midpoint()`].
            fn midpoint(self, rhs: Self) -> Self;
            /// Moves towards rhs based on the value d.
            ///
            /// See (e.g.) [`glam::Vec2::move_towards()`].
            fn move_towards(&self, rhs: Self, d: scalar) -> Self;
        }
    };
}
pub(crate) use vector_float_base_interface;

macro_rules! vector2_base_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// Creates a new vector from an array.
            fn from_array(array: [scalar; 2]) -> Self;
            /// `[x, y]`
            fn to_array(&self) -> [scalar; 2];
            /// Creates a 3D vector from self and the given z value.
            fn extend(self, z: scalar) -> vec3;
            /// Returns a vector mask containing the result of a == comparison for each element of `self` and `rhs`.
            fn cmpeq(self, rhs: Self) -> bvec2;
            /// Returns a vector mask containing the result of a != comparison for each element of `self` and `rhs`.
            fn cmpne(self, rhs: Self) -> bvec2;
            /// Returns a vector mask containing the result of a >= comparison for each element of `self` and `rhs`.
            fn cmpge(self, rhs: Self) -> bvec2;
            /// Returns a vector mask containing the result of a > comparison for each element of `self` and `rhs`.
            fn cmpgt(self, rhs: Self) -> bvec2;
            /// Returns a vector mask containing the result of a <= comparison for each element of `self` and `rhs`.
            fn cmple(self, rhs: Self) -> bvec2;
            /// Returns a vector mask containing the result of a < comparison for each element of `self` and `rhs`.
            fn cmplt(self, rhs: Self) -> bvec2;
            /// Creates a vector from the elements in `if_true` and `if_false`, selecting which to use for each element
            /// of `self`.
            ///
            /// A true element in the mask uses the corresponding element from `if_true`, and false uses the element
            /// from `if_false`.
            fn select(mask: bvec2, if_true: Self, if_false: Self) -> Self;
        }
    };
}
pub(crate) use vector2_base_interface;

macro_rules! vector3_base_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// Creates a new vector from an array.
            fn from_array(array: [scalar; 3]) -> Self;
            /// `[x, y, z]`
            fn to_array(&self) -> [scalar; 3];
            /// Creates a 3D vector from `self` and the given `w` value.
            fn extend(self, w: scalar) -> vec4;
            /// Creates a 2D vector by removing the `z` component.
            fn truncate(self) -> vec2;
            /// Cross product.
            fn cross(self, other: Self) -> Self;
            /// Returns a vector mask containing the result of a == comparison for each element of `self` and `rhs`.
            fn cmpeq(self, rhs: Self) -> bvec3;
            /// Returns a vector mask containing the result of a != comparison for each element of `self` and `rhs`.
            fn cmpne(self, rhs: Self) -> bvec3;
            /// Returns a vector mask containing the result of a >= comparison for each element of `self` and `rhs`.
            fn cmpge(self, rhs: Self) -> bvec3;
            /// Returns a vector mask containing the result of a > comparison for each element of `self` and `rhs`.
            fn cmpgt(self, rhs: Self) -> bvec3;
            /// Returns a vector mask containing the result of a <= comparison for each element of `self` and `rhs`.
            fn cmple(self, rhs: Self) -> bvec3;
            /// Returns a vector mask containing the result of a < comparison for each element of `self` and `rhs`.
            fn cmplt(self, rhs: Self) -> bvec3;
            /// Creates a vector from the elements in `if_true` and `if_false`, selecting which to use for each element
            /// of `self`.
            ///
            /// A true element in the mask uses the corresponding element from `if_true`, and false uses the element
            /// from `if_false`.
            fn select(mask: bvec3, if_true: Self, if_false: Self) -> Self;
            /// Replace the `z` component.
            fn with_z(self, z: scalar) -> Self;
        }
    };
}
pub(crate) use vector3_base_interface;

macro_rules! vector4_base_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// Creates a new vector from an array.
            fn from_array(array: [scalar; 4]) -> Self;
            /// `[x, y, z, w]`
            fn to_array(&self) -> [scalar; 4];
            /// Creates a 3D vector by removing the `w` component.
            fn truncate(self) -> vec3;
            /// Returns a vector mask containing the result of a == comparison for each element of `self` and `rhs`.
            fn cmpeq(self, rhs: Self) -> bvec4;
            /// Returns a vector mask containing the result of a != comparison for each element of `self` and `rhs`.
            fn cmpne(self, rhs: Self) -> bvec4;
            /// Returns a vector mask containing the result of a >= comparison for each element of `self` and `rhs`.
            fn cmpge(self, rhs: Self) -> bvec4;
            /// Returns a vector mask containing the result of a > comparison for each element of `self` and `rhs`.
            fn cmpgt(self, rhs: Self) -> bvec4;
            /// Returns a vector mask containing the result of a <= comparison for each element of `self` and `rhs`.
            fn cmple(self, rhs: Self) -> bvec4;
            /// Returns a vector mask containing the result of a < comparison for each element of `self` and `rhs`.
            fn cmplt(self, rhs: Self) -> bvec4;
            /// Creates a vector from the elements in `if_true` and `if_false`, selecting which to use for each element
            /// of `self`.
            ///
            /// A true element in the mask uses the corresponding element from `if_true`, and false uses the element
            /// from `if_false`.
            fn select(mask: bvec4, if_true: Self, if_false: Self) -> Self;
            /// Replace the `z` component.
            fn with_z(self, z: scalar) -> Self;
            /// Replace the `w` component.
            fn with_w(self, w: scalar) -> Self;
        }
    };
}
pub(crate) use vector4_base_interface;

macro_rules! vector2_signed_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// Returns a vector that is equal to `self` rotated by 90 degrees.
            fn perp(self) -> Self;
            /// The perpendicular dot product of `self` and `rhs`. Also known as the wedge product, 2D cross product,
            /// and determinant.
            fn perp_dot(self, other: Self) -> scalar;
        }
    };
}
pub(crate) use vector2_signed_interface;

macro_rules! vector2_float_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// See (e.g.) [`glam::Vec2::from_angle()`].
            fn from_angle(angle: angle) -> Self;
            /// See (e.g.) [`glam::Vec2::to_angle()`].
            fn to_angle(self) -> scalar;
            /// See (e.g.) [`glam::Vec2::angle_to()`].
            fn angle_to(self, rhs: Self) -> scalar;
            /// See (e.g.) [`glam::Vec2::rotate()`].
            fn rotate(self, other: Self) -> Self;
            /// Return a mask where each bit is set if the corresponding component is NaN.
            fn is_nan_mask(self) -> bvec2;
            /// See (e.g.) [`glam::Vec2::rotate_towards()`].
            fn rotate_towards(&self, rhs: Self, max_angle: angle) -> Self;
        }
    };
}
pub(crate) use vector2_float_interface;

macro_rules! vector3_float_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// See (e.g.) [`glam::Vec3::any_orthogonal_vector()`].
            fn any_orthogonal_vector(&self) -> Self;
            /// See (e.g.) [`glam::Vec3::any_orthonormal_vector()`].
            fn any_orthonormal_vector(&self) -> Self;
            /// See (e.g.) [`glam::Vec3::any_orthonormal_pair()`].
            fn any_orthonormal_pair(&self) -> (Self, Self);
            /// Return a boolean mask where `true` indicates that the component is NaN.
            fn is_nan_mask(self) -> bvec3;
        }
    };
}
pub(crate) use vector3_float_interface;

macro_rules! vector4_float_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// Return a mask where each bit is set if the corresponding component is NaN.
            fn is_nan_mask(self) -> bvec4;
        }
    };
}
pub(crate) use vector4_float_interface;

macro_rules! matrix_base_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// Returns `true` if any elements are `NaN`.
            fn is_nan(&self) -> bool;
            /// Returns `true` if all elements are finite.
            fn is_finite(&self) -> bool;
            /// Returns the transpose of `self`.
            fn transpose(&self) -> Self;
            /// Returns the inverse of `self`.
            fn inverse(&self) -> Self;
            /// Returns the determinant of `self`.
            fn determinant(&self) -> scalar;
            /// Takes the absolute value of each element in `self`.
            fn abs(&self) -> Self;
        }
    };
}
pub(crate) use matrix_base_interface;

macro_rules! matrix2_base_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// Transforms a 2D vector.
            fn mul_vec2(&self, vec: vec2) -> vec2;
            /// Creates a 2x2 matrix from two column vectors.
            fn from_cols(x_axis: vec2, y_axis: vec2) -> Self;
            /// Returns the matrix column for the given `index`.
            ///
            /// ### Panics
            ///
            /// Panics if index is greater than 1.
            fn col(&self, index: usize) -> vec2;
            /// Returns the matrix row for the given `index`.
            ///
            /// ### Panics
            ///
            /// Panics if index is greater than 1.
            fn row(&self, index: usize) -> vec2;
            /// Creates a 2x2 matrix containing the combining non-uniform `scale` and rotation of `angle`.
            fn from_scale_angle(vector: vec2, angle: angle) -> Self;
            /// Creates a 2x2 matrix containing a rotation of `angle`.
            fn from_angle(angle: scalar) -> Self;
        }
        // Split for recursion limit.
        crate::interface! {
            $mode =>
            /// Creates a 2x2 matrix from a 3x3 matrix, discarding the 2nd row and column.
            fn from_mat3(mat3: mat3) -> Self;
            /// Creates a 2x2 matrix with its diagonal set to `diagonal` and all other entries set to 0.
            fn from_diagonal(diagonal: vec2) -> Self;
            /// Multiplies two 2x2 matrices.
            fn mul_mat2(&self, other: ref_self) -> Self;
            /// Adds two 2x2 matrices.
            fn add_mat2(&self, other: ref_self) -> Self;
            /// Subtracts two 2x2 matrices.
            fn sub_mat2(&self, other: ref_self) -> Self;
            /// Creates a `[T; 4]` array storing data in column major order.
            fn to_cols_array(&self) -> [scalar; 4];
            /// Creates a `[[T; 2]; 2]` 2D array storing data in column major order.
            fn to_cols_array_2d(&self) -> [[scalar; 2]; 2];
            /// Creates a 2x2 matrix from a `[T; 4]` array stored in column major order.
            fn from_cols_array(array: ref_scalar_array_4) -> Self;
            /// Creates a 2x2 matrix from a `[[T; 2]; 2]` 2D array stored in column major order.
            fn from_cols_array_2d(m: ref_scalar_array_2_2) -> Self;
        }
    };
}
pub(crate) use matrix2_base_interface;

macro_rules! matrix3_base_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// Transforms a 3D vector.
            fn mul_vec3(&self, vec: vec3) -> vec3;
            /// Creates a 3x3 matrix from three column vectors.
            fn from_cols(x_axis: vec3, y_axis: vec3, z_axis: vec3) -> Self;
            /// Returns the matrix column for the given `index`.
            ///
            /// ### Panics
            ///
            /// Panics if index is greater than 2.
            fn col(&self, index: usize) -> vec3;
            /// Returns the matrix row for the given `index`.
            ///
            /// ### Panics
            ///
            /// Panics if index is greater than 2.
            fn row(&self, index: usize) -> vec3;
            /// Rotates the given 2D vector.
            ///
            /// This is the equivalent of multiplying `rhs` as a 3D vector where `z` is `0`.
            ///
            /// This method assumes that `self` contains a valid affine transform.
            ///
            /// ### Panics
            ///
            /// Will panic if the 2nd row of `self` is not `(0, 0, 1)` when `glam_assert` is enabled.
            fn transform_vector2(&self, vector: vec2) -> vec2;
            /// Transforms the given 2D point.
            ///
            /// This is the equivalent of multiplying `rhs` as a 3D vector where `z` is `1`.
            ///
            /// This method assumes that `self` contains a valid affine transform.
            ///
            /// ### Panics
            ///
            /// Will panic if the 2nd row of `self` is not `(0, 0, 1)` when `glam_assert` is enabled.
            fn transform_point2(&self, point: point2) -> point2;
        }
        // Split for recursion limit.
        crate::interface! {
            $mode =>
            /// Creates an affine transformation matrix from the given non-uniform 2D `scale`.
            ///
            /// See (e.g.) [`glam::Mat3::from_scale()`].
            fn from_scale(vector: vec2) -> Self;
            /// Creates an affine transformation matrix from the given 2D rotation `angle`.
            ///
            /// See (e.g.) [`glam::Mat3::from_angle()`].
            fn from_angle(angle: angle) -> Self;
            /// Creates an affine transformation matrix from the given 2D `translation`.
            ///
            /// See (e.g.) [`glam::Mat3::from_translation()`].
            fn from_translation(translation: vec2) -> Self;
            /// Creates an affine transformation matrix from the given 2D `scale`, rotation `angle` and `translation`.
            ///
            /// See (e.g.) [`glam::Mat3::from_scale_angle_translation()`].
            fn from_scale_angle_translation(scale: vec2, angle: angle, translation: vec2) -> Self;
            /// Creates a 3x3 matrix with its diagonal set to `diagonal` and all other entries set to 0.
            fn from_diagonal(diagonal: vec3) -> Self;
        }
        // Split for recursion limit.
        crate::interface! {
            $mode =>
            /// Creates an affine transformation matrix from the given 2x2 matrix.
            fn from_mat2(mat2: mat2) -> Self;
            /// Creates a 3x3 matrix from a 4x4 matrix, discarding the 4th row and column.
            fn from_mat4(mat4: mat4) -> Self;
            /// Multiplies two 3x3 matrices.
            fn mul_mat3(&self, other: ref_self) -> Self;
            /// Adds two 3x3 matrices.
            fn add_mat3(&self, other: ref_self) -> Self;
            /// Subtracts two 3x3 matrices.
            fn sub_mat3(&self, other: ref_self) -> Self;
            /// Creates a `[T; 9]` array storing data in column major order.
            fn to_cols_array(&self) -> [scalar; 9];
            /// Creates a `[[T; 3]; 3]` 2D array storing data in column major order.
            fn to_cols_array_2d(&self) -> [[scalar; 3]; 3];
            /// Creates a 3x3 matrix from a `[T; 9]` array stored in column major order.
            fn from_cols_array(array: ref_scalar_array_9) -> Self;
        }
    };
}
pub(crate) use matrix3_base_interface;

macro_rules! matrix4_base_interface {
    ($mode:tt) => {
        crate::interface! {
            $mode =>
            /// Transforms a 4D vector.
            ///
            /// See (e.g.) [`glam::Mat4::mul_vec4()`].
            fn mul_vec4(&self, vec: vec4) -> vec4;
            /// Transform 3D point.
            ///
            /// This assumes that the matrix is a valid affine matrix, and does not
            /// perform perspective correction.
            ///
            /// See [`glam::Mat4::transform_point3()`] or
            /// [`glam::DMat4::transform_point3()`] (depending on the scalar).
            fn transform_point3(&self, point: point3) -> point3;
            /// Transform 3D vector.
            ///
            /// See [`glam::Mat4::transform_vector3()`] or
            /// [`glam::DMat4::transform_vector3()`] (depending on the scalar).
            fn transform_vector3(&self, vector: vec3) -> vec3;
            /// Project 3D point.
            ///
            /// Transform the point, including perspective correction.
            ///
            /// See [`glam::Mat4::project_point3()`] or
            /// [`glam::DMat4::project_point3()`] (depending on the scalar).
            fn project_point3(&self, vector: point3) -> point3;

            /// Creates a 4x4 matrix from four column vectors.
            fn from_cols(x_axis: vec4, y_axis: vec4, z_axis: vec4, w_axis: vec4) -> Self;
            /// Returns the matrix column for the given `index`.
            ///
            /// ### Panics
            ///
            /// Panics if index is greater than 3.
            fn col(&self, index: usize) -> vec4;
            /// Returns the matrix row for the given `index`.
            ///
            /// ### Panics
            ///
            /// Panics if index is greater than 3.
            fn row(&self, index: usize) -> vec4;
        }
        // Split for recursion limit.
        crate::interface! {
            $mode =>
            /// Creates an affine transformation matrix containing the given 3D non-uniform `scale`.
            ///
            /// See (e.g.) [`glam::Mat4::from_scale()`].
            ///
            /// ### Panics
            ///
            /// Panics if all elements of `scale` are zero when `glam_assert` is enabled.
            fn from_scale(vector: vec3) -> Self;
            /// Creates an affine transformation matrix containing a 3D rotation around a normalized rotation `axis` of
            /// `angle`.
            ///
            /// See (e.g.) [`glam::Mat4::from_axis_angle()`].
            ///
            /// ### Panics
            ///
            /// Panics if `axis` is not normalized when `glam_assert` is enabled.
            fn from_axis_angle(axis: vec3, angle: angle) -> Self;
            /// Creates an affine transformation matrix from the given 3D `translation`.
            ///
            /// See (e.g.) [`glam::Mat4::from_translation()`].
            fn from_translation(translation: vec3) -> Self;
            /// Creates an affine transformation matrix from the given 3D `scale`, `rotation` and `translation`.
            ///
            /// See (e.g.) [`glam::Mat4::from_scale_rotation_translation()`].
            ///
            /// ### Panics
            ///
            /// Will panic if `rotation` is not normalized when `glam_assert` is enabled.
            fn from_scale_rotation_translation(scale: vec3, axis: quat, translation: vec3)
                -> Self;
        }

        // Split for recursion limit
        crate::interface! {
            $mode =>
            /// See (e.g.) [`glam::Mat4::look_at_lh()`].
            fn look_at_lh(eye: point3, center: point3, up: vec3) -> Self;
            /// See (e.g.) [`glam::Mat4::look_at_rh()`].
            fn look_at_rh(eye: point3, center: point3, up: vec3) -> Self;
            /// See (e.g.) [`glam::Mat4::perspective_rh_gl()`].
            fn perspective_rh_gl(fov_y_radians: angle, aspect_ratio: scalar, z_near: scalar, z_far: scalar) -> Self;
            /// See (e.g.) [`glam::Mat4::perspective_lh()`].
            fn perspective_lh(fov_y_radians: angle, aspect_ratio: scalar, z_near: scalar, z_far: scalar) -> Self;
            /// See (e.g.) [`glam::Mat4::perspective_rh()`].
            fn perspective_rh(fov_y_radians: angle, aspect_ratio: scalar, z_near: scalar, z_far: scalar) -> Self;
            /// See (e.g.) [`glam::Mat4::perspective_infinite_lh()`].
            fn perspective_infinite_lh(fov_y_radians: angle, aspect_ratio: scalar, z_near: scalar) -> Self;
            /// See (e.g.) [`glam::Mat4::perspective_infinite_reverse_lh()`].
            fn perspective_infinite_reverse_lh(fov_y_radians: angle, aspect_ratio: scalar, z_near: scalar) -> Self;
            /// See (e.g.) [`glam::Mat4::perspective_infinite_rh()`].
            fn perspective_infinite_rh(fov_y_radians: angle, aspect_ratio: scalar, z_near: scalar) -> Self;
            /// See (e.g.) [`glam::Mat4::perspective_infinite_reverse_rh()`].
            fn perspective_infinite_reverse_rh(fov_y_radians: angle, aspect_ratio: scalar, z_near: scalar) -> Self;
            /// See (e.g.) [`glam::Mat4::orthographic_rh_gl()`].
            fn orthographic_rh_gl(left: scalar, right: scalar, bottom: scalar, top: scalar, near: scalar, far: scalar) -> Self;
            /// See (e.g.) [`glam::Mat4::orthographic_lh()`].
            fn orthographic_lh(left: scalar, right: scalar, bottom: scalar, top: scalar, near: scalar, far: scalar) -> Self;
            /// See (e.g.) [`glam::Mat4::orthographic_rh()`].
            fn orthographic_rh(left: scalar, right: scalar, bottom: scalar, top: scalar, near: scalar, far: scalar) -> Self;
        }

        // Split for recursion limit
        crate::interface! {
            $mode =>
            /// Creates a 4x4 matrix with its diagonal set to `diagonal` and all other entries set to 0.
            fn from_diagonal(diagonal: vec4) -> Self;
            /// Creates an affine transformation matrix from the given 3D `translation`.
            ///
            /// See (e.g.) [`glam::Mat4::from_rotation_translation()`].
            ///
            /// ### Panics
            ///
            /// Will panic if `rotation` is not normalized when `glam_assert` is enabled.
            fn from_rotation_translation(rotation: quat, translation: vec3) -> Self;
            /// Creates an affine transformation matrix from the given `rotation` quaternion.
            ///
            /// See (e.g.) [`glam::Mat4::from_quat()`].
            ///
            /// ### Panics
            ///
            /// Will panic if `rotation` is not normalized when `glam_assert` is enabled.
            fn from_quat(quat: quat) -> Self;
            /// Create an affine transformation matrix from the given 3x3 linear transformation matrix.
            ///
            /// See (e.g.) [`glam::Mat4::from_mat3()`].
            fn from_mat3(mat3: mat3) -> Self;
            /// Multiplies two 4x4 matrices.
            fn mul_mat4(&self, other: ref_self) -> Self;
            /// Adds two 4x4 matrices.
            fn add_mat4(&self, other: ref_self) -> Self;
            /// Subtracts two 4x4 matrices.
            fn sub_mat4(&self, other: ref_self) -> Self;
            /// Creates a `[T; 16]` array storing data in column major order.
            fn to_cols_array(&self) -> [scalar; 16];
            /// Creates a `[[T; 4]; 4]` 2D array storing data in column major order.
            fn to_cols_array_2d(&self) -> [[scalar; 4]; 4];
            /// Creates a 4x4 matrix from a `[T; 16]` array stored in column major order.
            fn from_cols_array(array: ref_scalar_array_16) -> Self;
        }
    }
}
pub(crate) use matrix4_base_interface;

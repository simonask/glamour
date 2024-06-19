/// Generate the public Matrix interfaces.
macro_rules! matrixlike {
    ($base_type_name:ident, $n:tt) => {
        impl<T: crate::FloatScalar> $base_type_name<T> {
            crate::interfaces::matrix_base_interface!(struct);
            crate::impl_matrixlike::matrixlike!(@for_size $base_type_name, $n);

            /// Return true if the determinant is finite and nonzero.
            #[inline]
            #[must_use]
            pub fn is_invertible(&self) -> bool {
                let det = self.determinant();
                det.is_finite() && det != T::ZERO
            }

            /// Get the inverse if the matrix is invertible.
            #[inline]
            #[must_use]
            pub fn try_inverse(&self) -> Option<Self> {
                if self.is_invertible() {
                    Some(self.inverse())
                } else {
                    None
                }
            }

            /// Get the columns of the transposed matrix as a scalar array.
            #[inline]
            #[must_use]
            pub fn to_rows_array(&self) -> [T; $n*$n] {
                self.transpose().to_cols_array()
            }
            /// Get the columns of the transposed matrix.
            #[inline]
            #[must_use]
            pub fn to_rows_array_2d(&self) -> [[T; $n]; $n] {
                self.transpose().to_cols_array_2d()
            }
        }
    };
    (@for_size $base_type_name:ident, 2) => {
        crate::interfaces::matrix2_base_interface!(struct);

        /// Create from rows (i.e., transposed).
        #[inline]
        #[must_use]
        pub fn from_rows(r0: Vector2<T>, r1: Vector2<T>) -> Self {
            Self::from_cols(r0, r1).transpose()
        }
    };
    (@for_size $base_type_name:ident, 3) => {
        crate::interfaces::matrix3_base_interface!(struct);

        /// Create from rows (i.e., transposed).
        #[inline]
        #[must_use]
        pub fn from_rows(r0: Vector3<T>, r1: Vector3<T>, r2: Vector3<T>) -> Self {
            Self::from_cols(r0, r1, r2).transpose()
        }
    };
    (@for_size $base_type_name:ident, 4) => {
        crate::interfaces::matrix4_base_interface!(struct);

        /// Create from rows (i.e., transposed).
        #[inline]
        #[must_use]
        pub fn from_rows(r0: Vector4<T>, r1: Vector4<T>, r2: Vector4<T>, r3: Vector4<T>) -> Self {
            Self::from_cols(r0, r1, r2, r3).transpose()
        }
    };
}
pub(crate) use matrixlike;

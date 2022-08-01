use crate::{
    scalar::FloatScalar, AsRaw, FromRaw, Matrix2, Matrix3, Matrix4, Point2, Point3, Point4, Scalar,
    Size2, Size3, ToRaw, Unit, Vector2, Vector3, Vector4,
};
use encase::{
    internal::{BufferMut, BufferRef, CreateFrom, ReadFrom, Reader, WriteInto, Writer},
    matrix::{AsMutMatrixParts, AsRefMatrixParts, FromMatrixParts, MatrixScalar},
    vector::{AsMutVectorParts, AsRefVectorParts, FromVectorParts, VectorScalar},
    ShaderSize, ShaderType,
};

macro_rules! impl_encase_vector {
    ($base_type_name:ident, $raw_name:ident, $arity:literal) => {
        impl<T> ShaderType for $base_type_name<T>
        where
            T: Unit,
            <T::Scalar as Scalar>::$raw_name: ShaderType,
        {
            type ExtraMetadata = <<Self as ToRaw>::Raw as ShaderType>::ExtraMetadata;
            const METADATA: encase::private::Metadata<Self::ExtraMetadata> =
                <<Self as ToRaw>::Raw as ShaderType>::METADATA;
        }

        impl<T> ShaderSize for $base_type_name<T>
        where
            T: Unit,
            <T::Scalar as Scalar>::$raw_name: ShaderSize,
        {
        }

        impl<T> AsRefVectorParts<T::Scalar, $arity> for $base_type_name<T>
        where
            T: Unit,
            T::Scalar: VectorScalar,
        {
            #[inline]
            fn as_ref_parts(&self) -> &[T::Scalar; $arity] {
                self.as_array()
            }
        }

        impl<T> AsMutVectorParts<T::Scalar, $arity> for $base_type_name<T>
        where
            T: Unit,
            T::Scalar: VectorScalar,
        {
            #[inline]
            fn as_mut_parts(&mut self) -> &mut [T::Scalar; $arity] {
                self.as_array_mut()
            }
        }

        impl<T> FromVectorParts<T::Scalar, $arity> for $base_type_name<T>
        where
            T: Unit,
            T::Scalar: VectorScalar,
        {
            #[inline]
            fn from_parts(parts: [T::Scalar; $arity]) -> Self {
                Self::from_array(parts)
            }
        }

        impl<T> CreateFrom for $base_type_name<T>
        where
            T: Unit,
            <T::Scalar as Scalar>::$raw_name: CreateFrom,
        {
            #[inline]
            fn create_from<B>(reader: &mut Reader<B>) -> Self
            where
                B: BufferRef,
            {
                Self::from_raw(
                    <<T::Scalar as Scalar>::$raw_name as CreateFrom>::create_from(reader),
                )
            }
        }

        impl<T> ReadFrom for $base_type_name<T>
        where
            T: Unit,
            <T::Scalar as Scalar>::$raw_name: ReadFrom,
        {
            #[inline]
            fn read_from<B>(&mut self, reader: &mut Reader<B>)
            where
                B: BufferRef,
            {
                self.as_raw_mut().read_from(reader);
            }
        }

        impl<T> WriteInto for $base_type_name<T>
        where
            T: Unit,
            <T::Scalar as Scalar>::$raw_name: WriteInto,
        {
            #[inline]
            fn write_into<B>(&self, writer: &mut Writer<B>)
            where
                B: BufferMut,
            {
                self.as_raw().write_into(writer);
            }
        }
    };
}

macro_rules! impl_encase_matrix {
    ($base_type_name:ident, $raw_name:ident, $n:literal) => {
        impl<T> ShaderType for $base_type_name<T>
        where
            T: FloatScalar,
            T::$raw_name: ShaderType,
        {
            type ExtraMetadata = <<Self as ToRaw>::Raw as ShaderType>::ExtraMetadata;
            const METADATA: encase::private::Metadata<Self::ExtraMetadata> =
                <<Self as ToRaw>::Raw as ShaderType>::METADATA;
        }

        impl<T> ShaderSize for $base_type_name<T>
        where
            T: FloatScalar,
            T::$raw_name: ShaderType,
        {
        }

        impl<T> FromMatrixParts<T, $n, $n> for $base_type_name<T>
        where
            T: FloatScalar + MatrixScalar,
            T::$raw_name: FromMatrixParts<T, $n, $n>,
        {
            fn from_parts(parts: [[T; $n]; $n]) -> Self {
                Self::from_raw(<<Self as ToRaw>::Raw>::from_parts(parts))
            }
        }

        impl<T> AsRefMatrixParts<T, $n, $n> for $base_type_name<T>
        where
            T: FloatScalar + MatrixScalar,
            T::$raw_name: AsRefMatrixParts<T, $n, $n>,
        {
            fn as_ref_parts(&self) -> &[[T; $n]; $n] {
                self.as_raw().as_ref_parts()
            }
        }

        impl<T> AsMutMatrixParts<T, $n, $n> for $base_type_name<T>
        where
            T: FloatScalar + MatrixScalar,
            T::$raw_name: AsMutMatrixParts<T, $n, $n>,
        {
            fn as_mut_parts(&mut self) -> &mut [[T; $n]; $n] {
                self.as_raw_mut().as_mut_parts()
            }
        }

        impl<T> CreateFrom for $base_type_name<T>
        where
            T: FloatScalar + MatrixScalar,
            T::$raw_name: CreateFrom,
        {
            #[inline]
            fn create_from<B>(reader: &mut Reader<B>) -> Self
            where
                B: BufferRef,
            {
                Self::from_raw(
                    <<T::Scalar as FloatScalar>::$raw_name as CreateFrom>::create_from(reader),
                )
            }
        }

        impl<T> ReadFrom for $base_type_name<T>
        where
            T: FloatScalar + MatrixScalar,
            T::$raw_name: ReadFrom,
        {
            #[inline]
            fn read_from<B>(&mut self, reader: &mut Reader<B>)
            where
                B: BufferRef,
            {
                self.as_raw_mut().read_from(reader);
            }
        }

        impl<T> WriteInto for $base_type_name<T>
        where
            T: FloatScalar + MatrixScalar,
            T::$raw_name: WriteInto,
        {
            #[inline]
            fn write_into<B>(&self, writer: &mut Writer<B>)
            where
                B: BufferMut,
            {
                self.as_raw().write_into(writer);
            }
        }
    };
}

impl_encase_vector!(Vector2, Vec2, 2);
impl_encase_vector!(Vector3, Vec3, 3);
impl_encase_vector!(Vector4, Vec4, 4);
impl_encase_vector!(Point2, Vec2, 2);
impl_encase_vector!(Point3, Vec3, 3);
impl_encase_vector!(Point4, Vec4, 4);
impl_encase_vector!(Size2, Vec2, 2);
impl_encase_vector!(Size3, Vec3, 3);

impl_encase_matrix!(Matrix2, Mat2, 2);
impl_encase_matrix!(Matrix3, Mat3, 3);
impl_encase_matrix!(Matrix4, Mat4, 4);

#[cfg(test)]
mod tests {
    use encase::{
        matrix::MatrixScalar,
        private::{CreateFrom, ReadFrom, WriteInto},
        vector::VectorScalar,
    };

    use crate::prelude::*;

    enum F32 {}
    impl crate::Unit for F32 {
        type Scalar = f32;
    }
    enum I32 {}
    impl crate::Unit for I32 {
        type Scalar = i32;
    }
    enum U32 {}
    impl crate::Unit for U32 {
        type Scalar = u32;
    }

    fn assert_is_shader_type<T: encase::ShaderType + ReadFrom + WriteInto + CreateFrom>(_: T) {}

    fn assert_is_vector<
        T: encase::vector::FromVectorParts<S, N>
            + encase::vector::AsRefVectorParts<S, N>
            + encase::vector::AsMutVectorParts<S, N>
            + encase::internal::CreateFrom
            + encase::internal::ReadFrom
            + encase::internal::WriteInto,
        S: VectorScalar,
        const N: usize,
    >(
        _: T,
    ) {
    }

    fn assert_is_matrix<
        T: encase::matrix::FromMatrixParts<S, N, N>
            + encase::matrix::AsRefMatrixParts<S, N, N>
            + encase::matrix::AsMutMatrixParts<S, N, N>
            + encase::internal::CreateFrom
            + encase::internal::ReadFrom
            + encase::internal::WriteInto,
        S: MatrixScalar,
        const N: usize,
    >(
        _: T,
    ) {
    }

    #[test]
    fn static_assertions() {
        assert_is_shader_type(Vector2::<F32>::ZERO);
        assert_is_shader_type(Vector3::<F32>::ZERO);
        assert_is_shader_type(Vector4::<F32>::ZERO);
        assert_is_shader_type(Point2::<F32>::ZERO);
        assert_is_shader_type(Point3::<F32>::ZERO);
        assert_is_shader_type(Point4::<F32>::ZERO);
        assert_is_shader_type(Size2::<F32>::ZERO);
        assert_is_shader_type(Size3::<F32>::ZERO);

        assert_is_shader_type(Vector2::<I32>::ZERO);
        assert_is_shader_type(Vector3::<I32>::ZERO);
        assert_is_shader_type(Vector4::<I32>::ZERO);
        assert_is_shader_type(Point2::<I32>::ZERO);
        assert_is_shader_type(Point3::<I32>::ZERO);
        assert_is_shader_type(Point4::<I32>::ZERO);
        assert_is_shader_type(Size2::<I32>::ZERO);
        assert_is_shader_type(Size3::<I32>::ZERO);

        assert_is_shader_type(Vector2::<U32>::ZERO);
        assert_is_shader_type(Vector3::<U32>::ZERO);
        assert_is_shader_type(Vector4::<U32>::ZERO);
        assert_is_shader_type(Point2::<U32>::ZERO);
        assert_is_shader_type(Point3::<U32>::ZERO);
        assert_is_shader_type(Point4::<U32>::ZERO);
        assert_is_shader_type(Size2::<U32>::ZERO);
        assert_is_shader_type(Size3::<U32>::ZERO);

        assert_is_shader_type(Matrix2::<f32>::IDENTITY);
        assert_is_shader_type(Matrix3::<f32>::IDENTITY);
        assert_is_shader_type(Matrix4::<f32>::IDENTITY);

        assert_is_vector::<_, f32, 2>(Vector2::<F32>::ZERO);
        assert_is_vector::<_, i32, 2>(Vector2::<I32>::ZERO);
        assert_is_vector::<_, u32, 2>(Vector2::<U32>::ZERO);
        assert_is_vector::<_, f32, 3>(Vector3::<F32>::ZERO);
        assert_is_vector::<_, i32, 3>(Vector3::<I32>::ZERO);
        assert_is_vector::<_, u32, 3>(Vector3::<U32>::ZERO);
        assert_is_vector::<_, f32, 4>(Vector4::<F32>::ZERO);
        assert_is_vector::<_, i32, 4>(Vector4::<I32>::ZERO);
        assert_is_vector::<_, u32, 4>(Vector4::<U32>::ZERO);
        assert_is_vector::<_, f32, 2>(Point2::<F32>::ZERO);
        assert_is_vector::<_, i32, 2>(Point2::<I32>::ZERO);
        assert_is_vector::<_, u32, 2>(Point2::<U32>::ZERO);
        assert_is_vector::<_, f32, 3>(Point3::<F32>::ZERO);
        assert_is_vector::<_, i32, 3>(Point3::<I32>::ZERO);
        assert_is_vector::<_, u32, 3>(Point3::<U32>::ZERO);
        assert_is_vector::<_, f32, 4>(Point4::<F32>::ZERO);
        assert_is_vector::<_, i32, 4>(Point4::<I32>::ZERO);
        assert_is_vector::<_, u32, 4>(Point4::<U32>::ZERO);
        assert_is_vector::<_, f32, 2>(Size2::<F32>::ZERO);
        assert_is_vector::<_, i32, 2>(Size2::<I32>::ZERO);
        assert_is_vector::<_, u32, 2>(Size2::<U32>::ZERO);
        assert_is_vector::<_, f32, 3>(Size3::<F32>::ZERO);
        assert_is_vector::<_, i32, 3>(Size3::<I32>::ZERO);
        assert_is_vector::<_, u32, 3>(Size3::<U32>::ZERO);

        assert_is_matrix::<_, f32, 2>(Matrix2::<f32>::ZERO);
        assert_is_matrix::<_, f32, 3>(Matrix3::<f32>::ZERO);
        assert_is_matrix::<_, f32, 4>(Matrix4::<f32>::ZERO);
    }
}

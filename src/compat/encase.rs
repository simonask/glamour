use crate::{
    Matrix2, Matrix3, Matrix4, Point2, Point3, Point4, Scalar, Size2, Size3, Unit, Vector2,
    Vector3, Vector4,
};

// Adapted from `encase::impl_vector!`.
macro_rules! impl_vector {
    ($name:ident, $n:literal) => {
        impl<T: Unit> encase::private::AsRefVectorParts<T::Scalar, $n> for $name<T>
        where
            T::Scalar: encase::private::VectorScalar,
        {
            #[inline]
            #[cfg_attr(coverage, coverage(off))]
            fn as_ref_parts(&self) -> &[T::Scalar; $n] {
                ::core::convert::AsRef::as_ref(self)
            }
        }
        impl<T: Unit> encase::private::AsMutVectorParts<T::Scalar, $n> for $name<T>
        where
            T::Scalar: encase::private::VectorScalar,
        {
            #[inline]
            #[cfg_attr(coverage, coverage(off))]
            fn as_mut_parts(&mut self) -> &mut [T::Scalar; $n] {
                ::core::convert::AsMut::as_mut(self)
            }
        }
        impl<T: Unit> encase::private::FromVectorParts<T::Scalar, $n> for $name<T>
        where
            T::Scalar: encase::private::VectorScalar,
        {
            #[inline]
            #[cfg_attr(coverage, coverage(off))]
            fn from_parts(parts: [T::Scalar; $n]) -> Self {
                ::core::convert::From::from(parts)
            }
        }
        impl<T: Unit> encase::private::ShaderType for $name<T>
        where
            T::Scalar: encase::private::ShaderSize,
        {
            type ExtraMetadata = ();
            const METADATA: encase::private::Metadata<Self::ExtraMetadata> = {
                let size = encase::private::SizeValue::from(
                    <T::Scalar as encase::private::ShaderSize>::SHADER_SIZE,
                )
                .mul($n);
                let alignment = encase::private::AlignmentValue::from_next_power_of_two_size(size);
                encase::private::Metadata {
                    alignment,
                    has_uniform_min_alignment: false,
                    min_size: size,
                    extra: (),
                    is_pod: true,
                }
            };
        }
        impl<T: Unit> encase::private::ShaderSize for $name<T> where
            T::Scalar: encase::private::ShaderSize
        {
        }

        impl<T: Unit> encase::private::WriteInto for $name<T>
        where
            T::Scalar: encase::private::VectorScalar + encase::private::WriteInto,
        {
            #[inline]
            fn write_into<B: encase::private::BufferMut>(
                &self,
                writer: &mut encase::private::Writer<B>,
            ) {
                let elements =
                    encase::private::AsRefVectorParts::<T::Scalar, $n>::as_ref_parts(self);
                for el in elements {
                    encase::private::WriteInto::write_into(el, writer);
                }
            }
        }
        impl<T: Unit> encase::private::ReadFrom for $name<T>
        where
            T::Scalar: encase::private::VectorScalar + encase::private::ReadFrom,
        {
            #[inline]
            fn read_from<B: encase::private::BufferRef>(
                &mut self,
                reader: &mut encase::private::Reader<B>,
            ) {
                let elements =
                    encase::private::AsMutVectorParts::<T::Scalar, $n>::as_mut_parts(self);
                for el in elements {
                    encase::private::ReadFrom::read_from(el, reader);
                }
            }
        }
        impl<T: Unit> encase::private::CreateFrom for $name<T>
        where
            T::Scalar: encase::private::VectorScalar + encase::private::CreateFrom,
        {
            #[inline]
            fn create_from<B: encase::private::BufferRef>(
                reader: &mut encase::private::Reader<B>,
            ) -> Self {
                let elements =
                    ::core::array::from_fn(|_| encase::private::CreateFrom::create_from(reader));
                encase::private::FromVectorParts::<T::Scalar, $n>::from_parts(elements)
            }
        }
    };
}

impl_vector!(Vector2, 2);
impl_vector!(Vector3, 3);
impl_vector!(Vector4, 4);
impl_vector!(Point2, 2);
impl_vector!(Point3, 3);
impl_vector!(Point4, 4);
impl_vector!(Size2, 2);
impl_vector!(Size3, 3);

encase::impl_matrix!(2, 2, Matrix2<T>; (T: Scalar); using AsRef AsMut From);
encase::impl_matrix!(3, 3, Matrix3<T>; (T: Scalar); using AsRef AsMut From);
encase::impl_matrix!(4, 4, Matrix4<T>; (T: Scalar); using AsRef AsMut From);

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

    #[test]
    fn create_from2() {
        let data = [1.0f32, 2.0];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let v2: Vector2<F32> = buffer.create().unwrap();
        assert_eq!(v2, Vector2::<F32>::new(1.0, 2.0));
    }

    #[test]
    fn read_from2() {
        let data = [1.0f32, 2.0];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let mut v2: Vector2<F32> = Vector2::ZERO;
        buffer.read(&mut v2).unwrap();
        assert_eq!(v2, Vector2::<F32>::new(1.0, 2.0));
    }

    #[test]
    fn write_into2() {
        let mut data = [0.0f32, 0.0];
        let mut buffer =
            encase::DynamicUniformBuffer::new(bytemuck::cast_slice_mut::<_, u8>(&mut data));
        let v2 = Vector2::<F32>::new(1.0, 2.0);
        let n = buffer.write(&v2).unwrap();
        assert_eq!(n, 0);
        assert_eq!(v2, Vector2::<F32>::new(1.0, 2.0));
    }

    #[test]
    fn create_from3() {
        let data = [1.0f32, 2.0, 3.0];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let v3: Vector3<F32> = buffer.create().unwrap();
        assert_eq!(v3, Vector3::<F32>::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn read_from3() {
        let data = [1.0f32, 2.0, 3.0];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let mut v3: Vector3<F32> = Vector3::ZERO;
        buffer.read(&mut v3).unwrap();
        assert_eq!(v3, Vector3::<F32>::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn write_into3() {
        let mut data = [0.0f32, 0.0, 0.0];
        let mut buffer =
            encase::DynamicUniformBuffer::new(bytemuck::cast_slice_mut::<_, u8>(&mut data));
        let v3 = Vector3::<F32>::new(1.0, 2.0, 3.0);
        let n = buffer.write(&v3).unwrap();
        assert_eq!(n, 0);
        assert_eq!(v3, Vector3::<F32>::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn create_from4() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let v4: Vector4<F32> = buffer.create().unwrap();
        assert_eq!(v4, Vector4::<F32>::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn read_from4() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let mut v4: Vector4<F32> = Vector4::ZERO;
        buffer.read(&mut v4).unwrap();
        assert_eq!(v4, Vector4::<F32>::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn write_into4() {
        let mut data = [0.0f32, 0.0, 0.0, 0.0];
        let mut buffer =
            encase::DynamicUniformBuffer::new(bytemuck::cast_slice_mut::<_, u8>(&mut data));
        let v4 = Vector4::<F32>::new(1.0, 2.0, 3.0, 4.0);
        let n = buffer.write(&v4).unwrap();
        assert_eq!(n, 0);
        assert_eq!(v4, Vector4::<F32>::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn read_from_point2() {
        let data = [1.0f32, 2.0];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let mut x: Point2<F32> = Point2::ZERO;
        buffer.read(&mut x).unwrap();
        assert_eq!(x, Point2::<F32>::new(1.0, 2.0));
    }

    #[test]
    fn read_from_point3() {
        let data = [1.0f32, 2.0, 3.0];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let mut x: Point3<F32> = Point3::ZERO;
        buffer.read(&mut x).unwrap();
        assert_eq!(x, Point3::<F32>::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn read_from_point4() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let mut x: Point4<F32> = Point4::ZERO;
        buffer.read(&mut x).unwrap();
        assert_eq!(x, Point4::<F32>::new(1.0, 2.0, 3.0, 4.0));
    }

    #[test]
    fn read_from_size2() {
        let data = [1.0f32, 2.0];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let mut x: Size2<F32> = Size2::ZERO;
        buffer.read(&mut x).unwrap();
        assert_eq!(x, Size2::<F32>::new(1.0, 2.0));
    }

    #[test]
    fn read_from_size3() {
        let data = [1.0f32, 2.0, 3.0];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let mut x: Size3<F32> = Size3::ZERO;
        buffer.read(&mut x).unwrap();
        assert_eq!(x, Size3::<F32>::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn read_from_mat2() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let mut x: Matrix2<f32> = Matrix2::IDENTITY;
        buffer.read(&mut x).unwrap();
        assert_eq!(x, Matrix2::from(data));
    }

    #[test]
    fn read_from_mat3() {
        // Here it gets interesting: `mat3` has a different memory representation in shaders (represented as mat4)
        let data = [
            1.0f32, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0, 7.0, 8.0, 9.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let mut x: Matrix3<f32> = Matrix3::IDENTITY;
        buffer.read(&mut x).unwrap();
        assert_eq!(
            x,
            Matrix3::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        );
    }

    #[test]
    fn read_from_mat4() {
        let data = [
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ];
        let mut buffer = encase::DynamicUniformBuffer::new(bytemuck::cast_slice::<_, u8>(&data));
        let mut x: Matrix4<f32> = Matrix4::IDENTITY;
        buffer.read(&mut x).unwrap();
        assert_eq!(x, Matrix4::from(data));
    }
}

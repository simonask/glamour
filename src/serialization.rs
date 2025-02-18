use crate::{
    Angle, Box2, Box3, FloatScalar, Matrix2, Matrix3, Matrix4, Point2, Point3, Point4, Rect, Size2,
    Size3, Transform2, Transform3, Unit, Vector2, Vector3, Vector4,
};
use serde::ser::SerializeStruct;

macro_rules! impl_vectorlike {
    ($t:ident, $count:literal, $($field:ident),+) => {
        impl<T: Unit> serde::Serialize for $t<T> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut map = serializer.serialize_struct(stringify!($t), $count)?;
                $(
                    map.serialize_field(stringify!($field), &self.$field)?;
                )*
                map.end()
            }
        }

        impl<'de, T: Unit> serde::Deserialize<'de> for $t<T> {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct Visitor<T> {
                    _marker: core::marker::PhantomData<T>,
                }

                impl<'de, T: Unit> serde::de::Visitor<'de> for Visitor<T> {
                    type Value = $t<T>;

                    fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                        write!(formatter, concat!("a mapping or a sequence of length ", stringify!($count)))
                    }

                    fn visit_map<A: serde::de::MapAccess<'de>>(
                        self,
                        mut map: A,
                    ) -> Result<Self::Value, A::Error> {
                        $(
                            let mut $field = None;
                        )*
                        while let Some(key) = map.next_key_seed(AllowedFields {
                            allow_size: false,
                            len: $count,
                        })? {
                            match key {
                                $(
                                    Field::$field => {
                                        if $field.is_some() {
                                            return Err(serde::de::Error::duplicate_field(stringify!($field)));
                                        }
                                        $field = Some(map.next_value()?);
                                    }
                                )*
                                // Allowed fields tracked by AllowedFields
                                _ => unreachable!()
                            }
                        }
                        $(
                            let $field = $field.ok_or_else(|| serde::de::Error::missing_field(stringify!($field)))?;
                        )*

                        Ok($t { $($field),* })
                    }

                    #[allow(unused_assignments)]
                    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                    where
                        A: serde::de::SeqAccess<'de>,
                    {
                        let mut i = 0;
                        $(
                            let $field = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                            i += 1;
                        )*
                        Ok($t { $($field),* })
                    }
                }

                deserializer.deserialize_struct(
                    stringify!($t),
                    &ALLOWED_FIELDS_VECTORLIKE[..$count],
                    Visitor {
                        _marker: core::marker::PhantomData,
                    },
                )
            }
        }
    };
}

macro_rules! impl_sizelike {
    ($t:ident, $count:literal, $($field:ident | $field_alt:ident),+) => {
        impl<T: Unit> serde::Serialize for $t<T> {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                let mut map = serializer.serialize_struct(stringify!($t), $count)?;
                $(
                    map.serialize_field(stringify!($field), &self.$field)?;
                )*
                map.end()
            }
        }

        impl<'de, T: Unit> serde::Deserialize<'de> for $t<T> {
            fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
                struct Visitor<T> {
                    _marker: core::marker::PhantomData<T>,
                }

                impl<'de, T: Unit> serde::de::Visitor<'de> for Visitor<T> {
                    type Value = $t<T>;

                    fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                        write!(formatter, concat!("a mapping or a sequence of length ", stringify!($count)))
                    }

                    fn visit_map<A: serde::de::MapAccess<'de>>(
                        self,
                        mut map: A,
                    ) -> Result<Self::Value, A::Error> {
                        $(
                            let mut $field = None;
                        )*
                        while let Some(key) = map.next_key_seed(AllowedFields {
                            allow_size: true,
                            len: $count,
                        })? {
                            match key {
                                $(
                                    Field::$field | Field::$field_alt => {
                                        if $field.is_some() {
                                            return Err(serde::de::Error::duplicate_field(stringify!($field)));
                                        }
                                        $field = Some(map.next_value()?);
                                    }
                                )*
                                // Allowed fields tracked by AllowedFields
                                _ => unreachable!()
                            }
                        }
                        $(
                            let $field = $field.ok_or_else(|| serde::de::Error::missing_field(stringify!($field)))?;
                        )*

                        Ok($t { $($field),* })
                    }

                    #[allow(unused_assignments)]
                    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                    where
                        A: serde::de::SeqAccess<'de>,
                    {
                        let mut i = 0;
                        $(
                            let $field = seq.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(i, &self))?;
                            i += 1;
                        )*
                        Ok($t { $($field),* })
                    }
                }

                deserializer.deserialize_struct(
                    stringify!($t),
                    &ALLOWED_FIELDS_SIZELIKE[..$count * 2],
                    Visitor {
                        _marker: core::marker::PhantomData,
                    },
                )
            }
        }
    };
}

impl_vectorlike!(Vector2, 2, x, y);
impl_vectorlike!(Vector3, 3, x, y, z);
impl_vectorlike!(Vector4, 4, x, y, z, w);
impl_vectorlike!(Point2, 2, x, y);
impl_vectorlike!(Point3, 3, x, y, z);
impl_vectorlike!(Point4, 4, x, y, z, w);
impl_sizelike!(Size2, 2, width | x, height | y);
impl_sizelike!(Size3, 3, width | x, height | y, depth | z);

#[allow(non_camel_case_types)]
enum Field {
    x,
    y,
    z,
    w,
    width,
    height,
    depth,
}

struct AllowedFields {
    len: usize,
    allow_size: bool,
}

const ALLOWED_FIELDS_VECTORLIKE: &[&str] = &["x", "y", "z", "w"];
const ALLOWED_FIELDS_SIZELIKE: &[&str] = &["x", "width", "y", "height", "z", "depth"];

impl<'de> serde::de::DeserializeSeed<'de> for AllowedFields {
    type Value = Field;

    #[inline]
    fn deserialize<D: serde::de::Deserializer<'de>>(
        self,
        deserializer: D,
    ) -> Result<Self::Value, D::Error> {
        struct Visitor {
            allowed: AllowedFields,
        }

        impl serde::de::Visitor<'_> for Visitor {
            type Value = Field;

            #[inline]
            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(formatter, "a field name")
            }

            #[inline]
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                match value {
                    "x" => Ok(Field::x),
                    "y" => Ok(Field::y),
                    "z" if self.allowed.len >= 3 => Ok(Field::z),
                    "w" if self.allowed.len >= 4 => Ok(Field::w),
                    "width" if self.allowed.allow_size => Ok(Field::width),
                    "height" if self.allowed.allow_size => Ok(Field::height),
                    "depth" if self.allowed.allow_size && self.allowed.len >= 3 => Ok(Field::depth),
                    _ => {
                        let allowed_fields = if self.allowed.allow_size {
                            &ALLOWED_FIELDS_SIZELIKE[..self.allowed.len * 2]
                        } else {
                            &ALLOWED_FIELDS_VECTORLIKE[..self.allowed.len]
                        };
                        Err(serde::de::Error::unknown_field(value, allowed_fields))
                    }
                }
            }
        }

        deserializer.deserialize_identifier(Visitor { allowed: self })
    }
}

impl<Src, Dst> serde::Serialize for Transform2<Src, Dst>
where
    Src: Unit,
    Dst: Unit<Scalar = Src::Scalar>,
    Src::Scalar: FloatScalar,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.matrix.serialize(serializer)
    }
}

impl<'de, Src, Dst> serde::Deserialize<'de> for Transform2<Src, Dst>
where
    Src: Unit,
    Dst: Unit<Scalar = Src::Scalar>,
    Src::Scalar: FloatScalar,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let matrix: Matrix3<Src::Scalar> = serde::Deserialize::deserialize(deserializer)?;
        Ok(Transform2::from_matrix_unchecked(matrix))
    }
}

impl<Src, Dst> serde::Serialize for Transform3<Src, Dst>
where
    Src: Unit,
    Dst: Unit<Scalar = Src::Scalar>,
    Src::Scalar: FloatScalar,
{
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.matrix.serialize(serializer)
    }
}

impl<'de, Src, Dst> serde::Deserialize<'de> for Transform3<Src, Dst>
where
    Src: Unit,
    Dst: Unit<Scalar = Src::Scalar>,
    Src::Scalar: FloatScalar,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let matrix: Matrix4<Src::Scalar> = serde::Deserialize::deserialize(deserializer)?;
        Ok(Transform3::from_matrix_unchecked(matrix))
    }
}

impl<T: FloatScalar> serde::Serialize for Matrix2<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let cols: [T; 4] = (*self).into();
        cols.serialize(serializer)
    }
}

impl<'de, T: FloatScalar> serde::Deserialize<'de> for Matrix2<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let cols: [T; 4] = serde::Deserialize::deserialize(deserializer)?;
        Ok(cols.into())
    }
}

impl<T: FloatScalar> serde::Serialize for Matrix3<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let cols: [T; 9] = (*self).into();
        cols.serialize(serializer)
    }
}

impl<'de, T: FloatScalar> serde::Deserialize<'de> for Matrix3<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let cols: [T; 9] = serde::Deserialize::deserialize(deserializer)?;
        Ok(cols.into())
    }
}

impl<T: FloatScalar> serde::Serialize for Matrix4<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let cols: [T; 16] = (*self).into();
        cols.serialize(serializer)
    }
}

impl<'de, T: FloatScalar> serde::Deserialize<'de> for Matrix4<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let cols: [T; 16] = serde::Deserialize::deserialize(deserializer)?;
        Ok(cols.into())
    }
}

impl<T: FloatScalar> serde::Serialize for Angle<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.radians.serialize(serializer)
    }
}

impl<'de, T: FloatScalar> serde::Deserialize<'de> for Angle<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let radians: T = serde::Deserialize::deserialize(deserializer)?;
        Ok(Angle { radians })
    }
}

impl<T: Unit> serde::Serialize for Rect<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_struct("Rect", 2)?;
        map.serialize_field("origin", &self.origin)?;
        map.serialize_field("size", &self.size)?;
        map.end()
    }
}

enum RectField {
    Origin,
    Size,
}

impl<'de> serde::Deserialize<'de> for RectField {
    #[inline]
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;

        impl serde::de::Visitor<'_> for Visitor {
            type Value = RectField;

            #[inline]
            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(formatter, "a field name")
            }

            #[inline]
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                match value {
                    "origin" => Ok(RectField::Origin),
                    "size" => Ok(RectField::Size),
                    _ => Err(serde::de::Error::unknown_field(value, &["origin", "size"])),
                }
            }
        }

        deserializer.deserialize_identifier(Visitor)
    }
}

impl<'de, T: Unit> serde::Deserialize<'de> for Rect<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor<T> {
            _marker: core::marker::PhantomData<T>,
        }

        impl<'de, T: Unit> serde::de::Visitor<'de> for Visitor<T> {
            type Value = Rect<T>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(formatter, "a mapping with `origin` and `size` fields")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut origin = None;
                let mut size = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        RectField::Origin => {
                            if origin.is_some() {
                                return Err(serde::de::Error::duplicate_field("origin"));
                            }
                            origin = Some(map.next_value()?);
                        }
                        RectField::Size => {
                            if size.is_some() {
                                return Err(serde::de::Error::duplicate_field("size"));
                            }
                            size = Some(map.next_value()?);
                        }
                    }
                }
                let origin = origin.ok_or_else(|| serde::de::Error::missing_field("origin"))?;
                let size = size.ok_or_else(|| serde::de::Error::missing_field("size"))?;
                Ok(Rect { origin, size })
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let origin = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let size = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                Ok(Rect { origin, size })
            }
        }

        deserializer.deserialize_struct(
            "Rect",
            &["origin", "size"],
            Visitor {
                _marker: core::marker::PhantomData,
            },
        )
    }
}

impl<T: Unit> serde::Serialize for Box2<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_struct("Box2", 2)?;
        map.serialize_field("min", &self.min)?;
        map.serialize_field("max", &self.max)?;
        map.end()
    }
}

enum BoxField {
    Min,
    Max,
}

impl<'de> serde::Deserialize<'de> for BoxField {
    #[inline]
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;

        impl serde::de::Visitor<'_> for Visitor {
            type Value = BoxField;

            #[inline]
            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(formatter, "a field name")
            }

            #[inline]
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                match value {
                    "min" => Ok(BoxField::Min),
                    "max" => Ok(BoxField::Max),
                    _ => Err(serde::de::Error::unknown_field(value, &["min", "max"])),
                }
            }
        }

        deserializer.deserialize_identifier(Visitor)
    }
}

impl<'de, T: Unit> serde::Deserialize<'de> for Box2<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor<T> {
            _marker: core::marker::PhantomData<T>,
        }

        impl<'de, T: Unit> serde::de::Visitor<'de> for Visitor<T> {
            type Value = Box2<T>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(formatter, "a mapping with `min` and `max` fields")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut min = None;
                let mut max = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        BoxField::Min => {
                            if min.is_some() {
                                return Err(serde::de::Error::duplicate_field("min"));
                            }
                            min = Some(map.next_value()?);
                        }
                        BoxField::Max => {
                            if max.is_some() {
                                return Err(serde::de::Error::duplicate_field("max"));
                            }
                            max = Some(map.next_value()?);
                        }
                    }
                }
                let min = min.ok_or_else(|| serde::de::Error::missing_field("origin"))?;
                let max = max.ok_or_else(|| serde::de::Error::missing_field("size"))?;
                Ok(Box2 { min, max })
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let min = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let max = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                Ok(Box2 { min, max })
            }
        }

        deserializer.deserialize_struct(
            "Box2",
            &["min", "max"],
            Visitor {
                _marker: core::marker::PhantomData,
            },
        )
    }
}

impl<T: Unit> serde::Serialize for Box3<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_struct("Box3", 2)?;
        map.serialize_field("min", &self.min)?;
        map.serialize_field("max", &self.max)?;
        map.end()
    }
}

impl<'de, T: Unit> serde::Deserialize<'de> for Box3<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor<T> {
            _marker: core::marker::PhantomData<T>,
        }

        impl<'de, T: Unit> serde::de::Visitor<'de> for Visitor<T> {
            type Value = Box3<T>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(formatter, "a mapping with `min` and `max` fields")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut min = None;
                let mut max = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        BoxField::Min => {
                            if min.is_some() {
                                return Err(serde::de::Error::duplicate_field("min"));
                            }
                            min = Some(map.next_value()?);
                        }
                        BoxField::Max => {
                            if max.is_some() {
                                return Err(serde::de::Error::duplicate_field("max"));
                            }
                            max = Some(map.next_value()?);
                        }
                    }
                }
                let min = min.ok_or_else(|| serde::de::Error::missing_field("origin"))?;
                let max = max.ok_or_else(|| serde::de::Error::missing_field("size"))?;
                Ok(Box3 { min, max })
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let min = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let max = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                Ok(Box3 { min, max })
            }
        }

        deserializer.deserialize_struct(
            "Box3",
            &["min", "max"],
            Visitor {
                _marker: core::marker::PhantomData,
            },
        )
    }
}

#[cfg(test)]
#[allow(clippy::too_many_lines)]
mod tests {
    use crate::point;

    use super::*;
    use serde_test::*;

    #[test]
    fn vector2() {
        assert_ser_tokens(
            &Vector2::<f32> { x: 1.0, y: 2.0 },
            &[
                Token::Struct {
                    name: "Vector2",
                    len: 2,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::StructEnd,
            ],
        );
        assert_ser_tokens(
            &Vector2::<i64> { x: 1, y: 2 },
            &[
                Token::Struct {
                    name: "Vector2",
                    len: 2,
                },
                Token::Str("x"),
                Token::I64(1),
                Token::Str("y"),
                Token::I64(2),
                Token::StructEnd,
            ],
        );
        assert_de_tokens(
            &Vector2::<f32> { x: 1.0, y: 2.0 },
            &[
                Token::Struct {
                    name: "Vector2",
                    len: 2,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::StructEnd,
            ],
        );
        assert_de_tokens(
            &Vector2::<f32> { x: 1.0, y: 2.0 },
            &[
                Token::Seq { len: Some(2) },
                Token::F32(1.0),
                Token::F32(2.0),
                Token::SeqEnd,
            ],
        );

        assert_de_tokens_error::<Vector2<f32>>(
            &[Token::Seq { len: Some(1) }, Token::F32(1.0), Token::SeqEnd],
            "invalid length 1, expected a mapping or a sequence of length 2",
        );
        assert_de_tokens_error::<Vector2<f32>>(
            &[Token::Seq { len: None }, Token::F32(1.0), Token::SeqEnd],
            "invalid length 1, expected a mapping or a sequence of length 2",
        );
        assert_de_tokens_error::<Vector2<f32>>(
            &[Token::Map { len: Some(2) }, Token::MapEnd],
            "missing field `x`",
        );
        assert_de_tokens_error::<Vector2<f32>>(
            &[
                Token::Struct {
                    name: "Vector2",
                    len: 2,
                },
                Token::StructEnd,
            ],
            "missing field `x`",
        );
        assert_de_tokens_error::<Vector2<i32>>(
            &[
                Token::Struct {
                    name: "Vector2",
                    len: 2,
                },
                Token::I32(1),
                Token::StructEnd,
            ],
            "invalid type: integer `1`, expected a field name",
        );
        assert_de_tokens_error::<Vector2<f32>>(
            &[
                Token::Struct {
                    name: "Vector2",
                    len: 2,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("z"),
                Token::F32(2.0),
            ],
            "unknown field `z`, expected `x` or `y`",
        );
        assert_de_tokens_error::<Vector2<f32>>(
            &[
                Token::Struct {
                    name: "Vector2",
                    len: 3,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::Str("z"),
                Token::F32(3.0),
            ],
            "unknown field `z`, expected `x` or `y`",
        );
    }

    #[test]
    fn vector3() {
        assert_ser_tokens(
            &Vector3::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            &[
                Token::Struct {
                    name: "Vector3",
                    len: 3,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::Str("z"),
                Token::F32(3.0),
                Token::StructEnd,
            ],
        );
        assert_ser_tokens(
            &Vector3::<i64> { x: 1, y: 2, z: 3 },
            &[
                Token::Struct {
                    name: "Vector3",
                    len: 3,
                },
                Token::Str("x"),
                Token::I64(1),
                Token::Str("y"),
                Token::I64(2),
                Token::Str("z"),
                Token::I64(3),
                Token::StructEnd,
            ],
        );
        assert_de_tokens(
            &Vector3::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            &[
                Token::Struct {
                    name: "Vector3",
                    len: 3,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::Str("z"),
                Token::F32(3.0),
                Token::StructEnd,
            ],
        );
        assert_de_tokens(
            &Vector3::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0,
            },
            &[
                Token::Seq { len: Some(3) },
                Token::F32(1.0),
                Token::F32(2.0),
                Token::F32(3.0),
                Token::SeqEnd,
            ],
        );

        assert_de_tokens_error::<Vector3<f32>>(
            &[Token::Seq { len: Some(1) }, Token::F32(1.0), Token::SeqEnd],
            "invalid length 1, expected a mapping or a sequence of length 3",
        );
        assert_de_tokens_error::<Vector3<f32>>(
            &[
                Token::Seq { len: Some(2) },
                Token::F32(1.0),
                Token::F32(2.0),
                Token::SeqEnd,
            ],
            "invalid length 2, expected a mapping or a sequence of length 3",
        );
        assert_de_tokens_error::<Vector3<f32>>(
            &[Token::Seq { len: None }, Token::F32(1.0), Token::SeqEnd],
            "invalid length 1, expected a mapping or a sequence of length 3",
        );
        assert_de_tokens_error::<Vector3<f32>>(
            &[Token::Map { len: Some(2) }, Token::MapEnd],
            "missing field `x`",
        );
        assert_de_tokens_error::<Vector3<f32>>(
            &[
                Token::Struct {
                    name: "Vector3",
                    len: 2,
                },
                Token::StructEnd,
            ],
            "missing field `x`",
        );
        assert_de_tokens_error::<Vector3<i32>>(
            &[
                Token::Struct {
                    name: "Vector3",
                    len: 2,
                },
                Token::I32(1),
                Token::StructEnd,
            ],
            "invalid type: integer `1`, expected a field name",
        );
        assert_de_tokens_error::<Vector3<f32>>(
            &[
                Token::Struct {
                    name: "Vector3",
                    len: 2,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("w"),
                Token::F32(2.0),
            ],
            "unknown field `w`, expected one of `x`, `y`, `z`",
        );
        assert_de_tokens_error::<Vector3<f32>>(
            &[
                Token::Struct {
                    name: "Vector3",
                    len: 4,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::Str("z"),
                Token::F32(3.0),
                Token::Str("w"),
                Token::F32(4.0),
            ],
            "unknown field `w`, expected one of `x`, `y`, `z`",
        );
    }

    #[test]
    fn vector4() {
        assert_ser_tokens(
            &Vector4::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                w: 4.0,
            },
            &[
                Token::Struct {
                    name: "Vector4",
                    len: 4,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::Str("z"),
                Token::F32(3.0),
                Token::Str("w"),
                Token::F32(4.0),
                Token::StructEnd,
            ],
        );
        assert_ser_tokens(
            &Vector4::<i64> {
                x: 1,
                y: 2,
                z: 3,
                w: 4,
            },
            &[
                Token::Struct {
                    name: "Vector4",
                    len: 4,
                },
                Token::Str("x"),
                Token::I64(1),
                Token::Str("y"),
                Token::I64(2),
                Token::Str("z"),
                Token::I64(3),
                Token::Str("w"),
                Token::I64(4),
                Token::StructEnd,
            ],
        );
        assert_de_tokens(
            &Vector4::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                w: 4.0,
            },
            &[
                Token::Struct {
                    name: "Vector4",
                    len: 4,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::Str("z"),
                Token::F32(3.0),
                Token::Str("w"),
                Token::F32(4.0),
                Token::StructEnd,
            ],
        );
        assert_de_tokens(
            &Vector4::<f32> {
                x: 1.0,
                y: 2.0,
                z: 3.0,
                w: 4.0,
            },
            &[
                Token::Seq { len: Some(4) },
                Token::F32(1.0),
                Token::F32(2.0),
                Token::F32(3.0),
                Token::F32(4.0),
                Token::SeqEnd,
            ],
        );

        assert_de_tokens_error::<Vector4<f32>>(
            &[Token::Seq { len: Some(1) }, Token::F32(1.0), Token::SeqEnd],
            "invalid length 1, expected a mapping or a sequence of length 4",
        );
        assert_de_tokens_error::<Vector4<f32>>(
            &[
                Token::Seq { len: Some(2) },
                Token::F32(1.0),
                Token::F32(2.0),
                Token::SeqEnd,
            ],
            "invalid length 2, expected a mapping or a sequence of length 4",
        );
        assert_de_tokens_error::<Vector4<f32>>(
            &[Token::Seq { len: None }, Token::F32(1.0), Token::SeqEnd],
            "invalid length 1, expected a mapping or a sequence of length 4",
        );
        assert_de_tokens_error::<Vector4<f32>>(
            &[Token::Map { len: None }, Token::MapEnd],
            "missing field `x`",
        );
        assert_de_tokens_error::<Vector4<f32>>(
            &[
                Token::Struct {
                    name: "Vector4",
                    len: 2,
                },
                Token::StructEnd,
            ],
            "missing field `x`",
        );
        assert_de_tokens_error::<Vector4<i32>>(
            &[
                Token::Struct {
                    name: "Vector4",
                    len: 2,
                },
                Token::I32(1),
                Token::StructEnd,
            ],
            "invalid type: integer `1`, expected a field name",
        );
        assert_de_tokens_error::<Vector4<f32>>(
            &[
                Token::Struct {
                    name: "Vector4",
                    len: 2,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("q"),
                Token::F32(2.0),
            ],
            "unknown field `q`, expected one of `x`, `y`, `z`, `w`",
        );
        assert_de_tokens_error::<Vector4<f32>>(
            &[
                Token::Struct {
                    name: "Vector4",
                    len: 4,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::Str("z"),
                Token::F32(3.0),
                Token::Str("w"),
                Token::F32(4.0),
                Token::Str("q"),
                Token::F32(5.0),
            ],
            "unknown field `q`, expected one of `x`, `y`, `z`, `w`",
        );
    }

    #[test]
    fn point2() {
        assert_ser_tokens(
            &Point2::<f32> { x: 1.0, y: 2.0 },
            &[
                Token::Struct {
                    name: "Point2",
                    len: 2,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::StructEnd,
            ],
        );
        assert_ser_tokens(
            &Point2::<i64> { x: 1, y: 2 },
            &[
                Token::Struct {
                    name: "Point2",
                    len: 2,
                },
                Token::Str("x"),
                Token::I64(1),
                Token::Str("y"),
                Token::I64(2),
                Token::StructEnd,
            ],
        );
        assert_de_tokens(
            &Point2::<f32> { x: 1.0, y: 2.0 },
            &[
                Token::Struct {
                    name: "Point2",
                    len: 2,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::StructEnd,
            ],
        );
        assert_de_tokens(
            &Point2::<f32> { x: 1.0, y: 2.0 },
            &[
                Token::Seq { len: Some(2) },
                Token::F32(1.0),
                Token::F32(2.0),
                Token::SeqEnd,
            ],
        );

        assert_de_tokens_error::<Point2<f32>>(
            &[Token::Seq { len: Some(1) }, Token::F32(1.0), Token::SeqEnd],
            "invalid length 1, expected a mapping or a sequence of length 2",
        );
        assert_de_tokens_error::<Point2<f32>>(
            &[Token::Seq { len: None }, Token::F32(1.0), Token::SeqEnd],
            "invalid length 1, expected a mapping or a sequence of length 2",
        );
        assert_de_tokens_error::<Point2<f32>>(
            &[Token::Map { len: Some(2) }, Token::MapEnd],
            "missing field `x`",
        );
        assert_de_tokens_error::<Point2<f32>>(
            &[
                Token::Struct {
                    name: "Point2",
                    len: 2,
                },
                Token::StructEnd,
            ],
            "missing field `x`",
        );
        assert_de_tokens_error::<Point2<i32>>(
            &[
                Token::Struct {
                    name: "Point2",
                    len: 2,
                },
                Token::I32(1),
                Token::StructEnd,
            ],
            "invalid type: integer `1`, expected a field name",
        );
        assert_de_tokens_error::<Point2<f32>>(
            &[
                Token::Struct {
                    name: "Point2",
                    len: 2,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("z"),
                Token::F32(2.0),
            ],
            "unknown field `z`, expected `x` or `y`",
        );
        assert_de_tokens_error::<Point2<f32>>(
            &[
                Token::Struct {
                    name: "Point2",
                    len: 3,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::Str("z"),
                Token::F32(3.0),
            ],
            "unknown field `z`, expected `x` or `y`",
        );
    }

    #[test]
    fn size2() {
        assert_ser_tokens(
            &Size2::<f32> {
                width: 1.0,
                height: 2.0,
            },
            &[
                Token::Struct {
                    name: "Size2",
                    len: 2,
                },
                Token::Str("width"),
                Token::F32(1.0),
                Token::Str("height"),
                Token::F32(2.0),
                Token::StructEnd,
            ],
        );
        assert_ser_tokens(
            &Size2::<i64> {
                width: 1,
                height: 2,
            },
            &[
                Token::Struct {
                    name: "Size2",
                    len: 2,
                },
                Token::Str("width"),
                Token::I64(1),
                Token::Str("height"),
                Token::I64(2),
                Token::StructEnd,
            ],
        );
        assert_de_tokens(
            &Size2::<f32> {
                width: 1.0,
                height: 2.0,
            },
            &[
                Token::Struct {
                    name: "Size2",
                    len: 2,
                },
                Token::Str("width"),
                Token::F32(1.0),
                Token::Str("height"),
                Token::F32(2.0),
                Token::StructEnd,
            ],
        );
        assert_de_tokens(
            &Size2::<f32> {
                width: 1.0,
                height: 2.0,
            },
            &[
                Token::Struct {
                    name: "Size2",
                    len: 2,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::StructEnd,
            ],
        );
        assert_de_tokens(
            &Size2::<f32> {
                width: 1.0,
                height: 2.0,
            },
            &[
                Token::Seq { len: Some(2) },
                Token::F32(1.0),
                Token::F32(2.0),
                Token::SeqEnd,
            ],
        );

        assert_de_tokens_error::<Size2<f32>>(
            &[Token::Seq { len: Some(1) }, Token::F32(1.0), Token::SeqEnd],
            "invalid length 1, expected a mapping or a sequence of length 2",
        );
        assert_de_tokens_error::<Size2<f32>>(
            &[Token::Seq { len: None }, Token::F32(1.0), Token::SeqEnd],
            "invalid length 1, expected a mapping or a sequence of length 2",
        );
        assert_de_tokens_error::<Size2<f32>>(
            &[Token::Map { len: Some(2) }, Token::MapEnd],
            "missing field `width`",
        );
        assert_de_tokens_error::<Size2<f32>>(
            &[
                Token::Struct {
                    name: "Size2",
                    len: 2,
                },
                Token::StructEnd,
            ],
            "missing field `width`",
        );
        assert_de_tokens_error::<Size2<i32>>(
            &[
                Token::Struct {
                    name: "Size2",
                    len: 2,
                },
                Token::I32(1),
                Token::StructEnd,
            ],
            "invalid type: integer `1`, expected a field name",
        );
        assert_de_tokens_error::<Size2<f32>>(
            &[
                Token::Struct {
                    name: "Size2",
                    len: 2,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("z"),
                Token::F32(2.0),
            ],
            "unknown field `z`, expected one of `x`, `width`, `y`, `height`",
        );
        assert_de_tokens_error::<Size2<f32>>(
            &[
                Token::Struct {
                    name: "Size2",
                    len: 3,
                },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(2.0),
                Token::Str("z"),
                Token::F32(3.0),
            ],
            "unknown field `z`, expected one of `x`, `width`, `y`, `height`",
        );
    }

    #[test]
    fn rect() {
        let rect = Rect::<f32>::from_origin_and_size((10.0, 20.0), (30.0, 40.0));
        assert_ser_tokens(
            &rect,
            &[
                Token::Struct {
                    name: "Rect",
                    len: 2,
                },
                Token::Str("origin"),
                Token::Struct {
                    name: "Point2",
                    len: 2,
                },
                Token::Str("x"),
                Token::F32(10.0),
                Token::Str("y"),
                Token::F32(20.0),
                Token::StructEnd,
                Token::Str("size"),
                Token::Struct {
                    name: "Size2",
                    len: 2,
                },
                Token::Str("width"),
                Token::F32(30.0),
                Token::Str("height"),
                Token::F32(40.0),
                Token::StructEnd,
                Token::StructEnd,
            ],
        );

        assert_de_tokens(
            &rect,
            &[
                Token::Seq { len: Some(2) },
                Token::Seq { len: Some(2) },
                Token::F32(10.0),
                Token::F32(20.0),
                Token::SeqEnd,
                Token::Seq { len: Some(2) },
                Token::F32(30.0),
                Token::F32(40.0),
                Token::SeqEnd,
                Token::SeqEnd,
            ],
        );

        let serialized = serde_json::to_string(&rect).unwrap();
        assert_eq!(
            serialized,
            r#"{"origin":{"x":10.0,"y":20.0},"size":{"width":30.0,"height":40.0}}"#
        );
        let deserialized: Rect<f32> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(rect, deserialized);

        assert_de_tokens_error::<Rect<f32>>(
            &[Token::Map { len: Some(2) }, Token::I8(123)],
            "invalid type: integer `123`, expected a field name",
        );
        assert_de_tokens_error::<Rect<f32>>(
            &[
                Token::Map { len: Some(2) },
                Token::Str("origin"),
                Token::Map { len: Some(2) },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(1.0),
                Token::MapEnd,
                Token::Str("supposed to be size"),
                Token::Map { len: Some(2) },
            ],
            "unknown field `supposed to be size`, expected `origin` or `size`",
        );
        assert_de_tokens_error::<Rect<f32>>(
            &[Token::I8(123)],
            "invalid type: integer `123`, expected a mapping with `origin` and `size` fields",
        );
    }

    #[test]
    fn box2() {
        let box2 = Box2::<f32>::from_min_max((10.0, 20.0), (30.0, 40.0));
        assert_ser_tokens(
            &box2,
            &[
                Token::Struct {
                    name: "Box2",
                    len: 2,
                },
                Token::Str("min"),
                Token::Struct {
                    name: "Point2",
                    len: 2,
                },
                Token::Str("x"),
                Token::F32(10.0),
                Token::Str("y"),
                Token::F32(20.0),
                Token::StructEnd,
                Token::Str("max"),
                Token::Struct {
                    name: "Point2",
                    len: 2,
                },
                Token::Str("x"),
                Token::F32(30.0),
                Token::Str("y"),
                Token::F32(40.0),
                Token::StructEnd,
                Token::StructEnd,
            ],
        );

        assert_de_tokens(
            &box2,
            &[
                Token::Seq { len: Some(2) },
                Token::Seq { len: Some(2) },
                Token::F32(10.0),
                Token::F32(20.0),
                Token::SeqEnd,
                Token::Seq { len: Some(2) },
                Token::F32(30.0),
                Token::F32(40.0),
                Token::SeqEnd,
                Token::SeqEnd,
            ],
        );

        let serialized = serde_json::to_string(&box2).unwrap();
        assert_eq!(
            serialized,
            r#"{"min":{"x":10.0,"y":20.0},"max":{"x":30.0,"y":40.0}}"#
        );
        let deserialized: Box2<f32> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(box2, deserialized);

        assert_de_tokens_error::<Box2<f32>>(
            &[Token::Map { len: Some(2) }, Token::I8(123)],
            "invalid type: integer `123`, expected a field name",
        );
        assert_de_tokens_error::<Box2<f32>>(
            &[
                Token::Map { len: Some(2) },
                Token::Str("min"),
                Token::Map { len: Some(2) },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(1.0),
                Token::MapEnd,
                Token::Str("supposed to be max"),
                Token::Map { len: Some(2) },
            ],
            "unknown field `supposed to be max`, expected `min` or `max`",
        );
        assert_de_tokens_error::<Box2<f32>>(
            &[Token::I8(123)],
            "invalid type: integer `123`, expected a mapping with `min` and `max` fields",
        );
    }

    #[test]
    fn box3() {
        let box3 = Box3::<f32>::new(point!(10.0, 20.0, 30.0), point!(40.0, 50.0, 60.0));
        assert_ser_tokens(
            &box3,
            &[
                Token::Struct {
                    name: "Box3",
                    len: 2,
                },
                Token::Str("min"),
                Token::Struct {
                    name: "Point3",
                    len: 3,
                },
                Token::Str("x"),
                Token::F32(10.0),
                Token::Str("y"),
                Token::F32(20.0),
                Token::Str("z"),
                Token::F32(30.0),
                Token::StructEnd,
                Token::Str("max"),
                Token::Struct {
                    name: "Point3",
                    len: 3,
                },
                Token::Str("x"),
                Token::F32(40.0),
                Token::Str("y"),
                Token::F32(50.0),
                Token::Str("z"),
                Token::F32(60.0),
                Token::StructEnd,
                Token::StructEnd,
            ],
        );

        assert_de_tokens(
            &box3,
            &[
                Token::Seq { len: Some(2) },
                Token::Seq { len: Some(3) },
                Token::F32(10.0),
                Token::F32(20.0),
                Token::F32(30.0),
                Token::SeqEnd,
                Token::Seq { len: Some(3) },
                Token::F32(40.0),
                Token::F32(50.0),
                Token::F32(60.0),
                Token::SeqEnd,
                Token::SeqEnd,
            ],
        );

        let serialized = serde_json::to_string(&box3).unwrap();
        assert_eq!(
            serialized,
            r#"{"min":{"x":10.0,"y":20.0,"z":30.0},"max":{"x":40.0,"y":50.0,"z":60.0}}"#
        );
        let deserialized: Box3<f32> = serde_json::from_str(&serialized).unwrap();
        assert_eq!(box3, deserialized);

        assert_de_tokens_error::<Box3<f32>>(
            &[Token::Map { len: Some(2) }, Token::I8(123)],
            "invalid type: integer `123`, expected a field name",
        );
        assert_de_tokens_error::<Box3<f32>>(
            &[
                Token::Map { len: Some(2) },
                Token::Str("min"),
                Token::Map { len: Some(2) },
                Token::Str("x"),
                Token::F32(1.0),
                Token::Str("y"),
                Token::F32(1.0),
                Token::Str("z"),
                Token::F32(1.0),
                Token::MapEnd,
                Token::Str("supposed to be max"),
                Token::Map { len: Some(2) },
            ],
            "unknown field `supposed to be max`, expected `min` or `max`",
        );
        assert_de_tokens_error::<Box3<f32>>(
            &[Token::I8(123)],
            "invalid type: integer `123`, expected a mapping with `min` and `max` fields",
        );
    }
}

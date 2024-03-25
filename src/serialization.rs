use crate::{
    Angle, Box2, Box3, Matrix2, Matrix3, Matrix4, Point2, Point3, Point4, Rect, Size2, Size3,
    Transform2, Transform3, Unit, Vector2, Vector3, Vector4,
};
use serde::ser::SerializeStruct;

macro_rules! impl_vectorlike {
    ($count:literal, $($field:ident),+) => {};
}

impl<T: Unit> serde::Serialize for Vector2<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = match serializer.serialize_struct("Vector2", 2) {
            Ok(map) => map,
            Err(err) => return Err(err),
        };
        if let Err(err) = map.serialize_field("x", &self.x) {
            return Err(err);
        }
        if let Err(err) = map.serialize_field("y", &self.y) {
            return Err(err);
        }
        map.end()
    }
}

enum Field {
    X,
    Y,
    Z,
    W,
    Width,
    Height,
    Depth,
}

impl Field {
    #[inline]
    pub fn as_str(&self) -> &'static str {
        match self {
            Field::X => "x",
            Field::Y => "y",
            Field::Z => "z",
            Field::W => "w",
            Field::Width => "width",
            Field::Height => "height",
            Field::Depth => "depth",
        }
    }
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

        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = Field;

            #[inline]
            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(formatter, "a field name")
            }

            #[inline]
            fn visit_str<E: serde::de::Error>(self, value: &str) -> Result<Self::Value, E> {
                match value {
                    "x" => Ok(Field::X),
                    "y" => Ok(Field::Y),
                    "z" if self.allowed.len >= 3 => Ok(Field::Z),
                    "w" if self.allowed.len >= 4 => Ok(Field::W),
                    "width" if self.allowed.allow_size => Ok(Field::Width),
                    "height" if self.allowed.allow_size => Ok(Field::Height),
                    "depth" if self.allowed.allow_size && self.allowed.len >= 3 => Ok(Field::Depth),
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

impl<'de, T: Unit> serde::Deserialize<'de> for Vector2<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor<T> {
            _marker: core::marker::PhantomData<T>,
        }

        impl<'de, T: Unit> serde::de::Visitor<'de> for Visitor<T> {
            type Value = Vector2<T>;

            fn expecting(&self, formatter: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(formatter, "a mapping or a sequence of length 2")
            }

            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let mut x = None;
                let mut y = None;
                while let Some(key) = map.next_key_seed(AllowedFields {
                    allow_size: false,
                    len: 2,
                })? {
                    match key {
                        Field::X => {
                            if x.is_some() {
                                return Err(serde::de::Error::duplicate_field("x"));
                            }
                            x = Some(map.next_value()?);
                        }
                        Field::Y => {
                            if y.is_some() {
                                return Err(serde::de::Error::duplicate_field("y"));
                            }
                            y = Some(map.next_value()?);
                        }
                        _ => {
                            return Err(serde::de::Error::unknown_field(key.as_str(), &["x", "y"]));
                        }
                    }
                }
                let x = x.ok_or_else(|| serde::de::Error::missing_field("x"))?;
                let y = y.ok_or_else(|| serde::de::Error::missing_field("y"))?;
                Ok(Vector2 { x, y })
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let x = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let y = seq
                    .next_element()?
                    .ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                Ok(Vector2 { x, y })
            }
        }

        deserializer.deserialize_struct(
            "Vector2",
            &ALLOWED_FIELDS_VECTORLIKE[..2],
            Visitor {
                _marker: core::marker::PhantomData,
            },
        )
    }
}

#[cfg(test)]
mod tests {
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
        assert_de_tokens_error::<Vector2<f32>>(
            &[
                Token::Struct {
                    name: "Vector2",
                    len: 2,
                },
                Token::F32(1.0),
                Token::StructEnd,
            ],
            "invalid type: floating point `1`, expected a field name",
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

        assert_eq!(
            serde_json::to_string(&Vector2::<f32> { x: 1.0, y: 2.0 }).unwrap(),
            r#"{"x":1.0,"y":2.0}"#
        );
        assert_eq!(
            serde_json::from_str::<Vector2<f32>>(r#"{"x":1.0,"y":2.0}"#).unwrap(),
            Vector2::<f32> { x: 1.0, y: 2.0 }
        );
        assert_eq!(
            serde_json::from_str::<Vector2<f32>>(r#"[1.0,2.0]"#).unwrap(),
            Vector2::<f32> { x: 1.0, y: 2.0 }
        );
    }
}

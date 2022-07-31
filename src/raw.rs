/// By-value conversion to/from glam types.
///
/// This trait imposes no alignment requirements.
pub trait ToRaw: Sized {
    /// The underlying glam type.
    type Raw;

    /// By-value conversion to `Self::Raw`.
    fn to_raw(self) -> Self::Raw;

    /// By-value conversion from `Self::Raw`.
    fn from_raw(raw: Self::Raw) -> Self;
}

/// Reference conversion to glam types.
///
/// This is only one-way conversion, because the alignment of `Self` could be
/// larger than `Self::Raw` for some types.
pub trait AsRaw: ToRaw {
    /// By-ref conversion to `Self::Raw`.
    fn as_raw(&self) -> &Self::Raw;

    /// By-ref mutable conversion to `Self::Raw`.
    fn as_raw_mut(&mut self) -> &mut Self::Raw;
}

// For types where Self == Self::Raw.
macro_rules! impl_identity {
    ($t:ty) => {
        impl ToRaw for $t {
            type Raw = $t;

            fn to_raw(self) -> Self::Raw {
                self
            }

            fn from_raw(raw: Self::Raw) -> Self {
                raw
            }
        }

        impl AsRaw for $t {
            fn as_raw(&self) -> &Self::Raw {
                self
            }

            fn as_raw_mut(&mut self) -> &mut Self::Raw {
                self
            }
        }
    };
}

impl_identity!(());
impl_identity!(f32);
impl_identity!(f64);
impl_identity!(i32);
impl_identity!(u32);
impl_identity!(usize);
impl_identity!(isize);
impl_identity!(bool);
impl_identity!(glam::BVec2);
impl_identity!(glam::BVec3);
impl_identity!(glam::BVec4);
impl_identity!(glam::Quat);
impl_identity!(glam::DQuat);

impl<'a, T> ToRaw for &'a T
where
    T: AsRaw,
{
    type Raw = &'a T::Raw;

    fn to_raw(self) -> Self::Raw {
        self.as_raw()
    }

    fn from_raw(_: Self::Raw) -> Self {
        panic!("ref from raw conversion unsupported")
    }
}

impl<T, const N: usize> ToRaw for [T; N]
where
    T: ToRaw,
{
    type Raw = [T::Raw; N];

    fn to_raw(self) -> Self::Raw {
        self.map(T::to_raw)
    }

    fn from_raw(raw: Self::Raw) -> Self {
        raw.map(T::from_raw)
    }
}

impl<A, B> ToRaw for (A, B)
where
    A: ToRaw,
    B: ToRaw,
{
    type Raw = (A::Raw, B::Raw);

    fn to_raw(self) -> Self::Raw {
        (self.0.to_raw(), self.1.to_raw())
    }

    fn from_raw((a, b): Self::Raw) -> Self {
        (A::from_raw(a), B::from_raw(b))
    }
}

impl<T> ToRaw for Option<T>
where
    T: ToRaw,
{
    type Raw = Option<T::Raw>;

    fn to_raw(self) -> Self::Raw {
        self.map(T::to_raw)
    }

    fn from_raw(raw: Self::Raw) -> Self {
        raw.map(T::from_raw)
    }
}

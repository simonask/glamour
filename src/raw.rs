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
///
/// See also [`FromRawRef`].
pub trait AsRaw: ToRaw {
    /// By-ref conversion to `Self::Raw`.
    fn as_raw(&self) -> &Self::Raw;

    /// By-ref mutable conversion to `Self::Raw`.
    fn as_raw_mut(&mut self) -> &mut Self::Raw;
}

/// Reference conversion from glam types.
///
/// This is only one-way conversion, because the alignment of `Self::Raw` could
/// be larger than `Self` for some types.
///
/// See also [`AsRaw`].
pub trait FromRawRef: ToRaw {
    /// By-ref conversion from `Self::Raw`.
    fn from_raw_ref(raw: &Self::Raw) -> &Self;

    /// By-ref mutable conversion from `Self::Raw`.
    fn from_raw_mut(raw: &mut Self::Raw) -> &mut Self;
}

macro_rules! impl_primitive {
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

        impl FromRawRef for $t {
            fn from_raw_ref(raw: &Self::Raw) -> &Self {
                raw
            }

            fn from_raw_mut(raw: &mut Self::Raw) -> &mut Self {
                raw
            }
        }
    };
}

impl_primitive!(f32);
impl_primitive!(f64);
impl_primitive!(i32);
impl_primitive!(u32);
impl_primitive!(bool);
impl_primitive!(glam::BVec2);
impl_primitive!(glam::BVec3);
impl_primitive!(glam::BVec4);

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

impl<'a, T> ToRaw for &'a T
where
    T: AsRaw + FromRawRef,
{
    type Raw = &'a T::Raw;

    fn to_raw(self) -> Self::Raw {
        self.as_raw()
    }

    fn from_raw(raw: Self::Raw) -> Self {
        T::from_raw_ref(raw)
    }
}

impl<'a, T> ToRaw for &'a mut T
where
    T: AsRaw + FromRawRef,
{
    type Raw = &'a mut T::Raw;

    fn to_raw(self) -> Self::Raw {
        self.as_raw_mut()
    }

    fn from_raw(raw: Self::Raw) -> Self {
        T::from_raw_mut(raw)
    }
}

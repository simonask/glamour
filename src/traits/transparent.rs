use core::mem::{align_of, size_of, transmute_copy};

use bytemuck::Pod;

/// One-way `TransparentWrapper`.
///
/// # Safety
///
/// This is similar to `bytemuck::TransparentWrapper`, except that it only requires that references can be converted
/// **to** `T`, and not the other way around. In other words, the alignment of `Self` and `T` do not need to be equal,
/// but rather `align_of::<Self>() >= align_of::<T>()`.
pub unsafe trait Transparent: Pod {
    /// The inner type that shares a compatible representation with `Self`.
    type Wrapped: Pod;

    /// Wrap the inner type by copy.
    ///
    /// This is a no-op in most cases, except it may re-align the object if the alignment of `T` is higher.
    fn wrap(x: Self::Wrapped) -> Self {
        const {
            assert!(size_of::<Self::Wrapped>() == size_of::<Self>());
        }

        unsafe { transmute_copy(&x) }
    }

    /// Unwrap the inner type by copy.
    ///
    /// This is a no-op in most cases, except it may re-align the object if the alignment of `T` is higher.
    fn peel(x: Self) -> Self::Wrapped {
        const {
            assert!(size_of::<Self::Wrapped>() == size_of::<Self>());
        }

        unsafe { transmute_copy(&x) }
    }

    /// Convert a reference to the inner type.
    ///
    /// This asserts at compile time that the alignmen of `T` is less than or equal to the alignment of `Self`.
    fn peel_ref(x: &Self) -> &Self::Wrapped {
        const {
            assert!(size_of::<Self::Wrapped>() == size_of::<Self>());
            assert!(align_of::<Self::Wrapped>() <= align_of::<Self>());
        }

        unsafe { &*(x as *const Self).cast() }
    }

    /// Convert a mutable reference to the inner type.
    ///
    /// This asserts at compile time that the alignmen of `T` is less than or equal to the alignment of `Self`.
    fn peel_mut(x: &mut Self) -> &mut Self::Wrapped {
        const {
            assert!(size_of::<Self::Wrapped>() == size_of::<Self>());
            assert!(align_of::<Self::Wrapped>() <= align_of::<Self>());
        }

        unsafe { &mut *(x as *mut Self).cast() }
    }
}

/// Convenience function for calling [`Transparent::wrap()`].
pub fn wrap<T: Transparent>(a: T::Wrapped) -> T {
    Transparent::wrap(a)
}
/// Convenience function for calling [`Transparent::peel()`].
pub fn peel<T: Transparent>(a: T) -> T::Wrapped {
    Transparent::peel(a)
}
/// Convenience function for calling [`Transparent::peel_ref()`].
pub fn peel_ref<T: Transparent>(a: &T) -> &T::Wrapped {
    Transparent::peel_ref(a)
}
/// Convenience function for calling [`Transparent::peel_mut()`].
pub fn peel_mut<T: Transparent>(a: &mut T) -> &mut T::Wrapped {
    Transparent::peel_mut(a)
}
/// Convenience function for calling [`Transparent::peel()`] and [`Transparent::wrap()`].
pub fn rewrap<A: Transparent, B: Transparent<Wrapped = A::Wrapped>>(a: A) -> B {
    wrap(peel(a))
}

#[cfg(test)]
mod tests {
    use super::*;
    use bytemuck::Zeroable;

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct Bigger {
        a: u32,
        b: u32,
        c: u32,
        d: u32,
    }
    unsafe impl Zeroable for Bigger {}
    unsafe impl Pod for Bigger {}
    unsafe impl Transparent for Bigger {
        type Wrapped = Overaligned;
    }

    #[derive(Clone, Copy)]
    #[repr(C, align(16))]
    struct Overaligned {
        a: u32,
        b: u32,
        c: u32,
        d: u32,
    }
    unsafe impl Zeroable for Overaligned {}
    unsafe impl Pod for Overaligned {}
    unsafe impl Transparent for Overaligned {
        type Wrapped = Bigger;
    }

    #[test]
    fn invalid_cast() {
        assert_eq!(align_of::<Bigger>(), 4);
        assert_eq!(align_of::<Overaligned>(), 16);
        let bigger = Bigger {
            a: 1,
            b: 2,
            c: 3,
            d: 4,
        };
        let overaligned: Overaligned = Transparent::peel(bigger);
        assert_eq!(bigger.a, 1);
        assert_eq!(overaligned.a, 1);
        let _bigger: &Bigger = Transparent::peel_ref(&overaligned);
        // This should fail to compile:
        // let overaligned: &Overaligned = Transparent::peel_ref(bigger);
    }
}

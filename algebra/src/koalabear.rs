const KOALA_MODULUS: u32 = 2_130_706_433;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct KoalaBearField(pub u32);

impl KoalaBearField {
    pub fn new(x: u32) -> Self {
        KoalaBearField(x % KOALA_MODULUS)
    }

    pub fn zero() -> Self {
        KoalaBearField(0)
    }

    pub fn one() -> Self {
        KoalaBearField(1)
    }

    pub fn add(self, rhs: Self) -> Self {
        let mut sum = self.0 as u64 + rhs.0 as u64;
        if sum >= KOALA_MODULUS as u64 {
            sum -= KOALA_MODULUS as u64;
        }
        KoalaBearField(sum as u32)
    }

    pub fn sub(self, rhs: Self) -> Self {
        if self.0 >= rhs.0 {
            KoalaBearField(self.0 - rhs.0)
        } else {
            KoalaBearField(KOALA_MODULUS - (rhs.0 - self.0))
        }
    }

    pub fn mul(self, rhs: Self) -> Self {
        let prod = (self.0 as u64 * rhs.0 as u64) % (KOALA_MODULUS as u64);
        KoalaBearField(prod as u32)
    }

    pub fn inv(self) -> Option<Self> {
        let mut t = 0i64;
        let mut new_t = 1i64;
        let mut r = KOALA_MODULUS as i64;
        let mut new_r = self.0 as i64;

        while new_r != 0 {
            let quotient = r / new_r;
            (t, new_t) = (new_t, t - quotient * new_t);
            (r, new_r) = (new_r, r - quotient * new_r);
        }

        if r > 1 {
            return None;
        }

        if t < 0 {
            t += KOALA_MODULUS as i64;
        }

        Some(KoalaBearField(t as u32))
    }

    fn neg(self) -> Self {
        if self.0 == 0 { Self(0) } else { Self(KOALA_MODULUS - self.0) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_sub_mul() {
        let a = KoalaBearField::new(123456);
        let b = KoalaBearField::new(654321);
        let one = KoalaBearField::one();
        let zero = KoalaBearField::zero();

        assert_eq!(a.add(b).sub(b), a);
        assert_eq!(a.mul(one), a);
        assert_eq!(a.sub(a), zero);
    }

    #[test]
    fn test_inverse() {
        let a = KoalaBearField::new(123456);
        let inv = a.inv().expect("should have inverse");
        assert_eq!(a.mul(inv), KoalaBearField::one());
    }

    #[test]
    fn test_inverse_nonexistent() {
        let a = KoalaBearField::zero();
        assert_eq!(a.inv(), None);
    }

    #[test]
    fn test_negation() {
        let a = KoalaBearField::new(5);
        let neg_a = a.neg();
        assert_eq!(a.add(neg_a), KoalaBearField::zero());

        let zero = KoalaBearField::zero();
        assert_eq!(zero.neg(), KoalaBearField::zero());
    }
}

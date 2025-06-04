#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BigUint {
    /// Least significant limb first (little endian)
    pub limbs: Vec<u32>,
}

impl BigUint {
    /// Create a BigUint from a single u32 value
    pub fn from_u32(n: u32) -> Self {
        if n == 0 {
            Self { limbs: vec![] }
        } else {
            Self { limbs: vec![n] }
        }
    }

    /// Add another BigUint to self
    pub fn add_assign(&mut self, other: &BigUint) {
        let mut carry = 0u64;
        let max_len = self.limbs.len().max(other.limbs.len());

        self.limbs.resize(max_len, 0);

        for i in 0..max_len {
            let a = self.limbs[i] as u64;
            let b = if i < other.limbs.len() { other.limbs[i] as u64 } else { 0 };

            let sum = a + b + carry;
            self.limbs[i] = sum as u32;
            carry = sum >> 32;
        }

        if carry != 0 {
            self.limbs.push(carry as u32);
        }
    }

    /// Multiply self by a u32
    pub fn mul_u32(&self, rhs: u32) -> BigUint {
        let mut result = Vec::with_capacity(self.limbs.len() + 1);
        let mut carry = 0u64;

        for &limb in &self.limbs {
            let prod = (limb as u64) * (rhs as u64) + carry;
            result.push(prod as u32);
            carry = prod >> 32;
        }

        if carry != 0 {
            result.push(carry as u32);
        }

        BigUint { limbs: result }
    }

    /// Divide by u32 and return (quotient, remainder)
    pub fn divmod_u32(&self, divisor: u32) -> (BigUint, u32) {
        let mut result = Vec::with_capacity(self.limbs.len());
        let mut rem = 0u64;

        for &limb in self.limbs.iter().rev() {
            let dividend = (rem << 32) | limb as u64;
            result.push((dividend / divisor as u64) as u32);
            rem = dividend % divisor as u64;
        }

        result.reverse();
        while result.last() == Some(&0) {
            result.pop();
        }

        (BigUint { limbs: result }, rem as u32)
    }
}

use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::ops::{AddAssign, Mul};

impl AddAssign<&BigUint> for BigUint {
    fn add_assign(&mut self, rhs: &BigUint) {
        self.add_assign(rhs);
    }
}

impl Mul<u32> for BigUint {
    type Output = BigUint;
    fn mul(self, rhs: u32) -> BigUint {
        self.mul_u32(rhs)
    }
}

impl PartialOrd for BigUint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.limbs.iter().rev().cmp(other.limbs.iter().rev()))
    }
}

impl Display for BigUint {
    /// Convert to decimal string (slow but educational)
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        if self.limbs.is_empty() {
            return write!(f, "0");
        }

        let mut value = self.clone();
        let mut digits = vec![];

        while value > BigUint::from_u32(0) {
            let (q, r) = value.divmod_u32(10);
            digits.push((r as u8 + b'0') as char);
            value = q;
        }
        for ch in digits.iter().rev() {
            write!(f, "{ch}")?;
        }
        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_biguint_add_mul() {
        let mut a = BigUint::from_u32(123456789);
        let b = BigUint::from_u32(987654321);
        a += &b;

        let c = a.clone() * 1000;

        assert_eq!(a.to_string(), "1111111110");
        assert_eq!(c.to_string(), "1111111110000");
    }
}
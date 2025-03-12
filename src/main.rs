// workaround since sqrt() is not const in std
use const_soft_float::soft_f64::SoftF64;

const fn myfunc(q: f64, a: f64, b: f64, c: f64) -> f64 {
    a * q * q + b * q + c * SoftF64(b).sqrt().to_f64()
}

const fn calc_intensities<const N: usize>(q: &[f64; N]) -> [f64; N] {
    // using while since for loops are not const in Rust
    let mut result = [0.0; N];
    let mut index = 0;
    while index < N {
        result[index] = myfunc(q[index], 2.3, 4.2, -0.5);
        index += 1;
    }
    result
}

const fn sum(intensities: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut index = 0;
    while index < intensities.len() {
        sum += intensities[index];
        index += 1;
    }
    sum
}

fn main() {
    const I: &[f64] = calc_intensities(&[10.0, 20.0, 3.0, 4.0, 5.0, -2.0]).as_slice();
    const SUM: f64 = sum(&I);
    println!("{}", SUM);
}

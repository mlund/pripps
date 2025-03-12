// We need this for const sqrt
use const_soft_float::soft_f64::SoftF64;

// Just an arbitrary function that can be evaluated at compile time
const fn myfunc(q: f64, a: f64, b: f64, c: f64) -> f64 {
    a * q * q + b * q + c * SoftF64(b).sqrt().to_f64()
}
// Apply myfunc to q values at compile time
const fn calc_intensities<const N: usize>(q: &[f64; N]) -> [f64; N] {
    let mut result = [0.0; N];
    let a = 2.3;
    let b = 4.2;
    let c = -0.5;
    let mut index = 0;
    while index < N {
        result[index] = myfunc(q[index], a, b, c);
        index += 1;
    }
    result
}

// Sum intensities at compile time
const fn sum_intensities(intensities: &[f64]) -> f64 {
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
    const SUM: f64 = sum_intensities(&I);
    println!("{}", SUM);
}

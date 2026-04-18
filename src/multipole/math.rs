// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Numerical helpers for the Multipole expansion: spherical
//! coordinates, packed lower-triangular storage, normalized associated
//! Legendre functions, spherical Bessel functions, and complex
//! spherical harmonics. All crate-private, all specific to the
//! Multipole calculator's needs.

use crate::Vector3;
use num_complex::Complex;

/// `i^l` for l ≥ 0: cycles through `[1, i, -1, -i]` every four steps.
///
/// Used in the per-atom accumulation loop to multiply by the `i^l`
/// factor without a per-l trig call. PepsiSAXS inlines this as four
/// separate `l += 4` loops with manual sign/swap tricks
/// (`cSAS.cpp:2274-2322`) — a C-era manual vectorization. We trust
/// LLVM to handle a straight loop with a `const` lookup; revisit only
/// if profiling shows this branch is hot.
pub(crate) const I_POW: [Complex<f64>; 4] = [
    Complex::new(1.0, 0.0),
    Complex::I,
    Complex::new(-1.0, 0.0),
    Complex::new(0.0, -1.0),
];

/// Spherical coordinates in Å / rad / rad. Named-field struct so call
/// sites read `s.r`, `s.theta`, `s.phi` rather than destructuring a
/// tuple. Crate-local so the `From<Vector3>` impl satisfies the orphan
/// rule (neither `Vector3` nor stdlib tuples would work as the target).
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Spherical {
    pub r: f64,
    pub theta: f64,
    pub phi: f64,
}

impl From<Vector3> for Spherical {
    fn from(pos: Vector3) -> Self {
        let r = pos.norm();
        Self {
            r,
            // At the origin (r = 0) any θ is equivalent — the atom has
            // no direction — so pick 0 to avoid NaN from (z/0).acos().
            theta: if r > 0.0 { (pos.z / r).acos() } else { 0.0 },
            phi: pos.y.atan2(pos.x),
        }
    }
}

/// Packed lower-triangular storage for values indexed by `(l, m)`
/// with `m ≤ l ≤ l_max`. Length is `(l_max + 1) · (l_max + 2) / 2`.
///
/// The natural storage for spherical-harmonic coefficients: the upper
/// triangle is structurally empty (`m > l` is undefined) and we exploit
/// `Y_l^{-m} = (-1)^m · conj(Y_l^m)` elsewhere so negative m is never
/// materialized. Hides the `l·(l+1)/2 + m` index math behind
/// [`std::ops::Index<(usize, usize)>`] so callers write `tri[(l, m)]`.
#[derive(Debug, Clone)]
pub(crate) struct LowerTriangle<T> {
    data: Vec<T>,
}

impl<T: Clone + Default> LowerTriangle<T> {
    pub fn zeros(l_max: usize) -> Self {
        let len = (l_max + 1) * (l_max + 2) / 2;
        Self {
            data: vec![T::default(); len],
        }
    }
}

impl<T> LowerTriangle<T> {
    /// Upper-triangle access (`m > l`) lands on a valid but wrong
    /// cell because `l*(l+1)/2 + m` can still be in `data.len()`;
    /// the debug-assert catches the one case Rust's built-in
    /// vec-bounds-check can't. Out-of-range `l` falls through to
    /// the standard bounds-check panic on the subsequent access.
    #[inline]
    fn offset(&self, l: usize, m: usize) -> usize {
        debug_assert!(m <= l, "LowerTriangle: m ({m}) must not exceed l ({l})");
        l * (l + 1) / 2 + m
    }

    /// Mutable slice view of the packed storage. Used by
    /// `SphericalBasis::build` to pre-conjugate every entry in a
    /// single linear sweep without walking the `(l, m)` grid.
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> std::ops::Index<(usize, usize)> for LowerTriangle<T> {
    type Output = T;
    fn index(&self, (l, m): (usize, usize)) -> &T {
        &self.data[self.offset(l, m)]
    }
}

impl<T> std::ops::IndexMut<(usize, usize)> for LowerTriangle<T> {
    fn index_mut(&mut self, (l, m): (usize, usize)) -> &mut T {
        let i = self.offset(l, m);
        &mut self.data[i]
    }
}

/// Normalized associated Legendre functions `P̄_l^m(z)` for all
/// `(l, m)` with `m ≤ l ≤ l_max`, where `z = cos θ`. The normalization
/// factor `sqrt((2l+1)/(4π) · (l-m)!/(l+m)!)` is baked in — we never
/// compute factorials directly.
///
/// Uses the Holmes-Featherstone 2002 scaled recurrence
/// (*J. Geodesy* **76**, pp. 279-299,
/// <https://doi.org/10.1007/s00190-002-0216-2>), which handles `l_max`
/// up to ~2800 stably by rescaling `P̄_m^m` by `1e-280 · sin^m θ` to
/// prevent under/overflow near the poles at high m. The scale factor
/// is gradually consumed by successive `sin θ` multiplications; once
/// it drops below `1e-280` the rescaling is absorbed directly into the
/// stored values.
///
/// Ported from PepsiSAXS's `computeLegendreScaled` at
/// `cSAS.cpp:3001`, adapted to the [`LowerTriangle`] wrapper.
pub(crate) fn legendre_normalized(cos_theta: f64, l_max: usize) -> LowerTriangle<f64> {
    use std::f64::consts::PI;
    let four_pi = 4.0 * PI;
    let mut p = LowerTriangle::<f64>::zeros(l_max);

    // u = sin θ, computed stably from cos²θ for z close to ±1
    let z = cos_theta;
    let u = ((1.0 - z) * (1.0 + z)).sqrt();

    // m = 0 column: standard non-scaled recurrence.
    // P̄_0^0 = 1/√(4π), P̄_1^0 = √3 · z / √(4π),
    // P̄_l^0 = [(2l-1) z P̄_{l-1}^0 - (l-1) P̄_{l-2}^0] · √((2l+1)/(2l-1)) / l
    let mut pm2 = 1.0 / four_pi.sqrt();
    p[(0, 0)] = pm2;
    if l_max == 0 {
        return p;
    }
    let mut pm1 = 3f64.sqrt() * z / four_pi.sqrt();
    p[(1, 0)] = pm1;
    for l in 2..=l_max {
        let lf = l as f64;
        // Coefficients for the three-term recurrence on normalized P:
        //   P̄_l^0 = a · z · P̄_{l-1}^0 - b · P̄_{l-2}^0
        let a = (2.0f64.mul_add(lf, -1.0) * 2.0f64.mul_add(lf, 1.0) / (lf * lf)).sqrt();
        let b = (2.0f64.mul_add(lf, 1.0) * (lf - 1.0) * (lf - 1.0)
            / (2.0f64.mul_add(lf, -3.0) * lf * lf))
            .sqrt();
        let plm = (a * z).mul_add(pm1, -(b * pm2));
        p[(l, 0)] = plm;
        pm2 = pm1;
        pm1 = plm;
    }

    // m ≥ 1: Holmes-Featherstone scaled recurrence.
    // Start with a heavily scaled P̄_m^m (= scalef / √(4π)) and let
    // successive sin θ multiplications consume the scale factor.
    let scalef = 1e-280_f64;
    let mut pmm = scalef / four_pi.sqrt();
    let mut rescale = 1.0 / scalef;

    for m in 1..l_max {
        rescale *= u;
        let mf = m as f64;

        // P̄_m^m = -√((2m+1)/(2m)) · sin θ · P̄_{m-1}^{m-1}
        // (sin θ is absorbed in `rescale`; the negative sign is the
        // (-1)^m phase, applied once per m).
        pmm = -pmm * (2.0f64.mul_add(mf, 1.0).sqrt() / (2.0 * mf).sqrt());
        p[(m, m)] = pmm * rescale;

        // P̄_{m+1}^m = √(2m+3) · z · P̄_m^m
        let mut pm2m = pmm;
        let mut pm1m = z * 2.0f64.mul_add(mf, 3.0).sqrt() * pmm;
        if m < l_max {
            p[(m + 1, m)] = pm1m * rescale;
        }

        // P̄_l^m for l > m+1 via three-term recurrence.
        for l in (m + 2)..=l_max {
            let lf = l as f64;
            #[allow(clippy::suspicious_operation_groupings)]
            let a = (2.0f64.mul_add(lf, -1.0) * 2.0f64.mul_add(lf, 1.0) / ((lf - mf) * (lf + mf)))
                .sqrt();
            let b = (2.0f64.mul_add(lf, 1.0) * (lf - mf - 1.0) * (lf + mf - 1.0)
                / (2.0f64.mul_add(lf, -3.0) * (lf - mf) * (lf + mf)))
                .sqrt();
            let plm = (a * z).mul_add(pm1m, -(b * pm2m));
            p[(l, m)] = plm * rescale;
            pm2m = pm1m;
            pm1m = plm;
        }
    }

    // P̄_{l_max}^{l_max}: last diagonal entry.
    if l_max >= 1 {
        rescale *= u;
        let mf = l_max as f64;
        pmm = -pmm * (2.0f64.mul_add(mf, 1.0).sqrt() / (2.0 * mf).sqrt());
        p[(l_max, l_max)] = pmm * rescale;
    }

    p
}

/// Spherical Bessel functions `j_l(x)` for `l ∈ [0, l_max]`, written
/// into `out[0..=l_max]`. The caller provides the output slice, whose
/// `len()` determines `l_max + 1`. This lets the per-q-per-atom inner
/// loop reuse a single scratch buffer instead of allocating a fresh
/// `Vec<f64>` per call — the alternative (return `Vec<f64>`) was the
/// only regression flagged in the initial review.
///
/// Uses Miller's backward recurrence for numerical stability: forward
/// recurrence (`j_{l+1}(x) = (2l+1)/x · j_l - j_{l-1}`) is
/// catastrophically unstable for `l > x`, the regime pripps hits as
/// `q·r` stays moderate while `l_max` can be ~30-50. The backward
/// direction is well-conditioned: start far above `l_max` with
/// `(j_{L+1}, j_L) = (0, 1)`, iterate down to l=0, then renormalize
/// using the exact closed form `j_0(x) = sin(x)/x`.
///
/// Reference: *Numerical Recipes* §6.7 "Spherical Bessel Functions".
///
/// The `x < 1e-8` branch falls back to the small-argument power series
/// `j_l(x) ≈ x^l / (2l+1)!!` because `sin(x)/x` loses all precision as
/// `x → 0` in f64 (catastrophic cancellation in the subtraction
/// `sin(x) ≈ x`).
pub(crate) fn spherical_bessel(x: f64, out: &mut [f64]) {
    let l_max = out.len() - 1;

    if x.abs() < 1e-8 {
        // Small-argument series: j_l(x) = x^l / (2l+1)!! · (1 + O(x²))
        // (2l+1)!! = 1 · 3 · 5 · … · (2l+1), built incrementally.
        out[0] = 1.0;
        let mut xl = 1.0; // x^l accumulator
        let mut dfact = 1.0; // (2l+1)!! accumulator (= 1 for l=0)
        for (l, slot) in out.iter_mut().enumerate().skip(1) {
            xl *= x;
            dfact *= (2 * l + 1) as f64;
            *slot = xl / dfact;
        }
        return;
    }

    // Miller's backward recurrence. Starting l should be safely above
    // l_max; the exact choice is a trade-off between stability and
    // work. The NR recipe uses l_start ≈ l_max + ceil(sqrt(40 l_max));
    // we adopt the same with a floor of 15.
    let extra = (40.0 * l_max as f64).sqrt().ceil() as usize;
    let l_start = l_max + extra.max(15);

    let mut j_next = 0.0; // j_{l_start+1}
    let mut j_curr = 1.0; // j_{l_start}  (arbitrary seed; renormalized below)
    out.fill(0.0);

    for l in (0..=l_start).rev() {
        let lf = l as f64;
        let j_prev = (2.0f64.mul_add(lf, 1.0) / x).mul_add(j_curr, -j_next);
        if l <= l_max {
            out[l] = j_curr;
        }
        j_next = j_curr;
        j_curr = j_prev;
        // Guard against runaway growth in the unnormalized recurrence:
        // if |j_curr| gets huge, rescale everything to keep it in f64
        // range. Needed for l_max > ~60 or very small x.
        if j_curr.abs() > 1e200 {
            let scale = 1e-200;
            j_curr *= scale;
            j_next *= scale;
            for v in out.iter_mut() {
                *v *= scale;
            }
        }
    }

    // Renormalize: after the loop, j_next holds our unnormalized j_0(x).
    // True j_0(x) = sin(x) / x, so scale everything by the ratio.
    let true_j0 = x.sin() / x;
    let scale = true_j0 / j_next;
    for v in out.iter_mut() {
        *v *= scale;
    }
}

/// Fully-normalized complex spherical harmonics
/// `Y_l^m(θ, φ) = P̄_l^m(cos θ) · exp(imφ)` for all `(l, m)` with
/// `m ≥ 0`.
///
/// The `m < 0` half is recovered via
/// `Y_l^{-m} = (-1)^m · conj(Y_l^m)` and never materialized — a 2×
/// storage save matching PepsiSAXS's `Y_C[l][m]` layout at
/// `cSAS.cpp:2873`. The `exp(imφ)` factor is built incrementally
/// (`e^{i(m+1)φ} = e^{imφ} · e^{iφ}`) so we do one pair of trig calls
/// up front instead of `l_max` of them inside the loop.
pub(crate) fn spherical_harmonics(
    theta: f64,
    phi: f64,
    l_max: usize,
) -> LowerTriangle<Complex<f64>> {
    let legendre = legendre_normalized(theta.cos(), l_max);
    let mut out = LowerTriangle::<Complex<f64>>::zeros(l_max);

    let step = Complex::new(phi.cos(), phi.sin());
    let mut phase = Complex::new(1.0, 0.0); // exp(i·0·φ) = 1

    for m in 0..=l_max {
        for l in m..=l_max {
            out[(l, m)] = legendre[(l, m)] * phase;
        }
        phase *= step;
    }

    out
}

#[cfg(test)]
fn spherical_bessel_vec(x: f64, l_max: usize) -> Vec<f64> {
    let mut out = vec![0.0; l_max + 1];
    spherical_bessel(x, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f64::consts::PI;

    // ------- LowerTriangle ---------------------------------------

    #[test]
    fn lower_triangle_zeros_has_right_length() {
        let t = LowerTriangle::<f64>::zeros(5);
        assert_eq!(t.data.len(), 21); // 6 · 7 / 2
    }

    #[test]
    fn lower_triangle_index_roundtrip() {
        let mut t = LowerTriangle::<f64>::zeros(10);
        for l in 0..=10 {
            for m in 0..=l {
                t[(l, m)] = (l * 100 + m) as f64;
            }
        }
        for l in 0..=10 {
            for m in 0..=l {
                assert_eq!(t[(l, m)], (l * 100 + m) as f64);
            }
        }
    }

    #[test]
    fn lower_triangle_offset_is_packed() {
        let t = LowerTriangle::<f64>::zeros(5);
        let mut seen = Vec::new();
        for l in 0..=5 {
            for m in 0..=l {
                seen.push(t.offset(l, m));
            }
        }
        // Packed: 0, 1, 2, … with no gaps.
        for (i, &o) in seen.iter().enumerate() {
            assert_eq!(o, i);
        }
    }

    #[test]
    #[should_panic]
    #[cfg(debug_assertions)]
    fn lower_triangle_out_of_bounds_m_gt_l_panics() {
        let t = LowerTriangle::<f64>::zeros(5);
        let _ = t[(0, 1)]; // m > l — caught by debug_assert in offset()
    }

    #[test]
    #[should_panic]
    fn lower_triangle_out_of_bounds_l_gt_lmax_panics() {
        let t = LowerTriangle::<f64>::zeros(5);
        let _ = t[(6, 0)]; // l > l_max
    }

    #[test]
    fn lower_triangle_generic_over_complex() {
        let t = LowerTriangle::<Complex<f64>>::zeros(3);
        for l in 0..=3 {
            for m in 0..=l {
                assert_eq!(t[(l, m)], Complex::<f64>::default());
            }
        }
    }

    // ------- Spherical -------------------------------------------

    #[test]
    fn spherical_from_origin_is_zero() {
        let s = Spherical::from(Vector3::new(0.0, 0.0, 0.0));
        assert_eq!(s.r, 0.0);
        assert_eq!(s.theta, 0.0);
        assert_eq!(s.phi, 0.0);
    }

    #[test]
    fn spherical_from_z_axis() {
        let s = Spherical::from(Vector3::new(0.0, 0.0, 2.0));
        assert_relative_eq!(s.r, 2.0);
        assert_relative_eq!(s.theta, 0.0);
        // phi at z axis is degenerate (atan2(0,0) = 0); we accept 0.
        assert_eq!(s.phi, 0.0);
    }

    #[test]
    fn spherical_from_minus_z_axis() {
        let s = Spherical::from(Vector3::new(0.0, 0.0, -3.0));
        assert_relative_eq!(s.r, 3.0);
        assert_relative_eq!(s.theta, PI);
    }

    #[test]
    fn spherical_from_x_axis() {
        let s = Spherical::from(Vector3::new(1.5, 0.0, 0.0));
        assert_relative_eq!(s.r, 1.5);
        assert_relative_eq!(s.theta, PI / 2.0);
        assert_relative_eq!(s.phi, 0.0);
    }

    // ------- legendre_normalized ---------------------------------

    #[test]
    fn legendre_y00_value() {
        // P̄_0^0 = 1/√(4π) — identity independent of θ.
        let p = legendre_normalized(0.3, 5);
        assert_relative_eq!(p[(0, 0)], 1.0 / (4.0 * PI).sqrt(), epsilon = 1e-14);
    }

    #[test]
    fn legendre_y10_at_zenith() {
        // P̄_1^0(z) = √(3/(4π)) · z; at z = cos(0) = 1 → √(3/(4π))
        let p = legendre_normalized(1.0, 3);
        assert_relative_eq!(p[(1, 0)], (3.0 / (4.0 * PI)).sqrt(), epsilon = 1e-14);
    }

    #[test]
    fn legendre_normalized_orthogonality_m0() {
        // Normalized P̄_l^0 satisfies
        //   ∫_{-1}^{1} P̄_l^0 P̄_{l'}^0 dz = δ_{ll'} / (2π)
        // (half the unit-sphere orthogonality because we fold in the
        // φ integral as 2π later).
        //
        // Sanity-check with Simpson's rule on 1000 points for
        // l, l' ≤ 4.
        let n = 1000;
        let mut z = vec![0.0; n + 1];
        for (i, zi) in z.iter_mut().enumerate() {
            *zi = -1.0 + 2.0 * i as f64 / n as f64;
        }

        let mut pp = vec![Vec::new(); 5];
        for zi in &z {
            let tri = legendre_normalized(*zi, 4);
            for l in 0..=4 {
                pp[l].push(tri[(l, 0)]);
            }
        }

        // Simpson's 1/3 rule integral.
        let simpson = |values: &[f64]| -> f64 {
            let h = 2.0 / n as f64;
            let mut s = values[0] + values[n];
            for (i, v) in values.iter().enumerate().take(n).skip(1) {
                s += if i % 2 == 0 { 2.0 } else { 4.0 } * v;
            }
            s * h / 3.0
        };

        for l in 0..=4 {
            for lp in 0..=4 {
                let prod: Vec<f64> = pp[l].iter().zip(&pp[lp]).map(|(a, b)| a * b).collect();
                let integral = simpson(&prod);
                let expected = if l == lp { 1.0 / (2.0 * PI) } else { 0.0 };
                assert!(
                    (integral - expected).abs() < 1e-6,
                    "∫ P̄_{}^0 P̄_{}^0 dz = {}, expected {}",
                    l,
                    lp,
                    integral,
                    expected
                );
            }
        }
    }

    // ------- spherical_bessel ------------------------------------

    #[test]
    fn spherical_bessel_j0_small_x() {
        // j_0(x) = sin(x)/x; near x = 0 the series gives j_0(0) = 1.
        let j = spherical_bessel_vec(1e-10, 3);
        assert_relative_eq!(j[0], 1.0, epsilon = 1e-12);
    }

    #[test]
    fn spherical_bessel_small_x_series_matches() {
        // For x < 1e-8 the small-x branch returns j_l(x) ≈ x^l / (2l+1)!!
        let x = 1e-9;
        let j = spherical_bessel_vec(x, 4);
        assert_relative_eq!(j[0], 1.0, epsilon = 1e-12);
        assert_relative_eq!(j[1], x / 3.0, epsilon = 1e-12);
        assert_relative_eq!(j[2], x * x / 15.0, epsilon = 1e-12);
        assert_relative_eq!(j[3], x.powi(3) / 105.0, epsilon = 1e-12);
    }

    #[test]
    fn spherical_bessel_j0_at_pi_is_zero() {
        // j_0(π) = sin(π) / π = 0 (to rounding)
        let j = spherical_bessel_vec(PI, 5);
        assert!(j[0].abs() < 1e-14);
    }

    #[test]
    fn spherical_bessel_j1_at_half_pi() {
        // j_1(x) = sin(x)/x² - cos(x)/x
        // At x = π/2: j_1(π/2) = 1/(π/2)² - 0 = 4/π²
        let j = spherical_bessel_vec(PI / 2.0, 3);
        assert_relative_eq!(j[1], 4.0 / (PI * PI), epsilon = 1e-12);
    }

    #[test]
    fn spherical_bessel_high_l_finite() {
        // j_50(1.0) is extremely small (~1e-80) but must be finite.
        // Miller's recurrence stays stable; forward would have blown up.
        let j = spherical_bessel_vec(1.0, 60);
        assert!(j[50].is_finite());
        assert!(j[50].abs() < 1e-60); // tail is tiny but nonzero
    }

    // ------- spherical_harmonics ---------------------------------

    #[test]
    fn spherical_harmonics_y00_is_constant() {
        // Y_0^0 = 1/√(4π) independent of θ, φ
        let expected = 1.0 / (4.0 * PI).sqrt();
        for (theta, phi) in [
            (0.0, 0.0),
            (PI / 3.0, 0.7),
            (PI / 2.0, PI),
            (PI, 2.0f64.mul_add(PI, -0.1)),
        ] {
            let y = spherical_harmonics(theta, phi, 3);
            assert_relative_eq!(y[(0, 0)].re, expected, epsilon = 1e-14);
            assert_relative_eq!(y[(0, 0)].im, 0.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn spherical_harmonics_y10_at_zenith() {
        // Y_1^0(0, φ) = √(3/(4π)) · cos(0) = √(3/(4π))
        let y = spherical_harmonics(0.0, 0.0, 2);
        assert_relative_eq!(y[(1, 0)].re, (3.0 / (4.0 * PI)).sqrt(), epsilon = 1e-14);
    }

    #[test]
    fn spherical_harmonics_y11_vanishes_at_zenith() {
        // Y_1^1(0, φ) ∝ sin(0) = 0
        let y = spherical_harmonics(0.0, 0.7, 2);
        assert!(y[(1, 1)].norm() < 1e-14);
    }

    #[test]
    fn i_pow_cycles() {
        // Just confirm the const is right: 1, i, -1, -i
        assert_eq!(I_POW[0], Complex::new(1.0, 0.0));
        assert_eq!(I_POW[1], Complex::new(0.0, 1.0));
        assert_eq!(I_POW[2], Complex::new(-1.0, 0.0));
        assert_eq!(I_POW[3], Complex::new(0.0, -1.0));
    }
}

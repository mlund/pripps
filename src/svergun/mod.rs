// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Scattering intensity via Svergun's spherical-harmonic multipole
//! expansion.
//!
//! Reference: Svergun, Barberato, Koch — CRYSOL, *J. Appl. Cryst.*
//! **28**, 768 (1995). <https://doi.org/10.1107/S0021889895007047>
//!
//! Expands the plane wave `exp(i q·r)` in spherical harmonics so the
//! orientational average of `|A(q)|²` collapses onto a single sum
//! over multipole coefficients:
//!
//! ```text
//! A_l^m(q) = Σᵢ fᵢ(q) · i^l · Y_l^m*(θᵢ, φᵢ) · j_l(q·rᵢ)
//! I(q)     = 4π · Σ_lm |A_l^m(q)|²
//!          = 4π · Σ_l [ |A_l^0|² + 2 Σ_{m>0} |A_l^m|² ]
//! ```
//!
//! Complexity is `O(N · Q · L²)` where `N = atoms`, `Q = q-points`,
//! `L = l_max`. Linear in atoms, so this wins over the `O(N² · Q)`
//! [`crate::debye::DebyeIntensity`] once the cluster gets big enough
//! (break-even around `N ≈ 500` for typical globular proteins with
//! `l_max ≈ 20-30`).
//!
//! The numerical primitives (`Spherical`, `LowerTriangle`,
//! `legendre_normalized`, `spherical_bessel`, `spherical_harmonics`,
//! `I_POW`) live in the [`math`] submodule to keep this file focused
//! on the calculator pipeline. Solvent corrections (`Clm` excluded
//! volume, `Blm` hydration shell) that PepsiSAXS combines into
//! `|A - C + dp·B|²` are deliberately out of scope and belong to a
//! separate `SolventConfig` follow-up.

mod math;

use crate::q_sampler::QSampler;
use crate::{
    Intensity, IntensityCalculator, Result, Structure,
    formfactor::{self, FormFactorMap},
};
use math::{I_POW, LowerTriangle, Spherical, spherical_bessel, spherical_harmonics};
use num_complex::Complex;

/// Per-atom spherical-harmonic basis values: the spherical coordinates
/// of every atom and the conjugated `Y_l^m*` tables Svergun's
/// expansion evaluates at those coordinates.
///
/// Built once per structure by [`SphericalBasis::build`] and reused
/// across the q loop — the radial part `j_l(q·r)` is the only part of
/// the multipole kernel that depends on q, so it stays in the per-q
/// closure while everything angular lives here.
struct SphericalBasis {
    spherical: Vec<Spherical>,
    /// Per-atom, already-conjugated angular factors `Y_l^m*`.
    angular: Vec<LowerTriangle<Complex<f64>>>,
    l_max: usize,
}

impl SphericalBasis {
    fn build(structure: &Structure, l_max: usize) -> Self {
        let spherical: Vec<Spherical> =
            structure.pos.iter().copied().map(Spherical::from).collect();

        let angular: Vec<LowerTriangle<Complex<f64>>> = spherical
            .iter()
            .map(|s| {
                let mut y = spherical_harmonics(s.theta, s.phi, l_max);
                // Pre-conjugate once so the per-q inner loop doesn't
                // need `.conj()` per (atom, l, m). Saves one complex
                // negation per entry in the hot accumulation path.
                for c in y.data_mut() {
                    *c = c.conj();
                }
                y
            })
            .collect();

        Self {
            spherical,
            angular,
            l_max,
        }
    }
}

pub(crate) struct SvergunIntensity {
    sampler: QSampler,
    l_max_override: Option<usize>,
}

impl SvergunIntensity {
    pub(crate) fn new(
        formfactors: FormFactorMap,
        q_minmax: (f64, f64),
        num_q: usize,
        l_max: Option<usize>,
    ) -> Result<Self> {
        Ok(Self {
            sampler: QSampler::linear(formfactors, q_minmax.0, q_minmax.1, num_q),
            l_max_override: l_max,
        })
    }

    /// Resolve the effective expansion order: user override if given,
    /// else the Svergun criterion `⌈q_max · r_max⌉ + 2`.
    ///
    /// Uses `r_max` (farthest atom from origin) rather than `R_g`
    /// (radius of gyration) as a safe upper bound: `R_g` can
    /// under-estimate the criterion for asymmetric structures, and
    /// asking for too few multipoles silently gives a wrong answer.
    /// The `+2` absorbs the usual "rule of thumb" margin — tests
    /// confirm it's enough for agreement with Debye to 1% on small
    /// clusters.
    fn effective_l_max(&self, structure: &Structure) -> usize {
        if let Some(l) = self.l_max_override {
            return l;
        }
        let r_max: f64 = structure
            .pos
            .iter()
            .map(|p| p.norm())
            .fold(0.0_f64, f64::max);
        let q_max = self
            .sampler
            .q_values
            .iter()
            .copied()
            .fold(0.0_f64, f64::max);
        ((q_max * r_max).ceil() as usize) + 2
    }
}

impl IntensityCalculator for SvergunIntensity {
    fn calc_intensities(&self, structure: &Structure) -> Result<Intensity> {
        let l_max = self.effective_l_max(structure);
        log::info!(
            "Svergun expansion: l_max = {} over {} atoms, {} q-points",
            l_max,
            structure.pos.len(),
            self.sampler.q_values.len(),
        );
        let basis = SphericalBasis::build(structure, l_max);
        self.sampler
            .run_parallel(|q| svergun_at_q(&basis, &self.sampler.formfactors, &structure.ids, q))
    }
}

/// Compute `I(q)` at a single q from the precomputed angular basis.
fn svergun_at_q(
    basis: &SphericalBasis,
    formfactors: &FormFactorMap,
    atom_ids: &[usize],
    q: f64,
) -> Result<f64> {
    let l_max = basis.l_max;
    let ff = formfactor::formfactor_by_index(formfactors, q, atom_ids)?;
    let mut alm = LowerTriangle::<Complex<f64>>::zeros(l_max);

    // Per-q scratch for the radial Bessel factors, reused across all
    // atoms so the hot loop doesn't allocate per atom.
    let mut j_l = vec![0.0_f64; l_max + 1];

    // Accumulate A_l^m += f · i^l · Y*_l^m · j_l(q·r) over atoms.
    for ((s, y_conj), f) in basis
        .spherical
        .iter()
        .zip(&basis.angular)
        .zip(ff.iter().copied())
    {
        spherical_bessel(q * s.r, &mut j_l);
        for l in 0..=l_max {
            let il_jl = I_POW[l % 4] * j_l[l];
            for m in 0..=l {
                alm[(l, m)] += il_jl * y_conj[(l, m)] * f;
            }
        }
    }

    // I(q) = 4π · Σ_l [ |A_l^0|² + 2 · Σ_{m>0} |A_l^m|² ]
    //
    // The factor of 2 on m > 0 accounts for the negative-m half of
    // the sum: storing only m ≥ 0 and using the relation
    // Y_l^{-m} = (-1)^m · conj(Y_l^m) doubles the contribution of
    // each m > 0 entry to |A_l^m|² + |A_l^{-m}|² = 2 |A_l^m|².
    let sum: f64 = (0..=l_max)
        .map(|l| {
            let m0 = alm[(l, 0)].norm_sqr();
            let mpos: f64 = (1..=l).map(|m| 2.0 * alm[(l, m)].norm_sqr()).sum();
            m0 + mpos
        })
        .sum();

    Ok(4.0 * std::f64::consts::PI * sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debye::DebyeIntensity;
    use crate::formfactor::FormFactor;
    use crate::{AtomKind, Vector3};
    use std::collections::BTreeMap;

    /// Build a single-kind form-factor map: every atom uses
    /// `FormFactor::Constant(value)`, so `f(q) = value` for all q.
    fn constant_ff(value: f64) -> FormFactorMap {
        let mut m = BTreeMap::new();
        m.insert(AtomKind::new("X", 0), FormFactor::Constant(value));
        m
    }

    /// Build a tiny Structure from raw positions, all assigned to
    /// atom kind 0 (to match the single-kind `constant_ff` helper).
    fn structure_with_ids(positions: Vec<(f64, f64, f64)>) -> Structure {
        let n = positions.len();
        Structure {
            pos: positions
                .into_iter()
                .map(|(x, y, z)| Vector3::new(x, y, z))
                .collect(),
            ids: vec![0; n],
            box_sides: None,
        }
    }

    #[test]
    fn single_atom_at_origin_constant_ff() {
        // One atom at origin, f(q) = c. Only Y_0^0 contributes;
        // I(q) = 4π · |A_00|² · c² = c² (after normalization).
        let c = 3.0;
        let calc = SvergunIntensity::new(constant_ff(c), (0.1, 0.5), 10, None).unwrap();
        let structure = structure_with_ids(vec![(0.0, 0.0, 0.0)]);
        let out = calc.calc_intensities(&structure).unwrap();
        for i in &out.i {
            assert!((i - c * c).abs() < 1e-10, "expected {}, got {i}", c * c);
        }
    }

    #[test]
    fn agrees_with_debye_at_q_zero() {
        // At q=0 both reduce to (Σᵢ fᵢ)². Exact regardless of l_max.
        let c = 1.0;
        let structure = structure_with_ids(vec![
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 1.5),
        ]);
        let n = structure.pos.len() as f64;

        let svergun = SvergunIntensity::new(constant_ff(c), (0.0, 0.1), 5, None).unwrap();
        let s_out = svergun.calc_intensities(&structure).unwrap();
        let s0 = s_out.i[0];

        let debye = DebyeIntensity::new(constant_ff(c), (0.0, 0.1), 5).unwrap();
        let d_out = debye.calc_intensities(&structure).unwrap();
        let d0 = d_out.i[0];

        let expected = (n * c) * (n * c);
        assert!(
            (s0 - expected).abs() < 1e-10,
            "svergun q=0: {s0} ≠ {expected}"
        );
        assert!(
            (d0 - expected).abs() < 1e-10,
            "debye q=0: {d0} ≠ {expected}"
        );
        assert!(
            (s0 - d0).abs() < 1e-10,
            "svergun ≠ debye at q=0: {s0} vs {d0}"
        );
    }

    #[test]
    fn agrees_with_debye_on_two_atoms() {
        // Closed-form: I_Debye(q) = 2 + 2 · sinc(q · r_12). Svergun
        // truncates the expansion at finite l_max so disagreement is
        // truncation-order (a few × 1e-6 here); not a correctness bug.
        let structure = structure_with_ids(vec![(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)]);
        let qmin = 0.01;
        let qmax = 0.5;
        let nq = 20;

        let svergun = SvergunIntensity::new(constant_ff(1.0), (qmin, qmax), nq, None).unwrap();
        let debye = DebyeIntensity::new(constant_ff(1.0), (qmin, qmax), nq).unwrap();

        let s_out = svergun.calc_intensities(&structure).unwrap();
        let d_out = debye.calc_intensities(&structure).unwrap();

        assert_eq!(s_out.q, d_out.q);
        for ((q, &is), &id) in s_out.q.iter().zip(&s_out.i).zip(&d_out.i) {
            let rel = (is - id).abs() / id.max(1e-10);
            assert!(
                rel < 1e-4,
                "q = {q}: svergun = {is}, debye = {id}, rel err = {rel}"
            );
        }
    }

    #[test]
    fn agrees_with_debye_on_small_cluster() {
        // 10 deterministic positions inside a 15-Å box; Svergun vs
        // Debye at 20 q values should agree to < 1% rel err.
        // Positions are a simple lattice shifted off the origin so
        // no atom sits at r = 0.
        let positions: Vec<(f64, f64, f64)> = (0..10)
            .map(|i| {
                let f = i as f64;
                (1.5 + f * 1.1, 2.0 + (f * 0.7) % 5.0, 0.5 + (f * 1.3) % 6.0)
            })
            .collect();
        let structure = structure_with_ids(positions);

        let qmin = 0.01;
        let qmax = 0.5;
        let nq = 20;

        let svergun = SvergunIntensity::new(constant_ff(1.0), (qmin, qmax), nq, None).unwrap();
        let debye = DebyeIntensity::new(constant_ff(1.0), (qmin, qmax), nq).unwrap();

        let s_out = svergun.calc_intensities(&structure).unwrap();
        let d_out = debye.calc_intensities(&structure).unwrap();

        for ((q, &is), &id) in s_out.q.iter().zip(&s_out.i).zip(&d_out.i) {
            let rel = (is - id).abs() / id.max(1e-10);
            assert!(
                rel < 0.01,
                "q = {q}: svergun = {is}, debye = {id}, rel err = {rel}"
            );
        }
    }

    #[test]
    fn auto_l_max_respects_structure_extent() {
        // q_max · r_max = 1.0 · 50.0 = 50, so auto l_max = 52 (+2 margin).
        let structure = structure_with_ids(vec![(50.0, 0.0, 0.0)]);
        let calc = SvergunIntensity::new(constant_ff(1.0), (0.5, 1.0), 2, None).unwrap();
        let l = calc.effective_l_max(&structure);
        assert_eq!(l, 52);
    }

    #[test]
    fn explicit_l_max_overrides_auto() {
        // Would auto-select >> 5, but user forced 5.
        let structure = structure_with_ids(vec![(50.0, 0.0, 0.0)]);
        let calc = SvergunIntensity::new(constant_ff(1.0), (0.5, 1.0), 2, Some(5)).unwrap();
        assert_eq!(calc.effective_l_max(&structure), 5);
    }
}

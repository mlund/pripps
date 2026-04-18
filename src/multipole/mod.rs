// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Scattering intensity via Spherical-harmonic multipole
//! expansion.
//!
//! Reference: Multipole, Barberato, Koch — CRYSOL, *J. Appl. Cryst.*
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
//! on the calculator pipeline. Solvent corrections (excluded volume
//! and hydration shell) feed into the same multipole kernel as dummy
//! scatterer sets via [`crate::solvent::SolventConfig`]; the
//! three-contribution combine lives in [`combine_contributions`].

mod math;

use crate::q_sampler::QSampler;
use crate::solvent::{DummySet, SolventConfig};
use crate::{
    Contributions, Intensity, IntensityCalculator, Result, Structure, Vector3,
    formfactor::{self, FormFactorMap},
};
use itertools::Itertools;
use math::{I_POW, LowerTriangle, Spherical, spherical_bessel, spherical_harmonics};
use num_complex::Complex;
use rayon::prelude::*;

/// Per-atom spherical-harmonic basis values: the spherical coordinates
/// of every atom and the conjugated `Y_l^m*` tables that the multipole
/// expansion evaluates at those coordinates.
///
/// Built once per structure by [`SphericalBasis::build`] and reused
/// across the q loop — the radial part `j_l(q·r)` is the only part of
/// the multipole kernel that depends on q, so it stays in the per-q
/// closure while everything angular lives here.
pub(crate) struct SphericalBasis {
    spherical: Vec<Spherical>,
    /// Per-atom, already-conjugated angular factors `Y_l^m*`.
    angular: Vec<LowerTriangle<Complex<f64>>>,
    l_max: usize,
}

impl SphericalBasis {
    pub(crate) fn new(positions: &[Vector3], l_max: usize) -> Self {
        let spherical: Vec<Spherical> = positions.iter().copied().map(Spherical::from).collect();

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

pub(crate) struct MultipoleIntensity {
    sampler: QSampler,
    l_max_override: Option<usize>,
    solvent: Option<SolventConfig>,
}

impl MultipoleIntensity {
    pub(crate) fn new(
        formfactors: FormFactorMap,
        q_minmax: (f64, f64),
        num_q: usize,
        l_max: Option<usize>,
        solvent: Option<SolventConfig>,
    ) -> Result<Self> {
        Ok(Self {
            sampler: QSampler::linear(formfactors, q_minmax.0, q_minmax.1, num_q),
            l_max_override: l_max,
            solvent,
        })
    }

    fn effective_l_max(&self, structure: &Structure) -> usize {
        resolve_l_max(self.l_max_override, &structure.pos, &self.sampler.q_values)
    }
}

/// Resolve the effective expansion order: user override if given,
/// else the Multipole criterion `⌈q_max · r_max⌉ + 2`.
///
/// Uses `r_max` (farthest atom from origin) rather than `R_g`
/// (radius of gyration) as a safe upper bound: `R_g` can
/// under-estimate the criterion for asymmetric structures, and
/// asking for too few multipoles silently gives a wrong answer.
/// The `+2` absorbs the usual "rule of thumb" margin — tests
/// confirm it's enough for agreement with Debye to 1% on small
/// clusters.
pub(crate) fn resolve_l_max(
    override_val: Option<usize>,
    positions: &[Vector3],
    q_values: &[f64],
) -> usize {
    if let Some(l) = override_val {
        return l;
    }
    let r_max: f64 = positions.iter().map(|p| p.norm()).fold(0.0_f64, f64::max);
    let q_max = q_values.iter().copied().fold(0.0_f64, f64::max);
    ((q_max * r_max).ceil() as usize) + 2
}

impl IntensityCalculator for MultipoleIntensity {
    fn calc_intensities(&self, structure: &Structure) -> Result<Intensity> {
        let l_max = self.effective_l_max(structure);
        log::info!(
            "Multipole expansion: l_max = {} over {} atoms, {} q-points",
            l_max,
            structure.pos.len(),
            self.sampler.q_values.len(),
        );
        // Built here (before the branch) so both the vacuum and the
        // three-contribution paths share one `O(N · L²)` basis
        // construction. Building it twice would be wasted work in
        // the solvent path — only the excluded/hydration dummy bases
        // need to be fresh.
        let atoms_basis = SphericalBasis::new(&structure.pos, l_max);

        match self.solvent.as_ref().filter(|cfg| cfg.is_active()) {
            None => vacuum_intensity(&self.sampler, &atoms_basis, &structure.ids, l_max),
            Some(cfg) => solvent_intensity(&self.sampler, structure, &atoms_basis, cfg),
        }
    }
}

/// Per-worker scratch reused across every q iteration inside one
/// rayon worker. `bessel` is the `j_l(q·r)` scratch for
/// [`multipole_coefficients`]; `form_factors` is refilled per q to
/// hold the evaluated per-atom form factors.
//
// Bundled into one struct because rayon's `par_iter().map_init(…)`
// takes a single per-worker state value; splitting into two
// independent scratch buffers would require two parallel `map_init`
// passes, one per buffer, which is not what rayon offers.
#[derive(Clone)]
struct PerQBuffers {
    bessel: Vec<f64>,
    form_factors: Vec<f64>,
}

impl PerQBuffers {
    fn new(l_max: usize) -> Self {
        Self {
            bessel: vec![0.0; l_max + 1],
            form_factors: Vec::new(),
        }
    }
}

/// A single scatterer set (protein atoms or a dummy-atom collection)
/// paired with its precomputed spherical-harmonic basis. Bundling
/// the two together keeps call sites from passing three parallel
/// slices by accident.
//
// The basis is borrowed, not owned: each `SphericalBasis` is built
// once in the enclosing function and reused across rayon workers,
// so every `Contribution` holds a shared reference with the same
// `'a` lifetime as the dummy set (when present).
struct Contribution<'a> {
    dummies: Option<&'a DummySet>,
    basis: &'a SphericalBasis,
}

impl<'a> Contribution<'a> {
    const fn atoms(basis: &'a SphericalBasis) -> Self {
        Self {
            dummies: None,
            basis,
        }
    }

    // Both parameters separate (instead of constructing `basis` from
    // `dummies.positions()`) so the caller can build one basis per
    // contribution up front and share it across all rayon workers;
    // building inside would force each worker to re-run the
    // `O(N · L²)` basis construction.
    const fn from_dummies(dummies: &'a DummySet, basis: &'a SphericalBasis) -> Self {
        Self {
            dummies: Some(dummies),
            basis,
        }
    }

    /// Multipole coefficients for this contribution at a single q.
    /// Atoms use the calculator-wide form-factor map indexed by
    /// `atom_ids`; dummy sets carry their own per-atom form factors.
    fn coefficients(
        &self,
        formfactors: &FormFactorMap,
        atom_ids: &[usize],
        q: f64,
        buffers: &mut PerQBuffers,
    ) -> Result<LowerTriangle<Complex<f64>>> {
        match self.dummies {
            None => {
                // `formfactor_by_index` still allocates a `DVector`;
                // copying into the reusable scratch `buffers.form_factors`
                // keeps the downstream API uniform with the dummy path.
                let ff = formfactor::formfactor_by_index(formfactors, q, atom_ids)?;
                buffers.form_factors.clear();
                buffers.form_factors.extend_from_slice(ff.as_slice());
            }
            Some(set) => {
                set.form_factors_at(q, &mut buffers.form_factors)?;
            }
        }
        Ok(multipole_coefficients(
            self.basis,
            &buffers.form_factors,
            q,
            &mut buffers.bessel,
        ))
    }
}

/// Vacuum Multipole: one scatterer set, single-column [`Intensity`].
fn vacuum_intensity(
    sampler: &QSampler,
    atoms_basis: &SphericalBasis,
    atom_ids: &[usize],
    l_max: usize,
) -> Result<Intensity> {
    let atoms_contrib = Contribution::atoms(atoms_basis);
    let total: Vec<f64> = sampler
        .q_values
        .par_iter()
        .map_init(
            || PerQBuffers::new(l_max),
            |buffers, &q| {
                let coeffs =
                    atoms_contrib.coefficients(&sampler.formfactors, atom_ids, q, buffers)?;
                Ok(intensity_from_multipoles(&coeffs, l_max))
            },
        )
        .collect::<Result<Vec<_>>>()?;

    Ok(Intensity {
        q: sampler.q_values.clone(),
        total,
        contributions: None,
    })
}

/// Three-contribution Multipole path: atoms + excluded volume +
/// hydration shell, combined coherently at the coefficient level.
fn solvent_intensity(
    sampler: &QSampler,
    structure: &Structure,
    atoms_basis: &SphericalBasis,
    cfg: &SolventConfig,
) -> Result<Intensity> {
    let fit = build_model(sampler, structure, atoms_basis, cfg)?;
    Ok(fit.intensity(
        cfg.excluded.volume_scale(),
        cfg.hydration.contrast_density(),
    ))
}

/// Precompute the three per-q coefficient triangles for a structure.
/// All the expensive, fit-invariant work lives here; the returned
/// [`MultipoleModel`] is cheap to re-query with different fit knobs.
pub(crate) fn build_model(
    sampler: &QSampler,
    structure: &Structure,
    atoms_basis: &SphericalBasis,
    cfg: &SolventConfig,
) -> Result<MultipoleModel> {
    let l_max = atoms_basis.l_max;
    let excluded_dummies =
        cfg.excluded
            .dummies(structure, &sampler.formfactors, cfg.bulk_electron_density)?;
    let hydration_dummies = cfg.hydration.dummies(structure, &sampler.formfactors)?;

    let excluded_basis = SphericalBasis::new(excluded_dummies.positions(), l_max);
    let hydration_basis = SphericalBasis::new(hydration_dummies.positions(), l_max);

    let atoms_contrib = Contribution::atoms(atoms_basis);
    let excluded_contrib = Contribution::from_dummies(&excluded_dummies, &excluded_basis);
    let hydration_contrib = Contribution::from_dummies(&hydration_dummies, &hydration_basis);

    // Per-q: compute the three coefficient triangles. This is the
    // O(N · Q · L²) work we want to do exactly once per structure.
    let per_q: Vec<_> = sampler
        .q_values
        .par_iter()
        .map_init(
            || PerQBuffers::new(l_max),
            |buffers, &q| -> Result<_> {
                let atoms =
                    atoms_contrib.coefficients(&sampler.formfactors, &structure.ids, q, buffers)?;
                let excluded = excluded_contrib.coefficients(
                    &sampler.formfactors,
                    &structure.ids,
                    q,
                    buffers,
                )?;
                let hydration = hydration_contrib.coefficients(
                    &sampler.formfactors,
                    &structure.ids,
                    q,
                    buffers,
                )?;
                Ok((atoms, excluded, hydration))
            },
        )
        .collect::<Result<Vec<_>>>()?;

    let (atoms_coeffs, excluded_coeffs, hydration_coeffs): (Vec<_>, Vec<_>, Vec<_>) =
        per_q.into_iter().multiunzip();

    debug_assert_eq!(atoms_coeffs.len(), sampler.q_values.len());
    debug_assert_eq!(excluded_coeffs.len(), sampler.q_values.len());
    debug_assert_eq!(hydration_coeffs.len(), sampler.q_values.len());

    Ok(MultipoleModel {
        q_values: sampler.q_values.clone(),
        l_max,
        atoms_coeffs,
        excluded_coeffs,
        hydration_coeffs,
    })
}

/// Cached three-contribution Multipole state for a single structure.
///
/// Holds the per-q multipole coefficient triangles for atoms,
/// excluded volume, and hydration shell. The expensive O(N·Q·L²)
/// work runs once at construction (via
/// [`Pripps::multipole_model`]); the cheap O(Q·L²) combine
/// runs per [`intensity`](MultipoleModel::intensity) call with
/// different fit knobs `(volume_scale, contrast_density)`.
pub struct MultipoleModel {
    q_values: Vec<f64>,
    l_max: usize,
    atoms_coeffs: Vec<LowerTriangle<Complex<f64>>>,
    excluded_coeffs: Vec<LowerTriangle<Complex<f64>>>,
    hydration_coeffs: Vec<LowerTriangle<Complex<f64>>>,
}

impl MultipoleModel {
    /// Compute `I(q)` for the given fit parameters. Infallible —
    /// the inner loop is pure complex arithmetic over the cached
    /// coefficient triangles, no form-factor evaluation, no basis
    /// construction, no SASA tessellation.
    //
    // Sequential (not `par_iter`): the per-q combine is ~L² complex
    // multiply-adds (≈600 for L=34). At Q=100 that's ~60 K flops —
    // microseconds of work — and rayon's per-task scheduling
    // overhead would dominate. The expensive O(N·Q·L²) coefficient
    // build in `build_model` correctly uses `par_iter`.
    pub fn intensity(&self, volume_scale: f64, contrast_density: f64) -> Intensity {
        let per_q: Vec<(f64, f64, f64, f64)> = self
            .atoms_coeffs
            .iter()
            .zip(&self.excluded_coeffs)
            .zip(&self.hydration_coeffs)
            .map(|((a, e), h)| {
                let total =
                    combine_contributions(a, e, h, self.l_max, volume_scale, contrast_density);
                (
                    total,
                    intensity_from_multipoles(a, self.l_max),
                    intensity_from_multipoles(e, self.l_max),
                    intensity_from_multipoles(h, self.l_max),
                )
            })
            .collect();

        let (total, atoms, excluded, hydration): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) =
            per_q.into_iter().multiunzip();

        Intensity {
            q: self.q_values.clone(),
            total,
            contributions: Some(Contributions {
                atoms,
                excluded,
                hydration,
            }),
        }
    }

    /// The q grid the fit was prepared over.
    pub fn q_values(&self) -> &[f64] {
        &self.q_values
    }
}

/// Build the Multipole coefficients for one scatterer set.
/// The complex triangle is what lets the three contributions
/// recombine coherently later; if each set returned a squared
/// intensity the cross terms would already be lost.
// PepsiSAXS `cSAS.cpp::addAtom` accumulates the same `A_l^m`, `C_l^m`,
// or `B_l^m` triangle depending on which scatterer set it's called with.
fn multipole_coefficients(
    basis: &SphericalBasis,
    form_factors_at_q: &[f64],
    q: f64,
    j_l_scratch: &mut [f64],
) -> LowerTriangle<Complex<f64>> {
    let l_max = basis.l_max;
    let mut coeffs = LowerTriangle::<Complex<f64>>::zeros(l_max);

    debug_assert!(j_l_scratch.len() > l_max);
    let j_l = &mut j_l_scratch[..=l_max];

    for ((s, y_conj), f) in basis
        .spherical
        .iter()
        .zip(&basis.angular)
        .zip(form_factors_at_q.iter().copied())
    {
        spherical_bessel(q * s.r, j_l);
        for l in 0..=l_max {
            let il_jl = I_POW[l % 4] * j_l[l];
            for m in 0..=l {
                coeffs[(l, m)] += il_jl * y_conj[(l, m)] * f;
            }
        }
    }
    coeffs
}

/// Orientation-averaged intensity from one multipole triangle.
//
// Σ_l [ |c(l, 0)|² + 2 Σ_{m>0} |c(l, m)|² ], times 4π.
// The factor of 2 on m > 0 accounts for the negative-m half via the
// symmetry `Y_l^{-m} = (-1)^m · conj(Y_l^m)`: storing only m ≥ 0
// doubles each m > 0 entry's contribution.
fn intensity_from_multipoles(coeffs: &LowerTriangle<Complex<f64>>, l_max: usize) -> f64 {
    let sum: f64 = (0..=l_max)
        .map(|l| {
            let m_zero = coeffs[(l, 0)].norm_sqr();
            let m_positive: f64 = (1..=l).map(|m| 2.0 * coeffs[(l, m)].norm_sqr()).sum();
            m_zero + m_positive
        })
        .sum();
    4.0 * std::f64::consts::PI * sum
}

/// Orientation-averaged intensity of the solvent-corrected scatterer:
/// the protein contribution minus the scaled excluded-volume
/// contribution plus the scaled hydration contribution, combined
/// coherently at the coefficient level and then squared and summed.
// PepsiSAXS `cSAS.cpp::computeIntensity` uses the same shape,
// `|A - C + dp·B|²`, for its final intensity.
fn combine_contributions(
    atoms: &LowerTriangle<Complex<f64>>,
    excluded: &LowerTriangle<Complex<f64>>,
    hydration: &LowerTriangle<Complex<f64>>,
    l_max: usize,
    volume_scale: f64,
    contrast_density: f64,
) -> f64 {
    let sum: f64 = (0..=l_max)
        .map(|l| {
            let combined = |m: usize| -> Complex<f64> {
                atoms[(l, m)] - volume_scale * excluded[(l, m)]
                    + contrast_density * hydration[(l, m)]
            };
            let m_zero = combined(0).norm_sqr();
            let m_positive: f64 = (1..=l).map(|m| 2.0 * combined(m).norm_sqr()).sum();
            m_zero + m_positive
        })
        .sum();
    4.0 * std::f64::consts::PI * sum
}

/// Single-q vacuum intensity: the pre-refactor entry point, kept for
/// the `multipole_refactor_is_no_op` test which asserts that the
/// split kernel matches the original's output bit-for-bit.
#[cfg(test)]
fn multipole_at_q(
    basis: &SphericalBasis,
    formfactors: &FormFactorMap,
    atom_ids: &[usize],
    q: f64,
) -> Result<f64> {
    let ff = formfactor::formfactor_by_index(formfactors, q, atom_ids)?;
    let mut j_l = vec![0.0; basis.l_max + 1];
    Ok(intensity_from_multipoles(
        &multipole_coefficients(basis, ff.as_slice(), q, &mut j_l),
        basis.l_max,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::debye::DebyeIntensity;
    use crate::formfactor::FormFactor;
    use crate::{AtomKind, Vector3};
    /// Placeholder displaced volume attached to the single kind
    /// built by [`constant_ff`]. Size is arbitrary; chosen to be
    /// comparable to a real atomic volume so Fraser-exercising tests
    /// produce plausible orders of magnitude.
    const TEST_DISPLACED_VOLUME: f64 = 10.0;

    /// Placeholder water form-factor amplitude attached to the
    /// `H2O` kind in [`constant_ff`] so SASA-exercising tests have
    /// a shared water form factor to look up by name.
    const TEST_WATER_FF: f64 = 10.0;

    /// Build a single-kind form-factor map: every atom uses
    /// `FormFactor::Constant(value)`, so `f(q) = value` for all q.
    /// The kind carries placeholder `sigma` and `displaced_volume`
    /// so tests enabling `ExcludedVolume::Fraser` or
    /// `Hydration::Sasa` pass their per-atom checks. The map also
    /// includes an `H2O` kind with [`TEST_WATER_FF`] so SASA
    /// hydration tests find a water form factor by name.
    fn constant_ff(value: f64) -> FormFactorMap {
        let mut kind = AtomKind::new("X");
        kind.displaced_volume = Some(TEST_DISPLACED_VOLUME);
        kind.sigma = Some(2.0 * 1.7);
        let mut water = AtomKind::new("H2O");
        water.displaced_volume = Some(30.0);
        water.sigma = Some(2.0 * 1.4);
        FormFactorMap::from_entries(vec![
            (kind, FormFactor::Constant(value)),
            (water, FormFactor::Constant(TEST_WATER_FF)),
        ])
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
        let calc = MultipoleIntensity::new(constant_ff(c), (0.1, 0.5), 10, None, None).unwrap();
        let structure = structure_with_ids(vec![(0.0, 0.0, 0.0)]);
        let out = calc.calc_intensities(&structure).unwrap();
        for i in &out.total {
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

        let multipole = MultipoleIntensity::new(constant_ff(c), (0.0, 0.1), 5, None, None).unwrap();
        let s_out = multipole.calc_intensities(&structure).unwrap();
        let s0 = s_out.total[0];

        let debye = DebyeIntensity::new(constant_ff(c), (0.0, 0.1), 5).unwrap();
        let d_out = debye.calc_intensities(&structure).unwrap();
        let d0 = d_out.total[0];

        let expected = (n * c) * (n * c);
        assert!(
            (s0 - expected).abs() < 1e-10,
            "multipole q=0: {s0} ≠ {expected}"
        );
        assert!(
            (d0 - expected).abs() < 1e-10,
            "debye q=0: {d0} ≠ {expected}"
        );
        assert!(
            (s0 - d0).abs() < 1e-10,
            "multipole ≠ debye at q=0: {s0} vs {d0}"
        );
    }

    #[test]
    fn agrees_with_debye_on_two_atoms() {
        // Closed-form: I_Debye(q) = 2 + 2 · sinc(q · r_12). Multipole
        // truncates the expansion at finite l_max so disagreement is
        // truncation-order (a few × 1e-6 here); not a correctness bug.
        let structure = structure_with_ids(vec![(0.0, 0.0, 0.0), (4.0, 0.0, 0.0)]);
        let qmin = 0.01;
        let qmax = 0.5;
        let nq = 20;

        let multipole =
            MultipoleIntensity::new(constant_ff(1.0), (qmin, qmax), nq, None, None).unwrap();
        let debye = DebyeIntensity::new(constant_ff(1.0), (qmin, qmax), nq).unwrap();

        let s_out = multipole.calc_intensities(&structure).unwrap();
        let d_out = debye.calc_intensities(&structure).unwrap();

        assert_eq!(s_out.q, d_out.q);
        for ((q, &is), &id) in s_out.q.iter().zip(&s_out.total).zip(&d_out.total) {
            let rel = (is - id).abs() / id.max(1e-10);
            assert!(
                rel < 1e-4,
                "q = {q}: multipole = {is}, debye = {id}, rel err = {rel}"
            );
        }
    }

    #[test]
    fn agrees_with_debye_on_small_cluster() {
        // 10 deterministic positions inside a 15-Å box; Multipole vs
        // Debye at 20 q values should agree to < 1% rel err.
        // Positions are a simple lattice shifted off the origin so
        // no atom sits at r = 0.
        let positions: Vec<(f64, f64, f64)> = (0..10)
            .map(|i| {
                let f = i as f64;
                (
                    f.mul_add(1.1, 1.5),
                    2.0 + (f * 0.7) % 5.0,
                    0.5 + (f * 1.3) % 6.0,
                )
            })
            .collect();
        let structure = structure_with_ids(positions);

        let qmin = 0.01;
        let qmax = 0.5;
        let nq = 20;

        let multipole =
            MultipoleIntensity::new(constant_ff(1.0), (qmin, qmax), nq, None, None).unwrap();
        let debye = DebyeIntensity::new(constant_ff(1.0), (qmin, qmax), nq).unwrap();

        let s_out = multipole.calc_intensities(&structure).unwrap();
        let d_out = debye.calc_intensities(&structure).unwrap();

        for ((q, &is), &id) in s_out.q.iter().zip(&s_out.total).zip(&d_out.total) {
            let rel = (is - id).abs() / id.max(1e-10);
            assert!(
                rel < 0.01,
                "q = {q}: multipole = {is}, debye = {id}, rel err = {rel}"
            );
        }
    }

    #[test]
    fn auto_l_max_respects_structure_extent() {
        // q_max · r_max = 1.0 · 50.0 = 50, so auto l_max = 52 (+2 margin).
        let structure = structure_with_ids(vec![(50.0, 0.0, 0.0)]);
        let calc = MultipoleIntensity::new(constant_ff(1.0), (0.5, 1.0), 2, None, None).unwrap();
        let l = calc.effective_l_max(&structure);
        assert_eq!(l, 52);
    }

    #[test]
    fn explicit_l_max_overrides_auto() {
        // Would auto-select >> 5, but user forced 5.
        let structure = structure_with_ids(vec![(50.0, 0.0, 0.0)]);
        let calc = MultipoleIntensity::new(constant_ff(1.0), (0.5, 1.0), 2, Some(5), None).unwrap();
        assert_eq!(calc.effective_l_max(&structure), 5);
    }

    #[test]
    fn fraser_zero_scale_matches_vacuum() {
        // volume_scale = 0 zeros the excluded-volume coefficient
        // triangle, so the combine reduces to |A|² — bit-identical
        // to the vacuum path.
        let structure = structure_with_ids(vec![
            (0.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 3.0, 0.0),
            (0.0, 0.0, 1.5),
        ]);
        let qmin = 0.01;
        let qmax = 0.4;
        let nq = 10;

        let vacuum =
            MultipoleIntensity::new(constant_ff(1.0), (qmin, qmax), nq, None, None).unwrap();
        let solvent_cfg = SolventConfig {
            excluded: crate::ExcludedVolume::Fraser { volume_scale: 0.0 },
            ..Default::default()
        };
        let solvent_calc =
            MultipoleIntensity::new(constant_ff(1.0), (qmin, qmax), nq, None, Some(solvent_cfg))
                .unwrap();

        let v_out = vacuum.calc_intensities(&structure).unwrap();
        let s_out = solvent_calc.calc_intensities(&structure).unwrap();

        assert_eq!(v_out.total, s_out.total);
        let contributions = s_out
            .contributions
            .as_ref()
            .expect("active solvent populates contributions");
        assert_eq!(s_out.total, contributions.atoms);
    }

    #[test]
    fn active_fraser_populates_columns() {
        let structure = structure_with_ids(vec![(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]);
        let nq = 5;
        let solvent_cfg = SolventConfig {
            excluded: crate::ExcludedVolume::Fraser { volume_scale: 1.0 },
            ..Default::default()
        };
        let calc =
            MultipoleIntensity::new(constant_ff(1.0), (0.0, 0.3), nq, None, Some(solvent_cfg))
                .unwrap();
        let out = calc.calc_intensities(&structure).unwrap();

        assert_eq!(out.total.len(), nq);
        let contributions = out.contributions.expect("contributions populated");
        assert_eq!(contributions.atoms.len(), nq);
        assert_eq!(contributions.excluded.len(), nq);
        assert_eq!(contributions.hydration.len(), nq);
        // Fraser produces real dummies, so excluded must be non-zero.
        assert!(contributions.excluded.iter().any(|&v| v > 0.0));
        // Phase 2a has no hydration generator, so that column is zero.
        assert!(contributions.hydration.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn fraser_reduces_total_below_atoms_at_qzero() {
        // At q=0 with positive form factors and positive displaced
        // volumes, the protein-minus-excluded sum is smaller than
        // protein alone, so `total ≤ atoms` and the excluded column
        // is strictly positive.
        let structure = structure_with_ids(vec![(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]);
        let solvent_cfg = SolventConfig {
            excluded: crate::ExcludedVolume::Fraser { volume_scale: 1.0 },
            ..Default::default()
        };
        let calc =
            MultipoleIntensity::new(constant_ff(6.0), (0.0, 0.05), 2, None, Some(solvent_cfg))
                .unwrap();
        let out = calc.calc_intensities(&structure).unwrap();
        let contributions = out.contributions.expect("contributions populated");

        // The first q is 0 (inclusive qmin).
        assert_eq!(out.q[0], 0.0);
        assert!(contributions.excluded[0] > 0.0);
        assert!(out.total[0] < contributions.atoms[0]);
    }

    #[test]
    fn sasa_zero_contrast_matches_vacuum() {
        // contrast_density = 0 zeros the hydration term in
        // `combine_contributions`, so total must match the vacuum
        // path even though the SASA generator runs and populates
        // the `hydration` column.
        let structure = structure_with_ids(vec![(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 3.0, 0.0)]);
        let qmin = 0.01;
        let qmax = 0.4;
        let nq = 8;

        let vacuum =
            MultipoleIntensity::new(constant_ff(1.0), (qmin, qmax), nq, None, None).unwrap();
        let solvent_cfg = SolventConfig {
            hydration: crate::Hydration::Sasa {
                probe: 1.4,
                contrast_density: 0.0,
            },
            ..Default::default()
        };
        let sasa_calc =
            MultipoleIntensity::new(constant_ff(1.0), (qmin, qmax), nq, None, Some(solvent_cfg))
                .unwrap();

        let v_out = vacuum.calc_intensities(&structure).unwrap();
        let s_out = sasa_calc.calc_intensities(&structure).unwrap();

        assert_eq!(v_out.total, s_out.total);
        let contributions = s_out.contributions.expect("contributions populated");
        // Hydration column was still computed — non-zero even at
        // zero contrast because the SASA dummies exist and the
        // per-dummy `|B_lm|²` diagnostic is reported.
        assert!(contributions.hydration.iter().any(|&v| v > 0.0));
    }

    #[test]
    fn fraser_plus_sasa_populates_all_three_columns() {
        // Both excluded and hydration active — every diagnostic
        // column should carry a non-zero contribution and the
        // total differs from any single one.
        let structure = structure_with_ids(vec![(0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.5, 0.0)]);
        let solvent_cfg = SolventConfig {
            excluded: crate::ExcludedVolume::Fraser { volume_scale: 1.0 },
            hydration: crate::Hydration::Sasa {
                probe: 1.4,
                contrast_density: 0.03,
            },
            ..Default::default()
        };
        let calc =
            MultipoleIntensity::new(constant_ff(6.0), (0.0, 0.2), 5, None, Some(solvent_cfg))
                .unwrap();
        let out = calc.calc_intensities(&structure).unwrap();
        let c = out.contributions.expect("contributions populated");

        assert!(c.atoms.iter().any(|&v| v > 0.0));
        assert!(c.excluded.iter().any(|&v| v > 0.0));
        assert!(c.hydration.iter().any(|&v| v > 0.0));
        // Total is the coherent combine, not any single column.
        assert!(out.total[0] != c.atoms[0]);
        assert!(out.total[0] != c.excluded[0]);
        assert!(out.total[0] != c.hydration[0]);
    }

    #[test]
    fn multipole_refactor_is_no_op() {
        let structure =
            structure_with_ids(vec![(1.0, 2.0, 0.5), (-1.5, 0.0, 1.0), (0.0, -2.0, 0.75)]);
        let ffs = constant_ff(1.0);
        let l_max = 6;
        let basis = SphericalBasis::new(&structure.pos, l_max);
        let mut j_l = vec![0.0; l_max + 1];

        for q in [0.05, 0.1, 0.25, 0.4] {
            let via_helper = multipole_at_q(&basis, &ffs, &structure.ids, q).unwrap();
            let ff = formfactor::formfactor_by_index(&ffs, q, &structure.ids).unwrap();
            let coeffs = multipole_coefficients(&basis, ff.as_slice(), q, &mut j_l);
            let via_split = intensity_from_multipoles(&coeffs, l_max);
            assert_eq!(via_helper, via_split);
        }
    }
}

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
use itertools::{Itertools, izip};
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
    /// All atoms' conjugated angular factors `Y_l^m*` packed
    /// end-to-end: `n_atoms` contiguous triangles of `tri_len`
    /// complex entries. Flat (vs `Vec<LowerTriangle>`) so the
    /// atoms-outer kernel can hand a worker a plain slice and the
    /// hardware prefetcher sees one linear stream.
    angular: Vec<Complex<f64>>,
    l_max: usize,
}

impl SphericalBasis {
    pub(crate) fn new(positions: &[Vector3], l_max: usize) -> Self {
        let spherical: Vec<Spherical> = positions.iter().copied().map(Spherical::from).collect();

        let tri_len = (l_max + 1) * (l_max + 2) / 2;
        let mut angular = Vec::with_capacity(spherical.len() * tri_len);
        for s in &spherical {
            let mut y = spherical_harmonics(s.theta, s.phi, l_max);
            // Pre-conjugate once so the per-q inner loop doesn't need
            // `.conj()` per (atom, l, m). Saves one complex negation
            // per entry in the hot accumulation path.
            for c in y.data_mut() {
                *c = c.conj();
            }
            angular.extend_from_slice(y.data());
        }

        Self {
            spherical,
            angular,
            l_max,
        }
    }

    /// Number of `Complex<f64>` slots per atom's angular triangle.
    const fn tri_len(&self) -> usize {
        (self.l_max + 1) * (self.l_max + 2) / 2
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

    #[cfg(test)]
    fn effective_l_max(&self, structure: &Structure) -> usize {
        resolve_l_max(self.l_max_override, &structure.pos, &self.sampler.q_values)
    }
}

/// Resolve the effective expansion order **per q-point**: user
/// override if given, else the Multipole criterion `⌈q · r_max⌉ + 2`
/// evaluated at each q.
///
/// Per-q lets low-q entries use a much smaller truncation (at q=0
/// only the monopole matters) while high-q keeps the safe upper
/// bound. On a linear q-grid from 0 to q_max the mean work per
/// atom scales as ~L_max²/3 instead of L_max² — a 3× kernel
/// speed-up with no accuracy loss.
///
/// Uses `r_max` (farthest atom from origin) rather than `R_g`
/// (radius of gyration) as a safe upper bound: `R_g` can
/// under-estimate the criterion for asymmetric structures, and
/// asking for too few multipoles silently gives a wrong answer.
/// The `+2` absorbs the usual "rule of thumb" margin — tests
/// confirm it's enough for agreement with Debye to 1% on small
/// clusters.
/// Hard cap on the auto-selected expansion order. The `r_max`-based
/// criterion can ask for hundreds of multipoles on large complexes
/// fitted against WAXS-range data (e.g. a 16 000-atom structure at
/// q_max ≈ 6 Å⁻¹ wants l_max ≈ 450). Above ~300 the per-atom angular
/// storage becomes GB-scale and the kernel approaches the point
/// where the `direct` Fourier-transform scheme is a better choice
/// anyway. Cap + warn lets the user keep going with a noted loss
/// of high-q convergence; override with `--lmax <N>` to force a
/// larger value.
const AUTO_L_MAX_CAP: usize = 300;

pub(crate) fn resolve_l_max_per_q(
    override_val: Option<usize>,
    positions: &[Vector3],
    q_values: &[f64],
) -> Vec<usize> {
    if let Some(l) = override_val {
        return vec![l; q_values.len()];
    }
    let r_max = positions.iter().map(|p| p.norm()).fold(0.0_f64, f64::max);
    let raw: Vec<usize> = q_values
        .iter()
        .map(|&q| ((q * r_max).ceil() as usize) + 2)
        .collect();
    let max_raw = raw.iter().copied().max().unwrap_or(0);
    if max_raw > AUTO_L_MAX_CAP {
        log::warn!(
            "capping auto-selected multipole order at l_max = {} \
             (criterion asked for {} given r_max = {:.1} Å and \
             q_max = {:.3} Å⁻¹). High-q convergence may be incomplete; \
             override with `--lmax <N>` or switch to the `direct` scheme.",
            AUTO_L_MAX_CAP,
            max_raw,
            r_max,
            q_values.iter().copied().fold(0.0_f64, f64::max),
        );
    }
    raw.into_iter().map(|l| l.min(AUTO_L_MAX_CAP)).collect()
}

/// Scalar maximum over per-q expansion orders. Kept for tests that
/// assert on the global l_max; production code uses the per-q vector
/// directly.
#[cfg(test)]
pub(crate) fn resolve_l_max(
    override_val: Option<usize>,
    positions: &[Vector3],
    q_values: &[f64],
) -> usize {
    resolve_l_max_per_q(override_val, positions, q_values)
        .into_iter()
        .max()
        .unwrap_or(0)
}

impl IntensityCalculator for MultipoleIntensity {
    fn calc_intensities(&self, structure: &Structure) -> Result<Intensity> {
        // Center the structure so r_max (and hence l_max) reflects
        // the maximum extent from the geometric center, not from
        // the arbitrary coordinate origin. This can halve l_max
        // for off-center structures, making the O(N·Q·L²) expansion
        // significantly cheaper.
        let mut structure = structure.clone();
        structure.center();
        let l_max_per_q =
            resolve_l_max_per_q(self.l_max_override, &structure.pos, &self.sampler.q_values);
        let l_max = l_max_per_q.iter().copied().max().unwrap_or(0);
        let l_min = l_max_per_q.iter().copied().min().unwrap_or(0);
        log::info!(
            "Multipole expansion: l_max ∈ [{l_min}, {l_max}] over {} atoms, {} q-points",
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
            None => vacuum_intensity(&self.sampler, &atoms_basis, &structure.ids, &l_max_per_q),
            Some(cfg) => {
                solvent_intensity(&self.sampler, &structure, &atoms_basis, cfg, &l_max_per_q)
            }
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

    /// Multipole coefficients for this contribution at a single q,
    /// truncated at `l_max_q`. Atoms use the calculator-wide form-
    /// factor map indexed by `atom_ids`; dummy sets carry their own
    /// per-atom form factors. The returned triangle is allocated at
    /// the basis' *global* `l_max` so downstream per-q combines see
    /// a uniform shape (rows `l > l_max_q` stay zero).
    fn coefficients(
        &self,
        formfactors: &FormFactorMap,
        atom_ids: &[usize],
        q: f64,
        l_max_q: usize,
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
            l_max_q,
            &mut buffers.bessel,
        ))
    }
}

/// Vacuum Multipole: one scatterer set, single-column [`Intensity`].
fn vacuum_intensity(
    sampler: &QSampler,
    atoms_basis: &SphericalBasis,
    atom_ids: &[usize],
    l_max_per_q: &[usize],
) -> Result<Intensity> {
    let atoms_contrib = Contribution::atoms(atoms_basis);
    let coeffs_per_q = compute_coeffs(
        atoms_basis,
        &atoms_contrib,
        &sampler.formfactors,
        atom_ids,
        &sampler.q_values,
        l_max_per_q,
    )?;

    let total: Vec<f64> = coeffs_per_q
        .par_iter()
        .zip(l_max_per_q)
        .map(|(c, &l_q)| intensity_from_multipoles(c, l_q))
        .collect();

    Ok(Intensity {
        q: sampler.q_values.clone(),
        total,
        contributions: None,
        ..Default::default()
    })
}

/// Three-contribution Multipole path: atoms + excluded volume +
/// hydration shell, combined coherently at the coefficient level.
fn solvent_intensity(
    sampler: &QSampler,
    structure: &Structure,
    atoms_basis: &SphericalBasis,
    cfg: &SolventConfig,
    l_max_per_q: &[usize],
) -> Result<Intensity> {
    let fit = build_model(sampler, structure, atoms_basis, cfg, l_max_per_q)?;
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
    l_max_per_q: &[usize],
) -> Result<MultipoleModel> {
    let l_max = atoms_basis.l_max;
    let excluded_dummies = cfg.excluded.unit_scale_dummies(
        structure,
        &sampler.formfactors,
        cfg.bulk_electron_density,
    )?;
    let excluded_deriv_dummies = cfg.excluded.unit_scale_derivative_dummies(
        structure,
        &sampler.formfactors,
        cfg.bulk_electron_density,
    )?;
    let hydration_dummies = cfg.hydration.dummies(structure, &sampler.formfactors)?;

    let excluded_basis = SphericalBasis::new(excluded_dummies.positions(), l_max);
    let hydration_basis = SphericalBasis::new(hydration_dummies.positions(), l_max);

    let atoms_contrib = Contribution::atoms(atoms_basis);
    let excluded_contrib = Contribution::from_dummies(&excluded_dummies, &excluded_basis);
    let hydration_contrib = Contribution::from_dummies(&hydration_dummies, &hydration_basis);

    // Each contribution runs as its own atoms-outer, q-inner kernel
    // (when large enough) so every atom's y_conj triangle is read from
    // DRAM once and reused across all q-points in L2. Falls back to
    // the per-q schedule for small inputs where rayon fold overhead
    // dominates the bandwidth win.
    let atoms_coeffs = compute_coeffs(
        atoms_basis,
        &atoms_contrib,
        &sampler.formfactors,
        &structure.ids,
        &sampler.q_values,
        l_max_per_q,
    )?;
    let excluded_coeffs = compute_coeffs(
        &excluded_basis,
        &excluded_contrib,
        &sampler.formfactors,
        &structure.ids,
        &sampler.q_values,
        l_max_per_q,
    )?;
    let hydration_coeffs = compute_coeffs(
        &hydration_basis,
        &hydration_contrib,
        &sampler.formfactors,
        &structure.ids,
        &sampler.q_values,
        l_max_per_q,
    )?;

    // C1 derivative triangles (PepsiSAXS `Clm1`) for the first-order
    // width correction, or all-zero triangles for variants without one.
    let excluded_deriv_coeffs = match &excluded_deriv_dummies {
        Some(deriv_dummies) => {
            // C1 dummies sit at the same atom positions as C0, so the
            // spherical basis is identical — reuse it instead of rebuilding
            // the O(N·L²) angular/radial table.
            debug_assert_eq!(deriv_dummies.positions(), excluded_dummies.positions());
            let deriv_contrib = Contribution::from_dummies(deriv_dummies, &excluded_basis);
            compute_coeffs(
                &excluded_basis,
                &deriv_contrib,
                &sampler.formfactors,
                &structure.ids,
                &sampler.q_values,
                l_max_per_q,
            )?
        }
        None => sampler
            .q_values
            .iter()
            .map(|_| LowerTriangle::zeros(l_max))
            .collect(),
    };

    debug_assert_eq!(atoms_coeffs.len(), sampler.q_values.len());
    debug_assert_eq!(excluded_coeffs.len(), sampler.q_values.len());
    debug_assert_eq!(excluded_deriv_coeffs.len(), sampler.q_values.len());
    debug_assert_eq!(hydration_coeffs.len(), sampler.q_values.len());
    debug_assert_eq!(l_max_per_q.len(), sampler.q_values.len());

    Ok(MultipoleModel {
        q_values: sampler.q_values.clone(),
        l_max_per_q: l_max_per_q.to_vec(),
        atoms_coeffs,
        excluded_coeffs,
        excluded_deriv_coeffs,
        hydration_coeffs,
    })
}

/// Cached three-contribution Multipole state for a single structure.
///
/// Holds the per-q multipole coefficient triangles for atoms,
/// excluded volume, and hydration shell. The expensive O(N·Q·L²)
/// work runs once at construction (via
/// [`Pripps::multipole_model`](crate::Pripps::multipole_model)); the cheap O(Q·L²) combine
/// runs per [`intensity`](MultipoleModel::intensity) call with
/// different fit knobs `(volume_scale, contrast_density)`.
pub struct MultipoleModel {
    q_values: Vec<f64>,
    /// Per-q expansion orders. Each coefficient triangle is allocated
    /// at the *global* max of this vector (rows above `l_max_per_q[i]`
    /// stay zero for that q), so the combine just slices the used rows.
    l_max_per_q: Vec<usize>,
    atoms_coeffs: Vec<LowerTriangle<Complex<f64>>>,
    excluded_coeffs: Vec<LowerTriangle<Complex<f64>>>,
    /// First-order Taylor derivative (`C1`) coefficients of the excluded
    /// volume — PepsiSAXS' `Clm1`. All-zero triangles for variants
    /// without an analytic width correction.
    excluded_deriv_coeffs: Vec<LowerTriangle<Complex<f64>>>,
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
        let per_q: Vec<(f64, f64, f64, f64)> = izip!(
            &self.atoms_coeffs,
            &self.excluded_coeffs,
            &self.excluded_deriv_coeffs,
            &self.hydration_coeffs,
            &self.l_max_per_q,
        )
        .map(|(a, e, ed, h, &l_q)| {
            let total = combine_contributions(a, e, ed, h, l_q, volume_scale, contrast_density);
            (
                total,
                intensity_from_multipoles(a, l_q),
                intensity_from_multipoles(e, l_q),
                intensity_from_multipoles(h, l_q),
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
            ..Default::default()
        }
    }

    /// The q grid the fit was prepared over.
    pub fn q_values(&self) -> &[f64] {
        &self.q_values
    }
}

/// Minimum atom count above which [`compute_coeffs`] routes to the
/// atoms-outer kernel. Below this, the one-time form-factor
/// transpose and per-worker accumulator allocations dominate the
/// bandwidth savings and q-outer wins.
///
/// Calibrated from a sweep over synthetic clusters (n ∈ {100…20 000},
/// n_q ∈ {20, 50, 100}) on 10-core Apple Silicon: atoms-outer lost
/// 10-25 % for n ≤ 2 000 across all n_q, was break-even at n = 5 000,
/// and won 5-28 % for n ≥ 10 000. Picking the atom count alone (not
/// the n_atoms × n_q product) matches the data better — AO's fixed
/// per-call overhead (ff transpose + per-tile alloc + reduction)
/// scales with n_atoms and n_threads, not with n_q.
const ATOMS_OUTER_MIN_ATOMS: usize = 5_000;

/// Accumulate one atom's contribution to every triangle entry at a
/// single q: `coeffs[l,m] += (iˡ · jₗ(qr) · f) · y[l,m]` packed
/// lower-triangular. Shared between the q-outer and atoms-outer
/// kernels; `#[inline(always)]` keeps LLVM's autovectorisation
/// (NEON `ld2.2d`/`st2.2d` + `fmla`/`fmls` on ARM, 256-bit AVX2
/// equivalents on x86) intact at both call sites.
#[inline(always)]
fn accumulate_triangle(
    coeffs: &mut [Complex<f64>],
    y: &[Complex<f64>],
    j_l: &[f64],
    f: f64,
    l_max: usize,
) {
    let mut row_start = 0usize;
    for l in 0..=l_max {
        let a = I_POW[l % 4] * j_l[l] * f;
        let (a_re, a_im) = (a.re, a.im);
        let row_len = l + 1;
        let cs = &mut coeffs[row_start..row_start + row_len];
        let ys = &y[row_start..row_start + row_len];
        for (c, y) in cs.iter_mut().zip(ys) {
            c.re = a_re.mul_add(y.re, (-a_im).mul_add(y.im, c.re));
            c.im = a_re.mul_add(y.im, a_im.mul_add(y.re, c.im));
        }
        row_start += row_len;
    }
}

/// Per-atom form factors at every q, laid out atom-major so the
/// atoms-outer kernel reads a contiguous `n_q`-run per atom.
/// Parallel over q, one sequential transpose.
//
// Two-pass (q-major → transpose) because the existing form-factor
// primitives (`formfactor_by_index`, `DummySet::form_factors_at`)
// return a per-q vector over all atoms; producing atom-major
// directly would mean duplicating that evaluation logic. Transpose
// is O(n_atoms · n_q) memory writes (~73 MB for 1HTQ/100 q) — an
// order of magnitude below the kernel.
fn precompute_ff_atom_major(
    contribution: &Contribution<'_>,
    formfactors: &FormFactorMap,
    atom_ids: &[usize],
    q_values: &[f64],
    n_atoms: usize,
) -> Result<Vec<f64>> {
    let n_q = q_values.len();

    let q_major: Vec<Vec<f64>> = q_values
        .par_iter()
        .map(|&q| -> Result<Vec<f64>> {
            let mut row = Vec::with_capacity(n_atoms);
            match contribution.dummies {
                None => {
                    let ff = formfactor::formfactor_by_index(formfactors, q, atom_ids)?;
                    row.extend_from_slice(ff.as_slice());
                }
                Some(set) => set.form_factors_at(q, &mut row)?,
            }
            debug_assert_eq!(row.len(), n_atoms);
            Ok(row)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut atom_major = vec![0.0; n_atoms * n_q];
    for (q_idx, row) in q_major.iter().enumerate() {
        for (atom_idx, &v) in row.iter().enumerate() {
            atom_major[atom_idx * n_q + q_idx] = v;
        }
    }
    Ok(atom_major)
}

/// q-tile width. Smaller keeps the per-thread accumulator
/// (`q_block · tri_len · 16` bytes) within per-core L2; larger
/// amortises rayon scheduling across more atom-iters per tile.
/// Thresholds empirical — 10-core Apple Silicon → 4, 64-thread
/// Threadripper → 16; the binding constraint is atoms-per-thread.
fn pick_q_block(n_atoms: usize, n_threads: usize) -> usize {
    let atoms_per_thread = n_atoms.div_ceil(n_threads.max(1)).max(1);
    if atoms_per_thread >= 5000 {
        4
    } else if atoms_per_thread >= 2000 {
        8
    } else {
        16
    }
}

/// Atoms-outer, q-inner multipole kernel with q-tile blocking. Each
/// rayon worker owns one atom range and one accumulator per q-tile,
/// so every atom's `y_conj` triangle is read from DRAM once per tile
/// (not once per q). Results match the q-outer schedule to FP
/// tolerance, not bit-exactly — accumulation order depends on rayon
/// distribution; see `atoms_outer_matches_q_outer` for the bound.
fn atoms_outer_coefficients(
    basis: &SphericalBasis,
    ff_atom_major: &[f64],
    q_values: &[f64],
    l_max_per_q: &[usize],
) -> Vec<LowerTriangle<Complex<f64>>> {
    let l_max = basis.l_max;
    let tri_len = basis.tri_len();
    let n_q = q_values.len();
    let n_atoms = basis.spherical.len();
    debug_assert_eq!(basis.angular.len(), n_atoms * tri_len);
    debug_assert_eq!(ff_atom_major.len(), n_atoms * n_q);
    debug_assert_eq!(l_max_per_q.len(), n_q);

    // One atom range per rayon worker: rayon distributes but cannot
    // split further, so we allocate one accumulator per worker.
    // The earlier `fold`-based version churned the allocator on
    // many-core systems.
    let n_threads = rayon::current_num_threads().max(1);
    let chunk_size = n_atoms.div_ceil(n_threads);
    let atom_chunks: Vec<std::ops::Range<usize>> = (0..n_atoms)
        .step_by(chunk_size)
        .map(|start| start..(start + chunk_size).min(n_atoms))
        .collect();

    let q_block = pick_q_block(n_atoms, n_threads);
    // Pre-zeroed so the `+=` reduction below handles the first
    // partial uniformly with the rest — no copy_from_slice special
    // case for partial[0].
    let mut final_coeffs = vec![Complex::default(); n_q * tri_len];

    // q-tile outer (not atom outer) so that within one tile the hot
    // accumulator region is `block_len × tri_len × 16 B` (≈ L2) and
    // each atom's `y_conj` is read once per tile. Atom outer would
    // make the entire `n_q × tri_len` accumulator hot per atom.
    for q_start in (0..n_q).step_by(q_block) {
        let q_end = (q_start + q_block).min(n_q);
        let block_len = q_end - q_start;

        let partials: Vec<Vec<Complex<f64>>> = atom_chunks
            .par_iter()
            .map(|range| {
                let mut coeffs = vec![Complex::default(); block_len * tri_len];
                let mut j_l = vec![0.0; l_max + 1];
                for atom_idx in range.start..range.end {
                    let s = basis.spherical[atom_idx];
                    let y = &basis.angular[atom_idx * tri_len..(atom_idx + 1) * tri_len];
                    let ff = &ff_atom_major[atom_idx * n_q..(atom_idx + 1) * n_q];
                    for (local_q, c_row) in coeffs.chunks_exact_mut(tri_len).enumerate() {
                        let q_idx = q_start + local_q;
                        let l_q = l_max_per_q[q_idx];
                        // Shorter Bessel slice at low q: Miller's
                        // backward recurrence is O(l_start²) in work,
                        // so truncating saves real time.
                        let j_slice = &mut j_l[..=l_q];
                        spherical_bessel(q_values[q_idx] * s.r, j_slice);
                        accumulate_triangle(c_row, y, j_slice, ff[q_idx], l_q);
                    }
                }
                coeffs
            })
            .collect();

        let dst = &mut final_coeffs[q_start * tri_len..q_end * tri_len];
        for partial in &partials {
            for (d, p) in dst.iter_mut().zip(partial) {
                *d += p;
            }
        }
    }

    // Rewrap as `Vec<LowerTriangle>` so downstream consumers
    // (`intensity_from_multipoles`, `combine_contributions`,
    // `MultipoleModel`) can index by `(l, m)`. The flat form is an
    // internal detail of this kernel.
    final_coeffs
        .chunks_exact(tri_len)
        .map(|chunk| {
            let mut tri = LowerTriangle::<Complex<f64>>::zeros(l_max);
            tri.data_mut().copy_from_slice(chunk);
            tri
        })
        .collect()
}

/// Pick the atoms-outer kernel for large work, else the per-q
/// schedule. Shared dispatch for vacuum and solvent paths so both
/// honour the same threshold.
fn compute_coeffs(
    basis: &SphericalBasis,
    contribution: &Contribution<'_>,
    formfactors: &FormFactorMap,
    atom_ids: &[usize],
    q_values: &[f64],
    l_max_per_q: &[usize],
) -> Result<Vec<LowerTriangle<Complex<f64>>>> {
    debug_assert_eq!(l_max_per_q.len(), q_values.len());
    let n_atoms = basis.spherical.len();
    let basis_l_max = basis.l_max;

    if n_atoms >= ATOMS_OUTER_MIN_ATOMS {
        log::info!(
            "compute_coeffs: atoms-outer schedule (n_atoms = {n_atoms}, n_q = {})",
            q_values.len(),
        );
        let ff = precompute_ff_atom_major(contribution, formfactors, atom_ids, q_values, n_atoms)?;
        Ok(atoms_outer_coefficients(basis, &ff, q_values, l_max_per_q))
    } else {
        log::info!(
            "compute_coeffs: q-outer schedule (n_atoms = {n_atoms}, n_q = {})",
            q_values.len(),
        );
        q_values
            .par_iter()
            .zip(l_max_per_q)
            .map_init(
                || PerQBuffers::new(basis_l_max),
                |buffers, (&q, &l_q)| {
                    contribution.coefficients(formfactors, atom_ids, q, l_q, buffers)
                },
            )
            .collect::<Result<Vec<_>>>()
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
    l_max_q: usize,
    j_l_scratch: &mut [f64],
) -> LowerTriangle<Complex<f64>> {
    // Allocate the triangle at the basis' global l_max so downstream
    // per-q combines see a uniform shape; rows `l > l_max_q` stay
    // zero from the `zeros()` init and are skipped at sum time.
    let tri_len = basis.tri_len();
    let mut coeffs = LowerTriangle::<Complex<f64>>::zeros(basis.l_max);

    debug_assert!(l_max_q <= basis.l_max);
    debug_assert!(j_l_scratch.len() > l_max_q);
    let j_l = &mut j_l_scratch[..=l_max_q];

    for ((s, y), f) in basis
        .spherical
        .iter()
        .zip(basis.angular.chunks_exact(tri_len))
        .zip(form_factors_at_q.iter().copied())
    {
        spherical_bessel(q * s.r, j_l);
        accumulate_triangle(coeffs.data_mut(), y, j_l, f, l_max_q);
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
// `|A - C + dp·B|²`, for its final intensity, with the excluded
// amplitude `C = f0·C0 + f1·C1` split across a base and a first-order
// Taylor derivative basis (`Clm1`).
fn combine_contributions(
    atoms: &LowerTriangle<Complex<f64>>,
    excluded: &LowerTriangle<Complex<f64>>,
    excluded_deriv: &LowerTriangle<Complex<f64>>,
    hydration: &LowerTriangle<Complex<f64>>,
    l_max: usize,
    volume_scale: f64,
    contrast_density: f64,
) -> f64 {
    let (f0, f1) = crate::solvent::taylor_coefficients(volume_scale);
    let sum: f64 = (0..=l_max)
        .map(|l| {
            let combined = |m: usize| -> Complex<f64> {
                atoms[(l, m)] - f0 * excluded[(l, m)] - f1 * excluded_deriv[(l, m)]
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
        &multipole_coefficients(basis, ff.as_slice(), q, basis.l_max, &mut j_l),
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
            let coeffs = multipole_coefficients(&basis, ff.as_slice(), q, l_max, &mut j_l);
            let via_split = intensity_from_multipoles(&coeffs, l_max);
            assert_eq!(via_helper, via_split);
        }
    }

    #[test]
    fn atoms_outer_matches_q_outer() {
        // Call both kernels directly on the same moderate cluster
        // (3 375 atoms × 40 q) and compare entry-by-entry.
        // Accumulation order differs between the two schedules so
        // we assert agreement to `1e-10` relative rather than
        // bit-equality.
        let mut positions = Vec::new();
        let side = 15;
        for i in 0..side {
            for j in 0..side {
                for k in 0..side {
                    positions.push(((i as f64) * 1.7, (j as f64) * 1.7, (k as f64) * 1.7));
                }
            }
        }
        let structure = structure_with_ids(positions);
        assert_eq!(structure.pos.len(), 3375);
        let ffs = constant_ff(1.0);
        let l_max = 8;
        let q_values: Vec<f64> = (0..40).map(|i| 0.05 + (i as f64) * 0.01).collect();

        let basis = SphericalBasis::new(&structure.pos, l_max);

        let contrib = Contribution::atoms(&basis);
        let ff_matrix = precompute_ff_atom_major(
            &contrib,
            &ffs,
            &structure.ids,
            &q_values,
            structure.pos.len(),
        )
        .unwrap();
        let l_max_per_q = vec![l_max; q_values.len()];
        let coeffs_outer = atoms_outer_coefficients(&basis, &ff_matrix, &q_values, &l_max_per_q);

        let coeffs_q: Vec<_> = q_values
            .iter()
            .map(|&q| {
                let mut buf = PerQBuffers::new(l_max);
                contrib
                    .coefficients(&ffs, &structure.ids, q, l_max, &mut buf)
                    .unwrap()
            })
            .collect();

        assert_eq!(coeffs_outer.len(), coeffs_q.len());
        for (q_idx, (b, r)) in coeffs_outer.iter().zip(&coeffs_q).enumerate() {
            for l in 0..=l_max {
                for m in 0..=l {
                    let vb = b[(l, m)];
                    let vr = r[(l, m)];
                    let scale = vr.norm().max(1.0);
                    assert!(
                        (vb - vr).norm() <= 1e-10 * scale,
                        "q_idx={q_idx} (l,m)=({l},{m}) outer={vb:?} q-outer={vr:?}",
                    );
                }
            }
        }
    }
}

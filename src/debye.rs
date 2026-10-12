// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

use crate::q_sampler::QSampler;
use crate::solvent::SolventConfig;
use crate::{
    Contributions, Intensity, IntensityCalculator, Result, Structure, Vector3,
    formfactor::{self, FormFactorMap},
};
use itertools::Itertools;
use rayon::prelude::*;

/// Returns sinc(x) = sin(x)/x with graceful handling of x -> 0
#[inline]
pub fn sinc(x: f64) -> f64 {
    const NEARLY_ZERO: f64 = 1e-8;
    if x.abs() < NEARLY_ZERO {
        1.0
    } else {
        x.sin() / x
    }
}

pub(crate) struct DebyeIntensity {
    sampler: QSampler,
    solvent: Option<SolventConfig>,
}

impl DebyeIntensity {
    #[cfg(test)]
    pub(crate) fn new(
        formfactors: FormFactorMap,
        q_minmax: (f64, f64),
        num_q: usize,
    ) -> Result<Self> {
        Self::new_with_solvent(formfactors, q_minmax, num_q, None)
    }

    pub(crate) fn new_with_solvent(
        formfactors: FormFactorMap,
        q_minmax: (f64, f64),
        num_q: usize,
        solvent: Option<SolventConfig>,
    ) -> Result<Self> {
        Ok(Self {
            sampler: QSampler::linear(formfactors, q_minmax.0, q_minmax.1, num_q),
            solvent,
        })
    }
}

impl IntensityCalculator for DebyeIntensity {
    /// Calculate I(q) over a range of q values using the Debye formula
    fn calc_intensities(&self, structure: &Structure) -> Result<Intensity> {
        match self.solvent.as_ref().filter(|cfg| cfg.is_active()) {
            None => self.sampler.run_parallel(|q| {
                intensity_at_q(&structure.ids, &structure.pos, &self.sampler.formfactors, q)
            }),
            Some(cfg) => {
                let model = build_model(&self.sampler, structure, cfg)?;
                Ok(model.intensity(
                    cfg.excluded.volume_scale(),
                    cfg.hydration.contrast_density(),
                ))
            }
        }
    }
}

/// Precompute Debye pair sums for atoms, excluded volume, and hydration.
///
/// Unlike the vacuum Debye path, fitting needs repeated evaluations at
/// different `(volume_scale, contrast_density)`. The full Debye sum is
/// quadratic in those two scalar knobs, so caching the three self terms
/// and three cross terms per q makes refits cheap.
pub(crate) fn build_model(
    sampler: &QSampler,
    structure: &Structure,
    cfg: &SolventConfig,
) -> Result<DebyeModel> {
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

    let terms = sampler
        .q_values
        .par_iter()
        .map(|&q| {
            let atom_ff = formfactor::formfactor_by_index(&sampler.formfactors, q, &structure.ids)?;

            let mut excluded_ff = Vec::new();
            excluded_dummies.form_factors_at(q, &mut excluded_ff)?;

            let mut hydration_ff = Vec::new();
            hydration_dummies.form_factors_at(q, &mut hydration_ff)?;

            let atom_ff = atom_ff.as_slice();
            let mut terms = DebyeTerms {
                atoms: self_intensity_at_q(&structure.pos, atom_ff, q),
                excluded: self_intensity_at_q(excluded_dummies.positions(), &excluded_ff, q),
                hydration: self_intensity_at_q(hydration_dummies.positions(), &hydration_ff, q),
                atom_excluded: cross_intensity_at_q(
                    &structure.pos,
                    atom_ff,
                    excluded_dummies.positions(),
                    &excluded_ff,
                    q,
                ),
                atom_hydration: cross_intensity_at_q(
                    &structure.pos,
                    atom_ff,
                    hydration_dummies.positions(),
                    &hydration_ff,
                    q,
                ),
                excluded_hydration: cross_intensity_at_q(
                    excluded_dummies.positions(),
                    &excluded_ff,
                    hydration_dummies.positions(),
                    &hydration_ff,
                    q,
                ),
                ..DebyeTerms::default()
            };

            // First-order Taylor width-correction terms, only for the
            // Gaussian excluded variants that expose a `C1` basis.
            if let Some(deriv_dummies) = &excluded_deriv_dummies {
                let mut deriv_ff = Vec::new();
                deriv_dummies.form_factors_at(q, &mut deriv_ff)?;
                let deriv_pos = deriv_dummies.positions();
                terms.excluded_deriv = self_intensity_at_q(deriv_pos, &deriv_ff, q);
                terms.excluded_cross = cross_intensity_at_q(
                    excluded_dummies.positions(),
                    &excluded_ff,
                    deriv_pos,
                    &deriv_ff,
                    q,
                );
                terms.atom_excluded_deriv =
                    cross_intensity_at_q(&structure.pos, atom_ff, deriv_pos, &deriv_ff, q);
                terms.excluded_deriv_hydration = cross_intensity_at_q(
                    deriv_pos,
                    &deriv_ff,
                    hydration_dummies.positions(),
                    &hydration_ff,
                    q,
                );
            }

            Ok(terms)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(DebyeModel {
        q_values: sampler.q_values.clone(),
        terms,
    })
}

/// Cached Debye state for solvent fitting.
pub struct DebyeModel {
    q_values: Vec<f64>,
    terms: Vec<DebyeTerms>,
}

impl DebyeModel {
    /// Compute `I(q)` for the given solvent fit knobs.
    pub fn intensity(&self, volume_scale: f64, contrast_density: f64) -> Intensity {
        let per_q: Vec<(f64, f64, f64, f64)> = self
            .terms
            .iter()
            .map(|terms| {
                let total = terms.combine(volume_scale, contrast_density);
                (total, terms.atoms, terms.excluded, terms.hydration)
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

#[derive(Debug, Clone, Copy, Default)]
struct DebyeTerms {
    atoms: f64,
    excluded: f64,
    hydration: f64,
    atom_excluded: f64,
    atom_hydration: f64,
    excluded_hydration: f64,
    // First-order Taylor width-correction pair sums (all zero when the
    // excluded variant has no analytic derivative basis `C1`).
    /// S(C0, C1) — base × derivative excluded.
    excluded_cross: f64,
    /// S(C1, C1) — derivative excluded self.
    excluded_deriv: f64,
    /// S(atoms, C1) — atom × derivative excluded.
    atom_excluded_deriv: f64,
    /// S(C1, hydration) — derivative excluded × hydration.
    excluded_deriv_hydration: f64,
}

impl DebyeTerms {
    fn combine(self, volume_scale: f64, contrast_density: f64) -> f64 {
        let (f0, f1) = crate::solvent::taylor_coefficients(volume_scale);
        self.atoms
            + f0 * f0 * self.excluded
            + 2.0 * f0 * f1 * self.excluded_cross
            + f1 * f1 * self.excluded_deriv
            + contrast_density * contrast_density * self.hydration
            - 2.0 * f0 * self.atom_excluded
            - 2.0 * f1 * self.atom_excluded_deriv
            + 2.0 * contrast_density * self.atom_hydration
            - 2.0 * f0 * contrast_density * self.excluded_hydration
            - 2.0 * f1 * contrast_density * self.excluded_deriv_hydration
    }
}

/// Calculate I(q) using the Debye formula.
///
/// Distances and form factor products are computed on-the-fly to avoid N×N matrix allocations.
fn intensity_at_q(
    atom_ids: &[usize],
    positions: &[Vector3],
    formfactors: &FormFactorMap,
    q: f64,
) -> Result<f64> {
    log::trace!("Calculating 𝐼(𝑞) for 𝑞 = {q}");

    // TODO: Optimize by storing in static table, see std::sync::OnceLock
    let ff = formfactor::formfactor_by_index(formfactors, q, atom_ids)?;
    Ok(self_intensity_at_q(positions, ff.as_slice(), q))
}

/// Self Debye sum: Σ_i f_i² + 2 Σ_{i<j} f_i f_j sinc(q r_ij).
fn self_intensity_at_q(positions: &[Vector3], form_factors: &[f64], q: f64) -> f64 {
    let n = form_factors.len();
    let diagonal: f64 = form_factors.iter().map(|f| f * f).sum();
    if n < 2 {
        return diagonal;
    }

    let mut off_diagonal = 0.0;
    for i in 0..n - 1 {
        let fi = form_factors[i];
        let pi = &positions[i];
        for j in i + 1..n {
            let r = (pi - positions[j]).norm();
            off_diagonal += fi * form_factors[j] * sinc(q * r);
        }
    }

    2.0f64.mul_add(off_diagonal, diagonal)
}

/// Cross Debye sum between two disjoint scatterer groups:
/// Σ_i∈a Σ_j∈b f_i f_j sinc(q r_ij).
fn cross_intensity_at_q(
    a_positions: &[Vector3],
    a_form_factors: &[f64],
    b_positions: &[Vector3],
    b_form_factors: &[f64],
    q: f64,
) -> f64 {
    let mut total = 0.0;
    for (pa, &fa) in a_positions.iter().zip(a_form_factors) {
        for (pb, &fb) in b_positions.iter().zip(b_form_factors) {
            total += fa * fb * sinc(q * (pa - pb).norm());
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::formfactor::FormFactor;
    use crate::solvent::ExcludedVolume;
    use crate::{AtomKind, Structure};

    const RHO: f64 = 0.334;

    /// Four carbons with a displaced volume and a flat form factor.
    fn structure_and_map() -> (Structure, FormFactorMap) {
        let mut kind = AtomKind::new("C");
        kind.displaced_volume = Some(16.0);
        let map = FormFactorMap::from_entries(vec![(kind, FormFactor::Constant(6.0))]);
        let structure = Structure {
            pos: vec![
                Vector3::new(0.0, 0.0, 0.0),
                Vector3::new(3.8, 0.0, 0.0),
                Vector3::new(0.0, 4.1, 0.0),
                Vector3::new(1.2, 2.0, 3.0),
            ],
            ids: vec![0, 0, 0, 0],
            box_sides: None,
        };
        (structure, map)
    }

    fn fraser_cfg() -> SolventConfig {
        SolventConfig {
            excluded: ExcludedVolume::Fraser { volume_scale: 1.0 },
            ..Default::default()
        }
    }

    /// Exact Debye `|A − C(c1)|²` with `C` the fully `c1`-scaled Fraser
    /// dummies (the `c1^(2/3)` width baked in) — the ground truth the
    /// first-order model approximates. `excl_scale` multiplies the
    /// (already-built) dummy amplitudes: 1.0 for the exact set, `c1` for
    /// the unit-scale constant-width set.
    fn debye_atoms_minus_excluded(
        structure: &Structure,
        map: &FormFactorMap,
        excluded: &crate::solvent::DummySet,
        excl_scale: f64,
        q: f64,
    ) -> f64 {
        let atom_ff = formfactor::formfactor_by_index(map, q, &structure.ids).unwrap();
        let mut pos = structure.pos.clone();
        pos.extend_from_slice(excluded.positions());
        let mut ff: Vec<f64> = atom_ff.as_slice().to_vec();
        let mut excl_ff = Vec::new();
        excluded.form_factors_at(q, &mut excl_ff).unwrap();
        ff.extend(excl_ff.iter().map(|f| -excl_scale * f));
        self_intensity_at_q(&pos, &ff, q)
    }

    #[test]
    fn first_order_beats_constant_width_and_converges_quadratically() {
        let (structure, map) = structure_and_map();
        let cfg = fraser_cfg();
        let sampler = QSampler::linear(map.clone(), 0.0, 0.6, 16);
        let model = build_model(&sampler, &structure, &cfg).unwrap();

        // Unit base C0 dummies (constant-width reference basis).
        let base = ExcludedVolume::Fraser { volume_scale: 1.0 }
            .dummies(&structure, &map, RHO)
            .unwrap();

        let residuals = |c1: f64| -> (f64, f64, f64) {
            // Exact dummies carry c1 in prefactor AND width.
            let exact = ExcludedVolume::Fraser { volume_scale: c1 }
                .dummies(&structure, &map, RHO)
                .unwrap();
            let first_order = model.intensity(c1, 0.0).total;
            let mut err_fo = 0.0;
            let mut err_cw = 0.0;
            let mut max_fo = 0.0f64;
            for (i, &q) in sampler.q_values.iter().enumerate() {
                let i_exact = debye_atoms_minus_excluded(&structure, &map, &exact, 1.0, q);
                let i_cw = debye_atoms_minus_excluded(&structure, &map, &base, c1, q);
                err_fo += (first_order[i] - i_exact).powi(2);
                err_cw += (i_cw - i_exact).powi(2);
                max_fo = max_fo.max((first_order[i] - i_exact).abs());
            }
            (err_fo, err_cw, max_fo)
        };

        // At a realistic fit offset the first-order model must track the
        // exact width response better than freezing it.
        let (err_fo, err_cw, _) = residuals(1.05);
        assert!(
            err_fo < err_cw,
            "first-order residual {err_fo:.3e} should beat constant-width {err_cw:.3e}"
        );

        // First-order error is O((c1−1)²): halving the offset should
        // quarter the max error (allow a generous band).
        let (_, _, e2) = residuals(1.02);
        let (_, _, e4) = residuals(1.04);
        let ratio = e4 / e2;
        assert!(
            (3.0..=5.0).contains(&ratio),
            "expected ≈4× error growth for 2× offset, got {ratio:.2}"
        );
    }

    #[test]
    fn correction_vanishes_at_q_zero() {
        let (structure, map) = structure_and_map();
        let sampler = QSampler::linear(map.clone(), 0.0, 0.5, 8);
        let model = build_model(&sampler, &structure, &fraser_cfg()).unwrap();
        // q=0 is the first grid point; C1(0)=0 ⇒ every derivative term is 0.
        let t0 = model.terms[0];
        assert_eq!(t0.excluded_deriv, 0.0);
        assert_eq!(t0.excluded_cross, 0.0);
        assert_eq!(t0.atom_excluded_deriv, 0.0);
        assert_eq!(t0.excluded_deriv_hydration, 0.0);
    }

    #[test]
    fn zero_volume_scale_removes_excluded_contribution() {
        let (structure, map) = structure_and_map();
        let sampler = QSampler::linear(map.clone(), 0.0, 0.5, 8);
        let model = build_model(&sampler, &structure, &fraser_cfg()).unwrap();
        // c1=0 ⇒ f0=f1=0 ⇒ intensity is atoms-only (no hydration here).
        let total = model.intensity(0.0, 0.0).total;
        let atoms: Vec<f64> = model.terms.iter().map(|t| t.atoms).collect();
        for (got, want) in total.iter().zip(&atoms) {
            assert!((got - want).abs() < 1e-9);
        }
    }

    #[test]
    fn fit_bound_extremes_stay_finite() {
        let (structure, map) = structure_and_map();
        let sampler = QSampler::linear(map.clone(), 0.0, 0.6, 16);
        let model = build_model(&sampler, &structure, &fraser_cfg()).unwrap();
        // First-order Taylor can overshoot at the c1 bounds (0.5, 2.0);
        // it must not produce NaN/Inf even where it does.
        for c1 in [0.5, 2.0] {
            for &v in &model.intensity(c1, 0.05).total {
                assert!(v.is_finite(), "non-finite intensity at c1={c1}");
            }
        }
    }
}

//! `pripps` computes small-angle X-ray scattering (SAXS) intensity `I(q)`
//! from molecular structures. Three interchangeable calculators are offered
//! via [`IntensityScheme`]: the exact Debye summation, a Direct Fourier
//! transform over a periodic box, and a Multipole (spherical-harmonic)
//! expansion. Each can apply optional solvent corrections — excluded volume
//! and hydration — configured through [`SolventConfig`]. The high-level
//! entry point is [`Pripps`], whose [`Pripps::intensity`] runs a single
//! structure or trajectory-averages over an XTC, while the `*_model`
//! constructors prepare cached, reusable fits.

// Library-only builds (no `cli` feature) have no in-crate caller for the
// intensity calculators — they're only instantiated through `Pripps::intensity`.
// Suppress the per-item dead-code warnings in that configuration; external
// consumers (pyo3 bindings, other library users) reach them via `Pripps`.
#![cfg_attr(not(feature = "cli"), allow(dead_code))]
// Force every public item to carry a doc comment; the build fails
// otherwise. Keeps the library (and its pyo3/Python surface) documented.
#![deny(missing_docs)]

mod atomkind;
#[cfg(feature = "cli")]
pub mod cli;
mod debye;
mod error;
mod explicit;
mod fit;
mod formfactor;
#[cfg(feature = "gpu")]
mod gpu;
mod input;
mod multipole;
mod q_sampler;
mod solvent;
mod structure;
mod trajectory;

use std::{
    num::NonZeroUsize,
    path::{self, Path},
};

use debye::DebyeIntensity;
use explicit::DirectTransform;
use multipole::MultipoleIntensity;
use structure::Structure;
use trajectory::trajectory_average;

pub use debye::DebyeModel;
pub use error::{Error, Result};
pub use explicit::DirectModel;
pub use fit::{ExpData, FitResult, fit_parameters};
pub use multipole::MultipoleModel;
pub use solvent::{ExcludedVolume, Hydration, SolventConfig};

// Crate-internal: the form-factor table and atom-kind types are part of
// no public signature, so they stay off the public API surface.
pub(crate) use atomkind::{AtomKind, FindByName};
pub(crate) use formfactor::{
    FormFactorMap, read_formfactors, read_formfactors_merged, write_formfactors,
};

/// Crate-internal short alias. Not re-exported; public signatures use
/// `[f64; 3]` instead so the API doesn't leak `nalgebra` types.
pub(crate) type Vector3 = nalgebra::Vector3<f64>;

/// A scattering intensity curve: parallel q and I(q) vectors with
/// `q.len() == total.len()` by construction.
///
/// Two parallel `Vec<f64>` (instead of `Vec<(f64, f64)>`) so the
/// future `pyo3` / `numpy` bindings can expose each vector as a
/// zero-copy `ndarray` view. All pripps calculators return this type;
/// callers typically iterate via `.q.iter().zip(&curve.total)`.
///
/// `contributions` is populated when Debye, Multipole, or Direct runs
/// with an active [`SolventConfig`], carrying the per-contribution
/// diagnostic columns. It is `None` for vacuum calculations.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Intensity {
    /// Scattering vector magnitudes (Å⁻¹).
    pub q: Vec<f64>,
    /// Total intensity `I(q)` at each `q`.
    pub total: Vec<f64>,
    /// Per-contribution diagnostic columns; `None` for vacuum calculations.
    pub contributions: Option<Contributions>,
    /// At-concentration single-molecule form factor `P(q)`, present only when
    /// `--num-molecules` (a periodic box of identical molecules) is used. For a
    /// flexible molecule this is the *crowded* form factor, which an experiment
    /// (dividing by a dilute `P`) cannot access. `None` otherwise.
    pub form_factor: Option<Vec<f64>>,
    /// Apparent (effective) structure factor `S_eff(q) = I(q)/(N·P(q))`, present
    /// only with `--num-molecules`. `None` otherwise.
    pub effective_structure_factor: Option<Vec<f64>>,
    /// Raw self term `Σ_m |F_m(q)|² = N·P(q)` from the per-molecule partial sums;
    /// the trajectory-averaging carrier from which `form_factor` /
    /// `effective_structure_factor` are derived. Not part of the public output.
    pub(crate) monomer_self: Option<Vec<f64>>,
}

/// Per-contribution intensity curves emitted by the three-contribution
/// Multipole path, each parallel to [`Intensity::q`].
#[derive(Debug, Clone, PartialEq)]
pub struct Contributions {
    /// Intensity from the atoms (solute) alone.
    pub atoms: Vec<f64>,
    /// Excluded-volume (solvent displacement) contribution.
    pub excluded: Vec<f64>,
    /// Hydration-shell contribution.
    pub hydration: Vec<f64>,
}

/// Computational scheme for scattering intensity calculation, with its
/// parameters. Passed to [`Pripps::intensity`].
#[derive(Debug, Clone, Copy)]
pub enum IntensityScheme {
    /// Debye formula sampled on `nq` points in `[qmin, qmax]` (Å⁻¹).
    /// O(N²·Q) — simple, exact, best for small structures.
    Debye {
        /// Lower q bound (Å⁻¹).
        qmin: f64,
        /// Upper q bound (Å⁻¹).
        qmax: f64,
        /// Number of linearly spaced q points.
        nq: usize,
    },
    /// Direct Fourier transform on `pmax` Miller-index multiples. The
    /// periodic box is taken from the [`Structure`] — from a trajectory
    /// frame when averaging over an XTC, or via the `box_sides` override
    /// parameter of [`Pripps::intensity`] for single-frame calls.
    Direct {
        /// Maximum Miller-index multiple used to build q-vectors.
        pmax: usize,
    },
    /// GPU (wgpu) Direct Fourier transform, vacuum path only. Identical
    /// physics to [`Direct`](Self::Direct) but evaluated on the GPU in `f32`,
    /// for fast trajectory averaging. Requires the `gpu` build feature; with
    /// an active solvent or no available adapter it falls back to the CPU
    /// [`Direct`](Self::Direct) path.
    DirectGpu {
        /// Maximum Miller-index multiple used to build q-vectors.
        pmax: usize,
    },
    /// Spherical-harmonic multipole expansion
    /// (*Acta Cryst. A51*, 1995). O(N·Q·L²), linear in atom count —
    /// preferred for large structures and iterative fitting. `l_max`
    /// is the multipole truncation order; `None` auto-selects from
    /// the structure's extent and `qmax`.
    Multipole {
        /// Lower q bound (Å⁻¹).
        qmin: f64,
        /// Upper q bound (Å⁻¹).
        qmax: f64,
        /// `None` → auto-select at the Nyquist rate `2·l_max + 1`.
        nq: Option<usize>,
        /// Multipole truncation order; `None` auto-selects from the
        /// structure's extent and `qmax`.
        l_max: Option<usize>,
    },
}

/// Internal dispatch interface. Implementations compute I(q) for a single
/// structure; library consumers don't implement or name this trait.
pub(crate) trait IntensityCalculator: Sync {
    fn calc_intensities(&self, structure: &Structure) -> Result<Intensity>;
}

/// Resolve the q-grid density for the multipole kernel. If the
/// caller supplied an explicit `nq`, use it; otherwise pick
/// `2·L_max + 1` (Nyquist-rate for a band-limited spherical-
/// harmonic expansion at truncation L — see the Shannon sampling
/// theorem applied to SAXS by Moore 1980,
/// <https://doi.org/10.1107/S002188988001179X>). Floored at 21 so
/// tiny or very-low-q inputs still sample the theory curve
/// densely enough for smooth cubic-spline interpolation onto
/// experimental q values.
fn resolve_nq(
    requested: Option<usize>,
    positions: &[Vector3],
    qmax: f64,
    l_max_override: Option<usize>,
) -> usize {
    if let Some(n) = requested {
        return n;
    }
    let l_max = l_max_override.unwrap_or_else(|| {
        let center: Vector3 = if positions.is_empty() {
            Vector3::zeros()
        } else {
            positions.iter().sum::<Vector3>() / positions.len() as f64
        };
        let r_max = positions
            .iter()
            .map(|p| (p - center).norm())
            .fold(0.0_f64, f64::max);
        ((qmax * r_max).ceil() as usize) + 2
    });
    (2 * l_max + 1).max(21)
}

/// High-level interface to the Pripps library
pub struct Pripps {
    formfactors: FormFactorMap,
}

/// Options for trajectory-averaged intensity calculations over an XTC file.
pub struct TrajectoryOptions<'a> {
    /// Path to the XTC trajectory file.
    pub xtcfile: &'a path::Path,
    /// Process every `stride`-th frame.
    pub stride: NonZeroUsize,
    /// Optional path to a per-frame weights file for the average.
    pub weights: Option<&'a path::Path>,
}

impl Pripps {
    /// Construct from a single YAML form-factor definition file.
    pub fn from_yaml(path: &Path) -> Result<Self> {
        Self::from_formfactors(read_formfactors(path)?)
    }

    /// Construct from one or more YAML form-factor definition files,
    /// merged in order. When a species is defined in more than one file
    /// the later file wins and a warning is logged (see
    /// [`read_formfactors_merged`]). Symbolic specs (`!Alias` / `!United`
    /// / `!Linear`) may reference definitions from any of the files.
    pub fn from_yaml_files(paths: &[std::path::PathBuf]) -> Result<Self> {
        Self::from_formfactors(read_formfactors_merged(paths)?)
    }

    fn from_formfactors(formfactors: FormFactorMap) -> Result<Self> {
        if std::env::var("FF_TABLE").is_ok() {
            write_formfactors(&formfactors, Path::new("tabulated_formfactors.yaml"))?;
        }

        Ok(Self { formfactors })
    }

    fn atomkinds(&self) -> Vec<AtomKind> {
        self.formfactors.kinds().cloned().collect()
    }

    /// Prepare a stateful Multipole fit. Caches the per-q multipole
    /// coefficient triangles (atoms, excluded volume, hydration) so
    /// repeated calls to [`MultipoleModel::intensity`] with different
    /// `(volume_scale, contrast_density)` only pay the cheap
    /// O(Q·L²) combine, not the full O(N·Q·L²) basis + coefficient
    /// construction.
    pub fn multipole_model(
        &self,
        structure_file: &path::Path,
        qmin: f64,
        qmax: f64,
        nq: Option<usize>,
        l_max: Option<usize>,
        solvent: &SolventConfig,
    ) -> Result<MultipoleModel> {
        let mut structure = Structure::from_file(structure_file, &self.atomkinds())?;
        structure.center();
        let nq = resolve_nq(nq, &structure.pos, qmax, l_max);
        let sampler = q_sampler::QSampler::linear(self.formfactors.clone(), qmin, qmax, nq);
        let l_max_per_q = multipole::resolve_l_max_per_q(l_max, &structure.pos, &sampler.q_values);
        let l_max = l_max_per_q.iter().copied().max().unwrap_or(0);
        let l_min = l_max_per_q.iter().copied().min().unwrap_or(0);
        log::info!(
            "Multipole fit preparation: l_max ∈ [{l_min}, {l_max}] over {} atoms, {} q-points",
            structure.pos.len(),
            sampler.q_values.len(),
        );
        let atoms_basis = multipole::SphericalBasis::new(&structure.pos, l_max);
        multipole::build_model(&sampler, &structure, &atoms_basis, solvent, &l_max_per_q)
    }

    /// Prepare a stateful Debye fit. Caches the exact per-q Debye
    /// self and cross terms for atoms, excluded volume, and hydration
    /// so repeated calls to [`DebyeModel::intensity`] only recombine
    /// the cached terms with different `(volume_scale, contrast_density)`.
    pub fn debye_model(
        &self,
        structure_file: &path::Path,
        qmin: f64,
        qmax: f64,
        nq: usize,
        solvent: &SolventConfig,
    ) -> Result<DebyeModel> {
        let structure = Structure::from_file(structure_file, &self.atomkinds())?;
        let sampler = q_sampler::QSampler::linear(self.formfactors.clone(), qmin, qmax, nq);
        log::info!(
            "Debye fit preparation over {} atoms, {} q-points",
            structure.pos.len(),
            sampler.q_values.len(),
        );
        debye::build_model(&sampler, &structure, solvent)
    }

    /// Prepare a stateful direct/explicit fit. Caches the complex
    /// amplitudes for atoms, excluded volume, and hydration at every
    /// sampled q-vector so repeated calls to [`DirectModel::intensity`]
    /// only recombine the cached amplitudes with different
    /// `(volume_scale, contrast_density)`.
    pub fn direct_model(
        &self,
        structure_file: &path::Path,
        pmax: usize,
        box_sides: [f64; 3],
        solvent: &SolventConfig,
    ) -> Result<DirectModel> {
        let structure = Structure::from_file(structure_file, &self.atomkinds())?;
        let directions = explicit::directions(pmax);
        let box_sides = Vector3::from(box_sides);
        log::info!(
            "Direct fit preparation over {} atoms, {} q-vectors",
            structure.pos.len(),
            directions.len(),
        );
        explicit::build_model(
            &self.formfactors,
            &directions,
            &structure,
            &box_sides,
            solvent,
        )
    }

    /// Compute I(q) for a structure under the given [`IntensityScheme`].
    ///
    /// When `xtcfile` is `Some`, the structure is used as a template (for
    /// atom identities and count); each frame of the XTC is iterated,
    /// substituted into the template, and the resulting I(q) curves are
    /// averaged (weighted by the optional `weights` file if provided).
    ///
    /// `box_sides` provides an explicit periodic box for single-frame
    /// calls; it must be `None` when a trajectory is supplied (the per-
    /// frame box from the XTC takes precedence).
    pub fn intensity(
        &self,
        scheme: IntensityScheme,
        structure_file: &path::Path,
        xtc_opts: Option<TrajectoryOptions>,
        box_sides: Option<[f64; 3]>,
        solvent: Option<&SolventConfig>,
        num_molecules: Option<usize>,
    ) -> Result<Intensity> {
        let mut structure = Structure::from_file(structure_file, &self.atomkinds())?;
        if let Some(box_sides) = box_sides {
            if xtc_opts.is_some() {
                return Err(Error::usage(
                    "manual box length cannot be used with trajectories",
                ));
            }
            structure.box_sides = Some(Vector3::from(box_sides));
        }

        let active_solvent = solvent.filter(|cfg| cfg.is_active());

        // `--num-molecules N` (effective structure factor) needs the explicit
        // periodic-box method, vacuum, and an integer N. Validate up front so a
        // bad request fails immediately, before any (trajectory) work.
        if let Some(n) = num_molecules {
            if !matches!(
                scheme,
                IntensityScheme::Direct { .. } | IntensityScheme::DirectGpu { .. }
            ) {
                return Err(Error::usage(
                    "--num-molecules requires the `direct` scheme (the periodic-box method)",
                ));
            }
            if n == 0 {
                return Err(Error::usage("--num-molecules must be at least 1"));
            }
            if active_solvent.is_some() {
                return Err(Error::usage(
                    "--num-molecules supports vacuum runs only (no excluded-volume/hydration model)",
                ));
            }
            if !structure.pos.len().is_multiple_of(n) {
                return Err(Error::validation(format!(
                    "box atom count ({}) is not a multiple of --num-molecules ({n})",
                    structure.pos.len()
                )));
            }
            // A divisible count is not enough: the N blocks must actually be
            // identical contiguous molecules. Require every M-atom block to share
            // the first block's atom-kind sequence — this catches a count that
            // splits molecules or doesn't match the box composition (e.g. asking
            // for 7 molecules from a single 1001-atom protein).
            let m = structure.pos.len() / n;
            if m > 0
                && !structure
                    .ids
                    .chunks(m)
                    .all(|block| block == &structure.ids[..m])
            {
                return Err(Error::validation(format!(
                    "--num-molecules {n} does not partition the box into {n} identical \
                     contiguous molecules: the atom composition differs between blocks"
                )));
            }
            log::info!("Computing structure factor for {n} molecules ({m} atoms each)");
        }

        let calc: Box<dyn IntensityCalculator> = match scheme {
            IntensityScheme::Debye { qmin, qmax, nq } => {
                log::info!("Using Debye formula to calculate 𝐼(𝑞) for {qmin} ≤ 𝑞 ≤ {qmax} Å⁻¹");
                Box::new(DebyeIntensity::new_with_solvent(
                    self.formfactors.clone(),
                    (qmin, qmax),
                    nq,
                    active_solvent.cloned(),
                )?)
            }
            IntensityScheme::Direct { pmax } => {
                log::info!("Using direct Fourier transform with p_max = {pmax}");
                let transform =
                    DirectTransform::new(self.formfactors.clone(), pmax, active_solvent.cloned());
                Box::new(match num_molecules {
                    Some(n) => transform.with_num_molecules(n),
                    None => transform,
                })
            }
            IntensityScheme::DirectGpu { pmax } => {
                #[cfg(feature = "gpu")]
                {
                    // The structure-factor self term needs per-molecule partial
                    // sums, which only the CPU transform computes.
                    if let Some(n) = num_molecules {
                        log::warn!(
                            "--num-molecules is computed on the CPU direct transform (the GPU emits only |Σf|²)"
                        );
                        Box::new(
                            DirectTransform::new(self.formfactors.clone(), pmax, None)
                                .with_num_molecules(n),
                        )
                    }
                    // The GPU path covers the vacuum case only; with an active
                    // solvent or no usable adapter, fall back to the CPU Direct
                    // transform so results stay correct.
                    else if active_solvent.is_some() {
                        log::warn!(
                            "solvent corrections are not supported on the GPU; using CPU direct transform"
                        );
                        Box::new(DirectTransform::new(
                            self.formfactors.clone(),
                            pmax,
                            active_solvent.cloned(),
                        ))
                    } else {
                        match gpu::GpuDirectTransform::new(self.formfactors.clone(), pmax) {
                            Ok(calc) => {
                                log::info!(
                                    "Using GPU direct Fourier transform with p_max = {pmax}"
                                );
                                Box::new(calc) as Box<dyn IntensityCalculator>
                            }
                            Err(e) => {
                                log::warn!("GPU unavailable ({e}); using CPU direct transform");
                                Box::new(DirectTransform::new(self.formfactors.clone(), pmax, None))
                            }
                        }
                    }
                }
                #[cfg(not(feature = "gpu"))]
                {
                    let _ = pmax;
                    return Err(Error::usage(
                        "GPU direct transform requested but pripps was built without the `gpu` feature",
                    ));
                }
            }
            IntensityScheme::Multipole {
                qmin,
                qmax,
                nq,
                l_max,
            } => {
                let nq = resolve_nq(nq, &structure.pos, qmax, l_max);
                log::info!(
                    "Using Multipole expansion for 𝐼(𝑞) over {qmin} ≤ 𝑞 ≤ {qmax} Å⁻¹, {nq} q-points"
                );
                Box::new(MultipoleIntensity::new(
                    self.formfactors.clone(),
                    (qmin, qmax),
                    nq,
                    l_max,
                    active_solvent.cloned(),
                )?)
            }
        };

        let mut result = if let Some(xtc_opts) = xtc_opts {
            trajectory_average(calc.as_ref(), &structure, xtc_opts)?
        } else {
            calc.calc_intensities(&structure)?
        };

        // Derive the effective structure factor from the per-molecule self term
        // (`Σ_m|F_m|² = N·P`): S_eff = I_box/(N·P) = total/self; P = self/N. The
        // self term is the box self-scattering, so this is exact (no approximation).
        if let (Some(n), Some(self_term)) = (num_molecules, result.monomer_self.take()) {
            let n_f = n as f64;
            result.form_factor = Some(self_term.iter().map(|&s| s / n_f).collect());
            result.effective_structure_factor = Some(
                result
                    .total
                    .iter()
                    .zip(&self_term)
                    .map(|(&i, &s)| if s > 0.0 { i / s } else { f64::NAN })
                    .collect(),
            );
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debye_solvent_matches_cached_model_for_matching_knobs() {
        let (pripps, xyz) = yaml_and_xyz();
        let solvent = SolventConfig {
            excluded: ExcludedVolume::Fraser { volume_scale: 1.0 },
            hydration: Hydration::Sasa {
                probe: 1.8,
                contrast_density: 0.03,
            },
            ..Default::default()
        };
        let one_shot = pripps
            .intensity(
                IntensityScheme::Debye {
                    qmin: 0.0,
                    qmax: 0.3,
                    nq: 20,
                },
                &xyz,
                None,
                None,
                Some(&solvent),
                None,
            )
            .unwrap();

        let fit = pripps.debye_model(&xyz, 0.0, 0.3, 20, &solvent).unwrap();
        let from_fit = fit.intensity(1.0, 0.03);

        assert_eq!(one_shot.total, from_fit.total);
        assert_eq!(one_shot.contributions, from_fit.contributions);
    }

    #[test]
    fn direct_solvent_matches_cached_model_for_matching_knobs() {
        let (pripps, xyz) = yaml_and_xyz();
        let solvent = SolventConfig {
            excluded: ExcludedVolume::Fraser { volume_scale: 1.0 },
            hydration: Hydration::Sasa {
                probe: 1.8,
                contrast_density: 0.03,
            },
            ..Default::default()
        };
        let box_sides = [200.0; 3];
        let one_shot = pripps
            .intensity(
                IntensityScheme::Direct { pmax: 2 },
                &xyz,
                None,
                Some(box_sides),
                Some(&solvent),
                None,
            )
            .unwrap();

        let fit = pripps.direct_model(&xyz, 2, box_sides, &solvent).unwrap();
        let from_fit = fit.intensity(1.0, 0.03);

        assert_eq!(one_shot.total, from_fit.total);
        assert_eq!(one_shot.contributions, from_fit.contributions);
    }

    #[test]
    fn fraser_volume_scale_is_not_applied_twice() {
        let (pripps, xyz) = yaml_and_xyz();
        let scale = 1.7;
        let unit_solvent = SolventConfig {
            excluded: ExcludedVolume::Fraser { volume_scale: 1.0 },
            ..Default::default()
        };
        let scaled_solvent = SolventConfig {
            excluded: ExcludedVolume::Fraser {
                volume_scale: scale,
            },
            ..Default::default()
        };

        let debye_one_shot = pripps
            .intensity(
                IntensityScheme::Debye {
                    qmin: 0.0,
                    qmax: 0.2,
                    nq: 8,
                },
                &xyz,
                None,
                None,
                Some(&scaled_solvent),
                None,
            )
            .unwrap();
        let debye_model = pripps
            .debye_model(&xyz, 0.0, 0.2, 8, &unit_solvent)
            .unwrap();
        assert_eq!(
            debye_one_shot.total,
            debye_model.intensity(scale, 0.0).total
        );

        let direct_box = [200.0; 3];
        let direct_one_shot = pripps
            .intensity(
                IntensityScheme::Direct { pmax: 2 },
                &xyz,
                None,
                Some(direct_box),
                Some(&scaled_solvent),
                None,
            )
            .unwrap();
        let direct_model = pripps
            .direct_model(&xyz, 2, direct_box, &unit_solvent)
            .unwrap();
        assert_eq!(
            direct_one_shot.total,
            direct_model.intensity(scale, 0.0).total
        );

        let multipole_one_shot = pripps
            .intensity(
                IntensityScheme::Multipole {
                    qmin: 0.0,
                    qmax: 0.2,
                    nq: Some(8),
                    l_max: None,
                },
                &xyz,
                None,
                None,
                Some(&scaled_solvent),
                None,
            )
            .unwrap();
        let multipole_model = pripps
            .multipole_model(&xyz, 0.0, 0.2, Some(8), None, &unit_solvent)
            .unwrap();
        assert_eq!(
            multipole_one_shot.total,
            multipole_model.intensity(scale, 0.0).total
        );
    }

    fn yaml_and_xyz() -> (Pripps, std::path::PathBuf) {
        let yaml = std::path::Path::new("assets/poly_amino_acid.yaml");
        let xyz = std::path::PathBuf::from("tests/lysozyme/4lzt.xyz");
        (Pripps::from_yaml(yaml).unwrap(), xyz)
    }

    #[test]
    fn fit_matches_one_shot_for_matching_knobs() {
        let (pripps, xyz) = yaml_and_xyz();
        let solvent = SolventConfig {
            excluded: ExcludedVolume::Fraser { volume_scale: 1.0 },
            hydration: Hydration::Sasa {
                probe: 1.8,
                contrast_density: 0.03,
            },
            ..Default::default()
        };
        let one_shot = pripps
            .intensity(
                IntensityScheme::Multipole {
                    qmin: 0.0,
                    qmax: 0.3,
                    nq: Some(20),
                    l_max: None,
                },
                &xyz,
                None,
                None,
                Some(&solvent),
                None,
            )
            .unwrap();

        let fit = pripps
            .multipole_model(&xyz, 0.0, 0.3, Some(20), None, &solvent)
            .unwrap();
        let from_fit = fit.intensity(1.0, 0.03);

        assert_eq!(one_shot.total, from_fit.total);
        assert_eq!(one_shot.contributions, from_fit.contributions);
    }

    #[test]
    fn fit_zero_knobs_matches_vacuum() {
        let (pripps, xyz) = yaml_and_xyz();
        let solvent = SolventConfig {
            excluded: ExcludedVolume::Fraser { volume_scale: 1.0 },
            hydration: Hydration::Sasa {
                probe: 1.8,
                contrast_density: 0.03,
            },
            ..Default::default()
        };
        let fit = pripps
            .multipole_model(&xyz, 0.0, 0.3, Some(20), None, &solvent)
            .unwrap();
        let zero = fit.intensity(0.0, 0.0);
        let contributions = zero.contributions.as_ref().unwrap();
        assert_eq!(zero.total, contributions.atoms);
    }

    #[test]
    fn fit_intensity_is_pure() {
        let (pripps, xyz) = yaml_and_xyz();
        let solvent = SolventConfig {
            excluded: ExcludedVolume::Fraser { volume_scale: 1.0 },
            ..Default::default()
        };
        let fit = pripps
            .multipole_model(&xyz, 0.0, 0.3, Some(10), None, &solvent)
            .unwrap();
        let a = fit.intensity(1.0, 0.0);
        let b = fit.intensity(1.0, 0.0);
        assert_eq!(a.total, b.total);
    }

    #[test]
    fn fit_intensity_varies_with_knobs() {
        let (pripps, xyz) = yaml_and_xyz();
        let solvent = SolventConfig {
            excluded: ExcludedVolume::Fraser { volume_scale: 1.0 },
            hydration: Hydration::Sasa {
                probe: 1.8,
                contrast_density: 0.0,
            },
            ..Default::default()
        };
        let fit = pripps
            .multipole_model(&xyz, 0.0, 0.3, Some(10), None, &solvent)
            .unwrap();
        let a = fit.intensity(1.0, 0.0);
        let b = fit.intensity(1.0, 0.1);
        assert_ne!(a.total, b.total);
    }

    fn load_lyzexp() -> fit::ExpData {
        fit::ExpData::from_file(std::path::Path::new("tests/lysozyme/lyzexp.dat")).unwrap()
    }

    fn pepsi_intensity_scale(path: &std::path::Path) -> f64 {
        let contents = std::fs::read_to_string(path).unwrap();
        contents
            .lines()
            .find(|line| line.contains("Intensity scaling"))
            .and_then(|line| line.split_whitespace().last())
            .and_then(|value| value.parse().ok())
            .unwrap_or_else(|| panic!("missing intensity scaling in {}", path.display()))
    }

    fn pepsi_unscaled_iat_profile(path: &std::path::Path, scale: f64) -> Vec<(f64, f64)> {
        let contents = std::fs::read_to_string(path).unwrap();
        contents
            .lines()
            .filter(|line| {
                let trimmed = line.trim();
                !trimmed.is_empty() && !trimmed.starts_with('#')
            })
            .map(|line| {
                let fields: Vec<_> = line.split_whitespace().collect();
                assert!(
                    fields.len() >= 3,
                    "expected at least three PEPSI output columns in {line:?}"
                );
                let q = fields[0].parse::<f64>().unwrap();
                let iat = fields[2].parse::<f64>().unwrap() / scale;
                (q, iat)
            })
            .collect()
    }

    #[test]
    fn amino_acid_debye_matches_scaled_pepsi_iat_profile() {
        let scale = pepsi_intensity_scale(std::path::Path::new("tests/lysozyme/pepsi.log"));
        let reference =
            pepsi_unscaled_iat_profile(std::path::Path::new("tests/lysozyme/pepsi.out"), scale);
        let (pripps, xyz) = yaml_and_xyz();
        let intensity = pripps
            .intensity(
                IntensityScheme::Debye {
                    qmin: reference.first().unwrap().0,
                    qmax: reference.last().unwrap().0,
                    nq: reference.len(),
                },
                &xyz,
                None,
                None,
                None,
                None,
            )
            .unwrap();

        assert_eq!(intensity.q.len(), reference.len());

        let mut max_rel = 0.0_f64;
        let mut max_q = 0.0_f64;
        for ((&q, &calc), &(q_ref, expected)) in intensity
            .q
            .iter()
            .zip(&intensity.total)
            .zip(reference.iter())
        {
            assert!(
                (q - q_ref).abs() < 1.0e-12,
                "q grid mismatch: calculated {q}, reference {q_ref}"
            );
            let rel = ((calc - expected) / expected).abs();
            if rel > max_rel {
                max_rel = rel;
                max_q = q_ref;
            }
        }

        assert!(
            max_rel < 0.04,
            "PEPSI Iat profile differs by {max_rel:.4} at q={max_q:.3}"
        );
    }

    #[test]
    fn foxs_chi2_against_experiment() {
        // FoXS gets χ²/N ≈ 0.29 on 4lzt.pdb vs lyzexp.dat with
        // excluded volume only (no hydration, c₁=1). With the
        // FoXS-compatible Cromer-Mann form factors and the 16π
        // Fraser formula, pripps should match.
        let pripps =
            Pripps::from_yaml(std::path::Path::new("assets/foxs_formfactors.yaml")).unwrap();
        let pdb = std::path::PathBuf::from("tests/lysozyme/4lzt.pdb");
        let solvent = SolventConfig {
            excluded: ExcludedVolume::FraserFoxs { volume_scale: 1.0 },
            ..Default::default()
        };
        let intensity = pripps
            .intensity(
                IntensityScheme::Multipole {
                    qmin: 0.0,
                    qmax: 0.5,
                    nq: Some(500),
                    l_max: None,
                },
                &pdb,
                None,
                None,
                Some(&solvent),
                None,
            )
            .unwrap();

        let exp = load_lyzexp();
        let chi2 = fit::chi2_per_point(&intensity, &exp);
        // FoXS target: 0.29. With Cromer-Mann form factors and
        // the FoXS Fraser formula we should be within 0.5.
        assert!(
            chi2 < 0.5,
            "χ²/N = {chi2:.4} is too high (FoXS target: 0.29)"
        );
    }

    #[test]
    fn foxs_profile_matches_reference_file() {
        let reference =
            fit::ExpData::from_file(std::path::Path::new("tests/lysozyme/6lyz_foxs.dat")).unwrap();
        let pripps =
            Pripps::from_yaml(std::path::Path::new("assets/foxs_formfactors.yaml")).unwrap();
        let pdb = std::path::PathBuf::from("tests/lysozyme/6lyz.pdb");
        let solvent = SolventConfig {
            excluded: ExcludedVolume::FraserFoxs { volume_scale: 1.0 },
            ..Default::default()
        };
        let intensity = pripps
            .intensity(
                IntensityScheme::Multipole {
                    qmin: *reference.q.first().unwrap(),
                    qmax: *reference.q.last().unwrap(),
                    nq: Some(reference.q.len()),
                    l_max: None,
                },
                &pdb,
                None,
                None,
                Some(&solvent),
                None,
            )
            .unwrap();

        assert_eq!(intensity.q.len(), reference.q.len());

        let mut max_rel = 0.0_f64;
        let mut max_q = 0.0_f64;
        for ((&q, &calc), (&q_ref, &expected)) in intensity
            .q
            .iter()
            .zip(&intensity.total)
            .zip(reference.q.iter().zip(&reference.intensity))
        {
            assert!(
                (q - q_ref).abs() < 1.0e-12,
                "q grid mismatch: calculated {q}, reference {q_ref}"
            );
            let rel = ((calc - expected) / expected).abs();
            if rel > max_rel {
                max_rel = rel;
                max_q = q_ref;
            }
        }

        assert!(
            max_rel < 0.05,
            "FoXS reference profile differs by {max_rel:.4} at q={max_q:.3}"
        );
    }

    #[test]
    fn pepsi_pipeline_reasonable_chi2() {
        // PepsiSAXS form factors + PepsiSAXS Fraser (4π) should
        // produce a physically reasonable profile. Not as good as
        // the FoXS pipeline (different coefficients, different
        // denominator) but the shape should be plausible.
        let pripps = Pripps::from_yaml(std::path::Path::new("assets/pepsi_atoms.yaml")).unwrap();
        let pdb = std::path::PathBuf::from("tests/lysozyme/4lzt.pdb");
        let solvent = SolventConfig {
            excluded: ExcludedVolume::Fraser { volume_scale: 1.0 },
            ..Default::default()
        };
        let intensity = pripps
            .intensity(
                IntensityScheme::Multipole {
                    qmin: 0.0,
                    qmax: 0.5,
                    nq: Some(500),
                    l_max: None,
                },
                &pdb,
                None,
                None,
                Some(&solvent),
                None,
            )
            .unwrap();

        let exp = load_lyzexp();
        let chi2 = fit::chi2_per_point(&intensity, &exp);
        // PepsiSAXS pipeline with its own form factors and 4π
        // formula — should produce a reasonable profile (χ² < 10).
        assert!(
            chi2 < 10.0,
            "PepsiSAXS pipeline χ²/N = {chi2:.2} is too high"
        );
    }

    /// Write a synthetic two-frame XTC whose frames differ (the second is
    /// the first scaled by 1.2, so its Debye profile is distinct). The
    /// atom count matches `4lzt.xyz` (129), which is all `set_positions`
    /// checks — the coordinates here override the template's per frame.
    fn write_two_frame_xtc(path: &Path) {
        const NATOMS: usize = 129;
        let make_frame = |scale: f32, step: u32| {
            let mut positions = Vec::with_capacity(NATOMS * 3);
            for i in 0..NATOMS {
                positions.push(scale * 0.10 * i as f32); // x (nm)
                positions.push(scale * 0.15 * (i % 7) as f32); // y
                positions.push(scale * 0.20 * (i % 5) as f32); // z
            }
            let mut boxvec = [0.0f32; 9];
            boxvec[0] = 15.0;
            boxvec[4] = 15.0;
            boxvec[8] = 15.0;
            molly::Frame {
                step,
                time: step as f32,
                boxvec,
                precision: 1000.0,
                positions,
            }
        };
        let mut writer = molly::XTCWriter::create(path).unwrap();
        writer.write_frame(&make_frame(1.0, 0)).unwrap();
        writer.write_frame(&make_frame(1.2, 1)).unwrap();
    }

    /// Exercise XTC striding together with per-frame weights.
    ///
    /// Over a two-frame trajectory the Debye average (box-free, so no
    /// per-frame box is needed) is linear in the per-frame curves, which
    /// lets us pin down three properties at once:
    ///
    /// 1. a weight file selecting a single frame yields that frame alone,
    /// 2. uniform weighting equals the mean of the two single-frame curves,
    /// 3. `stride = 2` keeps only frame 0 and the weight renormalisation
    ///    rescales it back to a full average (not frame 0 × 1/N_frames).
    #[test]
    fn trajectory_stride_and_weights() {
        // 4lzt.xyz is a residue-bead structure, so it needs the CG
        // single-bead form factors from poly_amino_acid.yaml.
        let pripps = Pripps::from_yaml(Path::new("assets/poly_amino_acid.yaml")).unwrap();
        let template = std::path::PathBuf::from("tests/lysozyme/4lzt.xyz");
        let xtc = std::env::temp_dir().join("pripps_stride_test.xtc");
        write_two_frame_xtc(&xtc);

        let nz = |n| NonZeroUsize::new(n).unwrap();
        let write_weights = |name: &str, w: [f64; 2]| {
            let p = std::env::temp_dir().join(name);
            std::fs::write(&p, format!("{}\n{}\n", w[0], w[1])).unwrap();
            p
        };
        let run = |weights: Option<&Path>, stride: usize| {
            let opts = TrajectoryOptions {
                xtcfile: &xtc,
                stride: nz(stride),
                weights,
            };
            pripps
                .intensity(
                    IntensityScheme::Debye {
                        qmin: 0.05,
                        qmax: 0.3,
                        nq: 8,
                    },
                    &template,
                    Some(opts),
                    None,
                    None,
                    None,
                )
                .unwrap()
                .total
        };

        let w_first = write_weights("pripps_w_first.txt", [1.0, 0.0]);
        let w_second = write_weights("pripps_w_second.txt", [0.0, 1.0]);

        let i_f0 = run(Some(&w_first), 1); // all weight on frame 0
        let i_f1 = run(Some(&w_second), 1); // all weight on frame 1
        let i_avg = run(None, 1); // uniform over both frames
        let i_stride2 = run(None, 2); // stride skips frame 1 → frame 0 only

        let close = |a: &[f64], b: &[f64]| {
            a.len() == b.len()
                && a.iter()
                    .zip(b)
                    .all(|(x, y)| (x - y).abs() <= 1e-9 * x.abs().max(1.0))
        };

        // The two frames differ, so single-frame curves must not coincide.
        assert!(
            !close(&i_f0, &i_f1),
            "frames 0 and 1 should give different I(q)"
        );

        // Uniform weighting is the mean of the two single-frame curves.
        let mean: Vec<f64> = i_f0.iter().zip(&i_f1).map(|(a, b)| 0.5 * (a + b)).collect();
        assert!(
            close(&i_avg, &mean),
            "uniform weights must equal the per-frame mean"
        );

        // stride = 2 keeps only frame 0; renormalisation by the consumed
        // weight must recover frame 0 exactly, not frame 0 scaled by 1/N.
        assert!(
            close(&i_stride2, &i_f0),
            "stride=2 must select and renormalise to frame 0"
        );
    }

    /// `--num-molecules` structure factor: a single molecule gives S_eff ≡ 1 and
    /// `p ≡ total`; a box of two identical copies has `p` (the per-molecule form
    /// factor) equal to the single-molecule intensity; and the guards fire.
    #[test]
    fn num_molecules_structure_factor_and_guards() {
        use approx::assert_relative_eq;
        let dir = std::env::temp_dir();
        let yaml = dir.join("pripps_nmol_ff.yaml");
        std::fs::write(
            &yaml,
            "X: { formfactor: !Constant 1.0 }\nY: { formfactor: !Constant 1.0 }\n",
        )
        .unwrap();
        let one = dir.join("pripps_nmol_one.xyz");
        std::fs::write(&one, "2\nd\nX 0 0 -3\nX 0 0 3\n").unwrap();
        // Two copies of the same dumbbell, well separated.
        let two = dir.join("pripps_nmol_two.xyz");
        std::fs::write(&two, "4\nb\nX 0 0 -3\nX 0 0 3\nX 50 0 -3\nX 50 0 3\n").unwrap();
        // Divisible by 2 but not two identical molecules: blocks [X,Y] vs [Y,X].
        let mismatch = dir.join("pripps_nmol_mismatch.xyz");
        std::fs::write(&mismatch, "4\nm\nX 0 0 0\nY 1 0 0\nY 50 0 0\nX 51 0 0\n").unwrap();
        let pripps = Pripps::from_yaml(&yaml).unwrap();

        let run = |file: &Path, n: usize| {
            pripps
                .intensity(
                    IntensityScheme::Direct { pmax: 30 },
                    file,
                    None,
                    Some([200.0; 3]),
                    None,
                    Some(n),
                )
                .unwrap()
        };

        // N = 1: I_box == self ⇒ S_eff ≡ 1, p ≡ total.
        let single = run(&one, 1);
        let s_eff = single.effective_structure_factor.as_ref().unwrap();
        let p = single.form_factor.as_ref().unwrap();
        for ((&se, &pi), &ti) in s_eff.iter().zip(p).zip(&single.total) {
            assert_relative_eq!(se, 1.0, epsilon = 1e-9);
            assert_relative_eq!(pi, ti, epsilon = 1e-9);
        }

        // N = 2: the per-molecule form factor equals the single-molecule
        // intensity (same q-grid, identical molecule), and S_eff is present.
        let box2 = run(&two, 2);
        assert_eq!(box2.form_factor.as_ref().unwrap().len(), single.total.len());
        for (&p2, &t1) in box2.form_factor.as_ref().unwrap().iter().zip(&single.total) {
            assert_relative_eq!(p2, t1, epsilon = 1e-6);
        }

        // Guards.
        let err = |scheme, file: &Path, box_sides, solvent, n| {
            pripps
                .intensity(scheme, file, None, box_sides, solvent, Some(n))
                .is_err()
        };
        // box atom count not a multiple of N.
        assert!(err(
            IntensityScheme::Direct { pmax: 2 },
            &two,
            Some([200.0; 3]),
            None,
            3
        ));
        // non-direct scheme.
        assert!(err(
            IntensityScheme::Debye {
                qmin: 0.0,
                qmax: 0.3,
                nq: 8
            },
            &two,
            None,
            None,
            2
        ));
        // active solvent.
        let solvent = SolventConfig {
            excluded: ExcludedVolume::Fraser { volume_scale: 1.0 },
            ..Default::default()
        };
        assert!(err(
            IntensityScheme::Direct { pmax: 2 },
            &two,
            Some([200.0; 3]),
            Some(&solvent),
            2
        ));
        // divisible count, but the blocks are not identical molecules.
        assert!(err(
            IntensityScheme::Direct { pmax: 2 },
            &mismatch,
            Some([200.0; 3]),
            None,
            2
        ));
    }
}

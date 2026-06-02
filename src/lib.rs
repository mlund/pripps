// Library-only builds (no `cli` feature) have no in-crate caller for the
// intensity calculators — they're only instantiated through `Pripps::intensity`.
// Suppress the per-item dead-code warnings in that configuration; external
// consumers (pyo3 bindings, other library users) reach them via `Pripps`.
#![cfg_attr(not(feature = "cli"), allow(dead_code))]

mod atomkind;
#[cfg(feature = "cli")]
pub mod cli;
mod debye;
mod error;
mod explicit;
pub mod fit;
mod formfactor;
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

pub use atomkind::{AtomKind, FindByName};
pub use error::{Error, Result};
pub use formfactor::{FormFactor, FormFactorMap, read_formfactors, write_formfactors};
pub use multipole::MultipoleModel;
pub use solvent::{ExcludedVolume, Hydration, SolventConfig};

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
/// `contributions` is populated when Multipole runs with an active
/// [`SolventConfig`], carrying the per-contribution diagnostic
/// columns. It is `None` for vacuum Multipole, Debye, and Direct.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Intensity {
    pub q: Vec<f64>,
    pub total: Vec<f64>,
    pub contributions: Option<Contributions>,
}

/// Per-contribution intensity curves emitted by the three-contribution
/// Multipole path, each parallel to [`Intensity::q`].
#[derive(Debug, Clone, PartialEq)]
pub struct Contributions {
    pub atoms: Vec<f64>,
    pub excluded: Vec<f64>,
    pub hydration: Vec<f64>,
}

/// Computational scheme for scattering intensity calculation, with its
/// parameters. Passed to [`Pripps::intensity`].
#[derive(Debug, Clone, Copy)]
pub enum IntensityScheme {
    /// Debye formula sampled on `nq` points in `[qmin, qmax]` (Å⁻¹).
    /// O(N²·Q) — simple, exact, best for small structures.
    Debye { qmin: f64, qmax: f64, nq: usize },
    /// Direct Fourier transform on `pmax` Miller-index multiples. The
    /// periodic box is taken from the [`Structure`] — from a trajectory
    /// frame when averaging over an XTC, or via the `box_sides` override
    /// parameter of [`Pripps::intensity`] for single-frame calls.
    Direct { pmax: usize },
    /// Spherical-harmonic multipole expansion
    /// (*Acta Cryst. A51*, 1995). O(N·Q·L²), linear in atom count —
    /// preferred for large structures and iterative fitting. `l_max`
    /// is the multipole truncation order; `None` auto-selects from
    /// the structure's extent and `qmax`.
    Multipole {
        qmin: f64,
        qmax: f64,
        /// `None` → auto-select at the Nyquist rate `2·l_max + 1`.
        nq: Option<usize>,
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

pub struct TrajectoryOptions<'a> {
    pub xtcfile: &'a path::Path,
    pub stride: NonZeroUsize,
    pub weights: Option<&'a path::Path>,
}

impl Pripps {
    pub fn formfactors(&self) -> &FormFactorMap {
        &self.formfactors
    }

    /// Construct from a YAML form-factor definition file.
    pub fn from_yaml(path: &Path) -> Result<Self> {
        let formfactors = read_formfactors(path)?;

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
        if active_solvent.is_some() && matches!(scheme, IntensityScheme::Debye { .. }) {
            return Err(Error::usage(
                "solvent correction is not supported by the Debye scheme",
            ));
        }

        let calc: Box<dyn IntensityCalculator> = match scheme {
            IntensityScheme::Debye { qmin, qmax, nq } => {
                log::info!("Using Debye formula to calculate 𝐼(𝑞) for {qmin} ≤ 𝑞 ≤ {qmax} Å⁻¹");
                Box::new(DebyeIntensity::new(
                    self.formfactors.clone(),
                    (qmin, qmax),
                    nq,
                )?)
            }
            IntensityScheme::Direct { pmax } => {
                log::info!("Using direct Fourier transform with p_max = {pmax}");
                Box::new(DirectTransform::new(
                    self.formfactors.clone(),
                    pmax,
                    active_solvent.cloned(),
                ))
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

        if let Some(xtc_opts) = xtc_opts {
            trajectory_average(calc.as_ref(), &structure, xtc_opts)
        } else {
            calc.calc_intensities(&structure)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debye_rejects_active_solvent() {
        let yaml = r"C: { formfactor: !Constant 1.0 }";
        let path = std::env::temp_dir().join("pripps_debye_reject.yaml");
        std::fs::write(&path, yaml).unwrap();
        let pripps = Pripps::from_yaml(&path).unwrap();

        let pdb = std::env::temp_dir().join("pripps_debye_reject.pdb");
        std::fs::write(
            &pdb,
            "HEADER test\n\
             ATOM      1  C   ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n\
             END\n",
        )
        .unwrap();

        let solvent = SolventConfig {
            excluded: ExcludedVolume::Fraser { volume_scale: 1.0 },
            ..Default::default()
        };
        let err = pripps
            .intensity(
                IntensityScheme::Debye {
                    qmin: 0.0,
                    qmax: 0.1,
                    nq: 3,
                },
                &pdb,
                None,
                None,
                Some(&solvent),
            )
            .unwrap_err();
        assert!(matches!(err, Error::Usage(_)));
    }

    fn yaml_and_xyz() -> (Pripps, std::path::PathBuf) {
        let yaml = std::path::Path::new("assets/single_bead.yaml");
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

    #[test]
    fn foxs_chi2_against_experiment() {
        // FoXS gets χ²/N ≈ 0.29 on 6lyz.pdb vs lyzexp.dat with
        // excluded volume only (no hydration, c₁=1). With the
        // FoXS-compatible Cromer-Mann form factors and the 16π
        // Fraser formula, pripps should match.
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
                    qmin: 0.0,
                    qmax: 0.5,
                    nq: Some(500),
                    l_max: None,
                },
                &pdb,
                None,
                None,
                Some(&solvent),
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
    fn pepsi_pipeline_reasonable_chi2() {
        // PepsiSAXS form factors + PepsiSAXS Fraser (4π) should
        // produce a physically reasonable profile. Not as good as
        // the FoXS pipeline (different coefficients, different
        // denominator) but the shape should be plausible.
        let pripps = Pripps::from_yaml(std::path::Path::new("assets/single_bead.yaml")).unwrap();
        let pdb = std::path::PathBuf::from("tests/lysozyme/6lyz.pdb");
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
        let pripps = Pripps::from_yaml(Path::new("assets/single_bead.yaml")).unwrap();
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
}

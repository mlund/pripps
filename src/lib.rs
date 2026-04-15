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
mod formfactor;
mod input;
mod q_sampler;
mod solvent;
mod structure;
mod svergun;
mod trajectory;

use std::{
    collections::BTreeMap,
    path::{self, Path},
};

use debye::DebyeIntensity;
use explicit::DirectTransform;
use structure::Structure;
use svergun::SvergunIntensity;
use trajectory::trajectory_average;

pub use atomkind::{AtomKind, FindByName};
pub use error::{Error, Result};
pub use formfactor::{FormFactor, FormFactorMap, read_formfactors, write_formfactors};
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
/// `contributions` is populated when Svergun runs with an active
/// [`SolventConfig`], carrying the per-contribution diagnostic
/// columns. It is `None` for vacuum Svergun, Debye, and Direct.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Intensity {
    pub q: Vec<f64>,
    pub total: Vec<f64>,
    pub contributions: Option<Contributions>,
}

/// Per-contribution intensity curves emitted by the three-contribution
/// Svergun path, each parallel to [`Intensity::q`].
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
    /// Svergun's spherical-harmonic multipole expansion
    /// (*Acta Cryst. A51*, 1995). O(N·Q·L²), linear in atom count —
    /// preferred for large structures and iterative fitting. `l_max`
    /// is the multipole truncation order; `None` auto-selects from
    /// the structure's extent and `qmax`.
    Svergun {
        qmin: f64,
        qmax: f64,
        nq: usize,
        l_max: Option<usize>,
    },
}

/// Internal dispatch interface. Implementations compute I(q) for a single
/// structure; library consumers don't implement or name this trait.
pub(crate) trait IntensityCalculator: Sync {
    fn calc_intensities(&self, structure: &Structure) -> Result<Intensity>;
}

/// High-level interface to the Pripps library
pub struct Pripps {
    formfactors: BTreeMap<AtomKind, FormFactor>,
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
        self.formfactors.keys().cloned().collect()
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
        xyzfile: &path::Path,
        xtcfile: Option<&path::Path>,
        weights: Option<&path::Path>,
        box_sides: Option<[f64; 3]>,
        solvent: Option<&SolventConfig>,
    ) -> Result<Intensity> {
        let mut structure = Structure::from_file(xyzfile, &self.atomkinds())?;
        if let Some(box_sides) = box_sides {
            if xtcfile.is_some() {
                return Err(Error::usage(
                    "manual box length cannot be used with trajectories",
                ));
            }
            structure.box_sides = Some(Vector3::from(box_sides));
        }

        let active_solvent = solvent.filter(|cfg| cfg.is_active());
        if active_solvent.is_some() && !matches!(scheme, IntensityScheme::Svergun { .. }) {
            return Err(Error::usage(
                "solvent correction is only supported by the Svergun scheme",
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
                Box::new(DirectTransform::new(self.formfactors.clone(), pmax))
            }
            IntensityScheme::Svergun {
                qmin,
                qmax,
                nq,
                l_max,
            } => {
                log::info!(
                    "Using Svergun multipole expansion for 𝐼(𝑞) over {qmin} ≤ 𝑞 ≤ {qmax} Å⁻¹"
                );
                Box::new(SvergunIntensity::new(
                    self.formfactors.clone(),
                    (qmin, qmax),
                    nq,
                    l_max,
                    active_solvent.cloned(),
                )?)
            }
        };

        if let Some(xtc) = xtcfile {
            trajectory_average(calc.as_ref(), &structure, xtc, weights)
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
                None,
                Some(&solvent),
            )
            .unwrap_err();
        assert!(matches!(err, Error::Usage(_)));
    }
}

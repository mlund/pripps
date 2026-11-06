//! Scattering from a periodic box of identical molecules, frame by frame.
//!
//! A small public face over the direct (explicit q-vector) transform for
//! callers that generate frames themselves, such as a Monte Carlo run, rather
//! than reading structure files.

use crate::explicit::DirectTransform;
use crate::formfactor::{FormFactorMap, formfactors_from_yaml_str};
use crate::structure::Structure;
use crate::{AtomKind, Error, FindByName, IntensityCalculator, Result, Vector3};

/// Box intensity and per-molecule self term, averaged over q-vectors of
/// equal length. `S_eff = total / self_term`; `P = self_term / n_molecules`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BoxIntensity {
    /// Scattering vector magnitudes (Å⁻¹), ascending.
    pub q: Vec<f64>,
    /// `|Σ_m F_m(q)|²`
    pub total: Vec<f64>,
    /// `Σ_m |F_m(q)|²`
    pub self_term: Vec<f64>,
}

/// Evaluates box and self intensities for frames of a fixed set of molecules.
pub struct BoxScatterer {
    /// Form-factor id of every bead in the box, molecule after molecule.
    ids: Vec<usize>,
    backend: Backend,
}

enum Backend {
    Cpu(DirectTransform),
    #[cfg(feature = "gpu")]
    Gpu(crate::gpu::GpuDirectTransform),
}

impl BoxScatterer {
    /// GPU version of [`new_cpu`](Self::new_cpu), with phases in f32. Fails
    /// if no GPU adapter is available, so callers can fall back to the CPU.
    #[cfg(feature = "gpu")]
    pub async fn new_gpu(
        formfactors_yaml: &str,
        molecule_kinds: &[impl AsRef<str>],
        n_molecules: usize,
        p_max: usize,
    ) -> Result<Self> {
        let (formfactors, ids) = prepare(formfactors_yaml, molecule_kinds, n_molecules)?;
        let transform = crate::gpu::GpuDirectTransform::new_async(formfactors, p_max)
            .await?
            .with_num_molecules(n_molecules);
        Ok(Self {
            ids,
            backend: Backend::Gpu(transform),
        })
    }

    /// `formfactors_yaml` is the text of a pripps form-factor file;
    /// `molecule_kinds` names each bead of one molecule, matched exactly
    /// against the file's kinds (no united-atom resolution, which needs
    /// atomistic residue context). Positions passed to
    /// [`evaluate`](Self::evaluate) are `n_molecules` such molecules in order.
    /// `p_max` sets how many q-vector multiples are sampled per direction;
    /// the largest |q| must stay inside the form factors' range.
    pub fn new_cpu(
        formfactors_yaml: &str,
        molecule_kinds: &[impl AsRef<str>],
        n_molecules: usize,
        p_max: usize,
    ) -> Result<Self> {
        let (formfactors, ids) = prepare(formfactors_yaml, molecule_kinds, n_molecules)?;
        let transform =
            DirectTransform::new(formfactors, p_max, None).with_num_molecules(n_molecules);
        Ok(Self {
            ids,
            backend: Backend::Cpu(transform),
        })
    }

    /// Async because the GPU result is read back asynchronously in the
    /// browser. The CPU version completes immediately; the native GPU version
    /// blocks its thread while the device runs, as wgpu offers no wakeup.
    pub async fn evaluate(
        &self,
        positions: &[[f64; 3]],
        box_sides: [f64; 3],
    ) -> Result<BoxIntensity> {
        if positions.len() != self.ids.len() {
            return Err(Error::usage(format!(
                "expected {} bead positions, got {}",
                self.ids.len(),
                positions.len()
            )));
        }
        if !box_sides.iter().all(|&l| l.is_finite() && l > 0.0) {
            return Err(Error::usage(format!("invalid box sides {box_sides:?}")));
        }
        let structure = Structure {
            pos: positions.iter().map(|&p| Vector3::from(p)).collect(),
            ids: self.ids.clone(),
            box_sides: Some(Vector3::from(box_sides)),
        };
        let intensity = match &self.backend {
            Backend::Cpu(transform) => transform.calc_intensities(&structure)?,
            #[cfg(feature = "gpu")]
            Backend::Gpu(transform) => transform.calc_intensities_async(&structure).await?,
        };
        // Both backends set it when split into molecules; zeros keep the
        // lengths matched even if a degenerate path ever leaves it out.
        let self_term = intensity
            .self_term
            .unwrap_or_else(|| vec![0.0; intensity.q.len()]);
        Ok(BoxIntensity {
            self_term,
            q: intensity.q,
            total: intensity.total,
        })
    }
}

/// Form factors and box ids, shared by both constructors so their checks
/// can't drift apart.
fn prepare(
    formfactors_yaml: &str,
    molecule_kinds: &[impl AsRef<str>],
    n_molecules: usize,
) -> Result<(FormFactorMap, Vec<usize>)> {
    if n_molecules == 0 || molecule_kinds.is_empty() {
        return Err(Error::usage(
            "a box needs at least one molecule with one bead",
        ));
    }
    let formfactors = formfactors_from_yaml_str(formfactors_yaml)?;
    let ids = box_ids(&formfactors, molecule_kinds, n_molecules)?;
    Ok((formfactors, ids))
}

/// Form-factor ids for `n_molecules` copies of one molecule's beads. An
/// unknown kind is an error rather than a silent zero contribution.
fn box_ids(
    formfactors: &FormFactorMap,
    molecule_kinds: &[impl AsRef<str>],
    n_molecules: usize,
) -> Result<Vec<usize>> {
    let kinds: Vec<AtomKind> = formfactors.kinds().cloned().collect();
    let id_of = |name: &str| {
        kinds
            .find_name(name)
            .ok_or_else(|| Error::usage(format!("no form factor for bead kind '{name}'")))
    };
    let molecule: Vec<usize> = molecule_kinds
        .iter()
        .map(|kind| id_of(kind.as_ref()))
        .collect::<Result<_>>()?;
    Ok(molecule.repeat(n_molecules))
}

#[cfg(test)]
mod tests {
    use super::*;

    const UNIT: &str = "X: { formfactor: !Constant 1.0 }\n";

    fn evaluate(scatterer: &BoxScatterer, positions: &[[f64; 3]], l: f64) -> BoxIntensity {
        pollster::block_on(scatterer.evaluate(positions, [l; 3])).unwrap()
    }

    /// Two unit scatterers L/4 apart along x: each q-vector gives
    /// |1 + exp(iq·d)|² = 2 + 2cos(q_x d), with q_x d = ±π/2 or 0. Averaged
    /// per Miller shell: [100] (2+4+4)/3, [110] (4·2+2·4)/6, [111] 2.
    #[test]
    fn two_point_scatterers_give_the_worked_intensity() {
        let scatterer = BoxScatterer::new_cpu(UNIT, &["X"], 2, 1).unwrap();
        let result = evaluate(&scatterer, &[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], 40.0);
        let two_pi_over_l = std::f64::consts::TAU / 40.0;
        let expected_q = [1.0, 2f64.sqrt(), 3f64.sqrt()].map(|s| s * two_pi_over_l);
        for (q, expected) in result.q.iter().zip(expected_q) {
            approx::assert_relative_eq!(*q, expected, max_relative = 1e-12);
        }
        let expected_total = [10.0 / 3.0, 8.0 / 3.0, 2.0];
        for (total, expected) in result.total.iter().zip(expected_total) {
            approx::assert_relative_eq!(*total, expected, max_relative = 1e-12);
        }
        assert_eq!(result.self_term, [2.0; 3]);
    }

    /// q-vectors are commensurate with the box, so molecules need no
    /// unwrapping: moving one by a box length changes nothing.
    #[test]
    fn moving_a_molecule_by_the_box_length_changes_nothing() {
        let scatterer = BoxScatterer::new_cpu(UNIT, &["X", "X"], 2, 3).unwrap();
        let inside = [
            [1.0, 2.0, 3.0],
            [2.5, 2.0, 3.0],
            [20.0, 7.0, 11.0],
            [21.0, 8.0, 11.5],
        ];
        let mut shifted = inside;
        shifted[2][0] += 40.0;
        shifted[3][0] += 40.0;
        let (a, b) = (
            evaluate(&scatterer, &inside, 40.0),
            evaluate(&scatterer, &shifted, 40.0),
        );
        for (x, y) in a.total.iter().zip(&b.total) {
            approx::assert_relative_eq!(x, y, max_relative = 1e-9);
        }
    }

    /// A bead without a form factor would silently scatter nothing.
    #[test]
    fn unknown_bead_kind_is_an_error_naming_it() {
        let error = BoxScatterer::new_cpu(UNIT, &["X", "NTR"], 1, 1)
            .err()
            .unwrap();
        assert!(error.to_string().contains("NTR"), "{error}");
    }

    /// The GPU evaluates the phases in f32, so it agrees with the CPU only
    /// to f32 precision.
    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_matches_cpu() {
        let kinds = ["X", "X", "X"];
        let Ok(gpu) = pollster::block_on(BoxScatterer::new_gpu(UNIT, &kinds, 4, 4)) else {
            eprintln!("no GPU adapter; skipping");
            return;
        };
        let cpu = BoxScatterer::new_cpu(UNIT, &kinds, 4, 4).unwrap();
        let positions: Vec<[f64; 3]> = (0..12)
            .map(|i| {
                let t = i as f64;
                [
                    (7.3 * t) % 50.0,
                    (3.1 * t * t) % 50.0,
                    (11.7 * t + 2.0) % 50.0,
                ]
            })
            .collect();
        let (a, b) = (
            evaluate(&cpu, &positions, 50.0),
            evaluate(&gpu, &positions, 50.0),
        );
        assert_eq!(a.q.len(), b.q.len());
        for (x, y) in a
            .total
            .iter()
            .chain(&a.self_term)
            .zip(b.total.iter().chain(&b.self_term))
        {
            approx::assert_relative_eq!(x, y, max_relative = 1e-4, epsilon = 1e-4);
        }
    }

    /// The path `pripps direct --num-molecules` takes: form factors and beads
    /// read from files. BoxScatterer must resolve the same names to the same
    /// form factors from YAML text.
    #[test]
    fn matches_the_file_based_direct_path() {
        let yaml_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/assets/poly_amino_acid_combined.yaml"
        );
        let kinds = ["LYS", "GLU", "ALA"];
        let positions = [
            [1.0, 2.0, 3.0],
            [4.5, 2.0, 3.5],
            [3.0, 5.0, 1.0],
            [20.0, 21.0, 19.0],
            [23.5, 20.5, 22.0],
            [21.0, 24.0, 18.5],
        ];
        let l = 60.0;
        let mut xyz = format!("{}\nfixture\n", positions.len());
        for (kind, [x, y, z]) in kinds.iter().cycle().zip(&positions) {
            xyz += &format!("{kind} {x} {y} {z}\n");
        }
        let dir = tempfile::tempdir().unwrap();
        let xyz_path = dir.path().join("box.xyz");
        std::fs::write(&xyz_path, xyz).unwrap();

        let formfactors =
            crate::formfactor::read_formfactors(std::path::Path::new(yaml_path)).unwrap();
        let atomkinds: Vec<_> = formfactors.kinds().cloned().collect();
        let mut structure = Structure::from_file(&xyz_path, &atomkinds).unwrap();
        structure.box_sides = Some(Vector3::from([l; 3]));
        let reference = DirectTransform::new(formfactors, 3, None)
            .with_num_molecules(2)
            .calc_intensities(&structure)
            .unwrap();

        let yaml = std::fs::read_to_string(yaml_path).unwrap();
        let scatterer = BoxScatterer::new_cpu(&yaml, &kinds, 2, 3).unwrap(); // |q| ≤ 0.54 Å⁻¹, inside the 0.75 range of residue form factors
        let result = evaluate(&scatterer, &positions, l);
        assert_eq!(result.q, reference.q);
        assert_eq!(result.total, reference.total);
        assert_eq!(Some(result.self_term), reference.self_term);
    }

    #[test]
    fn empty_box_is_an_error() {
        assert!(BoxScatterer::new_cpu(UNIT, &["X"], 0, 1).is_err());
        assert!(BoxScatterer::new_cpu(UNIT, &[] as &[&str], 2, 1).is_err());
    }

    /// A zero, negative or NaN side would give inf/NaN curves, not an error.
    #[test]
    fn invalid_box_is_an_error() {
        let scatterer = BoxScatterer::new_cpu(UNIT, &["X"], 1, 1).unwrap();
        for sides in [
            [0.0, 40.0, 40.0],
            [40.0, -1.0, 40.0],
            [40.0, 40.0, f64::NAN],
        ] {
            assert!(pollster::block_on(scatterer.evaluate(&[[0.0; 3]], sides)).is_err());
        }
    }

    #[test]
    fn wrong_number_of_positions_is_an_error() {
        let scatterer = BoxScatterer::new_cpu(UNIT, &["X"], 2, 1).unwrap();
        let result = pollster::block_on(scatterer.evaluate(&[[0.0; 3]], [40.0; 3]));
        assert!(result.is_err());
    }
}

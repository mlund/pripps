use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

use std::num::NonZeroUsize;
use std::path::PathBuf;

use ::pripps::{
    DebyeModel, DirectModel, ExcludedVolume, ExpData, FitResult, Hydration, Intensity,
    IntensityScheme, MultipoleModel, Pripps, SolventConfig, TrajectoryOptions, fit_parameters,
};

// ----------------------------------------------------------------------------
// Model / solvent selection enums (also accepted as plain strings everywhere)
// ----------------------------------------------------------------------------

/// Excluded-volume model. Pass this enum or the equivalent lowercase
/// string (e.g. `pripps.Excluded.Foxs` or `"foxs"`).
#[pyclass(name = "Excluded", module = "pripps", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
enum PyExcluded {
    Disabled,
    Fraser,
    Foxs,
    Grid,
    Voronoi,
    PerKind,
}

/// Hydration-shell model. Pass this enum or the equivalent lowercase
/// string (e.g. `pripps.Hydration.Sasa` or `"sasa"`).
#[pyclass(name = "Hydration", module = "pripps", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
enum PyHydration {
    Disabled,
    Sasa,
    Grid,
    Voronoi,
}

/// Scattering-intensity calculator. Pass this enum or the equivalent
/// lowercase string (e.g. `pripps.Scheme.Multipole` or `"multipole"`).
#[pyclass(name = "Scheme", module = "pripps", eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq)]
enum PyScheme {
    Multipole,
    Debye,
    Direct,
}

/// Canonical lowercase name for an `excluded` argument that is either a
/// string, a `pripps.Excluded`, or omitted (`None` → no excluded volume).
fn excluded_name(obj: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    let Some(o) = obj else {
        return Ok("none".to_string());
    };
    if let Ok(e) = o.extract::<PyExcluded>() {
        return Ok(match e {
            PyExcluded::Disabled => "none",
            PyExcluded::Fraser => "fraser",
            PyExcluded::Foxs => "foxs",
            PyExcluded::Grid => "grid",
            PyExcluded::Voronoi => "voronoi",
            PyExcluded::PerKind => "per-kind",
        }
        .to_string());
    }
    if let Ok(s) = o.extract::<String>() {
        return Ok(s.to_lowercase());
    }
    Err(PyTypeError::new_err(
        "`excluded` must be a str or pripps.Excluded",
    ))
}

/// Canonical lowercase name for a `hydration` argument that is either a
/// string, a `pripps.Hydration`, or omitted (`None` → no hydration).
fn hydration_name(obj: Option<&Bound<'_, PyAny>>) -> PyResult<String> {
    let Some(o) = obj else {
        return Ok("none".to_string());
    };
    if let Ok(e) = o.extract::<PyHydration>() {
        return Ok(match e {
            PyHydration::Disabled => "none",
            PyHydration::Sasa => "sasa",
            PyHydration::Grid => "grid",
            PyHydration::Voronoi => "voronoi",
        }
        .to_string());
    }
    if let Ok(s) = o.extract::<String>() {
        return Ok(s.to_lowercase());
    }
    Err(PyTypeError::new_err(
        "`hydration` must be a str or pripps.Hydration",
    ))
}

/// Resolve a `model` argument (string, `pripps.Scheme`, or omitted →
/// Multipole) to a [`PyScheme`].
fn scheme_from(obj: Option<&Bound<'_, PyAny>>) -> PyResult<PyScheme> {
    let Some(o) = obj else {
        return Ok(PyScheme::Multipole);
    };
    if let Ok(e) = o.extract::<PyScheme>() {
        return Ok(e);
    }
    if let Ok(s) = o.extract::<String>() {
        return match s.to_lowercase().as_str() {
            "multipole" => Ok(PyScheme::Multipole),
            "debye" => Ok(PyScheme::Debye),
            "direct" => Ok(PyScheme::Direct),
            other => Err(PyValueError::new_err(format!(
                "unknown model: {other} (expected `multipole`, `debye`, or `direct`)"
            ))),
        };
    }
    Err(PyTypeError::new_err(
        "`model` must be a str or pripps.Scheme",
    ))
}

// ----------------------------------------------------------------------------
// Result types
// ----------------------------------------------------------------------------

/// Result of an intensity calculation: parallel NumPy arrays.
///
/// `q` and `total` are always present. `atoms`, `excluded` and
/// `hydration` are the per-contribution curves. Cached models always
/// populate them (zero where a contribution is disabled); they are
/// `None` only when the calculation produced no per-contribution
/// breakdown — currently trajectory averages.
///
/// The NumPy arrays are built once and shared on every attribute access
/// (a cheap reference clone), so repeated `.q` / `.total` reads in e.g.
/// plotting loops don't re-allocate.
#[pyclass(name = "Intensity", module = "pripps")]
struct PyIntensity {
    q: Py<PyArray1<f64>>,
    total: Py<PyArray1<f64>>,
    atoms: Option<Py<PyArray1<f64>>>,
    excluded: Option<Py<PyArray1<f64>>>,
    hydration: Option<Py<PyArray1<f64>>>,
    p: Option<Py<PyArray1<f64>>>,
    s_eff: Option<Py<PyArray1<f64>>>,
    n_points: usize,
}

impl PyIntensity {
    fn new(py: Python<'_>, inner: Intensity) -> Self {
        let arr = |v: Vec<f64>| v.into_pyarray(py).unbind();
        let opt = |v: Option<Vec<f64>>| v.map(arr);
        let n_points = inner.q.len();
        let (atoms, excluded, hydration) = match inner.contributions {
            Some(c) => (
                Some(arr(c.atoms)),
                Some(arr(c.excluded)),
                Some(arr(c.hydration)),
            ),
            None => (None, None, None),
        };
        Self {
            q: arr(inner.q),
            total: arr(inner.total),
            atoms,
            excluded,
            hydration,
            p: opt(inner.form_factor),
            s_eff: opt(inner.effective_structure_factor),
            n_points,
        }
    }
}

#[pymethods]
impl PyIntensity {
    /// Scattering vector magnitudes q (Å⁻¹).
    #[getter]
    fn q(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.q.clone_ref(py)
    }

    /// Total intensity I(q).
    #[getter]
    fn total(&self, py: Python<'_>) -> Py<PyArray1<f64>> {
        self.total.clone_ref(py)
    }

    /// Atoms-only (in-vacuo) contribution, or `None` if unavailable.
    #[getter]
    fn atoms(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.atoms.as_ref().map(|a| a.clone_ref(py))
    }

    /// Excluded-volume contribution, or `None` if unavailable.
    #[getter]
    fn excluded(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.excluded.as_ref().map(|a| a.clone_ref(py))
    }

    /// Hydration-shell contribution, or `None` if unavailable.
    #[getter]
    fn hydration(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.hydration.as_ref().map(|a| a.clone_ref(py))
    }

    /// At-concentration single-molecule form factor P(q), or `None` unless
    /// `num_molecules` was given.
    #[getter]
    fn p(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.p.as_ref().map(|a| a.clone_ref(py))
    }

    /// Effective structure factor S_eff(q) = I/(N·P), or `None` unless
    /// `num_molecules` was given.
    #[getter]
    fn s_eff(&self, py: Python<'_>) -> Option<Py<PyArray1<f64>>> {
        self.s_eff.as_ref().map(|a| a.clone_ref(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "Intensity(points={}, contributions={})",
            self.n_points,
            if self.atoms.is_some() {
                "present"
            } else {
                "none"
            }
        )
    }
}

/// Best-fit parameters from [`MultipoleModel.fit`] and friends.
#[pyclass(name = "FitResult", module = "pripps")]
struct PyFitResult {
    inner: FitResult,
}

#[pymethods]
impl PyFitResult {
    /// Fitted excluded-volume scale (PepsiSAXS `r₀` / `c₁`).
    #[getter]
    fn volume_scale(&self) -> f64 {
        self.inner.volume_scale
    }

    /// Fitted hydration contrast density (`c₂`).
    #[getter]
    fn contrast_density(&self) -> f64 {
        self.inner.contrast_density
    }

    /// Minimised χ²/N at the best fit.
    #[getter]
    fn chi2(&self) -> f64 {
        self.inner.chi2
    }

    /// Optimal linear scale between theory and experiment.
    #[getter]
    fn scale(&self) -> f64 {
        self.inner.scale
    }

    fn __repr__(&self) -> String {
        format!(
            "FitResult(volume_scale={:.4}, contrast_density={:.4}, chi2={:.4}, scale={:.4e})",
            self.inner.volume_scale, self.inner.contrast_density, self.inner.chi2, self.inner.scale
        )
    }
}

// ----------------------------------------------------------------------------
// Pripps + cached models
// ----------------------------------------------------------------------------

/// Scattering calculator loaded from a form-factor YAML file.
///
/// Construct with `Pripps(atomfile)` — a single path or a list of paths
/// merged in order (latest wins on duplicate species) — then either build a cached model
/// (`multipole_model(...)` / `debye_model(...)` / `direct_model(...)`)
/// for repeated/fitting use, or call `intensity(...)` for a one-shot
/// calculation (the only path that averages over an XTC trajectory).
#[pyclass(name = "Pripps", module = "pripps")]
struct PyPripps(Pripps);

#[pymethods]
impl PyPripps {
    /// Load a pripps instance from one or more form-factor YAML files.
    ///
    /// `atomfile` is either a single path (`Pripps("a.yaml")`) or a list
    /// of paths (`Pripps(["base.yaml", "override.yaml"])`). Multiple files
    /// are merged in order; when a species appears in more than one file
    /// the last one listed wins (a warning is logged for each override).
    #[new]
    fn new(atomfile: &Bound<'_, PyAny>) -> PyResult<Self> {
        let paths: Vec<PathBuf> = if let Ok(single) = atomfile.extract::<String>() {
            vec![single.into()]
        } else if let Ok(many) = atomfile.extract::<Vec<String>>() {
            many.into_iter().map(PathBuf::from).collect()
        } else {
            return Err(PyTypeError::new_err(
                "`atomfile` must be a str or a list of str",
            ));
        };
        Pripps::from_yaml_files(&paths)
            .map(Self)
            .map_err(into_py_err)
    }

    /// Prepare a cached Multipole model. The expensive per-structure
    /// setup runs once; `MultipoleModel.intensity(volume_scale,
    /// contrast_density)` and `MultipoleModel.fit(...)` are then cheap.
    /// `excluded` / `hydration` accept a string or a `pripps.Excluded` /
    /// `pripps.Hydration` enum (omit for none).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        structure_file,
        *,
        qmin = 0.0,
        qmax = 0.5,
        nq = 100,
        l_max = None,
        excluded = None,
        hydration = None,
        probe = 1.8,
        spacing = 3.10516,
        thickness = 3.0,
        excluded_spacing = 1.0,
        bulk_electron_density = 0.334,
    ))]
    fn multipole_model(
        &self,
        structure_file: &str,
        qmin: f64,
        qmax: f64,
        nq: usize,
        l_max: Option<usize>,
        excluded: Option<Bound<'_, PyAny>>,
        hydration: Option<Bound<'_, PyAny>>,
        probe: f64,
        spacing: f64,
        thickness: f64,
        excluded_spacing: f64,
        bulk_electron_density: f64,
    ) -> PyResult<PyMultipoleModel> {
        let cfg = solvent_basis(
            excluded.as_ref(),
            hydration.as_ref(),
            excluded_spacing,
            probe,
            spacing,
            thickness,
            bulk_electron_density,
        )?;
        self.0
            .multipole_model(structure_file.as_ref(), qmin, qmax, Some(nq), l_max, &cfg)
            .map(PyMultipoleModel)
            .map_err(into_py_err)
    }

    /// Prepare a cached Debye model (exact, O(N²·Q) — best for small
    /// structures). See `multipole_model` for the solvent arguments.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        structure_file,
        *,
        qmin = 0.0,
        qmax = 0.5,
        nq = 100,
        excluded = None,
        hydration = None,
        probe = 1.8,
        spacing = 3.10516,
        thickness = 3.0,
        excluded_spacing = 1.0,
        bulk_electron_density = 0.334,
    ))]
    fn debye_model(
        &self,
        structure_file: &str,
        qmin: f64,
        qmax: f64,
        nq: usize,
        excluded: Option<Bound<'_, PyAny>>,
        hydration: Option<Bound<'_, PyAny>>,
        probe: f64,
        spacing: f64,
        thickness: f64,
        excluded_spacing: f64,
        bulk_electron_density: f64,
    ) -> PyResult<PyDebyeModel> {
        let cfg = solvent_basis(
            excluded.as_ref(),
            hydration.as_ref(),
            excluded_spacing,
            probe,
            spacing,
            thickness,
            bulk_electron_density,
        )?;
        self.0
            .debye_model(structure_file.as_ref(), qmin, qmax, nq, &cfg)
            .map(PyDebyeModel)
            .map_err(into_py_err)
    }

    /// Prepare a cached direct/explicit model (discrete q-vectors in a
    /// cubic box of `side_length` Å). See `multipole_model` for the
    /// solvent arguments.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        structure_file,
        *,
        pmax = 6,
        side_length = 200.0,
        excluded = None,
        hydration = None,
        probe = 1.8,
        spacing = 3.10516,
        thickness = 3.0,
        excluded_spacing = 1.0,
        bulk_electron_density = 0.334,
    ))]
    fn direct_model(
        &self,
        structure_file: &str,
        pmax: usize,
        side_length: f64,
        excluded: Option<Bound<'_, PyAny>>,
        hydration: Option<Bound<'_, PyAny>>,
        probe: f64,
        spacing: f64,
        thickness: f64,
        excluded_spacing: f64,
        bulk_electron_density: f64,
    ) -> PyResult<PyDirectModel> {
        let cfg = solvent_basis(
            excluded.as_ref(),
            hydration.as_ref(),
            excluded_spacing,
            probe,
            spacing,
            thickness,
            bulk_electron_density,
        )?;
        self.0
            .direct_model(structure_file.as_ref(), pmax, [side_length; 3], &cfg)
            .map(PyDirectModel)
            .map_err(into_py_err)
    }

    /// One-shot intensity for a single structure or an XTC trajectory
    /// average. `model` selects the calculator (`"multipole"` /
    /// `"debye"` / `"direct"` or a `pripps.Scheme`). Pass `xtc=` to
    /// average over a trajectory (with optional `stride` and per-frame
    /// `weights`); trajectory averages carry no per-contribution
    /// breakdown. `volume_scale` / `contrast_density` are baked into the
    /// solvent model here (this path is not cached).
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        structure_file,
        *,
        model = None,
        xtc = None,
        stride = 1,
        weights = None,
        box_sides = None,
        excluded = None,
        hydration = None,
        qmin = 0.0,
        qmax = 0.5,
        nq = None,
        l_max = None,
        pmax = 6,
        probe = 1.8,
        spacing = 3.10516,
        thickness = 3.0,
        excluded_spacing = 1.0,
        bulk_electron_density = 0.334,
        volume_scale = 1.0,
        contrast_density = 0.03,
        num_molecules = None,
    ))]
    fn intensity(
        &self,
        py: Python<'_>,
        structure_file: &str,
        model: Option<Bound<'_, PyAny>>,
        xtc: Option<String>,
        stride: usize,
        weights: Option<String>,
        box_sides: Option<Bound<'_, PyAny>>,
        excluded: Option<Bound<'_, PyAny>>,
        hydration: Option<Bound<'_, PyAny>>,
        qmin: f64,
        qmax: f64,
        nq: Option<usize>,
        l_max: Option<usize>,
        pmax: usize,
        probe: f64,
        spacing: f64,
        thickness: f64,
        excluded_spacing: f64,
        bulk_electron_density: f64,
        volume_scale: f64,
        contrast_density: f64,
        num_molecules: Option<usize>,
    ) -> PyResult<PyIntensity> {
        let scheme = match scheme_from(model.as_ref())? {
            PyScheme::Multipole => IntensityScheme::Multipole {
                qmin,
                qmax,
                nq,
                l_max,
            },
            PyScheme::Debye => IntensityScheme::Debye {
                qmin,
                qmax,
                nq: nq.unwrap_or(100),
            },
            PyScheme::Direct => IntensityScheme::Direct { pmax },
        };
        let cfg = solvent_config(
            &excluded_name(excluded.as_ref())?,
            volume_scale,
            excluded_spacing,
            &hydration_name(hydration.as_ref())?,
            probe,
            spacing,
            thickness,
            contrast_density,
            bulk_electron_density,
        )?;
        let box_sides = match box_sides {
            None => None,
            Some(b) => Some(parse_box(&b)?),
        };
        let stride = NonZeroUsize::new(stride)
            .ok_or_else(|| PyValueError::new_err("`stride` must be >= 1"))?;
        let xtc_path = xtc.map(PathBuf::from);
        let weights_path = weights.map(PathBuf::from);
        let xtc_opts = xtc_path.as_deref().map(|p| TrajectoryOptions {
            xtcfile: p,
            stride,
            weights: weights_path.as_deref(),
        });
        self.0
            .intensity(
                scheme,
                structure_file.as_ref(),
                xtc_opts,
                box_sides,
                Some(&cfg),
                num_molecules,
            )
            .map(|inner| PyIntensity::new(py, inner))
            .map_err(into_py_err)
    }
}

/// Macro-free helper: define a cached-model pyclass with `intensity`,
/// `fit`, and `q_values`. (Written out per type for clear docstrings.)
macro_rules! cached_model {
    ($Py:ident, $Core:ty, $name:literal, $doc:literal) => {
        #[doc = $doc]
        #[pyclass(name = $name, module = "pripps")]
        struct $Py($Core);

        #[pymethods]
        impl $Py {
            /// Compute I(q) for the given fit knobs, returning an
            /// `Intensity` (numpy `.q` / `.total` / `.atoms` / ...).
            #[pyo3(signature = (volume_scale = 1.0, contrast_density = 0.03))]
            fn intensity(
                &self,
                py: Python<'_>,
                volume_scale: f64,
                contrast_density: f64,
            ) -> PyIntensity {
                PyIntensity::new(py, self.0.intensity(volume_scale, contrast_density))
            }

            /// Fit `(volume_scale, contrast_density)` to experimental
            /// data. `q`/`intensity` are equal-length arrays; `sigma`
            /// (uncertainties) defaults to ones (unweighted χ²).
            #[pyo3(signature = (q, intensity, sigma = None))]
            fn fit(
                &self,
                q: Vec<f64>,
                intensity: Vec<f64>,
                sigma: Option<Vec<f64>>,
            ) -> PyResult<PyFitResult> {
                run_fit(q, intensity, sigma, |c1, c2| self.0.intensity(c1, c2))
            }

            /// The q grid the model was prepared over.
            #[getter]
            fn q_values<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
                self.0.q_values().to_vec().into_pyarray(py)
            }
        }
    };
}

cached_model!(
    PyMultipoleModel,
    MultipoleModel,
    "MultipoleModel",
    "A cached Multipole model from `Pripps.multipole_model`."
);
cached_model!(
    PyDebyeModel,
    DebyeModel,
    "DebyeModel",
    "A cached Debye model from `Pripps.debye_model`."
);
cached_model!(
    PyDirectModel,
    DirectModel,
    "DirectModel",
    "A cached direct/explicit model from `Pripps.direct_model`."
);

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

/// Build a `SolventConfig` for a cached model: the model is built at a
/// unit basis and the `volume_scale`/`contrast_density` knobs are applied
/// later in `intensity()`, so placeholder values are used here.
fn solvent_basis(
    excluded: Option<&Bound<'_, PyAny>>,
    hydration: Option<&Bound<'_, PyAny>>,
    excluded_spacing: f64,
    probe: f64,
    spacing: f64,
    thickness: f64,
    bulk_electron_density: f64,
) -> PyResult<SolventConfig> {
    solvent_config(
        &excluded_name(excluded)?,
        1.0,
        excluded_spacing,
        &hydration_name(hydration)?,
        probe,
        spacing,
        thickness,
        0.0,
        bulk_electron_density,
    )
}

#[allow(clippy::too_many_arguments)]
fn solvent_config(
    excluded: &str,
    volume_scale: f64,
    excluded_spacing: f64,
    hydration: &str,
    probe: f64,
    spacing: f64,
    thickness: f64,
    contrast_density: f64,
    bulk_electron_density: f64,
) -> PyResult<SolventConfig> {
    let excluded = match excluded {
        "none" | "disabled" => ExcludedVolume::Disabled,
        "fraser" => ExcludedVolume::Fraser { volume_scale },
        "foxs" => ExcludedVolume::FraserFoxs { volume_scale },
        "grid" => ExcludedVolume::Grid {
            spacing: excluded_spacing,
            volume_scale,
        },
        "voronoi" => ExcludedVolume::Voronoi { volume_scale },
        "per-kind" | "perkind" => ExcludedVolume::PerKind { volume_scale },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown excluded-volume model: {other} (expected `none`, `fraser`, `foxs`, `grid`, `voronoi`, or `per-kind`)"
            )));
        }
    };
    let hydration = match hydration {
        "none" | "disabled" => Hydration::Disabled,
        "sasa" => Hydration::Sasa {
            probe,
            contrast_density,
        },
        "grid" => Hydration::Grid {
            spacing,
            thickness,
            probe,
            contrast_density,
        },
        "voronoi" => Hydration::Voronoi {
            probe,
            contrast_density,
        },
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown hydration model: {other} (expected `none`, `sasa`, `grid`, or `voronoi`)"
            )));
        }
    };
    Ok(SolventConfig {
        excluded,
        hydration,
        bulk_electron_density,
    })
}

/// Parse a `box_sides` argument: a single float (cubic box) or a 3-tuple.
fn parse_box(o: &Bound<'_, PyAny>) -> PyResult<[f64; 3]> {
    if let Ok(a) = o.extract::<[f64; 3]>() {
        return Ok(a);
    }
    if let Ok(v) = o.extract::<f64>() {
        return Ok([v; 3]);
    }
    Err(PyValueError::new_err(
        "`box_sides` must be a float (cubic) or a 3-sequence of floats",
    ))
}

/// Run the core (volume_scale, contrast_density) fit using `eval` as the
/// per-parameter intensity evaluator.
fn run_fit<F: Fn(f64, f64) -> Intensity>(
    q: Vec<f64>,
    intensity: Vec<f64>,
    sigma: Option<Vec<f64>>,
    eval: F,
) -> PyResult<PyFitResult> {
    let exp = ExpData::from_arrays(q, intensity, sigma.unwrap_or_default()).map_err(into_py_err)?;
    let inner = fit_parameters(&exp, eval).map_err(into_py_err)?;
    Ok(PyFitResult { inner })
}

fn into_py_err(err: ::pripps::Error) -> PyErr {
    use ::pripps::Error::*;
    let msg = err.to_string();
    match err {
        File { .. } | Io(_) => PyIOError::new_err(msg),
        Yaml(_)
        | UnknownAtom(_)
        | UndefinedFormFactor(_)
        | QOutOfRange { .. }
        | Validation(_)
        | Usage(_) => PyValueError::new_err(msg),
        Internal(_) => PyRuntimeError::new_err(msg),
    }
}

/// Fast small-angle scattering (SAXS/SANS) intensities for atomic and
/// coarse-grained structures.
///
/// Load a form-factor table with `Pripps(atomfile)`, build a cached model
/// with `Pripps.multipole_model(structure_file, ...)`, then call
/// `model.intensity(volume_scale, contrast_density)` (→ `Intensity` with
/// numpy `.q` / `.total` / `.atoms` / `.excluded` / `.hydration`) or
/// `model.fit(q, intensity, sigma)` (→ `FitResult`). For trajectory
/// averaging use the one-shot `Pripps.intensity(model=..., xtc=...)`.
#[pymodule]
fn pripps(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Bridge the core library's `log` records into Python's `logging`
    // module so messages (e.g. the grid hydration shell summary)
    // surface when callers configure `logging`. Without this no `log`
    // backend is installed in the extension and every record is
    // dropped silently.
    pyo3_log::init();
    m.add_class::<PyPripps>()?;
    m.add_class::<PyMultipoleModel>()?;
    m.add_class::<PyDebyeModel>()?;
    m.add_class::<PyDirectModel>()?;
    m.add_class::<PyIntensity>()?;
    m.add_class::<PyFitResult>()?;
    m.add_class::<PyExcluded>()?;
    m.add_class::<PyHydration>()?;
    m.add_class::<PyScheme>()?;
    Ok(())
}

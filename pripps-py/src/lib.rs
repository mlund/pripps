use numpy::IntoPyArray;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use ::pripps::{
    DebyeModel, DirectModel, ExcludedVolume, Hydration, Intensity, MultipoleModel, Pripps,
    SolventConfig,
};

#[pyclass(name = "Pripps", module = "pripps")]
struct PyPripps(Pripps);

#[pymethods]
impl PyPripps {
    /// Load a pripps instance from a form-factor YAML file.
    #[new]
    fn new(atomfile: &str) -> PyResult<Self> {
        Pripps::from_yaml(atomfile.as_ref())
            .map(Self)
            .map_err(into_py_err)
    }

    /// Prepare a stateful Multipole fit. Caches per-q coefficient
    /// triangles so `MultipoleModel.intensity` is cheap to call
    /// repeatedly with different `(volume_scale, contrast_density)`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        structure_file,
        *,
        qmin = 0.0,
        qmax = 0.5,
        nq = 100,
        l_max = None,
        excluded = "none",
        volume_scale = 1.0,
        excluded_spacing = 1.0,
        hydration = "none",
        probe = 1.8,
        spacing = 3.10516,
        thickness = 3.0,
        contrast_density = 0.03,
        bulk_electron_density = 0.334,
    ))]
    fn multipole_model(
        &self,
        structure_file: &str,
        qmin: f64,
        qmax: f64,
        nq: usize,
        l_max: Option<usize>,
        excluded: &str,
        volume_scale: f64,
        excluded_spacing: f64,
        hydration: &str,
        probe: f64,
        spacing: f64,
        thickness: f64,
        contrast_density: f64,
        bulk_electron_density: f64,
    ) -> PyResult<PyMultipoleModel> {
        let cfg = solvent_from_kwargs(
            excluded,
            volume_scale,
            excluded_spacing,
            hydration,
            probe,
            spacing,
            thickness,
            contrast_density,
            bulk_electron_density,
        )?;
        self.0
            .multipole_model(structure_file.as_ref(), qmin, qmax, Some(nq), l_max, &cfg)
            .map(PyMultipoleModel)
            .map_err(into_py_err)
    }

    /// Prepare a stateful Debye fit. Caches exact per-q Debye terms
    /// so `DebyeModel.intensity` is cheap to call repeatedly with
    /// different `(volume_scale, contrast_density)`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        structure_file,
        *,
        qmin = 0.0,
        qmax = 0.5,
        nq = 100,
        excluded = "none",
        volume_scale = 1.0,
        excluded_spacing = 1.0,
        hydration = "none",
        probe = 1.8,
        spacing = 3.10516,
        thickness = 3.0,
        contrast_density = 0.03,
        bulk_electron_density = 0.334,
    ))]
    fn debye_model(
        &self,
        structure_file: &str,
        qmin: f64,
        qmax: f64,
        nq: usize,
        excluded: &str,
        volume_scale: f64,
        excluded_spacing: f64,
        hydration: &str,
        probe: f64,
        spacing: f64,
        thickness: f64,
        contrast_density: f64,
        bulk_electron_density: f64,
    ) -> PyResult<PyDebyeModel> {
        let cfg = solvent_from_kwargs(
            excluded,
            volume_scale,
            excluded_spacing,
            hydration,
            probe,
            spacing,
            thickness,
            contrast_density,
            bulk_electron_density,
        )?;
        self.0
            .debye_model(structure_file.as_ref(), qmin, qmax, nq, &cfg)
            .map(PyDebyeModel)
            .map_err(into_py_err)
    }

    /// Prepare a stateful direct/explicit fit. Caches per-q-vector
    /// amplitudes so `DirectModel.intensity` is cheap to call
    /// repeatedly with different `(volume_scale, contrast_density)`.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        structure_file,
        *,
        pmax = 6,
        side_length = 200.0,
        excluded = "none",
        volume_scale = 1.0,
        excluded_spacing = 1.0,
        hydration = "none",
        probe = 1.8,
        spacing = 3.10516,
        thickness = 3.0,
        contrast_density = 0.03,
        bulk_electron_density = 0.334,
    ))]
    fn direct_model(
        &self,
        structure_file: &str,
        pmax: usize,
        side_length: f64,
        excluded: &str,
        volume_scale: f64,
        excluded_spacing: f64,
        hydration: &str,
        probe: f64,
        spacing: f64,
        thickness: f64,
        contrast_density: f64,
        bulk_electron_density: f64,
    ) -> PyResult<PyDirectModel> {
        let cfg = solvent_from_kwargs(
            excluded,
            volume_scale,
            excluded_spacing,
            hydration,
            probe,
            spacing,
            thickness,
            contrast_density,
            bulk_electron_density,
        )?;
        self.0
            .direct_model(structure_file.as_ref(), pmax, [side_length; 3], &cfg)
            .map(PyDirectModel)
            .map_err(into_py_err)
    }
}

#[pyclass(name = "MultipoleModel", module = "pripps")]
struct PyMultipoleModel(MultipoleModel);

#[pymethods]
impl PyMultipoleModel {
    /// Compute I(q) for the given fit parameters. Returns a dict of
    /// numpy arrays: `q`, `total`, `atoms`, `excluded`, `hydration`.
    fn intensity<'py>(
        &self,
        py: Python<'py>,
        volume_scale: f64,
        contrast_density: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = self.0.intensity(volume_scale, contrast_density);
        intensity_to_dict(py, out)
    }

    /// The q grid the fit was prepared over.
    #[getter]
    fn q_values<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.0.q_values().to_vec().into_pyarray(py)
    }
}

#[pyclass(name = "DebyeModel", module = "pripps")]
struct PyDebyeModel(DebyeModel);

#[pymethods]
impl PyDebyeModel {
    /// Compute I(q) for the given fit parameters. Returns a dict of
    /// numpy arrays: `q`, `total`, `atoms`, `excluded`, `hydration`.
    fn intensity<'py>(
        &self,
        py: Python<'py>,
        volume_scale: f64,
        contrast_density: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = self.0.intensity(volume_scale, contrast_density);
        intensity_to_dict(py, out)
    }

    /// The q grid the fit was prepared over.
    #[getter]
    fn q_values<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.0.q_values().to_vec().into_pyarray(py)
    }
}

#[pyclass(name = "DirectModel", module = "pripps")]
struct PyDirectModel(DirectModel);

#[pymethods]
impl PyDirectModel {
    /// Compute I(q) for the given fit parameters. Returns a dict of
    /// numpy arrays: `q`, `total`, `atoms`, `excluded`, `hydration`.
    fn intensity<'py>(
        &self,
        py: Python<'py>,
        volume_scale: f64,
        contrast_density: f64,
    ) -> PyResult<Bound<'py, PyDict>> {
        let out = self.0.intensity(volume_scale, contrast_density);
        intensity_to_dict(py, out)
    }

    /// The duplicate-averaged q grid the fit was prepared over.
    #[getter]
    fn q_values<'py>(&self, py: Python<'py>) -> Bound<'py, numpy::PyArray1<f64>> {
        self.0.q_values().to_vec().into_pyarray(py)
    }
}

#[allow(clippy::too_many_arguments)]
fn solvent_from_kwargs(
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

fn intensity_to_dict<'py>(py: Python<'py>, intensity: Intensity) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("q", intensity.q.into_pyarray(py))?;
    dict.set_item("total", intensity.total.into_pyarray(py))?;
    if let Some(c) = intensity.contributions {
        dict.set_item("atoms", c.atoms.into_pyarray(py))?;
        dict.set_item("excluded", c.excluded.into_pyarray(py))?;
        dict.set_item("hydration", c.hydration.into_pyarray(py))?;
    }
    Ok(dict)
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
    Ok(())
}

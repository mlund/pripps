// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.
// You may obtain a copy of the license at
//
//     http://www.apache.org/licenses/license-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the license is distributed on an "as is" basis,
// without warranties or conditions of any kind, either express or implied.
// See the license for the specific language governing permissions and
// limitations under the license.

use crate::debye::sinc;
use crate::error::IoResultExt;
use crate::{AtomKind, Error, Result};
use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::{cmp::Ordering, collections::BTreeMap, fs::File, ops::Add, path::Path};

/// A flat table pairing each [`AtomKind`] with its runtime [`FormFactor`],
/// stored in atom-kind id order. Lookups by id are `O(1)` vector indexes.
//
// Vec rather than `BTreeMap<AtomKind, FormFactor>`: every consumer
// iterates or indexes by id, and id *is* the entry's position, so a
// map adds lookup ceremony for no benefit. Keeping the type a
// newtype (rather than a bare `Vec`) lets Phase 2b+ add invariants
// without churning call sites.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FormFactorMap {
    entries: Vec<(AtomKind, FormFactor)>,
}

impl FormFactorMap {
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &(AtomKind, FormFactor)> {
        self.entries.iter()
    }

    /// Atom kind at id `id`. Panics on out-of-range id.
    pub(crate) fn kind(&self, id: usize) -> &AtomKind {
        &self.entries[id].0
    }

    pub(crate) fn kinds(&self) -> impl Iterator<Item = &AtomKind> {
        self.entries.iter().map(|(k, _)| k)
    }

    pub(crate) fn form_factors(&self) -> impl Iterator<Item = &FormFactor> {
        self.entries.iter().map(|(_, f)| f)
    }

    /// Linear-search the table for the kind named `name` and return
    /// a borrow of its form factor. Used by the solvent generators
    /// (SASA, Phase 2c/2d) to look up role-specific kinds like
    /// `H2O` without exposing the underlying storage.
    pub(crate) fn form_factor_by_name(&self, name: &str) -> Option<&FormFactor> {
        self.entries
            .iter()
            .find(|(k, _)| k.name() == name)
            .map(|(_, f)| f)
    }

    pub(crate) const fn from_entries(entries: Vec<(AtomKind, FormFactor)>) -> Self {
        Self { entries }
    }
}

/// Runtime form factor model, 𝐹(𝑞).
///
/// Always directly usable: every variant can be evaluated on a valid q. The
/// declarative variants (`Alias`, `United`, `Linear`) only exist on
/// [`FormFactorSpec`] and are flattened into one of the four variants below
/// by [`read_formfactors`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum FormFactor {
    /// 𝐹(𝑞) = 𝑐
    Constant(f64),
    /// 𝐹(𝑞) = 𝑎 + 𝑏𝑞 + 𝑐𝑞² + … valid within an inclusive q range
    Polynomial { qrange: (f64, f64), coeff: Vec<f64> },
    /// Interpolated 𝐹(𝑞) from a list of (𝑞, 𝐹) pairs
    Table(Vec<(f64, f64)>),
    /// Multi-exponential, 𝐹(𝑞) = 𝑐 + ∑ 𝑎ᵢ exp(−𝑏ᵢ𝑞²), valid within an inclusive q range
    Gaussian {
        qrange: (f64, f64),
        ab: Vec<(f64, f64)>,
        c: f64,
    },
    /// 𝐹(𝑞) = 𝑎 · exp(−𝑏𝑞²). Const-constructible, no heap
    /// allocation. Used by the Fraser excluded-volume dummies where
    /// each atom gets its own single-term Gaussian sized by its
    /// displaced volume.
    SingleGaussian { a: f64, b: f64 },
}

impl FormFactor {
    /// Build a single-Gaussian form factor. Convenience constructor
    /// for dummy atoms (see [`crate::ExcludedVolume::Fraser`]).
    pub const fn single_gaussian(a: f64, b: f64) -> Self {
        Self::SingleGaussian { a, b }
    }

    pub fn qrange(&self) -> (f64, f64) {
        use FormFactor::*;
        match self {
            Constant(_) | SingleGaussian { .. } => (0.0, f64::INFINITY),
            Gaussian { qrange, .. } => *qrange,
            Polynomial { qrange, .. } => *qrange,
            Table(table) => (table[0].0, table[table.len() - 1].0),
        }
    }

    pub fn in_qrange(&self, q: f64) -> bool {
        let (lo, hi) = self.qrange();
        q >= lo && q <= hi
    }

    /// Tabulate `F(q)` at the given q values. Errors if any q is out of range.
    pub fn to_table(&self, qvalues: impl IntoIterator<Item = f64>) -> Result<Vec<(f64, f64)>> {
        qvalues
            .into_iter()
            .map(|q| self.eval(q).map(|f| (q, f)))
            .try_collect()
    }

    pub fn eval(&self, q: f64) -> Result<f64> {
        use FormFactor::*;
        let (lo, hi) = self.qrange();
        if !(lo..=hi).contains(&q) {
            return Err(Error::QOutOfRange { q, lo, hi });
        }
        Ok(match self {
            Constant(c) => *c,
            SingleGaussian { a, b } => a * (-b * q * q).exp(),
            Gaussian { ab, c, .. } => eval_gaussian(q, ab, *c),
            Polynomial { coeff, .. } => eval_polynomium(coeff, q),
            Table(table) => interpolate(table, q),
        })
    }
}

/// Declarative form-factor specification as read from YAML.
///
/// Contains the four runtime variants plus three symbolic ones (`Alias`,
/// `United`, `Linear`) that reference other entries in the spec map by name.
/// Resolved into a [`FormFactor`] by [`read_formfactors`]; never seen by the
/// scattering calculators or by library consumers.
#[derive(Debug, Clone, Deserialize, PartialEq)]
enum FormFactorSpec {
    Constant(f64),
    Polynomial {
        qrange: (f64, f64),
        coeff: Vec<f64>,
    },
    Table(Vec<(f64, f64)>),
    Gaussian {
        qrange: (f64, f64),
        ab: Vec<(f64, f64)>,
        c: f64,
    },
    /// Refers by name to another entry in the spec map; replaced with a clone
    /// of that entry during resolution.
    Alias(String),
    /// United atom: a central site flanked by orbiters with distance offsets.
    /// 𝐹′(𝑞) = 𝐹(𝑞) + ∑ 𝑤ᵢ⋅𝐹ᵢ(𝑞)⋅sinc(𝑞𝑟ᵢ). Resolved to a `Table`.
    United {
        center: String,
        orbit: Vec<(String, f64, Option<f64>)>,
    },
    /// Linear combination, 𝐹(𝑞) = ∑ 𝑤ᵢ⋅𝐹ᵢ(𝑞). Resolved to a `Table`.
    Linear(Vec<(String, f64)>),
}

impl FormFactorSpec {
    /// Symbolic variants return `(0, ∞)` — the resolver passes rely on this to
    /// skip range filtering for not-yet-flattened entries, not as a correctness
    /// claim.
    fn qrange(&self) -> (f64, f64) {
        use FormFactorSpec::*;
        match self {
            Constant(_) => (0.0, f64::INFINITY),
            Gaussian { qrange, .. } => *qrange,
            Polynomial { qrange, .. } => *qrange,
            Table(table) => (table[0].0, table[table.len() - 1].0),
            Alias(_) | United { .. } | Linear(_) => (0.0, f64::INFINITY),
        }
    }

    /// Bails on unresolved symbolic variants. Only called from the resolver
    /// passes, after the relevant pass has flattened its own variant.
    fn eval(&self, q: f64) -> Result<f64> {
        use FormFactorSpec::*;
        let (lo, hi) = self.qrange();
        if !(lo..=hi).contains(&q) {
            return Err(Error::QOutOfRange { q, lo, hi });
        }
        Ok(match self {
            Constant(c) => *c,
            Gaussian { ab, c, .. } => eval_gaussian(q, ab, *c),
            Polynomial { coeff, .. } => eval_polynomium(coeff, q),
            Table(table) => interpolate(table, q),
            Alias(name) => {
                return Err(Error::internal(format!(
                    "unexpected alias form factor '{name}'"
                )));
            }
            Linear(_) => return Err(Error::internal("unexpected linear form factor")),
            United { center, .. } => {
                return Err(Error::internal(format!(
                    "unexpected united form factor '{center}'"
                )));
            }
        })
    }
}

impl TryFrom<FormFactorSpec> for FormFactor {
    type Error = Error;

    fn try_from(spec: FormFactorSpec) -> Result<Self> {
        use FormFactorSpec as S;
        Ok(match spec {
            S::Constant(c) => Self::Constant(c),
            S::Polynomial { qrange, coeff } => Self::Polynomial { qrange, coeff },
            S::Table(t) => Self::Table(t),
            S::Gaussian { qrange, ab, c } => Self::Gaussian { qrange, ab, c },
            S::Alias(name) => {
                return Err(Error::internal(format!(
                    "unresolved alias '{name}' reached conversion boundary"
                )));
            }
            S::United { center, .. } => {
                return Err(Error::internal(format!(
                    "unresolved united '{center}' reached conversion boundary"
                )));
            }
            S::Linear(_) => {
                return Err(Error::internal(
                    "unresolved linear form factor reached conversion boundary",
                ));
            }
        })
    }
}

/// Atom specification as read from YAML.
#[derive(Debug, Clone, Deserialize, PartialEq)]
struct AtomSpec {
    /// Radius of the atom in Å.
    radius: Option<f64>,
    /// Displaced solvent volume in Å³.
    displaced_volume: Option<f64>,
    /// Form factor for the atom.
    #[serde(alias = "ff")]
    formfactor: FormFactorSpec,
}

// If `displaced_volume` is absent but `radius` is present,
// derive V = (4/3)πr³. Explicit `displaced_volume` in the YAML
// takes precedence — needed for united-atom groups (NH, SH)
// where the displaced volume includes implicit hydrogens that
// the effective radius doesn't account for.
impl From<(&str, &AtomSpec)> for AtomKind {
    fn from((name, spec): (&str, &AtomSpec)) -> Self {
        let displaced_volume = spec.displaced_volume.or_else(|| {
            spec.radius
                .map(|r| (4.0 / 3.0) * std::f64::consts::PI * r.powi(3))
        });
        Self {
            name: name.to_string(),
            sigma: spec.radius.map(|r| 2.0 * r),
            displaced_volume,
        }
    }
}

/// Evaluate Gaussian form factor, 𝐹(𝑞) = 𝑐 + ∑ 𝑎ᵢ exp(−𝑏ᵢ𝑠²)
/// where 𝑠 = 𝑞/(4π) (the International Tables / Cromer-Mann
/// convention). Both PepsiSAXS and FoXS use this convention;
/// published 𝑏 values are always in 𝑠² units.
#[inline]
fn eval_gaussian(q: f64, ab: &[(f64, f64)], c: f64) -> f64 {
    let s2 = q * q / (16.0 * std::f64::consts::PI * std::f64::consts::PI);
    ab.iter()
        .map(|(a, b)| a * f64::exp(-b * s2))
        .sum::<f64>()
        .add(c)
}

/// Evaluate polynomium form factor, 𝐹(𝑞) = ∑ 𝑐ᵢ𝑞ⁱ
#[inline]
fn eval_polynomium(coeff: &[f64], q: f64) -> f64 {
    coeff
        .iter()
        .enumerate()
        .map(|(i, c)| c * q.powi(i as i32))
        .sum()
}

/// Linear interpolation using binary search; requires x sorted (x,y) table
#[inline]
fn interpolate(table: &[(f64, f64)], x: f64) -> f64 {
    debug_assert!(table.is_sorted());
    assert!(table.len() >= 2);

    // Binary search to find interval
    let mut low = 0;
    let mut high = table.len() - 1;
    while low < high {
        let mid = (low + high) / 2;
        match x.partial_cmp(&table[mid].0).unwrap() {
            Ordering::Less => high = mid,
            Ordering::Greater => low = mid + 1,
            Ordering::Equal => return table[mid].1,
        }
    }
    // After binary search, `low` points to the second point in the interval
    let (x0, y0) = table[low - 1];
    let (x1, y1) = table[low];
    y0 + (x - x0) * (y1 - y0) / (x1 - x0)
}

/// Read YAML file with form factors for different atom types (atomic or coarse grained)
pub fn read_formfactors(path: &Path) -> Result<FormFactorMap> {
    let file = File::open(path).with_path(path)?;
    let mut specs: BTreeMap<String, AtomSpec> = yaml_serde::from_reader(file)?;

    convert_alias(&mut specs)?;
    convert_linear(&mut specs)?;
    convert_united(&mut specs)?;

    // Spec → runtime: the three resolvers above guarantee every remaining
    // variant is one of the four concrete ones, so `try_from` only errors if
    // a resolver pass has a bug.
    let entries: Vec<(AtomKind, FormFactor)> = specs
        .into_iter()
        .map(|(name, spec)| {
            let kind = AtomKind::from((&*name, &spec));
            let ff = FormFactor::try_from(spec.formfactor)?;
            Ok((kind, ff))
        })
        .collect::<Result<_>>()?;

    let formfactor = FormFactorMap::from_entries(entries);
    log::info!(
        "Read form-factor definitions from {} for {} species",
        path.display(),
        formfactor.len(),
    );
    Ok(formfactor)
}

/// Convert all formfactors to a TABLE and Write to YAML file
///
/// The q-interval is hard-coded to [0.0, 0.5] with 100 points.
pub fn write_formfactors(formfactor: &FormFactorMap, path: &Path) -> Result<()> {
    log::info!("Exporting tabulated form factors to {}", path.display());

    let qvalues = crate::q_sampler::linear_q_values(0.0, 0.5, 100);
    let formfactor: Vec<_> = formfactor
        .iter()
        .map(|(kind, ff)| ff.to_table(qvalues.clone()).map(|t| (kind.name(), t)))
        .try_collect()?;

    let mut file = File::create(path).with_path(path)?;
    yaml_serde::to_writer(&mut file, &formfactor)?;
    Ok(())
}

/// Create N vector of form factors at given q value (N = number of atoms)
pub fn formfactor_by_index(
    formfactors: &FormFactorMap,
    q: f64,
    atom_ids: &[usize],
) -> Result<DVector<f64>> {
    // Evaluate each atom kind's form factor once, then broadcast to atoms.
    let f: Vec<_> = formfactors
        .form_factors()
        .map(|ff| ff.eval(q))
        .try_collect()?;

    Ok(DVector::from_iterator(
        atom_ids.len(),
        atom_ids.iter().map(|id| f[*id]),
    ))
}

/// Create an S x S matrix of form factor products at a given q value
/// (S = number of atom kinds). Library-facing helper; not called from
/// the CLI path but exercised by a unit test.
#[allow(dead_code)]
pub fn formfactor_product_by_id(formfactors: &FormFactorMap, q: f64) -> Result<DMatrix<f64>> {
    let f = DVector::from_vec(
        formfactors
            .form_factors()
            .map(|ff| ff.eval(q))
            .try_collect()?,
    );
    Ok(&f * f.transpose())
}

fn convert_linear(specs: &mut BTreeMap<String, AtomSpec>) -> Result<()> {
    const TABLE_LEN: usize = 100;
    const MAX_Q: f64 = 2.0; // Å⁻¹

    let no_linear: BTreeMap<String, AtomSpec> = specs
        .iter()
        .filter(|(_, spec)| !matches!(spec.formfactor, FormFactorSpec::Linear(_)))
        .map(|(name, spec)| (name.clone(), spec.clone()))
        .collect();

    for spec in specs.values_mut() {
        let FormFactorSpec::Linear(components) = &spec.formfactor else {
            continue;
        };
        let components = components.clone();

        let get_ff = |name: &str| {
            no_linear
                .get(name)
                .map(|s| &s.formfactor)
                .ok_or_else(|| Error::UndefinedFormFactor(name.to_string()))
        };

        // 𝐹(𝑞) = ∑ 𝑤ᵢ⋅𝐹ᵢ(𝑞)
        let lin_combination = |q| -> Result<(f64, f64)> {
            components
                .iter()
                .map(|(name, weight)| Ok(weight * get_ff(name)?.eval(q)?))
                .sum::<Result<f64>>()
                .map(|f| (q, f))
        };

        // Find highest q_min and lowest q_max in the components
        let mut qmin = 0.0;
        let mut qmax = MAX_Q;
        for (name, ..) in components.iter() {
            let qrange = get_ff(name)?.qrange();
            qmin = qrange.0.max(qmin);
            qmax = qrange.1.min(qmax);
        }
        let qvalues = crate::q_sampler::linear_q_values(qmin, qmax, TABLE_LEN);

        let table = qvalues.iter().cloned().map(lin_combination).try_collect()?;
        spec.formfactor = FormFactorSpec::Table(table);
    }
    Ok(())
}

fn convert_alias(specs: &mut BTreeMap<String, AtomSpec>) -> Result<()> {
    let no_alias: BTreeMap<String, FormFactorSpec> = specs
        .iter()
        .filter(|(_, spec)| !matches!(spec.formfactor, FormFactorSpec::Alias(_)))
        .map(|(name, spec)| (name.clone(), spec.formfactor.clone()))
        .collect();

    for spec in specs.values_mut() {
        let FormFactorSpec::Alias(name) = &spec.formfactor else {
            continue;
        };
        spec.formfactor = no_alias
            .get(name)
            .ok_or_else(|| Error::UndefinedFormFactor(name.to_string()))?
            .clone();
    }
    Ok(())
}

fn convert_united(specs: &mut BTreeMap<String, AtomSpec>) -> Result<()> {
    const TABLE_SIZE: usize = 100;
    const MAX_Q: f64 = 2.0; // Å⁻¹

    let no_united: BTreeMap<String, AtomSpec> = specs
        .iter()
        .filter(|(_, spec)| !matches!(spec.formfactor, FormFactorSpec::United { .. }))
        .map(|(name, spec)| (name.clone(), spec.clone()))
        .collect();

    // Evaluates single united form factor to a tabulated form factor
    let eval = |ff: &FormFactorSpec| -> Result<FormFactorSpec> {
        let FormFactorSpec::United { center, orbit } = ff else {
            return Err(Error::internal("united form factor expected"));
        };
        let center_ff = &no_united
            .get(center)
            .map(|s| &s.formfactor)
            .ok_or_else(|| Error::UndefinedFormFactor(center.clone()))?;

        let (qmin, mut qmax) = center_ff.qrange();
        if qmax.is_infinite() {
            log::debug!("Infinite q-range for '{center}'; setting max q to {MAX_Q} Å⁻¹");
            qmax = MAX_Q;
        }

        let mut table: Vec<_> = (0..=TABLE_SIZE)
            .map(|i| qmin + (qmax - qmin) * i as f64 / TABLE_SIZE as f64)
            .map(|q| center_ff.eval(q).map(|f| (q, f)))
            .try_collect()?;

        for (name, offset, weight) in orbit {
            let orbit_ff = &no_united
                .get(name)
                .map(|s| &s.formfactor)
                .ok_or_else(|| Error::UndefinedFormFactor(name.clone()))?;
            let weight = weight.unwrap_or(1.0);
            for (q, f) in table.iter_mut() {
                *f += weight * orbit_ff.eval(*q)? * sinc(*q * offset);
            }
        }
        Ok(FormFactorSpec::Table(table))
    };

    for spec in specs.values_mut() {
        if matches!(spec.formfactor, FormFactorSpec::United { .. }) {
            spec.formfactor = eval(&spec.formfactor)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_formfactor_eval() {
        let ff = FormFactor::Polynomial {
            qrange: (0.0, 1.0),
            coeff: vec![1.0, 2.0, 3.0, 4.0, 5.0],
        };
        assert_relative_eq!(ff.eval(0.0).unwrap(), 1.0);
        assert_relative_eq!(ff.eval(1.0).unwrap(), 15.0);
    }

    #[test]
    fn error_variant_for_out_of_range_q() {
        let ff = FormFactor::Polynomial {
            qrange: (0.0, 1.0),
            coeff: vec![1.0],
        };
        let err = ff.eval(2.0).unwrap_err();
        assert!(
            matches!(err, Error::QOutOfRange { .. }),
            "expected QOutOfRange, got {err:?}",
        );
    }

    // Test Table and linear interpolation
    #[test]
    fn test_formfactor_eval_table() {
        let ff = FormFactor::Table(vec![(0.0, 1.0), (0.5, 2.0), (1.0, 3.0), (1.5, 6.0)]);
        assert_relative_eq!(ff.eval(0.0).unwrap(), 1.0); // first
        assert_relative_eq!(ff.eval(0.5).unwrap(), 2.0); // second
        assert_relative_eq!(ff.eval(1.0).unwrap(), 3.0); // third
        assert_relative_eq!(ff.eval(1.5).unwrap(), 6.0); // fourth
        assert_relative_eq!(ff.eval(0.25).unwrap(), 1.5); // between first and second
        assert_relative_eq!(ff.eval(0.75).unwrap(), 2.5); // between second and third
        assert!(ff.eval(-0.1).is_err()); // out of q range
        assert!(ff.eval(2.1).is_err()); // out of q range
    }

    #[test]
    fn single_gaussian_evaluates_in_q2_convention() {
        // SingleGaussian is for Fraser dummies — uses bare q²
        // convention (NOT the Cromer-Mann s² convention that
        // Gaussian uses). At q=0 both give F=a; at q>0 they
        // differ because SingleGaussian decays faster.
        let a = 5.486;
        let b = 0.42;
        let single = FormFactor::single_gaussian(a, b);
        assert_relative_eq!(single.eval(0.0).unwrap(), a);
        // At q=0.5: a * exp(-b * 0.25) = 5.486 * exp(-0.105) = 4.939
        let expected = a * (-b * 0.25_f64).exp();
        assert_relative_eq!(single.eval(0.5).unwrap(), expected);
        // Const-constructible: this compiles at const position.
        const _: FormFactor = FormFactor::single_gaussian(1.0, 2.0);
    }

    #[test]
    fn gaussian_evaluates_in_s2_convention() {
        // Gaussian uses the International Tables s² = (q/(4π))²
        // convention. At SAXS angles the form factor barely decays.
        let ff = FormFactor::Gaussian {
            qrange: (0.0, f64::INFINITY),
            ab: vec![(6.0, 20.0)],
            c: 0.0,
        };
        assert_relative_eq!(ff.eval(0.0).unwrap(), 6.0);
        // At q=0.5: s² = 0.25/(16π²) ≈ 0.001583
        // F = 6 * exp(-20 * 0.001583) = 6 * exp(-0.0317) ≈ 5.813
        let s2 = 0.25 / (16.0 * std::f64::consts::PI * std::f64::consts::PI);
        let expected = 6.0 * (-20.0 * s2).exp();
        assert_relative_eq!(ff.eval(0.5).unwrap(), expected, epsilon = 1e-10);
    }

    #[test]
    fn test_formfactor_product_at_q() {
        let formfactors = FormFactorMap::from_entries(vec![
            (AtomKind::new("A"), FormFactor::Constant(1.0)),
            (AtomKind::new("B"), FormFactor::Constant(2.0)),
        ]);
        let matrix = formfactor_product_by_id(&formfactors, 1.0).unwrap();
        assert_relative_eq!(matrix[(0, 0)], 1.0);
        assert_relative_eq!(matrix[(0, 1)], 2.0);
        assert_relative_eq!(matrix[(1, 1)], 4.0);
    }

    /// Round-trip the shipped YAML through `read_formfactors` and verify every
    /// resolved entry evaluates finitely at a q inside all declared ranges.
    /// Exercises the `Alias` / `United` / `Gaussian` / `Table` resolver paths
    /// end-to-end and pins the spec→runtime pipeline.
    #[test]
    fn test_read_formfactors_single_bead() {
        let path = Path::new("assets/single_bead.yaml");
        let ff = read_formfactors(path).expect("parse single_bead.yaml");
        assert!(!ff.is_empty(), "expected at least one species");
        for (kind, f) in ff.iter() {
            let v = f.eval(0.1).unwrap_or_else(|e| {
                panic!("eval failed for {} at q=0.1: {e}", kind.name());
            });
            assert!(v.is_finite(), "non-finite F(0.1) for {}", kind.name());
        }
    }
}

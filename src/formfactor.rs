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

use crate::{AtomKind, AtomKindBuilder, sinc};
use anyhow::{Result, anyhow, bail};
use faunus::topology::Value;
use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    fs::File,
    ops::Add,
    path::Path,
};

/// Used to deserialize from YAML file
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
struct AtomRecord {
    /// Radius of the atom in Å
    radius: Option<f64>,
    /// Solvent radius in Å. TODO: better name
    dsv: Option<f64>,
    /// Form factor for the atom
    #[serde(alias = "ff")]
    formfactor: FormFactor,
}

impl AtomRecord {
    /// Helper to convert input file record to AtomKind
    ///
    /// The optional `dsv` field is saved as a custom property.
    fn to_atomkind(&self, name: &str, id: usize) -> Result<AtomKind> {
        let custom = self
            .dsv
            .map(|dsv| HashMap::from([("dsv".to_owned(), Value::Float(dsv))]));

        AtomKindBuilder::default()
            .name(name)
            .id(id)
            .sigma(2.0 * self.radius.unwrap_or(0.0))
            .custom(custom.unwrap_or_default())
            .build()
            .map_err(|e| anyhow!("invalid atom info: {e}"))
    }
}

/// Map of form factors for each atomkind
pub type FormFactorMap = BTreeMap<AtomKind, FormFactor>;

/// Models for site form factors, F(q)
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum FormFactor {
    /// 𝐹(𝑞) = constant for all q values
    Constant(f64),
    /// 𝐹(𝑞) = 𝑎 + 𝑏𝑞 + 𝑐𝑞² + ... valid within an inclusive q range
    Polynomial { qrange: (f64, f64), coeff: Vec<f64> },
    /// Interpolated 𝐹(𝑞) from a list of (𝑞,𝐹) pairs
    Table(Vec<(f64, f64)>),
    /// Multi-exponential, 𝐹(𝑞) = 𝑐 + ∑ 𝑎ᵢ exp(−𝑏ᵢ𝑞²)
    Gaussian {
        /// Valid q range, inclusive
        qrange: (f64, f64),
        /// List of (a, b) pairs for each of the exponential terms
        ab: Vec<(f64, f64)>,
        /// Offset
        c: f64,
    },
    /// 𝐹(𝑞) is the same an another defined form factor.
    ///
    /// Used only during deserialization where it is replaced with the actual form factor
    Alias(String),
    /// United form factor where a central atom is flanked by multiple other atoms with distance offsets, 𝑟ᵢ
    ///
    /// 𝐹′(𝑞) = 𝐹(𝑞) + ∑ 𝑤ᵢ⋅𝐹ᵢ(𝑞)⋅sinc(𝑞𝑟ᵢ)
    ///
    /// Used only during deserialization where it is replaced with a tabulated form factor
    United {
        /// Central atom
        center: String,
        /// Orbiting atoms given by a list of tuples (name, offset in Å, optional weight). Default weights, 𝑤ᵢ=1.
        orbit: Vec<(String, f64, Option<f64>)>,
    },
    /// Linear combination of two or more form factors
    /// 𝐹(𝑞) = ∑ 𝑤ᵢ⋅𝐹ᵢ(𝑞)
    /// Used only during deserialization where it is replaced with a tabulated form factor
    Linear(Vec<(String, f64)>),
}

impl FormFactor {
    /// Get the q-range for the form factor
    pub fn qrange(&self) -> (f64, f64) {
        use FormFactor::*;
        match self {
            Constant(_) => (0.0, f64::INFINITY),
            Gaussian { qrange, .. } => *qrange,
            Polynomial { qrange, .. } => *qrange,
            Table(table) => (table[0].0, table[table.len() - 1].0),
            // These should be converted and filtered out during deserialization
            Alias(_) => (0.0, f64::INFINITY),
            United { .. } => (0.0, f64::INFINITY),
            Linear { .. } => (0.0, f64::INFINITY),
        }
    }
    /// Determine if a q value is within valid q-range
    pub fn in_qrange(&self, q: f64) -> bool {
        let qrange = self.qrange();
        q >= qrange.0 && q <= qrange.1
    }

    /// Convert any form factor to a tabulated form factor in the given q range
    ///
    /// Errors if the q values are outside valid range (checked by `eval()`)
    pub fn to_table(&self, qvalues: impl IntoIterator<Item = f64>) -> Result<Vec<(f64, f64)>> {
        qvalues
            .into_iter()
            .map(|q| self.eval(q).map(|f| (q, f)))
            .try_collect()
    }

    /// Evaluate form factor, 𝐹(𝑞)
    pub fn eval(&self, q: f64) -> Result<f64> {
        use FormFactor::*;
        if !self.in_qrange(q) {
            bail!("q value outside of form factor range");
        }

        let formfactor = match self {
            Constant(c) => *c,
            Gaussian { ab, c, .. } => eval_gaussian(q, ab, *c),
            Polynomial { coeff, .. } => eval_polynomium(coeff, q),
            Table(table) => interpolate(table, q),
            // These should be converted and filtered out during deserialization
            Alias(name) => bail!("Unexpected alias form factor '{name}'"),
            Linear { .. } => bail!("unexpected linear form factor"),
            United { center, .. } => bail!("unexpected united form factor '{center}'"),
        };
        Ok(formfactor)
    }
}

/// Evaluate Gaussian form factor, 𝐹(𝑞) = 𝑐 + ∑ 𝑎ᵢ exp(−𝑏ᵢ𝑞²)
#[inline]
fn eval_gaussian(q: f64, ab: &[(f64, f64)], c: f64) -> f64 {
    ab.iter()
        .map(|(a, b)| a * f64::exp(-b * q * q))
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
    let file = File::open(path)
        .or_else(|e| bail!("error opening formfactor file {}: {}", path.display(), e))?;

    let mut formfactor: BTreeMap<String, AtomRecord> =
        serde_yaml::from_reader(file).map_err(|e| anyhow!("error parsing formfactor file: {e}"))?;

    convert_alias(&mut formfactor)?;
    convert_linear(&mut formfactor)?;
    convert_united(&mut formfactor)?;

    // (String, AtomRecord) -> (AtomKind, FormFactor)
    let formfactor: FormFactorMap = formfactor
        .into_iter()
        .enumerate()
        .map(|(id, (name, record))| {
            record
                .to_atomkind(&name, id)
                .map(|kind| (kind, record.formfactor))
        })
        .try_collect()?;

    assert!(formfactor.keys().map(|kind| kind.id()).is_sorted());
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

    let qvalues = crate::debye::make_q_values(0.0, 0.5, 100);
    let formfactor: Vec<_> = formfactor
        .iter()
        .map(|(kind, ff)| ff.to_table(qvalues.clone()).map(|t| (kind.name(), t)))
        .try_collect()?;

    let mut file = File::create(path)?;
    serde_yaml::to_writer(&mut file, &formfactor)
        .map_err(|e| anyhow!("error writing formfactor file {}: {e}", path.display()))
}

/// Helper to convert "Linear" form factors to tabulated form factors
fn convert_linear(formfactor: &mut BTreeMap<String, AtomRecord>) -> Result<()> {
    const TABLE_LEN: usize = 100;
    const MAX_Q: f64 = 2.0; // Å⁻¹

    let no_linear = &formfactor
        .iter()
        .filter(|(_, record)| !matches!(record.formfactor, FormFactor::Linear { .. }))
        .map(|(name, ff)| (name.clone(), ff.clone()))
        .collect::<BTreeMap<_, _>>();

    for (_, record) in formfactor.iter_mut() {
        let FormFactor::Linear(components) = record.formfactor.clone() else {
            continue;
        };

        let get_ff = |name| {
            no_linear
                .get(name)
                .map(|r| &r.formfactor)
                .ok_or_else(|| anyhow!("undefined form factor '{name}'"))
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
        let qvalues = crate::debye::make_q_values(qmin, qmax, TABLE_LEN);

        let table = qvalues.iter().cloned().map(lin_combination).try_collect()?;
        record.formfactor = FormFactor::Table(table);
    }
    Ok(())
}

/// Helper to convert alias form factors to the actual form factors
fn convert_alias(name_formfactor: &mut BTreeMap<String, AtomRecord>) -> Result<()> {
    // Map of form without aliases
    let no_alias = &name_formfactor
        .iter()
        .map(|(name, record)| (name, record.formfactor.clone()))
        .filter(|(_, ff)| !matches!(ff, FormFactor::Alias(_)))
        .map(|(name, ff)| (name.clone(), ff.clone()))
        .collect::<BTreeMap<_, _>>();

    // Convert all alias variants to the actual form factor variant that they refer to
    for (_, record) in name_formfactor.iter_mut() {
        let FormFactor::Alias(name) = record.formfactor.clone() else {
            continue;
        };
        record.formfactor = no_alias
            .get(&name)
            .ok_or_else(|| anyhow!("undefined aliased form factor '{name}'"))?
            .clone();
    }
    Ok(())
}

/// Helper to convert all united variants to combined, tabulated form factors using the Debye formula
fn convert_united(formfactor: &mut BTreeMap<String, AtomRecord>) -> Result<()> {
    const TABLE_SIZE: usize = 100;
    const MAX_Q: f64 = 2.0; // Å⁻¹

    let no_united = &formfactor
        .iter()
        .filter(|(_, record)| !matches!(record.formfactor, FormFactor::United { .. }))
        .map(|(name, ff)| (name.clone(), ff.clone()))
        .collect::<BTreeMap<_, _>>();

    // Evaluates single united form factor to a tabulated form factor
    let eval = |ff: &FormFactor| {
        let FormFactor::United { center, orbit } = ff else {
            bail!("united form factor expected");
        };
        let center_ff = &no_united
            .get(center)
            .map(|r| &r.formfactor)
            .ok_or_else(|| anyhow!("undefined central atom form factor '{center}'"))?;

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
                .map(|r| &r.formfactor)
                .ok_or_else(|| anyhow!("undefined orbiting atom form factor '{name}'"))?;
            let weight = weight.unwrap_or(1.0);
            for (q, f) in table.iter_mut() {
                *f += weight * orbit_ff.eval(*q)? * sinc(*q * offset);
            }
        }
        Ok(FormFactor::Table(table))
    };

    for (_, record) in formfactor.iter_mut() {
        if let FormFactor::United { .. } = record.formfactor {
            record.formfactor = eval(&record.formfactor)?;
        }
    }
    Ok(())
}

/// Create N vector of form factors at given q value (N = number of atoms)
pub fn formfactor_by_index(
    formfactors: &FormFactorMap,
    q: f64,
    atom_ids: &[usize],
) -> Result<DVector<f64>> {
    // form factors for each atom kind
    let f: Vec<_> = formfactors.values().map(|f| f.eval(q)).try_collect()?;

    Ok(DVector::from_iterator(
        atom_ids.len(),
        atom_ids.iter().map(|id| f[*id]),
    ))
}

/// Create S x S matrix of form factor products at a given q value (S = number of atom types)
pub fn formfactor_product_by_id(formfactors: &FormFactorMap, q: f64) -> Result<DMatrix<f64>> {
    let f = DVector::from_vec(formfactors.values().map(|f| f.eval(q)).try_collect()?);
    Ok(&f * f.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomKindBuilder;
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
    fn test_formfactor_product_at_q() {
        let formfactors = vec![
            (
                AtomKindBuilder::default().name("A").id(0).build().unwrap(),
                FormFactor::Constant(1.0),
            ),
            (
                AtomKindBuilder::default().name("B").id(1).build().unwrap(),
                FormFactor::Constant(2.0),
            ),
        ]
        .into_iter()
        .collect();
        let matrix = formfactor_product_by_id(&formfactors, 1.0).unwrap();
        assert_relative_eq!(matrix[(0, 0)], 1.0);
        assert_relative_eq!(matrix[(0, 1)], 2.0);
        assert_relative_eq!(matrix[(1, 1)], 4.0);
    }
}

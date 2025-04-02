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

use crate::{AtomKind, AtomKindBuilder};
use anyhow::{Result, bail};
use itertools::Itertools;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::{collections::BTreeMap, ops::Deref};

/// Used to deserialize from YAML file
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
struct AtomRecord {
    radius: Option<f64>,
    formfactor: FormFactor,
}

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
    /// Multi-exponential, 𝐹(𝑞) = 𝑐 + ∑𝑎ᵢ exp(−𝑏ᵢ𝑞²)
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
    /// 𝐹′(𝑞) = 𝐹(𝑞) + ∑ 𝑤ᵢ⋅𝐹ᵢ(𝑞)⋅sin(𝑞𝑟ᵢ)/𝑞𝑟ᵢ
    ///
    /// Used only during deserialization where it is replaced with a tabulated form factor
    United {
        /// Central atom
        center: String,
        /// Orbiting atoms given by a list of tuples (name, offset in Å, optional weight). Default weights, 𝑤ᵢ=1.
        orbit: Vec<(String, f64, Option<f64>)>,
    },
}

impl FormFactor {
    /// Get the q-range for the form factor
    pub fn qrange(&self) -> (f64, f64) {
        use FormFactor::*;
        match self {
            Constant(_) => (0.0, f64::INFINITY),
            Polynomial { qrange, .. } => *qrange,
            Table(table) => (table[0].0, table[table.len() - 1].0),
            Gaussian { qrange, .. } => *qrange,
            Alias(_) => (0.0, f64::INFINITY),
            United { .. } => (0.0, f64::INFINITY),
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
    pub fn to_tabulated(&self, qvalues: impl IntoIterator<Item = f64>) -> Result<FormFactor> {
        let table = qvalues
            .into_iter()
            .map(|q| self.eval(q).map(|f| (q, f)))
            .collect::<Result<Vec<_>>>()?;
        Ok(FormFactor::Table(table))
    }

    /// Evaluate form factor, 𝐹(𝑞)
    pub fn eval(&self, q: f64) -> Result<f64> {
        use FormFactor::*;
        if !self.in_qrange(q) {
            bail!("q value outside of form factor range");
        }
        let formfactor = match self {
            Constant(c) => *c,
            Polynomial { coeff, .. } => eval_polynomium(coeff, q),
            Table(table) => interpolate(table, q),
            Gaussian { ab, c, .. } => eval_gaussian(q, ab, *c),
            // These should be converted and filtered out during deserialization
            Alias(name) => bail!("Unexpected alias form factor '{name}'"),
            United { center, .. } => bail!("unexpected united form factor '{center}'"),
        };
        Ok(formfactor)
    }
    /// Evaluate form factor, 𝐹(𝑞), with added solvent
    ///
    /// 𝐼(𝑞) = 𝐹(𝑞) + 𝑤⋅𝐹_solvent(𝑞)⋅sin(𝑞⋅offset)/(𝑞⋅offset)
    pub fn eval_with_solvent(
        &self,
        q: f64,
        solvent: &FormFactor,
        offset: f64,
        weight: Option<f64>,
    ) -> Result<f64> {
        let intensity = self.eval(q)?
            + weight.unwrap_or(1.0) * solvent.eval(q)? * f64::sin(q * offset) / (q * offset);
        Ok(intensity)
    }
}

/// Evaluate Gaussian form factor, 𝐹(𝑞) = 𝑐 + ∑𝑎ᵢ exp(−𝑏ᵢ𝑞²)
#[inline]
fn eval_gaussian(q: f64, ab: &[(f64, f64)], c: f64) -> f64 {
    ab.iter()
        .map(|(a, b)| a * f64::exp(-b * q * q))
        .sum::<f64>()
        + c
}

/// Evaluate polynomium form factor, 𝐹(𝑞) = c₀ + c₁𝑞 + c₂𝑞² + ...
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
    debug_assert!(table.len() >= 2);
    let last = table.len() - 1;
    if x == table[0].0 {
        table[0].1
    } else if x >= table[last].0 {
        table[last].1
    } else {
        let compare_x = |(xi, _): &(f64, f64)| xi.partial_cmp(&x).unwrap();
        let i = table.binary_search_by(compare_x).unwrap_or(last - 1);
        let (x1, y1) = table[i];
        let (x2, y2) = table[i + 1];
        y1 + (y2 - y1) / (x2 - x1) * (x - x1)
    }
}

/// Read YAML file with form factors for different atom types (atomic or coarse grained)
pub fn read_formfactors(path: &std::path::Path) -> Result<FormFactorMap> {
    let file =
        std::fs::File::open(path).or_else(|e| bail!("error opening formfactor file: {e}"))?;

    let mut formfactor: BTreeMap<String, AtomRecord> = serde_yaml::from_reader(file)
        .map_err(|e| anyhow::anyhow!("error parsing formfactor file: {e}"))?;

    convert_alias(&mut formfactor)?;
    convert_united(&mut formfactor)?;

    let formfactor: FormFactorMap = formfactor
        .into_iter()
        .enumerate()
        .map(|(id, (name, record))| {
            AtomKindBuilder::default()
                .name(name)
                .id(id)
                .sigma(2.0 * record.radius.unwrap_or(0.0))
                .build()
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
        if let FormFactor::Alias(name) = record.formfactor.clone() {
            match no_alias.get(&name) {
                Some(ff) => record.formfactor = ff.clone(),
                None => bail!("undefined aliased form factor '{name}'"),
            }
        }
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
    let eval = |ff: &FormFactor| -> Result<FormFactor> {
        let FormFactor::United { center, orbit } = ff else {
            bail!("united form factor expected");
        };
        let center_ff = no_united
            .get(center)
            .ok_or_else(|| anyhow::anyhow!("undefined central atom form factor '{center}'"))?;

        let (qmin, mut qmax) = center_ff.formfactor.qrange();
        if qmax.is_infinite() {
            log::debug!("Infinite q-range for '{center}'; setting max q to {MAX_Q} Å⁻¹");
            qmax = MAX_Q;
        }

        let mut table = (0..=TABLE_SIZE)
            .map(|i| qmin + (qmax - qmin) * i as f64 / TABLE_SIZE as f64)
            .map(|q| center_ff.formfactor.eval(q).map(|f| (q, f)))
            .collect::<Result<Vec<_>>>()?;

        for (name, r, weight) in orbit {
            let orbit_ff = no_united
                .get(name)
                .ok_or_else(|| anyhow::anyhow!("undefined orbiting atom form factor '{name}'"))?;
            let weight = weight.unwrap_or(1.0);
            for (q, f) in table.iter_mut() {
                let qr = q.deref() * r;
                *f += weight * orbit_ff.formfactor.eval(*q)? * qr.sin() / qr;
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
    let ff_by_id = formfactors
        .values()
        .map(|f| f.eval(q))
        .collect::<Result<Vec<_>>>()?;

    Ok(DVector::from_iterator(
        atom_ids.len(),
        atom_ids.iter().map(|id| ff_by_id[*id]),
    ))
}

/// Create S x S matrix of form factor products at a given q value (S = atom id)
pub fn formfactor_product_by_id(formfactors: &FormFactorMap, q: f64) -> Result<DMatrix<f64>> {
    let f_vec = formfactors
        .values()
        .map(|f| f.eval(q))
        .collect::<Result<Vec<_>>>()?;
    let f = DVector::from_vec(f_vec);
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
        let ff = FormFactor::Table(vec![(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]);
        assert_relative_eq!(ff.eval(0.0).unwrap(), 1.0);
        assert_relative_eq!(ff.eval(0.1).unwrap(), 1.1);
        assert_relative_eq!(ff.eval(0.5).unwrap(), 1.5);
        assert_relative_eq!(ff.eval(1.0).unwrap(), 2.0);
        assert_relative_eq!(ff.eval(1.5).unwrap(), 2.5);
        assert_relative_eq!(ff.eval(1.9).unwrap(), 2.9);
        assert_relative_eq!(ff.eval(2.0).unwrap(), 3.0);
        let ff = FormFactor::Table(vec![(0.0, 1.0), (1.0, 2.0)]);
        assert_relative_eq!(ff.eval(0.0).unwrap(), 1.0);
        assert_relative_eq!(ff.eval(0.5).unwrap(), 1.5);
        assert_relative_eq!(ff.eval(1.0).unwrap(), 2.0);
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

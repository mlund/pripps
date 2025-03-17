// Copyright 2025 Isabel Vinterbladh, Rémi Vuillemot, Sergei Grudinin and Mikael Lund
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
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::collections::BTreeMap;

/// Models for site form factors, F(q)
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum FormFactor {
    /// F(q)=constant for all q values
    Constant(f64),
    /// F(q) given by an arbitrary length polynomial valid within an inclusive q range
    Polynomial { qrange: (f64, f64), coeff: Vec<f64> },
    /// F(q) given by a list of `(q, F)` pairs
    Table(Vec<(f64, f64)>),
    /// Nine-coefficient exponential (Guo 1995, doi:10.1107/s0108767395010038, Eq. 3)
    NineCoeff {
        /// Valid q range, inclusive
        qrange: (f64, f64),
        /// (a, b) pairs for each of the four exponential terms
        ab: [(f64, f64); 4],
        /// Offset
        c: f64,
    },
    /// F(q) is the same an another defined form factor
    Alias(String),
}

impl FormFactor {
    /// Determine if a q value is within valid q-range
    pub fn in_qrange(&self, q: f64) -> bool {
        use FormFactor::*;
        match self {
            Constant(_) | Alias(_) => true,
            NineCoeff { qrange, .. } => q >= qrange.0 && q <= qrange.1,
            Polynomial { qrange, .. } => q >= qrange.0 && q <= qrange.1,
            Table(table) => q >= table[0].0 && q <= table[table.len() - 1].0,
        }
    }

    /// Evaluate the form factor, F(q)
    pub fn eval(&self, q: f64) -> Result<f64> {
        use FormFactor::*;
        if !self.in_qrange(q) {
            bail!("q value outside of form factor range");
        }
        let formfactor = match self {
            Constant(c) => *c,
            Polynomial { coeff, .. } => coeff
                .iter()
                .enumerate()
                .map(|(i, c)| c * q.powi(i as i32))
                .sum(),
            Table(table) => {
                // Linear interpolation using binary search; requires sorted table
                if q == table[0].0 {
                    table[0].1
                } else if q >= table[table.len() - 1].0 {
                    table[table.len() - 1].1
                } else {
                    let i = table
                        .binary_search_by(|(q_i, _)| q.partial_cmp(q_i).unwrap())
                        .unwrap_or(table.len() - 2);
                    let (q1, f1) = table[i];
                    let (q2, f2) = table[i + 1];
                    f1 + (f2 - f1) * (q - q1) / (q2 - q1)
                }
            }
            NineCoeff { ab, c, .. } => {
                // https://doi.org/10.1107/s0108767395010038, Eq. 3
                // TODO: Should c be inside or outside the sum?
                ab.iter()
                    .map(|(a, b)| a * f64::exp(-b * q * q))
                    .sum::<f64>()
                    + c
            }
            Alias(name) => bail!("Invalid SameAs form factor {}", name),
        };
        Ok(formfactor)
    }
}

/// Read YAML file with form factors for different atom types (atomic or coarse grained)
pub fn read_formfactors(path: &std::path::PathBuf) -> Result<BTreeMap<AtomKind, FormFactor>> {
    let file =
        std::fs::File::open(path).or_else(|e| bail!("error opening formfactor file: {e}"))?;

    let mut name_formfactor: BTreeMap<String, FormFactor> = serde_yaml::from_reader(file)
        .map_err(|e| anyhow::anyhow!("error parsing formfactor file: {e}"))?;

    // Map of form without aliases
    let no_alias = &name_formfactor
        .iter()
        .filter(|(_, ff)| !matches!(ff, FormFactor::Alias(_)))
        .map(|(name, ff)| (name.clone(), ff.clone()))
        .collect::<BTreeMap<_, _>>();

    // Convert all alias variants to the actual form factor variant that they refer to
    for (_, formfactor) in name_formfactor.iter_mut() {
        if let FormFactor::Alias(name) = formfactor {
            match no_alias.get(name) {
                Some(ff) => *formfactor = ff.clone(),
                None => bail!("Aliased form factor {} is not defined", name),
            }
        }
    }

    let formfactor: BTreeMap<AtomKind, FormFactor> = name_formfactor
        .into_iter()
        .enumerate()
        .map(|(id, (name, formfactor))| {
            let kind = AtomKindBuilder::default()
                .name(name)
                .id(id)
                .build()
                .unwrap(); // todo: handle error
            (kind, formfactor)
        })
        .collect();

    assert!(formfactor.keys().map(|kind| kind.id()).is_sorted());
    log::info!(
        "Read form-factor definitions from {} for {} species",
        path.display(),
        formfactor.len(),
    );
    Ok(formfactor)
}

/// Create S x S matrix of form factor products at a given q value (S = atom id)
pub fn formfactor_product(
    formfactors: &BTreeMap<AtomKind, FormFactor>,
    q: f64,
) -> Result<DMatrix<f64>> {
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
        let matrix = formfactor_product(&formfactors, 1.0).unwrap();
        assert_relative_eq!(matrix[(0, 0)], 1.0);
        assert_relative_eq!(matrix[(0, 1)], 2.0);
        assert_relative_eq!(matrix[(1, 1)], 4.0);
    }
}

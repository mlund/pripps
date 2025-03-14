// Copyright 2025 Isabel Vinterbladh, Sergei Grudinin and Mikael Lund
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

use crate::AtomKind;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::collections::BTreeMap;

/// Models for site form factors, F(q)
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum FormFactor {
    /// F(q)=constant for all q values
    Constant(f64),
    /// F(q) given by a polynomial
    Polynomial([f64; 5]),
    /// F(q) given by a table of 10 q and F(q) values
    Table(Vec<(f64, f64)>),
}

impl FormFactor {
    /// Evaluate the form factor, F(q)
    pub fn eval(&self, q: f64) -> f64 {
        match self {
            FormFactor::Constant(c) => *c,
            FormFactor::Polynomial(coeff) => coeff
                .iter()
                .enumerate()
                .map(|(i, c)| c * q.powi(i as i32))
                .sum(),
            FormFactor::Table(_table) => {
                todo!("interpolation of form factor table")
            }
        }
    }
}

/// Read YAML file with form factors for different atom types (atomic or coarse grained)
pub fn read_formfactors(path: &std::path::PathBuf) -> Result<BTreeMap<AtomKind, FormFactor>> {
    let file = std::fs::File::open(path)
        .or_else(|e| anyhow::bail!("error opening formfactor file: {e}"))?;

    let name_formfactor: BTreeMap<String, FormFactor> = serde_yaml::from_reader(file)
        .map_err(|e| anyhow::anyhow!("error parsing formfactor file: {e}"))?;

    let kind_formfactor: BTreeMap<AtomKind, FormFactor> = name_formfactor
        .into_iter()
        .enumerate()
        .map(|(id, (name, formfactor))| {
            let kind = crate::AtomKindBuilder::default()
                .name(name)
                .id(id)
                .build()
                .unwrap(); // todo: handle error
            (kind, formfactor)
        })
        .collect();

    assert!(kind_formfactor.keys().map(|kind| kind.id()).is_sorted());
    Ok(kind_formfactor)
}

/// Create S x S matrix of form factor products at a given q value (S = atom id)
pub fn formfactor_product_at_q(
    formfactors: &BTreeMap<AtomKind, FormFactor>,
    q: f64,
) -> nalgebra::DMatrix<f64> {
    let formfactors: Vec<_> = formfactors.values().collect();
    let n = formfactors.len();
    let mut matrix = nalgebra::DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        let f1 = formfactors[i].eval(q);
        for j in i..n {
            let f2 = formfactors[j].eval(q);
            matrix[(i, j)] = f1 * f2;
            matrix[(j, i)] = f1 * f2;
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomKindBuilder;
    use approx::assert_relative_eq;

    #[test]
    fn test_formfactor_eval() {
        let ff = FormFactor::Polynomial([1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_relative_eq!(ff.eval(0.0), 1.0);
        assert_relative_eq!(ff.eval(1.0), 15.0);
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
        let matrix = formfactor_product_at_q(&formfactors, 1.0);
        assert_relative_eq!(matrix[(0, 0)], 1.0);
        assert_relative_eq!(matrix[(0, 1)], 2.0);
        assert_relative_eq!(matrix[(1, 1)], 4.0);
    }
}

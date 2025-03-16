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

use crate::{AtomKind, AtomKindBuilder};
use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::collections::BTreeMap;

/// Models for site form factors, F(q)
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum FormFactor {
    /// F(q)=constant for all q values
    Constant(f64),
    /// F(q) given by a polynomial valid within an inclusive q range
    Polynomial { qrange: (f64, f64), coeff: [f64; 5] },
    /// F(q) given by a table of q and F(q) pairs
    Table(Vec<(f64, f64)>),
}

impl FormFactor {
    /// Determine if a q value is within valid q-range
    pub fn in_qrange(&self, q: f64) -> bool {
        use FormFactor::*;
        match self {
            Constant(_) => true,
            Polynomial { qrange, .. } => q >= qrange.0 && q <= qrange.1,
            Table(table) => q >= table[0].0 && q <= table[table.len() - 1].0,
        }
    }

    /// Evaluate the form factor, F(q)
    pub fn eval(&self, q: f64) -> Result<f64> {
        use FormFactor::*;
        if !self.in_qrange(q) {
            anyhow::bail!("q value outside of form factor range");
        }
        let f = match self {
            Constant(c) => *c,
            Polynomial { coeff, .. } => coeff
                .iter()
                .enumerate()
                .map(|(i, c)| c * q.powi(i as i32))
                .sum(),
            Table(..) => {
                // TODO: implement linear interpolation
                // _table is a sorted list of (q, I) pairs so we can use *binary search*
                log::warn!("interpolation not yet implemented - assuming unity");
                1.0
            }
        };
        Ok(f)
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
            let kind = AtomKindBuilder::default()
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
            coeff: [1.0, 2.0, 3.0, 4.0, 5.0],
        };
        assert_relative_eq!(ff.eval(0.0).unwrap(), 1.0);
        assert_relative_eq!(ff.eval(1.0).unwrap(), 15.0);
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

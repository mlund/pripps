use crate::AtomKind;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_yaml;
use std::collections::HashMap;

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
            FormFactor::Polynomial(p) => {
                let mut i = 0;
                let mut sum = 0.0;
                while i < p.len() {
                    sum += p[i] * q.powi(i as i32);
                    i += 1;
                }
                sum
            }
            FormFactor::Table(t) => {
                let mut i = 0;
                while i < t.len() - 1 && t[i].0 < q {
                    i += 1;
                }
                let (q1, f1) = t[i];
                let (q2, f2) = t[i + 1];
                f1 + (f2 - f1) * (q - q1) / (q2 - q1)
            }
        }
    }

    // Evaluate the formfactor for a range of q values
    pub fn eval_range(&self, q_values: &[f64]) -> Vec<(f64, f64)> {
        q_values.iter().map(|q| (*q, self.eval(*q))).collect()
    }
}

/// Read YAML file with form factors for different atom types (atomic or coarse grained)
pub fn read_formfactors(path: &std::path::PathBuf) -> Result<HashMap<AtomKind, FormFactor>> {
    let file = std::fs::File::open(path)
        .or_else(|e| anyhow::bail!("error opening formfactor file: {e}"))?;
    let name_formfactor: HashMap<String, FormFactor> = serde_yaml::from_reader(file)
        .map_err(|e| anyhow::anyhow!("error parsing formfactor file: {e}"))?;
    let kind_formfactor: HashMap<AtomKind, FormFactor> = name_formfactor
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
    Ok(kind_formfactor)
}

/// Create a hashmap of evaluated form form factors for a range of q values. Key = atom id
pub fn eval_formfactors(
    formfactors: &HashMap<AtomKind, FormFactor>,
    q_values: &[f64],
) -> HashMap<usize, Vec<(f64, f64)>> {
    formfactors
        .iter()
        .map(|(kind, formfactor)| (kind.id(), formfactor.eval_range(q_values)))
        .collect()
}

use crate::{AtomKind, FormFactor, Structure, Vector3};
use std::collections::HashMap;

#[allow(unused)]
pub fn calc_intensity(
    structure: Structure,
    formfactors: &HashMap<AtomKind, FormFactor>,
    qspan: (f64, f64),
    nq: usize,
) -> Vec<(f64, f64)> {
    let q_values: Vec<f64> = (0..nq)
        .map(|i| qspan.0 + (qspan.1 - qspan.0) * i as f64 / (nq - 1) as f64)
        .collect();

    let distances = distance_matrix(&structure.pos);

    todo!("Calculate the scattering intensity using the Debye formula")
}

// Calculate formfactor product and store in N x N matrix for a given q value
#[allow(unused)]
pub fn formfactor_matrix(
    strucure: &Structure,
    formfactors: HashMap<AtomKind, FormFactor>,
    q_values: &[f64],
) -> Vec<nalgebra::DMatrix<f64>> {
    for id in strucure.kinds.iter() {
        // let formfactor = formfactors.get(id).unwrap();
    }
    Vec::new()
}

// Calculate distance matrix and store in N x N matrix
pub fn distance_matrix(positions: &[Vector3]) -> nalgebra::DMatrix<f64> {
    let n = positions.len();
    let mut matrix = nalgebra::DMatrix::<f64>::zeros(n, n);
    for i in 0..n - 1 {
        for j in i + 1..n {
            let r = (&positions[i] - &positions[j]).norm();
            matrix[(i, j)] = r;
            matrix[(j, i)] = r;
        }
    }
    matrix
}

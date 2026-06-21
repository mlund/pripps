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

//! Effective structure factor S_eff(q) and the decoupling parameter β(q).
//!
//! For a concentrated solution of identical particles the measured intensity
//! factorizes (Kotlarchyk & Chen 1983, <https://doi.org/10.1063/1.446055>) as
//!
//! ```text
//!     I(q) = N · P(q) · [ 1 + β(q) · (S(q) − 1) ]
//! ```
//!
//! where `P(q) = ⟨|F(q)|²⟩_Ω` is the single-particle form factor, `S(q)` the
//! centre–centre structure factor, and
//!
//! ```text
//!     β(q) = |⟨F(q)⟩_Ω|² / ⟨|F(q)|²⟩_Ω
//! ```
//!
//! the decoupling parameter (the fraction of the inter-particle structure
//! factor that survives orientational averaging). pripps emits β(q) and the
//! measured `S_eff(q) = I(q)/(N·P(q))`; inverting to the true `S(q) = 1 +
//! (S_eff−1)/β` is ill-conditioned near β nodes (⟨F⟩ → 0) and is left to the
//! caller, who can mask the nodes with whatever floor suits the data.
//!
//! `⟨F⟩_Ω` and `P(q)` are properties of a *single* monomer molecule. Unlike
//! the box intensity — which is orientationally averaged by the physical
//! ensemble of many randomly oriented copies — a single molecule has no
//! ensemble, so its orientational averages are computed analytically: `P(q)`
//! via the exact Debye sum and `⟨F⟩_Ω = Σ_i f_i·sinc(q|r_i − c|)` about the
//! molecule centroid `c`. (Sampling one fixed orientation along discrete
//! q-directions would *not* average over orientation.)

use crate::debye::{self_intensity_at_q, sinc};
use crate::explicit::eval_atom_form_factors;
use crate::formfactor::FormFactorMap;
use crate::{Result, Structure, StructureFactor, Vector3};
use rayon::prelude::*;

/// Geometric centroid of a set of positions.
fn centroid(positions: &[Vector3]) -> Vector3 {
    if positions.is_empty() {
        return Vector3::zeros();
    }
    positions.iter().sum::<Vector3>() / positions.len() as f64
}

/// Compute the decoupling parameter β(q) and the effective structure factor
/// `S_eff(q) = I(q)/(N·P(q))` from a `monomer` whole molecule, the box
/// intensity `box_intensity`, and the molecule count `n`. `box_intensity` and
/// `q_values` are parallel to the output curves.
pub(crate) fn from_monomer(
    monomer: &Structure,
    formfactors: &FormFactorMap,
    box_intensity: &[f64],
    q_values: &[f64],
    n: usize,
) -> Result<StructureFactor> {
    // Distances from the centroid are q-independent, so compute them once.
    let center = centroid(&monomer.pos);
    let distances: Vec<f64> = monomer.pos.iter().map(|r| (r - center).norm()).collect();
    let n_f = n as f64;

    // Per q, combine the orientationally averaged amplitude ⟨F⟩ (sinc) and the
    // single-molecule form factor P(q) = ⟨|F|²⟩ (exact Debye sum) into β and
    // S_eff in one parallel pass over the (q, box intensity) grid.
    let (beta, effective_structure_factor): (Vec<f64>, Vec<f64>) = q_values
        .par_iter()
        .zip(box_intensity.par_iter())
        .map(|(&q, &i)| {
            let ff = eval_atom_form_factors(q, &monomer.ids, formfactors)?;
            let mean_f: f64 = distances
                .iter()
                .zip(&ff)
                .map(|(&d, &f)| f * sinc(q * d))
                .sum();
            let p = self_intensity_at_q(&monomer.pos, &ff, q);
            let beta = if p > 0.0 { mean_f * mean_f / p } else { 1.0 };
            let s_eff = if p > 0.0 { i / (n_f * p) } else { 1.0 };
            Ok((beta, s_eff))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip();

    Ok(StructureFactor {
        n_molecules: n,
        beta,
        effective_structure_factor,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AtomKind;
    use crate::formfactor::FormFactor;
    use approx::assert_relative_eq;

    fn dumbbell() -> (FormFactorMap, Structure) {
        let ff = FormFactorMap::from_entries(vec![(AtomKind::new("X"), FormFactor::Constant(1.0))]);
        let monomer = Structure {
            pos: vec![Vector3::new(0.0, 0.0, -3.0), Vector3::new(0.0, 0.0, 3.0)],
            ids: vec![0, 0],
            box_sides: None,
        };
        (ff, monomer)
    }

    #[test]
    fn beta_is_one_at_low_q_and_decays() {
        let (ff, monomer) = dumbbell();
        let q = vec![1e-4, 0.1, 0.4, 1.0];
        // N = 1, box intensity = P (single molecule) so S_eff = 1.
        let p: Vec<f64> = q
            .iter()
            .map(|&q| self_intensity_at_q(&monomer.pos, &[1.0, 1.0], q))
            .collect();
        let dec = from_monomer(&monomer, &ff, &p, &q, 1).unwrap();

        assert_relative_eq!(dec.beta[0], 1.0, epsilon = 1e-6);
        assert!(dec.beta.iter().all(|&b| b <= 1.0 + 1e-9));
        assert!(dec.beta.windows(2).all(|w| w[1] <= w[0] + 1e-12));
        // Box == single monomer ⇒ S_eff ≡ 1.
        for &s_eff in &dec.effective_structure_factor {
            assert_relative_eq!(s_eff, 1.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn s_eff_scales_with_box_intensity() {
        let (ff, monomer) = dumbbell();
        let q = [0.1];
        let p = [self_intensity_at_q(&monomer.pos, &[1.0, 1.0], q[0])];
        // Box intensity 5 % above N·P ⇒ S_eff = 1.05 (N = 1).
        let dec = from_monomer(&monomer, &ff, &[p[0] * 1.05], &q, 1).unwrap();
        assert_relative_eq!(dec.effective_structure_factor[0], 1.05, epsilon = 1e-9);
    }
}

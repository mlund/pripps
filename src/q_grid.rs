//! Which q-vectors a periodic box is sampled at.
//!
//! Only reciprocal-lattice vectors q = 2π(h/Lx, k/Ly, l/Lz) give
//! PBC-invariant amplitudes. The lattice grid uses all of them up to q_max,
//! binned in |q|, so each bin averages many directions instead of the 13
//! Miller directions' 3–6 vectors.

use crate::{Error, Result};
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// q-vectors to evaluate, and how their intensities are grouped into |q|.
#[derive(Clone, Debug)]
pub struct QGrid {
    kind: Kind,
}

#[derive(Clone, Debug)]
enum Kind {
    /// 13 low-index directions × p = 1..=p_max, grouped by exact |q| at
    /// evaluation time, so any box works.
    Miller {
        p_max: usize,
    },
    Lattice(Lattice),
}

#[derive(Clone, Debug)]
struct Lattice {
    box_sides: [f64; 3],
    /// Kept vectors (one of each ±n pair), bin after bin.
    vectors: Vec<[i32; 3]>,
    /// Bin of each kept vector.
    bins: Vec<usize>,
    /// Mean |q| over all of a bin's vectors, kept or not.
    q: Vec<f64>,
}

impl QGrid {
    /// Every lattice vector with |q| ≤ `q_max` (Å⁻¹) in bins of width
    /// 2π/max(L); bins with more than `max_per_bin` vectors keep an evenly
    /// spread subset of that size.
    pub fn lattice(box_sides: [f64; 3], q_max: f64, max_per_bin: usize) -> Result<Self> {
        if !box_sides.iter().all(|&l| l.is_finite() && l > 0.0) {
            return Err(Error::usage(format!("invalid box sides {box_sides:?}")));
        }
        let mut shells = shells(box_sides, q_max);
        if shells.is_empty() {
            return Err(Error::usage(format!(
                "q_max {q_max} Å⁻¹ is below the shortest lattice vector of box {box_sides:?}"
            )));
        }
        let mut lattice = Lattice {
            box_sides,
            vectors: Vec::new(),
            bins: Vec::new(),
            q: Vec::new(),
        };
        for (bin, shell) in shells.iter_mut().enumerate() {
            lattice
                .q
                .push(shell.iter().map(|v| v.q).sum::<f64>() / shell.len() as f64);
            for v in spread_subset(shell, max_per_bin.max(1)) {
                lattice.vectors.push(v.n);
                lattice.bins.push(bin);
            }
        }
        Ok(Self {
            kind: Kind::Lattice(lattice),
        })
    }

    /// The 13 [100], [110] and [111] directions at p = 1..=`p_max`: few
    /// vectors per |q|, but valid for any box.
    pub fn miller(p_max: usize) -> Self {
        Self {
            kind: Kind::Miller { p_max },
        }
    }

    /// Bin q-values (Å⁻¹), ascending; empty for Miller grids, whose |q|
    /// follow from the box at evaluation.
    pub fn q(&self) -> &[f64] {
        match &self.kind {
            Kind::Miller { .. } => &[],
            Kind::Lattice(lattice) => &lattice.q,
        }
    }

    /// Vectors evaluated per bin; they set each bin's sampling noise.
    /// Empty for Miller grids.
    pub fn vectors_per_bin(&self) -> Vec<usize> {
        match &self.kind {
            Kind::Miller { .. } => vec![],
            Kind::Lattice(lattice) => {
                let mut counts = vec![0; lattice.q.len()];
                lattice.bins.iter().for_each(|&b| counts[b] += 1);
                counts
            }
        }
    }
}

impl QGrid {
    /// Integer lattice indices n of the q-vectors, q = 2π(n_x/L_x, …).
    pub(crate) fn indices(&self) -> Vec<[i32; 3]> {
        match &self.kind {
            Kind::Miller { p_max } => crate::explicit::miller_indices(*p_max),
            Kind::Lattice(lattice) => lattice.vectors.clone(),
        }
    }

    /// A lattice grid's bins assume its box; a Miller grid fits any.
    pub(crate) fn check_box(&self, box_sides: [f64; 3]) -> Result<()> {
        let Kind::Lattice(lattice) = &self.kind else {
            return Ok(());
        };
        // Tolerant, as the same box may be recomputed or widened from f32.
        let same = (0..3)
            .all(|a| (box_sides[a] - lattice.box_sides[a]).abs() <= 1e-9 * lattice.box_sides[a]);
        if same {
            return Ok(());
        }
        Err(Error::usage(format!(
            "box {box_sides:?} differs from the {:?} the q-grid was built for",
            lattice.box_sides
        )))
    }

    /// Mean of per-vector `rows` (|q|, values), in `directions()` order,
    /// per |q| group.
    pub(crate) fn group<const C: usize>(&self, rows: Vec<(f64, [f64; C])>) -> Vec<(f64, [f64; C])> {
        let Kind::Lattice(lattice) = &self.kind else {
            return crate::explicit::average_by_qnorm(rows);
        };
        assert_eq!(rows.len(), lattice.bins.len(), "one row per q-vector");
        let mut sums = vec![[0.0; C]; lattice.q.len()];
        for (&bin, (_, values)) in lattice.bins.iter().zip(&rows) {
            sums[bin].iter_mut().zip(values).for_each(|(s, v)| *s += v);
        }
        let counts = self.vectors_per_bin();
        lattice
            .q
            .iter()
            .zip(sums)
            .zip(counts)
            .map(|((&q, sum), n)| (q, sum.map(|s| s / n as f64)))
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
struct LatticeVector {
    n: [i32; 3],
    q: f64,
}

/// Lattice vectors with |q| ≤ q_max, one of each ±n pair (I(q) = I(−q)),
/// grouped into non-empty |q| bins of width 2π/max(L).
fn shells(box_sides: [f64; 3], q_max: f64) -> Vec<Vec<LatticeVector>> {
    let width = 2.0 * PI / box_sides.iter().cloned().fold(0.0, f64::max);
    // Same slack as the |q| test below, so a vector at exactly q_max counts.
    let n_max = box_sides.map(|l| (q_max * (1.0 + 1e-12) * l / (2.0 * PI)).floor() as i32);
    let mut bins: BTreeMap<i64, Vec<LatticeVector>> = BTreeMap::new();
    for h in -n_max[0]..=n_max[0] {
        for k in -n_max[1]..=n_max[1] {
            for l in -n_max[2]..=n_max[2] {
                let n = [h, k, l];
                if !upper_half(n) {
                    continue;
                }
                let q = (0..3)
                    .map(|a| (2.0 * PI * n[a] as f64 / box_sides[a]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if q <= q_max * (1.0 + 1e-12) {
                    let bin = (q / width).round() as i64;
                    bins.entry(bin).or_default().push(LatticeVector { n, q });
                }
            }
        }
    }
    bins.into_values().collect()
}

/// First non-zero component positive: one representative of each ±n pair.
fn upper_half([h, k, l]: [i32; 3]) -> bool {
    h > 0 || (h == 0 && (k > 0 || (k == 0 && l > 0)))
}

/// All vectors if they fit, else `cap` of them at an even stride through
/// the bin sorted by direction, so the subset still spans the sphere.
fn spread_subset(shell: &mut [LatticeVector], cap: usize) -> Vec<LatticeVector> {
    if shell.len() <= cap {
        return shell.to_vec();
    }
    shell.sort_by(|a, b| direction_key(a).total_cmp(&direction_key(b)));
    (0..cap).map(|i| shell[i * shell.len() / cap]).collect()
}

/// Orders directions by polar band, then azimuth within it.
fn direction_key(v: &LatticeVector) -> f64 {
    let [x, y, z] = v.n.map(f64::from);
    let norm = (x * x + y * y + z * z).sqrt();
    let band = ((z / norm + 1.0) * 8.0).floor();
    band * 10.0 + y.atan2(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::SQRT_2;

    const L: f64 = 405.0;

    #[test]
    fn first_bin_holds_the_unit_and_face_diagonal_vectors() {
        // |n| = 1: 3 vectors; |n| = √2: 6 (one per ± pair); both round to bin 1.
        let grid = QGrid::lattice([L; 3], 0.023, usize::MAX).unwrap();
        assert_eq!(grid.vectors_per_bin(), vec![9]);
        let expected = 2.0 * PI / L * (3.0 + 6.0 * SQRT_2) / 9.0;
        approx::assert_relative_eq!(grid.q()[0], expected, max_relative = 1e-12);
    }

    #[test]
    fn bin_counts_match_a_brute_force_count() {
        // Independent count (Python, all integer triples, halved for ±n).
        let grid = QGrid::lattice([L; 3], 0.5, usize::MAX).unwrap();
        assert_eq!(
            grid.vectors_per_bin()[..8],
            [9, 31, 49, 105, 175, 225, 301, 381]
        );
    }

    #[test]
    fn the_cap_thins_large_bins_only_and_keeps_their_q() {
        let full = QGrid::lattice([L; 3], 0.5, usize::MAX).unwrap();
        let capped = QGrid::lattice([L; 3], 0.5, 100).unwrap();
        for (&all, &kept) in full.vectors_per_bin().iter().zip(&capped.vectors_per_bin()) {
            assert_eq!(kept, all.min(100));
        }
        assert_eq!(full.q(), capped.q());
    }

    /// ⟨n̂n̂ᵀ⟩ = I/3 for directions spread over the sphere.
    #[allow(clippy::needless_range_loop)] // a tensor reads best by index
    #[test]
    fn a_capped_bin_still_spans_the_sphere() {
        let grid = QGrid::lattice([L; 3], 0.5, 50).unwrap();
        let Kind::Lattice(lattice) = &grid.kind else {
            unreachable!()
        };
        let last = lattice.q.len() - 1;
        let mut tensor = [[0.0; 3]; 3];
        let picked: Vec<_> = (0..lattice.vectors.len())
            .filter(|&i| lattice.bins[i] == last)
            .map(|i| lattice.vectors[i].map(f64::from))
            .collect();
        for v in &picked {
            let norm2: f64 = v.iter().map(|x| x * x).sum();
            for a in 0..3 {
                for b in 0..3 {
                    tensor[a][b] += v[a] * v[b] / norm2 / picked.len() as f64;
                }
            }
        }
        // Only one of each ±n pair is kept, which leaves n̂n̂ᵀ unchanged.
        for a in 0..3 {
            for b in 0..3 {
                let expected = if a == b { 1.0 / 3.0 } else { 0.0 };
                approx::assert_abs_diff_eq!(tensor[a][b], expected, epsilon = 0.03);
            }
        }
    }

    #[test]
    fn impossible_grids_are_errors() {
        assert!(QGrid::lattice([L; 3], 0.01, 10).is_err()); // below 2π/L
        assert!(QGrid::lattice([L, 0.0, L], 0.5, 10).is_err());
        assert!(QGrid::lattice([L, f64::NAN, L], 0.5, 10).is_err());
    }
}

// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Criterion benchmarks comparing Debye vs Multipole intensity
//! calculators across a range of structure sizes.
//!
//! Tests go through the public `Pripps::intensity` API, so the
//! measured time includes file parsing, calculator construction
//! (including `SphericalBasis::build` for Multipole), and the actual
//! compute. This is what a library consumer actually pays per call —
//! a pure kernel benchmark would need a different entry point.
//!
//! Size picks: 100 / 500 / 5000 synthetic structures on a cubic
//! lattice (lives in tempfiles built once at bench startup). The
//! theoretical Debye-vs-Multipole crossover is around N ≈ 500 for
//! typical l_max; in practice Multipole is already ahead at N = 100
//! because the setup cost is dominated by the per-frame Debye pair
//! sum even at that size.
//!
//! Run:
//! ```sh
//! cargo bench --bench intensity
//! ```

use criterion::{BenchmarkId, Criterion, SamplingMode, criterion_group, criterion_main};
use pripps::{IntensityScheme, Pripps};
use std::hint::black_box;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Duration;
use tempfile::{NamedTempFile, TempDir};

/// Minimal YAML form factor file covering the five element types
/// used by the Pedersen/PepsiSAXS atomic PDBs (`N C O S H`). Each
/// element gets a constant form factor so the bench measures the
/// geometric and multipole kernels instead of per-atom form-factor
/// evaluation cost — `FormFactor::eval` would otherwise be a
/// sizeable fraction of total time for Gaussian or Table variants.
///
/// We include elements we don't use in the synthetic fixtures (`N`,
/// `O`, `S`, `H`) so the same YAML can later be pointed at real
/// atomic PDBs without needing a second config file.
const BENCH_FF_YAML: &str = "\
N: { formfactor: !Constant 1.0 }
C: { formfactor: !Constant 1.0 }
O: { formfactor: !Constant 1.0 }
S: { formfactor: !Constant 1.0 }
H: { formfactor: !Constant 1.0 }
";

/// Q-range parameters tuned for a realistic fitting workflow: 50
/// points from 0 to 0.5 Å⁻¹. Matches what users pass through the
/// CLI's `debye --qmax 0.5 --nq 50` shape. Enough q-points to
/// exercise the per-q rayon parallelism without making the inner
/// loop so long that Debye's O(N²·Q) at N=5000 becomes minutes per
/// iteration.
const QMIN: f64 = 0.0;
const QMAX: f64 = 0.5;
const NQ: usize = 50;

/// Build a synthetic PDB structure of `n` atoms on a 3-Å cubic
/// lattice and write it to `dir/<name>.pdb`.
///
/// Deterministic layout (vs random positions) makes bench runs
/// reproducible run-to-run — a single timing outlier is a real
/// perturbation, not noise from different positions.
///
/// 3 Å spacing matches the scale of real proteins (Cα–Cα ≈ 3.8 Å)
/// so `r_max` and the Multipole criterion `l_max ≈ q_max · r_max`
/// land in a realistic range (l_max ≈ 10 at N=100, ≈ 25 at N=500,
/// ≈ 50 at N=5000).
///
/// All atoms are carbon — one of the five element types in
/// [`BENCH_FF_YAML`] so the form-factor lookup succeeds.
fn write_synthetic_pdb(dir: &Path, name: &str, n: usize) -> PathBuf {
    let path = dir.join(format!("{name}.pdb"));
    let mut file = std::fs::File::create(&path).unwrap();

    writeln!(file, "HEADER    SYNTHETIC BENCH FIXTURE").unwrap();
    let side = (n as f64).cbrt().ceil() as usize;
    let mut written = 0usize;
    'outer: for i in 0..side {
        for j in 0..side {
            for k in 0..side {
                if written == n {
                    break 'outer;
                }
                let x = i as f64 * 3.0;
                let y = j as f64 * 3.0;
                let z = k as f64 * 3.0;
                // Fixed-width PDB ATOM record. Element column is C.
                writeln!(
                    file,
                    "ATOM  {serial:5}  C   ALA A{res:4}    {x:8.3}{y:8.3}{z:8.3}  1.00  0.00           C",
                    serial = written + 1,
                    res = (written % 9999) + 1,
                )
                .unwrap();
                written += 1;
            }
        }
    }
    writeln!(file, "END").unwrap();
    path
}

/// Write the form-factor YAML into a tempfile and return it. The
/// file is kept alive by the returned `NamedTempFile` — dropping it
/// unlinks the path.
fn write_yaml() -> NamedTempFile {
    let mut yaml = NamedTempFile::new().unwrap();
    yaml.write_all(BENCH_FF_YAML.as_bytes()).unwrap();
    yaml.flush().unwrap();
    yaml
}

fn bench_intensity(c: &mut Criterion) {
    let yaml = write_yaml();
    let pripps = Pripps::from_yaml(yaml.path()).unwrap();

    // Synthetic fixtures live in one tempdir for the whole bench run.
    // `TempDir` cleans up on drop; files stay alive until the end of
    // `bench_intensity`.
    let fixtures = TempDir::new().unwrap();
    let synthetic: Vec<(usize, PathBuf)> = [100usize, 500, 5000]
        .iter()
        .map(|&n| (n, write_synthetic_pdb(fixtures.path(), &format!("n{n}"), n)))
        .collect();

    let mut group = c.benchmark_group("intensity");

    // Flat sampling: one iteration per sample, summing the raw
    // timings without the parabolic regression criterion uses by
    // default. Right for long per-iter benches where fitting 10+
    // iterations per sample (criterion's default) would blow past
    // the measurement window.
    group.sampling_mode(SamplingMode::Flat);

    // 20 samples is a compromise: default is 100, which would push
    // the n=5000 Debye bench to ~8 minutes per run. 20 gives a
    // tighter confidence interval than 10 without being painful to
    // run repeatedly.
    group.sample_size(20);

    // 12-second target covers the slowest single iteration (n=5000
    // Debye is ~5 s) so criterion doesn't warn about overflow. The
    // total bench time is still sample_size × per_iter regardless.
    group.measurement_time(Duration::from_secs(12));

    for (n, pdb) in &synthetic {
        let id = format!("n{n}");

        group.bench_function(BenchmarkId::new("debye", &id), |b| {
            b.iter(|| {
                let scheme = IntensityScheme::Debye {
                    qmin: QMIN,
                    qmax: QMAX,
                    nq: NQ,
                };
                // black_box: prevents LLVM from eliminating the call
                // because the returned `Intensity` is otherwise
                // unused. Without this the optimizer could realize
                // that each iteration is idempotent and skip most of
                // the work.
                black_box(
                    pripps
                        .intensity(scheme, pdb, None, None, None, None)
                        .unwrap(),
                );
            });
        });

        group.bench_function(BenchmarkId::new("multipole", &id), |b| {
            b.iter(|| {
                let scheme = IntensityScheme::Multipole {
                    qmin: QMIN,
                    qmax: QMAX,
                    nq: Some(NQ),
                    // None = auto-select l_max from structure extent,
                    // matching the default CLI behaviour.
                    l_max: None,
                };
                black_box(
                    pripps
                        .intensity(scheme, pdb, None, None, None, None)
                        .unwrap(),
                );
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_intensity);
criterion_main!(benches);

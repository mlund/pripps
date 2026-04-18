// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Fitting against experimental SAXS data.
//!
//! Provides χ²/N computation with analytical linear scale factor,
//! experimental data parsing, and L-BFGS-B minimisation over the
//! (volume_scale, contrast_density) fit parameters.

use crate::error::IoResultExt;
use crate::{Intensity, Result};
use std::io::BufRead;
use std::path::Path;

/// Experimental SAXS data: q, I(q), σ(q).
#[derive(Debug, Clone)]
pub struct ExpData {
    pub q: Vec<f64>,
    pub intensity: Vec<f64>,
    pub sigma: Vec<f64>,
}

impl ExpData {
    /// Load from a file. Auto-detects CSV (comma-separated) vs
    /// whitespace-separated. Skips comment lines starting with
    /// `#` or alphabetic characters, and blank lines.
    pub fn from_file(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).with_path(path)?;
        let reader = std::io::BufReader::new(file);

        let mut q = Vec::new();
        let mut intensity = Vec::new();
        let mut sigma = Vec::new();

        for line in reader.lines() {
            let line = line.with_path(path)?;
            let trimmed = line.trim();
            if trimmed.is_empty()
                || trimmed.starts_with('#')
                || trimmed.starts_with(|c: char| c.is_alphabetic())
            {
                continue;
            }
            // Auto-detect delimiter: comma → CSV, otherwise whitespace
            let parts: Vec<&str> = if trimmed.contains(',') {
                trimmed.split(',').map(str::trim).collect()
            } else {
                trimmed.split_whitespace().collect()
            };
            if parts.len() < 2 {
                continue;
            }
            let (Ok(qi), Ok(ii)) = (parts[0].parse::<f64>(), parts[1].parse::<f64>()) else {
                continue;
            };
            let si = parts
                .get(2)
                .and_then(|s| s.parse::<f64>().ok())
                .unwrap_or(1.0);
            q.push(qi);
            intensity.push(ii);
            sigma.push(si);
        }

        if q.is_empty() {
            return Err(crate::Error::validation(format!(
                "no data points found in {} (expected 2–3 columns: q, I [, σ]; \
                 whitespace or comma separated)",
                path.display()
            )));
        }

        // σ=1 fallback means unweighted χ² — warn the user since
        // fit quality metrics won't be comparable to weighted fits.
        if sigma.iter().all(|&s| s == 1.0) {
            log::warn!(
                "No σ column found in {}; using σ=1 (unweighted χ²)",
                path.display()
            );
        }

        log::info!(
            "Loaded {} experimental data points from {} (q = {:.4}–{:.4} Å⁻¹)",
            q.len(),
            path.display(),
            q.first().unwrap(),
            q.last().unwrap()
        );

        Ok(Self {
            q,
            intensity,
            sigma,
        })
    }

    /// Maximum q value in the experimental data.
    pub fn q_max(&self) -> f64 {
        self.q.iter().copied().fold(0.0_f64, f64::max)
    }
}

/// Linear interpolation of `(xs, ys)` at point `x`.
pub fn interpolate_at(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    if x <= xs[0] {
        return ys[0];
    }
    if x >= *xs.last().unwrap() {
        return *ys.last().unwrap();
    }
    let i = xs.partition_point(|&v| v < x);
    if i == 0 {
        return ys[0];
    }
    let (x0, x1) = (xs[i - 1], xs[i]);
    let (y0, y1) = (ys[i - 1], ys[i]);
    y0 + (x - x0) * (y1 - y0) / (x1 - x0)
}

/// Optimal linear scale factor between theory and experiment:
/// `scale = Σ(I_th · I_exp / σ²) / Σ(I_th² / σ²)`.
pub fn compute_scale(theory: &Intensity, exp: &ExpData) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for i in 0..exp.q.len() {
        let i_th = interpolate_at(&theory.q, &theory.total, exp.q[i]);
        let s2 = exp.sigma[i] * exp.sigma[i];
        num += i_th * exp.intensity[i] / s2;
        den += i_th * i_th / s2;
    }
    num / den
}

/// Compute χ²/N between a theoretical curve and experimental data,
/// with the optimal linear scale factor determined analytically.
/// Caches interpolated values to avoid redundant lookups.
pub fn chi2_per_point(theory: &Intensity, exp: &ExpData) -> f64 {
    let n = exp.q.len() as f64;
    // Interpolate once and cache: the scale factor and χ² both
    // need the same interpolated values, so computing them up
    // front halves the number of binary searches per evaluation.
    let interpolated: Vec<f64> = exp
        .q
        .iter()
        .map(|&qi| interpolate_at(&theory.q, &theory.total, qi))
        .collect();

    let (num, den) = interpolated
        .iter()
        .zip(&exp.intensity)
        .zip(&exp.sigma)
        .fold((0.0, 0.0), |(num, den), ((&i_th, &i_exp), &sigma)| {
            let s2 = sigma * sigma;
            (num + i_th * i_exp / s2, den + i_th * i_th / s2)
        });
    let scale = num / den;

    let chi2: f64 = interpolated
        .iter()
        .zip(&exp.intensity)
        .zip(&exp.sigma)
        .map(|((&i_th, &i_exp), &sigma)| {
            let residual = (scale * i_th - i_exp) / sigma;
            residual * residual
        })
        .sum();
    chi2 / n
}

/// Result of a fit against experimental data.
#[derive(Debug, Clone)]
pub struct FitResult {
    pub volume_scale: f64,
    pub contrast_density: f64,
    pub chi2: f64,
    pub scale: f64,
}

/// Nelder-Mead simplex minimisation for 2 parameters.
/// Returns `(x, y, f(x, y))` at the best vertex.
///
/// Hand-rolled to avoid pulling in argmin + transitive deps
/// (rand, nalgebra 0.34, etc.) for a 2-parameter problem where
/// the simplex algorithm is ~40 lines.
fn nelder_mead_2d(cost: impl Fn(f64, f64) -> f64, initial: [(f64, f64); 3]) -> (f64, f64, f64) {
    const MAX_ITER: usize = 500;
    const TOLERANCE: f64 = 1e-10;

    let mut simplex: [(f64, f64, f64); 3] = initial.map(|(x, y)| (x, y, cost(x, y)));

    for _ in 0..MAX_ITER {
        // Sort: simplex[0] = best, simplex[2] = worst
        simplex.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        // Convergence: spread of costs across the simplex
        if simplex[2].2 - simplex[0].2 < TOLERANCE {
            break;
        }

        // Centroid of the two best vertices
        let cx = 0.5 * (simplex[0].0 + simplex[1].0);
        let cy = 0.5 * (simplex[0].1 + simplex[1].1);

        // Reflect the worst vertex through the centroid
        let rx = 2.0 * cx - simplex[2].0;
        let ry = 2.0 * cy - simplex[2].1;
        let fr = cost(rx, ry);

        if fr < simplex[0].2 {
            // Reflected is best so far — try expanding further
            let ex = 3.0 * cx - 2.0 * simplex[2].0;
            let ey = 3.0 * cy - 2.0 * simplex[2].1;
            let fe = cost(ex, ey);
            simplex[2] = if fe < fr { (ex, ey, fe) } else { (rx, ry, fr) };
        } else if fr < simplex[1].2 {
            // Reflected is better than second-worst — accept
            simplex[2] = (rx, ry, fr);
        } else {
            // Contract: move worst vertex toward centroid
            let kx = 0.5 * (cx + simplex[2].0);
            let ky = 0.5 * (cy + simplex[2].1);
            let fk = cost(kx, ky);
            if fk < simplex[2].2 {
                simplex[2] = (kx, ky, fk);
            } else {
                // Shrink: move all vertices toward the best
                for i in 1..3 {
                    simplex[i].0 = 0.5 * (simplex[i].0 + simplex[0].0);
                    simplex[i].1 = 0.5 * (simplex[i].1 + simplex[0].1);
                    simplex[i].2 = cost(simplex[i].0, simplex[i].1);
                }
            }
        }

        log::debug!(
            "Nelder-Mead: χ²/N = {:.6}, c₁ = {:.4}, c₂ = {:.4}",
            simplex[0].2,
            simplex[0].0,
            simplex[0].1
        );
    }

    simplex.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    (simplex[0].0, simplex[0].1, simplex[0].2)
}

/// Fit (volume_scale, contrast_density) against experimental data
/// using Nelder-Mead simplex. The `eval` closure takes (c1, c2)
/// and returns a theoretical `Intensity`.
pub fn fit_parameters<F>(exp: &ExpData, eval: F) -> Result<FitResult>
where
    F: Fn(f64, f64) -> Intensity,
{
    let cost = |c1: f64, c2: f64| chi2_per_point(&eval(c1, c2), exp);

    log::info!(
        "Fit: initial χ²/N = {:.4} at c₁=1.0, c₂=0.03",
        cost(1.0, 0.03)
    );

    let (c1, c2, chi2) = nelder_mead_2d(cost, [(1.0, 0.03), (1.05, 0.03), (1.0, 0.10)]);

    let theory = eval(c1, c2);
    let scale = compute_scale(&theory, exp);

    log::info!("Fit: χ²/N = {chi2:.4}, c₁ = {c1:.4}, c₂ = {c2:.4}, scale = {scale:.6e}");

    Ok(FitResult {
        volume_scale: c1,
        contrast_density: c2,
        chi2,
        scale,
    })
}

/// Write the fitted curve as CSV: q, I_exp, I_fit, sigma.
pub fn write_fit_csv(path: &Path, exp: &ExpData, theory: &Intensity, scale: f64) -> Result<()> {
    use std::io::{BufWriter, Write};
    let file = std::fs::File::create(path).with_path(path)?;
    let mut out = BufWriter::new(file);
    log::info!("Writing fitted curve to {}", path.display());
    writeln!(out, "q,i_exp,i_fit,sigma")?;
    for i in 0..exp.q.len() {
        let i_fit = scale * interpolate_at(&theory.q, &theory.total, exp.q[i]);
        writeln!(
            out,
            "{:.6},{:.6},{:.6},{:.6}",
            exp.q[i], exp.intensity[i], i_fit, exp.sigma[i]
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interpolate_at_boundaries() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![10.0, 20.0, 30.0];
        assert_eq!(interpolate_at(&xs, &ys, -1.0), 10.0);
        assert_eq!(interpolate_at(&xs, &ys, 3.0), 30.0);
        assert_eq!(interpolate_at(&xs, &ys, 0.5), 15.0);
        assert_eq!(interpolate_at(&xs, &ys, 1.5), 25.0);
    }

    #[test]
    fn load_lyzexp() {
        let exp = ExpData::from_file(std::path::Path::new("tests/lysozyme/lyzexp.dat")).unwrap();
        assert!(exp.q.len() > 100);
        assert!(exp.q[0] > 0.0);
        assert!(exp.intensity[0] > 0.0);
        assert!(exp.sigma[0] > 0.0);
    }
}

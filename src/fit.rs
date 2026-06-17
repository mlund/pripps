// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Fitting against experimental SAXS data.
//!
//! Provides χ²/N computation with an analytical linear scale factor,
//! experimental data parsing, and box-constrained Nelder-Mead
//! minimisation over the (volume_scale, contrast_density) fit
//! parameters.

use crate::error::IoResultExt;
use crate::{Intensity, Result};
use std::io::BufRead;
use std::path::Path;

/// Experimental SAXS data: q, I(q), σ(q).
#[derive(Debug, Clone)]
pub struct ExpData {
    /// Scattering vector magnitudes q (Å⁻¹).
    pub q: Vec<f64>,
    /// Measured scattering intensity I(q) at each q.
    pub intensity: Vec<f64>,
    /// Experimental uncertainty σ(q) on each intensity (σ=1 if absent).
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

/// Natural cubic spline on a borrowed `(xs, ys)` table.
/// Second-derivative recurrence follows Numerical Recipes §3.3.
#[derive(Debug, Clone)]
pub(crate) struct CubicSpline<'a> {
    xs: &'a [f64],
    ys: &'a [f64],
    y2: Vec<f64>,
}

impl<'a> CubicSpline<'a> {
    pub(crate) fn new(xs: &'a [f64], ys: &'a [f64]) -> Self {
        let n = xs.len();
        let mut y2 = vec![0.0; n];
        if n < 3 {
            return Self { xs, ys, y2 };
        }
        let mut u = vec![0.0; n];
        for i in 1..n - 1 {
            let sig = (xs[i] - xs[i - 1]) / (xs[i + 1] - xs[i - 1]);
            let p = sig * y2[i - 1] + 2.0;
            y2[i] = (sig - 1.0) / p;
            u[i] = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])
                - (ys[i] - ys[i - 1]) / (xs[i] - xs[i - 1]);
            u[i] = (6.0 * u[i] / (xs[i + 1] - xs[i - 1]) - sig * u[i - 1]) / p;
        }
        for i in (0..n - 1).rev() {
            y2[i] = y2[i] * y2[i + 1] + u[i];
        }
        Self { xs, ys, y2 }
    }

    pub(crate) fn eval(&self, x: f64) -> f64 {
        if x <= self.xs[0] {
            return self.ys[0];
        }
        if x >= *self.xs.last().unwrap() {
            return *self.ys.last().unwrap();
        }
        let i = self.xs.partition_point(|&v| v < x);
        if i == 0 {
            return self.ys[0];
        }
        let (x0, x1) = (self.xs[i - 1], self.xs[i]);
        let (y0, y1) = (self.ys[i - 1], self.ys[i]);
        let (d0, d1) = (self.y2[i - 1], self.y2[i]);
        let h = x1 - x0;
        let a = (x1 - x) / h;
        let b = (x - x0) / h;
        a * y0 + b * y1 + ((a * a * a - a) * d0 + (b * b * b - b) * d1) * h * h / 6.0
    }
}

/// Optimal linear scale factor between theory and experiment:
/// `scale = Σ(I_th · I_exp / σ²) / Σ(I_th² / σ²)`.
pub fn compute_scale(theory: &Intensity, exp: &ExpData) -> f64 {
    let spline = CubicSpline::new(&theory.q, &theory.total);
    let (num, den) = exp.q.iter().zip(&exp.intensity).zip(&exp.sigma).fold(
        (0.0, 0.0),
        |(num, den), ((&q, &i_exp), &sigma)| {
            let i_th = spline.eval(q);
            let s2 = sigma * sigma;
            (num + i_th * i_exp / s2, den + i_th * i_th / s2)
        },
    );
    num / den
}

/// Compute χ²/N between a theoretical curve and experimental data,
/// with the optimal linear scale factor determined analytically.
pub fn chi2_per_point(theory: &Intensity, exp: &ExpData) -> f64 {
    let n = exp.q.len() as f64;
    let spline = CubicSpline::new(&theory.q, &theory.total);
    let interpolated: Vec<f64> = exp.q.iter().map(|&qi| spline.eval(qi)).collect();

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
    /// Fitted excluded-volume scaling factor c₁.
    pub volume_scale: f64,
    /// Fitted hydration-shell contrast density c₂.
    pub contrast_density: f64,
    /// Minimised χ²/N at the fitted parameters.
    pub chi2: f64,
    /// Optimal linear scale factor between theory and experiment.
    pub scale: f64,
}

/// Physical bounds on fit parameters. `c₂ ≤ 0.1` is ~10% fractional
/// excess density (pepsi's `d_rho = 0.0334 e/Å³ ÷ ρ_water`); `c₁`
/// is wider than pepsi's ±5% radius range so non-Fraser EV (voronoi,
/// grid) whose nominal volumes differ in scale still fit in-range.
const C1_BOUNDS: (f64, f64) = (0.5, 2.0);
const C2_BOUNDS: (f64, f64) = (0.0, 0.1);

/// Box-constrained Nelder-Mead: every trial vertex is clamped into
/// `C1_BOUNDS × C2_BOUNDS` before the cost evaluation.
fn nelder_mead_2d(cost: impl Fn(f64, f64) -> f64, initial: [(f64, f64); 3]) -> (f64, f64, f64) {
    const MAX_ITER: usize = 500;
    const TOLERANCE: f64 = 1e-10;

    let eval = |x: f64, y: f64| {
        let cx = x.clamp(C1_BOUNDS.0, C1_BOUNDS.1);
        let cy = y.clamp(C2_BOUNDS.0, C2_BOUNDS.1);
        (cx, cy, cost(cx, cy))
    };

    let mut simplex: [(f64, f64, f64); 3] = initial.map(|(x, y)| eval(x, y));

    for _ in 0..MAX_ITER {
        simplex.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

        if simplex[2].2 - simplex[0].2 < TOLERANCE {
            break;
        }

        let cx = 0.5 * (simplex[0].0 + simplex[1].0);
        let cy = 0.5 * (simplex[0].1 + simplex[1].1);

        let rx = 2.0 * cx - simplex[2].0;
        let ry = 2.0 * cy - simplex[2].1;
        let (rx, ry, fr) = eval(rx, ry);

        if fr < simplex[0].2 {
            let ex = 3.0 * cx - 2.0 * simplex[2].0;
            let ey = 3.0 * cy - 2.0 * simplex[2].1;
            let (ex, ey, fe) = eval(ex, ey);
            simplex[2] = if fe < fr { (ex, ey, fe) } else { (rx, ry, fr) };
        } else if fr < simplex[1].2 {
            simplex[2] = (rx, ry, fr);
        } else {
            let kx = 0.5 * (cx + simplex[2].0);
            let ky = 0.5 * (cy + simplex[2].1);
            let (kx, ky, fk) = eval(kx, ky);
            if fk < simplex[2].2 {
                simplex[2] = (kx, ky, fk);
            } else {
                for i in 1..3 {
                    let sx = 0.5 * (simplex[i].0 + simplex[0].0);
                    let sy = 0.5 * (simplex[i].1 + simplex[0].1);
                    simplex[i] = eval(sx, sy);
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
        "Fit: initial χ²/N = {:.4} at excluded volume (c₁) = 1.0, hydration shell (c₂) = 0.03 \
         (bounds: c₁ ∈ [{:.2}, {:.2}], c₂ ∈ [{:.3}, {:.4}])",
        cost(1.0, 0.03),
        C1_BOUNDS.0,
        C1_BOUNDS.1,
        C2_BOUNDS.0,
        C2_BOUNDS.1
    );

    let (c1, c2, chi2) = nelder_mead_2d(cost, [(1.0, 0.03), (1.08, 0.03), (1.0, 0.015)]);

    let theory = eval(c1, c2);
    let scale = compute_scale(&theory, exp);

    let on_bound = |v: f64, (lo, hi): (f64, f64)| v <= lo + 1e-6 || v >= hi - 1e-6;
    if on_bound(c1, C1_BOUNDS) || on_bound(c2, C2_BOUNDS) {
        log::warn!(
            "Fit hit bound: c₁ = {c1:.4} (∈ [{:.2}, {:.2}]), c₂ = {c2:.4} (∈ [{:.3}, {:.4}]). \
             The fit wanted to go further — check the structure and experimental data.",
            C1_BOUNDS.0,
            C1_BOUNDS.1,
            C2_BOUNDS.0,
            C2_BOUNDS.1
        );
    }

    log::info!(
        "Fit: χ²/N = {chi2:.4}, excluded volume (c₁) = {c1:.4}, \
         hydration shell (c₂) = {c2:.4}, scale = {scale:.6e}"
    );

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
    let spline = CubicSpline::new(&theory.q, &theory.total);
    for ((&q, &i_exp), &sigma) in exp.q.iter().zip(&exp.intensity).zip(&exp.sigma) {
        let i_fit = scale * spline.eval(q);
        writeln!(out, "{q:.6},{i_exp:.6},{i_fit:.6},{sigma:.6}")?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_spline_matches_linear_on_linear_data() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![10.0, 20.0, 30.0];
        let spline = CubicSpline::new(&xs, &ys);
        assert_eq!(spline.eval(-1.0), 10.0);
        assert_eq!(spline.eval(3.0), 30.0);
        assert_eq!(spline.eval(0.5), 15.0);
        assert_eq!(spline.eval(1.5), 25.0);
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

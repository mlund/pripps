// Copyright 2026 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Fitting against experimental SAXS data.
//!
//! Provides χ²/N computation with an analytical linear scale factor,
//! experimental data parsing, and bounded minimisation over the
//! (volume_scale, contrast_density) fit parameters.

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

    /// Build from in-memory arrays. `q` and `intensity` must be the
    /// same non-zero length; `sigma` is either empty (→ σ=1, unweighted
    /// χ²) or the same length as `q`.
    pub fn from_arrays(q: Vec<f64>, intensity: Vec<f64>, sigma: Vec<f64>) -> Result<Self> {
        if q.is_empty() {
            return Err(crate::Error::validation("experimental data is empty"));
        }
        if intensity.len() != q.len() {
            return Err(crate::Error::validation(format!(
                "q ({}) and intensity ({}) length mismatch",
                q.len(),
                intensity.len()
            )));
        }
        let sigma = if sigma.is_empty() {
            vec![1.0; q.len()]
        } else if sigma.len() == q.len() {
            sigma
        } else {
            return Err(crate::Error::validation(format!(
                "q ({}) and sigma ({}) length mismatch",
                q.len(),
                sigma.len()
            )));
        };
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

/// Default physical bounds for fitted excluded-volume scale `c₁`.
///
/// This is wider than pepsi's ±5% radius range so non-Fraser EV
/// (voronoi, grid) whose nominal volumes differ in scale still fit in-range.
pub const DEFAULT_C1_BOUNDS: (f64, f64) = (0.5, 2.0);

/// Default physical bounds for fitted hydration contrast `c₂`.
///
/// `c₂ ≤ 0.1` is roughly 10% fractional excess density, corresponding
/// to pepsi's `d_rho = 0.0334 e/Å³ ÷ ρ_water`.
pub const DEFAULT_C2_BOUNDS: (f64, f64) = (0.0, 0.1);

/// Box-constrained Nelder-Mead: every trial vertex is clamped into
/// the caller-supplied bounds before the cost evaluation.
fn nelder_mead_2d(
    cost: impl Fn(f64, f64) -> f64,
    initial: [(f64, f64); 3],
    c1_bounds: (f64, f64),
    c2_bounds: (f64, f64),
) -> (f64, f64, f64) {
    const MAX_ITER: usize = 500;
    const TOLERANCE: f64 = 1e-10;

    let eval = |x: f64, y: f64| {
        let cx = x.clamp(c1_bounds.0, c1_bounds.1);
        let cy = y.clamp(c2_bounds.0, c2_bounds.1);
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

fn bounded_offset(start: f64, bounds: (f64, f64), preferred_delta: f64) -> f64 {
    let delta = preferred_delta.abs().min((bounds.1 - bounds.0) / 2.0);
    if preferred_delta >= 0.0 {
        if start + delta <= bounds.1 {
            start + delta
        } else {
            start - delta
        }
    } else if start - delta >= bounds.0 {
        start - delta
    } else {
        start + delta
    }
}

fn initial_simplex(c1_bounds: (f64, f64), c2_bounds: (f64, f64)) -> [(f64, f64); 3] {
    let c1 = 1.0_f64.clamp(c1_bounds.0, c1_bounds.1);
    let c2 = 0.03_f64.clamp(c2_bounds.0, c2_bounds.1);
    [
        (c1, c2),
        (bounded_offset(c1, c1_bounds, 0.08), c2),
        (c1, bounded_offset(c2, c2_bounds, -0.015)),
    ]
}

/// Fit (volume_scale, contrast_density) against experimental data
/// using Nelder-Mead simplex. The `eval` closure takes (c1, c2)
/// and returns a theoretical `Intensity`.
pub fn fit_parameters<F>(exp: &ExpData, eval: F) -> Result<FitResult>
where
    F: Fn(f64, f64) -> Intensity,
{
    fit_parameters_with_bounds(exp, eval, DEFAULT_C1_BOUNDS, DEFAULT_C2_BOUNDS)
}

/// Fit (volume_scale, contrast_density) against experimental data
/// using Nelder-Mead simplex over caller-supplied bounds.
pub fn fit_parameters_with_bounds<F>(
    exp: &ExpData,
    eval: F,
    c1_bounds: (f64, f64),
    c2_bounds: (f64, f64),
) -> Result<FitResult>
where
    F: Fn(f64, f64) -> Intensity,
{
    validate_bounds("c1", c1_bounds)?;
    validate_bounds("c2", c2_bounds)?;

    let cost = |c1: f64, c2: f64| chi2_per_point(&eval(c1, c2), exp);
    let initial = initial_simplex(c1_bounds, c2_bounds);
    let (initial_c1, initial_c2) = initial[0];

    log::info!(
        "Fit: initial χ²/N = {:.4} at excluded volume (c₁) = {:.4}, hydration shell (c₂) = {:.4} \
         (bounds: c₁ ∈ [{:.2}, {:.2}], c₂ ∈ [{:.3}, {:.4}])",
        cost(initial_c1, initial_c2),
        initial_c1,
        initial_c2,
        c1_bounds.0,
        c1_bounds.1,
        c2_bounds.0,
        c2_bounds.1
    );

    let (c1, c2, chi2) = nelder_mead_2d(cost, initial, c1_bounds, c2_bounds);

    let theory = eval(c1, c2);
    let scale = compute_scale(&theory, exp);

    warn_if_on_bound(c1, c2, c1_bounds, c2_bounds);

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

/// Fit `(volume_scale, contrast_density)` against experimental data
/// by exhaustively evaluating an endpoint-inclusive rectangular grid
/// over the same physical bounds as [`fit_parameters`].
///
/// `c1_points` and `c2_points` must both be at least 2. A finite grid
/// finds the best sampled point, not necessarily the continuous optimum
/// between grid points.
pub fn fit_parameters_grid<F>(
    exp: &ExpData,
    eval: F,
    c1_points: usize,
    c2_points: usize,
) -> Result<FitResult>
where
    F: Fn(f64, f64) -> Intensity,
{
    fit_parameters_grid_with_bounds(
        exp,
        eval,
        c1_points,
        c2_points,
        DEFAULT_C1_BOUNDS,
        DEFAULT_C2_BOUNDS,
    )
}

/// Fit `(volume_scale, contrast_density)` against experimental data
/// by exhaustively evaluating an endpoint-inclusive rectangular grid
/// over caller-supplied fit bounds.
///
/// `c1_points` and `c2_points` must both be at least 2. A finite grid
/// finds the best sampled point, not necessarily the continuous optimum
/// between grid points.
pub fn fit_parameters_grid_with_bounds<F>(
    exp: &ExpData,
    eval: F,
    c1_points: usize,
    c2_points: usize,
    c1_bounds: (f64, f64),
    c2_bounds: (f64, f64),
) -> Result<FitResult>
where
    F: Fn(f64, f64) -> Intensity,
{
    if c1_points < 2 || c2_points < 2 {
        return Err(crate::Error::validation(format!(
            "grid fit requires at least 2 points per axis (got c1_points={c1_points}, \
             c2_points={c2_points})"
        )));
    }
    validate_bounds("c1", c1_bounds)?;
    validate_bounds("c2", c2_bounds)?;

    let grid_value = |(lo, hi): (f64, f64), points: usize, i: usize| {
        lo + (hi - lo) * i as f64 / (points - 1) as f64
    };

    log::info!(
        "Grid fit: exhaustive search over {c1_points}×{c2_points} points \
         (c₁ ∈ [{:.2}, {:.2}], c₂ ∈ [{:.3}, {:.4}])",
        c1_bounds.0,
        c1_bounds.1,
        c2_bounds.0,
        c2_bounds.1
    );

    let mut best: Option<(f64, f64, f64)> = None;
    for i in 0..c1_points {
        let c1 = grid_value(c1_bounds, c1_points, i);
        for j in 0..c2_points {
            let c2 = grid_value(c2_bounds, c2_points, j);
            let chi2 = chi2_per_point(&eval(c1, c2), exp);
            if !chi2.is_finite() {
                continue;
            }
            if best.is_none_or(|(_, _, best_chi2)| chi2 < best_chi2) {
                best = Some((c1, c2, chi2));
            }
        }
    }

    let Some((c1, c2, chi2)) = best else {
        return Err(crate::Error::internal(
            "grid fit did not find any finite χ² value",
        ));
    };

    let theory = eval(c1, c2);
    let scale = compute_scale(&theory, exp);

    warn_if_on_bound(c1, c2, c1_bounds, c2_bounds);

    log::info!(
        "Grid fit: χ²/N = {chi2:.4}, excluded volume (c₁) = {c1:.4}, \
         hydration shell (c₂) = {c2:.4}, scale = {scale:.6e}"
    );

    Ok(FitResult {
        volume_scale: c1,
        contrast_density: c2,
        chi2,
        scale,
    })
}

/// Warns when the optimum lies on a bound, i.e. the true minimum may be outside the box.
fn warn_if_on_bound(c1: f64, c2: f64, c1_bounds: (f64, f64), c2_bounds: (f64, f64)) {
    let on_bound = |v: f64, (lo, hi): (f64, f64)| v <= lo + 1e-6 || v >= hi - 1e-6;
    if on_bound(c1, c1_bounds) || on_bound(c2, c2_bounds) {
        log::warn!(
            "Fit hit bound: c₁ = {c1:.4} (∈ [{:.2}, {:.2}]), c₂ = {c2:.4} (∈ [{:.3}, {:.4}]). \
             The fit wanted to go further — check the structure and experimental data.",
            c1_bounds.0,
            c1_bounds.1,
            c2_bounds.0,
            c2_bounds.1
        );
    }
}

fn validate_bounds(name: &str, (lo, hi): (f64, f64)) -> Result<()> {
    if !lo.is_finite() || !hi.is_finite() {
        return Err(crate::Error::validation(format!(
            "{name} bounds must be finite"
        )));
    }
    if lo >= hi {
        return Err(crate::Error::validation(format!(
            "{name} lower bound must be less than upper bound"
        )));
    }
    Ok(())
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

    #[test]
    fn nelder_mead_initial_simplex_respects_custom_bounds() {
        let simplex = initial_simplex((2.0, 3.0), (0.2, 0.4));

        for &(c1, c2) in &simplex {
            assert!((2.0..=3.0).contains(&c1));
            assert!((0.2..=0.4).contains(&c2));
        }
        assert_ne!(simplex[0], simplex[1]);
        assert_ne!(simplex[0], simplex[2]);
    }

    #[test]
    fn nelder_mead_rejects_invalid_bounds() {
        let exp =
            ExpData::from_arrays(vec![0.0, 1.0, 2.0], vec![1.0, 1.25, 0.4], Vec::new()).unwrap();
        let err = fit_parameters_with_bounds(
            &exp,
            |c1, c2| Intensity {
                q: vec![0.0, 1.0, 2.0],
                total: vec![1.0, c1, 10.0 * c2],
                contributions: None,
                ..Default::default()
            },
            (2.0, 0.5),
            DEFAULT_C2_BOUNDS,
        )
        .unwrap_err();

        assert!(err.to_string().contains("lower bound"));
    }

    #[test]
    fn grid_fit_finds_sampled_minimum() {
        let exp =
            ExpData::from_arrays(vec![0.0, 1.0, 2.0], vec![1.0, 1.25, 0.4], Vec::new()).unwrap();
        let fit = fit_parameters_grid(
            &exp,
            |c1, c2| Intensity {
                q: vec![0.0, 1.0, 2.0],
                total: vec![1.0, c1, 10.0 * c2],
                contributions: None,
                ..Default::default()
            },
            7,
            6,
        )
        .unwrap();

        assert!((fit.volume_scale - 1.25).abs() < 1e-12);
        assert!((fit.contrast_density - 0.04).abs() < 1e-12);
        assert!(fit.chi2 < 1e-24);
    }

    #[test]
    fn grid_fit_rejects_degenerate_axes() {
        let exp =
            ExpData::from_arrays(vec![0.0, 1.0, 2.0], vec![1.0, 1.25, 0.4], Vec::new()).unwrap();
        let err = fit_parameters_grid(
            &exp,
            |c1, c2| Intensity {
                q: vec![0.0, 1.0, 2.0],
                total: vec![1.0, c1, 10.0 * c2],
                contributions: None,
                ..Default::default()
            },
            1,
            6,
        )
        .unwrap_err();

        assert!(err.to_string().contains("at least 2 points"));
    }

    #[test]
    fn grid_fit_accepts_custom_c2_bounds() {
        let exp =
            ExpData::from_arrays(vec![0.0, 1.0, 2.0], vec![1.0, 1.25, 12.0], Vec::new()).unwrap();
        let fit = fit_parameters_grid_with_bounds(
            &exp,
            |c1, c2| Intensity {
                q: vec![0.0, 1.0, 2.0],
                total: vec![1.0, c1, 10.0 * c2],
                contributions: None,
                ..Default::default()
            },
            7,
            6,
            (0.5, 2.0),
            (0.0, 2.0),
        )
        .unwrap();

        assert!((fit.volume_scale - 1.25).abs() < 1e-12);
        assert!((fit.contrast_density - 1.2).abs() < 1e-12);
        assert!(fit.chi2 < 1e-24);
    }

    #[test]
    fn grid_fit_rejects_invalid_bounds() {
        let exp =
            ExpData::from_arrays(vec![0.0, 1.0, 2.0], vec![1.0, 1.25, 12.0], Vec::new()).unwrap();
        let err = fit_parameters_grid_with_bounds(
            &exp,
            |c1, c2| Intensity {
                q: vec![0.0, 1.0, 2.0],
                total: vec![1.0, c1, 10.0 * c2],
                contributions: None,
                ..Default::default()
            },
            7,
            6,
            (0.5, 2.0),
            (2.0, 0.0),
        )
        .unwrap_err();

        assert!(err.to_string().contains("lower bound"));
    }
}

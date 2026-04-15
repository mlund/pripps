// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Shared infrastructure for per-q-parallel intensity calculators.
//!
//! Every scheme pripps ships ([`crate::debye::DebyeIntensity`],
//! [`crate::svergun::SvergunIntensity`], …) follows the same outer shape:
//! a grid of q values, a form-factor map, and a closure that turns a
//! single q into `I(q)` using whatever per-structure state the
//! calculator precomputed. [`QSampler`] captures exactly that pattern
//! so the calculators themselves only have to supply the per-q body.

use crate::{Intensity, Result, formfactor::FormFactorMap};
use rayon::prelude::*;

/// Linearly-spaced q grid from `qmin` to `qmax` (inclusive) sampled at
/// `nq` points. Used by [`QSampler::linear`] and by form-factor tables
/// inside `formfactor.rs` that need a plain `Vec<f64>`.
///
/// The `.max(1)` on the denominator handles the degenerate `nq = 1`
/// case (a legal single-q grid) without division by zero.
pub(crate) fn linear_q_values(qmin: f64, qmax: f64, nq: usize) -> Vec<f64> {
    let denom = (nq - 1).max(1) as f64;
    (0..nq)
        .map(|i| qmin + (qmax - qmin) * i as f64 / denom)
        .collect()
}

/// A grid of q values plus the form-factor map they'll be looked up in,
/// with a parallel driver for per-q closures. Shared between all
/// intensity calculators that follow the "for each q, compute I(q)"
/// structure.
#[derive(Debug, Clone)]
pub(crate) struct QSampler {
    pub q_values: Vec<f64>,
    pub formfactors: FormFactorMap,
}

impl QSampler {
    /// Construct from an arbitrary q grid. Accepts any iterator of
    /// `f64` so callers can pass `Vec`, `[f64; N]`, lazy iterator
    /// chains, or experimental q values read from a data file.
    pub fn new<I>(formfactors: FormFactorMap, q_values: I) -> Self
    where
        I: IntoIterator<Item = f64>,
    {
        Self {
            formfactors,
            q_values: q_values.into_iter().collect(),
        }
    }

    /// Linearly-spaced q grid from `qmin` to `qmax` inclusive,
    /// `nq` points. Convenience wrapper around [`Self::new`] for the
    /// common CLI case.
    pub fn linear(formfactors: FormFactorMap, qmin: f64, qmax: f64, nq: usize) -> Self {
        Self::new(formfactors, linear_q_values(qmin, qmax, nq))
    }

    /// Evaluate `f(q)` for every stored q in parallel and return the
    /// `(q, I(q))` curve in input order.
    ///
    /// `f` may close over any per-structure state the calculator
    /// precomputed before calling `run_parallel` — that's the whole
    /// point of this closure-based shape. Expensive preprocessing
    /// (e.g. per-atom spherical harmonics) lives outside the closure
    /// and is shared by all workers via immutable reference; per-q
    /// scratch lives inside the closure and is per-worker.
    pub fn run_parallel<F>(&self, f: F) -> Result<Intensity>
    where
        F: Fn(f64) -> Result<f64> + Sync,
    {
        let total: Vec<f64> = self
            .q_values
            .par_iter()
            .map(|&q| f(q))
            .collect::<Result<Vec<_>>>()?;
        Ok(Intensity {
            q: self.q_values.clone(),
            total,
            ..Default::default()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Error;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn empty_ffs() -> FormFactorMap {
        FormFactorMap::default()
    }

    #[test]
    fn new_from_vec() {
        let s = QSampler::new(empty_ffs(), vec![0.1, 0.2, 0.3]);
        assert_eq!(s.q_values, vec![0.1, 0.2, 0.3]);
    }

    #[test]
    fn new_from_array() {
        let s = QSampler::new(empty_ffs(), [0.1, 0.2]);
        assert_eq!(s.q_values, vec![0.1, 0.2]);
    }

    #[test]
    fn new_from_range_map() {
        let s = QSampler::new(empty_ffs(), (0..5).map(|i| i as f64 * 0.01));
        assert_eq!(s.q_values, vec![0.0, 0.01, 0.02, 0.03, 0.04]);
    }

    #[test]
    fn run_parallel_preserves_input_order() {
        let s = QSampler::new(empty_ffs(), [0.3, 0.1, 0.2]);
        let out = s.run_parallel(|q| Ok(q * q + 1.0)).unwrap();
        assert_eq!(out.q, vec![0.3, 0.1, 0.2]);
        assert_eq!(
            out.total,
            vec![0.3 * 0.3 + 1.0, 0.1 * 0.1 + 1.0, 0.2 * 0.2 + 1.0]
        );
    }

    #[test]
    fn run_parallel_calls_closure_exactly_once_per_q() {
        let s = QSampler::new(empty_ffs(), (0..50).map(|i| i as f64));
        let counter = AtomicUsize::new(0);
        s.run_parallel(|_q| {
            counter.fetch_add(1, Ordering::Relaxed);
            Ok(0.0)
        })
        .unwrap();
        assert_eq!(counter.load(Ordering::Relaxed), 50);
    }

    #[test]
    fn run_parallel_propagates_error() {
        let s = QSampler::new(empty_ffs(), vec![0.1, 0.2, 0.3]);
        let out = s.run_parallel(|q| {
            if q > 0.15 {
                Err(Error::validation("boom"))
            } else {
                Ok(q)
            }
        });
        let err = out.unwrap_err();
        assert!(matches!(err, Error::Validation(_)));
    }

    #[test]
    fn run_parallel_empty_q_values() {
        let s = QSampler::new(empty_ffs(), Vec::<f64>::new());
        let out = s.run_parallel(|_q| Ok(1.0)).unwrap();
        assert!(out.q.is_empty());
        assert!(out.total.is_empty());
    }

    #[test]
    fn run_parallel_single_q_value() {
        let s = QSampler::new(empty_ffs(), [0.42]);
        let out = s.run_parallel(|q| Ok(q * 2.0)).unwrap();
        assert_eq!(out.q, vec![0.42]);
        assert_eq!(out.total, vec![0.84]);
    }

    #[test]
    fn linear_endpoints() {
        let s = QSampler::linear(empty_ffs(), 0.0, 1.0, 11);
        assert_eq!(s.q_values.len(), 11);
        assert_eq!(s.q_values[0], 0.0);
        assert_eq!(s.q_values[10], 1.0);
    }

    #[test]
    fn linear_single_q() {
        let s = QSampler::linear(empty_ffs(), 0.42, 0.42, 1);
        assert_eq!(s.q_values, vec![0.42]);
    }
}

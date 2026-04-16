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

use itertools::Itertools;
use molly::{Frame, XTCReader, selection::AtomSelection};
use std::iter::zip;
use std::{f64, path};

#[cfg(feature = "cli")]
use indicatif::ProgressBar;

use crate::error::IoResultExt;
use crate::structure::Structure;
use crate::{Error, Intensity, IntensityCalculator, Result, Vector3};

/// Iterator over (weighted) frames in an XTC file
pub struct XtcIterator {
    reader: XTCReader<std::fs::File>,
    weights: Option<Vec<f64>>,
    cnt: usize,
    /// Total number of frames in the trajectory
    pub num_frames: usize,
}

impl XtcIterator {
    /// Open an XTC file for iteration with an optional file with weights for each frame
    pub fn open(path: &path::Path, weights: Option<&path::Path>) -> Result<Self> {
        let reader = XTCReader::open(path).with_path(path)?;
        let weights = weights.map(read_weights).transpose()?;
        let mut xtc = Self {
            reader,
            weights,
            cnt: 0,
            num_frames: 0,
        };
        xtc.num_frames = xtc.num_frames()?;
        if let Some(w) = xtc.weights.as_ref()
            && w.len() != xtc.num_frames
        {
            return Err(Error::validation(format!(
                "number of weights ({}) does not match number of frames in trajectory ({})",
                w.len(),
                xtc.num_frames,
            )));
        }
        Ok(xtc)
    }

    /// Get the number of frames in the trajectory
    fn num_frames(&mut self) -> Result<usize> {
        let mut num_frames = 0;
        while self
            .reader
            .read_frame_with_selection(&mut Frame::default(), &AtomSelection::Until(0))
            .is_ok()
        {
            num_frames += 1;
        }
        self.reader.home()?;
        Ok(num_frames)
    }
}

impl Iterator for XtcIterator {
    type Item = (Frame, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let mut frame = Frame::default();
        self.reader
            .read_frame(&mut frame)
            .map(|_| {
                // When no explicit weights file was given, each frame
                // gets a uniform 1/N weight so the accumulation loop
                // can always do `Σ wᵢ·vᵢ` without a separate `/N`.
                let weight = self
                    .weights
                    .as_ref()
                    .map(|w| w[self.cnt])
                    .unwrap_or(1.0 / self.num_frames as f64);
                self.cnt += 1;
                (frame, weight)
            })
            .ok()
    }
}

fn read_weights(path: &path::Path) -> Result<Vec<f64>> {
    let mut weights: Vec<f64> = std::fs::read_to_string(path)
        .with_path(path)?
        .lines()
        .map(|w| w.parse())
        .try_collect()
        .map_err(|e: std::num::ParseFloatError| {
            Error::validation(format!("weight parse error: {e}"))
        })?;

    normalize_vector(weights.as_mut_slice())?;

    log::info!("Read {} weights from {}", weights.len(), path.display());
    Ok(weights)
}

/// Normalize a vector so that the sum of its elements is 1
fn normalize_vector(v: &mut [f64]) -> Result<()> {
    let sum: f64 = v.iter().sum();
    if !sum.abs().is_normal() {
        return Err(Error::validation("sum of weights is zero or infinite"));
    }
    for w in v.iter_mut() {
        *w /= sum;
    }
    Ok(())
}

/// Get side lengths from a frame (Å)
///
/// Assumes that values are in nm and converts to Å
pub fn get_side_lengths(frame: &molly::Frame) -> Option<Vector3> {
    const ANGSTROM_PER_NM: f64 = 10.0;
    let sides = frame.boxvec.to_cols_array_2d();
    let sides =
        ANGSTROM_PER_NM * Vector3::new(sides[0][0] as f64, sides[1][1] as f64, sides[2][2] as f64);
    log::trace!("frame side-length = {sides:?}");
    sides.norm_squared().gt(&f64::EPSILON).then_some(sides)
}

/// Weighted average of I(q) over all frames in an XTC trajectory.
///
/// The template `structure` is cloned and used as per-frame scratch —
/// `set_positions` refreshes both coordinates and (if the frame carries
/// one) the periodic box on every iteration. The frame loop is serial;
/// each calculator parallelizes internally over q values / q vectors.
pub fn trajectory_average(
    calc: &dyn IntensityCalculator,
    structure: &Structure,
    xtc_path: &path::Path,
    weights: Option<&path::Path>,
) -> Result<Intensity> {
    let reader = XtcIterator::open(xtc_path, weights)?;
    let num_frames = reader.num_frames;
    #[cfg(feature = "cli")]
    let progress = ProgressBar::new(num_frames as u64);

    log::info!("Averaging 𝐼(𝑞) over {num_frames} frames");

    let mut mean = Intensity::default();
    let mut structure = structure.clone();

    for (frame, weight) in reader {
        structure.set_positions(&frame)?;
        let curve = calc.calc_intensities(&structure)?;

        if mean.q.is_empty() {
            mean.q = curve.q.clone();
            mean.total = vec![0.0; curve.total.len()];
        }

        for (acc, v) in zip(&mut mean.total, &curve.total) {
            *acc += weight * v;
        }

        #[cfg(feature = "cli")]
        progress.inc(1);
    }

    Ok(mean)
}

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
use std::num::NonZeroUsize;
use std::path;
use std::time::Instant;

#[cfg(feature = "cli")]
use indicatif::ProgressBar;

use crate::error::IoResultExt;
use crate::structure::Structure;
use crate::{Error, Intensity, IntensityCalculator, Result, Vector3};

/// Iterator over (weighted) frames in an XTC file.
///
/// Frames are reached by seeking to precomputed per-frame byte offsets,
/// so a `stride > 1` decodes only the frames it yields — the skipped
/// frames cost a cheap seek rather than a full decompression. The frame
/// count is likewise derived from the offset table (headers + seeks),
/// without decompressing any coordinates.
pub struct XtcIterator {
    reader: XTCReader<std::fs::File>,
    /// One weight per frame, indexed by the original frame number; all
    /// ones when no weights file was given. Left unnormalized — the
    /// averager divides by the weight it actually consumes, so any global
    /// scaling cancels.
    weights: Vec<f64>,
    /// Start byte offset of every frame in the trajectory.
    offsets: Box<[u64]>,
    /// Stride between yielded frames (≥ 1).
    stride: usize,
    /// Index into `offsets` of the next frame to yield.
    next_idx: usize,
}

impl XtcIterator {
    /// Open an XTC file for iteration, yielding every `stride`-th frame.
    ///
    /// `weights` is an optional file of per-frame weights (one per line),
    /// indexed by the original frame number rather than the strided
    /// position, so the weight stays attached to its frame regardless of
    /// stride.
    pub fn open(
        path: &path::Path,
        weights: Option<&path::Path>,
        stride: NonZeroUsize,
    ) -> Result<Self> {
        let mut reader = XTCReader::open(path).with_path(path)?;
        // Header-and-seek scan: counts frames and records their byte
        // offsets without decompressing any coordinates.
        let offsets = reader.determine_offsets(None).with_path(path)?;
        let weights = match weights {
            Some(path) => {
                let w = read_weights(path)?;
                if w.len() != offsets.len() {
                    return Err(Error::validation(format!(
                        "number of weights ({}) does not match number of frames in trajectory ({})",
                        w.len(),
                        offsets.len(),
                    )));
                }
                w
            }
            None => vec![1.0; offsets.len()],
        };
        Ok(Self {
            reader,
            weights,
            offsets,
            stride: stride.get(),
            next_idx: 0,
        })
    }
}

impl Iterator for XtcIterator {
    type Item = (Frame, f64);

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.next_idx;
        let &offset = self.offsets.get(idx)?;
        self.next_idx += self.stride;

        let mut frame = Frame::default();
        self.reader
            .read_frame_at_offset::<false>(&mut frame, offset, &AtomSelection::All)
            .ok()?;

        Some((frame, self.weights[idx]))
    }

    /// Exact: every remaining strided index yields exactly one frame.
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self
            .offsets
            .len()
            .saturating_sub(self.next_idx)
            .div_ceil(self.stride);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for XtcIterator {}

/// Read per-frame weights, one per line. Weights are returned as-is (not
/// normalized): the trajectory averager divides by the weight it consumes,
/// so only the relative magnitudes matter. The total is validated to be
/// finite and non-zero so the eventual division is well-defined.
fn read_weights(path: &path::Path) -> Result<Vec<f64>> {
    let weights: Vec<f64> = std::fs::read_to_string(path)
        .with_path(path)?
        .lines()
        .map(|w| w.parse())
        .try_collect()
        .map_err(|e: std::num::ParseFloatError| {
            Error::validation(format!("weight parse error: {e}"))
        })?;

    let sum: f64 = weights.iter().sum();
    if !sum.abs().is_normal() {
        return Err(Error::validation("sum of weights is zero or infinite"));
    }

    log::info!("Read {} weights from {}", weights.len(), path.display());
    Ok(weights)
}

/// Get side lengths from a frame (Å)
///
/// Assumes that values are in nm and converts to Å
pub fn get_side_lengths(frame: &molly::Frame) -> Option<Vector3> {
    const ANGSTROM_PER_NM: f64 = 10.0;
    // molly 0.6's BoxVec is [f32; 9] in column-major 3x3 layout;
    // the orthorhombic side lengths are the (0,0), (1,1), (2,2)
    // diagonal entries at indices 0, 4, 8.
    let b = &frame.boxvec;
    let sides = ANGSTROM_PER_NM * Vector3::new(b[0] as f64, b[4] as f64, b[8] as f64);
    log::trace!("frame side-length = {sides:?}");
    sides.norm_squared().gt(&f64::EPSILON).then_some(sides)
}

/// Weighted average of I(q) over all frames in an XTC trajectory.
///
/// The template `structure` is cloned and used as per-frame scratch —
/// `set_positions` refreshes both coordinates and (if the frame carries
/// one) the periodic box on every iteration. The frame loop is serial;
/// each calculator parallelizes internally over q values / q vectors.
///
/// Curves are accumulated positionally against the q-grid of the first frame,
/// so a **fixed box** (constant volume) is assumed: a fluctuating (NPT) box
/// shifts the per-frame Direct q-grid and would misalign the average.
pub fn trajectory_average(
    calc: &dyn IntensityCalculator,
    structure: &Structure,
    xtc_opts: crate::TrajectoryOptions,
) -> Result<Intensity> {
    // The offset scan walks every frame header in the file (see
    // `XtcIterator::open`); for multi-gigabyte trajectories this is a
    // noticeable up-front cost, so time it separately from the averaging.
    let scan_start = Instant::now();
    let reader = XtcIterator::open(xtc_opts.xtcfile, xtc_opts.weights, xtc_opts.stride)?;
    let num_frames = reader.len();
    log::info!(
        "Scanned frame offsets in {:.2} s",
        scan_start.elapsed().as_secs_f64()
    );

    #[cfg(feature = "cli")]
    let progress = ProgressBar::new(num_frames as u64);

    log::info!("Averaging 𝐼(𝑞) over {num_frames} frames");

    let mut mean = Intensity::default();
    let mut structure = structure.clone();
    let mut weight_sum = 0.0;

    // Count frames we actually consume: `XtcIterator` stops early on a frame
    // read error (its `next` swallows the error), so `num_frames` (the offset
    // table length) can overstate what was processed. Time against the real
    // count so ms/frame stays honest.
    let mut processed = 0usize;
    let average_start = Instant::now();
    for (frame, weight) in reader {
        weight_sum += weight;
        structure.set_positions(&frame)?;
        let curve = calc.calc_intensities(&structure)?;

        if mean.q.is_empty() {
            mean.q = curve.q.clone();
            mean.total = vec![0.0; curve.total.len()];
            // Mirror `total` for the per-molecule self term, so S_eff is formed
            // as a ratio of weighted averages after the loop.
            mean.self_term = curve.self_term.as_ref().map(|s| vec![0.0; s.len()]);
        }

        for (acc, v) in zip(&mut mean.total, &curve.total) {
            *acc += weight * v;
        }
        if let (Some(acc), Some(v)) = (mean.self_term.as_mut(), &curve.self_term) {
            for (a, x) in zip(acc, v) {
                *a += weight * x;
            }
        }

        processed += 1;
        #[cfg(feature = "cli")]
        progress.inc(1);
    }

    let elapsed = average_start.elapsed().as_secs_f64();
    log::info!(
        "Averaged {processed} frames in {elapsed:.2} s ({:.1} ms/frame)",
        1e3 * elapsed / processed.max(1) as f64
    );

    // The *consumed* weights can sum to zero even though `read_weights` checked
    // the full set (striding may select a zero-sum subset of signed weights);
    // dividing by it would turn every average into NaN, so reject it.
    if !weight_sum.abs().is_normal() {
        return Err(Error::validation(format!(
            "weights of the {processed} averaged frames sum to {weight_sum}; cannot normalize"
        )));
    }

    // Normalize the accumulated sums by the total weight to get the weighted
    // averages (`total` and the self term divided by the same weight).
    for acc in mean
        .total
        .iter_mut()
        .chain(mean.self_term.iter_mut().flatten())
    {
        *acc /= weight_sum;
    }

    Ok(mean)
}

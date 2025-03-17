// Copyright 2025 Isabel Vinterbladh, Rémi Vuillemot, Sergei Grudinin and Mikael Lund
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

use anyhow::{Result, bail};
use itertools::Itertools;
use molly::{Frame, XTCReader, selection::AtomSelection};
use std::path;

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
        let reader = XTCReader::open(path)?;
        let weights = weights.map(read_weights).transpose()?;
        let mut xtc = Self {
            reader,
            weights,
            cnt: 0,
            num_frames: 0,
        };
        xtc.num_frames = xtc.num_frames()?;
        if let Some(w) = xtc.weights.as_ref() {
            if w.len() != xtc.num_frames {
                bail!("number of weights does not match number of frames in trajectory");
            }
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
        match self.reader.read_frame(&mut frame) {
            Ok(_) => {
                let weight = self.weights.as_ref().map(|w| w[self.cnt]).unwrap_or(1.0);
                self.cnt += 1;
                Some((frame, weight))
            }
            Err(_) => None,
        }
    }
}

fn read_weights(path: &path::Path) -> Result<Vec<f64>> {
    let read_error = |e| bail!("error reading weights: {e}");

    let mut weights: Vec<_> = std::fs::read_to_string(path)
        .or_else(read_error)?
        .lines()
        .map(|w| w.parse())
        .try_collect()
        .map_err(|e| anyhow::anyhow!("weight error: {e}"))?;

    normalize_vector(weights.as_mut_slice())?;

    log::info!("Read {} weights from {}", weights.len(), path.display());
    Ok(weights)
}

/// Normalize a vector so that the sum of its elements is 1
fn normalize_vector(v: &mut [f64]) -> Result<()> {
    let sum: f64 = v.iter().sum();

    if sum.abs() < 1e-6 {
        bail!("sum of weights is zero");
    }

    for w in v.iter_mut() {
        *w /= sum;
    }
    Ok(())
}

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

use anyhow::{Result, bail};
use std::ops::{Div, IndexMut};

/// Struct representing a 2D table where one axis ("x") is linear.
///
/// x values are mapped to indices in the grid whereby only y-data is stored.
/// Lookup complexity is O(1):
///
/// `(𝑥ᵢ, 𝑦ᵢ) = (𝑥₀ + 𝑖Δx, data[𝑖])`
///
/// This is also known as a regular grid or implicit coordinate mapping.
///
#[derive(Debug, Clone, PartialEq)]
pub struct RegularGrid<T> {
    /// Constant spacing between x values
    dx: f64,
    /// Offset (starting point) of the x values.
    x0: f64,
    /// Mapped data (y-values)
    data: Vec<T>,
}

impl<T: Default + Clone> RegularGrid<T> {
    /// Create new regular grid with given range and spacing.
    ///
    /// Elements are initialized with the default value of `T`.
    pub fn new(xrange: &(f64, f64), dx: f64) -> Result<Self> {
        if dx <= 0.0 {
            bail!("dx must be positive");
        }
        let len = (xrange.1 - xrange.0).div(dx).round() as isize + 1;
        if len <= 0 {
            bail!("invalid or empty grid size");
        }
        let data = vec![T::default(); len as usize];
        Ok(RegularGrid {
            dx,
            x0: xrange.0,
            data,
        })
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    /// Find index of element at given x coordinate.
    fn index(&self, x: f64) -> usize {
        (x - self.x0).div(self.dx).round() as usize
    }
    /// Get reference to element at given x coordinate.
    pub fn get(&self, x: f64) -> &T {
        &self.data[self.index(x)]
    }
    /// Get mutable reference to element at given x coordinate.
    pub fn get_mut(&mut self, x: f64) -> &mut T {
        self.data.index_mut(self.index(x))
    }
    /// Iterator over `(x, &y)` pairs
    pub fn iter(&self) -> impl Iterator<Item = (f64, &T)> {
        self.data
            .iter()
            .enumerate()
            .map(|(i, y)| (self.x0 + i as f64 * self.dx, y))
    }
    /// Iterator over `(x, &mut y)` pairs
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (f64, &mut T)> {
        self.data
            .iter_mut()
            .enumerate()
            .map(|(i, y)| (self.x0 + i as f64 * self.dx, y))
    }
    /// Get x range (min, max)
    pub fn x_range(&self) -> (f64, f64) {
        (self.x0, self.x0 + (self.data.len() - 1) as f64 * self.dx)
    }
}

impl RegularGrid<f64> {
    /// Interpolate y-values
    ///
    /// # Panics
    ///
    /// Panics if `x` is outside the grid range.
    ///
    pub fn interpolate(&self, x: f64) -> f64 {
        let i = (x - self.x0).div(self.dx) as usize;
        let x0 = self.x0 + i as f64 * self.dx;
        let x1 = x0 + self.dx;
        let y0 = self.data[i];
        let y1 = self.data[i + 1];
        y0 + (x - x0) * (y1 - y0) / (x1 - x0)
    }
}

#[test]
fn test_regulargrid() {
    let mut grid = RegularGrid::<usize>::new(&(-1.0, 1.0), 0.5).unwrap();
    assert!(!grid.is_empty());
    assert_eq!(grid.len(), 5);
    for (i, (_x, y)) in grid.iter_mut().enumerate() {
        *y = i;
    }
    assert_eq!(grid.get(-1.0), &0);
    assert_eq!(grid.get(-0.5), &1);
    assert_eq!(grid.get(0.0), &2);
    assert_eq!(grid.get(0.5), &3);
    assert_eq!(grid.get(1.0), &4);

    let xrange = grid.x_range();
    assert!(xrange.0 == -1.0);
    assert_eq!(xrange.1, 1.0);

    let sum_y = grid.iter().map(|(_x, y)| y).sum::<usize>();
    assert_eq!(sum_y, 10);

    // Test interpolation for T=f64
    let mut grid = RegularGrid::<f64>::new(&(-1.0, 1.0), 0.5).unwrap();
    *grid.get_mut(-1.0) = 10.0;
    *grid.get_mut(-0.5) = 20.0;
    *grid.get_mut(0.0) = 40.0;
    *grid.get_mut(0.5) = 80.0;
    *grid.get_mut(1.0) = 120.0;
    assert_eq!(grid.interpolate(-1.0), 10.0);
    assert_eq!(grid.interpolate(-0.75), 15.0);
    assert_eq!(grid.interpolate(-0.25), 30.0);
    assert_eq!(grid.interpolate(0.0), 40.0);
    assert_eq!(grid.interpolate(0.25), 60.0);
    assert_eq!(grid.interpolate(0.75), 100.0);
}

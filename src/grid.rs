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

/// Struct representing a 2D grid of implicitly mapped data.
///
/// x values are implicitly mapped to indices in the grid by assuming
/// they are evenly spaced.
#[derive(Debug, Clone)]
pub struct Implicit2DGrid<T> {
    /// Mapped data
    data: Vec<T>,
    /// Spacing between grid points along x.
    dx: f64,
    /// Offset (starting point) of the x grid.
    x0: f64,
}

impl<T: Default + Clone> Implicit2DGrid<T> {
    pub fn new(xrange: &(f64, f64), dx: f64) -> Result<Self> {
        if dx <= 0.0 {
            bail!("dx must be positive");
        }
        let len = (xrange.1 - xrange.0).div(dx).round() as isize + 1;
        if len <= 0 {
            bail!("invalid or empty grid size");
        }
        let data = vec![T::default(); len as usize];
        Ok(Implicit2DGrid {
            data,
            dx,
            x0: xrange.0,
        })
    }
    pub fn len(&self) -> usize {
        self.data.len()
    }
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    fn lookup_index(&self, x: f64) -> usize {
        (x - self.x0).div(self.dx).round() as usize
    }
    pub fn get(&self, x: f64) -> &T {
        &self.data[self.lookup_index(x)]
    }
    pub fn get_mut(&mut self, x: f64) -> &mut T {
        self.data.index_mut(self.lookup_index(x))
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

#[test]
fn test_implicit2dgrid() {
    let mut grid = Implicit2DGrid::<usize>::new(&(-1.0, 1.0), 0.5).unwrap();
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
}

/// A trait for implicit coordinate mapping of f(x) to a regular set of x values.
pub trait RegularGrid<T: Clone>
where
    Self: AsRef<[T]> + AsMut<[T]>,
{
    #[inline]
    fn lookup_index(x: f64, dx: f64, x0: Option<f64>) -> Option<usize> {
        debug_assert!(dx > 0.0);
        let x0 = x0.unwrap_or(0.0);
        let index = (x - x0).div(dx).round() as isize;
        index.ge(&0).then_some(index as usize)
    }

    /// Returns f(x) assuming a regularly spaced set of x-values.
    ///
    /// # Arguments
    ///
    /// * `x` - The value
    /// * `dx` - The spacing between grid points. Must be positive.
    /// * `x0` - The optional offset (starting point) of the grid. Defaults to `0.0` if `None`.
    ///
    /// # Returns
    ///
    /// Returns `Some(&T)` if the computed index is positive,
    /// otherwise returns `None`.
    ///
    /// # Panics
    ///
    /// Panics in debug mode if `dx` is not greater than `0.0`.
    fn lookup(&self, x: f64, dx: f64, x0: Option<f64>) -> Option<&T> {
        Self::lookup_index(x, dx, x0).and_then(|i| self.as_ref().get(i))
    }

    fn lookup_mut(&mut self, x: f64, dx: f64, x0: Option<f64>) -> Option<&mut T> {
        Self::lookup_index(x, dx, x0).and_then(|i| self.as_mut().get_mut(i))
    }

    /// Calculate the size of the given x-range and resolution
    fn grid_size(xrange: &(f64, f64), dx: f64) -> usize {
        Self::lookup_index(xrange.1, dx, Some(xrange.0)).unwrap() + 1
    }

    /// Calculate x from index, 𝑥ᵢ = 𝑥₀ + 𝑖Δx
    fn x_from_index(index: usize, dx: f64, x0: Option<f64>) -> f64 {
        x0.unwrap_or(0.0) + index as f64 * dx
    }

    /// Create vector of f(x) values corresponding to the x coordinates
    fn from_grid(xrange: &(f64, f64), dx: f64, value: T) -> Self;

    /// Iterator over `(x,&y)` pairs
    fn iter_grid<'a>(&'a self, dx: f64, x0: Option<f64>) -> impl Iterator<Item = (f64, &'a T)>
    where
        T: 'a;
}

impl<T: Clone> RegularGrid<T> for Vec<T> {
    fn from_grid(xrange: &(f64, f64), dx: f64, value: T) -> Self {
        vec![value; Self::grid_size(xrange, dx)]
    }

    fn iter_grid<'a>(&'a self, dx: f64, x0: Option<f64>) -> impl Iterator<Item = (f64, &'a T)>
    where
        T: 'a,
    {
        self.iter()
            .enumerate()
            .map(move |(i, y)| (Self::x_from_index(i, dx, x0), y))
    }
}

#[test]
fn test_regulargrid() {
    assert_eq!(Vec::<f64>::grid_size(&(0.0, 3.0), 1.0), 4);
    assert_eq!(Vec::<f64>::grid_size(&(-1.0, 1.0), 0.5), 5);

    let v = vec![1, 2, 3, 4];
    assert_eq!(v.lookup(0.0, 1.0, None), Some(&1));
    assert_eq!(v.lookup(2.4, 1.0, None), Some(&3));
    assert_eq!(v.lookup(2.5, 1.0, None), Some(&4));
    assert_eq!(v.lookup(0.5, 0.5, Some(-1.0)), Some(&4));
    assert_eq!(v.lookup(-1.0, 1.0, None), None); // Negative index

    let dx = 0.5;
    let mut v = Vec::from_grid(&(-1.0, 1.0), dx, 0.0);
    assert_eq!(v.len(), 5);
    *v.lookup_mut(0.9, dx, Some(-1.0)).unwrap() = 3.14;
    assert_eq!(v.lookup(0.9, dx, Some(-1.0)).unwrap(), &3.14);

    let pairs: Vec<_> = v.iter_grid(dx, Some(-1.0)).collect();
    assert_eq!(pairs.len(), 5);
    assert_eq!(
        pairs,
        vec![
            (-1.0, &0.0),
            (-0.5, &0.0),
            (0.0, &0.0),
            (0.5, &0.0),
            (1.0, &3.14)
        ]
    );
}

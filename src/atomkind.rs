// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Local atom-kind type and name-lookup trait.
//!
//! Replaces the (much heavier) `faunus::topology::{AtomKind, AtomKindBuilder,
//! FindByName, Value}` set. `sigma` and `dsv` are parsed from the YAML
//! atom-file and preserved here for the planned SASA re-introduction via
//! `voronota-ltr`.

use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

/// A named species in the scattering calculation.
///
/// Identity is `id` alone: [`Eq`], [`Ord`], and [`Hash`] ignore the data
/// fields. This matches the `BTreeMap<AtomKind, _>` iteration-order-equals-id
/// invariant that `formfactor::read_formfactors` asserts and
/// `formfactor::formfactor_by_index` relies on.
#[derive(Debug, Clone)]
pub struct AtomKind {
    /// Stable numeric id used for indexing into form-factor vectors.
    pub id: usize,
    /// Human-readable name (e.g. "C", "ALA", "H2O").
    pub name: String,
    /// 2 × radius (Å), parsed from the YAML `radius` field. Consumed by the
    /// planned voronota-ltr SASA path.
    pub sigma: Option<f64>,
    /// Solvent-excluded radius (Å), parsed from the YAML `dsv` field.
    /// Consumed by the planned voronota-ltr SASA path.
    pub dsv: Option<f64>,
}

impl AtomKind {
    pub fn new(name: impl Into<String>, id: usize) -> Self {
        Self {
            id,
            name: name.into(),
            sigma: None,
            dsv: None,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }
    pub fn id(&self) -> usize {
        self.id
    }
    pub fn sigma(&self) -> Option<f64> {
        self.sigma
    }
    pub fn dsv(&self) -> Option<f64> {
        self.dsv
    }
}

impl PartialEq for AtomKind {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}
impl Eq for AtomKind {}
impl Hash for AtomKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}
impl PartialOrd for AtomKind {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for AtomKind {
    fn cmp(&self, other: &Self) -> Ordering {
        self.id.cmp(&other.id)
    }
}

/// Look up an [`AtomKind`] by name in a slice.
pub trait FindByName {
    fn find_name(&self, name: &str) -> Option<&AtomKind>;
}

impl FindByName for [AtomKind] {
    fn find_name(&self, name: &str) -> Option<&AtomKind> {
        self.iter().find(|k| k.name == name)
    }
}

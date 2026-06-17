// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Local atom-kind type and name-lookup trait.
//!
//! Replaces the (much heavier) `faunus::topology::{AtomKind, AtomKindBuilder,
//! FindByName, Value}` set. `sigma` and `displaced_volume` are parsed from
//! the YAML atom-file and consumed by the solvent-correction path
//! (`ExcludedVolume::Fraser` and the planned SASA path).

/// A named species in the scattering calculation.
///
/// `AtomKind` values live inside a [`crate::FormFactorMap`] and are
/// addressed positionally: the map's index is the kind's numeric id,
/// so the struct itself doesn't need to carry one.
//
// The id field was removed to eliminate redundant state — storing
// the position inside the struct that already lives at that
// position meant the `from_entries` constructor needed a
// debug-only invariant check every time, and Fraser kind lookups
// went through an `id()` getter that was itself just a Vec index.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomKind {
    /// Human-readable name (e.g. "C", "ALA", "H2O").
    pub name: String,
    /// 2 × radius (Å), parsed from the YAML `radius` field.
    pub sigma: Option<f64>,
    /// Displaced solvent volume in Å³, parsed from the YAML
    /// `displaced_volume` field. `None` is legal for coarse-grained
    /// kinds that a given structure never references;
    /// [`crate::ExcludedVolume::Fraser`] errors if it sees `None` for
    /// a kind that is referenced.
    pub displaced_volume: Option<f64>,
}

impl AtomKind {
    /// Create a kind with the given name and no physical parameters
    /// (`sigma` and `displaced_volume` are both `None`). Test-only helper.
    #[cfg(test)]
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            sigma: None,
            displaced_volume: None,
        }
    }

    /// Human-readable name (e.g. "C", "ALA", "H2O").
    pub fn name(&self) -> &str {
        &self.name
    }
    /// 2 × radius (Å), or `None` if unset.
    pub const fn sigma(&self) -> Option<f64> {
        self.sigma
    }
    /// Displaced solvent volume in Å³, or `None` if unset.
    pub const fn displaced_volume(&self) -> Option<f64> {
        self.displaced_volume
    }
}

/// Look up an [`AtomKind`] by name in an ordered slice and return
/// its positional index.
//
// Returns the index rather than a `&AtomKind` because the only
// caller (`Structure::from_file`) immediately stores that index as
// the per-atom id in the structure — skipping the borrow step
// keeps the one useful bit of information and drops the rest.
pub trait FindByName {
    /// Return the positional index of the first kind whose name
    /// matches `name`, or `None` if there is no match.
    fn find_name(&self, name: &str) -> Option<usize>;
}

impl FindByName for [AtomKind] {
    fn find_name(&self, name: &str) -> Option<usize> {
        self.iter().position(|k| k.name == name)
    }
}

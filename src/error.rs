// Copyright 2025 Mikael Lund
//
// Licensed under the Apache license, version 2.0 (the "license");
// you may not use this file except in compliance with the license.

//! Typed error enum used at every public API boundary. Variants are
//! chosen to map cleanly to Python exception classes for the upcoming
//! `pyo3` bindings: `File` / `Io` → `IOError`, `Yaml` / `UnknownAtom` /
//! `UndefinedFormFactor` / `QOutOfRange` / `Validation` / `Usage` →
//! `ValueError`, `Internal` → `RuntimeError`.

use std::path::{Path, PathBuf};

/// All failures produced by the pripps library.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// I/O error with a known file path — attached by [`IoResultExt::with_path`]
    /// at every site that reads or writes a user-supplied file.
    #[error("{}: {source}", .path.display())]
    File {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// I/O error without an attached path — used for errors from deeper
    /// library calls (e.g. molly mid-trajectory reads) where pripps
    /// doesn't hold the file path at the error site.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    /// YAML parsing or serialization failure.
    #[error("YAML error: {0}")]
    Yaml(#[from] yaml_serde::Error),

    /// Atom name in a structure file not found in the form-factor map.
    #[error("unknown atom name: {0}")]
    UnknownAtom(String),

    /// Reference to a form factor by name that isn't defined (alias,
    /// united-atom center, linear-combination component, …).
    #[error("undefined form factor: {0}")]
    UndefinedFormFactor(String),

    /// q value outside a form factor's declared valid range.
    #[error("q value {q} outside form factor range [{lo}, {hi}]")]
    QOutOfRange { q: f64, lo: f64, hi: f64 },

    /// Data-shape / invariant failures on user input: frame↔structure
    /// atom-count mismatch, weight/frame count mismatch, empty input
    /// file, degenerate numeric state, parse failures in pripps's own
    /// readers.
    #[error("{0}")]
    Validation(String),

    /// API misuse: incompatible argument combinations at the public
    /// boundary.
    #[error("{0}")]
    Usage(String),

    /// Broken crate-internal invariant. Should be unreachable in
    /// practice; reaching here indicates a resolver bug.
    #[error("internal: {0}")]
    Internal(String),
}

impl Error {
    pub(crate) fn validation(msg: impl Into<String>) -> Self {
        Self::Validation(msg.into())
    }
    pub(crate) fn usage(msg: impl Into<String>) -> Self {
        Self::Usage(msg.into())
    }
    pub(crate) fn internal(msg: impl Into<String>) -> Self {
        Self::Internal(msg.into())
    }
}

/// Crate-wide result alias.
pub type Result<T> = std::result::Result<T, Error>;

/// Attach a file path to an `io::Result`, producing an [`Error::File`]
/// with both path and underlying cause in its `Display`.
pub(crate) trait IoResultExt<T> {
    fn with_path(self, path: &Path) -> Result<T>;
}

impl<T> IoResultExt<T> for std::io::Result<T> {
    fn with_path(self, path: &Path) -> Result<T> {
        self.map_err(|source| Error::File {
            path: path.to_path_buf(),
            source,
        })
    }
}

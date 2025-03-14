# Partial Rust Implementation of PePSi (Pripps)

This is a tool to calculate the scattering intensity of molecular structures.
Multiple form factor models are supported, including constant, polynomial, gaussian, and table-based models.

## Usage

```sh
pripps --atomfile assets/formfactors.yaml debye --input structure.xyz
```

## Features

- Multiple form factor models
- File formats: `xyz`
- Multithreaded
- Unit tested
- Overindulgent error handling and memory safetiness

## Form Factor Models

Atom definitions, which may be atomic or coarse grained, are defined in a separate YAML file,
see e.g. `assets/formfactors.yaml`.
An input structure may use arbitrary combinations of the following form factor models:

Keyword      | F(q) calculation
------------ | ----------------------------------------------
`Constant`   | `F(q) = c`
`Polynomial` | `F(q) = a + b * q + ...`
`Gaussian`   | Five gaussian approximation
`Table`      | Array of (q, F(q)) pairs used for interpolation

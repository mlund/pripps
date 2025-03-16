# Partial Rust Implementation of PePSi (Pripps)

This is a tool to calculate the scattering intensity of molecular structures.
Multiple models are supported, including constant, polynomial, gaussian, and table-based models.

## Features

- Multiple form factor models
- Multithreaded
- Trajectory averaging
- File formats: `xyz`, `xtc`
- Unit tested
- Excessive error handling and memory safetiness

## Usage

```sh
# I(q) using the Debye formula
pripps --atomfile assets/single_bead.yaml debye --input protein.xyz
```

Compile and run directly from the source repository (requires a Rust installation):

```sh
cd pripps/
cargo run --release -- --atomfile assets/single_bead.yaml debye --input protein.xyz
```

## Form Factor Models

Atom definitions, which may be atomic or coarse grained, are defined in a separate YAML file,
see e.g. `assets/single_bead.yaml`.
An input structure may use arbitrary combinations of the following form factor models:

Keyword      | Description
------------ | ----------------------------------------------
`Constant`   | 𝐹(𝑞) = 𝑐
`Polynomial` | 𝐹(𝑞) = 𝑎 + 𝑏𝑞 + 𝑐𝑞² + ...
`Gaussian`   | Five gaussian approximation
`Table`      | Array of `[𝑞, 𝐹(𝑞)]` pairs used for interpolation


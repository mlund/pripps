# Partial Rust Implementation of PePSi (Pripps)

This is a tool to calculate the scattering intensity of molecular structures.
Multiple models are supported, including constant, polynomial, gaussian, and table-based models.

## Features

- Multiple form factor models
- Multithreaded
- Weighted trajectory averaging
- File formats: `xyz`, `xtc`
- Unit tested
- Excessive error handling and memory safetiness

## Usage

Compile and run directly from the source repository
[(requires a Rust installation)](https://www.rust-lang.org/tools/install):

```sh
git clone git@github.com:mlund/pripps.git
cd pripps/
cargo run --release -- -h
```

## Form Factor Models

Atom definitions, which may be atomic or coarse grained, are defined in a separate YAML file,
see e.g. `assets/single_bead.yaml`.
An input structure may use arbitrary combinations of the following form factor models:

Description                                | YAML example
------------------------------------------ | ----------------------------------
𝐹(𝑞) = 𝑐                                    | `!Constant 1.0`
𝐹(𝑞) = 𝑎 + 𝑏𝑞 + 𝑐𝑞² + ...                   | `!Polynomial { qrange: [0.0, 1.0], coeff: [1.0, 2.0, 3.0, ...] }`
Array of `[𝑞, 𝐹(𝑞)]` pairs                 | `!Table [ [0.0, 8.9], ...]`
Alias for a previously defined form factor | `!Alias LYS`
[Nine coefficient approximation](https://doi.org/10.1107/S0108767395010038) (four a-b pairs + c) | `!NineCoeff { qrange: [0.0, 1.0], ab: [ [0.5, 0.7], .. ], c = 2.0 }`

# Partial Rust Implementation of PePSi (Pripps)

This is a tool to calculate the scattering intensity of molecular structures.
Multiple models are supported, including constant, polynomial, gaussian, and table-based models.

## Features

- Multiple form factor models
- Multithreaded
- Weighted trajectory averaging
- Unit tested
- Excessive error handling and memory safetiness

## Usage

Explore available options with `--help`. Note that to run directly from the source repository,
replace `pripps` with `cargo run --release --`, see section below.
```sh
pripps --help
```

Calculate the scattering intensity for a coarse grained lysozyme structure
```sh
pripps --atomfile assets/single_bead.yaml --input assets/4lzt.xyz debye
```

### Building from source

This requires a [Rust installation)](https://www.rust-lang.org/tools/install):

```sh
git clone git@github.com:mlund/pripps.git
cd pripps/
cargo run --release -- --help
```

## Form Factor Models

Atom definitions, which may be atomic or coarse grained, are defined in a separate YAML file,
see e.g. `assets/single_bead.yaml`.
Input structures (xyz, gro, pdb) may use species with form factors defined by any combination of the following models:

Description                                | Keyword
------------------------------------------ | ----------
𝐹(𝑞) = 𝑐                                   | `!Constant`
𝐹(𝑞) = 𝑎 + 𝑏𝑞 + 𝑐𝑞² + ...                  | `!Polynomial`
𝐹(𝑞) = 𝑐 + ∑𝑎ᵢ exp(−𝑏ᵢ𝑞²)                  | `!Gaussian`
Array of `[𝑞, 𝐹]` pairs                 | `!Table`
Alias for a previously defined form factor | `!Alias`

### Examples

```yaml
# atoms.yaml
O: !Constant 1.0
N: !Table [ [0.0, 8.9], [0.1, 7.8], ...]
C: !Gaussian { qrange: [0.0, 1.0], ab: [[0.1, 2.0], [0.2, 3.0], ...], c: 0.4 }
CA: !Alias C
ALA: !Polynomial { qrange: [0.0, 0.75], coeff: [8.9, 0.0 ,5.8, ...]}
```

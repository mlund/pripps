# Partial Rust Implementation of PePSi (Pripps)

This is a tool to calculate the scattering intensity of molecular structures.
Multiple models are supported, including constant, polynomial, gaussian, and table-based models.

## Features

- Multiple form factor models
- Multithreaded
- Weighted trajectory averaging
- Unit tested
- Excessive error handling and memory safetiness

## Installing

This requires a [Rust installation)](https://www.rust-lang.org/tools/install):
```sh
cargo install --git git@github.com:mlund/pripps.git
```

## Usage

Explore available options with `--help`. Note that to run directly from the source repository,
replace `pripps` with `cargo run --release --`, see section above.
```sh
pripps --help
```

Calculate the scattering intensity for a coarse grained lysozyme structure
```sh
pripps --atomfile assets/single_bead.yaml --input tests/4lzt/4lzt.xyz debye
```

### Form Factor Models

Atom definitions, which may be atomic or coarse grained, are defined in a separate YAML file,
see e.g. `assets/single_bead.yaml`.
Input structures (xyz, gro, pdb) may use species with form factors defined by any combination of the following models:

Description                                | Keyword
------------------------------------------ | ----------
𝐹(𝑞) = 𝑐                                   | `!Constant`
𝐹(𝑞) = 𝑎 + 𝑏𝑞 + 𝑐𝑞² + ...                  | `!Polynomial`
𝐹(𝑞) = 𝑐 + ∑𝑎ᵢ exp(−𝑏ᵢ𝑞²)                  | `!Gaussian`
𝐹(𝑞) = 𝐹c(𝑞) + ∑𝑤ᵢ𝐹ᵢ(𝑞)⋅sinc(𝑞𝑟ᵢ)          | `!United`
𝐹(𝑞) = ∑ 𝑤ᵢ⋅𝐹ᵢ(𝑞)                          | `!Linear`
Array of `[𝑞, 𝐹]` pairs                    | `!Table`
Alias for a previously defined form factor | `!Alias`

#### Examples

```yaml
# atoms.yaml
# radius is in Ångström and optional
H: { formfactor: !Constant 1.0, radius: 1.0 }
# 'ff' is an alias for 'formfactor'
N: { ff: !Table [ [0.0, 8.9], [0.1, 7.8], ...] }
C: { ff: !Gaussian { qrange: [0.0, 1.0], ab: [[0.1, 2.0], [0.2, 3.0], ...], c: 0.4 } }
# United C and H where H is offset by 1.5 Å and added with weight 1:
CH: { ff: !United { center: C, orbit: [[H, 1.5, 1.0]] } }
CA: { ff: !Alias C }
ALA: { ff: !Polynomial { qrange: [0.0, 0.75], coeff: [8.9, 0.0 ,5.8, ...]} }
```

## Python bindings

The `pripps-py/` subcrate provides Python bindings via [pyo3](https://pyo3.rs) +
[maturin](https://maturin.rs). Install into your Python environment with
[uv](https://docs.astral.sh/uv/):

```sh
cd pripps-py
uv pip install -e .
```

Or without uv:

```sh
cd pripps-py
pip install maturin
maturin develop
```

Minimal example — compute a solvent-corrected SAXS profile:

```python
import pripps

pr = pripps.Pripps(atomfile="assets/single_bead.yaml")
fit = pr.svergun_model(
    structure_file="tests/4lzt/4lzt.xyz",
    qmin=0.0,
    qmax=0.5,
    nq=100,
    excluded="fraser",
    volume_scale=1.0,
    hydration="sasa",
    probe=1.8,
    contrast_density=0.03,
    bulk_electron_density=0.334,
)

# Returns a dict of numpy arrays: q, total, atoms, excluded, hydration
out = fit.intensity(volume_scale=1.0, contrast_density=0.03)
print(out["q"][:3], out["total"][:3])
```

The expensive per-structure work (basis construction, coefficient
triangles, SASA tessellation) runs once inside `svergun_model`.
Each `fit.intensity(volume_scale, contrast_density)` call re-runs only
the cheap coefficient-level combine — fast enough for
`scipy.optimize.minimize` or a grid search over the two fit knobs.

### Environmental variables

The following environmental variables can be used for additional configuration:

Variable        | Description
--------------- | -------------
`FF_TABLE`      | If set formfactors will be exported as `tabulated_formfactors.yaml`
`RUST_LOG`      | Set log level for Rust code (`Debug`, `Trace`, `Warn`, `Info`)

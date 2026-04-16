# Pripps — SAXS Intensity from Molecular Structures

Pripps computes small-angle X-ray scattering (SAXS) intensity
profiles from PDB, XYZ, or GRO structures, with optional solvent
corrections (excluded volume + hydration shell). It is fast enough
for iterative fitting against experimental data.

## Quick start (Python)

### Installation

Requires a Rust toolchain and Python ≥ 3.9.

```sh
git clone https://github.com/mlund/pripps.git
cd pripps/pripps-py
uv pip install -e .          # with uv (recommended)
# pip install maturin && maturin develop   # without uv
```

### Computing a SAXS profile

```python
import pripps

pr = pripps.Pripps(atomfile="assets/single_bead.yaml")
model = pr.svergun_model(
    structure_file="tests/4lzt/4lzt.pdb",
    qmin=0.0,
    qmax=0.5,
    nq=100,
    excluded="fraser",       # Fraser-MacRae-Suzuki excluded volume
    volume_scale=1.0,        # fit knob c₁ (scales excluded volume)
    hydration="sasa",        # FoXS-style SASA-weighted hydration shell
    probe=1.8,               # SASA probe radius (Å)
    contrast_density=0.03,   # fit knob c₂ (scales hydration)
    bulk_electron_density=0.334,  # e/Å³, pure water
)

# Returns a dict of numpy arrays
out = model.intensity(volume_scale=1.0, contrast_density=0.03)
print(out["q"][:3], out["total"][:3])
```

The returned dict contains:

| Key | Description |
|-----|-------------|
| `q` | Scattering vector magnitudes (Å⁻¹) |
| `total` | Total intensity I(q) |
| `atoms` | Protein-only (vacuum) contribution |
| `excluded` | Excluded-volume contribution |
| `hydration` | Hydration-shell contribution |

### Fitting against experimental data

The expensive per-structure work (basis construction, SASA
tessellation, per-q coefficient triangles) runs once in
`svergun_model`. Each `model.intensity(volume_scale,
contrast_density)` call re-runs only the cheap coefficient-level
combine — fast enough for scipy or a grid search.

A full working example is in `pripps-py/examples/fit_lysozyme.py`,
which fits the shipped `tests/4lzt/SASDMF2.dat` dataset:

```sh
python pripps-py/examples/fit_lysozyme.py
```

The core pattern:

```python
import numpy as np
from scipy.optimize import minimize

data = np.loadtxt("experimental.dat", comments="#")
q_exp, i_exp, sigma = data[:, 0], data[:, 1], data[:, 2]

def chi2(params):
    c1, c2 = params
    out = model.intensity(volume_scale=c1, contrast_density=c2)
    theory = np.interp(q_exp, out["q"], out["total"])
    scale = np.sum(theory * i_exp / sigma**2) / np.sum((theory / sigma) ** 2)
    return np.sum(((scale * theory - i_exp) / sigma) ** 2) / len(q_exp)

result = minimize(
    chi2,
    x0=[1.0, 0.03],
    bounds=[(0.95, 1.05), (-0.05, 0.1)],
    method="L-BFGS-B",
)
print(f"c1={result.x[0]:.4f}, c2={result.x[1]:.4f}, χ²/N={result.fun:.2f}")
```

### Solvent models

| Parameter | Values | Description |
|-----------|--------|-------------|
| `excluded` | `"none"`, `"fraser"` | Excluded-volume model |
| `hydration` | `"none"`, `"sasa"` | Hydration-shell model |
| `volume_scale` | float (default 1.0) | Scales excluded volume (PepsiSAXS `r₀`) |
| `contrast_density` | float (default 0.03) | Scales hydration shell (FoXS `c₂`) |
| `probe` | float (default 1.8) | SASA probe radius in Å |
| `bulk_electron_density` | float (default 0.334) | Bulk solvent density in e/Å³ |

## Form factor definitions

Atom definitions (atomic or coarse-grained) are specified in a YAML
file. Each entry maps a species name to a form factor model and
optional physical properties:

```yaml
H: { formfactor: !Constant 1.0, radius: 1.0 }
N: { ff: !Table [ [0.0, 8.9], [0.1, 7.8], ...] }
C: { ff: !Gaussian { qrange: [0.0, 1.0], ab: [[0.1, 2.0]], c: 0.4 },
     radius: 1.577 }
# United atom with explicit displaced_volume (includes implicit H):
NH: { radius: 1.009, displaced_volume: 7.606,
      ff: !United { center: N, orbit: [[H, 1.009, 1]] } }
CA: { ff: !Alias C }
ALA: { ff: !Polynomial { qrange: [0.0, 0.75], coeff: [8.9, 0.0, 5.8, ...]} }
```

Supported form factor models:

| Model | Keyword | Formula |
|-------|---------|---------|
| Constant | `!Constant` | F(q) = c |
| Polynomial | `!Polynomial` | F(q) = a + bq + cq² + … |
| Gaussian | `!Gaussian` | F(q) = c + Σ aᵢ exp(−bᵢq²) |
| United atom | `!United` | F(q) = Fc(q) + Σ wᵢ Fᵢ(q) sinc(q rᵢ) |
| Linear combination | `!Linear` | F(q) = Σ wᵢ Fᵢ(q) |
| Tabulated | `!Table` | Array of [q, F] pairs |
| Alias | `!Alias` | Reference to another entry |

| Property | Unit | Used by | Auto-derived? |
|----------|------|---------|---------------|
| `radius` | Å | SASA hydration, Fraser excluded volume | No — must be specified |
| `displaced_volume` | ų | Fraser excluded volume | Yes — defaults to (4/3)πr³ from `radius` if omitted |

When `radius` is present but `displaced_volume` is omitted, the
displaced volume is automatically derived as the sphere volume
(4/3)πr³. Specify `displaced_volume` explicitly only for united-atom
groups (e.g. NH, SH) where the displaced volume includes implicit
hydrogens that the effective radius does not account for.

## Command-line interface (Rust)

### Installing

Requires a [Rust toolchain](https://www.rust-lang.org/tools/install):

```sh
cargo install --git https://github.com/mlund/pripps.git
```

### Usage

```sh
pripps --help

# Debye formula (exact, O(N²))
pripps -a atoms.yaml -i structure.pdb debye

# Svergun multipole expansion (fast, O(N·L²))
pripps -a atoms.yaml -i structure.pdb svergun

# With solvent correction — CSV output with 5 columns
pripps -a atoms.yaml -i structure.pdb -o intensity.csv \
    svergun --excluded fraser --hydration sasa

# Trajectory averaging
pripps -a atoms.yaml -i structure.pdb -x trajectory.xtc svergun
```

Output is CSV: `q,total` for vacuum, or
`q,total,atoms,excluded,hydration` when solvent is active.

### Environment variables

| Variable | Description |
|----------|-------------|
| `FF_TABLE` | If set, export resolved form factors as `tabulated_formfactors.yaml` |
| `RUST_LOG` | Log level: `Debug`, `Trace`, `Warn`, `Info` |

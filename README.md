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

pr = pripps.Pripps(atomfile="assets/foxs_formfactors.yaml")
model = pr.multipole_model(
    structure_file="tests/lysozyme/6lyz.pdb",
    qmin=0.0,
    qmax=0.5,
    nq=500,
    excluded="foxs",         # FoXS-style excluded volume (16π Fraser)
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
`multipole_model`. Each `model.intensity(volume_scale,
contrast_density)` call re-runs only the cheap coefficient-level
combine — fast enough for scipy or a grid search.

A full working example is in `pripps-py/examples/fit_lysozyme.py`,
which fits lysozyme against the shipped `tests/lysozyme/lyzexp.dat`
experimental dataset and reproduces FoXS's χ²/N ≈ 0.20:

```sh
python pripps-py/examples/fit_lysozyme.py
```

The core pattern:

```python
import numpy as np
from scipy.optimize import minimize

# experimental data: q (Å⁻¹), I(q), σ(q)
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
    x0=[1.0, 0.2],
    bounds=[(0.95, 1.12), (-2.0, 4.0)],
    method="L-BFGS-B",
)
print(f"c1={result.x[0]:.4f}, c2={result.x[1]:.4f}, χ²/N={result.fun:.4f}")
```

### Solvent models

| Parameter | Values | Description |
|-----------|--------|-------------|
| `excluded` | `"disabled"`, `"fraser"`, `"foxs"`, `"grid"`, `"voronoi"` | Excluded-volume model |
| `hydration` | `"disabled"`, `"sasa"`, `"grid"`, `"voronoi"` | Hydration-shell model |
| `volume_scale` | float (default 1.0) | Scales excluded volume (PepsiSAXS `r₀`) |
| `excluded_spacing` | float (default 1.0) | Grid spacing (Å) for `excluded="grid"` |
| `contrast_density` | float (default 0.03) | Scales hydration shell (~0.03 for SASA, ~0.017 for grid) |
| `probe` | float (default 1.8) | SASA/grid shell probe radius in Å |
| `spacing` | float (default 1.5) | Grid spacing (Å) for `hydration="grid"` |
| `thickness` | float (default 3.0) | Shell thickness (Å) for `hydration="grid"` |
| `bulk_electron_density` | float (default 0.334) | Bulk solvent density in e/Å³ |

### Excluded-volume models

The excluded volume accounts for bulk solvent displaced by the
protein. Each model places dummy scatterers whose contribution is
subtracted from the atomic scattering via
`|A − volume_scale·C + contrast_density·B|²`.

| Model | Description | Requires |
|-------|-------------|----------|
| `fraser` | One Gaussian dummy per atom, sized by `displaced_volume` from the YAML file. Uses the PepsiSAXS/CRYSOL 4π denominator. Best for atomic-resolution structures with tabulated volumes. | `displaced_volume` per atom kind |
| `foxs` | Same as `fraser` but with the FoXS 16π denominator, giving a sharper Gaussian that subtracts less at high q. | `displaced_volume` per atom kind |
| `grid` | Fills the molecular envelope with uniform grid cells, each carrying a Fraser Gaussian based on cell volume (`spacing³`). Independent of per-atom displaced volumes. | `radius` per atom kind |
| `voronoi` | Per-atom Fraser Gaussians sized by individual Voronoi cell volumes from the radical tessellation (probe=0). Derives displaced volumes geometrically — no tabulated volumes needed. Achieves χ²≈0.20 on lysozyme, comparable to `fraser`. | `radius` per atom kind |

The `fraser` and `foxs` models use per-atom displaced volumes from
the form-factor YAML (derived from `radius` as `(4/3)πr³` when
`displaced_volume` is not explicitly set). The `grid` and `uniform`
models need only atomic radii.

### Hydration-shell models

The hydration shell accounts for the first water layer around the
protein, which has a higher electron density than bulk solvent.

| Model | Description |
|-------|-------------|
| `sasa` | FoXS-style: one dummy per atom weighted by solvent-accessible surface area fraction (Voronoi tessellation). |
| `grid` | PepsiSAXS-style: water dummies on a Cartesian grid in a shell around the molecular envelope. |
| `voronoi` | Explicit solvent spheres on the molecular surface via voronota-ltr tessellation. Dummies sit at actual water-accessible positions (not atom centers). Probe-insensitive and needs only `radius`. |

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
| Gaussian | `!Gaussian` | F(q) = c + Σ aᵢ exp(−bᵢs²), s = q/(4π) |
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

### Shipped form-factor files

| File | Description | Best for |
|------|-------------|----------|
| `assets/single_bead.yaml` | PepsiSAXS coefficients + coarse-grained residues | Cα / single-bead models, `--excluded fraser` |
| `assets/foxs_formfactors.yaml` | Cromer-Mann (Int. Tables Vol. C), FoXS-compatible volumes and SASA radii | Atomic PDB files, `--excluded foxs` |

Both files use Gaussian b values in the standard s² = (q/4π)²
convention (International Tables / Cromer-Mann). The coefficients
differ slightly (different parameterisations of the same physics)
and the volume/radius tables match their respective tools.

For atomic-resolution PDB files, use `foxs_formfactors.yaml` with
`--excluded foxs` — this reproduces FoXS's χ²/N ≈ 0.29 on
lysozyme (unfitted) and ≈ 0.20 (fitted with hydration).

**PDB atom mapping:** for PDB files, pripps uses residue context
to map each heavy atom to its united-atom form-factor type (e.g.
ALA:CB → CH₃, LYS:NZ → NH₃), folding in implicit hydrogen
scattering. This matching the approach used by FoXS
(`FormFactorTable.cpp:679-715`).

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

# Multipole expansion (fast, O(N·L²))
pripps -a atoms.yaml -i structure.pdb multipole

# With solvent correction — CSV output with 5 columns
pripps -a atoms.yaml -i structure.pdb -o intensity.csv \
    multipole --excluded fraser --hydration sasa

# Fit against experimental data
pripps -a assets/foxs_formfactors.yaml -i structure.pdb \
    -f experimental.dat -o fit.csv \
    multipole --excluded foxs --hydration sasa

# Trajectory averaging
pripps -a atoms.yaml -i structure.pdb -x trajectory.xtc multipole
```

Output is CSV: `q,total` for vacuum, or
`q,total,atoms,excluded,hydration` when solvent is active.
With `--fit`, output is `q,i_exp,i_fit,sigma` at experimental
q points. The experimental file is auto-detected as CSV or
whitespace-separated (3 columns: q, I, σ).

### Environment variables

| Variable | Description |
|----------|-------------|
| `FF_TABLE` | If set, export resolved form factors as `tabulated_formfactors.yaml` |
| `RUST_LOG` | Log level: `Debug`, `Trace`, `Warn`, `Info` |

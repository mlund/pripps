# Pripps — SAXS Intensity from Molecular Structures

Pripps computes small-angle X-ray scattering (SAXS) intensity
profiles from PDB, XYZ, or GRO structures, with optional solvent
corrections (excluded volume + hydration shell). It is fast enough
for iterative fitting against experimental data.

## Quick start (Python)

### Installation

Install straight from GitHub with [uv](https://docs.astral.sh/uv/). The
extension is built from source, so a
[Rust toolchain](https://www.rust-lang.org/tools/install) and Python ≥ 3.9 are
required:

```sh
uv pip install "git+https://github.com/mlund/pripps.git#subdirectory=pripps-py"
```

To also get the example structures and form-factor assets used below, clone the
repo and install in editable mode instead:

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
    excluded="foxs",         			   # FoXS-style excluded volume (16π Fraser)
    volume_scale=1.0,        			   # fit knob c₁ (scales excluded volume)
    hydration="sasa",  # FoXS-style SASA-weighted hydration shell
    probe=1.8,  # SASA probe radius (Å)
    contrast_density=0.03,  # fit knob c₂ (scales hydration)
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

`multipole_model` does the expensive per-structure setup once and
caches the intermediate scattering amplitudes. Subsequent
`model.intensity(volume_scale, contrast_density)` calls only apply
the fit knobs to the cached amplitudes — typically under a
millisecond each — so wrapping them in `scipy.minimize` or a grid
search is fluent.

For small structures or exact reference checks, `debye_model` accepts
the same solvent keyword arguments and exposes the same
`model.intensity(volume_scale, contrast_density)` interface. It caches
the exact Debye self and cross terms for atoms, excluded volume, and
hydration before fitting.
`direct_model` does the same for the explicit discrete-q transform,
caching the complex amplitudes before recombining and averaging
duplicate q magnitudes.

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
| `excluded` | `"disabled"`, `"fraser"`, `"foxs"`, `"grid"`, `"voronoi"`, `"per-kind"` | Excluded-volume model |
| `hydration` | `"disabled"`, `"sasa"`, `"grid"`, `"voronoi"` | Hydration-shell model |
| `volume_scale` | float (default 1.0) | Scales excluded volume (PepsiSAXS `r₀`) |
| `excluded_spacing` | float (default 1.0) | Grid spacing (Å) for `excluded="grid"` |
| `contrast_density` | float (default 0.03) | Fractional hydration-layer excess density (c₂ in `A − c₁·C + c₂·B`) — pepsi's `d_rho / ρ_water`, ~0.1 for a 10 % excess layer |
| `probe` | float (default: 1.8 for SASA/grid, 6.0 for voronoi) | Water probe radius in Å |
| `spacing` | float (default 3.10516) | Grid spacing (Å) for `hydration="grid"`; the bulk-water linear spacing, so each cell holds ~1 water molecule |
| `thickness` | float (default 3.0) | Shell thickness (Å) for `hydration="grid"` |
| `bulk_electron_density` | float (default 0.334) | Bulk solvent density in e/Å³ |

### Excluded-volume models

The excluded volume accounts for bulk solvent displaced by the
protein. Each model places dummy scatterers whose contribution is
subtracted from the atomic scattering via
`|A − volume_scale·C + contrast_density·B|²`.

| Model | Description | Requires |
|-------|-------------|----------|
| `fraser` | One Gaussian dummy per atom, sized by `displaced_volume` from the YAML. Dummy form factor decays faster with q → smaller excluded-volume subtraction at high q. Matches PepsiSAXS / CRYSOL. | `displaced_volume` per atom kind |
| `foxs` | Same formula as `fraser` but with the FoXS 16π denominator: dummy form factor decays more slowly with q → larger excluded-volume subtraction at high q. Matches FoXS. | `displaced_volume` per atom kind |
| `grid` | Fills the molecular envelope with uniform grid cells, each carrying a Fraser Gaussian sized by cell volume (`spacing³`). Independent of per-atom displaced volumes. | `radius` per atom kind |
| `voronoi` | Per-atom Fraser Gaussians sized by individual Voronoi cell volumes. Derives displaced volumes geometrically — no tabulated volumes needed. Achieves χ²≈0.20 on lysozyme, comparable to `fraser`. | `radius` per atom kind |
| `per-kind` | Uses per-kind excluded-solvent form factors stored as `excluded_solvent` beside each atom/residue `formfactor` in the same YAML file. | `excluded_solvent` for referenced atom kinds |

The `fraser` and `foxs` models use per-atom displaced volumes from
the form-factor YAML (derived from `radius` as `(4/3)πr³` when
`displaced_volume` is not explicitly set). The `grid` and `voronoi`
models need only atomic radii.

### Hydration-shell models

The hydration shell accounts for the first water layer around the
protein, which has a higher electron density than bulk solvent.

All three schemes weight their dummies so `weight × F_water(q)`
represents the scattering from bulk-density water in the dummy's
associated volume — so `contrast_density` has a consistent
physical meaning (fractional excess density above bulk) across
schemes. Pepsi's `d_rho` in e/Å³ maps as `contrast_density ≈ d_rho / ρ_water`.

| Model | Description |
|-------|-------------|
| `sasa` | FoXS-style: one dummy per atom, weight = SASA area × 3 Å / 30 Å³ (accessible-area × shell-thickness → bulk-water count). Best with atomic-resolution YAMLs. |
| `grid` | PepsiSAXS-style: water dummies on a Cartesian grid in a shell around the molecular envelope, weight = waters per cell at bulk density. |
| `voronoi` | Explicit solvent spheres on the molecular surface via voronota-ltr, weight = cell volume / 30 Å³. Dummies sit at actual water-accessible positions (not atom centers). Needs only `radius`. Larger probe (default 6 Å) smooths the granular shell into an envelope-like layer that avoids spurious first-minimum interference. |

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
# Linear combination of other entries (no geometry), F(q) = Σ wᵢ Fᵢ(q):
H2O: { ff: !Linear [[O, 1.0], [H, 2.0]] }
ALA: { ff: !Polynomial { qrange: [0.0, 0.75], coeff: [8.9, 0.0, 5.8, ...]},
       excluded_solvent: !Polynomial { qrange: [0.0, 0.75], coeff: [7.1, 0.0, 4.2, ...]} }
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
| `excluded_solvent` | form factor | `per-kind` excluded volume | No — optional; required per species when using `--excluded per-kind` |

When `radius` is present but `displaced_volume` is omitted, the
displaced volume is automatically derived as the sphere volume
(4/3)πr³. Specify `displaced_volume` explicitly only for united-atom
groups (e.g. NH, SH) where the displaced volume includes implicit
hydrogens that the effective radius does not account for.

The optional `excluded_solvent` field stores a *second* form factor
(any of the models above) for the bulk solvent each species displaces,
sitting beside its in-vacuo `formfactor` in the same entry. It is read
only by the `per-kind` excluded-volume model (`--excluded per-kind`, or
`excluded="per-kind"` from Python), which subtracts this form factor —
scaled by `volume_scale` — instead of building a Gaussian dummy from
`radius`/`displaced_volume`. Every species present in the structure
must define it when `--excluded per-kind` is selected.

### Shipped form-factor files

| File | Description | Best for |
|------|-------------|----------|
| `assets/foxs_formfactors.yaml` | Cromer-Mann (Int. Tables Vol. C) atomic form factors, FoXS-compatible volumes and SASA radii | Atomic PDBs with `--excluded foxs` |
| `assets/gaussian_atoms.yaml` | Mixed table: residue single-bead polynomials plus atomic/united-atom Gaussians (CH/CH₂/NH/…) | Atomic PDBs with PepsiSAXS-style solvent models |
| `assets/stovgaard2010.yaml` | Per-residue form-factor centroids (one-body model, q ∈ [0, 0.75] Å⁻¹) from Stovgaard et al. 2010, <https://doi.org/10.1186/1471-2105-11-429> | Residue / Cα single-bead models |
| `assets/poly_amino_acid.yaml` | Amino-acid bead polynomials with a separate `excluded_solvent` polynomial per bead | Cα / single-bead with `--excluded per-kind` |
| `assets/poly_amino_acid_combined.yaml` | Amino-acid bead polynomials with the excluded solvent already folded into the form factor | Cα / single-bead with `--excluded disabled` |
| `assets/poly_martini.yaml` | Martini-3 bead polynomials with a separate `excluded_solvent` polynomial per bead | Martini-3 CG with `--excluded per-kind` |
| `assets/poly_martini_combined.yaml` | Martini-3 bead polynomials with the excluded solvent already folded into the form factor | Martini-3 CG with `--excluded disabled` |

Gaussian files use b values in the standard s² = (q/4π)² convention
(International Tables / Cromer-Mann). The atomic coefficients differ
(Cromer-Mann vs Stovgaard-2010) and the volume/radius tables match
their respective tools.

**Don't mix files with non-matching `--excluded` flags.** Each
YAML + excluded-volume convention is a self-consistent bundle:
`foxs_formfactors.yaml` was calibrated jointly with FoXS's
`K = 16π` Gaussian, while `gaussian_atoms.yaml` was calibrated with
PepsiSAXS's `K = 4π`. Swapping the denominator gives atomic and
excluded contributions that were never fit together and yields
subtly biased intensities.

For atomic-resolution PDB files, use `foxs_formfactors.yaml` with
`--excluded foxs` — this reproduces FoXS's χ²/N ≈ 0.29 on
lysozyme (unfitted) and ≈ 0.20 (fitted with hydration).

**PDB atom mapping:** for PDB files, pripps uses residue context
to map each heavy atom to its united-atom form-factor type (e.g.
ALA:CB → CH₃, LYS:NZ → NH₃), folding in implicit hydrogen
scattering — same convention as FoXS.

## Performance

The multipole expansion is **linear in atom count**, so it beats
the `O(N²)` Debye schedule above a few hundred atoms and stays
practical for atomic-resolution PDBs up into the hundreds of
thousands of atoms.

- **Auto-selected truncation order per q-point.** Low q uses
  only a handful of multipoles, high q uses a convergent upper
  bound. The default is conservative on non-globular shapes
  (elongated complexes, stacked rings, multi-domain assemblies)
  where radius-of-gyration-based heuristics in other tools can
  silently under-converge at high q.
- **Auto-selected q-grid density at the Nyquist rate** (`2·L + 1`
  points) when `--nq` is omitted. The theory `I(q)` is band-limited
  by the multipole truncation, so sampling any denser just computes
  redundant points — experimental q values are reached via natural
  cubic-spline interpolation on the theory grid. Shannon-Nyquist
  sampling applied to SAXS is discussed by Moore,
  [*J. Appl. Cryst.* **13**, 168-175 (1980)](https://doi.org/10.1107/S002188988001179X).
- **Parallel across CPU cores and SIMD-vectorised.** No tuning
  needed; the workload scales with hardware.
- **Cached fit setup.** `multipole_model` amortises the
  per-structure work so fit loops spend their time in the cheap
  recombination, not the expensive scattering calculation.

Illustrative wall-clock on a 90 672-atom structure, 100 q-points,
full atomic-resolution form factors (vacuum):

| Hardware | wall |
|---|---:|
| 10-core Apple Silicon laptop | ~4 s |
| 64-thread AMD Threadripper 3970X (`-C target-cpu=native`) | ~9 s |

### Tuning knobs

- `--lmax <N>` (CLI) or `l_max=N` (Python) overrides the
  auto-selected truncation order. Useful for reproducing a fixed
  reference, timing sweeps, or trading a little high-q accuracy
  for extra speed on very elongated structures — the default is
  correct but conservative for those.
- `--nq <N>` (CLI) overrides the auto-selected q-grid density.
  The default `2·L_max + 1` is enough to represent the band-
  limited theory without aliasing; pass a larger value to
  oversample the output curve for visualisation.

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

# Multipole expansion (linear in atoms, truncation auto-selected)
pripps -a atoms.yaml -i structure.pdb multipole
pripps -a atoms.yaml -i structure.pdb multipole --lmax 68  # fixed order

# With solvent correction — CSV output with 5 columns
pripps -a atoms.yaml -i structure.pdb -o intensity.csv \
    multipole --excluded fraser --hydration sasa
pripps -a atoms.yaml -i structure.pdb -o intensity.csv \
    debye --excluded fraser --hydration sasa

# Fit against experimental data (box-constrained Nelder-Mead over
# c₁ ∈ [0.5, 2.0] and c₂ ∈ [0.0, 0.1]; warns if the fit hits a bound)
pripps -a assets/foxs_formfactors.yaml -i structure.pdb \
    -f experimental.dat -o fit.csv \
    multipole --excluded foxs --hydration sasa
pripps -a assets/foxs_formfactors.yaml -i structure.pdb \
    -f experimental.dat -o fit.csv \
    debye --excluded foxs --hydration sasa
pripps -a assets/foxs_formfactors.yaml -i structure.pdb \
    -f experimental.dat -o fit.csv \
    direct --excluded foxs --hydration sasa

# Trajectory averaging — use `direct` so each frame's periodic box
# is honoured (multipole ignores PBC).
pripps -a atoms.yaml -i structure.pdb -x trajectory.xtc direct
```

### Choosing a scheme

| Scheme | Best for | Notes |
|---|---|---|
| `multipole` | Single structures, fitting loops | Fastest for large atomic PDBs; no periodic boundaries. |
| `direct` | MD trajectories with periodic box | Evaluates I(q) on discrete Miller-index q-vectors; supports the same solvent fit knobs for single-frame fits. |
| `debye` | Small clusters, reference checks | Exact pair-distance sum with the same solvent and fit knobs as multipole; quadratic in atoms, fine under a few hundred. |

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

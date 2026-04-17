#!/usr/bin/env python3
"""Fit (volume_scale, contrast_density) against experimental lysozyme SAXS data.

Uses the FoXS-compatible Cromer-Mann form factors, the FoXS-style
excluded-volume formula (16π denominator), and SASA-weighted
hydration. Reproduces FoXS's χ²/N ≈ 0.20 on lysozyme.

Run from the repository root (after `cd pripps-py && uv pip install -e .`):

    python pripps-py/examples/fit_lysozyme.py
"""

import numpy as np
import pripps
from scipy.optimize import minimize

# Load experimental data (q, I, sigma).
# The lyzexp.dat file has a text header line and a trailing byte;
# parse manually to be robust.
data = []
for line in open("tests/lysozyme/lyzexp.dat"):
    line = line.strip()
    if not line or line[0].isalpha():
        continue
    try:
        parts = line.split()
        data.append([float(parts[0]), float(parts[1]), float(parts[2])])
    except (ValueError, IndexError):
        pass
data = np.array(data)
q_exp, i_exp, sigma = data[:, 0], data[:, 1], data[:, 2]

# Prepare the Svergun model — expensive work runs once here.
# Uses FoXS-compatible Cromer-Mann form factors with the 16π
# Fraser excluded-volume formula and SASA hydration.
pr = pripps.Pripps(atomfile="assets/foxs_formfactors.yaml")
model = pr.svergun_model(
    structure_file="tests/lysozyme/6lyz.pdb",
    qmin=0.0,
    qmax=float(q_exp[-1]),
    nq=500,
    excluded="foxs",
    hydration="sasa",
)


def chi2(params):
    """χ² between the theoretical curve and experimental data."""
    c1, c2 = params
    out = model.intensity(volume_scale=c1, contrast_density=c2)
    theory = np.interp(q_exp, out["q"], out["total"])
    scale = np.sum(theory * i_exp / sigma**2) / np.sum((theory / sigma) ** 2)
    return np.sum(((scale * theory - i_exp) / sigma) ** 2) / len(q_exp)


# Grid search to find a good starting point
print("Grid search over (volume_scale, contrast_density)...")
best_chi2 = float("inf")
best_params = (1.0, 0.0)
for c1 in np.linspace(0.95, 1.12, 18):
    for c2 in np.linspace(-2.0, 4.0, 61):
        val = chi2([c1, c2])
        if val < best_chi2:
            best_chi2 = val
            best_params = (c1, c2)
print(f"  Best grid point: c1={best_params[0]:.3f}, c2={best_params[1]:.3f}, χ²/N={best_chi2:.4f}")

# Refine with L-BFGS-B
result = minimize(
    chi2,
    x0=list(best_params),
    bounds=[(0.95, 1.12), (-2.0, 4.0)],
    method="L-BFGS-B",
)
c1_opt, c2_opt = result.x
print(f"  Optimised:       c1={c1_opt:.4f}, c2={c2_opt:.4f}, χ²/N={result.fun:.4f}")
print(f"  FoXS reference:  χ²/N ≈ 0.20")

# Write the fitted curve
out = model.intensity(volume_scale=c1_opt, contrast_density=c2_opt)
theory = np.interp(q_exp, out["q"], out["total"])
scale = np.sum(theory * i_exp / sigma**2) / np.sum((theory / sigma) ** 2)

np.savetxt(
    "fit_lysozyme.csv",
    np.column_stack([q_exp, i_exp, scale * theory]),
    delimiter=",",
    header="q,i_exp,i_fit",
    comments="",
)
print(f"  Fitted curve written to fit_lysozyme.csv (scale={scale:.6e})")

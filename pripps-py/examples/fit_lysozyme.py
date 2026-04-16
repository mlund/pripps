#!/usr/bin/env python3
"""Fit (volume_scale, contrast_density) against experimental lysozyme SAXS data.

Uses the SASDMF2 dataset shipped in the repository and the
single-bead coarse-grained form factors. Demonstrates the full
pripps Python workflow: prepare a model once, then call
`model.intensity(c1, c2)` cheaply inside scipy's minimiser.

Run from the repository root (after `cd pripps-py && uv pip install -e .`):

    python pripps-py/examples/fit_lysozyme.py
"""

import numpy as np
import pripps
from scipy.optimize import minimize

# Load experimental data (q, I, sigma) — skip comment lines.
# Clip to q ≤ 0.5 Å⁻¹: the coarse-grained form factors in
# single_bead.yaml are valid up to 0.75, and the signal-to-noise
# of this dataset is best below ~0.5.
QMAX = 0.5
data = np.loadtxt("tests/4lzt/SASDMF2.dat", comments="#")
mask = data[:, 0] <= QMAX
q_exp = data[mask, 0]
i_exp = data[mask, 1]
sigma = data[mask, 2]

# Prepare the Svergun model — expensive work runs once here
pr = pripps.Pripps(atomfile="assets/single_bead.yaml")
model = pr.svergun_model(
    structure_file="tests/4lzt/4lzt.pdb",
    qmin=float(q_exp[0]),
    qmax=QMAX,
    nq=200,
    excluded="fraser",
    hydration="sasa",
)

def chi2(params):
    """χ² between the theoretical curve and experimental data."""
    c1, c2 = params
    out = model.intensity(volume_scale=c1, contrast_density=c2)
    # Interpolate the model onto the experimental q grid
    theory = np.interp(q_exp, out["q"], out["total"])
    # Optimal linear scale factor (analytical minimum)
    scale = np.sum(theory * i_exp / sigma**2) / np.sum((theory / sigma) ** 2)
    return np.sum(((scale * theory - i_exp) / sigma) ** 2) / len(q_exp)

# Grid search to find a good starting point
print("Grid search over (volume_scale, contrast_density)...")
best_chi2 = float("inf")
best_params = (1.0, 0.0)
for c1 in np.linspace(0.90, 1.10, 21):
    for c2 in np.linspace(-0.05, 0.10, 16):
        val = chi2([c1, c2])
        if val < best_chi2:
            best_chi2 = val
            best_params = (c1, c2)
print(f"  Best grid point: c1={best_params[0]:.3f}, c2={best_params[1]:.4f}, χ²/N={best_chi2:.4f}")

# Refine with L-BFGS-B
result = minimize(
    chi2,
    x0=list(best_params),
    bounds=[(0.85, 1.15), (-0.10, 0.15)],
    method="L-BFGS-B",
)
c1_opt, c2_opt = result.x
print(f"  Optimised:       c1={c1_opt:.4f}, c2={c2_opt:.4f}, χ²/N={result.fun:.4f}")

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
print(f"  Fitted curve written to fit_lysozyme.csv (scale={scale:.6f})")

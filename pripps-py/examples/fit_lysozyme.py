#!/usr/bin/env python3
"""Fit (volume_scale, contrast_density) against experimental lysozyme SAXS data.

Uses the FoXS-compatible Cromer-Mann form factors, the FoXS-style
excluded-volume formula (16π denominator), and SASA-weighted
hydration. Reproduces FoXS's χ²/N ≈ 0.20 on lysozyme.

Demonstrates BOTH fitting styles:
  (a) the built-in `model.fit(..., method="nelder-mead")` convenience, and
  (b) a manual `scipy.optimize.minimize` over the typed `Intensity`
      result (`.q` / `.total`) for full control.

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

# Prepare the Multipole model — expensive work runs once here.
# Uses FoXS-compatible Cromer-Mann form factors with the 16π
# Fraser excluded-volume formula and SASA hydration. Model choice
# may be a string ("foxs") or an enum (pripps.Excluded.Foxs).
pr = pripps.Pripps(atomfile="assets/foxs_formfactors.yaml")
model = pr.multipole_model(
    structure_file="tests/lysozyme/6lyz.pdb",
    qmin=0.0,
    qmax=float(q_exp[-1]),
    nq=500,
    excluded=pripps.Excluded.Foxs,
    hydration=pripps.Hydration.Sasa,
)

# (a) Built-in fit — one call, returns a typed FitResult.
fit = model.fit(q_exp, i_exp, sigma, method="nelder-mead")
print("Built-in model.fit():")
print(f"  {fit!r}")
print("  FoXS reference: χ²/N ≈ 0.20")


# (b) Manual fit with scipy for full control over the objective.
# The typed Intensity gives numpy `.q` / `.total` directly.
def chi2(params):
    c1, c2 = params
    out = model.intensity(volume_scale=c1, contrast_density=c2)
    theory = np.interp(q_exp, out.q, out.total)
    scale = np.sum(theory * i_exp / sigma**2) / np.sum((theory / sigma) ** 2)
    return np.sum(((scale * theory - i_exp) / sigma) ** 2) / len(q_exp)


result = minimize(
    chi2,
    x0=[1.0, 0.0],  # neutral start — independent of (a), to confirm they agree
    bounds=[(0.95, 1.12), (-2.0, 4.0)],
    method="L-BFGS-B",
)
print("Manual scipy.minimize():")
print(f"  c1={result.x[0]:.4f}, c2={result.x[1]:.4f}, χ²/N={result.fun:.4f}")

# Write the fitted curve (using the built-in fit's parameters).
out = model.intensity(
    volume_scale=fit.volume_scale, contrast_density=fit.contrast_density
)
theory = np.interp(q_exp, out.q, out.total)
np.savetxt(
    "fit_lysozyme.csv",
    np.column_stack([q_exp, i_exp, fit.scale * theory]),
    delimiter=",",
    header="q,i_exp,i_fit",
    comments="",
)
print(f"Fitted curve written to fit_lysozyme.csv (scale={fit.scale:.6e})")

# --- Trajectory averaging (one-shot; the only path that reads an XTC) ---
# Uncomment with a real trajectory to average I(q) over frames:
# avg = pr.intensity(
#     structure_file="tests/lysozyme/6lyz.pdb",
#     model="multipole",
#     xtc="trajectory.xtc",
#     stride=10,
#     excluded="foxs",
#     hydration="sasa",
#     contrast_density=0.03,
# )
# print(avg.q, avg.total)   # note: trajectory averages have no per-contribution breakdown

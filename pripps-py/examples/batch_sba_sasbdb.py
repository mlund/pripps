#!/usr/bin/env python3
"""Run the testing_single-bead_approximation.ipynb workflow over SASBDB cache entries.

For each cache directory containing both:
  - raw_cleaned.xyz or raw_cleaned_martini3_aa.pdb
  - exp.dat

this script fits (volume_scale, contrast_density) with the same setup
as testing_single-bead_approximation.ipynb:
  - atomfile: assets/poly_amino_acid.yaml (or poly_martini.yaml for Martini3)
  - excluded: polynomial using the atomfile's excluded_solvent entries
  - hydration: disabled
  - optimizer: L-BFGS-B with bounds [(0.95, 1.12), (-2.0, 4.0)] or change them in the script
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pripps
from scipy.optimize import minimize


def load_exp_data(path: Path, qmax: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse q, I, sigma from a loose 3-column SAXS file."""
    rows: list[tuple[float, float, float]] = []
    with path.open() as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 3:
                continue
            try:
                q, i_exp, sigma = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            if sigma <= 0.0 or q < 0.0 or q > qmax:
                continue
            rows.append((q, i_exp, sigma))

    if not rows:
        raise ValueError(f"no valid q/I/sigma rows found in {path}")

    data = np.array(rows, dtype=float)
    return data[:, 0], data[:, 1], data[:, 2]


def weighted_scale(theory: np.ndarray, i_exp: np.ndarray, sigma: np.ndarray) -> float:
    num = np.sum(theory * i_exp / sigma**2)
    den = np.sum((theory / sigma) ** 2)
    return float(num / den)


def fit_single(
    pr: pripps.Pripps,
    xyz_path: Path,
    exp_path: Path,
    qmax: float,
    nq: int,
) -> dict[str, float | np.ndarray]:
    q_exp, i_exp, sigma = load_exp_data(exp_path, qmax=qmax)

    model = pr.debye_model(
        structure_file=str(xyz_path),
        qmin=0.0,
        qmax=qmax,
        nq=nq,
        excluded="polynomial",
        hydration="grid",
        volume_scale=1.0,
        probe=1.8,
        contrast_density=0.03,
        bulk_electron_density=0.334,
        spacing = 3.8,
    )

    def chi2(params: np.ndarray) -> float:
        c1, c2 = float(params[0]), float(params[1])
        out = model.intensity(volume_scale=c1, contrast_density=c2)
        theory = np.interp(q_exp, out["q"], out["total"])
        scale = weighted_scale(theory, i_exp, sigma)
        return float(np.sum(((scale * theory - i_exp) / sigma) ** 2) / len(q_exp))

    result = minimize(
        chi2,
        x0=[1.0, 0.2],
        bounds=[(-2.0, 3.0), (-2.0, 4.0)],
        method="L-BFGS-B",
    )

    c1_opt, c2_opt = float(result.x[0]), float(result.x[1])
    out = model.intensity(volume_scale=c1_opt, contrast_density=c2_opt)
    theory_interp = np.interp(q_exp, out["q"], out["total"])
    scale = weighted_scale(theory_interp, i_exp, sigma)
    i_fit = scale * theory_interp

    return {
        "c1": c1_opt,
        "c2": c2_opt,
        "chi2": float(result.fun),
        "scale": scale,
        "q_exp": q_exp,
        "i_exp": i_exp,
        "sigma": sigma,
        "i_fit": i_fit,
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=repo_root / "sasbdb_bench" / "cache",
        help="Directory with SASBDB entry subdirectories (default: %(default)s)",
    )
    parser.add_argument(
        "--atomfile",
        type=Path,
        default=repo_root / "assets" / "poly_martini.yaml",
        help="Form-factor YAML with formfactor and excluded_solvent entries",
    )
    parser.add_argument(
        "--qmax",
        type=float,
        default=0.5,
        help="Upper q limit (Å^-1) used for both model and data filtering",
    )
    parser.add_argument(
        "--nq",
        type=int,
        default=500,
        help="Number of q points in Debye model",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of SASBDB IDs to process",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional output path for summary CSV (default: <cache-dir>/sba_summary.csv)",
    )
    args = parser.parse_args()

    cache_dir = args.cache_dir.resolve()
    summary_csv = args.summary_csv or (cache_dir / "sba_summary_martini.csv")

    pr = pripps.Pripps(atomfile=str(args.atomfile))

    entries = sorted(d for d in cache_dir.iterdir() if d.is_dir())
    if args.only:
        keep = set(args.only)
        entries = [d for d in entries if d.name in keep]

    summary_rows: list[dict[str, str | float | int]] = []

    for entry in entries:
        sid = entry.name
        xyz_path = entry / "raw_cleaned_martini3_aa.pdb"
        exp_path = entry / "exp.dat"
        if not xyz_path.is_file() or not exp_path.is_file():
            continue

        try:
            fit = fit_single(
                pr=pr,
                xyz_path=xyz_path,
                exp_path=exp_path,
                qmax=float(args.qmax),
                nq=int(args.nq),
            )
        except Exception as exc:
            print(f"{sid}: FAILED ({exc})")
            continue

        fit_csv = entry / "sba_fit_martini.csv"
        np.savetxt(
            fit_csv,
            np.column_stack([fit["q_exp"], fit["i_exp"], fit["i_fit"], fit["sigma"]]),
            delimiter=",",
            header="q,i_exp,i_fit,sigma",
            comments="",
        )

        print(
            f"{sid}: c1={fit['c1']:.4f}, c2={fit['c2']:.4f}, "
            f"chi2/N={fit['chi2']:.4f}, n={len(fit['q_exp'])}"
        )

        summary_rows.append(
            {
                "sid": sid,
                "c1": fit["c1"],
                "c2": fit["c2"],
                "chi2_per_point": fit["chi2"],
                "scale": fit["scale"],
                "n_points": len(fit["q_exp"]),
                "xyz_file": str(xyz_path),
                "exp_file": str(exp_path),
                "fit_csv": str(fit_csv),
            }
        )

    if not summary_rows:
        raise SystemExit("No datasets were processed.")

    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with summary_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sid",
                "c1",
                "c2",
                "chi2_per_point",
                "scale",
                "n_points",
                "xyz_file",
                "exp_file",
                "fit_csv",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"Wrote summary: {summary_csv}")


if __name__ == "__main__":
    main()

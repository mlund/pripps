#!/usr/bin/env python3
"""Batch single-bead-approximation (SBA) fits over SASBDB cache entries.

Runs the testing_single-bead_approximation.ipynb workflow against a
directory of cached SASBDB (Small Angle Scattering Biological Data Bank)
entries.

For each cache directory containing both:
  - raw_cleaned.xyz or raw_cleaned_martini3_aa.pdb
  - exp.dat

this script fits (volume_scale, contrast_density) with the same setup
as testing_single-bead_approximation.ipynb:
  - atomfile: assets/poly_amino_acid.yaml (or poly_martini.yaml for Martini3)
  - excluded: per-kind using the atomfile's excluded_solvent entries
  - hydration: grid
  - fit: built-in `model.fit(q, I, sigma)` (pripps' library optimizer)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pripps


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
        excluded="per-kind",
        hydration="grid",
        probe=1.8,
        bulk_electron_density=0.334,
        spacing=3.8,
    )

    # Built-in fit of (volume_scale, contrast_density) against the data.
    fit = model.fit(q_exp, i_exp, sigma)

    out = model.intensity(
        volume_scale=fit.volume_scale, contrast_density=fit.contrast_density
    )
    theory_interp = np.interp(q_exp, out.q, out.total)
    i_fit = fit.scale * theory_interp

    return {
        "c1": fit.volume_scale,
        "c2": fit.contrast_density,
        "chi2": fit.chi2,
        "scale": fit.scale,
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

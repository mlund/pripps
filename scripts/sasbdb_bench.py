#!/usr/bin/env python3
"""Bootstrap SASBDB benchmark for pripps.

Downloads experimental curves and deposited atomic models for the
19 main-paper SASBDB entries in Lytje & Pedersen (2025) Fig. 5,
normalises each model with `pdbfixer`, runs pripps with
`voronoi` excluded-volume and hydration, and tabulates χ²/N.

All network and pdbfixer outputs are cached under
`sasbdb_bench/cache/<SASD_ID>/` so re-runs skip any stage whose
output already exists. Use `--refetch` to force a re-download.

Usage:
    python scripts/sasbdb_bench.py                         # run everything
    python scripts/sasbdb_bench.py --only SASDJG5 SASDJQ4  # subset
    python scripts/sasbdb_bench.py --refetch               # redownload

Requires:
    pripps binary on $PATH or at ./target/release/pripps
    pdbfixer on $PATH (OpenMM)
    assets/foxs_formfactors.yaml relative to repo root
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import urllib.request

# Atomic-resolution SASBDB IDs from Lytje & Pedersen (2025) Fig. 5.
# Additional IDs listed in Fig. S1 of the supporting information are
# not yet included here — add them once Table S2 is available.
SASBDB_IDS = [
    "SASDA92",
    "SASDAW3",
    "SASDEM6",
    "SASDF86",
    "SASDJF5",
    "SASDJG5",
    "SASDJQ4",
    "SASDJU5",
    "SASDKG4",
    "SASDL82",
    "SASDMB5",
    "SASDME4",
    "SASDMZ9",
    "SASDP39",
    "SASDT75",
    "SASDT85",
    "SASDT95",
    # SASDEL9 is a dummy-residue envelope model, not atomistic —
    # excluded from the atomistic comparison.
    # SASDDD3 is a hard dataset where the deposited PDB disagrees
    # with the deposited curve (pepsi and pripps both get χ² ≈ 78,
    # vs the paper's Fig. 5 value of 5–8 from a different input).
]

REPO = pathlib.Path(__file__).resolve().parent.parent
BENCH = REPO / "sasbdb_bench"
CACHE = BENCH / "cache"
FF_YAML_DEFAULT = REPO / "assets" / "foxs_formfactors.yaml"
# Override via --atomfile if you want gaussian_atoms.yaml for the
# PepsiSAXS-style K=4π convention.

SASBDB_API = "https://www.sasbdb.org/rest-api/entry/summary/{}/"


def pripps_binary():
    """Find the pripps binary — prefer local release build, fall back to PATH."""
    local = REPO / "target" / "release" / "pripps"
    if local.is_file() and os.access(local, os.X_OK):
        return str(local)
    found = shutil.which("pripps")
    if found is None:
        sys.exit("pripps binary not found; build with `cargo build --release`")
    return found


def fetch(url, dest, refetch=False):
    """Download `url` to `dest` unless it already exists."""
    if dest.is_file() and not refetch:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def fetch_meta(sid, refetch=False):
    """Fetch SASBDB entry JSON, return the parsed dict."""
    path = CACHE / sid / "meta.json"
    fetch(SASBDB_API.format(sid), path, refetch=refetch)
    return json.loads(path.read_text())


def preclean_pdb(src, dst):
    """Strip `MODEL` / `ENDMDL` records: some SASBDB depositions use
    malformed widths (e.g. 'MODEL 1 ' with trailing spaces) that
    break pdbfixer's fixed-column parser. Safe to drop — we want a
    single conformer anyway."""
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.startswith(("MODEL", "ENDMDL")):
                continue
            fout.write(line)


def strip_explicit_hydrogens(src, dst):
    """Remove ATOM / HETATM lines for hydrogen atoms. pripps' YAML
    uses united-atom form factors that fold implicit H scattering
    into the heavy-atom entries (ALA:CB → CH₃ etc.), so explicit Hs
    in the PDB would be looked up as unknown atom names."""
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            if line.startswith(("ATOM  ", "HETATM")):
                # Element column is 77-78 (1-indexed); atom name is 13-16.
                element = line[76:78].strip() if len(line) >= 78 else ""
                name = line[12:16].strip()
                is_h = element == "H" or (
                    not element and name.lstrip("0123456789").startswith("H")
                )
                if is_h:
                    continue
            fout.write(line)


def filter_expdata(src, dst, q_max=6.0, q_scale=1.0):
    """Drop rows with σ ≤ 0 or q > `q_max` from a SAXS .dat file,
    optionally multiplying q by `q_scale` first (use 0.1 to convert
    nm⁻¹ → Å⁻¹). Sigma ≤ 0 makes χ² NaN and crashes the Nelder-Mead
    fit; q > q_max (after scaling) falls outside the form-factor
    YAML's defined range (shipped YAMLs use qrange [0.0, 6.0])."""
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                fout.write(line)
                continue
            parts = stripped.split()
            if len(parts) < 3:
                fout.write(line)
                continue
            try:
                q = float(parts[0]) * q_scale
                intensity = parts[1]
                sigma = float(parts[2])
            except ValueError:
                fout.write(line)
                continue
            if sigma > 0 and q <= q_max:
                fout.write(f"{q:.6e} {intensity} {sigma:.6e}\n")


def preprocess(sid, refetch=False):
    """Ensure `exp.dat` and `final.pdb` exist for `sid`. Returns the
    paths, or (None, None) if the entry has no atomistic model."""
    d = CACHE / sid
    d.mkdir(parents=True, exist_ok=True)
    meta = fetch_meta(sid, refetch=refetch)

    fits = meta.get("fits") or []
    if not fits or not fits[0].get("models"):
        print(f"{sid}: no fitted model", file=sys.stderr)
        return None, None
    m = fits[0]["models"][0]
    if m.get("type_of_model") != "atomic":
        print(
            f"{sid}: skipping, model type = {m.get('type_of_model')}", file=sys.stderr
        )
        return None, None

    raw_exp = d / "exp_raw.dat"
    fetch(meta["intensities_data"], raw_exp, refetch=refetch)
    # SASBDB stores `angular_unit` per entry. Convert nm⁻¹ → Å⁻¹
    # so pripps (which assumes Å⁻¹) interprets q consistently with
    # the structure, whose coordinates are always in Å.
    angular_unit = meta.get("angular_unit", "1/A")
    q_scale = 0.1 if angular_unit == "1/nm" else 1.0
    exp = d / "exp.dat"
    if not exp.is_file() or refetch:
        filter_expdata(raw_exp, exp, q_scale=q_scale)

    raw = d / "raw.pdb"
    fetch(m["model_data"], raw, refetch=refetch)

    # Stage 1: preclean MODEL records so pdbfixer can parse.
    cleaned = d / "raw_cleaned.pdb"
    if not cleaned.is_file() or refetch:
        preclean_pdb(raw, cleaned)

    # Stage 2: pdbfixer (missing heavy atoms, gaps, nonstandard residues,
    # drop heterogens). Explicit hydrogens may remain from the input.
    fixed = d / "fixed.pdb"
    if not fixed.is_file() or refetch:
        cmd = [
            "pdbfixer",
            str(cleaned),
            "--add-atoms=heavy",
            "--add-residues",
            "--replace-nonstandard",
            "--keep-heterogens=none",
            f"--output={fixed}",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    # Stage 3: strip explicit hydrogens — pripps' foxs YAML uses
    # united-atom form factors and doesn't register H2/HG1/HA2/…
    final = d / "final.pdb"
    if not final.is_file() or refetch:
        strip_explicit_hydrogens(fixed, final)
    return exp, final


def parse_expdata_qmax(path):
    """Find the last q value in a SAXS .dat file (3 columns: q, I, σ).
    Skip headers and blank lines. Used to size pripps' q-grid to match."""
    qmax = 0.0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                q = float(parts[0])
                if q > qmax:
                    qmax = q
            except ValueError:
                continue
    return qmax


def run_pripps(exp, fixed, out_csv, log_path, pripps, atomfile, excluded, hydration):
    """Invoke pripps with fit mode. Returns (returncode, stderr_text)."""
    qmax = parse_expdata_qmax(exp)
    # Cap at 6.0 Å⁻¹ — the atomic form-factor Gaussians in the
    # shipped YAMLs are defined over qrange [0.0, 6.0]. Going past
    # that errors out.
    qmax = max(min(qmax * 1.05, 6.0), 0.5)
    env = os.environ.copy()
    env["RUST_LOG"] = "info"
    cmd = [
        pripps,
        "-a",
        str(atomfile),
        "-i",
        str(fixed),
        "-f",
        str(exp),
        "-o",
        str(out_csv),
        "multipole",
        "--qmax",
        f"{qmax:.3f}",
        # Intentionally no --nq: pripps auto-selects at the
        # Shannon-Nyquist rate `2·L_max + 1` (floored at 21).
        "--excluded",
        excluded,
        "--hydration",
        hydration,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    log_path.write_text(r.stderr)
    return r.returncode, r.stderr


def chi2_from_fit_csv(csv_path):
    """Compute reduced χ² from a pripps fit CSV (q, I_exp, I_fit, σ).
    Pripps already applies the analytical scale factor to I_fit."""
    import math

    n, total = 0, 0.0
    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i == 0 or not line.strip():
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            try:
                _, i_exp, i_fit, sigma = (float(x) for x in parts[:4])
            except ValueError:
                continue
            if sigma <= 0.0:
                continue
            total += ((i_fit - i_exp) / sigma) ** 2
            n += 1
    return (total / n) if n else math.nan, n


def run_pepsi(pepsi_bin, fixed, exp, out_dir):
    """Run local Pepsi-SAXS on the same (structure, data) pair pripps
    is benchmarked against. Returns χ² (pepsi's own `Chi^2` line) or
    NaN on failure. Cached per-entry as `pepsi_fit.log`."""
    import math

    log_path = out_dir / "pepsi_fit.log"
    out_path = out_dir / "pepsi_fit.out"
    if not log_path.is_file():
        # Pepsi writes output relative to its own cwd; run it in the entry dir.
        r = subprocess.run(
            [
                str(pepsi_bin),
                str(fixed.resolve()),
                str(exp.resolve()),
                "-o",
                str(out_path.resolve()),
            ],
            cwd=str(out_dir),
            capture_output=True,
            text=True,
        )
        log_path.write_text(r.stdout + "\n---STDERR---\n" + r.stderr)
        if r.returncode != 0:
            return math.nan
    text = log_path.read_text()
    for line in text.splitlines():
        if line.startswith("Chi^2"):
            parts = line.rsplit(":", 1)
            if len(parts) == 2:
                try:
                    return float(parts[1].strip())
                except ValueError:
                    pass
    return math.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", nargs="*", help="subset of SASBDB IDs to run")
    ap.add_argument(
        "--refetch", action="store_true", help="ignore cache and re-download everything"
    )
    ap.add_argument(
        "--excluded",
        default="foxs",
        choices=["disabled", "fraser", "foxs", "grid", "voronoi"],
    )
    ap.add_argument(
        "--hydration", default="sasa", choices=["disabled", "sasa", "grid", "voronoi"]
    )
    ap.add_argument(
        "--atomfile",
        type=pathlib.Path,
        default=FF_YAML_DEFAULT,
        help=f"form-factor YAML (default: {FF_YAML_DEFAULT.relative_to(REPO)})",
    )
    ap.add_argument(
        "--label",
        default=None,
        help="tag baked into fit output filename (default: "
        "'<atomfile-stem>_<excluded>_<hydration>')",
    )
    ap.add_argument(
        "--with-pepsi",
        action="store_true",
        help="also run Pepsi-SAXS on each entry for side-by-side comparison",
    )
    ap.add_argument(
        "--pepsi",
        default=str(pathlib.Path("~/bin/Pepsi-SAXS").expanduser()),
        help="path to Pepsi-SAXS binary (default: ~/bin/Pepsi-SAXS)",
    )
    args = ap.parse_args()

    args.atomfile = args.atomfile.resolve()
    if args.label is None:
        args.label = f"{args.atomfile.stem}_{args.excluded}_{args.hydration}"
    if not args.atomfile.is_file():
        sys.exit(f"atomfile not found: {args.atomfile}")

    ids = args.only if args.only else SASBDB_IDS
    pripps = pripps_binary()
    pepsi = pathlib.Path(args.pepsi).expanduser()
    if args.with_pepsi and not pepsi.is_file():
        sys.exit(f"pepsi binary not found: {pepsi}")
    CACHE.mkdir(parents=True, exist_ok=True)

    print(f"# pripps: {pripps}")
    try:
        af_display = args.atomfile.relative_to(REPO)
    except ValueError:
        af_display = args.atomfile
    print(f"# atomfile = {af_display}")
    print(f"# excluded = {args.excluded}, hydration = {args.hydration}")
    print(f"# label    = {args.label}")
    if args.with_pepsi:
        print(f"# pepsi    = {pepsi}")
    print(f"# cache dir: {CACHE}")
    print()
    hdr = f"{'SASBDB':<9} {'n_atoms':>8} {'n_pts':>6} {'pripps':>8}"
    if args.with_pepsi:
        hdr += f" {'pepsi':>8}"
    hdr += "  status"
    print(hdr)
    print("-" * (70 if args.with_pepsi else 60))

    def fmt(val, cached=False, w=8):
        if val != val:  # NaN
            return f"{'—':>{w}}"
        return f"{val:{w}.3f}"

    results = []
    for sid in ids:
        try:
            exp, fixed = preprocess(sid, refetch=args.refetch)
        except Exception as e:
            row = f"{sid:<9} {'—':>8} {'—':>6} {'—':>8}"
            if args.with_pepsi:
                row += f" {'—':>8}"
            print(f"{row}  fetch error: {e}")
            continue
        if exp is None:
            row = f"{sid:<9} {'—':>8} {'—':>6} {'—':>8}"
            if args.with_pepsi:
                row += f" {'—':>8}"
            print(f"{row}  skipped (no atomic model)")
            continue

        out_csv = CACHE / sid / f"fit_{args.label}.csv"
        log_path = CACHE / sid / f"fit_{args.label}.log"

        with open(fixed) as f:
            n_atoms = sum(1 for L in f if L.startswith(("ATOM", "HETATM")))

        # pripps
        status = "ok"
        if out_csv.is_file() and not args.refetch:
            chi2, npts = chi2_from_fit_csv(out_csv)
            status = "cached"
        else:
            rc, stderr = run_pripps(
                exp,
                fixed,
                out_csv,
                log_path,
                pripps,
                args.atomfile,
                args.excluded,
                args.hydration,
            )
            if rc != 0:
                short = stderr.strip().splitlines()[-1] if stderr.strip() else "unknown"
                row = f"{sid:<9} {n_atoms:>8} {'—':>6} {'—':>8}"
                if args.with_pepsi:
                    row += f" {'—':>8}"
                print(f"{row}  pripps failed: {short[:30]}")
                continue
            chi2, npts = chi2_from_fit_csv(out_csv)

        # pepsi
        pepsi_chi2 = float("nan")
        if args.with_pepsi:
            pepsi_chi2 = run_pepsi(pepsi, fixed, exp, CACHE / sid)

        row = f"{sid:<9} {n_atoms:>8} {npts:>6} {fmt(chi2)}"
        if args.with_pepsi:
            row += f" {fmt(pepsi_chi2)}"
        print(f"{row}  {status}")
        results.append((sid, n_atoms, npts, chi2, pepsi_chi2))

    if results:
        import math

        pripps_vals = sorted(r[3] for r in results if r[3] == r[3])
        pepsi_vals = sorted(r[4] for r in results if r[4] == r[4])
        med_p = pripps_vals[len(pripps_vals) // 2] if pripps_vals else math.nan
        print()
        print(f"# {len(results)} entries, pripps median χ²/N = {med_p:.3f}", end="")
        if args.with_pepsi and pepsi_vals:
            med_pe = pepsi_vals[len(pepsi_vals) // 2]
            print(f", pepsi median χ²/N = {med_pe:.3f}")
        else:
            print()


if __name__ == "__main__":
    main()

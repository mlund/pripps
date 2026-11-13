# Pepsi-SAXS excluded-volume reference fixture (4lzt)

Committed reference for the excluded-volume cross-check test
(`pripps_excluded_matches_pepsi_ev_fixture` in `src/lib.rs`). Generated once
with the Pepsi-SAXS binary; the test reads only the committed `.dat`/`.log`,
never the binary.

## Generation

```
Pepsi-SAXS 4lzt.pdb lyzexp.dat --dro 0 -o 4lzt_pepsi_ev.dat
```

- Binary: Pepsi-SAXS version 2.6 (29 Jan 2020), `~/bin/Pepsi-SAXS`.
- `--dro 0` fixes the hydration-shell contrast to 0, isolating atoms +
  excluded volume so the comparison targets the Fraser model directly.
- Fitted parameters (from `4lzt_pepsi_ev.log`): mean atomic radius
  `rm = 1.61072 Å`, best `r0 = 1.58495 Å`, `d_rho = 0`, `Chi2 = 0.75`.

## Convention conversion

Pepsi scales the atomic **radius** (`γ = r0/rm`); pripps scales the displaced
**volume** (`c1 = volume_scale`), with `V ∝ r³`, so

```
c1 = (r0/rm)³ = (1.58495/1.61072)³ = 0.95277
```

The public v2.6 binary has no `--r0scale` flag and auto-fits `r0`, which for
4lzt lands near `rm`; `--dro 0` pulls it to `γ ≈ 0.984`, giving a modest but
non-trivial width-correction offset. The strict validator for the first-order
Taylor width term is the analytic exact-width test in `src/debye.rs`
(`first_order_beats_constant_width_and_converges_quadratically`); this fixture
is a cross-code sanity anchor for the full excluded-volume pipeline.

## Columns (`.dat`)

`q   I_exp   dI_exp   I_fit` — the test uses `q` (col 0) and `I_fit` (col 3).

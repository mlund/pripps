"""Smoke tests for the pripps Python bindings.

Run from the `pripps-py/` directory after `maturin develop`:

    cd pripps-py && maturin develop && python -m unittest tests/test_smoke.py
"""

import unittest

import numpy as np
import pripps


def load_lyzexp():
    """Load experimental lysozyme SAXS data (q, I, sigma)."""
    data = []
    for line in open("../tests/lysozyme/lyzexp.dat"):
        line = line.strip()
        if not line or line[0].isalpha():
            continue
        try:
            parts = line.split()
            data.append([float(parts[0]), float(parts[1]), float(parts[2])])
        except (ValueError, IndexError):
            pass
    return np.array(data)


def chi2_per_point(q_theory, i_theory, exp):
    """Compute χ²/N between theoretical and experimental curves."""
    q_exp, i_exp, sigma = exp[:, 0], exp[:, 1], exp[:, 2]
    theory = np.interp(q_exp, q_theory, i_theory)
    scale = np.sum(theory * i_exp / sigma**2) / np.sum((theory / sigma) ** 2)
    return np.sum(((scale * theory - i_exp) / sigma) ** 2) / len(q_exp)


class TestMultipoleModel(unittest.TestCase):
    def test_prepare_and_call(self):
        pr = pripps.Pripps("../assets/poly_amino_acid.yaml")
        fit = pr.multipole_model(
            "../tests/lysozyme/4lzt.xyz",
            qmin=0.0,
            qmax=0.5,
            nq=50,
            excluded="fraser",
            hydration="sasa",
        )
        out = fit.intensity(1.0, 0.03)
        self.assertEqual(out["q"].shape, (50,))
        self.assertEqual(out["total"].shape, (50,))
        self.assertTrue(np.all(np.isfinite(out["total"])))
        self.assertTrue(np.all(out["atoms"] > 0))
        self.assertIsNone(fit.grid_hydration_points)

    def test_refit_produces_different_curves(self):
        pr = pripps.Pripps("../assets/poly_amino_acid.yaml")
        fit = pr.multipole_model(
            "../tests/lysozyme/4lzt.xyz",
            qmin=0.0,
            qmax=0.5,
            nq=50,
            excluded="fraser",
            hydration="sasa",
        )
        a = fit.intensity(0.95, 0.02)
        b = fit.intensity(1.05, 0.05)
        self.assertFalse(np.allclose(a["total"], b["total"]))

    def test_bad_excluded_raises(self):
        pr = pripps.Pripps("../assets/poly_amino_acid.yaml")
        with self.assertRaises(ValueError) as ctx:
            pr.multipole_model(
                "../tests/lysozyme/4lzt.xyz",
                qmin=0.0,
                qmax=0.5,
                nq=10,
                excluded="garbage",
            )
        self.assertIn("garbage", str(ctx.exception))


class TestPepsiFit(unittest.TestCase):
    """PepsiSAXS-style fit: Fraser excluded volume (4π) + grid hydration.

    Uses 4lzt.pdb to match the PepsiSAXS v3.0 fitted reference:

        Pepsi-SAXS 4lzt.pdb lyzexp.dat

    Pepsi fits r0 and d_rho via a 100×100 grid search and achieves
    χ² = 0.272 (r0=1.614, d_rho=0.0334 e/Å³, probe=3 Å,
    thickness=3 Å, 381 shell points).

    Pripps with L-BFGS-B over (c1, c2) achieves χ²/N ≈ 0.20
    (c1=1.012, c2=0.014). PepsiSAXS default d_rho is 5% of bulk
    density = 0.05 × 0.334 ≈ 0.017 e/Å³.
    """

    # PepsiSAXS default: 5% of bulk electron density
    PEPSI_DEFAULT_CONTRAST: float = 0.05 * 0.334

    @classmethod
    def setUpClass(cls):
        cls.exp = load_lyzexp()
        q_max = float(cls.exp[-1, 0])
        pr = pripps.Pripps("../assets/gaussian_atoms.yaml")
        cls.model = pr.multipole_model(
            "../tests/lysozyme/4lzt.pdb",
            qmin=0.0,
            qmax=q_max,
            nq=500,
            excluded="fraser",
            hydration="grid",
            probe=3.0,
            spacing=4.0,
            thickness=3.0,
            contrast_density=cls.PEPSI_DEFAULT_CONTRAST,
        )

    def test_pepsi_fit_chi2(self):
        """Fitted χ²/N should be competitive with PepsiSAXS (~0.27)."""
        from scipy.optimize import minimize

        def objective(params):
            c1, c2 = params
            out = self.model.intensity(volume_scale=c1, contrast_density=c2)
            return chi2_per_point(out["q"], out["total"], self.exp)

        result = minimize(
            objective,
            x0=[1.0, self.PEPSI_DEFAULT_CONTRAST],
            bounds=[(0.95, 1.12), (-2.0, 4.0)],
            method="L-BFGS-B",
        )
        chi2 = result.fun
        # PepsiSAXS v3.0 gets χ²/N = 0.272; pripps gets ≈ 0.20
        self.assertLess(chi2, 0.5, f"PepsiSAXS-style fit χ²/N = {chi2:.4f}")

    def test_grid_hydration_populates_contributions(self):
        """Grid hydration should populate atoms, excluded, and hydration."""
        out = self.model.intensity(volume_scale=1.0, contrast_density=self.PEPSI_DEFAULT_CONTRAST)
        self.assertIn("atoms", out)
        self.assertIn("excluded", out)
        self.assertIn("hydration", out)
        self.assertTrue(np.all(np.isfinite(out["total"])))
        self.assertIsInstance(self.model.grid_hydration_points, int)
        self.assertGreater(self.model.grid_hydration_points, 0)


if __name__ == "__main__":
    unittest.main()

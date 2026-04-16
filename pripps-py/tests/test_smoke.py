"""Smoke tests for the pripps Python bindings.

Run from the `pripps-py/` directory after `maturin develop`:

    cd pripps-py && maturin develop && python -m unittest tests/test_smoke.py
"""

import unittest

import numpy as np
import pripps


class TestSvergunModel(unittest.TestCase):
    def test_prepare_and_call(self):
        pr = pripps.Pripps("../assets/single_bead.yaml")
        fit = pr.svergun_model(
            "../tests/4lzt/4lzt.xyz",
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

    def test_refit_produces_different_curves(self):
        pr = pripps.Pripps("../assets/single_bead.yaml")
        fit = pr.svergun_model(
            "../tests/4lzt/4lzt.xyz",
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
        pr = pripps.Pripps("../assets/single_bead.yaml")
        with self.assertRaises(ValueError) as ctx:
            pr.svergun_model(
                "../tests/4lzt/4lzt.xyz",
                qmin=0.0,
                qmax=0.5,
                nq=10,
                excluded="garbage",
            )
        self.assertIn("garbage", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

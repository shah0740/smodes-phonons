"""Validated against two real ISOTROPY/ISODISTORT INVARIANTS outputs for
Ca3Mn2O7 (space group 139, I4/mmm; irreps GM5-, X2+, X3-) -- differing only
in which order-parameter direction was chosen for the degenerate X-point
irreps, confirming the parsed polynomial form itself doesn't depend on that
choice. Fitting and prediction are validated against synthetic data built
from known coefficients (exact recovery, no noise) rather than real DFT
energies, which this project doesn't compute itself.
"""

import unittest
from pathlib import Path

import numpy as np

from smodes_phonons.landau import (
    LandauError,
    fit_landau_coefficients,
    parse_data_table,
    parse_inline_terms,
    parse_invariants,
    polynomial_energy,
    polynomial_gradient,
    predict_equilibrium,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / name).read_text()


class TestParseInvariants(unittest.TestCase):
    EXPECTED_TERMS = {
        (2, 0, 0), (0, 2, 0), (0, 0, 2),
        (1, 1, 1),
        (4, 0, 0), (2, 2, 0), (0, 4, 0), (2, 0, 2), (0, 2, 2), (0, 0, 4),
    }

    def test_p1_example_parses_correctly(self):
        inv = parse_invariants(load("CMO_invariants_P1.txt"))
        self.assertEqual(inv.variables, ["a", "b", "c"])
        self.assertEqual(set(inv.terms), self.EXPECTED_TERMS)
        self.assertEqual(len(inv.terms), 10)
        self.assertEqual(inv.space_group_number, 139)
        self.assertEqual(inv.space_group_symbol, "I4/mmm")

    def test_p3_example_gives_the_identical_term_set(self):
        # Different order-parameter direction choice for X2+/X3- (P3 vs
        # P1), same resulting invariant polynomial form.
        inv = parse_invariants(load("CMO_invariants_P3.txt"))
        self.assertEqual(set(inv.terms), self.EXPECTED_TERMS)

    def test_irrep_cross_reference_parsed(self):
        inv = parse_invariants(load("CMO_invariants_P1.txt"))
        labels = {irrep.variable: irrep.irrep_label for irrep in inv.irreps}
        self.assertEqual(labels, {"a": "GM5-", "b": "X2+", "c": "X3-"})

    def test_term_label_formatting(self):
        inv = parse_invariants(load("CMO_invariants_P1.txt"))
        self.assertIn("a b c", [inv.term_label(t) for t in inv.terms])
        self.assertIn("a^4", [inv.term_label(t) for t in inv.terms])

    def test_missing_invariants_section_raises(self):
        with self.assertRaises(LandauError):
            parse_invariants("Space Group: 139 I4/mmm D4h-17\nno polynomials here\n")

    def test_degree_mismatch_raises(self):
        text = "Invariant Polynomials\nDegree 2\n\na3\n"  # a^3 under a "Degree 2" header
        with self.assertRaises(LandauError):
            parse_invariants(text)

    def test_unparseable_token_raises(self):
        text = "Invariant Polynomials\nDegree 2\n\n$$\n"
        with self.assertRaises(LandauError):
            parse_invariants(text)

    def test_empty_polynomial_section_raises(self):
        text = "Invariant Polynomials\n"
        with self.assertRaises(LandauError):
            parse_invariants(text)


class TestParseInlineTerms(unittest.TestCase):
    def test_comma_separated(self):
        inv = parse_inline_terms("a2,b2,c2,abc,a4,a2b2,b4,a2c2,b2c2,c4")
        self.assertEqual(inv.variables, ["a", "b", "c"])
        self.assertEqual(set(inv.terms), TestParseInvariants.EXPECTED_TERMS)

    def test_whitespace_separated(self):
        inv = parse_inline_terms("a2 b2 c2 abc a4")
        self.assertEqual(set(inv.terms), {(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 1), (4, 0, 0)})

    def test_matches_real_invariants_file(self):
        from_file = parse_invariants(load("CMO_invariants_P1.txt"))
        inline = parse_inline_terms("a2,b2,c2,abc,a4,a2b2,b4,a2c2,b2c2,c4")
        self.assertEqual(set(from_file.terms), set(inline.terms))
        self.assertEqual(from_file.variables, inline.variables)

    def test_empty_raises(self):
        with self.assertRaises(LandauError):
            parse_inline_terms("   ")

    def test_malformed_token_raises(self):
        with self.assertRaises(LandauError):
            parse_inline_terms("a2, $$")


class TestFitLandauCoefficients(unittest.TestCase):
    """Exact recovery from noise-free synthetic data -- the fit is a
    linear system, so with enough independent data points it should
    recover known coefficients to floating-point precision."""

    def setUp(self):
        self.invariants = parse_inline_terms("a2,b2,c2,abc,a4,a2b2,b4,a2c2,b2c2,c4")
        self.terms = [tuple(0 for _ in self.invariants.variables)] + list(self.invariants.terms)
        self.true_coefficients = np.array([5.0, -2.0, -1.0, 0.5, 0.3, 1.0, 0.2, 1.0, 0.1, 0.15, 1.0])

    def _true_energy(self, a, b, c):
        from smodes_phonons.landau import _monomial_value

        resolved = np.array([a, b, c])
        return float(sum(coef * _monomial_value(t, resolved) for coef, t in zip(self.true_coefficients, self.terms)))

    def _synthetic_data(self, n=30, seed=0):
        rng = np.random.default_rng(seed)
        data = []
        for _ in range(n):
            a, b, c = rng.uniform(-1, 1, size=3)
            data.append(({"a": a, "b": b, "c": c}, self._true_energy(a, b, c)))
        return data

    def test_recovers_known_coefficients_exactly(self):
        fit = fit_landau_coefficients(self.invariants, self._synthetic_data())
        np.testing.assert_allclose(fit.coefficients, self.true_coefficients, atol=1e-8)
        self.assertAlmostEqual(fit.r_squared, 1.0, places=10)

    def test_dict_and_sequence_value_forms_agree(self):
        data_dict = self._synthetic_data(n=15, seed=1)
        data_seq = [((v["a"], v["b"], v["c"]), e) for v, e in data_dict]
        fit_dict = fit_landau_coefficients(self.invariants, data_dict)
        fit_seq = fit_landau_coefficients(self.invariants, data_seq)
        np.testing.assert_allclose(fit_dict.coefficients, fit_seq.coefficients, atol=1e-10)

    def test_too_few_data_points_raises(self):
        with self.assertRaises(LandauError):
            fit_landau_coefficients(self.invariants, self._synthetic_data(n=3))

    def test_single_data_point_raises(self):
        with self.assertRaises(LandauError):
            fit_landau_coefficients(self.invariants, [({"a": 0.0, "b": 0.0, "c": 0.0}, 1.0)])

    def test_missing_variable_in_data_point_raises(self):
        with self.assertRaises(LandauError):
            fit_landau_coefficients(self.invariants, self._synthetic_data() + [({"a": 0.1, "b": 0.1}, 1.0)])


class TestPredictEquilibrium(unittest.TestCase):
    def test_simple_stable_quadratic_predicts_origin(self):
        invariants = parse_inline_terms("a2")
        data = [({"a": a}, 1.0 + 3.0 * a**2) for a in np.linspace(-1, 1, 10)]
        fit = fit_landau_coefficients(invariants, data)
        equilibrium = predict_equilibrium(fit, bound=1.0, n_starts=5)
        self.assertAlmostEqual(equilibrium["a"], 0.0, places=6)

    def test_double_well_predicts_nonzero_lower_energy_than_origin(self):
        # E = -a^2 + a^4 has minima at a = +-1/sqrt(2), both lower than E(0)=0.
        invariants = parse_inline_terms("a2,a4")
        data = [({"a": a}, -(a**2) + a**4) for a in np.linspace(-1.5, 1.5, 12)]
        fit = fit_landau_coefficients(invariants, data)
        equilibrium = predict_equilibrium(fit, bound=1.0, n_starts=15)
        self.assertAlmostEqual(abs(equilibrium["a"]), 1.0 / np.sqrt(2), places=4)
        self.assertLess(polynomial_energy(fit, equilibrium), polynomial_energy(fit, {"a": 0.0}))

    def test_gradient_is_zero_at_the_predicted_equilibrium(self):
        invariants = parse_inline_terms("a2,b2,ab,a4,b4")
        rng = np.random.default_rng(2)
        terms = [(), (2, 0), (0, 2), (1, 1), (4, 0), (0, 4)]
        true_coeffs = np.array([0.0, -1.0, -1.0, 0.4, 1.0, 1.0])
        from smodes_phonons.landau import _monomial_value

        data = []
        for _ in range(20):
            a, b = rng.uniform(-1, 1, size=2)
            e = float(sum(c * _monomial_value(t, np.array([a, b])) for c, t in zip(true_coeffs, terms)))
            data.append(({"a": a, "b": b}, e))
        fit = fit_landau_coefficients(invariants, data)
        equilibrium = predict_equilibrium(fit, bound=1.0, n_starts=20)
        grad = polynomial_gradient(fit, equilibrium)
        self.assertLess(float(np.linalg.norm(grad)), 1e-4)


class TestParseDataTable(unittest.TestCase):
    def test_tab_delimited(self):
        text = "a\tb\tc\tenergy\n0.0\t0.0\t0.0\t-10.0\n0.1\t0.0\t0.0\t-9.9\n"
        data = parse_data_table(text, ["a", "b", "c"])
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0], ({"a": 0.0, "b": 0.0, "c": 0.0}, -10.0))

    def test_comma_delimited(self):
        text = "a,b,energy\n0.0,0.0,-10.0\n0.1,0.0,-9.9\n"
        data = parse_data_table(text, ["a", "b"])
        self.assertEqual(len(data), 2)

    def test_column_order_independent_of_variables_order(self):
        text = "energy\tb\ta\n-10.0\t0.0\t0.0\n"
        data = parse_data_table(text, ["a", "b"])
        self.assertEqual(data[0], ({"a": 0.0, "b": 0.0}, -10.0))

    def test_comments_and_blank_lines_skipped(self):
        text = "# comment\na,energy\n\n0.1,-9.9\n"
        data = parse_data_table(text, ["a"])
        self.assertEqual(len(data), 1)

    def test_missing_column_raises(self):
        with self.assertRaises(LandauError):
            parse_data_table("a,energy\n0.0,-10.0\n", ["a", "b"])

    def test_empty_table_raises(self):
        with self.assertRaises(LandauError):
            parse_data_table("", ["a"])

    def test_malformed_number_raises(self):
        with self.assertRaises(LandauError):
            parse_data_table("a,energy\nfoo,-10.0\n", ["a"])


if __name__ == "__main__":
    unittest.main()

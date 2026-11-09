"""the developer notes: sum of all irrep 'total' values == 3 * N_primitive_parent * n_star_arms.

Catches a parser silently dropping an irrep whose header formatting differs
from the ones already seen -- a failure mode that otherwise produces output
that looks entirely reasonable (fewer irreps, no error).
"""

import unittest
from pathlib import Path

from smodes_phonons import parse_smodes_output

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# (fixture filename, N_primitive_parent, n_star_arms, expected sum)
SUM_RULE_CASES = [
    ("STO_findsym_Gamma", 5, 1, 15),
    ("STO_findsym_R", 5, 1, 15),
    ("STO_findsym_M", 5, 3, 45),
    ("TiO2_findsym_GM", 6, 1, 18),
    ("CMO_findsym_X", 12, 2, 72),
]


class TestModeCountSumRule(unittest.TestCase):
    def test_sum_rule_holds_for_every_fixture(self):
        for name, n_prim, arms, expected in SUM_RULE_CASES:
            with self.subTest(fixture=name):
                out = parse_smodes_output((FIXTURES / name).read_text())
                actual = sum(ir.total_modes for ir in out.irreps)
                expected_from_rule = 3 * n_prim * arms
                self.assertEqual(expected, expected_from_rule)
                self.assertEqual(
                    actual,
                    expected_from_rule,
                    f"{name}: sum of total_modes across {len(out.irreps)} irreps is "
                    f"{actual}, expected 3*{n_prim}*{arms}={expected_from_rule}",
                )

    def test_per_irrep_total_equals_degeneracy_times_modes(self):
        for name, *_ in SUM_RULE_CASES:
            out = parse_smodes_output((FIXTURES / name).read_text())
            for ir in out.irreps:
                with self.subTest(fixture=name, irrep=ir.label):
                    # n_modes is derived as total_modes // degeneracy, so also
                    # check against the actual number of parsed mode blocks --
                    # that's the check that would catch the mode-parsing loop
                    # itself drifting out of sync with the header.
                    self.assertEqual(ir.total_modes, ir.degeneracy * ir.n_modes)
                    # `len(ir.modes)` equals `n_modes` only for an ORDINARY
                    # irrep. A COMPLEX PAIR (`D1+D2+`) prints every partner
                    # rather than one representative, so it has `degeneracy`
                    # times as many blocks -- real HfO2 D has 18 for 9 distinct
                    # modes. Asserting the ordinary relation there is simply
                    # false; it passes today only because no committed fixture
                    # has a complex irrep, which is luck rather than design.
                    expected_blocks = ir.n_modes * (ir.degeneracy if ir.is_complex_pair else 1)
                    self.assertEqual(len(ir.modes), expected_blocks)


if __name__ == "__main__":
    unittest.main()


# (smodes fixture, parent POSCAR, findsym log, N_primitive, n_star_arms)
OPD_CONSTRUCTION_CASES = [
    ("STO_findsym_Gamma", "POSCAR_STO", "findsym_7.1.6_STO.log", 5, 1),
    ("STO_findsym_M", "POSCAR_STO", "findsym_7.1.6_STO.log", 5, 3),
    ("STO_findsym_R", "POSCAR_STO", "findsym_7.1.6_STO.log", 5, 1),
    ("CMO_findsym_X", "POSCAR_CMO", "findsym_7.1.6_CMO.log", 12, 2),
    ("TiO2_findsym_X", "POSCAR_TiO2", "findsym_7.1.6_TiO2.log", 6, 2),
    ("graphite_findsym_K", "POSCAR_graphite", "findsym_7.1.6_graphite.log", 4, 2),
    ("CsV3Sb5_findsym_K", "POSCAR_CsV3Sb5", "findsym_7.1.6_CsV3Sb5.log", 9, 2),
]


class TestSumRuleAppliedToOpdConstruction(unittest.TestCase):
    """The SAME sum rule, applied to the OPD CONSTRUCTION rather than the parser.

    `TestModeCountSumRule` above checks that `parse_smodes_output` doesn't
    silently drop an irrep. It says nothing about whether the direction
    machinery then actually BUILDS every order-parameter direction those
    irreps declare -- and for a long time it did not, silently:

        k-point       sum rule   hosted (before)
        TiO2 X            36          22
        CsV3Sb5 K         54          46

    Both shortfalls were within-arm partners being dropped without error.
    `tile_blend_components_onto_joint_cell` derives its within-arm target
    directions from mode 0 and reuses them for every mode, which only holds
    while every mode of the irrep shares mode 0's orbit; real TiO2 X1/X2 and
    CsV3Sb5 K5/K6 have modes in genuinely different orbits, so
    `construct_partner` correctly raised for those and they were skipped.
    Fixed by a per-mode fallback (a mode that cannot reach mode 0's targets
    searches for its own), but the reason it went unnoticed for so long is
    that nothing ever counted the constructed OPDs against the sum rule.

    Physicist's own framing, which is what this test encodes: STO's M point
    has 10 atoms in its cell, so 30 modes, times 3 for degeneracy, halved
    because the other half are the Gamma-point modes -- 45 OPDs at M. That
    is exactly `3 * N_primitive * n_star_arms` and exactly what the
    construction must host.

    Bloch quadrature partners ARE counted here. At a time-reversal-paired
    k-point (`2k` not integral -- graphite/CsV3Sb5 K) the star's two arms
    come back exactly parallel (cos = +1.0000, measured), so the quadrature
    SUBSTITUTES for the redundant arm rather than adding to a full arm set.
    Real graphite K1 (degeneracy 2) is then `(a,0)` = arm 1 and `(0,a)` =
    arm 1's quadrature -- two genuinely orthogonal directions -- instead of
    the four (spanning only two) that hosting both arms produced.
    """

    def test_every_declared_opd_is_actually_constructed(self):
        from smodes_phonons import Structure
        from smodes_phonons.decomposition import build_joint_opd_cell

        for name, poscar, log_name, n_prim, arms in OPD_CONSTRUCTION_CASES:
            with self.subTest(fixture=name):
                out = parse_smodes_output((FIXTURES / name).read_text())
                parent = Structure.from_poscar(FIXTURES / poscar)
                findsym_log = (FIXTURES / log_name).read_text()

                declared = sum(ir.total_modes for ir in out.irreps)
                self.assertEqual(
                    declared, 3 * n_prim * arms,
                    f"{name}: fixture itself disagrees with the sum rule",
                )

                _reference, patterns, _n_components = build_joint_opd_cell(
                    list(out.irreps), parent, findsym_log_text=findsym_log,
                )
                # Count EVERY hosted pattern. A Bloch quadrature partner
                # substitutes for a redundant arm at a time-reversal-paired
                # k-point rather than adding to a full arm set, so it is a
                # member of the sum rule's own count, not an extra.
                hosted = len(patterns)
                self.assertEqual(
                    hosted, declared,
                    f"{name}: joint cell hosts {hosted} order-parameter directions, "
                    f"but the irreps there declare {declared} "
                    f"(3*{n_prim}*{arms}) -- some OPD was silently dropped",
                )

    def test_quadrature_partners_appear_only_where_k_is_not_self_inverse(self):
        """A quadrature partner is real physics at a time-reversal-paired
        k-point and must NOT be invented anywhere else. At a self-inverse k
        (`2k` integral) the phase factors are purely real and the quadrature
        vanishes identically, so `bloch_quadrature_partner` returns None and
        the count here must be exactly zero -- confirming the guard holds
        without any k-point-family special-casing.
        """
        import numpy as np

        from smodes_phonons import Structure
        from smodes_phonons.decomposition import build_joint_opd_cell

        for name, poscar, log_name, _n_prim, _arms in OPD_CONSTRUCTION_CASES:
            with self.subTest(fixture=name):
                out = parse_smodes_output((FIXTURES / name).read_text())
                parent = Structure.from_poscar(FIXTURES / poscar)
                _ref, patterns, _n = build_joint_opd_cell(
                    list(out.irreps), parent,
                    findsym_log_text=(FIXTURES / log_name).read_text(),
                )
                quadrature = sum(1 for (_l, _a, key) in patterns if "quad" in key)

                k = np.array([float(x) for x in out.irreps[0].k_vector])
                self_inverse = bool(np.allclose(2.0 * k, np.round(2.0 * k), atol=1e-6))
                if self_inverse:
                    self.assertEqual(
                        quadrature, 0,
                        f"{name}: k={k} is self-inverse, so a quadrature partner "
                        "cannot exist, yet some were constructed",
                    )
                else:
                    self.assertGreater(
                        quadrature, 0,
                        f"{name}: k={k} is time-reversal-paired, so every OPD "
                        "should gain a quadrature partner, but none were built",
                    )

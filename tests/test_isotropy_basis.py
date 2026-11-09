"""v2.0 OPD basis built from ISOTROPY's irrep matrices (isotropy_basis.py).

Subprocess-free: every `iso` output is a saved real one
(fixtures/isotropy_basis/iso_cache, named by the script's key), with each
material's real POSCAR, findsym 7.1.6 log and smodes output.

The four materials are the hardest mapping cases under the old basis: STO X5-
(45-degree rule), NaCl L3+ (30 degrees plus a mirrored arm), Si L3+
(non-symmorphic, origin choice 2, per-arm signs) and graphite K5 (hexagonal
quadrature). End to end through the CLI, ISOTROPY's own letters reproduce every
one of ISOTROPY's strata with this basis: 22/22, 22/22, 22/22, 7/7
(2026-11-02, letters_transfer.py).
"""

import unittest
from pathlib import Path

import numpy as np

from smodes_phonons import isotropy_basis as ib
from smodes_phonons import smodes
from smodes_phonons.structure import Structure

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "isotropy_basis"
CASES = {
    # material: (space group, k-point file, irrep)
    "STO": (221, "smodes_X", "X5-"),
    "NaCl": (225, "smodes_L", "L3+"),
    "Si": (227, "smodes_L", "L3+"),
    "graphite": (194, "smodes_K", "K5"),
}


def _fixture_runner(script: str) -> str:
    path = FIXTURES / "iso_cache" / f"{ib.script_key(script)}.txt"
    if not path.is_file():
        raise AssertionError(f"no saved iso output for this script (key {ib.script_key(script)}):\n{script}")
    return path.read_text()


def _load(material):
    sg, smodes_file, label = CASES[material]
    parent = Structure.from_poscar(FIXTURES / material / "POSCAR")
    log = (FIXTURES / material / "findsym.log").read_text()
    irrep = smodes.parse_smodes_output((FIXTURES / material / smodes_file).read_text()).irrep(label)
    return sg, parent, log, irrep


class _WithFixtureRunner(unittest.TestCase):
    def setUp(self):
        self._previous = ib._iso_runner
        ib.set_iso_runner(_fixture_runner)
        self.addCleanup(ib.set_iso_runner, self._previous)


class TestParsers(_WithFixtureRunner):
    def test_cartesian_operations_are_transposed_into_column_convention(self):
        sg, parent, log, _irrep = _load("STO")
        frame = ib.standard_frame(parent, log)
        ops = ib.parse_cartesian_operations(ib._run(ib.build_operations_input(sg, frame.standard_lattice_parameters)))
        self.assertEqual([op.name for op in ops], ["C4z+", "C31+", "I"])
        # iso PRINTS C4z+ as [[0,1,0],[-1,0,0],[0,0,1]]; acting on column
        # vectors (x -> y, y -> -x) it is the transpose:
        np.testing.assert_allclose(ops[0].rotation_cartesian, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-9)

    def test_matrices_are_snapped_to_exact_values(self):
        sg, parent, log, irrep = _load("NaCl")
        rep = ib.representation(irrep, parent, sg, log)
        entries = np.unique(np.round(np.abs(np.concatenate([D.ravel() for _R, _t, D in rep.generators])), 12))
        allowed = {0.0, 0.5, round(np.sqrt(3) / 2, 12), 1.0}
        self.assertTrue(set(entries) <= allowed, entries)


class TestFrame(_WithFixtureRunner):
    def test_si_frame_comes_from_findsym_including_origin_choice_2(self):
        _sg, parent, log, _irrep = _load("Si")
        frame = ib.standard_frame(parent, log)
        np.testing.assert_allclose(frame.Mt, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(frame.origin, [0.625, 0.625, 0.125], atol=1e-6)
        self.assertEqual(len(frame.standard_centrings), 3)      # fcc

    def test_every_generator_is_a_symmetry_of_the_parent(self):
        for material in CASES:
            sg, parent, log, irrep = _load(material)
            rep = ib.representation(irrep, parent, sg, log)
            multipliers = ib.cell_multipliers(rep)
            reference = ib.tiled_parent(parent, multipliers)
            for R, t, _D in rep.generators:
                ib._permutation(reference, multipliers, R, t)    # raises if not


class TestConstructedBasis(_WithFixtureRunner):
    EXPECTED_ARMS = {
        "STO": [1, 1, 2, 2, 3, 3],
        "NaCl": [1, 1, 2, 2, 3, 3, 4, 4],
        "Si": [1, 1, 2, 2, 3, 3, 4, 4],
        "graphite": [1, 1, 1, 1],
    }

    def test_arm_structure_comes_from_isotropys_translation_matrices(self):
        for material, expected in self.EXPECTED_ARMS.items():
            sg, parent, log, irrep = _load(material)
            arms, pairs = ib.arm_structure(ib.representation(irrep, parent, sg, log))
            self.assertEqual(arms, expected, material)
            # quadrature only at the non-self-inverse k (graphite K)
            self.assertEqual(bool(pairs), material == "graphite", material)

    def test_graphite_k5_quadrature_pairs(self):
        sg, parent, log, irrep = _load("graphite")
        _arms, pairs = ib.arm_structure(ib.representation(irrep, parent, sg, log))
        self.assertEqual(pairs, [(0, 2), (1, 3)])

    def test_every_mode_transforms_exactly_by_isotropys_matrices(self):
        """`joint_basis` runs `_check_representation` on every mode and raises
        if the basis is not orthonormal or does not transform by ISOTROPY's
        matrices -- so building it at all is the test."""
        for material in CASES:
            sg, parent, log, irrep = _load(material)
            joint = ib.joint_basis([(irrep, [m.displacements for m in irrep.modes])], parent, sg, log)
            for W in joint.bases[irrep.label]:
                self.assertEqual(W.shape, (irrep.degeneracy, joint.reference.n_atoms, 3))

    def test_same_species_modes_stay_separate(self):
        sg, parent, log, irrep = _load("STO")
        joint = ib.joint_basis([(irrep, [m.displacements for m in irrep.modes])], parent, sg, log)
        flat = [W.reshape(W.shape[0], -1) for W in joint.bases[irrep.label]]
        for a in range(len(flat)):
            for b in range(a + 1, len(flat)):
                self.assertLess(np.abs(flat[a] @ flat[b].T).max(), 1e-10)

    def test_reference_cell_keeps_species_contiguous_for_a_poscar(self):
        sg, parent, log, irrep = _load("NaCl")
        joint = ib.joint_basis([(irrep, [irrep.modes[0].displacements])], parent, sg, log)
        species = joint.reference.species
        blocks = [s for i, s in enumerate(species) if i == 0 or s != species[i - 1]]
        self.assertEqual(len(blocks), len(set(species)))

    def test_old_drop_in_keys_order_like_the_old_construction(self):
        from smodes_phonons.decomposition import _opd_sort_key

        sg, parent, log, irrep = _load("STO")
        _ref, patterns, _n = ib.build_joint_opd_cell([irrep], parent, sg, log)
        m0 = sorted((k for k in patterns if k[2].split("__")[0] == "m0"), key=_opd_sort_key)
        self.assertEqual([(arm, name) for _l, arm, name in m0],
                         [(1, "m0"), (1, "m0__w1"), (2, "m0"), (2, "m0__w1"), (3, "m0"), (3, "m0__w1")])

    def test_a_wrong_frame_is_caught_not_silently_used(self):
        sg, parent, log, irrep = _load("Si")
        rep = ib.representation(irrep, parent, sg, log)
        R, t, D = rep.generators[0]
        rep.generators[0] = (R, t + np.array([0.125, 0.0, 0.0]), D)   # a wrong origin
        multipliers = ib.cell_multipliers(rep)
        reference = ib.tiled_parent(parent, multipliers)
        with self.assertRaises(ib.IsotropyBasisError):
            ib.group_elements(rep, reference, parent, multipliers)




if __name__ == "__main__":
    unittest.main()


class TestDecompositionRoundTripsOnTheIsotropyBasis(unittest.TestCase):
    """v2.0 replacement for the old multi-arm BLEND tests. The old basis was
    built arm by arm, so a distortion over several arms needed separate
    detection (`is_multi_arm_blend`). The ISOTROPY basis already holds every
    arm, within-arm partner and quadrature partner on one cell, so a "blend"
    is just several non-zero positions: every structure below is built from
    the PARENT plus ISOTROPY-basis patterns, and `decomposition` must return
    each injected amplitude exactly, zero everywhere else, zero residual.

    Real fixtures; `iso` output from the suite's saved cache (conftest)."""

    MAIN = Path(__file__).resolve().parent / "fixtures"
    CASES = {
        # name: (poscar, smodes output, findsym log, k, {(irrep, mode, position): amplitude})
        "CMO X2+ two arms (centring-driven)": (
            "POSCAR_CMO", "CMO_findsym_X", "findsym_7.1.6_CMO.log", "X",
            {("X2+", 0, 0): 0.001, ("X2+", 0, 1): 0.001}),
        "CMO cross-irrep X2+ arm 1 + X3- arm 2": (
            "POSCAR_CMO", "CMO_findsym_X", "findsym_7.1.6_CMO.log", "X",
            {("X2+", 0, 0): 0.0013, ("X3-", 0, 1): -0.0007}),
        "TiO2 X1 two arms x within-arm": (
            "POSCAR_TiO2", "TiO2_findsym_X", "findsym_7.1.6_TiO2.log", "X",
            {("X1", 0, 0): 0.001, ("X1", 0, 3): 0.0005}),
        "STO M5- three arms x within-arm": (
            "POSCAR_STO", "STO_findsym_M", "findsym_7.1.6_STO.log", "M",
            {("M5-", 0, 0): 0.001, ("M5-", 0, 3): -0.0004}),
        "graphite K5 native + quadrature": (
            "POSCAR_graphite", "graphite_findsym_K", "findsym_7.1.6_graphite.log", "K",
            {("K5", 0, 0): 0.001, ("K5", 0, 2): 0.0007}),
    }

    def test_every_injected_position_comes_back_exactly(self):
        from smodes_phonons.decomposition import check_stoichiometry, decompose_at_k_vector
        from smodes_phonons.opd import LETTERS

        for name, (poscar, smodes_file, log_file, k, injected) in self.CASES.items():
            with self.subTest(case=name):
                parent = Structure.from_poscar(self.MAIN / poscar)
                log = (self.MAIN / log_file).read_text()
                irreps = list(smodes.parse_smodes_output((self.MAIN / smodes_file).read_text()).irreps)
                wanted = sorted({label for label, _m, _p in injected})
                chosen = [irrep for irrep in irreps if irrep.label in wanted]
                joint = ib.joint_basis([(irrep, ib.seeds_of(irrep)) for irrep in chosen], parent,
                                       ib.space_group_from_log(log), log)
                displacement = sum(a * joint.bases[label][mode][position]
                                   for (label, mode, position), a in injected.items())
                sub = joint.reference.displaced(displacement)

                result = decompose_at_k_vector(k, irreps, parent, sub,
                                               stoichiometry_multiplier=check_stoichiometry(parent, sub),
                                               findsym_log_text=log)
                self.assertLess(result.residual_norm_angstrom, 1e-9)
                got = {(ma.label, ma.mode_index, ma.direction): ma.amplitude_angstrom for ma in result.amplitudes}
                for (label, mode, position), a in injected.items():
                    self.assertAlmostEqual(got[(label, mode, LETTERS[position])], a, places=9, msg=name)
                stray = [v for key, v in got.items()
                         if (key[0], key[1], LETTERS.index(key[2])) not in injected]
                self.assertLess(max(abs(v) for v in stray), 1e-9, name)


TOP_FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestArmKVectors(unittest.TestCase):
    """Each arm's Q-vector, read off ISOTROPY's translation matrices -- the
    `Q^(1), Q^(2), ...` the OPD report prints. Arm N's Q must be the
    wavevector of exactly the components ISOTROPY's letters put on arm N."""

    CASES = {
        # STO M2-: three arms of the M star (cubic, primitive).
        "STO M2-": ("POSCAR_STO", "STO_findsym_M", "findsym_7.1.6_STO.log", "M2-",
                    [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]),
        # CMO X3-: two arms that differ by (0,1,0) -- a reciprocal vector of the
        # conventional cell but NOT of the body-centred crystal; only the
        # centring translation separates them.
        "CMO X3-": ("POSCAR_CMO", "CMO_findsym_X", "findsym_7.1.6_CMO.log", "X3-",
                    [(0.5, 0.5, 0.0), (-0.5, 0.5, 0.0)]),
        # CsV3Sb5 M1+: the published hexagonal triple-Q star.
        "CsV3Sb5 M1+": ("POSCAR_CsV3Sb5", "CsV3Sb5_findsym_M", "findsym_7.1.6_CsV3Sb5.log", "M1+",
                        [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (-0.5, 0.5, 0.0)]),
        # graphite K5: +-K form ONE quadrature arm in ISOTROPY's notation.
        "graphite K5": ("POSCAR_graphite", "graphite_findsym_K", "findsym_7.1.6_graphite.log", "K5",
                        [(1 / 3, 1 / 3, 0.0)]),
    }

    def test_each_arm_gets_its_own_wavevector(self):
        for name, (poscar, smodes_file, log_name, label, expected) in self.CASES.items():
            with self.subTest(name):
                parent = Structure.from_poscar(TOP_FIXTURES / poscar)
                log = (TOP_FIXTURES / log_name).read_text()
                irrep = smodes.parse_smodes_output((TOP_FIXTURES / smodes_file).read_text()).irrep(label)
                rep = ib.representation(irrep, parent, ib.resolve_space_group([irrep], log), log)
                frame = ib.standard_frame(parent, log)
                k = np.linalg.inv(frame.Mt).T @ np.array([float(x) for x in irrep.k_vector])
                rotations = ib.point_group_rotations(parent, rep.space_group, log)
                got = ib.arm_k_vectors(rep, k, rotations)
                np.testing.assert_allclose(np.array(got), np.array(expected), atol=1e-9)


class TestPointGroupRotations(unittest.TestCase):
    def test_group_orders(self):
        for poscar, log_name, order in (("POSCAR_STO", "findsym_7.1.6_STO.log", 48),
                                        ("POSCAR_TiO2", "findsym_7.1.6_TiO2.log", 16),
                                        ("POSCAR_graphite", "findsym_7.1.6_graphite.log", 24),
                                        ("POSCAR_HfO2", "findsym_7.1.6_HfO2.log", 4)):
            with self.subTest(poscar):
                parent = Structure.from_poscar(TOP_FIXTURES / poscar)
                log = (TOP_FIXTURES / log_name).read_text()
                rotations = ib.point_group_rotations(parent, ib.space_group_from_log(log), log)
                self.assertEqual(len(rotations), order)
                # each is a symmetry of the parent's metric
                G = parent.lattice @ parent.lattice.T
                for R in rotations:
                    np.testing.assert_allclose(R.T @ G @ R, G, atol=1e-6)


class TestReduceCell(unittest.TestCase):
    """`freeze` writes the SAME lattice on its shortest near-orthogonal basis,
    following the parent's axes."""

    def test_real_cmo_p5_echelon_basis_is_reduced(self):
        parent = Structure.from_poscar(TOP_FIXTURES / "POSCAR_CMO")
        echelon = np.array([[1.0, -3.0, -2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 2.0]])
        reduced = ib.reduce_cell(echelon, parent.lattice)
        # same lattice: an integer unimodular change of basis
        change = reduced @ np.linalg.inv(echelon)
        np.testing.assert_allclose(change, np.round(change), atol=1e-9)
        self.assertAlmostEqual(abs(np.linalg.det(change)), 1.0, places=9)
        self.assertGreater(np.linalg.det(reduced), 0)          # right-handed
        lengths = np.linalg.norm(reduced @ parent.lattice, axis=1)
        np.testing.assert_allclose(sorted(lengths), [5.2697, 5.2697, 19.4163], atol=1e-3)
        before = np.linalg.norm(echelon @ parent.lattice, axis=1)
        self.assertGreater(before.max(), 39.0)   # 39.9 A before reduction

    def test_an_already_reduced_cell_is_unchanged(self):
        parent = Structure.from_poscar(TOP_FIXTURES / "POSCAR_STO")
        np.testing.assert_allclose(ib.reduce_cell(np.diag([2.0, 2.0, 1.0]), parent.lattice),
                                   np.diag([2.0, 2.0, 1.0]), atol=1e-12)

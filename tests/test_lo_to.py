"""Validates the LO-TO non-analytic correction's math and its integration
into phonons.diagonalize_phonons. No real Born-charge/dielectric-tensor
OUTCAR fixture exists yet (that's still pending from the physicist, for
STO's cubic Pm-3m phase and BaTiO3's tetragonal P4mm phase); these tests
instead check the properties any correct implementation of Gonze & Lee
(1997) MUST have, independent of which numbers a real DFPT run would
actually produce -- split between:

  - TestModePolarization / TestNonAnalyticCorrection: clean, noise-free
    synthetic cases (hand-built SymmetryMode/Born-charge/dielectric-tensor
    arrays, exact answers computable by hand) -- rank-1 structure, the
    acoustic sum rule protecting a TRUE rigid translation exactly (to
    machine precision, via a textbook 2-atom diatomic-chain force-constant
    matrix), rotational invariance of an isotropic response, and
    anisotropic direction-dependence.
  - TestLoToIntegrationWithRealStoGm4Minus: the same correction applied to
    REAL STO GM4- SMODES mode data (tests/fixtures/STO_findsym_Gamma, the
    same fixture test_phonons.py's real GM4- tests use) with synthetic --
    but physically sensible -- Born charges, confirming the correction
    plugs into diagonalize_phonons's own real, finite-displacement-derived
    force-constant matrix and produces a nontrivial, direction-sensitive
    result. GM4-'s real soft-mode instability (-66.5 cm^-1, STO's own
    well-known ferroelectric instability) is itself strongly polar, so
    real data doesn't offer a mode that's both exactly acoustic AND
    trivially trackable across the correction the way the synthetic case
    is -- the exact acoustic-sum-rule claim is proven on that clean
    synthetic case instead, not re-attempted here.

Once the physicist's real BaTiO3/STO Born-charge OUTCAR data lands, a
further fixture-based test (real Z*/epsilon_inf numbers, cross-checked
against literature LO-TO values for one or both materials) should be
added alongside these.
"""

import unittest
from pathlib import Path

import numpy as np

from smodes_phonons import Structure, parse_smodes_output
from smodes_phonons import isotropy_basis
from smodes_phonons.lo_to import (
    LoToError,
    group_equivalent_directions,
    mode_effective_charges,
    mode_polarization,
    non_analytic_correction,
    static_dielectric_tensor,
)
from smodes_phonons.masses import mass_for_species
from smodes_phonons.phonons import PhononError, diagonalize_phonons
from smodes_phonons.smodes import SymmetryMode
from smodes_phonons.vasp import parse_outcar_forces

from _vasp_data import requires_vasp_data


def _point_group(structure, log_name):
    """ISOTROPY's point group in row-vector form -- what the CLI passes."""
    log = (FIXTURES / log_name).read_text()
    rotations = isotropy_basis.point_group_rotations(structure, isotropy_basis.space_group_from_log(log), log)
    return [(R.T, None) for R in rotations]

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# SrTiO3's own formal ionic charges: Sr O2- Ti4+ O3(2-) -> 2 + 4 - 6 = 0,
# satisfying the acoustic sum rule (sum_kappa Z*_kappa = 0) exactly for an
# isotropic (scalar x identity) Born-charge model. Not real DFPT Born
# charges (those are typically "anomalous", e.g. Ti well above +4) -- good
# enough to exercise the correction's own math correctly.
_ISOTROPIC_CHARGES = {"Sr": 2.0, "Ti": 4.0, "O": -2.0}

_VOLUME_ANGSTROM3 = 4.0**3  # a plausible cubic STO-like cell volume for these synthetic checks


def _isotropic_born_charges(atom_species: list, charges: dict = _ISOTROPIC_CHARGES) -> np.ndarray:
    return np.array([charges[sp] * np.eye(3) for sp in atom_species])


def _anisotropic_born_charges(atom_species: list) -> np.ndarray:
    # Same net neutrality (sum over atoms is still exactly zero per
    # Cartesian component), but each atom's own tensor is now genuinely
    # anisotropic (different in-plane vs out-of-plane response) -- the
    # tetragonal-symmetry case.
    per_species_diag = {"Sr": (2.5, 2.5, 1.5), "Ti": (5.0, 5.0, 3.0), "O": (-2.5, -2.5, -1.5)}
    return np.array([np.diag(per_species_diag[sp]) for sp in atom_species])


def _load_gm4_minus():
    out = parse_smodes_output((FIXTURES / "STO_findsym_Gamma").read_text())
    return out.irrep("GM4-")


def _mode_forces():
    return [parse_outcar_forces(FIXTURES / "STO_Irrep_GM4-" / f"Mode_{i}" / "OUTCAR") for i in range(4)]


class TestModePolarization(unittest.TestCase):
    def test_zero_for_a_pattern_orthogonal_to_the_field_response(self):
        # A single atom, born charge only couples x, mode only moves z.
        born = np.array([[[1.0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        mode = np.array([[0.0, 0.0, 1.0]])
        self.assertAlmostEqual(mode_polarization(mode, born, [1, 0, 0]), 0.0)

    def test_nonzero_when_mode_and_field_response_align(self):
        born = np.array([[[1.0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        mode = np.array([[1.0, 0.0, 0.0]])
        self.assertAlmostEqual(mode_polarization(mode, born, [1, 0, 0]), 1.0)

    def test_shape_mismatch_raises(self):
        born = np.zeros((2, 3, 3))
        mode = np.zeros((3, 3))
        with self.assertRaises(LoToError):
            mode_polarization(mode, born, [1, 0, 0])

    def test_zero_direction_raises(self):
        born = np.zeros((1, 3, 3))
        mode = np.zeros((1, 3))
        with self.assertRaises(LoToError):
            mode_polarization(mode, born, [0, 0, 0])


class TestNonAnalyticCorrection(unittest.TestCase):
    """Real GM4- SMODES data: every one of its 4 basis modes has SMODES's
    own native displacement pattern exactly along x (confirmed directly --
    see module-level note below) -- so tests here that need a nonzero
    correction use direction=[1,0,0], not an arbitrary axis."""

    def setUp(self):
        self.irrep = _load_gm4_minus()

    def test_correction_matrix_is_rank_one(self):
        born = _isotropic_born_charges(self.irrep.atom_species)
        eps = np.eye(3) * 6.0
        correction = non_analytic_correction(self.irrep.modes, born, eps, [1, 0, 0], _VOLUME_ANGSTROM3)
        self.assertEqual(correction.shape, (4, 4))
        # Allow for floating-point noise around the true rank.
        singular_values = np.linalg.svd(correction, compute_uv=False)
        self.assertGreater(singular_values[0], 1e-8)
        self.assertTrue(np.all(singular_values[1:] < 1e-8 * singular_values[0]))

    def test_correction_is_symmetric(self):
        born = _anisotropic_born_charges(self.irrep.atom_species)
        eps = np.diag([5.0, 5.0, 8.0])
        correction = non_analytic_correction(self.irrep.modes, born, eps, [0.9, 0.1, 0.3], _VOLUME_ANGSTROM3)
        np.testing.assert_allclose(correction, correction.T)

    def test_non_positive_direction_dot_epsilon_dot_direction_raises(self):
        born = _isotropic_born_charges(self.irrep.atom_species)
        eps = np.diag([1.0, 1.0, -1.0])  # unphysical, but exercises the guard
        with self.assertRaises(LoToError):
            non_analytic_correction(self.irrep.modes, born, eps, [0, 0, 1], _VOLUME_ANGSTROM3)

    def test_non_positive_volume_raises(self):
        born = _isotropic_born_charges(self.irrep.atom_species)
        eps = np.eye(3) * 6.0
        with self.assertRaises(LoToError):
            non_analytic_correction(self.irrep.modes, born, eps, [1, 0, 0], -1.0)

    def test_atom_count_mismatch_raises(self):
        born = np.zeros((self.irrep.n_atoms - 1, 3, 3))
        eps = np.eye(3) * 6.0
        with self.assertRaises(LoToError):
            non_analytic_correction(self.irrep.modes, born, eps, [1, 0, 0], _VOLUME_ANGSTROM3)

    def test_isotropic_response_is_invariant_under_simultaneous_rotation(self):
        # The rigorous way to check "isotropic (cubic) case gives a
        # direction-INDEPENDENT splitting": rotate a mode's own
        # displacement pattern, an isotropic Born charge, and the query
        # direction all TOGETHER by the same arbitrary rotation -- the
        # coupling strength must come back identical, since nothing in an
        # isotropic response singles out a lab-frame axis. (Testing this
        # against real GM4- data directly, by comparing different FIXED
        # lab-frame directions like [1,0,0] vs [0,0,1], would NOT work --
        # SMODES only ever prints one native partner direction per
        # degenerate mode, so an unaligned query direction legitimately
        # gives zero coupling; that's correct physics, not something this
        # test should be confused by.)
        rng = np.random.default_rng(0)
        z = 3.0
        born = np.array([z * np.eye(3)])
        mode_direction = np.array([1.0, 0.0, 0.0])
        query_direction = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)  # partial overlap with mode_direction
        baseline = mode_polarization(np.array([mode_direction]), born, query_direction)
        self.assertNotAlmostEqual(baseline, 0.0)  # a meaningful, nonzero case to rotate

        A = rng.normal(size=(3, 3))
        rotation, _ = np.linalg.qr(A)
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1

        rotated_born = np.array([rotation @ born[0] @ rotation.T])
        rotated_mode = np.array([rotation @ mode_direction])
        rotated_query = rotation @ query_direction
        rotated = mode_polarization(rotated_mode, rotated_born, rotated_query)

        self.assertAlmostEqual(baseline, rotated, places=8)

    def test_anisotropic_dielectric_gives_direction_dependent_splitting(self):
        # Both directions overlap the real mode's own x-polarization (so
        # neither trivially zeroes out), but tilt differently relative to
        # the anisotropic dielectric tensor's own xx vs zz components --
        # exactly the effect that matters for BaTiO3's real P4mm phase.
        born = _anisotropic_born_charges(self.irrep.atom_species)
        eps = np.diag([5.0, 5.0, 8.0])
        correction_a = non_analytic_correction(self.irrep.modes, born, eps, [1, 0, 0], _VOLUME_ANGSTROM3)
        correction_b = non_analytic_correction(self.irrep.modes, born, eps, [1, 0, 0.5], _VOLUME_ANGSTROM3)
        self.assertFalse(np.allclose(correction_a, correction_b))

    def test_acoustic_sum_rule_protects_a_true_rigid_translation_exactly(self):
        # A clean (noise-free) 2-atom diatomic model, hand-built so the
        # exact math is checkable by hand: two raw SMODES-style modes,
        # atom A alone moving (1,0,0) and atom B alone moving (1,0,0).
        # Phi = k*[[1,-1],[-1,1]] is the textbook diatomic-chain force-
        # constant matrix -- [1,1]/sqrt2 (rigid translation, both atoms
        # move together) is its exact eigenvector with eigenvalue 0 (the
        # "acoustic" branch); [1,-1]/sqrt2 (atoms move oppositely) is the
        # "optical" branch with eigenvalue 2k. Charge-neutral Born charges
        # (z_A = -z_B) make s = [z_A, z_B] exactly orthogonal to [1,1] (by
        # construction: z_A + z_B = 0) and NOT orthogonal to [1,-1] (since
        # z_A - z_B != 0) -- so the correction must leave the acoustic
        # eigenvalue EXACTLY unchanged while shifting the optical one.
        mode_a = SymmetryMode(species="A", displacements=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        mode_b = SymmetryMode(species="B", displacements=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        modes = [mode_a, mode_b]

        k = 5.0
        phi = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
        born = np.array([[[1.0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-1.0, 0, 0], [0, 0, 0], [0, 0, 0]]])
        eps = np.eye(3) * 6.0
        volume = 64.0

        correction = non_analytic_correction(modes, born, eps, [1, 0, 0], volume)
        corrected_phi = phi + correction

        acoustic = np.array([1.0, 1.0]) / np.sqrt(2)
        optical = np.array([1.0, -1.0]) / np.sqrt(2)

        # Phi @ acoustic = 0 already (textbook diatomic chain); the
        # correction must add EXACTLY zero on top -- not just small, exact
        # to machine precision, since this is a noise-free hand-built case.
        np.testing.assert_allclose(corrected_phi @ acoustic, np.zeros(2), atol=1e-12)
        # The optical eigenvalue, by contrast, must have genuinely shifted
        # away from its uncorrected value (2k).
        optical_eigenvalue_uncorrected = optical @ phi @ optical
        optical_eigenvalue_corrected = optical @ corrected_phi @ optical
        self.assertNotAlmostEqual(optical_eigenvalue_uncorrected, optical_eigenvalue_corrected)


class TestLoToIntegrationWithRealStoGm4Minus(unittest.TestCase):
    """The physics-correctness checks -- real GM4- SMODES mode data, real
    (uncorrected) diagonalization already validated in test_phonons.py,
    with synthetic Born charges/dielectric tensors chosen to exercise
    specific, independently-known-correct behaviors.

    Every one of GM4-'s 4 real basis modes has SMODES's own native
    displacement pattern exactly along x (confirmed directly, not
    assumed): Sr/Ti/O/O each print displacements = [[1,0,0], 0, 0, ...]
    for whichever single atom moves in that mode. A query direction with
    NO x-component (e.g. [0,0,1]) would therefore give an all-zero
    correction here regardless of whether the physics is otherwise right
    or wrong -- every test below uses a direction with a nonzero x
    component so it's actually exercising the correction, not
    accidentally testing a degenerate no-op case.
    """

    def setUp(self):
        self.irrep = _load_gm4_minus()
        self.forces = _mode_forces()
        self.baseline = diagonalize_phonons(self.irrep, self.forces)

    @requires_vasp_data
    def test_correction_changes_the_real_spectrum_nontrivially(self):
        # REFERENCE_CM in test_phonons.py: [-66.50, 2.84, 156.91, 534.37].
        # -66.50 is STO's real, well-known ferroelectric soft-mode
        # instability -- itself strongly polar, not a bystander -- so
        # unlike the clean synthetic diatomic case above, real STO GM4-
        # doesn't offer an eigenvalue that's both (a) unambiguously
        # trackable across the correction by simple magnitude-sorting and
        # (b) guaranteed uncoupled from the field. The EXACT acoustic-
        # sum-rule protection is proven rigorously on a noise-free
        # synthetic case in TestNonAnalyticCorrection instead; what's
        # checked here is the weaker, still-meaningful real-data claim:
        # applying a genuinely polar correction to a real, finite-
        # displacement-derived force-constant matrix visibly changes the
        # resulting spectrum, not silently a no-op.
        born = _isotropic_born_charges(self.irrep.atom_species)
        eps = np.eye(3) * 6.0
        correction = non_analytic_correction(self.irrep.modes, born, eps, [1, 0, 0], _VOLUME_ANGSTROM3)
        corrected = diagonalize_phonons(self.irrep, self.forces, additional_force_constant_correction=correction)

        self.assertFalse(
            np.allclose(sorted(self.baseline.frequencies_cm), sorted(corrected.frequencies_cm), atol=1e-6)
        )

    @requires_vasp_data
    def test_isotropic_case_gives_the_same_splitting_for_plus_and_minus_x(self):
        # A weaker, real-data version of the rotation-invariance property
        # (the full version is tested abstractly in TestNonAnalyticCorrection,
        # where the mode direction is under direct control) -- for an
        # isotropic response, reversing the query direction must leave the
        # magnitude of the correction (hence the corrected spectrum)
        # unchanged, since the correction only ever depends on
        # (q_hat . Z*)^2-type quantities.
        born = _isotropic_born_charges(self.irrep.atom_species)
        eps = np.eye(3) * 6.0

        results = []
        for direction in ([1, 0, 0], [-1, 0, 0]):
            correction = non_analytic_correction(self.irrep.modes, born, eps, direction, _VOLUME_ANGSTROM3)
            result = diagonalize_phonons(self.irrep, self.forces, additional_force_constant_correction=correction)
            results.append(sorted(result.frequencies_cm))

        np.testing.assert_allclose(results[0], results[1], atol=1e-8)

    @requires_vasp_data
    def test_anisotropic_case_gives_direction_dependent_splitting(self):
        # Tetragonal-style case: genuinely different splitting for a pure
        # x direction vs. one tilted toward z -- exactly the effect that
        # matters for BaTiO3's real P4mm phase. Both directions have a
        # nonzero x-component, so both couple to the real mode's own
        # native x-polarization; only the dielectric tensor's anisotropy
        # (xx=5 vs zz=8) makes the two outcomes differ.
        born = _anisotropic_born_charges(self.irrep.atom_species)
        eps = np.diag([5.0, 5.0, 8.0])

        correction_x = non_analytic_correction(self.irrep.modes, born, eps, [1, 0, 0], _VOLUME_ANGSTROM3)
        correction_tilted = non_analytic_correction(self.irrep.modes, born, eps, [1, 0, 1], _VOLUME_ANGSTROM3)

        result_x = diagonalize_phonons(self.irrep, self.forces, additional_force_constant_correction=correction_x)
        result_tilted = diagonalize_phonons(
            self.irrep, self.forces, additional_force_constant_correction=correction_tilted
        )

        self.assertFalse(
            np.allclose(sorted(result_x.frequencies_cm), sorted(result_tilted.frequencies_cm), atol=1e-3)
        )


class TestDiagonalizePhononsCorrectionParameter(unittest.TestCase):
    def setUp(self):
        self.irrep = _load_gm4_minus()
        self.forces = _mode_forces()

    @requires_vasp_data
    def test_none_reproduces_baseline_exactly(self):
        baseline = diagonalize_phonons(self.irrep, self.forces)
        explicit_none = diagonalize_phonons(self.irrep, self.forces, additional_force_constant_correction=None)
        np.testing.assert_array_equal(baseline.frequencies_cm, explicit_none.frequencies_cm)

    @requires_vasp_data
    def test_wrong_shape_raises_clear_error(self):
        wrong_shape = np.zeros((3, 3))
        with self.assertRaises(PhononError):
            diagonalize_phonons(self.irrep, self.forces, additional_force_constant_correction=wrong_shape)


class TestGroupEquivalentDirections(unittest.TestCase):
    """Real point-group data for two materials already used elsewhere in
    this project's fixtures -- cubic STO (all of x/y/z equivalent) and
    tetragonal TiO2 (x=y, z distinct) -- covering the two cases discussed
    with the physicist (STO's cubic Pm-3m phase, direction-independent;
    BaTiO3's tetragonal P4mm phase, direction-dependent)."""

    def test_cubic_sto_collapses_x_y_z_into_one_group(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        ops = _point_group(structure, "findsym_7.1.6_STO.log")
        self.assertEqual(len(ops), 48)
        directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        groups = group_equivalent_directions(directions, ops, structure.lattice)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0]), [0, 1, 2])

    def test_cubic_sto_keeps_the_body_diagonal_as_its_own_group(self):
        # [1,1,1] is a genuine (3-fold) symmetry axis of a cubic crystal,
        # but NOT related to [1,0,0] by any single point-group operation --
        # face-normal and body-diagonal directions are different orbits
        # under Oh, even though the crystal as a whole is fully cubic.
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        ops = _point_group(structure, "findsym_7.1.6_STO.log")
        self.assertEqual(len(ops), 48)
        directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]]
        groups = group_equivalent_directions(directions, ops, structure.lattice)
        self.assertEqual(len(groups), 2)
        sizes = sorted(len(g) for g in groups)
        self.assertEqual(sizes, [1, 3])

    def test_tetragonal_tio2_separates_z_from_x_equals_y(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_TiO2")
        ops = _point_group(structure, "findsym_7.1.6_TiO2.log")
        self.assertEqual(len(ops), 16)
        directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        groups = group_equivalent_directions(directions, ops, structure.lattice)
        self.assertEqual(len(groups), 2)
        sizes = sorted(len(g) for g in groups)
        self.assertEqual(sizes, [1, 2])
        # x (index 0) and y (index 1) are the equivalent pair; z (index 2) alone.
        xy_group = next(g for g in groups if len(g) == 2)
        z_group = next(g for g in groups if len(g) == 1)
        self.assertEqual(sorted(xy_group), [0, 1])
        self.assertEqual(z_group, [2])

    def test_plus_and_minus_direction_are_equivalent_even_with_no_rotations(self):
        # q_hat and -q_hat give identical LO-TO physics regardless of
        # crystal symmetry -- even with an EMPTY verified-operations list
        # (e.g. triclinic, no symmetry beyond identity), [1,0,0] and
        # [-1,0,0] must still collapse into one group.
        directions = [[1, 0, 0], [-1, 0, 0], [0, 1, 0]]
        groups = group_equivalent_directions(directions, [], np.eye(3))
        self.assertEqual(len(groups), 2)
        sizes = sorted(len(g) for g in groups)
        self.assertEqual(sizes, [1, 2])

    def test_zero_direction_raises(self):
        with self.assertRaises(LoToError):
            group_equivalent_directions([[0, 0, 0]], [], np.eye(3))

    def test_every_index_covered_exactly_once(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        ops = _point_group(structure, "findsym_7.1.6_STO.log")
        self.assertEqual(len(ops), 48)
        directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [1, 1, 0]]
        groups = group_equivalent_directions(directions, ops, structure.lattice)
        covered = sorted(i for group in groups for i in group)
        self.assertEqual(covered, list(range(len(directions))))


class TestModeEffectiveChargesAndStaticDielectricTensor(unittest.TestCase):
    """Validated via the Lyddane-Sachs-Teller relation, cross-checked
    against the already-validated non_analytic_correction/diagonalize_
    phonons pipeline directly -- not just trusted as a from-scratch
    derivation. Real elements (O, Ti) used for their real, distinct masses
    (mass_for_species requires a real symbol), not to represent any real
    O-Ti bonding physics in this synthetic 2-atom toy case."""

    def _diatomic_setup(self):
        mode_a = SymmetryMode(species="O", displacements=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))
        mode_b = SymmetryMode(species="Ti", displacements=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
        modes = [mode_a, mode_b]
        m_a, m_b = mass_for_species("O"), mass_for_species("Ti")

        k = 5.0
        phi = k * np.array([[1.0, -1.0], [-1.0, 1.0]])
        mass_mat = np.sqrt(np.outer([m_a, m_b], [m_a, m_b]))
        dyn_mat = phi / mass_mat
        dyn_eig, dyn_vec = np.linalg.eigh(dyn_mat)

        born = np.array(
            [[[1.0, 0, 0], [0, 0, 0], [0, 0, 0]], [[-1.0, 0, 0], [0, 0, 0], [0, 0, 0]]]
        )  # charge-neutral: z_O + z_Ti = 0
        return modes, phi, mass_mat, dyn_eig, dyn_vec, born

    def test_acoustic_mode_effective_charge_is_exactly_zero(self):
        modes, _, _, _, dyn_vec, born = self._diatomic_setup()
        zbar = mode_effective_charges(modes, dyn_vec, born)
        acoustic_idx = 0  # eigh's ascending order puts the 0-eigenvalue (acoustic) mode first
        np.testing.assert_allclose(zbar[acoustic_idx], 0.0, atol=1e-12)
        self.assertFalse(np.allclose(zbar[1], 0.0))  # optical mode is genuinely polar

    def test_matches_lyddane_sachs_teller_relation_exactly(self):
        # epsilon(0)/epsilon_inf must equal (omega_LO/omega_TO)^2 for the
        # SAME direction/material -- computed here two independent ways:
        # once via static_dielectric_tensor's own mode sum, once via
        # diagonalizing the ALREADY-validated non_analytic_correction's
        # own LO-shifted force-constant matrix directly. If these two
        # genuinely independent computations agree exactly, the formula's
        # overall prefactor/units are correct, not just plausible-looking.
        modes, phi, mass_mat, dyn_eig, dyn_vec, born = self._diatomic_setup()
        eps_inf = np.eye(3) * 4.0
        volume = 64.0
        direction = [1, 0, 0]

        zbar = mode_effective_charges(modes, dyn_vec, born)
        eps0, unstable_excluded = static_dielectric_tensor(zbar, dyn_eig, eps_inf, volume)
        self.assertFalse(unstable_excluded)

        correction = non_analytic_correction(modes, born, eps_inf, direction, volume)
        corrected_eig, _ = np.linalg.eigh((phi + correction) / mass_mat)

        to2 = max(dyn_eig)
        lo2 = max(corrected_eig)
        lst_predicted_eps0_xx = eps_inf[0, 0] * (lo2 / to2)

        self.assertAlmostEqual(lst_predicted_eps0_xx, eps0[0, 0], places=10)
        # Untouched off-axis components -- born charges only ever had an
        # x-component in this setup.
        np.testing.assert_allclose(eps0[1:, 1:], eps_inf[1:, 1:])

    def test_unstable_contributing_mode_is_excluded_and_flagged(self):
        modes, _, _, dyn_eig, dyn_vec, born = self._diatomic_setup()
        # Force the (otherwise stable) optical mode eigenvalue negative,
        # simulating a real unstable, polar mode (e.g. STO's own GM4-
        # ferroelectric soft mode) -- its own contribution must be
        # excluded from the sum, not included as a nonsensical negative
        # or blown-up term.
        unstable_eig = dyn_eig.copy()
        unstable_eig[1] = -0.05

        eps_inf = np.eye(3) * 4.0
        zbar = mode_effective_charges(modes, dyn_vec, born)
        eps0, unstable_excluded = static_dielectric_tensor(zbar, unstable_eig, eps_inf, 64.0)

        self.assertTrue(unstable_excluded)
        # With the only polar mode excluded, epsilon(0) reduces to bare epsilon_inf.
        np.testing.assert_allclose(eps0, eps_inf)

    def test_non_positive_volume_raises(self):
        modes, _, _, dyn_eig, dyn_vec, born = self._diatomic_setup()
        zbar = mode_effective_charges(modes, dyn_vec, born)
        with self.assertRaises(LoToError):
            static_dielectric_tensor(zbar, dyn_eig, np.eye(3) * 4.0, -1.0)

    @requires_vasp_data
    def test_real_sto_gm4_minus_does_not_produce_nan_or_inf(self):
        # Real, noisy finite-displacement data: confirms the formula stays
        # numerically well-behaved (no NaN/inf) even with a near-but-not-
        # exactly-zero acoustic-like mode (2.84 cm^-1, a small residual
        # polar character from real DFT numerical noise -- see
        # TestLoToIntegrationWithRealStoGm4Minus's own module-level note)
        # combined with a genuinely unstable, strongly polar mode
        # (-66.5 cm^-1). Not a claim about a specific "correct" epsilon(0)
        # value for this noisy fixture -- see this module's own docstring.
        irrep = _load_gm4_minus()
        forces = _mode_forces()
        result = diagonalize_phonons(irrep, forces)
        born = _isotropic_born_charges(irrep.atom_species)
        eps_inf = np.eye(3) * 6.0

        zbar = mode_effective_charges(irrep.modes, result.mode_basis_eigenvectors, born)
        eps0, unstable_excluded = static_dielectric_tensor(
            zbar, result.dynamical_matrix_eigenvalues, eps_inf, _VOLUME_ANGSTROM3
        )
        self.assertTrue(unstable_excluded)  # STO GM4-'s own real soft-mode instability
        self.assertTrue(np.all(np.isfinite(eps0)))


if __name__ == "__main__":
    unittest.main()

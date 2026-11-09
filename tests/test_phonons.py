"""Core validation for phonons.py, against three complete real VASP calculations.

The tolerances here are not arbitrary: legacy's reference frequencies were
computed with c = 3E10 (a known bug, fixed here to 2.99792458e10), so every
computed frequency is expected to differ from legacy's by ~0.07% -- not zero,
and not more than roughly 0.2%. Eigenvalues_fcon comparisons go through
sqrt(abs(.)) because legacy's reported values are the (mislabeled)
square root of the true force-constant eigenvalue -- see phonons.py's
module docstring.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from smodes_phonons import parse_smodes_output
from smodes_phonons.phonons import PhononError, diagonalize_phonons, load_phonon_result, save_phonon_result
from smodes_phonons.vasp import parse_outcar_forces

from _vasp_data import requires_vasp_data

FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Relative tolerance covers the deliberate c-constant fix (~0.07%) plus a
# small margin for legacy's rounded reference printouts (2 decimal places).
_FREQ_RTOL = 0.003


def _mode_forces(irrep_dir: str, n_modes: int) -> list:
    return [parse_outcar_forces(FIXTURES / irrep_dir / f"Mode_{i}" / "OUTCAR") for i in range(n_modes)]


def _undistorted_forces(irrep_dir: str):
    return parse_outcar_forces(FIXTURES / irrep_dir / "Undistorted_lattice" / "OUTCAR")


class TestStoGM4Minus(unittest.TestCase):
    """No undistorted-lattice case; includes one unstable (imaginary) mode."""

    REFERENCE_CM = [-66.50, 2.84, 156.91, 534.37]

    def setUp(self):
        out = parse_smodes_output((FIXTURES / "STO_findsym_Gamma").read_text())
        self.irrep = out.irrep("GM4-")
        self.result = diagonalize_phonons(self.irrep, _mode_forces("STO_Irrep_GM4-", 4))

    @requires_vasp_data
    def test_frequencies_match_reference_within_tolerance(self):
        computed = sorted(self.result.frequencies_cm)
        for c, r in zip(computed, sorted(self.REFERENCE_CM)):
            self.assertAlmostEqual(c / r, 1.0, delta=_FREQ_RTOL)

    @requires_vasp_data
    def test_frequencies_are_ascending(self):
        self.assertEqual(list(self.result.frequencies_cm), sorted(self.result.frequencies_cm))

    @requires_vasp_data
    def test_unstable_mode_is_negative_real_not_complex(self):
        self.assertTrue(np.isrealobj(self.result.frequencies_cm))
        self.assertLess(min(self.result.frequencies_cm), 0)

    @requires_vasp_data
    def test_thz_and_mev_are_simple_ratios_of_cm(self):
        np.testing.assert_allclose(self.result.frequencies_thz, self.result.frequencies_cm / 33.356)
        np.testing.assert_allclose(self.result.frequencies_mev, self.result.frequencies_cm / 8.066)

    @requires_vasp_data
    def test_force_constant_eigenvalues_match_legacys_mislabeled_reference(self):
        # legacy's Eigenvalues_fcon: 4.11, 2.42, 0.69 (unstable), 0.03 -- these
        # are sqrt(|true eigenvalue|), not the eigenvalue itself.
        legacy_reported = sorted([4.11, 2.42, 0.69, 0.03])
        computed_sqrt = sorted(np.sqrt(np.abs(self.result.force_constant_eigenvalues)))
        for c, r in zip(computed_sqrt, legacy_reported):
            self.assertAlmostEqual(c, r, delta=0.02)

    @requires_vasp_data
    def test_highest_frequency_atomic_displacement_eigenvector_matches_reference(self):
        # Reference: Eigenvectors_Atom_disp "Phonon Displacement 0" (the 534
        # cm^-1 mode in legacy's own ordering).
        reference = np.array(
            [
                [0.0119250, 0.0, 0.0],
                [-0.0004444, 0.0, 0.0],
                [-0.7383471, 0.0, 0.0],
                [0.4768129, 0.0, 0.0],
                [0.4768129, 0.0, 0.0],
            ]
        )
        idx = int(np.argmax(self.result.frequencies_cm))
        computed = self.result.atomic_displacement_eigenvectors[idx]
        matches = np.allclose(computed, reference, atol=1e-5) or np.allclose(computed, -reference, atol=1e-5)
        self.assertTrue(matches, f"computed:\n{computed}\nreference:\n{reference}")

    @requires_vasp_data
    def test_wrong_number_of_mode_forces_raises(self):
        with self.assertRaises(PhononError):
            diagonalize_phonons(self.irrep, _mode_forces("STO_Irrep_GM4-", 4)[:3])


class TestTiO2GM5Plus(unittest.TestCase):
    """Undistorted-lattice subtraction case; all 3 modes stable."""

    REFERENCE_CM = [620.84, 136.58, 166.67]

    def setUp(self):
        out = parse_smodes_output((FIXTURES / "TiO2_findsym_GM").read_text())
        self.irrep = out.irrep("GM5+")
        self.result = diagonalize_phonons(
            self.irrep,
            _mode_forces("TiO2_Irrep_GM5+", 3),
            undistorted_forces=_undistorted_forces("TiO2_Irrep_GM5+"),
        )

    @requires_vasp_data
    def test_frequencies_match_reference_within_tolerance(self):
        computed = sorted(self.result.frequencies_cm)
        for c, r in zip(computed, sorted(self.REFERENCE_CM)):
            self.assertAlmostEqual(c / r, 1.0, delta=_FREQ_RTOL)

    @requires_vasp_data
    def test_all_modes_stable(self):
        self.assertTrue(all(f > 0 for f in self.result.frequencies_cm))

    @requires_vasp_data
    def test_without_undistorted_subtraction_gives_different_answer(self):
        # Sanity check that the subtraction actually does something for this
        # material (it has free Wyckoff parameters) -- omitting it should not
        # coincidentally give the same frequencies.
        without = diagonalize_phonons(self.irrep, _mode_forces("TiO2_Irrep_GM5+", 3))
        self.assertFalse(np.allclose(sorted(without.frequencies_cm), sorted(self.result.frequencies_cm), rtol=1e-3))


class TestCmoX3Minus(unittest.TestCase):
    """Undistorted-lattice subtraction; 6 modes, 2 unstable -- the fullest stress test."""

    REFERENCE_CM = [428.66, 340.07, -143.86, -55.30, 165.92, 192.78]

    def setUp(self):
        out = parse_smodes_output((FIXTURES / "CMO_findsym_X").read_text())
        self.irrep = out.irrep("X3-")
        self.result = diagonalize_phonons(
            self.irrep,
            _mode_forces("CMO_Irrep_X3-", 6),
            undistorted_forces=_undistorted_forces("CMO_Irrep_X3-"),
        )

    @requires_vasp_data
    def test_frequencies_match_reference_within_tolerance(self):
        computed = sorted(self.result.frequencies_cm)
        for c, r in zip(computed, sorted(self.REFERENCE_CM)):
            self.assertAlmostEqual(c / r, 1.0, delta=_FREQ_RTOL)

    @requires_vasp_data
    def test_exactly_two_unstable_modes(self):
        self.assertEqual(sum(1 for f in self.result.frequencies_cm if f < 0), 2)

    @requires_vasp_data
    def test_six_modes_all_outputs_consistent_length(self):
        self.assertEqual(len(self.result.frequencies_cm), 6)
        self.assertEqual(len(self.result.force_constant_eigenvalues), 6)
        self.assertEqual(len(self.result.force_constant_eigenvectors), 6)
        self.assertEqual(len(self.result.dynamical_matrix_eigenvectors), 6)
        self.assertEqual(len(self.result.atomic_displacement_eigenvectors), 6)
        for vec in self.result.atomic_displacement_eigenvectors:
            self.assertEqual(vec.shape, (self.irrep.n_atoms, 3))
            self.assertAlmostEqual(float(np.linalg.norm(vec)), 1.0, places=10)


class TestSaveLoadPhononResult(unittest.TestCase):
    def setUp(self):
        out = parse_smodes_output((FIXTURES / "STO_findsym_Gamma").read_text())
        self.irrep = out.irrep("GM4-")
        self.result = diagonalize_phonons(self.irrep, _mode_forces("STO_Irrep_GM4-", 4))

    @requires_vasp_data
    def test_round_trip_matches_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phonons.npz"
            save_phonon_result(self.result, path)
            reloaded = load_phonon_result(path)

        np.testing.assert_array_equal(reloaded.frequencies_cm, self.result.frequencies_cm)
        np.testing.assert_array_equal(reloaded.frequencies_thz, self.result.frequencies_thz)
        np.testing.assert_array_equal(reloaded.frequencies_mev, self.result.frequencies_mev)
        np.testing.assert_array_equal(
            reloaded.force_constant_eigenvalues, self.result.force_constant_eigenvalues
        )
        for a, b in zip(reloaded.force_constant_eigenvectors, self.result.force_constant_eigenvectors):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(reloaded.dynamical_matrix_eigenvectors, self.result.dynamical_matrix_eigenvectors):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(
            reloaded.atomic_displacement_eigenvectors, self.result.atomic_displacement_eigenvectors
        ):
            np.testing.assert_array_equal(a, b)


if __name__ == "__main__":
    unittest.main()

import stat
import unittest
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from smodes_phonons import (
    SmodesOutputError,
    SymmetryInfo,
    WyckoffPosition,
    build_smodes_input,
    k_vector_label_from_irrep,
    parse_smodes_output,
    run_smodes,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / name).read_text()


class TestStoGamma(unittest.TestCase):
    def setUp(self):
        self.out = parse_smodes_output(load("STO_findsym_Gamma"))

    def test_header(self):
        self.assertEqual(self.out.title, "STO_findsym")
        self.assertEqual(self.out.space_group_number, 221)
        self.assertEqual(self.out.space_group_symbol, "Pm-3m")
        np.testing.assert_allclose(
            self.out.lattice_parameters, (3.8986, 3.8986, 3.8986, 90.0, 90.0, 90.0)
        )
        self.assertEqual(
            self.out.wyckoff_positions,
            [
                WyckoffPosition("Sr", "a"),
                WyckoffPosition("Ti", "b"),
                WyckoffPosition("O", "c"),
            ],
        )

    def test_irrep_labels_and_counts(self):
        self.assertEqual([ir.label for ir in self.out.irreps], ["GM4-", "GM5-"])

    def test_gm4_minus(self):
        ir = self.out.irrep("GM4-")
        self.assertEqual(ir.mulliken, "T1u")
        self.assertEqual(ir.k_label, "GM")
        self.assertEqual(ir.k_vector, (Fraction(0), Fraction(0), Fraction(0)))
        self.assertEqual(ir.degeneracy, 3)
        self.assertEqual(ir.total_modes, 12)
        self.assertEqual(ir.n_modes, 4)
        self.assertEqual(ir.n_translational_modes, 3)
        self.assertTrue(ir.ir_active)
        self.assertFalse(ir.raman_active)
        self.assertEqual(ir.n_atoms, 5)
        self.assertEqual(ir.atom_species, ["Sr", "Ti", "O", "O", "O"])
        self.assertEqual(len(ir.modes), 4)

        # mode 0: only atom 1 (Sr, index 0) moves, in +x
        m0 = ir.modes[0]
        self.assertEqual(m0.species, "Sr")
        expected0 = np.zeros((5, 3))
        expected0[0] = [1.0, 0.0, 0.0]
        np.testing.assert_allclose(m0.displacements, expected0)

        # mode 3: atoms 4 and 5 (both O, indices 3, 4) move together in +x
        m3 = ir.modes[3]
        self.assertEqual(m3.species, "O")
        expected3 = np.zeros((5, 3))
        expected3[3] = [1.0, 0.0, 0.0]
        expected3[4] = [1.0, 0.0, 0.0]
        np.testing.assert_allclose(m3.displacements, expected3)

    def test_gm5_minus(self):
        ir = self.out.irrep("GM5-")
        self.assertEqual(ir.mulliken, "T2u")
        self.assertEqual(ir.degeneracy, 3)
        self.assertEqual(ir.total_modes, 3)
        self.assertEqual(ir.n_modes, 1)
        self.assertEqual(ir.n_translational_modes, 0)
        self.assertFalse(ir.ir_active)
        self.assertFalse(ir.raman_active)

        m0 = ir.modes[0]
        self.assertEqual(m0.species, "O")
        expected = np.zeros((5, 3))
        expected[2] = [0.0, 0.0, -1.0]  # atom 3, index 2
        expected[4] = [0.0, 0.0, 1.0]  # atom 5, index 4
        np.testing.assert_allclose(m0.displacements, expected)

    def test_normalized_mode_is_unit_frobenius_norm(self):
        ir = self.out.irrep("GM5-")
        normalized = ir.modes[0].normalized
        self.assertAlmostEqual(np.linalg.norm(normalized), 1.0)


class TestStoM(unittest.TestCase):
    def setUp(self):
        self.out = parse_smodes_output(load("STO_findsym_M"))

    def test_irrep_labels(self):
        self.assertEqual(
            [ir.label for ir in self.out.irreps],
            ["M1+", "M2+", "M3+", "M4+", "M5+", "M2-", "M3-", "M5-"],
        )

    def test_k_vector_is_half_half_zero(self):
        ir = self.out.irrep("M1+")
        self.assertEqual(ir.k_label, "M")
        self.assertEqual(ir.k_vector, (Fraction(1, 2), Fraction(1, 2), Fraction(0)))

    def test_m2_minus_has_two_independent_modes_different_species(self):
        ir = self.out.irrep("M2-")
        self.assertEqual(ir.degeneracy, 3)
        self.assertEqual(ir.total_modes, 6)
        self.assertEqual(ir.n_modes, 2)
        self.assertEqual(ir.modes[0].species, "Ti")
        self.assertEqual(ir.modes[1].species, "O")

    def test_m5_minus_three_modes(self):
        ir = self.out.irrep("M5-")
        self.assertEqual(ir.degeneracy, 6)
        self.assertEqual(ir.total_modes, 18)
        self.assertEqual(ir.n_modes, 3)
        self.assertEqual([m.species for m in ir.modes], ["Sr", "Ti", "O"])

    def test_supercell_has_ten_atoms(self):
        for ir in self.out.irreps:
            self.assertEqual(ir.n_atoms, 10)
            self.assertEqual(ir.superlattice.shape, (3, 3))
            self.assertEqual(ir.atom_positions.shape, (10, 3))


class TestStoR(unittest.TestCase):
    def setUp(self):
        self.out = parse_smodes_output(load("STO_findsym_R"))

    def test_irrep_labels(self):
        self.assertEqual(
            [ir.label for ir in self.out.irreps], ["R5+", "R2-", "R3-", "R4-", "R5-"]
        )

    def test_k_vector_is_half_half_half(self):
        ir = self.out.irrep("R5+")
        self.assertEqual(ir.k_vector, (Fraction(1, 2), Fraction(1, 2), Fraction(1, 2)))

    def test_r4_minus_two_modes(self):
        ir = self.out.irrep("R4-")
        self.assertEqual(ir.degeneracy, 3)
        self.assertEqual(ir.total_modes, 6)
        self.assertEqual(ir.n_modes, 2)
        self.assertEqual(ir.modes[0].species, "Sr")
        self.assertEqual(ir.modes[1].species, "O")

    def test_r2_minus_nondegenerate_single_mode_many_atoms(self):
        ir = self.out.irrep("R2-")
        self.assertEqual(ir.degeneracy, 1)
        self.assertEqual(ir.n_modes, 1)
        mode = ir.modes[0]
        self.assertEqual(mode.species, "O")
        moving = np.count_nonzero(np.linalg.norm(mode.displacements, axis=1) > 1e-8)
        self.assertEqual(moving, 6)


class TestTiO2Gamma(unittest.TestCase):
    def setUp(self):
        self.out = parse_smodes_output(load("TiO2_findsym_GM"))

    def test_header(self):
        self.assertEqual(self.out.title, "TiO2_findsym")
        self.assertEqual(self.out.space_group_number, 141)
        self.assertEqual(self.out.space_group_symbol, "I4_1/amd")
        self.assertEqual(
            self.out.wyckoff_positions,
            [
                WyckoffPosition("Ti", "b"),
                WyckoffPosition("O", "e", z=-0.41753),
            ],
        )

    def test_irrep_labels(self):
        self.assertEqual(
            [ir.label for ir in self.out.irreps],
            ["GM1+", "GM2+", "GM5+", "GM3-", "GM4-", "GM5-"],
        )

    def test_raman_and_ir_flags(self):
        self.assertTrue(self.out.irrep("GM1+").raman_active)
        self.assertFalse(self.out.irrep("GM1+").ir_active)
        self.assertTrue(self.out.irrep("GM3-").ir_active)
        self.assertEqual(self.out.irrep("GM3-").n_translational_modes, 1)
        self.assertEqual(self.out.irrep("GM5-").n_translational_modes, 2)
        self.assertFalse(self.out.irrep("GM4-").raman_active)
        self.assertFalse(self.out.irrep("GM4-").ir_active)

    def test_gm5_plus_three_modes_two_atoms_each(self):
        ir = self.out.irrep("GM5+")
        self.assertEqual(ir.degeneracy, 2)
        self.assertEqual(ir.n_modes, 3)
        self.assertEqual([m.species for m in ir.modes], ["Ti", "O", "O"])
        for mode in ir.modes:
            moving = np.count_nonzero(np.linalg.norm(mode.displacements, axis=1) > 1e-8)
            self.assertEqual(moving, 2)


class TestSpeciesInvariantAcrossAllFixtures(unittest.TestCase):
    """the developer notes: every SMODES mode must involve exactly one species. Verify this
    holds -- without exception -- across every mode in every irrep we have real
    fixture data for, rather than merely spot-checking a few by hand."""

    def test_every_mode_in_every_fixture_has_single_species(self):
        for name in (
            "STO_findsym_Gamma",
            "STO_findsym_M",
            "STO_findsym_R",
            "TiO2_findsym_GM",
        ):
            out = parse_smodes_output(load(name))
            for ir in out.irreps:
                self.assertEqual(len(ir.modes), ir.n_modes)
                for mode in ir.modes:
                    self.assertIsInstance(mode.species, str)
                    self.assertIn(mode.species, ir.atom_species)


class TestMalformedInput(unittest.TestCase):
    def test_not_smodes_output(self):
        with self.assertRaises(SmodesOutputError):
            parse_smodes_output("this is not smodes output\n" * 20)

    def test_truncated_mid_irrep(self):
        text = load("STO_findsym_Gamma")
        truncated = "\n".join(text.splitlines()[:30])  # cuts off mid Symmetry-modes block
        with self.assertRaises(SmodesOutputError):
            parse_smodes_output(truncated)

    def test_missing_k_vector_before_irrep(self):
        lines = load("STO_findsym_Gamma").splitlines()
        # drop the "k vector GM" header + its following '****' line
        del lines[13:15]
        with self.assertRaises(SmodesOutputError):
            parse_smodes_output("\n".join(lines))


class TestMultipleKVectorsInOneFile(unittest.TestCase):
    """Not exercised by any of the real fixtures (the physicist's workflow always
    requests one k-vector per run), but generator_smodes_input.sh supports
    requesting several -- exercise that code path with a small synthetic example."""

    def _synthetic(self) -> str:
        return """smodes, version 1.2.4, January 1999
Harold T. Stokes
Brigham Young University
Synthetic
Space Group:
221 Pm-3m
Lattice Parameters (a,b,c,alpha,beta,gamma)
   3.89860   3.89860   3.89860  90.00000  90.00000  90.00000
Wyckoff position of each type of atom
Sr a
***********************************************
k vector GM = (0,0,0)
***********************************************
Irrep GM1+
Degeneracy:  1
Total number of modes:   1
Vectors defining superlattice
   3.89860   0.00000   0.00000
   0.00000   3.89860   0.00000
   0.00000   0.00000   3.89860
atom  type   position
  1   Sr     0.00000   0.00000   0.00000
Symmetry modes:
atom  type   displacement
------------------------------------------
  1   Sr     1.00000   0.00000   0.00000
***********************************************
k vector X = (1/2,0,0)
***********************************************
Irrep X1
Degeneracy:  1
Total number of modes:   1
Vectors defining superlattice
   3.89860   0.00000   0.00000
   0.00000   3.89860   0.00000
   0.00000   0.00000   3.89860
atom  type   position
  1   Sr     0.00000   0.00000   0.00000
Symmetry modes:
atom  type   displacement
------------------------------------------
  1   Sr     0.00000   1.00000   0.00000
***********************************************
End of output
"""

    def test_two_k_vector_sections(self):
        out = parse_smodes_output(self._synthetic())
        self.assertEqual([ir.label for ir in out.irreps], ["GM1+", "X1"])
        self.assertEqual(out.irrep("GM1+").k_label, "GM")
        self.assertEqual(out.irrep("X1").k_label, "X")
        self.assertEqual(out.irrep("X1").k_vector, (Fraction(1, 2), Fraction(0), Fraction(0)))


class TestBuildSmodesInput(unittest.TestCase):
    def test_no_free_params(self):
        symmetry = SymmetryInfo(
            title="STO_findsym",
            space_group_number=221,
            lattice_parameters=(3.8986, 3.8986, 3.8986, 90.0, 90.0, 90.0),
            wyckoff_positions=[
                WyckoffPosition("Sr", "a"),
                WyckoffPosition("Ti", "b"),
                WyckoffPosition("O", "c"),
            ],
        )
        text = build_smodes_input(symmetry, "GM")
        self.assertEqual(
            text,
            "STO_findsym\n"
            "221\n"
            "3.89860 3.89860 3.89860 90.00000 90.00000 90.00000\n"
            "3\n"
            "Sr a\n"
            "Ti b\n"
            "O c\n"
            "1\n"
            "GM\n",
        )

    def test_free_params(self):
        symmetry = SymmetryInfo(
            title="TiO2_findsym",
            space_group_number=141,
            lattice_parameters=(3.77632, 3.77632, 9.57643, 90.0, 90.0, 90.0),
            wyckoff_positions=[
                WyckoffPosition("Ti", "b"),
                WyckoffPosition("O", "e", z=-0.41753),
            ],
        )
        text = build_smodes_input(symmetry, "X")
        lines = text.splitlines()
        self.assertEqual(lines[0], "TiO2_findsym")
        self.assertEqual(lines[4], "Ti b")
        self.assertEqual(lines[5], "O e 0 0 -0.41753")
        self.assertEqual(lines[6], "1")
        self.assertEqual(lines[7], "X")

    def test_rejects_a_list_of_k_vectors(self):
        # Only one k-vector per smodes_input, ever -- the legacy README's
        # "sometimes errors occur, redo step 2" warning is exactly about
        # batching k-vectors. A list doesn't have a sane single-string
        # rendering, so passing one is a caller bug we want to surface
        # immediately rather than silently stringify.
        symmetry = SymmetryInfo(
            title="STO_findsym",
            space_group_number=221,
            lattice_parameters=(3.8986, 3.8986, 3.8986, 90.0, 90.0, 90.0),
            wyckoff_positions=[WyckoffPosition("Sr", "a")],
        )
        with self.assertRaises(TypeError):
            build_smodes_input(symmetry, ["GM", "X"])

    def test_matches_real_sto_smodes_input_byte_for_byte(self):
        # tests/fixtures/STO_findsym_smodes_input is the real input the physicist fed
        # to smodes for STO_findsym_Gamma. Rebuild it from the header smodes
        # echoed back in its own output, and require an exact match.
        out = parse_smodes_output(load("STO_findsym_Gamma"))
        symmetry = SymmetryInfo(
            title=out.title,
            space_group_number=out.space_group_number,
            lattice_parameters=out.lattice_parameters,
            wyckoff_positions=out.wyckoff_positions,
        )
        text = build_smodes_input(symmetry, "GM")
        self.assertEqual(text, load("STO_findsym_smodes_input"))

    def test_matches_real_tio2_smodes_input_byte_for_byte(self):
        # Same check for TiO2 -- this is the one that exercises the fixed-vs-
        # free Wyckoff axis distinction ("O e 0 0 -0.41753").
        out = parse_smodes_output(load("TiO2_findsym_GM"))
        symmetry = SymmetryInfo(
            title=out.title,
            space_group_number=out.space_group_number,
            lattice_parameters=out.lattice_parameters,
            wyckoff_positions=out.wyckoff_positions,
        )
        text = build_smodes_input(symmetry, "GM")
        self.assertEqual(text, load("TiO2_findsym_smodes_input"))

    def test_matches_real_cmo_smodes_input_byte_for_byte(self):
        # Same check again for Ca3Mn2O7 (CMO) -- a genuinely different space
        # group (139, tetragonal) with repeated species across multiple
        # Wyckoff orbits (Ca at both 'b' and 'e', O at 'a', 'e', and 'g').
        # tests/fixtures/CMO_findsym_smodes_input is the real, matched input that
        # produced tests/fixtures/CMO_findsym_X (k-vector X).
        out = parse_smodes_output(load("CMO_findsym_X"))
        symmetry = SymmetryInfo(
            title=out.title,
            space_group_number=out.space_group_number,
            lattice_parameters=out.lattice_parameters,
            wyckoff_positions=out.wyckoff_positions,
        )
        text = build_smodes_input(symmetry, "X")
        self.assertEqual(text, load("CMO_findsym_smodes_input"))


class TestRunSmodes(unittest.TestCase):
    def test_missing_binary_raises(self):
        from smodes_phonons import smodes as smodes_module

        original = smodes_module.isotropy.find_binary
        smodes_module.isotropy.find_binary = lambda name: None
        try:
            with self.assertRaises(FileNotFoundError):
                run_smodes("irrelevant input")
        finally:
            smodes_module.isotropy.find_binary = original

    def test_subprocess_wiring_with_stub_binary(self):
        with TemporaryDirectory() as tmp:
            stub = Path(tmp) / "fake_smodes"
            stub.write_text("#!/bin/sh\ncat\n")
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
            output = run_smodes("hello\n", binary=str(stub))
        self.assertEqual(output, "hello\n")

    def test_nonzero_exit_raises(self):
        with TemporaryDirectory() as tmp:
            stub = Path(tmp) / "fake_smodes_fail"
            stub.write_text("#!/bin/sh\necho 'boom' >&2\nexit 1\n")
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
            with self.assertRaises(RuntimeError):
                run_smodes("hello\n", binary=str(stub))


class TestKVectorLabelFromIrrep(unittest.TestCase):
    def test_every_real_irrep_label_across_all_fixtures(self):
        expected_k_label = {
            "STO_findsym_Gamma": "GM",
            "STO_findsym_M": "M",
            "STO_findsym_R": "R",
            "TiO2_findsym_GM": "GM",
            "CMO_findsym_X": "X",
        }
        for name, k_label in expected_k_label.items():
            out = parse_smodes_output(load(name))
            for ir in out.irreps:
                with self.subTest(fixture=name, label=ir.label):
                    self.assertEqual(k_vector_label_from_irrep(ir.label), k_label)

    def test_malformed_label_raises(self):
        with self.assertRaises(ValueError):
            k_vector_label_from_irrep("bogus")

    def test_empty_label_raises(self):
        with self.assertRaises(ValueError):
            k_vector_label_from_irrep("")


class TestIrrepNotFoundErrorMentionsWyckoffSetting(unittest.TestCase):
    def test_error_lists_available_labels_and_notes_wyckoff_setting(self):
        out = parse_smodes_output(load("STO_findsym_Gamma"))
        with self.assertRaises(KeyError) as ctx:
            out.irrep("R4+")
        message = str(ctx.exception)
        self.assertIn("GM4-", message)
        self.assertIn("GM5-", message)
        self.assertIn("Wyckoff setting", message)


if __name__ == "__main__":
    unittest.main()

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from smodes_phonons import (
    DISPLACEMENT_AMPLITUDE_ANGSTROM,
    Structure,
    parse_smodes_output,
    supercell_structure,
    write_mode_poscars,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / name).read_text()


class TestSupercellStructure(unittest.TestCase):
    def setUp(self):
        out = parse_smodes_output(load("STO_findsym_Gamma"))
        self.irrep = out.irrep("GM4-")

    def test_lattice_species_and_atom_count_match_irrep(self):
        s = supercell_structure(self.irrep, comment="test")
        np.testing.assert_allclose(s.lattice, self.irrep.superlattice)
        self.assertEqual(s.species, self.irrep.atom_species)
        self.assertEqual(s.n_atoms, self.irrep.n_atoms)

    def test_cart_coords_match_irrep_atom_positions(self):
        s = supercell_structure(self.irrep, comment="test")
        np.testing.assert_allclose(s.cart_coords, self.irrep.atom_positions, atol=1e-10)


class TestWriteModePoscars(unittest.TestCase):
    def test_gm4_minus_writes_four_modes_at_correct_amplitude(self):
        out = parse_smodes_output(load("STO_findsym_Gamma"))
        irrep = out.irrep("GM4-")
        undistorted = supercell_structure(irrep, comment="STO GM4-")

        with TemporaryDirectory() as tmp:
            irrep_dir = Path(tmp) / "Irrep_GM4-"
            paths = write_mode_poscars(irrep, irrep_dir, comment="STO GM4-")

            self.assertEqual(len(paths), irrep.n_modes)
            self.assertEqual(len(paths), 4)
            for i, path in enumerate(paths):
                self.assertEqual(path, irrep_dir / f"Mode_{i}" / "POSCAR")
                self.assertTrue(path.is_file())

                reloaded = Structure.from_poscar(path)
                self.assertEqual(reloaded.species, undistorted.species)
                np.testing.assert_allclose(reloaded.lattice, undistorted.lattice)

                delta = reloaded.cart_coords - undistorted.cart_coords
                self.assertAlmostEqual(
                    float(np.linalg.norm(delta)), DISPLACEMENT_AMPLITUDE_ANGSTROM, places=10
                )

    def test_gm5_minus_single_mode(self):
        out = parse_smodes_output(load("STO_findsym_Gamma"))
        irrep = out.irrep("GM5-")

        with TemporaryDirectory() as tmp:
            irrep_dir = Path(tmp) / "Irrep_GM5-"
            paths = write_mode_poscars(irrep, irrep_dir, comment="STO GM5-")

        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0], irrep_dir / "Mode_0" / "POSCAR")

    def test_rerun_overwrites_silently(self):
        out = parse_smodes_output(load("STO_findsym_Gamma"))
        irrep = out.irrep("GM5-")

        with TemporaryDirectory() as tmp:
            irrep_dir = Path(tmp) / "Irrep_GM5-"
            first = write_mode_poscars(irrep, irrep_dir, comment="first")
            second = write_mode_poscars(irrep, irrep_dir, comment="second")
            self.assertEqual(first, second)
            text = second[0].read_text()
            self.assertIn("second", text.splitlines()[0])

    def test_undistorted_poscar_always_written(self):
        out = parse_smodes_output(load("STO_findsym_Gamma"))
        irrep = out.irrep("GM4-")
        undistorted = supercell_structure(irrep, comment="STO GM4-")

        with TemporaryDirectory() as tmp:
            irrep_dir = Path(tmp) / "Irrep_GM4-"
            write_mode_poscars(irrep, irrep_dir, comment="STO GM4-")

            undistorted_path = irrep_dir / "Undistorted" / "POSCAR"
            self.assertTrue(undistorted_path.is_file())

            reloaded = Structure.from_poscar(undistorted_path)
            self.assertEqual(reloaded.species, undistorted.species)
            np.testing.assert_allclose(reloaded.lattice, undistorted.lattice)
            # No displacement applied -- must match the undistorted supercell exactly.
            np.testing.assert_allclose(reloaded.cart_coords, undistorted.cart_coords, atol=1e-10)


if __name__ == "__main__":
    unittest.main()

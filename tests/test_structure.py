import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from smodes_phonons import PoscarFormatError, Structure

SRTIO3_DIRECT = """\
SrTiO3 cubic
1.0
3.9 0.0 0.0
0.0 3.9 0.0
0.0 0.0 3.9
Sr Ti O
1 1 3
Direct
0.0 0.0 0.0
0.5 0.5 0.5
0.5 0.5 0.0
0.5 0.0 0.5
0.0 0.5 0.5
"""


def write(dir_path: Path, name: str, text: str) -> Path:
    path = dir_path / name
    path.write_text(text)
    return path


class TestFromPoscarDirect(unittest.TestCase):
    def test_basic_parse(self):
        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "POSCAR", SRTIO3_DIRECT)
            s = Structure.from_poscar(path)

        self.assertEqual(s.comment, "SrTiO3 cubic")
        self.assertEqual(s.n_atoms, 5)
        self.assertEqual(s.species, ["Sr", "Ti", "O", "O", "O"])
        np.testing.assert_allclose(s.lattice, np.eye(3) * 3.9)
        np.testing.assert_allclose(s.frac_coords[1], [0.5, 0.5, 0.5])


class TestScaleFactor(unittest.TestCase):
    def test_positive_scale_multiplies_lattice(self):
        text = SRTIO3_DIRECT.replace("1.0\n3.9 0.0 0.0", "2.0\n3.9 0.0 0.0", 1)
        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "POSCAR", text)
            s = Structure.from_poscar(path)
        np.testing.assert_allclose(s.lattice, np.eye(3) * 7.8)

    def test_negative_scale_targets_volume(self):
        # raw lattice is 3.9x3.9x3.9 => raw volume 59.319 A^3. Ask for volume 8.0 A^3
        # instead => each raw vector scales by (8.0/59.319)**(1/3).
        text = SRTIO3_DIRECT.replace("1.0\n3.9 0.0 0.0", "-8.0\n3.9 0.0 0.0", 1)
        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "POSCAR", text)
            s = Structure.from_poscar(path)
        self.assertAlmostEqual(np.linalg.det(s.lattice), 8.0, places=10)

    def test_zero_scale_rejected(self):
        text = SRTIO3_DIRECT.replace("1.0\n3.9 0.0 0.0", "0.0\n3.9 0.0 0.0", 1)
        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "POSCAR", text)
            with self.assertRaises(PoscarFormatError):
                Structure.from_poscar(path)

    def test_multi_value_scale_line_rejected(self):
        text = SRTIO3_DIRECT.replace("1.0\n3.9 0.0 0.0", "1.0 1.0 1.0\n3.9 0.0 0.0", 1)
        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "POSCAR", text)
            with self.assertRaises(PoscarFormatError):
                Structure.from_poscar(path)


class TestCartesianMode(unittest.TestCase):
    def test_cartesian_matches_direct(self):
        cartesian = """\
SrTiO3 cubic
1.0
3.9 0.0 0.0
0.0 3.9 0.0
0.0 0.0 3.9
Sr Ti O
1 1 3
Cartesian
0.0 0.0 0.0
1.95 1.95 1.95
1.95 1.95 0.0
1.95 0.0 1.95
0.0 1.95 1.95
"""
        with TemporaryDirectory() as tmp:
            direct = Structure.from_poscar(write(Path(tmp), "POSCAR_d", SRTIO3_DIRECT))
            cart = Structure.from_poscar(write(Path(tmp), "POSCAR_c", cartesian))
        np.testing.assert_allclose(direct.frac_coords, cart.frac_coords, atol=1e-12)

    def test_cartesian_coords_scaled_by_scale_factor(self):
        # scale=2.0 must apply to Cartesian coordinates too, not just the lattice.
        # Raw (pre-scale) values here are real values / 2.0.
        cartesian_scaled = """\
SrTiO3 cubic
2.0
1.95 0.0 0.0
0.0 1.95 0.0
0.0 0.0 1.95
Sr Ti O
1 1 3
Cartesian
0.0 0.0 0.0
0.975 0.975 0.975
0.975 0.975 0.0
0.975 0.0 0.975
0.0 0.975 0.975
"""
        with TemporaryDirectory() as tmp:
            s = Structure.from_poscar(write(Path(tmp), "POSCAR", cartesian_scaled))
        np.testing.assert_allclose(s.lattice, np.eye(3) * 3.9)
        np.testing.assert_allclose(s.frac_coords[1], [0.5, 0.5, 0.5])


class TestSelectiveDynamics(unittest.TestCase):
    def test_selective_dynamics_line_skipped(self):
        text = SRTIO3_DIRECT.replace("Direct\n", "Selective dynamics\nDirect\n", 1)
        text = text.replace("0.0 0.0 0.0\n", "0.0 0.0 0.0 T T T\n", 1)
        with TemporaryDirectory() as tmp:
            s = Structure.from_poscar(write(Path(tmp), "POSCAR", text))
        self.assertEqual(s.n_atoms, 5)
        np.testing.assert_allclose(s.frac_coords[0], [0.0, 0.0, 0.0])


class TestSpeciesValidation(unittest.TestCase):
    def test_missing_species_line_rejected(self):
        # VASP4-style POSCAR: no symbols line, counts appear where symbols should be.
        text = SRTIO3_DIRECT.replace("Sr Ti O\n1 1 3\n", "1 1 3\n5\n", 1)
        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "POSCAR", text)
            with self.assertRaises(PoscarFormatError):
                Structure.from_poscar(path)

    def test_mismatched_species_and_counts_rejected(self):
        text = SRTIO3_DIRECT.replace("Sr Ti O\n1 1 3\n", "Sr Ti O\n1 1 3 1\n", 1)
        with TemporaryDirectory() as tmp:
            path = write(Path(tmp), "POSCAR", text)
            with self.assertRaises(PoscarFormatError):
                Structure.from_poscar(path)

    def test_discontiguous_species_rejected_on_write(self):
        s = Structure(
            comment="bad order",
            lattice=np.eye(3) * 3.9,
            species=["Sr", "Ti", "Sr"],
            frac_coords=np.zeros((3, 3)),
        )
        with self.assertRaises(ValueError):
            s.species_counts


class TestRoundTrip(unittest.TestCase):
    def test_write_then_read_back(self):
        with TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            original = Structure.from_poscar(write(tmp, "POSCAR", SRTIO3_DIRECT))
            out_path = tmp / "POSCAR_out"
            original.to_poscar(out_path)
            reread = Structure.from_poscar(out_path)

        np.testing.assert_allclose(reread.lattice, original.lattice)
        np.testing.assert_allclose(reread.frac_coords, original.frac_coords)
        self.assertEqual(reread.species, original.species)


class TestCoordinateConversion(unittest.TestCase):
    def test_frac_cart_round_trip(self):
        with TemporaryDirectory() as tmp:
            s = Structure.from_poscar(write(Path(tmp), "POSCAR", SRTIO3_DIRECT))
        cart = s.cart_coords
        np.testing.assert_allclose(s.cart_to_frac(cart), s.frac_coords, atol=1e-12)

    def test_displaced_shifts_atoms_and_does_not_mutate_original(self):
        with TemporaryDirectory() as tmp:
            s = Structure.from_poscar(write(Path(tmp), "POSCAR", SRTIO3_DIRECT))
        disp = np.zeros((s.n_atoms, 3))
        disp[0] = [0.001, 0.0, 0.0]
        moved = s.displaced(disp)

        np.testing.assert_allclose(moved.cart_coords[0], s.cart_coords[0] + [0.001, 0.0, 0.0])
        np.testing.assert_allclose(s.cart_coords[0], [0.0, 0.0, 0.0])  # original untouched


if __name__ == "__main__":
    unittest.main()

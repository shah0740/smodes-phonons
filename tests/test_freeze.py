"""freeze.py's atom-alignment and reduction-mapping helpers, on real data:
nearest-neighbour alignment of independently tiled STO GM4- / M2- cells, and
mapping findsym-reduced structures back onto the cell they came from (the
mechanism `--magmom` relies on).
"""

import unittest
from pathlib import Path

import numpy as np

from smodes_phonons import Structure, parse_smodes_output
from smodes_phonons.freeze import (
    FreezeError,
    align_by_nearest_neighbor,
    search_alignment_shift,
)
from smodes_phonons.supercell import nice_diagonal_multipliers, tile_onto_supercell

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / name).read_text()




class TestAlignByNearestNeighbor(unittest.TestCase):
    def setUp(self):
        self.structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        gm4 = parse_smodes_output(load("STO_findsym_Gamma")).irrep("GM4-")
        m2 = parse_smodes_output(load("STO_findsym_M")).irrep("M2-")
        # Both irreps tiled independently onto M2-'s parent-axis cell (2,2,1).
        anchor = self.structure.lattice
        multipliers = nice_diagonal_multipliers(m2.superlattice, anchor)
        self.reference, _ = tile_onto_supercell(gm4, anchor, multipliers, {}, comment="ref")
        self.other, _ = tile_onto_supercell(m2, anchor, multipliers, {}, comment="other")

    def test_real_independently_tiled_irreps_all_align(self):
        perm = align_by_nearest_neighbor(
            self.reference.species, self.reference.frac_coords, self.other.species, self.other.frac_coords,
            self.reference.lattice,
        )
        self.assertEqual(len(perm), self.reference.n_atoms)
        self.assertEqual(len(set(perm)), len(perm))  # a genuine permutation, no atom used twice
        for i, j in enumerate(perm):
            self.assertEqual(self.reference.species[i], self.other.species[j])

    def test_atom_count_mismatch_raises(self):
        with self.assertRaises(FreezeError):
            align_by_nearest_neighbor(
                self.reference.species, self.reference.frac_coords,
                self.other.species[:-1], self.other.frac_coords[:-1],
                self.reference.lattice,
            )

    def test_far_displaced_atom_raises(self):
        corrupted_frac = self.other.frac_coords.copy()
        corrupted_frac[0] = corrupted_frac[0] + 0.4  # far beyond the default 0.3 A tolerance
        with self.assertRaises(FreezeError):
            align_by_nearest_neighbor(
                self.reference.species, self.reference.frac_coords,
                self.other.species, corrupted_frac,
                self.reference.lattice,
            )




class TestSearchAlignmentShift(unittest.TestCase):
    """Moved here from `decomposition._search_origin_shift` this session
    (see its own docstring) once the identical symptom -- a genuine
    structure, off by one fixed, data-derivable translation -- turned up
    for a second reason (combining star arms of a non-symmorphic irrep,
    see TestCombineStarArmsNonSymmorphic below), not just decomposition's
    own external-origin-convention case. `decomposition.py`'s own tests
    (`test_recovers_exact_amplitudes_despite_an_arbitrary_origin_shift`,
    `TestRealBaTiO3RhombohedralDecomposition`) already cover that original
    use through `internal_displacement`; this is the direct unit-level
    check of the moved function itself."""

    def test_finds_an_arbitrary_data_derived_shift(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        shift = np.array([0.13, 0.27, 0.61])
        shifted_frac = (structure.frac_coords - shift) % 1.0
        sub = Structure(
            comment="shifted", lattice=structure.lattice, species=list(structure.species), frac_coords=shifted_frac
        )
        found_shift, perm = search_alignment_shift(structure, sub, tol=0.3)
        self.assertIsNotNone(found_shift)
        np.testing.assert_allclose(found_shift % 1.0, (-shift) % 1.0, atol=1e-6)
        self.assertEqual(len(set(perm)), structure.n_atoms)

    def test_no_shift_needed_when_already_aligned(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        found_shift, perm = search_alignment_shift(structure, structure, tol=0.3)
        np.testing.assert_allclose(found_shift, [0.0, 0.0, 0.0], atol=1e-6)
        self.assertEqual(perm, list(range(structure.n_atoms)))

    def test_returns_none_when_genuinely_unalignable(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        garbage_frac = np.random.RandomState(0).rand(structure.n_atoms, 3)
        sub = Structure(
            comment="garbage", lattice=structure.lattice, species=list(structure.species), frac_coords=garbage_frac
        )
        found_shift, perm = search_alignment_shift(structure, sub, tol=0.05)
        self.assertIsNone(found_shift)
        self.assertIsNone(perm)









def _min_image_match(reference, built):
    """Greedy species-constrained nearest-neighbour match under the MINIMUM
    IMAGE convention. Returns `(max_distance, permutation)`.

    THE MINIMUM IMAGE IS LOAD-BEARING, not a refinement. These are PERIODIC
    structures, so a raw Cartesian difference is not a distance between
    atoms: two IDENTICAL crystals whose atoms are wrapped differently into
    the same cell read as one full lattice vector apart. Measured on exactly
    the case this module guards -- real CMO GM5- on a 24-atom cell -- the raw
    metric reported 19.0554 A, which is precisely CMO's own `c`. The greedy
    one-to-one assignment then compounds it, because an early bad match
    forces worse ones later, turning 3.5 A into 15.5 A.

    Under minimum image the same comparison gives 1.2e-05 A and a pattern
    cosine of 1.000000 -- the structures are the same crystal. `freeze.py`'s
    own `align_by_nearest_neighbor` has always used minimum image; this test
    was the outlier.

    `test_a_genuinely_wrong_cell_is_still_rejected` is the negative control:
    a metric that accepts everything would pass these tests too, so the
    correction is only trustworthy alongside a case it still rejects.
    """
    lattice = np.asarray(built.lattice, dtype=np.float64)
    inverse = np.linalg.inv(lattice)
    unmatched = list(range(built.n_atoms))
    perm = [None] * reference.n_atoms
    max_dist = 0.0
    for i, (pos, species) in enumerate(zip(reference.cart_coords, reference.species)):
        best = None
        for j in unmatched:
            if built.species[j] != species:
                continue
            delta = (pos - built.cart_coords[j]) @ inverse
            delta -= np.round(delta)
            distance = float(np.linalg.norm(delta @ lattice))
            if best is None or distance < best[0]:
                best = (distance, j)
        assert best is not None, f"no unmatched {species} atom remains"
        max_dist = max(max_dist, best[0])
        perm[i] = best[1]
        unmatched.remove(best[1])
    return max_dist, np.asarray(perm)


class TestMinImageMatchRejectsAWrongCell(unittest.TestCase):
    """Negative control for `_min_image_match`.

    Correcting the metric to use minimum image makes a previously-failing
    comparison pass, so the obvious worry is that it now accepts anything.
    It does not: a structure that is genuinely a different arrangement --
    not merely the same one wrapped differently -- is still rejected."""

    def test_a_genuinely_wrong_cell_is_still_rejected(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        gm5 = parse_smodes_output(load("CMO_findsym_GM")).irrep("GM5-")
        pattern = gm5.modes[0].normalized
        b = np.asarray(structure.lattice) @ np.linalg.inv(np.asarray(gm5.superlattice))
        truth, _disp = tile_onto_supercell(
            gm5, gm5.superlattice, np.round(b).astype(int), {"p": [pattern]}, comment="truth"
        )

        # Same cell and same species, but one atom of each species displaced
        # by a large NON-lattice vector -- a real rearrangement, which no
        # amount of periodic wrapping can undo.
        moved = np.array(truth.frac_coords, dtype=np.float64)
        for species in dict.fromkeys(truth.species):
            index = truth.species.index(species)
            moved[index] = (moved[index] + np.array([0.37, 0.21, 0.29])) % 1.0
        wrong = Structure(
            comment="deliberately wrong", lattice=truth.lattice,
            species=list(truth.species), frac_coords=moved,
        )

        max_dist, _perm = _min_image_match(truth, wrong)
        self.assertGreater(
            max_dist, 1e-2,
            "minimum image must not make a genuinely different arrangement look identical",
        )

    def test_the_same_structure_wrapped_differently_is_accepted(self):
        """The other half of the control: a pure lattice-vector shift of
        every atom is the SAME crystal and must compare as identical, which
        is exactly what the raw-Cartesian metric got wrong."""
        structure = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        gm5 = parse_smodes_output(load("CMO_findsym_GM")).irrep("GM5-")
        pattern = gm5.modes[0].normalized
        b = np.asarray(structure.lattice) @ np.linalg.inv(np.asarray(gm5.superlattice))
        truth, _disp = tile_onto_supercell(
            gm5, gm5.superlattice, np.round(b).astype(int), {"p": [pattern]}, comment="truth"
        )

        shifted = Structure(
            comment="same crystal, wrapped differently", lattice=truth.lattice,
            species=list(truth.species),
            frac_coords=(np.array(truth.frac_coords, dtype=np.float64) + np.array([0.0, 0.0, 1.0])),
        )
        max_dist, _perm = _min_image_match(truth, shifted)
        self.assertLess(max_dist, 1e-9)












if __name__ == "__main__":
    unittest.main()

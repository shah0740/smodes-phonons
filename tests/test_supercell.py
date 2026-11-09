import unittest
from pathlib import Path

import numpy as np

from smodes_phonons import Structure, parse_smodes_output
from smodes_phonons.supercell import (
    SupercellError,
    _aligning_rotation,
    _find_integer_representation,
    map_parent_values_to_native,
    nice_diagonal_multipliers,
    nice_supercell_matrix,
    tile_onto_supercell,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / name).read_text()


class TestNiceDiagonalMultipliers(unittest.TestCase):
    def setUp(self):
        self.structure_sto = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        self.structure_cmo = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        self.structure_tio2 = Structure.from_poscar(FIXTURES / "POSCAR_TiO2")

    def test_gamma_point_is_trivial(self):
        out = parse_smodes_output(load("STO_findsym_Gamma"))
        n = nice_diagonal_multipliers(out.irrep("GM4-").superlattice, self.structure_sto.lattice)
        self.assertEqual(n, (1, 1, 1))

    def test_tio2_anatase_gamma_point_is_trivial(self):
        # tests/fixtures/POSCAR_TiO2 was originally rutile -- a genuinely different
        # polymorph from TiO2_findsym_GM's anatase, so this raised
        # SupercellError until the physicist replaced it with the correct
        # matched anatase structure (see the developer notes).
        out = parse_smodes_output(load("TiO2_findsym_GM"))
        n = nice_diagonal_multipliers(out.irrep("GM5+").superlattice, self.structure_tio2.lattice)
        self.assertEqual(n, (1, 1, 1))

    def test_m_point_doubles_along_ab_only(self):
        out = parse_smodes_output(load("STO_findsym_M"))
        n = nice_diagonal_multipliers(out.irrep("M2-").superlattice, self.structure_sto.lattice)
        self.assertEqual(n, (2, 2, 1))

    def test_r_point_doubles_along_all_three(self):
        out = parse_smodes_output(load("STO_findsym_R"))
        n = nice_diagonal_multipliers(out.irrep("R5+").superlattice, self.structure_sto.lattice)
        self.assertEqual(n, (2, 2, 2))

    def test_centered_space_group_cmo_x_point(self):
        # Space group 139 (I4/mmm, body-centered) -- the relationship between
        # POSCAR_CMO's conventional cell and the X3- superlattice is
        # half-integer, the signature of I-centering. Must still resolve to
        # a clean answer: double along a,b, not c.
        out = parse_smodes_output(load("CMO_findsym_X"))
        n = nice_diagonal_multipliers(out.irrep("X3-").superlattice, self.structure_cmo.lattice)
        self.assertEqual(n, (2, 2, 1))

    def test_result_is_a_genuine_integer_supercell(self):
        out = parse_smodes_output(load("STO_findsym_R"))
        superlattice = out.irrep("R5+").superlattice
        n = nice_diagonal_multipliers(superlattice, self.structure_sto.lattice)
        target = np.diag(n) @ self.structure_sto.lattice
        m = target @ np.linalg.inv(superlattice)
        np.testing.assert_allclose(m, np.round(m), atol=1e-2)

    def test_incommensurate_lattice_raises(self):
        # Fraction.limit_denominator(24) is quite forgiving -- most
        # "unrelated" matrices still admit a decent small-denominator
        # approximation within the 1% relative tolerance. A deterministic
        # way to force a genuine failure: scale the superlattice by an
        # irrational factor (sqrt(2)) and pass a tight max_denominator, so
        # no rational approximation is close enough.
        out = parse_smodes_output(load("STO_findsym_M"))
        superlattice = out.irrep("M2-").superlattice
        bogus_reference = (2.0**0.5) * superlattice
        with self.assertRaises(SupercellError):
            nice_diagonal_multipliers(superlattice, bogus_reference, max_denominator=1)


class TestTileOntoSupercell(unittest.TestCase):
    def setUp(self):
        self.structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        out = parse_smodes_output(load("STO_findsym_M"))
        self.irrep = out.irrep("M2-")  # 2 modes: Ti-only, then O-only
        self.multipliers = nice_diagonal_multipliers(self.irrep.superlattice, self.structure.lattice)

    def test_atom_count_scales_by_determinant(self):
        # Not n1*n2*n3 directly -- that's the multiple of the *primitive*
        # cell, but M2-'s own superlattice is already 2x the primitive cell
        # (the M-point folds once), so the atom-count multiplier is
        # det(target_lattice @ inv(superlattice)), not n1*n2*n3.
        displacement_sets = {"atomic_displacement": [m.displacements for m in self.irrep.modes]}
        big_structure, _ = tile_onto_supercell(
            self.irrep, self.structure.lattice, self.multipliers, displacement_sets, comment="test"
        )
        target = np.diag(self.multipliers) @ self.structure.lattice
        det_m = round(abs(np.linalg.det(target @ np.linalg.inv(self.irrep.superlattice))))
        self.assertEqual(big_structure.n_atoms, self.irrep.n_atoms * det_m)

    def test_species_grouped_contiguously(self):
        # Structure.species_counts raises if species aren't contiguous --
        # calling it (via to_poscar during setUp-less construction) is itself
        # the test that tiling didn't interleave species across native-cell copies.
        displacement_sets = {"atomic_displacement": [m.displacements for m in self.irrep.modes]}
        big_structure, _ = tile_onto_supercell(
            self.irrep, self.structure.lattice, self.multipliers, displacement_sets, comment="test"
        )
        counts = big_structure.species_counts  # raises ValueError if not contiguous
        total = sum(n for _, n in counts)
        self.assertEqual(total, big_structure.n_atoms)

    def test_no_duplicate_positions(self):
        displacement_sets = {"atomic_displacement": [m.displacements for m in self.irrep.modes]}
        big_structure, _ = tile_onto_supercell(
            self.irrep, self.structure.lattice, self.multipliers, displacement_sets, comment="test"
        )
        keys = {tuple(np.round(row, 5)) for row in big_structure.frac_coords}
        self.assertEqual(len(keys), big_structure.n_atoms)

    def test_single_species_per_mode_survives_tiling(self):
        displacement_sets = {"atomic_displacement": [m.displacements for m in self.irrep.modes]}
        big_structure, big_sets = tile_onto_supercell(
            self.irrep, self.structure.lattice, self.multipliers, displacement_sets, comment="test"
        )
        for mode_idx, expected_species in enumerate(["Ti", "O"]):
            arr = big_sets["atomic_displacement"][mode_idx]
            moving = {
                sp
                for sp, row in zip(big_structure.species, arr)
                if np.linalg.norm(row) > 1e-8
            }
            self.assertEqual(moving, {expected_species})

    def test_mapped_displacements_are_unit_normalized(self):
        displacement_sets = {"atomic_displacement": [m.displacements for m in self.irrep.modes]}
        _, big_sets = tile_onto_supercell(
            self.irrep, self.structure.lattice, self.multipliers, displacement_sets, comment="test"
        )
        for arr in big_sets["atomic_displacement"]:
            self.assertAlmostEqual(float(np.linalg.norm(arr)), 1.0, places=10)

    def test_multiple_named_displacement_sets(self):
        displacement_sets = {
            "atomic_displacement": [m.displacements for m in self.irrep.modes],
            "force_constant": [m.displacements * 2 for m in self.irrep.modes],
        }
        _, big_sets = tile_onto_supercell(
            self.irrep, self.structure.lattice, self.multipliers, displacement_sets, comment="test"
        )
        self.assertEqual(set(big_sets.keys()), {"atomic_displacement", "force_constant"})
        self.assertEqual(len(big_sets["force_constant"]), 2)

    def test_result_round_trips_through_poscar(self):
        import tempfile

        displacement_sets = {"atomic_displacement": [m.displacements for m in self.irrep.modes]}
        big_structure, _ = tile_onto_supercell(
            self.irrep, self.structure.lattice, self.multipliers, displacement_sets, comment="test"
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "POSCAR_superlattice"
            big_structure.to_poscar(path)
            reloaded = Structure.from_poscar(path)
        self.assertEqual(reloaded.n_atoms, big_structure.n_atoms)
        self.assertEqual(reloaded.species, big_structure.species)


class TestCmoTiling(unittest.TestCase):
    """The centered-space-group stress test, end to end."""

    def test_cmo_x3_minus_tiles_cleanly(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        out = parse_smodes_output(load("CMO_findsym_X"))
        irrep = out.irrep("X3-")
        multipliers = nice_diagonal_multipliers(irrep.superlattice, structure.lattice)
        displacement_sets = {"atomic_displacement": [m.displacements for m in irrep.modes]}
        big_structure, big_sets = tile_onto_supercell(
            irrep, structure.lattice, multipliers, displacement_sets, comment="cmo test"
        )
        self.assertEqual(multipliers, (2, 2, 1))
        self.assertEqual(big_structure.n_atoms, irrep.n_atoms * 4)
        keys = {tuple(np.round(row, 5)) for row in big_structure.frac_coords}
        self.assertEqual(len(keys), big_structure.n_atoms)
        for arr in big_sets["atomic_displacement"]:
            self.assertAlmostEqual(float(np.linalg.norm(arr)), 1.0, places=10)


class TestTio2Tiling(unittest.TestCase):
    """Anatase TiO2's GM5+ mode -- multiplier (1,1,1), but NOT a trivial
    exact match: the fixed POSCAR_TiO2 is the 12-atom *conventional*
    body-centered cell (I4_1/amd -- the "I" is body-centering, same
    signature as CMO's I4/mmm), while GM5+'s own native superlattice is the
    6-atom primitive cell -- exactly double the volume via a non-diagonal
    integer tiling matrix (det 2), the same kind of centered-space-group
    relationship CMO's X3- exercises, just discovered here via the same
    diagonal-multiplier machinery rather than a distinct code path. Real
    fixture data, undistorted-lattice-subtraction case (free O Wyckoff
    parameter)."""

    def test_tio2_gm5_plus_tiles_cleanly(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_TiO2")
        out = parse_smodes_output(load("TiO2_findsym_GM"))
        irrep = out.irrep("GM5+")
        multipliers = nice_diagonal_multipliers(irrep.superlattice, structure.lattice)
        displacement_sets = {"atomic_displacement": [m.displacements for m in irrep.modes]}
        big_structure, big_sets = tile_onto_supercell(
            irrep, structure.lattice, multipliers, displacement_sets, comment="tio2 test"
        )
        self.assertEqual(multipliers, (1, 1, 1))
        target = np.diag(multipliers) @ structure.lattice
        det_m = round(abs(np.linalg.det(target @ np.linalg.inv(irrep.superlattice))))
        self.assertEqual(det_m, 2)
        self.assertEqual(big_structure.n_atoms, irrep.n_atoms * det_m)
        keys = {tuple(np.round(row, 5)) for row in big_structure.frac_coords}
        self.assertEqual(len(keys), big_structure.n_atoms)
        for arr in big_sets["atomic_displacement"]:
            self.assertAlmostEqual(float(np.linalg.norm(arr)), 1.0, places=10)


class TestAligningRotation(unittest.TestCase):
    def test_recovers_a_known_rotation(self):
        rng = np.random.default_rng(0)
        source = rng.normal(size=(3, 3))
        # A genuine proper rotation (det +1), built via QR + a determinant fix.
        q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(q) < 0:
            q[:, -1] *= -1
        target = source @ q
        recovered = _aligning_rotation(source, target)
        np.testing.assert_allclose(recovered, q, atol=1e-8)
        np.testing.assert_allclose(source @ recovered, target, atol=1e-8)

    def test_result_is_orthogonal(self):
        rng = np.random.default_rng(1)
        source = rng.normal(size=(3, 3))
        target = rng.normal(size=(3, 3))
        r = _aligning_rotation(source, target)
        np.testing.assert_allclose(r @ r.T, np.eye(3), atol=1e-8)


class TestFindIntegerRepresentation(unittest.TestCase):
    def test_finds_a_simple_diagonal_doubling(self):
        g_ref = np.diag([4.0, 5.0, 6.0])
        g_target = np.diag([16.0, 20.0, 24.0])  # doubled lattice vectors -> 4x lengths^2
        m = _find_integer_representation(g_ref, g_target)
        self.assertIsNotNone(m)
        np.testing.assert_allclose(m @ g_ref @ m.T, g_target, atol=1e-6)
        self.assertGreater(np.linalg.det(m), 0)

    def test_returns_none_when_no_relationship_exists(self):
        g_ref = np.diag([4.0, 5.0, 6.0])
        g_target = np.diag([7.3, 9.1, 11.7])  # no small integer combination reaches these
        m = _find_integer_representation(g_ref, g_target, max_component=4)
        self.assertIsNone(m)

    def test_candidates_finds_a_real_exact_gram_tie(self):
        # Real BaTiO3 R3m data (cubic parent vs. its own real 15-atom
        # hexagonal-setting supercell): two genuinely different integer
        # matrices, [[-1,-1,0],[0,1,-1],[1,-1,-1]] and
        # [[1,-1,0],[0,1,-1],[1,1,1]], both satisfy the Gram match with
        # IDENTICAL full-matrix error -- confirmed this session while
        # fixing `decomposition._search_target_matrix`'s own inability to
        # resolve real BaTiO3 R3m data (see the developer notes). This is a genuine,
        # Gram-tensor-blind ambiguity (the two candidates are related by a
        # rotation that is not itself a symmetry of the cubic lattice, so
        # no amount of Gram-tensor scoring can distinguish them) -- only
        # one of them actually aligns real atom positions correctly, which
        # is why `_search_target_matrix` needs every tied candidate, not
        # just one.
        from smodes_phonons.supercell import _find_integer_representation_candidates

        a = 3.99037921
        g_ref = np.eye(3) * a * a  # cubic
        m1 = np.array([[-1, -1, 0], [0, 1, -1], [1, -1, -1]], dtype=float)
        g_target = m1 @ g_ref @ m1.T

        candidates = _find_integer_representation_candidates(g_ref, g_target, max_component=8, tol=0.05)
        self.assertGreaterEqual(len(candidates), 2)
        for m in candidates[:2]:
            np.testing.assert_allclose(m @ g_ref @ m.T, g_target, atol=1e-6)
            self.assertGreater(np.linalg.det(m), 0)
        # The two candidates are genuinely different matrices (not the
        # same one returned twice).
        self.assertFalse(np.array_equal(candidates[0], candidates[1]))


class TestNiceSupercellMatrix(unittest.TestCase):
    """HfO2 (space group 14, P2_1/c, monoclinic) is this project's only
    fixture needing genuinely non-diagonal supercell matrices: SMODES's own
    superlattice for each non-Gamma k-point disagrees with POSCAR_HfO2's
    Cartesian embedding by a real rotation (and, for some k-points, by
    admixture too) -- see supercell.py's module docstring. Real findsym
    cross-check (done interactively while developing this fix, not
    re-checked here) confirms undistorted/distorted structures built on
    these supercells report space groups 14/P2_1/c and 2/P-1 respectively
    for all four k-points."""

    def setUp(self):
        self.structure = Structure.from_poscar(FIXTURES / "POSCAR_HfO2")

    def test_gamma_point_stays_on_the_diagonal_fast_path(self):
        # Regression guard: nice_supercell_matrix must not change behavior
        # for materials/k-points the plain diagonal search already handles.
        out = parse_smodes_output(load("STO_findsym_R"))
        m = nice_supercell_matrix(out.irrep("R5+").superlattice, Structure.from_poscar(FIXTURES / "POSCAR_STO").lattice)
        np.testing.assert_array_equal(m, np.diag([2, 2, 2]))

    def test_z_a_b_c_points_all_resolve_with_determinant_two(self):
        # irrep.n_atoms == 24, POSCAR_HfO2 has 12 -- volume ratio 2 is the
        # only physically sensible answer for all four k-points.
        for k in ["Z", "A", "B", "C"]:
            out = parse_smodes_output(load(f"HfO2_findsym_{k}"))
            irrep = out.irrep(f"{k}1")
            m = nice_supercell_matrix(irrep.superlattice, self.structure.lattice)
            self.assertEqual(round(np.linalg.det(m)), 2, msg=f"k={k}")

    def test_c_point_avoids_the_spurious_large_denominator_diagonal_match(self):
        # nice_diagonal_multipliers alone silently "succeeds" here with a
        # physically meaningless (2, 2, 23) (volume ratio 92!) because
        # Fraction.limit_denominator(24) finds *some* approximation within
        # tolerance even though row 2 has no true diagonal relationship.
        # nice_supercell_matrix's smaller internal fast-path denominator
        # budget must reject this and fall through to the correct,
        # non-diagonal, determinant-2 answer instead.
        out = parse_smodes_output(load("HfO2_findsym_C"))
        irrep = out.irrep("C1")
        spurious = nice_diagonal_multipliers(irrep.superlattice, self.structure.lattice, max_denominator=24)
        self.assertEqual(spurious, (2, 2, 23))

        m = nice_supercell_matrix(irrep.superlattice, self.structure.lattice)
        self.assertFalse(np.array_equal(m, np.diag(np.diag(m))))
        self.assertEqual(round(np.linalg.det(m)), 2)

    def test_z_point_matrix_reproduces_the_irrep_metric(self):
        # The defining property of nice_supercell_matrix's result: M applied
        # to the reference lattice's metric tensor must reproduce the
        # irrep's own superlattice metric (up to the rotation that
        # tile_onto_supercell recovers separately) -- this is what
        # _find_integer_representation actually searches for.
        out = parse_smodes_output(load("HfO2_findsym_Z"))
        irrep = out.irrep("Z1")
        m = nice_supercell_matrix(irrep.superlattice, self.structure.lattice)
        g_ref = self.structure.lattice @ self.structure.lattice.T
        g_target = irrep.superlattice @ irrep.superlattice.T
        np.testing.assert_allclose(m @ g_ref @ m.T, g_target, atol=1e-2)

    def test_unrelated_lattices_still_raise(self):
        out = parse_smodes_output(load("HfO2_findsym_Z"))
        irrep = out.irrep("Z1")
        bogus_reference = (2.0**0.5) * irrep.superlattice
        with self.assertRaises(SupercellError):
            nice_supercell_matrix(irrep.superlattice, bogus_reference, max_admixture_component=4)


class TestTileOntoSupercellRotationFallback(unittest.TestCase):
    """tile_onto_supercell must transparently apply the rotation
    nice_supercell_matrix's non-diagonal result implies -- this is the
    integration point that actually unblocks `irrep`/`phonons` for HfO2's
    non-Gamma k-points, not just the matrix search in isolation."""

    def setUp(self):
        self.structure = Structure.from_poscar(FIXTURES / "POSCAR_HfO2")

    def _check(self, k_label):
        out = parse_smodes_output(load(f"HfO2_findsym_{k_label}"))
        irrep = out.irrep(f"{k_label}1")
        m = nice_supercell_matrix(irrep.superlattice, self.structure.lattice)
        displacement_sets = {"atomic_displacement": [mode.displacements for mode in irrep.modes]}
        big_structure, big_sets = tile_onto_supercell(
            irrep, self.structure.lattice, m, displacement_sets, comment=f"hfo2 {k_label}"
        )
        self.assertEqual(big_structure.n_atoms, irrep.n_atoms)
        keys = {tuple(np.round(row, 4)) for row in big_structure.frac_coords}
        self.assertEqual(len(keys), big_structure.n_atoms)
        for arr in big_sets["atomic_displacement"]:
            self.assertAlmostEqual(float(np.linalg.norm(arr)), 1.0, places=6)
        return big_structure, big_sets

    def test_z_point_tiles_cleanly(self):
        self._check("Z")

    def test_a_point_tiles_cleanly(self):
        # A is the k-point that needs genuine admixture *and* rotation
        # jointly, not just permutation+rotation like Z -- the harder,
        # user-approved "full joint solver" case.
        self._check("A")

    def test_b_point_tiles_cleanly(self):
        self._check("B")

    def test_c_point_tiles_cleanly(self):
        self._check("C")

    def test_result_round_trips_through_poscar(self):
        import tempfile

        big_structure, _ = self._check("A")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "POSCAR_superlattice"
            big_structure.to_poscar(path)
            reloaded = Structure.from_poscar(path)
        self.assertEqual(reloaded.n_atoms, big_structure.n_atoms)
        self.assertEqual(reloaded.species, big_structure.species)






class TestMapParentValuesToNativeOriginShift(unittest.TestCase):
    """A real HPC bug report (Born effective charges from a DFPT run on
    TiO2's real 12-atom conventional parent, needing reduction onto a
    Gamma-point irrep's own smaller 6-atom native cell) surfaced a real
    origin-convention mismatch `map_parent_values_to_native` did not
    previously search for: TiO2's own GM3-/GM5- native atom reconstruction
    sits at a genuine, nonzero, fixed fractional shift `(-0.5, 0.25, 0.125)`
    relative to the parent POSCAR's own atoms -- confirmed directly (the
    unshifted direct match fails at ~1.5 A for the very first atom, far
    beyond any noise floor; the shift found here reconciles every one of
    GM3-'s 6 native atoms to a distinct parent atom, residual ~1e-5 A).
    CMO's own GM3-/X3- (`TestMagmomTilingRealCmo` above) need NO shift at
    all -- this is a real, material-dependent difference, not a universal
    correction, which is exactly why the fix tries the zero-shift match
    FIRST and only escalates on failure."""

    def setUp(self):
        self.parent = Structure.from_poscar(FIXTURES / "POSCAR_TiO2")

    def test_gm3_minus_needs_and_finds_the_real_origin_shift(self):
        irrep = parse_smodes_output(load("TiO2_findsym_GM")).irrep("GM3-")
        self.assertEqual(irrep.n_atoms, 6)
        self.assertEqual(self.parent.n_atoms, 12)
        # Label each parent atom with its own index (via a distinguishable
        # scalar) so the found mapping can be checked directly, not just
        # "didn't raise."
        parent_values = list(range(self.parent.n_atoms))
        native_values = map_parent_values_to_native(irrep, self.parent, parent_values)
        self.assertEqual(len(native_values), irrep.n_atoms)
        # Every mapped index must genuinely be the correct species.
        for species, parent_idx in zip(irrep.atom_species, native_values):
            self.assertEqual(self.parent.species[parent_idx], species)
        # A real, correct mapping cannot alias two different native atoms
        # onto the same parent atom (this is a reduction, not a genuine
        # multi-image case, unlike STO's own R-point).
        self.assertEqual(len(set(native_values)), irrep.n_atoms)

    def test_gm5_minus_also_needs_the_shift(self):
        # A second real, independently-checked irrep at the same k-point,
        # confirming this isn't specific to GM3- alone.
        irrep = parse_smodes_output(load("TiO2_findsym_GM")).irrep("GM5-")
        self.assertEqual(irrep.n_atoms, 6)
        parent_values = list(range(self.parent.n_atoms))
        native_values = map_parent_values_to_native(irrep, self.parent, parent_values)
        for species, parent_idx in zip(irrep.atom_species, native_values):
            self.assertEqual(self.parent.species[parent_idx], species)

    def test_still_raises_when_no_shift_can_reconcile_species(self):
        # The escalation is a real search, not a blanket "always succeed" --
        # a genuinely wrong-material parent (no matching species at all)
        # must still raise, even after trying every candidate shift.
        irrep = parse_smodes_output(load("TiO2_findsym_GM")).irrep("GM3-")
        bogus_parent = Structure(
            comment="bogus",
            lattice=self.parent.lattice,
            species=["Fe"] * self.parent.n_atoms,
            frac_coords=self.parent.frac_coords,
        )
        with self.assertRaises(SupercellError):
            map_parent_values_to_native(irrep, bogus_parent, [0.0] * self.parent.n_atoms)


if __name__ == "__main__":
    unittest.main()

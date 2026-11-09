import gzip
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from smodes_phonons import Structure, parse_smodes_output
from smodes_phonons.vasp import (
    LEPSILON_INCAR_OVERRIDES,
    SCF_INCAR_OVERRIDES,
    KpointsError,
    VaspOutputError,
    apply_incar_overrides,
    format_magmom,
    parse_born_effective_charges,
    parse_dielectric_tensor,
    parse_kpoints_mesh,
    parse_magmom,
    parse_outcar_forces,
    scaled_kpoints_mesh,
)

from _vasp_data import requires_vasp_data

# Modeled directly on the real legacy/INCAR_fcon template (never modified,
# per the developer notes -- this is a representative copy for test purposes).
_LEGACY_STYLE_INCAR = (
    "SYSTEM=BaSnO3\n"
    "\n"
    "NSW=150\n"
    "\n"
    "ISMEAR=-5\n"
    "\n"
    "LWAVE = .FALSE.\n"
    "LCHARG = .FALSE.\n"
    "#LASPH =.TRUE.\n"
    "\n"
    "PREC= Accurate\n"
    "ENCUT = 500\n"
    "EDIFF=1e-8\n"
    "NCORE=4\n"
    "#LEPSILON=.TRUE.\n"
    "GGA=PS\n"
    "\n"
    "LREAL = .FALSE.\n"
    "ISYM=0\n"
    "ISIF=3\n"
    "IBRION=2\n"
    "EDIFFG=-0.001\n"
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestParseOutcarForcesRealFixtures(unittest.TestCase):
    def _check(self, outcar_path: Path, reference_path: Path):
        forces = parse_outcar_forces(outcar_path)
        reference = np.loadtxt(reference_path)
        np.testing.assert_allclose(forces, reference)

    @requires_vasp_data
    def test_sto_all_modes(self):
        base = FIXTURES / "STO_Irrep_GM4-"
        for i in range(4):
            with self.subTest(mode=i):
                self._check(base / f"Mode_{i}" / "OUTCAR", base / f"Mode_{i}" / f"force_raw_output_Mode_{i}")

    @requires_vasp_data
    def test_tio2_all_modes_and_undistorted(self):
        base = FIXTURES / "TiO2_Irrep_GM5+"
        for i in range(3):
            with self.subTest(mode=i):
                self._check(base / f"Mode_{i}" / "OUTCAR", base / f"Mode_{i}" / f"force_raw_output_Mode_{i}")
        self._check(
            base / "Undistorted_lattice" / "OUTCAR",
            base / "Undistorted_lattice" / "force_raw_output_undistorted",
        )

    @requires_vasp_data
    def test_cmo_all_modes_and_undistorted(self):
        base = FIXTURES / "CMO_Irrep_X3-"
        for i in range(6):
            with self.subTest(mode=i):
                self._check(base / f"Mode_{i}" / "OUTCAR", base / f"Mode_{i}" / f"force_raw_output_Mode_{i}")
        self._check(
            base / "Undistorted_lattice" / "OUTCAR",
            base / "Undistorted_lattice" / "force_raw_output_undistorted",
        )

    @requires_vasp_data
    def test_returned_shape_matches_atom_count(self):
        forces = parse_outcar_forces(FIXTURES / "STO_Irrep_GM4-" / "Mode_0" / "OUTCAR")
        self.assertEqual(forces.shape, (5, 3))


class TestGzipHandling(unittest.TestCase):
    @requires_vasp_data
    def test_gzipped_outcar_parses_identically(self):
        real_outcar = FIXTURES / "STO_Irrep_GM4-" / "Mode_0" / "OUTCAR"
        with TemporaryDirectory() as tmp:
            gz_path = Path(tmp) / "OUTCAR.gz"
            with open(real_outcar, "rb") as src, gzip.open(gz_path, "wb") as dst:
                dst.write(src.read())

            plain = parse_outcar_forces(real_outcar)
            gzipped = parse_outcar_forces(gz_path)
        np.testing.assert_allclose(plain, gzipped)


class TestMalformedInput(unittest.TestCase):
    def test_missing_total_force_raises(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text("this is not an OUTCAR\n" * 5)
            with self.assertRaises(VaspOutputError):
                parse_outcar_forces(path)

    def test_takes_last_total_force_block(self):
        # Synthetic: no real fixture has more than one ionic step (all are
        # IBRION=-1 static calcs), but a relaxation OUTCAR would have one
        # TOTAL-FORCE block per ionic step -- the last is the correct one.
        text = """junk before
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000      0.00000      0.00000         1.000000      0.000000      0.000000
 -----------------------------------------------------------------------------------
    total drift:                               0.000000      0.000000      0.000000
some text between ionic steps
 POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
      0.00000      0.00000      0.00000         2.000000      0.000000      0.000000
 -----------------------------------------------------------------------------------
    total drift:                               0.000000      0.000000      0.000000
"""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text(text)
            forces = parse_outcar_forces(path)
        np.testing.assert_allclose(forces, [[2.0, 0.0, 0.0]])

    def test_negative_leading_coordinate_does_not_confuse_separator_detection(self):
        # Regression guard: an atom row whose first (position) column is
        # negative starts with '-' after stripping, same as the dashed
        # separator line -- must not be mistaken for one.
        text = """ POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
     -1.94930      0.00000      0.00000         0.500000      0.000000      0.000000
      1.94930      0.00000      0.00000        -0.500000      0.000000      0.000000
 -----------------------------------------------------------------------------------
    total drift:                               0.000000      0.000000      0.000000
"""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text(text)
            forces = parse_outcar_forces(path)
        np.testing.assert_allclose(forces, [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]])

    def test_no_atom_rows_raises(self):
        text = """ POSITION                                       TOTAL-FORCE (eV/Angst)
 -----------------------------------------------------------------------------------
 -----------------------------------------------------------------------------------
    total drift:                               0.000000      0.000000      0.000000
"""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text(text)
            with self.assertRaises(VaspOutputError):
                parse_outcar_forces(path)


class TestParseKpointsMesh(unittest.TestCase):
    def test_typical_automatic_mesh(self):
        text = "Automatic mesh\n0\nGamma\n10 10 10\n0 0 0\n"
        kmesh = parse_kpoints_mesh(text)
        self.assertEqual(kmesh.comment, "Automatic mesh")
        self.assertEqual(kmesh.scheme, "Gamma")
        self.assertEqual(kmesh.mesh, (10, 10, 10))
        self.assertEqual(kmesh.shift, (0.0, 0.0, 0.0))

    def test_mesh_without_shift_line(self):
        text = "Automatic mesh\n0\nMonkhorst-Pack\n6 6 4\n"
        kmesh = parse_kpoints_mesh(text)
        self.assertEqual(kmesh.scheme, "Monkhorst-Pack")
        self.assertEqual(kmesh.mesh, (6, 6, 4))
        self.assertIsNone(kmesh.shift)

    def test_explicit_kpoint_list_raises(self):
        # line 2 != "0" means an explicit list of k-points, not a mesh --
        # scaling doesn't make sense for that, so this must not be guessed at.
        text = "Explicit list\n2\nReciprocal\n0 0 0 1\n0.5 0 0 1\n"
        with self.assertRaises(KpointsError):
            parse_kpoints_mesh(text)

    def test_too_short_raises(self):
        with self.assertRaises(KpointsError):
            parse_kpoints_mesh("only\ntwo lines\n")

    def test_malformed_mesh_line_raises(self):
        text = "Automatic mesh\n0\nGamma\nnot a mesh\n"
        with self.assertRaises(KpointsError):
            parse_kpoints_mesh(text)


class TestScaledKpointsMesh(unittest.TestCase):
    def test_matches_worked_example(self):
        # The user's own worked example: a 4x4x4 cube with a 10x10x10 mesh,
        # doubled along one axis -> half as many k-points along that axis.
        kmesh = parse_kpoints_mesh("Automatic mesh\n0\nGamma\n10 10 10\n0 0 0\n")
        original = np.diag([4.0, 4.0, 4.0])
        new = np.diag([8.0, 4.0, 4.0])
        scaled = scaled_kpoints_mesh(kmesh, original, new)
        self.assertEqual(scaled.mesh, (5, 10, 10))
        self.assertEqual(scaled.scheme, "Gamma")
        self.assertEqual(scaled.shift, kmesh.shift)

    def test_shrinking_the_cell_increases_mesh(self):
        kmesh = parse_kpoints_mesh("Automatic mesh\n0\nGamma\n5 5 5\n")
        original = np.diag([8.0, 8.0, 8.0])
        new = np.diag([4.0, 4.0, 4.0])
        scaled = scaled_kpoints_mesh(kmesh, original, new)
        self.assertEqual(scaled.mesh, (10, 10, 10))

    def test_minimum_one_clamping(self):
        kmesh = parse_kpoints_mesh("Automatic mesh\n0\nGamma\n2 2 2\n")
        original = np.diag([4.0, 4.0, 4.0])
        new = np.diag([400.0, 400.0, 400.0])  # huge supercell multiplier
        scaled = scaled_kpoints_mesh(kmesh, original, new)
        self.assertEqual(scaled.mesh, (1, 1, 1))

    def test_write_round_trips(self):
        kmesh = parse_kpoints_mesh("Automatic mesh\n0\nGamma\n10 10 10\n0 0 0\n")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "KPOINTS"
            kmesh.write(path)
            reloaded = parse_kpoints_mesh(path.read_text())
        self.assertEqual(reloaded.mesh, kmesh.mesh)
        self.assertEqual(reloaded.scheme, kmesh.scheme)
        self.assertEqual(reloaded.shift, kmesh.shift)

    def test_write_without_shift(self):
        kmesh = parse_kpoints_mesh("Automatic mesh\n0\nMonkhorst-Pack\n6 6 4\n")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "KPOINTS"
            kmesh.write(path)
            text = path.read_text()
        self.assertEqual(len(text.splitlines()), 4)


class TestScaledKpointsMeshRealSupercells(unittest.TestCase):
    """The user's own worked example is a trivial diagonal supercell
    (8,0,0/0,4,0/0,0,4). Real SMODES supercells are almost never that simple
    -- they're rotated/skewed relative to the parent axes (that's the whole
    reason this feature exists). These use real, non-diagonal SMODES output
    to make sure the per-row-length-ratio rule degrades sensibly rather than
    only being validated on a case simple enough to not need it."""

    def test_sto_m_point_in_plane_rotated_supercell(self):
        # M2-'s superlattice is rotated 45 degrees in the a-b plane and
        # lengthened by sqrt(2) there, c unchanged -- expect a matching
        # reduction in the first two mesh directions only.
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        out = parse_smodes_output((FIXTURES / "STO_findsym_M").read_text())
        superlattice = out.irrep("M2-").superlattice

        row_lengths = np.linalg.norm(superlattice, axis=1)
        self.assertAlmostEqual(row_lengths[0], row_lengths[1], places=6)
        self.assertGreater(row_lengths[0], row_lengths[2])  # a,b rows longer than c row

        kmesh = parse_kpoints_mesh("Automatic mesh\n0\nGamma\n10 10 10\n0 0 0\n")
        scaled = scaled_kpoints_mesh(kmesh, structure.lattice, superlattice)
        self.assertEqual(scaled.mesh, (7, 7, 10))

    def test_sto_r_point_uniformly_folded_supercell(self):
        # R5+ folds along all three directions uniformly -- every superlattice
        # row has the same length, so the mesh should reduce uniformly too.
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        out = parse_smodes_output((FIXTURES / "STO_findsym_R").read_text())
        superlattice = out.irrep("R5+").superlattice

        row_lengths = np.linalg.norm(superlattice, axis=1)
        self.assertAlmostEqual(row_lengths[0], row_lengths[1], places=6)
        self.assertAlmostEqual(row_lengths[1], row_lengths[2], places=6)

        kmesh = parse_kpoints_mesh("Automatic mesh\n0\nGamma\n10 10 10\n0 0 0\n")
        scaled = scaled_kpoints_mesh(kmesh, structure.lattice, superlattice)
        self.assertEqual(scaled.mesh, (7, 7, 7))

    def test_cmo_x_point_non_cubic_parent_and_skewed_supercell(self):
        # The hardest real case available: a non-cubic (tetragonal) parent
        # cell, and a body-centered-derived superlattice whose rows mix all
        # three parent axes unevenly -- this is what "if the POSCAR in the
        # Mode folder has a weird shape" actually looks like in practice.
        structure = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        out = parse_smodes_output((FIXTURES / "CMO_findsym_X").read_text())
        superlattice = out.irrep("X3-").superlattice

        kmesh = parse_kpoints_mesh("Automatic mesh\n0\nGamma\n6 6 2\n0 0 0\n")
        scaled = scaled_kpoints_mesh(kmesh, structure.lattice, superlattice)
        # Asymmetric reduction -- rows 1,2 pick up a large c-axis component
        # (c is ~5x longer than a,b in this material) and shrink hard; row 3
        # stays purely in-plane and shrinks only modestly.
        self.assertEqual(scaled.mesh, (2, 2, 7))
        # Every value comes from a real, independently-computed row length --
        # cross-check none of them collapsed to the minimum-1 clamp by
        # accident (that would mask a real error as a trivial pass).
        self.assertTrue(all(n > 1 for n in scaled.mesh))


class TestApplyIncarOverrides(unittest.TestCase):
    def test_scf_overrides_force_nsw_and_ibrion_leave_everything_else(self):
        result = apply_incar_overrides(_LEGACY_STYLE_INCAR, SCF_INCAR_OVERRIDES)
        lines = result.splitlines()
        self.assertIn("NSW = 0", lines)
        self.assertIn("IBRION = -1", lines)
        # ISYM=0 is forced too (finite-displacement forces need VASP's own
        # symmetry detection disabled, or it can silently fail to notice a
        # tiny 0.001 A symmetry-breaking displacement and symmetrize away
        # part of the very force being measured) -- the template's own
        # ISYM=0 happens to already have this value, but it's still
        # rewritten in place to the override's own "TAG = value" format,
        # not left in its original no-space "ISYM=0" form, since it's now
        # an actively-forced tag rather than a passed-through one.
        self.assertIn("ISYM = 0", lines)
        self.assertNotIn("ISYM=0", lines)
        # A relax-only NSW=150/IBRION=2 must be gone, replaced in place --
        # not merely appended alongside the old value.
        self.assertNotIn("NSW=150", lines)
        self.assertNotIn("IBRION=2", lines)
        # ISIF is untouched -- VASP ignores it at NSW=0/IBRION=-1 (see
        # SCF_INCAR_OVERRIDES's own docstring), so the user's own ISIF=3
        # passes straight through.
        self.assertIn("ISIF=3", lines)
        # Every material-specific tag survives byte-for-byte.
        for line in ["SYSTEM=BaSnO3", "ENCUT = 500", "GGA=PS", "NCORE=4", "PREC= Accurate"]:
            self.assertIn(line, lines)
        # Comments pass through untouched too.
        self.assertIn("#LASPH =.TRUE.", lines)

    def test_empty_overrides_is_a_pure_pass_through(self):
        # The `freeze` case: no tags forced at all, template written as-is.
        result = apply_incar_overrides(_LEGACY_STYLE_INCAR, {})
        self.assertEqual(result, _LEGACY_STYLE_INCAR)

    def test_override_case_insensitive_and_whitespace_tolerant(self):
        template = "nsw = 150\nibrion=2\n"
        result = apply_incar_overrides(template, {"NSW": "0", "IBRION": "-1"})
        self.assertIn("NSW = 0", result.splitlines())
        self.assertIn("IBRION = -1", result.splitlines())
        self.assertEqual(len(result.splitlines()), 2)  # replaced in place, not appended

    def test_override_appended_when_tag_absent_from_template(self):
        template = "SYSTEM=Test\nENCUT=500\n"
        result = apply_incar_overrides(template, {"NSW": "0", "IBRION": "-1"})
        lines = result.splitlines()
        self.assertIn("SYSTEM=Test", lines)
        self.assertIn("ENCUT=500", lines)
        self.assertIn("NSW = 0", lines)
        self.assertIn("IBRION = -1", lines)

    def test_commented_out_tag_is_not_matched_a_new_active_line_is_appended(self):
        # A commented '#ISIF=3' must not be treated as an existing ISIF
        # line -- the override has to be a real, active line.
        template = "SYSTEM=Test\n#ISIF=3\n"
        result = apply_incar_overrides(template, {"ISIF": "2"})
        lines = result.splitlines()
        self.assertIn("#ISIF=3", lines)  # untouched
        self.assertIn("ISIF = 2", lines)  # appended, active

    def test_lepsilon_overrides_force_lepsilon_and_ibrion(self):
        result = apply_incar_overrides(_LEGACY_STYLE_INCAR, LEPSILON_INCAR_OVERRIDES)
        lines = result.splitlines()
        self.assertIn("LEPSILON = .TRUE.", lines)
        self.assertIn("IBRION = -1", lines)
        # The relax-only NSW=150 (irrelevant to a LEPSILON run) is left
        # alone -- LEPSILON_INCAR_OVERRIDES only forces LEPSILON/IBRION.
        self.assertIn("NSW=150", lines)
        for line in ["SYSTEM=BaSnO3", "ENCUT = 500", "GGA=PS"]:
            self.assertIn(line, lines)


class TestMagmom(unittest.TestCase):
    """Validated against the physicist's own real magnetic/DFT+U material
    (Ca3Mn2O7, tests/fixtures/CMO_INCAR_magnetic -- real MAGMOM/LDAU
    settings, AFM Mn ordering) -- see supercell.py's own
    TestMagmomTilingRealCmo for the full parent-to-supercell pipeline this
    parsing/formatting feeds into."""

    def test_parses_real_cmo_magmom_shorthand(self):
        text = (FIXTURES / "CMO_INCAR_magnetic").read_text()
        values = parse_magmom(text)
        self.assertEqual(len(values), 24)
        # 6 non-magnetic Ca, AFM Mn (+3,-3,+3,-3), 14 non-magnetic O.
        self.assertEqual(values[:6], [0.0] * 6)
        self.assertEqual(values[6:10], [3.0, -3.0, 3.0, -3.0])
        self.assertEqual(values[10:], [0.0] * 14)

    def test_no_magmom_tag_returns_none(self):
        self.assertIsNone(parse_magmom(_LEGACY_STYLE_INCAR))

    def test_format_uses_repeat_shorthand_for_runs(self):
        self.assertEqual(format_magmom([0, 0, 0, 5, 5, -5]), "3*0  2*5  -5")

    def test_format_single_value_no_shorthand(self):
        self.assertEqual(format_magmom([1.5]), "1.5")

    def test_round_trips_through_incar_text(self):
        values = [0.0, 0.0, 3.0, -3.0, 3.0, -3.0, 0.0]
        text = f"MAGMOM = {format_magmom(values)}\n"
        self.assertEqual(parse_magmom(text), values)

    def test_malformed_token_raises(self):
        with self.assertRaises(VaspOutputError):
            parse_magmom("MAGMOM = 3*x 5\n")

    def test_format_empty_list_raises(self):
        with self.assertRaises(ValueError):
            format_magmom([])


# Modeled on VASP's own documented LEPSILON=.TRUE. OUTCAR format -- NOT
# copied from a real OUTCAR (none exists in this project's fixtures yet;
# the physicist is generating real STO/BaTiO3 LEPSILON data). Good enough
# to exercise the parsers' own logic; header text/spacing will be checked
# against real output once available (see vasp.py's own docstring notes).
_BORN_CHARGES_OUTCAR = """
 BORN EFFECTIVE CHARGES (in e, cummulative output)
 -------------------------------------------------


 ion    1
    1     2.90420     0.00000     0.00000
    2     0.00000     2.90420     0.00000
    3     0.00000     0.00000     2.90420

 ion    2
    1    -2.02458     0.00000     0.00000
    2     0.00000    -2.02458     0.00000
    3     0.00000     0.00000    -2.02458

 ion    3
    1     1.50000     0.10000     0.00000
    2     0.05000     1.55000     0.00000
    3     0.00000     0.00000     1.60000


 MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)
 ------------------------------------------------------
       6.246632     0.000000     0.000000
       0.000000     6.246632     0.000000
       0.000000     0.000000     6.246632
 ------------------------------------------------------
"""

_IONIC_ONLY_DIELECTRIC_OUTCAR = """
 MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION
 ------------------------------------------------------
       1.000000     0.000000     0.000000
       0.000000     1.000000     0.000000
       0.000000     0.000000     1.000000
 ------------------------------------------------------
"""


class TestParseBornEffectiveCharges(unittest.TestCase):
    def test_parses_three_ions_in_order(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text(_BORN_CHARGES_OUTCAR)
            born = parse_born_effective_charges(path)

        self.assertEqual(born.shape, (3, 3, 3))
        np.testing.assert_allclose(born[0], np.eye(3) * 2.9042)
        np.testing.assert_allclose(born[1], np.eye(3) * -2.02458)
        np.testing.assert_allclose(born[2], [[1.5, 0.1, 0.0], [0.05, 1.55, 0.0], [0.0, 0.0, 1.6]])

    def test_gzip_supported(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR.gz"
            with gzip.open(path, "wt") as fh:
                fh.write(_BORN_CHARGES_OUTCAR)
            born = parse_born_effective_charges(path)
        self.assertEqual(born.shape, (3, 3, 3))

    def test_missing_header_raises(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text("nothing relevant here\n")
            with self.assertRaises(VaspOutputError):
                parse_born_effective_charges(path)

    def test_malformed_row_raises(self):
        text = _BORN_CHARGES_OUTCAR.replace("    1     2.90420     0.00000     0.00000", "    1     garbage")
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text(text)
            with self.assertRaises(VaspOutputError):
                parse_born_effective_charges(path)


class TestParseDielectricTensor(unittest.TestCase):
    def test_parses_electronic_tensor(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text(_BORN_CHARGES_OUTCAR)
            eps = parse_dielectric_tensor(path)
        np.testing.assert_allclose(eps, np.eye(3) * 6.246632)

    def test_ionic_only_block_raises_a_clear_explanatory_error(self):
        # Must not silently fall back to the ionic-contribution tensor --
        # that would double-count the phonon response the Born charges
        # already carry in the non-analytic correction formula.
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text(_IONIC_ONLY_DIELECTRIC_OUTCAR)
            with self.assertRaises(VaspOutputError) as ctx:
                parse_dielectric_tensor(path)
        self.assertIn("electronic", str(ctx.exception))

    def test_missing_block_raises(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "OUTCAR"
            path.write_text("nothing relevant here\n")
            with self.assertRaises(VaspOutputError):
                parse_dielectric_tensor(path)


class TestBornChargesRealFixtures(unittest.TestCase):
    """Real VASP LEPSILON=.TRUE. calculations, supplied by the physicist --
    a real .gz-compressed OUTCAR for cubic SrTiO3 (space group 221) and
    tetragonal ferroelectric BaTiO3 (space group 99, P4mm), the exact data
    the developer notes' LO-TO/BEC section had flagged as still pending. Physically
    independent checks (not just "parses without crashing"): the acoustic
    sum rule (charge neutrality -- Born charges must sum to ~0 across all
    atoms, a real physical constraint the DFT calculation itself has to
    satisfy, not something this parser enforces), and known real physics
    (Ti's anomalous/giant Born charge, ~7x nominal +4, a textbook signature
    of dynamical charge transfer in perovskites; O's anisotropic charge,
    larger along its own Ti-O bond direction than transverse to it)."""

    @requires_vasp_data
    def test_sto_born_charges_satisfy_acoustic_sum_rule_and_known_physics(self):
        born = parse_born_effective_charges(FIXTURES / "BEC_STO" / "OUTCAR.gz")
        self.assertEqual(born.shape, (5, 3, 3))
        # Charge neutrality: summing every atom's own Born charge tensor
        # must return (approximately) zero -- a real constraint on the DFT
        # calculation, not something this parser can enforce by construction.
        np.testing.assert_allclose(born.sum(axis=0), np.zeros((3, 3)), atol=2e-4)
        sr, ti = born[0], born[1]
        np.testing.assert_allclose(sr, np.eye(3) * 2.5515, atol=2e-4)
        # Ti's real, well-known "anomalous" Born charge -- far above its
        # formal +4 oxidation state, the textbook signature of dynamical
        # Ti-O charge transfer that helps drive ferroelectricity in
        # perovskites.
        np.testing.assert_allclose(ti, np.eye(3) * 7.4093, atol=2e-4)
        self.assertGreater(ti[0, 0], 7.0)
        # Each oxygen's own charge is anisotropic: larger (in magnitude)
        # along its own Ti-O bond axis than transverse to it -- confirmed
        # per-atom, not just "some large value exists somewhere".
        for o in born[2:]:
            diag = np.abs(np.diag(o))
            self.assertAlmostEqual(float(np.max(diag)), 5.8725, places=3)
            self.assertAlmostEqual(float(np.min(diag)), 2.0441, places=3)

    @requires_vasp_data
    def test_sto_dielectric_tensor_isotropic_cubic(self):
        eps = parse_dielectric_tensor(FIXTURES / "BEC_STO" / "OUTCAR.gz")
        np.testing.assert_allclose(eps, np.eye(3) * 6.425368, atol=1e-4)

    @requires_vasp_data
    def test_bto_born_charges_satisfy_acoustic_sum_rule_and_tetragonal_anisotropy(self):
        born = parse_born_effective_charges(FIXTURES / "BEC_BTO_99" / "OUTCAR.gz")
        self.assertEqual(born.shape, (5, 3, 3))
        np.testing.assert_allclose(born.sum(axis=0), np.zeros((3, 3)), atol=2e-4)
        ti = born[1]
        # Tetragonal (P4mm) symmetry breaks cubic Ti's isotropic Born
        # charge into a distinct in-plane (xx=yy) vs. c-axis (zz) value --
        # the real, physical fingerprint of the ferroelectric distortion
        # along c, not just numerical noise (the split is large: ~1.6 e).
        self.assertAlmostEqual(float(ti[0, 0]), float(ti[1, 1]), places=3)
        self.assertGreater(abs(ti[0, 0] - ti[2, 2]), 1.0)

    @requires_vasp_data
    def test_bto_dielectric_tensor_tetragonal_anisotropy(self):
        eps = parse_dielectric_tensor(FIXTURES / "BEC_BTO_99" / "OUTCAR.gz")
        self.assertAlmostEqual(float(eps[0, 0]), float(eps[1, 1]), places=4)
        self.assertNotAlmostEqual(float(eps[0, 0]), float(eps[2, 2]), places=2)


if __name__ == "__main__":
    unittest.main()

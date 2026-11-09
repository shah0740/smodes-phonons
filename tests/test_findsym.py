import stat
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from smodes_phonons import (
    FindsymLogError,
    Structure,
    WyckoffPosition,
    align_by_nearest_neighbor,
    build_findsym_input,
    build_smodes_input,
    detect_findsym_version,
    parse_findsym_log,
    parse_reduced_structure,
    parse_smodes_output,
    reduce_to_primitive_cell,
    run_findsym,
)
from smodes_phonons import findsym, isotropy

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TestBuildFindsymInput(unittest.TestCase):
    def test_sto_species_indices_and_field_order(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        text = build_findsym_input(structure, title="STO structure", tolerance=0.0)
        lines = text.splitlines()

        self.assertEqual(lines[0], "STO structure")
        self.assertEqual(lines[1], "0")
        self.assertEqual(lines[2], "1")
        # 3 lattice vector lines, then identity transform, then atom count
        self.assertEqual(lines[6], "1 0 0")
        self.assertEqual(lines[7], "0 1 0")
        self.assertEqual(lines[8], "0 0 1")
        self.assertEqual(lines[9], "5")
        # Sr, Ti, O, O, O -> distinct species get distinct 1-based indices,
        # not legacy's blanket "1" for every atom.
        self.assertEqual(lines[10], "1 2 3 3 3")
        self.assertEqual(len(lines), 11 + 5)  # header + 5 position lines

    def test_tio2_species_indices(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_TiO2")
        text = build_findsym_input(structure, title="TiO2 structure", tolerance=0.0)
        lines = text.splitlines()
        # Ti x4, O x8 -> Ti=1, O=2 (anatase conventional cell, matching the
        # real TiO2_findsym_GM pair -- see the developer notes, the TiO2 fixture was
        # corrected from a mismatched rutile POSCAR to a matched anatase one)
        self.assertEqual(lines[9], "12")
        self.assertEqual(lines[10], "1 1 1 1 2 2 2 2 2 2 2 2")

    def test_tolerance_is_rendered_on_line_two(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        text = build_findsym_input(structure, title="t", tolerance=0.01)
        self.assertEqual(text.splitlines()[1], "0.01")

    def test_lattice_uses_scaled_structure_lattice_not_raw_poscar_lines(self):
        # Regression guard for the POSCAR-scale-factor bug: build_findsym_input
        # must read structure.lattice (already scale-corrected), never re-derive
        # from raw POSCAR text.
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        text = build_findsym_input(structure, title="t", tolerance=0.0)
        lat_lines = text.splitlines()[3:6]
        parsed = [[float(x) for x in line.split()] for line in lat_lines]
        for row, expected_row in zip(parsed, structure.lattice.tolist()):
            for value, expected in zip(row, expected_row):
                self.assertAlmostEqual(value, expected, places=8)


def _synthetic_findsym_log(n_atoms: int, positions: list, wyckoff_section: str) -> str:
    """A minimal but structurally real findsym.log, for edge cases no real
    fixture exercises (mixed-species orbit, atom-count mismatch, multiple
    free Wyckoff axes). Modeled on tests/fixtures/CMO_findsym's real section order
    and headers -- never added to legacy/, which stays real-fixtures-only.
    """
    position_lines = "\n".join(
        f"  {i + 1}   {x:.5f}   {y:.5f}   {z:.5f}" for i, (x, y, z) in enumerate(positions)
    )
    type_line = "  " + "  ".join(["1"] * n_atoms)
    return f"""findsym, Version 2.5, February 2001
Written by Harold T. Stokes and Dorian M. Hatch
Brigham Young University

Sample input file to identify_space_group.  BSO_1uc
Accuracy of data:
 0.000E+00
Form of lattice parameters:  1 = vectors, 2 = lengths and angles
1
Lattice vectors in cartesian coordinates:
   4.00000   0.00000   0.00000
   0.00000   4.00000   0.00000
   0.00000   0.00000   4.00000
Lattice parameters, a,b,c,alpha,beta,gamma:
   4.00000   4.00000   4.00000  90.00000  90.00000  90.00000
Vectors defining the unit cell (dimensionless coordinates):
   1.00000   0.00000   0.00000
   0.00000   1.00000   0.00000
   0.00000   0.00000   1.00000
Number of atoms in unit cell:
 {n_atoms}
Type of each atom:
{type_line}
Position of each atom (dimensionless coordinates)
{position_lines}
Space Group 221  Oh-1      Pm-3m
Values of a,b,c,alpha,beta,gamma:
   4.00000   4.00000   4.00000  90.00000  90.00000  90.00000

Atomic positions in terms of a,b,c:

{wyckoff_section}
"""


class TestParseFindsymLog(unittest.TestCase):
    def test_real_cmo_fixture(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        text = (FIXTURES / "CMO_findsym").read_text()
        info = parse_findsym_log(text, structure, title="CMO_findsym")

        self.assertEqual(info.title, "CMO_findsym")
        self.assertEqual(info.space_group_number, 139)
        self.assertEqual(info.lattice_parameters, (3.72627, 3.72627, 19.05536, 90.0, 90.0, 90.0))
        self.assertEqual(
            info.wyckoff_positions,
            [
                WyckoffPosition("Ca", "b"),
                WyckoffPosition("Ca", "e", z=0.31599),
                WyckoffPosition("Mn", "e", z=0.09970),
                WyckoffPosition("O", "a"),
                WyckoffPosition("O", "e", z=0.19799),
                WyckoffPosition("O", "g", z=0.09784),
            ],
        )

    def test_matches_real_cmo_smodes_input_byte_for_byte(self):
        # End-to-end: real POSCAR -> real findsym.log -> parse_findsym_log ->
        # build_smodes_input must reproduce the real smodes_input exactly.
        # tests/fixtures/CMO_findsym_smodes_input is the real, matched input that
        # produced tests/fixtures/CMO_findsym_X (k-vector X).
        structure = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        text = (FIXTURES / "CMO_findsym").read_text()
        info = parse_findsym_log(text, structure, title="CMO_findsym")
        built = build_smodes_input(info, "X")
        self.assertEqual(built, (FIXTURES / "CMO_findsym_smodes_input").read_text())

    def test_cross_validates_against_independently_parsed_smodes_output(self):
        # tests/fixtures/CMO_findsym (findsym log) and tests/fixtures/CMO_findsym_X (smodes
        # output) were generated independently. What parse_findsym_log
        # recovers from the log must agree with what parse_smodes_output
        # recovers from the completely separate output file.
        structure = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        from_log = parse_findsym_log((FIXTURES / "CMO_findsym").read_text(), structure, title="CMO_findsym")
        from_output = parse_smodes_output((FIXTURES / "CMO_findsym_X").read_text())

        self.assertEqual(from_log.space_group_number, from_output.space_group_number)
        self.assertEqual(from_log.lattice_parameters, from_output.lattice_parameters)
        self.assertEqual(from_log.wyckoff_positions, from_output.wyckoff_positions)

    def test_mixed_species_in_one_orbit_raises(self):
        structure = Structure(
            comment="synthetic",
            lattice=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            species=["Sr", "Ti"],
            frac_coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
        )
        log = _synthetic_findsym_log(
            2,
            [(0, 0, 0), (0.5, 0.5, 0.5)],
            "Wyckoff position a\n   1   0.00000   0.00000   0.00000\n   2   0.50000   0.50000   0.50000",
        )
        with self.assertRaises(FindsymLogError):
            parse_findsym_log(log, structure, title="t")

    def test_atom_count_mismatch_raises(self):
        structure = Structure(
            comment="synthetic",
            lattice=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            species=["Sr", "Ti", "O"],
            frac_coords=[[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0, 0]],
        )
        log = _synthetic_findsym_log(
            2,
            [(0, 0, 0), (0.5, 0.5, 0.5)],
            "Wyckoff position a\n   1   0.00000   0.00000   0.00000",
        )
        with self.assertRaises(FindsymLogError):
            parse_findsym_log(log, structure, title="t")

    def test_two_free_axes(self):
        structure = Structure(
            comment="synthetic",
            lattice=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            species=["Sr", "Sr"],
            frac_coords=[[0.12345, 0.67890, 0.0], [0.87655, 0.32110, 0.0]],
        )
        log = _synthetic_findsym_log(
            2,
            [(0.12345, 0.67890, 0.0), (0.87655, 0.32110, 0.0)],
            "Wyckoff position h, x =  0.12345, y =  0.67890\n"
            "   1   0.12345   0.67890   0.00000\n"
            "   2   0.87655   0.32110   0.00000",
        )
        info = parse_findsym_log(log, structure, title="t")
        wyc = info.wyckoff_positions[0]
        self.assertEqual(wyc.letter, "h")
        self.assertAlmostEqual(wyc.x, 0.12345)
        self.assertAlmostEqual(wyc.y, 0.67890)
        self.assertIsNone(wyc.z)

    def test_three_free_axes(self):
        structure = Structure(
            comment="synthetic",
            lattice=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            species=["Sr"],
            frac_coords=[[0.11111, 0.22222, 0.33333]],
        )
        log = _synthetic_findsym_log(
            1,
            [(0.11111, 0.22222, 0.33333)],
            "Wyckoff position i, x =  0.11111, y =  0.22222, z =  0.33333\n"
            "   1   0.11111   0.22222   0.33333",
        )
        info = parse_findsym_log(log, structure, title="t")
        wyc = info.wyckoff_positions[0]
        self.assertEqual(wyc.letter, "i")
        self.assertAlmostEqual(wyc.x, 0.11111)
        self.assertAlmostEqual(wyc.y, 0.22222)
        self.assertAlmostEqual(wyc.z, 0.33333)

    def test_missing_atomic_positions_section_raises(self):
        structure = Structure(
            comment="synthetic",
            lattice=[[4, 0, 0], [0, 4, 0], [0, 0, 4]],
            species=["Sr"],
            frac_coords=[[0, 0, 0]],
        )
        log = _synthetic_findsym_log(1, [(0, 0, 0)], "").replace(
            "Atomic positions in terms of a,b,c:\n", "Something else entirely\n"
        )
        with self.assertRaises(FindsymLogError):
            parse_findsym_log(log, structure, title="t")


class TestRunFindsym(unittest.TestCase):
    def _make_stub(self, tmp: str, script: str) -> str:
        stub = Path(tmp) / "fake_findsym"
        stub.write_text(script)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        return str(stub)

    def test_missing_binary_raises(self):
        from smodes_phonons import findsym as findsym_module

        original = findsym_module.isotropy.find_binary
        findsym_module.isotropy.find_binary = lambda name: None
        try:
            with self.assertRaises(FileNotFoundError):
                run_findsym("irrelevant input")
        finally:
            findsym_module.isotropy.find_binary = original

    def test_writes_findsym_log_in_temp_cwd_and_returns_contents(self):
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(
                tmp,
                "#!/bin/sh\n"
                "cat > /dev/null\n"  # consume stdin
                "echo 'fake findsym log contents' > findsym.log\n",
            )
            log_text = run_findsym("irrelevant input\n", binary=stub)
        self.assertEqual(log_text, "fake findsym log contents\n")

    def test_isodata_env_var_is_set_for_subprocess(self):
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(
                tmp,
                "#!/bin/sh\n"
                "cat > /dev/null\n"
                'echo "ISODATA=$ISODATA" > findsym.log\n',
            )
            log_text = run_findsym("x\n", binary=stub, isodata="/some/data/path/")
        self.assertEqual(log_text, "ISODATA=/some/data/path/\n")

    def test_nonzero_exit_raises(self):
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(
                tmp, "#!/bin/sh\ncat > /dev/null\necho boom >&2\nexit 1\n"
            )
            with self.assertRaises(RuntimeError):
                run_findsym("x\n", binary=stub)

    def test_missing_log_after_success_raises(self):
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(tmp, "#!/bin/sh\ncat > /dev/null\n")  # never writes findsym.log
            with self.assertRaises(RuntimeError):
                run_findsym("x\n", binary=stub)

    def test_each_run_gets_a_fresh_cwd(self):
        # Two concurrent-ish runs must not see each other's leftover findsym.log.
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(
                tmp,
                "#!/bin/sh\n"
                "cat > /dev/null\n"
                "if [ -f findsym.log ]; then echo STALE > findsym.log; "
                "else echo FRESH > findsym.log; fi\n",
            )
            first = run_findsym("x\n", binary=stub)
            second = run_findsym("x\n", binary=stub)
        self.assertEqual(first, "FRESH\n")
        self.assertEqual(second, "FRESH\n")

    def test_separate_findsym_cif_is_appended_when_log_has_no_inline_cif(self):
        # Real 7.1.6 does this for some space-group settings (e.g. hexagonal
        # axes) -- confirmed directly against graphite/SiO2/BaTiO3 real runs,
        # see direction.py and the developer notes. Without this, that data is silently
        # lost, since only findsym.log used to be read.
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(
                tmp,
                "#!/bin/sh\n"
                "cat > /dev/null\n"
                "echo 'fake log body, no CIF marker here' > findsym.log\n"
                "echo '# CIF file created by FINDSYM, version 7.1.3' > findsym.cif\n"
                "echo 'fake cif body' >> findsym.cif\n",
            )
            log_text = run_findsym("x\n", binary=stub)
        self.assertIn("fake log body, no CIF marker here", log_text)
        self.assertIn("# CIF file created by FINDSYM", log_text)
        self.assertIn("fake cif body", log_text)

    def test_separate_findsym_cif_is_not_double_appended_when_log_already_has_one(self):
        # STO/TiO2/CMO's real logs embed the CIF directly -- don't append the
        # separate file's content a second time in that case.
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(
                tmp,
                "#!/bin/sh\n"
                "cat > /dev/null\n"
                "printf 'fake log body\\n# CIF file created by FINDSYM\\ninline cif body\\n' > findsym.log\n"
                "echo 'this must NOT appear in the result' > findsym.cif\n",
            )
            log_text = run_findsym("x\n", binary=stub)
        self.assertEqual(log_text.count("# CIF file created"), 1)
        self.assertNotIn("this must NOT appear in the result", log_text)

    def test_no_separate_cif_file_is_fine(self):
        # Pre-6.x logs never have a findsym.cif at all -- must not error.
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(tmp, "#!/bin/sh\ncat > /dev/null\necho 'plain old log' > findsym.log\n")
            log_text = run_findsym("x\n", binary=stub)
        self.assertEqual(log_text, "plain old log\n")


class TestDetectFindsymVersion(unittest.TestCase):
    """detect_findsym_version probes with empty stdin -- both real findsym
    versions print their banner as the first output line before reading any
    input at all (confirmed against the real 7.1.6 binary), so a stub that
    does the same is a faithful test double."""

    def _make_stub(self, tmp: str, script: str) -> str:
        stub = Path(tmp) / "fake_findsym"
        stub.write_text(script)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        return str(stub)

    def test_detects_new_version(self):
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(
                tmp, "#!/bin/sh\ncat > /dev/null\necho 'FINDSYM, Version 7.1.6, Oct 2023'\n"
            )
            self.assertEqual(detect_findsym_version(binary=stub), (7, 1, 6))

    def test_detects_legacy_version(self):
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(
                tmp, "#!/bin/sh\ncat > /dev/null\necho 'findsym, Version 2.5, February 2001'\n"
            )
            self.assertEqual(detect_findsym_version(binary=stub), (2, 5))

    def test_unparseable_output_raises(self):
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(tmp, "#!/bin/sh\ncat > /dev/null\n")  # no output at all
            with self.assertRaises(FindsymLogError):
                detect_findsym_version(binary=stub)

    def test_missing_binary_raises(self):
        with self.assertRaises(FileNotFoundError):
            detect_findsym_version(binary="/no/such/findsym/binary")

    def test_probe_does_not_leave_a_stray_findsym_log(self):
        # Regression guard: the probe call runs findsym with no cwd isolation
        # bug once left a findsym.log in the caller's own working directory.
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(
                tmp,
                "#!/bin/sh\ncat > /dev/null\necho 'FINDSYM, Version 7.1.6, Oct 2023'\necho stray > findsym.log\n",
            )
            with TemporaryDirectory() as cwd_dir:
                original_cwd = Path.cwd()
                import os

                os.chdir(cwd_dir)
                try:
                    detect_findsym_version(binary=stub)
                finally:
                    os.chdir(original_cwd)
                self.assertFalse((Path(cwd_dir) / "findsym.log").exists())


class TestDetectFindsymVersionCaching(unittest.TestCase):
    """The probe is cached per binary. A findsym binary's version cannot
    change while a command runs, but `detect_findsym_version` is called on
    EVERY direction-basis findsym escalation -- so each escalation cost two
    subprocess invocations (the probe, then the real run) when the probe's
    answer was already known. Measured before the fix: a stub-binary CLI
    test doing 4 escalations invoked findsym 8 times."""

    def _counting_stub(self, tmp: str, version: str, counter: Path, name="fake_findsym") -> str:
        stub = Path(tmp) / name
        stub.write_text(
            f"#!/bin/sh\ncat > /dev/null\necho x >> {counter}\n"
            f"echo 'FINDSYM, Version {version}, Oct 2023'\n"
        )
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        return str(stub)

    def test_repeated_probes_invoke_the_binary_once(self):
        with TemporaryDirectory() as tmp:
            counter = Path(tmp) / "calls"
            counter.write_text("")
            stub = self._counting_stub(tmp, "7.1.6", counter)
            versions = [detect_findsym_version(binary=stub) for _ in range(5)]

            self.assertEqual(versions, [(7, 1, 6)] * 5)
            self.assertEqual(counter.read_text().count("x"), 1)

    def test_a_rewritten_binary_at_the_same_path_is_a_cache_miss(self):
        """The cache key is (path, mtime, size), NOT the bare path. Tests
        routinely point `isotropy.find_binary` at a freshly written stub and
        several write stubs of DIFFERENT versions during one process; a
        path-only key would hand the second one the first one's version --
        silently selecting the wrong input format tier for every later
        call."""
        with TemporaryDirectory() as tmp:
            counter = Path(tmp) / "calls"
            counter.write_text("")

            self.assertEqual(
                detect_findsym_version(binary=self._counting_stub(tmp, "7.1.6", counter)),
                (7, 1, 6),
            )
            time.sleep(0.01)  # ensure a distinct mtime_ns
            self.assertEqual(
                detect_findsym_version(binary=self._counting_stub(tmp, "2.5", counter)),
                (2, 5),
            )
            self.assertEqual(counter.read_text().count("x"), 2)

    def test_distinct_binaries_are_cached_separately(self):
        with TemporaryDirectory() as tmp:
            counter = Path(tmp) / "calls"
            counter.write_text("")
            new = self._counting_stub(tmp, "7.1.6", counter, name="new_findsym")
            old = self._counting_stub(tmp, "2.5", counter, name="old_findsym")

            self.assertEqual(detect_findsym_version(binary=new), (7, 1, 6))
            self.assertEqual(detect_findsym_version(binary=old), (2, 5))
            self.assertEqual(detect_findsym_version(binary=new), (7, 1, 6))
            self.assertEqual(detect_findsym_version(binary=old), (2, 5))
            self.assertEqual(counter.read_text().count("x"), 2)

    def test_an_unstattable_binary_still_probes_rather_than_caching(self):
        """A path that cannot be stat'd must fall through to the real probe
        and its real error, never be cached as a success."""
        with self.assertRaises((FileNotFoundError, OSError, IndexError)):
            detect_findsym_version(binary="/nonexistent/definitely/not/findsym")


class TestBuildFindsymInputKeyword(unittest.TestCase):
    """6.x+ keyword format (version[0] >= 6) -- see findsym.py's module
    docstring for exactly how this was determined and validated against the
    real 7.1.6 binary."""

    def test_sto_keyword_format(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        text = build_findsym_input(structure, title="STO test", tolerance=0.0, version=(7, 1, 6))
        lines = text.splitlines()

        self.assertEqual(lines[0], "!useKeyWords")
        self.assertIn("!title", lines)
        self.assertIn("STO test", lines)
        self.assertIn("!atomCount", lines)
        self.assertIn("!atomType", lines)
        # Real chemical symbols straight from the POSCAR, not type indices --
        # !atomType accepts them directly in 6.x+ (unlike the positional
        # format's integer-index requirement).
        atom_type_idx = lines.index("!atomType")
        self.assertEqual(lines[atom_type_idx + 1], "Sr Ti O O O")
        self.assertIn("!atomPosition", lines)
        self.assertIn("!latticeParameters", lines)

    def test_zero_tolerance_omits_tolerance_keywords(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        text = build_findsym_input(structure, title="t", tolerance=0.0, version=(7, 1, 6))
        self.assertNotIn("!latticeTolerance", text)
        self.assertNotIn("!atomicPositionTolerance", text)

    def test_nonzero_tolerance_sets_both_lattice_and_position_keywords(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        text = build_findsym_input(structure, title="t", tolerance=0.01, version=(7, 1, 6))
        lines = text.splitlines()
        self.assertIn("!latticeTolerance", lines)
        self.assertIn("!atomicPositionTolerance", lines)
        lat_idx = lines.index("!latticeTolerance")
        pos_idx = lines.index("!atomicPositionTolerance")
        self.assertEqual(lines[lat_idx + 1], "0.01")
        self.assertEqual(lines[pos_idx + 1], "0.01")

    def test_default_version_is_legacy_positional_format(self):
        # build_findsym_input's default (version unspecified) must stay the
        # pre-6.x positional format -- every existing caller and fixture
        # (and the HPC cluster, the real production environment) depends on
        # this not silently changing.
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        text = build_findsym_input(structure, title="t", tolerance=0.0)
        self.assertFalse(text.startswith("!useKeyWords"))
        self.assertEqual(text.splitlines()[0], "t")

    def test_lattice_parameters_are_correct_for_cubic_structure(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        text = build_findsym_input(structure, title="t", tolerance=0.0, version=(7, 1, 6))
        lines = text.splitlines()
        idx = lines.index("!latticeParameters")
        a, b, c, alpha, beta, gamma = (float(x) for x in lines[idx + 1].split())
        self.assertAlmostEqual(a, 3.901, places=6)
        self.assertAlmostEqual(b, 3.901, places=6)
        self.assertAlmostEqual(c, 3.901, places=6)
        self.assertAlmostEqual(alpha, 90.0, places=6)
        self.assertAlmostEqual(beta, 90.0, places=6)
        self.assertAlmostEqual(gamma, 90.0, places=6)


class TestParseFindsymLogKeyword(unittest.TestCase):
    """Validated against real FINDSYM 7.1.6 output (tests/fixtures/), not
    just the older sample in reference/ -- real 7.1.6 logs add a per-Wyckoff-
    block "RMS/MAX position shift" line the reference sample doesn't have."""

    def test_real_sto_fixture(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        text = (FIXTURES / "findsym_7.1.6_STO.log").read_text()
        info = parse_findsym_log(text, structure, title="STO_findsym")

        self.assertEqual(info.space_group_number, 221)
        self.assertEqual(
            info.wyckoff_positions,
            [WyckoffPosition("Sr", "a"), WyckoffPosition("Ti", "b"), WyckoffPosition("O", "c")],
        )

    def test_real_tio2_fixture_with_free_parameter(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_TiO2")
        text = (FIXTURES / "findsym_7.1.6_TiO2.log").read_text()
        info = parse_findsym_log(text, structure, title="TiO2_findsym")

        self.assertEqual(info.space_group_number, 141)
        self.assertEqual(len(info.wyckoff_positions), 2)
        letters = {w.letter for w in info.wyckoff_positions}
        self.assertEqual(letters, {"b", "e"})
        o_position = next(w for w in info.wyckoff_positions if w.species == "O")
        self.assertIsNotNone(o_position.z)
        self.assertIsNone(o_position.x)
        self.assertIsNone(o_position.y)

    def test_real_cmo_fixture_centered_space_group_multiple_free_orbits(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        text = (FIXTURES / "findsym_7.1.6_CMO.log").read_text()
        info = parse_findsym_log(text, structure, title="CMO_findsym")

        self.assertEqual(info.space_group_number, 139)
        self.assertEqual(len(info.wyckoff_positions), 6)
        species_present = {w.species for w in info.wyckoff_positions}
        self.assertEqual(species_present, {"Ca", "Mn", "O"})
        free_param_orbits = [w for w in info.wyckoff_positions if w.z is not None]
        self.assertEqual(len(free_param_orbits), 4)  # Ca(e), Mn(e), O(e), O(g)

    def test_occupancy_column_and_shift_line_do_not_break_parsing(self):
        # The real fixtures already exercise this, but a focused check: the
        # extra trailing occupancy value on each atom row and the
        # "RMS position shift = ..." line after each block must not be
        # mistaken for a new Wyckoff header or an atom row.
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        text = (FIXTURES / "findsym_7.1.6_STO.log").read_text()
        info = parse_findsym_log(text, structure, title="STO_findsym")
        self.assertEqual(len(info.wyckoff_positions), 3)

    def test_trailing_cif_block_does_not_break_parsing(self):
        # Real 7.1.6 logs end with a "# CIF file created by FINDSYM..." block
        # that has nothing to do with the Wyckoff section -- parsing must
        # stop cleanly at the separator line before it, not choke on CIF text.
        text = (FIXTURES / "findsym_7.1.6_STO.log").read_text()
        self.assertIn("# CIF file created by FINDSYM", text)  # sanity: fixture really has one
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        info = parse_findsym_log(text, structure, title="STO_findsym")
        self.assertEqual(info.space_group_number, 221)

    def test_trailing_cif_block_without_preceding_separator_line(self):
        # Real 7.1.6 logs generated with a nonzero tolerance (and some
        # space-group settings even at tolerance=0 -- HfO2, graphite, SiO2,
        # BaTiO3) omit the dashed separator line before "# CIF file
        # created", going straight from the last Wyckoff block's "RMS
        # position shift" line to the CIF marker. Parsing must stop cleanly
        # there too, not just when a separator is present.
        text = (FIXTURES / "findsym_7.1.6_HfO2.log").read_text()
        self.assertNotIn("------------------------------------------\n# CIF file created", text)
        self.assertIn("# CIF file created by FINDSYM", text)  # sanity: fixture really has one
        structure = Structure.from_poscar(FIXTURES / "POSCAR_HfO2")
        info = parse_findsym_log(text, structure, title="HfO2_findsym")
        self.assertEqual(info.space_group_number, 14)
        self.assertEqual(len(info.wyckoff_positions), 3)

    def test_matches_legacy_format_result_for_same_real_structure(self):
        # STO's pre-6.x fixture is smodes output text, not a raw findsym log
        # (see the developer notes), so this cross-checks against CMO instead, which has
        # both a real 2.5-era findsym log (tests/fixtures/CMO_findsym) and (via this
        # test class's own fixture) a real 7.1.6-era one for the same
        # structure -- same physics, different tool version, must agree.
        structure = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        legacy_info = parse_findsym_log((FIXTURES / "CMO_findsym").read_text(), structure, title="t")
        keyword_info = parse_findsym_log(
            (FIXTURES / "findsym_7.1.6_CMO.log").read_text(), structure, title="t"
        )
        self.assertEqual(legacy_info.space_group_number, keyword_info.space_group_number)
        self.assertEqual(
            {(w.species, w.letter) for w in legacy_info.wyckoff_positions},
            {(w.species, w.letter) for w in keyword_info.wyckoff_positions},
        )


class TestRunFindsymInvocationStyle(unittest.TestCase):
    """run_findsym detects invocation style from the input text itself (does
    it start with !useKeyWords), confirmed empirically against the real
    7.1.6 binary: piping keyword-format input via stdin makes it silently
    fall back to positional parsing and error out, so 6.x+ input must be
    passed as a filename argument instead."""

    def _make_stub(self, tmp: str, script: str) -> str:
        stub = Path(tmp) / "fake_findsym"
        stub.write_text(script)
        stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
        return str(stub)

    def test_keyword_input_passed_as_file_argument_not_stdin(self):
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(
                tmp,
                "#!/bin/sh\n"
                'if [ -n "$1" ] && [ -f "$1" ]; then\n'
                '  echo "got file: $1" > findsym.log\n'
                "else\n"
                '  echo "no file argument" > findsym.log\n'
                "fi\n"
                "cat > /dev/null\n",  # drain any stdin so the process doesn't hang
            )
            log_text = run_findsym("!useKeyWords\n!title\nx\n", binary=stub)
        self.assertIn("got file: findsym_input.in", log_text)

    def test_positional_input_passed_via_stdin_not_file_argument(self):
        with TemporaryDirectory() as tmp:
            stub = self._make_stub(
                tmp,
                "#!/bin/sh\n"
                'if [ -n "$1" ]; then\n'
                '  echo "unexpected file argument: $1" > findsym.log\n'
                "else\n"
                "  cat > findsym.log\n"  # echo stdin back into the log
                "fi\n",
            )
            log_text = run_findsym("plain positional input\n", binary=stub)
        self.assertEqual(log_text, "plain positional input\n")


class TestParseReducedStructure(unittest.TestCase):
    """Real findsym 7.1.6 logs, captured from the actual binary (not
    hand-written) -- an exactly-periodic 2x2x2 BaTiO3 cubic supercell (40
    atoms, no distortion applied at all) and a 2x2x2 supercell of CMO's own
    24-atom CONVENTIONAL cell (192 atoms), each run through real findsym
    and saved as `findsym_7.1.6_BaTiO3_cubic_2x2x2.log` / `findsym_7.1.6_
    CMO_2x2x2.log`."""

    def test_batio3_2x2x2_reduces_to_the_real_5_atom_primitive_cell(self):
        big = Structure.from_poscar(FIXTURES / "POSCAR_BaTiO3_cubic_2x2x2")
        log_text = (FIXTURES / "findsym_7.1.6_BaTiO3_cubic_2x2x2.log").read_text()

        reduced = parse_reduced_structure(log_text, big)
        self.assertIsNotNone(reduced)
        self.assertEqual(reduced.n_atoms, 5)
        self.assertEqual(sorted(reduced.species_counts), [("Ba", 1), ("O", 3), ("Ti", 1)])

        # Same physical crystal as the real, independently-built cubic
        # parent fixture -- same lattice PARAMETERS (a rotated but
        # equally-valid choice of primitive vectors is expected and fine;
        # findsym's own reduction has no reason to match this project's
        # own axis convention) and the identical set of fractional
        # positions.
        parent = Structure.from_poscar(FIXTURES / "POSCAR_BaTiO3_cubic")
        self.assertTrue(
            np.allclose(sorted(np.linalg.norm(reduced.lattice, axis=1)), sorted(np.linalg.norm(parent.lattice, axis=1)))
        )
        reduced_positions = {(sp, tuple(np.round(fc, 4))) for sp, fc in zip(reduced.species, reduced.frac_coords)}
        parent_positions = {(sp, tuple(np.round(fc, 4))) for sp, fc in zip(parent.species, parent.frac_coords)}
        self.assertEqual(reduced_positions, parent_positions)

    def test_cmo_2x2x2_of_conventional_cell_reduces_to_the_true_primitive(self):
        # CMO's own I4/mmm is body-centered -- findsym's real reduction of
        # this 192-atom (2x2x2 of the 24-atom conventional cell) structure
        # goes all the way to the 12-atom TRUE primitive cell (Ca3Mn2O7's
        # own real reduced formula), not just back to the 24-atom
        # conventional cell -- confirmed directly against the real binary,
        # not assumed from the space group's own centering type alone.
        big = Structure.from_poscar(FIXTURES / "POSCAR_CMO_2x2x2")
        log_text = (FIXTURES / "findsym_7.1.6_CMO_2x2x2.log").read_text()

        reduced = parse_reduced_structure(log_text, big)
        self.assertIsNotNone(reduced)
        self.assertEqual(reduced.n_atoms, 12)
        self.assertEqual(sorted(reduced.species_counts), [("Ca", 3), ("Mn", 2), ("O", 7)])

    def test_r_centered_batio3_r3m_reduces_to_the_true_5_atom_rhombohedral_primitive(self):
        # BaTiO3's real R3m phase (space group 160), padded into a 2x2x2,
        # exactly-periodic 40-atom supercell. R3m is R-centered in
        # findsym's own standard hexagonal-axes setting -- `Vectors a,b,c:`
        # there describes the CONVENTIONAL, 3x, R-centered cell (196.9
        # A^3), but the log's plain-text Wyckoff block only ever lists the
        # PRIMITIVE-cell-worth of atom representatives (5 atoms: 1 Ba + 1
        # Ti + 3 O), confirmed independently by the same log's own CIF
        # block (atom-site multiplicities 3, 3, 9 = 15 = 3x5). Naively
        # pairing the conventional lattice with these 5 positions gives an
        # internally-inconsistent, 3x-too-sparse structure -- this is
        # exactly the real bug found and fixed this session.
        big = Structure.from_poscar(FIXTURES / "POSCAR_BaTiO3_R3m_primitive_2x2x2")
        log_text = (FIXTURES / "findsym_7.1.6_BaTiO3_R3m_primitive_2x2x2.log").read_text()

        reduced = parse_reduced_structure(log_text, big)
        self.assertIsNotNone(reduced)
        self.assertEqual(reduced.n_atoms, 5)
        self.assertEqual(sorted(reduced.species_counts), [("Ba", 1), ("O", 3), ("Ti", 1)])

        # The true primitive volume (65.6 A^3, confirmed independently
        # this session via real findsym on this exact fixture reporting
        # the same space group, 160) -- not the 3x-inflated 196.9 A^3 the
        # bug produced.
        parent = Structure.from_poscar(FIXTURES / "POSCAR_BaTiO3_primitive")
        self.assertAlmostEqual(
            abs(np.linalg.det(reduced.lattice)), abs(np.linalg.det(parent.lattice)), places=3
        )

        # The real, load-bearing consequence: the reduced structure's atom
        # positions now actually align against the real parent (translation
        # only -- no rotation search needed), where the pre-fix, 3x-inflated
        # structure could not align at all.
        aligned = align_by_nearest_neighbor(
            parent.species, parent.frac_coords, reduced.species, reduced.frac_coords, parent.lattice, tol=0.3
        )
        self.assertEqual(len(aligned), 5)

    def test_no_reduction_when_vectors_are_already_identity(self):
        # The plain (non-doubled) CMO structure's own real log: ratio > 1
        # (a smaller primitive exists in principle), but findsym's own
        # `Vectors a,b,c:` stays identity here -- the conventional,
        # centered description is what findsym reports as its own
        # "Vectors a,b,c:", not a further reduction -- so this correctly
        # returns None rather than a same-size "reduced" structure.
        structure = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        log_text = (FIXTURES / "findsym_7.1.6_CMO.log").read_text()
        self.assertIsNone(parse_reduced_structure(log_text, structure))

    def test_pre_6x_log_has_no_vectors_line_returns_none(self):
        structure = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        log_text = (FIXTURES / "CMO_findsym").read_text()
        self.assertIsNone(parse_reduced_structure(log_text, structure))


class TestReduceToPrimitiveCell(unittest.TestCase):
    """`reduce_to_primitive_cell` closes the one real gap `parse_reduced_
    structure` alone leaves open: a structure that's ALREADY findsym's own
    standard conventional-cell description (not a padded supercell of
    anything), for a centered space group, where `Vectors a,b,c:` never
    exposes any reduction transform at all -- confirmed for both of this
    project's real fixtures that hit this exact case (BaTiO3's own real
    R3m data and CMO's own real I4/mmm data, both GIVEN DIRECTLY, not
    padded)."""

    def test_batio3_r3m_given_directly_reduces_to_true_primitive(self):
        # POSCAR_BaTiO3 is the real, originally-supplied 15-atom R-centered
        # hexagonal setting -- TestParseReducedStructure's own R-centered
        # test uses a DIFFERENT (padded 2x2x2) fixture where `Vectors
        # a,b,c:` DOES expose a transform; this one does not, and needs
        # the new bucket-2 fallback specifically.
        structure = Structure.from_poscar(FIXTURES / "POSCAR_BaTiO3")
        log_text = (FIXTURES / "findsym_7.1.6_BaTiO3.log").read_text()
        self.assertIsNone(parse_reduced_structure(log_text, structure))

        reduced = reduce_to_primitive_cell(structure, log_text)
        self.assertIsNotNone(reduced)
        self.assertEqual(reduced.n_atoms, 5)
        self.assertEqual(sorted(reduced.species_counts), [("Ba", 1), ("O", 3), ("Ti", 1)])

        primitive = Structure.from_poscar(FIXTURES / "POSCAR_BaTiO3_primitive")
        self.assertAlmostEqual(
            abs(np.linalg.det(reduced.lattice)), abs(np.linalg.det(primitive.lattice)), places=6
        )
        aligned = align_by_nearest_neighbor(
            primitive.species, primitive.frac_coords, reduced.species, reduced.frac_coords,
            primitive.lattice, tol=0.3,
        )
        self.assertEqual(len(aligned), 5)

    def test_cmo_given_directly_reduces_to_true_primitive(self):
        # POSCAR_CMO is the real, originally-supplied 24-atom I-centered
        # conventional cell -- the same fixture TestParseReducedStructure's
        # own test already confirms `parse_reduced_structure` alone
        # returns None for.
        structure = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        log_text = (FIXTURES / "findsym_7.1.6_CMO.log").read_text()

        reduced = reduce_to_primitive_cell(structure, log_text)
        self.assertIsNotNone(reduced)
        self.assertEqual(reduced.n_atoms, 12)
        self.assertEqual(sorted(reduced.species_counts), [("Ca", 3), ("Mn", 2), ("O", 7)])
        self.assertAlmostEqual(
            abs(np.linalg.det(reduced.lattice)), abs(np.linalg.det(structure.lattice)) / 2, places=6
        )

    def test_padded_batio3_still_reduces_via_bucket_one(self):
        # Confirms reduce_to_primitive_cell degrades to exactly
        # parse_reduced_structure's own already-validated behavior when
        # bucket 1 alone already succeeds -- no double-reduction, no
        # different answer.
        big = Structure.from_poscar(FIXTURES / "POSCAR_BaTiO3_R3m_primitive_2x2x2")
        log_text = (FIXTURES / "findsym_7.1.6_BaTiO3_R3m_primitive_2x2x2.log").read_text()
        reduced = reduce_to_primitive_cell(big, log_text)
        self.assertIsNotNone(reduced)
        self.assertEqual(reduced.n_atoms, 5)

    def test_already_primitive_structure_returns_none(self):
        # STO's own Gamma-point structure is already primitive (ratio 1) --
        # nothing to reduce, in either bucket.
        structure = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        log_text = (FIXTURES / "findsym_7.1.6_STO.log").read_text()
        self.assertIsNone(reduce_to_primitive_cell(structure, log_text))


if __name__ == "__main__":
    unittest.main()


class TestIsotropySubprocessTimeout(unittest.TestCase):
    """known issue 6 in the developer notes -- every real ISOTROPY-suite subprocess call
    must carry a wall-clock ceiling.

    `detect_findsym_version` always passed `timeout=30`, but `run_findsym`,
    `run_smodes` and `run_iso` passed none, so a wedged invocation blocked its
    caller forever. Seen in practice: a findsym Docker container at 100% CPU
    TEN HOURS after launch, on input findsym normally answers in seconds.
    """

    def test_timeout_is_minutes_not_seconds(self):
        # Generous on purpose: a big supercell is legitimately slow, and a
        # tight limit would kill correct work rather than a hang.
        self.assertGreaterEqual(isotropy.ISOTROPY_SUBPROCESS_TIMEOUT_SECONDS, 120)

    def test_every_real_subprocess_call_passes_the_timeout(self):
        # Pinned by SOURCE, because the failure mode is an omission: the call
        # works perfectly until the day something hangs, so no ordinary
        # behavioural test would ever notice it missing.
        import inspect
        from smodes_phonons import iso as iso_module
        from smodes_phonons import smodes as smodes_module
        for func in (findsym.run_findsym, smodes_module.run_smodes, iso_module.run_iso):
            source = inspect.getsource(func)
            self.assertIn("subprocess.run", source, func.__name__)
            self.assertIn("timeout=", source, f"{func.__name__} has no subprocess timeout")
            self.assertIn("TimeoutExpired", source,
                          f"{func.__name__} does not convert the timeout into a clear error")

    def test_timeout_raises_a_named_error_not_a_bare_subprocess_one(self):
        # A caller should be able to catch this specifically, and the message
        # has to mention the Docker caveat: on macOS, killing the client does
        # NOT kill the container, so the CPU keeps burning after the hang ends.
        self.assertTrue(issubclass(isotropy.IsotropyTimeoutError, RuntimeError))
        import inspect
        source = inspect.getsource(findsym.run_findsym)
        self.assertIn("docker", source.lower())


class TestReSettledAxesAreReadFromTheStandardCell(unittest.TestCase):
    """Real Pnma CaTiO3/SrSnO3: findsym re-settles the axes, and the lattice
    paired with the Wyckoff positions must come from the RE-SETTLED cell.

    findsym prints two lattice-parameter lines. `Lattice parameters,
    a,b,c,...:` echoes the INPUT; `Values of a,b,c,...:` is the standard
    setting it re-settled into, and sits immediately above `Atomic positions
    ... in terms of a,b,c:` -- the frame every Wyckoff parameter below it is
    expressed in. Reading the echo silently pairs standard-setting Wyckoff
    parameters with the input's axes.

    Identical whenever `Vectors a,b,c:` is the identity, which is every other
    parent fixture here -- so nothing caught this for the project's whole
    life. These two materials are supplied in the Pbnm-like axis order (long
    axis third) and findsym re-settles to standard Pnma (long axis second),
    giving an EXACTLY Pnma structure -- every symmetry check passes -- with
    Ti-O at 1.496 A against the relaxed 1.951 A, hence ~14 eV/A of force on
    an undisplaced reference and imaginary frequencies on every branch.
    """

    CASES = (
        # material, POSCAR, log, input echo (a,b,c), re-settled standard (a,b,c)
        ("CaTiO3", "POSCAR_CaTiO3", "findsym_7.1.6_CaTiO3.log",
         (5.33824, 5.44564, 7.59926), (5.44564, 7.59926, 5.33824)),
        ("SrSnO3", "POSCAR_SrSnO3", "findsym_7.1.6_SrSnO3.log",
         (5.71677, 5.74478, 8.08891), (5.74478, 8.08891, 5.71677)),
    )

    def test_lattice_comes_from_the_re_settled_cell_not_the_input_echo(self):
        for name, poscar, log, echoed, standard in self.CASES:
            with self.subTest(material=name):
                structure = Structure.from_poscar(FIXTURES / poscar)
                info = parse_findsym_log((FIXTURES / log).read_text(), structure, name)
                self.assertEqual(
                    tuple(round(v, 5) for v in info.lattice_parameters[:3]), standard
                )
                self.assertNotEqual(
                    tuple(round(v, 5) for v in info.lattice_parameters[:3]), echoed
                )

    def test_the_smodes_input_carries_the_re_settled_cell(self):
        """The load-bearing consumer: smodes applies the Wyckoff formulas to
        whatever cell it is handed, so this line IS the bug."""
        for name, poscar, log, _echoed, standard in self.CASES:
            with self.subTest(material=name):
                structure = Structure.from_poscar(FIXTURES / poscar)
                info = parse_findsym_log((FIXTURES / log).read_text(), structure, name)
                lattice_line = build_smodes_input(info, "GM").splitlines()[2]
                self.assertEqual(
                    tuple(round(float(v), 5) for v in lattice_line.split()[:3]), standard
                )


class TestReconstructionGuard(unittest.TestCase):
    """The parsed lattice and the parsed positions must describe the same
    structure, measured setting-independently.

    findsym legitimately changes origin, axis order, atom order and idealizes
    positions -- none of which moves an interatomic distance. So comparing the
    sorted minimum-image distance spectrum against the input catches a
    frame mismatch without caring about any of the legitimate differences.
    """

    def test_guard_accepts_every_real_log_whose_axes_were_re_settled(self):
        for poscar, log in (("POSCAR_CaTiO3", "findsym_7.1.6_CaTiO3.log"),
                            ("POSCAR_SrSnO3", "findsym_7.1.6_SrSnO3.log")):
            with self.subTest(log=log):
                structure = Structure.from_poscar(FIXTURES / poscar)
                parse_findsym_log((FIXTURES / log).read_text(), structure, "x")

    def test_guard_rejects_the_input_echo_lattice(self):
        """Pin the guard against the exact defect: force the old behaviour and
        require a raise, so reverting the parser fix cannot pass silently."""
        original = findsym._parse_lattice_parameters

        def read_the_input_echo(lines):
            _, line = findsym._find_value_after(lines, findsym._LATTICE_PARAMS_HEADER)
            return tuple(float(x) for x in line.split())

        findsym._parse_lattice_parameters = read_the_input_echo
        try:
            for poscar, log in (("POSCAR_CaTiO3", "findsym_7.1.6_CaTiO3.log"),
                                ("POSCAR_SrSnO3", "findsym_7.1.6_SrSnO3.log")):
                with self.subTest(log=log):
                    structure = Structure.from_poscar(FIXTURES / poscar)
                    with self.assertRaises(FindsymLogError) as caught:
                        parse_findsym_log((FIXTURES / log).read_text(), structure, "x")
                    self.assertIn("does not match the structure it was given", str(caught.exception))
        finally:
            findsym._parse_lattice_parameters = original

    def test_distance_spectrum_is_invariant_to_setting(self):
        """The property the guard rests on: permuting axes and shifting the
        origin must not move the fingerprint, or the guard would false-positive
        on every legitimate re-settling."""
        structure = Structure.from_poscar(FIXTURES / "POSCAR_CaTiO3")
        lattice = np.asarray(structure.lattice, dtype=float)
        frac = np.asarray(structure.frac_coords, dtype=float)

        reference = findsym._distance_spectrum(lattice, frac)
        permuted = findsym._distance_spectrum(
            lattice[[1, 2, 0]], (frac[:, [1, 2, 0]] + np.array([0.37, 0.11, 0.62])) % 1.0
        )
        self.assertLess(float(np.max(np.abs(reference - permuted))), 1e-9)


class TestParseSymopString(unittest.TestCase):
    """A regression guard for the exact bug hit while building this: the
    per-component tokenizer initially set R[row, axis] instead of
    R[axis, row] (row/column transposed) -- silently well-defined (still a
    genuine, invertible linear map) but wrong, only caught by cross-checking
    generated Wyckoff-orbit multiplicities against FINDSYM's own reported
    values. Moved from the v1.x test_direction.py with the parser."""

    def test_identity(self):
        R, t = findsym._parse_symop_string("x,y,z")
        np.testing.assert_array_equal(R, np.eye(3))
        np.testing.assert_array_equal(t, np.zeros(3))

    def test_cyclic_permutation_row_vector_convention(self):
        # "y,z,x" means new_x=old_y, new_y=old_z, new_z=old_x. Under the
        # row-vector convention (new = old @ R), that requires
        # R[1,0]=R[2,1]=R[0,2]=1, NOT R[0,1]=R[1,2]=R[2,0]=1 (the transposed,
        # wrong version this project's own exploratory script initially had).
        R, t = findsym._parse_symop_string("y,z,x")
        old = np.array([1.0, 0.0, 0.0])  # pure x
        new = old @ R
        np.testing.assert_allclose(new, [0.0, 0.0, 1.0])  # should become pure z
        np.testing.assert_array_equal(t, np.zeros(3))

    def test_translation_with_fraction(self):
        R, t = findsym._parse_symop_string("-x,y+1/2,-z")
        np.testing.assert_array_equal(R, np.diag([-1.0, 1.0, -1.0]))
        np.testing.assert_allclose(t, [0.0, 0.5, 0.0])

    def test_negative_fraction_and_decimal_free_parameter(self):
        R, t = findsym._parse_symop_string("-y+1/4,-x+1/4,-z+3/4")
        expected_R = np.zeros((3, 3))
        expected_R[1, 0] = -1  # "y" term (negative) in output component 0
        expected_R[0, 1] = -1  # "x" term (negative) in output component 1
        expected_R[2, 2] = -1
        np.testing.assert_array_equal(R, expected_R)
        np.testing.assert_allclose(t, [0.25, 0.25, 0.75])

    def test_malformed_string_raises(self):
        with self.assertRaises(findsym.FindsymLogError):
            findsym._parse_symop_string("x,y")  # only 2 components, not 3


class TestParseOriginShift(unittest.TestCase):
    def test_nonzero_shift(self):
        text = "blah\nOrigin at    0.00000   0.75000   0.62500\nblah"
        shift = findsym.parse_origin_shift(text)
        np.testing.assert_allclose(shift, [0.0, 0.75, 0.625])

    def test_missing_line_gives_zero(self):
        shift = findsym.parse_origin_shift("no origin line here at all")
        np.testing.assert_array_equal(shift, np.zeros(3))


class TestParseAxisTransform(unittest.TestCase):
    def test_missing_section_gives_identity(self):
        M = findsym.parse_axis_transform("no vectors section here at all")
        np.testing.assert_array_equal(M, np.eye(3))

    def test_identity_section_gives_identity(self):
        text = "Vectors a,b,c:\n   1.00000   0.00000   0.00000\n   0.00000   1.00000   0.00000\n   0.00000   0.00000   1.00000\n"
        M = findsym.parse_axis_transform(text)
        np.testing.assert_array_equal(M, np.eye(3))

    def test_unimodular_relabeling_is_parsed(self):
        # A genuine same-volume axis relabeling (|det|==1) -- real example
        # from BaTiO3's tiled GM3 log.
        text = "Vectors a,b,c:\n  -0.00000  -1.00000   0.00000\n   1.00000   1.00000   0.00000\n  -0.00000   0.00000   1.00000\n"
        M = findsym.parse_axis_transform(text)
        expected = np.array([[0, -1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(M, expected, atol=1e-5)
        self.assertAlmostEqual(abs(np.linalg.det(M)), 1.0)

    def test_a_reducing_transform_is_now_kept_not_discarded(self):
        # FINDSYM reporting a genuinely SMALLER reduced cell (|det| < 1,
        # because it reduced a tiled supercell back toward the primitive) --
        # real example from CMO's tiled X3- log. This used to fall back to
        # the identity, on the reasoning that the caller's coordinates must
        # not be converted through a smaller cell.
        #
        # That was too blunt: the conjugation `inv(M) @ R @ M` is well
        # defined for any invertible M, and discarding the transform threw
        # away the operations the caller's own cell CAN host.
        #
        # Why discarding it looked correct for so long: for THIS matrix the
        # x and y scales are equal, so any operation mixing x and y commutes
        # with it and conjugation is a no-op -- identity happened to give the
        # right answer for CMO, by arithmetic rather than by luck. It is
        # wrong wherever the mixed axes have different scales, e.g. graphite's
        # L2 (diag(-1, -0.5, 0.5)), which found 1 operation and now finds 8.
        text = "Vectors a,b,c:\n  -0.50000   0.00000   0.00000\n  -0.00000  -0.50000   0.00000\n   0.00000   0.00000   1.00000\n"
        M = findsym.parse_axis_transform(text)
        expected = np.array([[-0.5, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 1.0]])
        np.testing.assert_allclose(M, expected, atol=1e-9)

        # The property that made the old fallback safe for this matrix, now
        # stated explicitly rather than relied on implicitly: an in-plane
        # x<->y operation is unchanged by conjugation through it.
        swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(np.linalg.inv(M) @ swap @ M, swap, atol=1e-9)

    def test_a_degenerate_transform_still_falls_back_to_identity(self):
        text = "Vectors a,b,c:\n  0.0 0.0 0.0\n  0.0 0.0 0.0\n  0.0 0.0 0.0\n"
        np.testing.assert_array_equal(findsym.parse_axis_transform(text), np.eye(3))

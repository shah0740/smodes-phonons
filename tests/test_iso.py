import stat
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from smodes_phonons import iso
from smodes_phonons.iso import (
    IsoOutputError,
    IsoSubgroupMatch,
    build_iso_input,
    parse_iso_subgroup_output,
    run_iso,
)

FIXTURES = Path(__file__).parent / "fixtures"

# Real output captured directly from the actual `iso` binary (Isotropy,
# Version 9.6.1) for parent 221 (Pm-3m), subgroup 99 (P4mm) -- this is the
# exact triple (99/38/160 <-> (a,0,0)/(a,a,0)/(a,a,a)) the developer notes'
# direction-vector work already cites from earlier manual `iso` validation.
_ISO_OUTPUT_99 = """Isotropy, Version 9.6.1, Jan 2022
Harold T. Stokes, Dorian M. Hatch, and Branton J. Campbell
Brigham Young University
Current setting is International (new ed.) with conventional basis vectors.
******Irrep (ML) Subgroup Dir
GM4-       99 P4mm  P1  (a,0,0)
*
"""

_ISO_OUTPUT_38 = """Isotropy, Version 9.6.1, Jan 2022
Harold T. Stokes, Dorian M. Hatch, and Branton J. Campbell
Brigham Young University
Current setting is International (new ed.) with conventional basis vectors.
******Irrep (ML) Subgroup Dir
GM4-       38 Amm2  P2  (a,a,0)
GM5-       38 Amm2  P2  (a,a,0)
*
"""

_ISO_OUTPUT_160 = """Isotropy, Version 9.6.1, Jan 2022
Harold T. Stokes, Dorian M. Hatch, and Branton J. Campbell
Brigham Young University
Current setting is International (new ed.) with conventional basis vectors.
******Irrep (ML) Subgroup Dir
GM4-       160 R3m  P3  (a,a,a)
M5-        160 R3m  P6  (a,a;a,a;a,a)
*
"""

_ISO_OUTPUT_NO_TABLE = "Isotropy, Version 9.6.1, Jan 2022\n*Syntax error: V\n"


class TestBuildIsoInput(unittest.TestCase):
    def test_renders_exact_validated_command_script(self):
        text = build_iso_input(221, 99)
        self.assertEqual(
            text,
            "v par 221\nv sub 99\nsh irrep\nsh sub\nsh dir vec\nd iso\nquit\n",
        )


class TestRunIso(unittest.TestCase):
    def test_missing_binary_raises(self):
        from smodes_phonons import iso as iso_module

        original = iso_module.isotropy.find_binary
        iso_module.isotropy.find_binary = lambda name: None
        try:
            with self.assertRaises(FileNotFoundError):
                run_iso("irrelevant input")
        finally:
            iso_module.isotropy.find_binary = original

    def test_subprocess_wiring_with_stub_binary(self):
        with TemporaryDirectory() as tmp:
            stub = Path(tmp) / "fake_iso"
            stub.write_text("#!/bin/sh\ncat\n")
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
            output = run_iso("hello\n", binary=str(stub))
        self.assertEqual(output, "hello\n")

    def test_nonzero_exit_raises(self):
        with TemporaryDirectory() as tmp:
            stub = Path(tmp) / "fake_iso_fail"
            stub.write_text("#!/bin/sh\necho 'boom' >&2\nexit 1\n")
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
            with self.assertRaises(RuntimeError):
                run_iso("input\n", binary=str(stub))


class TestParseIsoSubgroupOutput(unittest.TestCase):
    """Parses real `iso` output captured directly from the actual binary --
    same three (parent, subgroup) pairs the developer notes' direction-vector
    validation table already cites (99/38/160 <-> (a,0,0)/(a,a,0)/(a,a,a)
    for STO's real GM4-)."""

    def test_single_row_p4mm(self):
        matches = parse_iso_subgroup_output(_ISO_OUTPUT_99)
        self.assertEqual(
            matches,
            [IsoSubgroupMatch(irrep="GM4-", subgroup_number=99, subgroup_symbol="P4mm", direction_index="P1", direction="(a,0,0)")],
        )

    def test_two_rows_amm2(self):
        matches = parse_iso_subgroup_output(_ISO_OUTPUT_38)
        self.assertEqual(len(matches), 2)
        self.assertEqual([m.irrep for m in matches], ["GM4-", "GM5-"])
        self.assertTrue(all(m.direction == "(a,a,0)" for m in matches))
        self.assertTrue(all(m.direction_index == "P2" for m in matches))

    def test_multi_component_direction_r3m(self):
        matches = parse_iso_subgroup_output(_ISO_OUTPUT_160)
        self.assertEqual(len(matches), 2)
        gm4, m5 = matches
        self.assertEqual(gm4.irrep, "GM4-")
        self.assertEqual(gm4.direction, "(a,a,a)")
        self.assertEqual(gm4.direction_index, "P3")
        self.assertEqual(m5.irrep, "M5-")
        # A 6-component order parameter -- kept as raw text, not parsed
        # into a tuple (see IsoSubgroupMatch's own docstring).
        self.assertEqual(m5.direction, "(a,a;a,a;a,a)")
        self.assertEqual(m5.direction_index, "P6")

    def test_the_p_token_is_a_sequential_index_not_a_point_group(self):
        # Direct confirmation of the module docstring's own claim: P1/P2/P3
        # track (a,0,0)/(a,a,0)/(a,a,a) purely by position among GM4-'s own
        # three directions, not any independent crystallographic meaning.
        p1 = parse_iso_subgroup_output(_ISO_OUTPUT_99)[0]
        p2 = parse_iso_subgroup_output(_ISO_OUTPUT_38)[0]
        p3 = parse_iso_subgroup_output(_ISO_OUTPUT_160)[0]
        self.assertEqual((p1.direction_index, p2.direction_index, p3.direction_index), ("P1", "P2", "P3"))

    def test_missing_table_raises(self):
        with self.assertRaises(IsoOutputError):
            parse_iso_subgroup_output(_ISO_OUTPUT_NO_TABLE)


if __name__ == "__main__":
    unittest.main()


class TestIrrepDirectionsQuery(unittest.TestCase):
    """`iso`'s per-irrep stratum table -- the input `freeze --map` is built on.

    Parsed from SAVED TEXT, never a live subprocess, per the architecture
    rule that every parser must be separable from the program producing its
    input. The fixtures are real `iso` 9.6.1 output.
    """

    def test_command_script_asks_by_irrep_not_by_subgroup(self):
        script = iso.build_irrep_directions_input(141, "P1")
        # FULL WORDS deliberately: iso accepts both forms (verified identical
        # output) and the long one is self-documenting for anyone reading the
        # script who does not know iso's command language.
        self.assertIn("value parent 141", script)
        self.assertIn("value irrep p1", script)      # lower-cased: iso's own form
        self.assertIn("show direction vector", script)
        self.assertIn("display isotropy", script)
        # The inverse query's own keyword must NOT appear -- that one asks
        # which irreps reach a given subgroup and prints an extra column.
        self.assertNotIn("value subgroup", script)

    def test_parses_real_tio2_p1_table(self):
        text = (FIXTURES / "iso_directions_141_P1.txt").read_text()
        strata = iso.parse_irrep_directions_output(text)
        self.assertEqual(len(strata), 11)
        primary = [s for s in strata if s.is_primary]
        self.assertEqual(
            [(s.subgroup_number, s.direction_index) for s in primary],
            [(121, "P1"), (122, "P3"), (12, "P4"), (15, "P11")],
        )

    def test_parses_real_cmo_p5_table(self):
        text = (FIXTURES / "iso_directions_139_P5.txt").read_text()
        primary = [s for s in iso.parse_irrep_directions_output(text) if s.is_primary]
        # The six primary strata a physicist supplied by hand, reproduced
        # exactly -- this is the table Issue 19 was mistakenly tested against
        # for TiO2 (space group 141, not 139).
        self.assertEqual(
            [s.subgroup_number for s in primary], [122, 70, 121, 74, 69, 72]
        )

    def test_coupled_strata_are_excluded_from_primary(self):
        text = (FIXTURES / "iso_directions_141_P1.txt").read_text()
        coupled = [s for s in iso.parse_irrep_directions_output(text) if not s.is_primary]
        self.assertTrue(coupled)
        # A C-row holds a LINEAR COMBINATION of two free parameters in one
        # position, which our --direction grammar deliberately cannot express.
        self.assertTrue(any("+" in s.direction for s in coupled))

    def test_direction_converts_straight_into_our_grammar(self):
        text = (FIXTURES / "iso_directions_141_P1.txt").read_text()
        by_index = {s.direction_index: s for s in iso.parse_irrep_directions_output(text)}
        # Issue 10's coefficient grammar was specified against this output,
        # so only the parentheses come off -- 1.414a is already our form.
        self.assertEqual(by_index["P4"].as_direction_spec(), "a,1.414a,a,0")
        self.assertEqual(by_index["P11"].as_direction_spec(), "a,0.707a,0,0.707a")
        self.assertEqual(by_index["P3"].as_direction_spec(), "a,0,0,0")

    def test_n_components_gates_a_label_mismatch(self):
        # iso is queried by space group NUMBER plus irrep LABEL, and labels
        # depend on the Wyckoff setting -- so iso's P1 need not be our P1.
        # The component count is the cheap gate: it must equal our own
        # degeneracy or the two are different irreps entirely.
        for name in ("iso_directions_141_P1.txt", "iso_directions_139_P5.txt"):
            strata = iso.parse_irrep_directions_output((FIXTURES / name).read_text())
            self.assertEqual({s.n_components for s in strata}, {4})

    def test_missing_table_raises_rather_than_returning_nothing(self):
        with self.assertRaises(iso.IsoOutputError):
            iso.parse_irrep_directions_output("Isotropy, Version 9.6.1\nno table here\n")

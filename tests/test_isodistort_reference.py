"""ISODISTORT order-parameter-direction tables as ground truth.

Real ISODISTORT output supplied by the physicist for STO's M5- and M5+
(parent 221 Pm-3m, k = M = (1/2,1/2,0)), used as an authoritative,
externally-computed reference rather than a self-consistency check.

Two things it pins down that nothing else here does:

1. **The OPD notation itself.** ISODISTORT writes an M-point order
   parameter as `(a,b;c,d;e,f)` -- semicolon-separated groups, one per STAR
   ARM, with one entry per WITHIN-ARM component. So the notation directly
   encodes `degeneracy == n_arms * within_arm`, and can be checked against
   what this codebase derives independently from SMODES output.

2. **Which subgroup each direction produces.** Independent, and the reason
   the tables are kept in full: `(a,0;0,0;0,0)` and `(a,a;0,0;0,0)` differ
   only in the SECOND component of the SAME arm, and give 65 Cmmm vs 51
   Pmma. That is external proof the within-arm partner is a genuinely
   distinct physical direction -- which is what showed a rank-3-of-6 basis
   for M5- to be a construction bug rather than a property of the irrep.

Confirmed against real findsym, freezing each direction built from the
orbit+SVD within-arm basis:

    M5-  (a,0;0,0;0,0) -> 65    ISODISTORT P2  -> 65 Cmmm
    M5-  (a,a;0,0;0,0) -> 51    ISODISTORT P1  -> 51 Pmma
    M5+  (a,0;0,0;0,0) -> 67    ISODISTORT P2  -> 67 Cmma
    M5+  (a,a;0,0;0,0) -> 53    ISODISTORT P10 -> 53 Pmna
    M5-  (a,0;a,0;a,0) -> 199   ISODISTORT P11 -> 199 I2_13
    M5-  (a,0;b,0;0,0) -> 74    ISODISTORT C11 -> 74 Imma   (built with a=b)

Those runs are not repeated here -- each is a real findsym subprocess, and
this project keeps subprocess work out of the default suite (see the
`phonons` direction-basis note in the developer notes: an earlier version that ran
findsym unconditionally took the suite from ~20s to ~157s).
"""

import math
import re
import unittest
import zipfile
from pathlib import Path

import numpy as np

from smodes_phonons import parse_smodes_output

FIXTURES = Path(__file__).resolve().parent / "fixtures"

_DIRECTION = re.compile(r"^(\S+)\s+\(([^)]*)\)\s+(\d+)\s+(\S+)")


def parse_isodistort_directions(text: str) -> list:
    """[(tag, opd_string, space_group_number, symbol), ...]."""
    rows = []
    for line in text.splitlines():
        match = _DIRECTION.match(line.strip())
        if match and match.group(1)[0] in "PCS46":
            rows.append((match.group(1), match.group(2), int(match.group(3)), match.group(4)))
    return rows


class TestIsodistortReference(unittest.TestCase):
    CASES = [("ISODISTORT_STO_M5-_directions.txt", "M5-", 26),
             ("ISODISTORT_STO_M5+_directions.txt", "M5+", 14)]

    def test_tables_parse_and_are_complete(self):
        for name, _label, expected_rows in self.CASES:
            with self.subTest(fixture=name):
                rows = parse_isodistort_directions((FIXTURES / name).read_text())
                self.assertEqual(len(rows), expected_rows)
                for tag, opd, number, _symbol in rows:
                    self.assertTrue(1 <= number <= 230, f"{tag}: bad space group {number}")
                    self.assertIn(";", opd, f"{tag}: M-point OPD must be arm-grouped")

    def test_opd_notation_matches_our_arm_and_within_arm_decomposition(self):
        """`(a,b;c,d;e,f)` says 3 arms x 2 within-arm; our SMODES-derived
        degeneracy must agree. This is the check that would have caught a
        wrong arm/within-arm split immediately."""
        out = parse_smodes_output((FIXTURES / "STO_findsym_M").read_text())
        by_label = {irrep.label: irrep for irrep in out.irreps}

        for name, label, _rows in self.CASES:
            with self.subTest(irrep=label):
                rows = parse_isodistort_directions((FIXTURES / name).read_text())
                # The most general direction (all free parameters) shows the
                # full shape; every row must share it.
                shapes = {
                    (len(opd.split(";")), len(opd.split(";")[0].split(",")))
                    for _tag, opd, _n, _s in rows
                }
                self.assertEqual(len(shapes), 1, f"{label}: inconsistent OPD shapes {shapes}")
                n_arms, within_arm = shapes.pop()
                self.assertEqual(n_arms, 3, f"{label}: M point has 3 star arms")
                self.assertEqual(
                    n_arms * within_arm, by_label[label].degeneracy,
                    f"{label}: ISODISTORT says {n_arms} arms x {within_arm} within-arm = "
                    f"{n_arms * within_arm}, SMODES says degeneracy "
                    f"{by_label[label].degeneracy}",
                )

    def test_the_within_arm_partner_changes_the_subgroup(self):
        """Directions differing ONLY in one arm's second component give
        different space groups -- external proof that the within-arm partner
        is a distinct physical direction, not a relabelling of the first."""
        rows = dict(
            (opd.replace(" ", ""), number)
            for _tag, opd, number, _sym in
            parse_isodistort_directions((FIXTURES / "ISODISTORT_STO_M5-_directions.txt").read_text())
        )
        self.assertEqual(rows["a,0;0,0;0,0"], 65)
        self.assertEqual(rows["a,a;0,0;0,0"], 51)
        self.assertNotEqual(rows["a,0;0,0;0,0"], rows["a,a;0,0;0,0"])

        plus = dict(
            (opd.replace(" ", ""), number)
            for _tag, opd, number, _sym in
            parse_isodistort_directions((FIXTURES / "ISODISTORT_STO_M5+_directions.txt").read_text())
        )
        self.assertEqual(plus["a,0;0,0;0,0"], 67)
        self.assertEqual(plus["a,a;0,0;0,0"], 53)


_SUBGROUP_S = re.compile(r"\bs=(\d+)")
_SUBGROUP_SYMBOL = re.compile(r"^\s*\d+\s+([PABCIFR])\S*,\s*basis=", re.MULTILINE)
_CENTERING_MULTIPLICITY = {"P": 1, "A": 2, "B": 2, "C": 2, "I": 2, "R": 3, "F": 4}
_CELL = re.compile(
    r"a=([\d.]+), b=([\d.]+), c=([\d.]+), "
    r"alpha=([\d.]+), beta=([\d.]+), gamma=([\d.]+)"
)
_AMPLITUDE = re.compile(r"^(\S*\[.*\]\S*)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$")
_NORMFACTOR = re.compile(r"^(\S*\[.*\]\S*)\s+normfactor\s*=\s*([\d.]+)\s*$")
_VECTOR_ROW = re.compile(
    r"^(?:\S+)?\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+"
    r"(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$"
)


def _lattice_matrix(a, b, c, alpha, beta, gamma):
    """Rows = Cartesian lattice vectors, standard crystallographic setting."""
    al, be, ga = (math.radians(v) for v in (alpha, beta, gamma))
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    return np.array([
        [a, 0.0, 0.0],
        [b * math.cos(ga), b * math.sin(ga), 0.0],
        [cx, cy, math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))],
    ])


def _isodistort_texts():
    """(name, text) for every 'complete modes details' fixture available."""
    for path in sorted(FIXTURES.glob("ISODISTORT_*_modes_*.txt")):
        yield path.name, path.read_text()
    with zipfile.ZipFile(FIXTURES / "CMO_BTO_decomposition.zip") as archive:
        for name in sorted(archive.namelist()):
            if "complete modes details" in name and name.endswith(".txt"):
                yield name.rsplit("/", 1)[-1], archive.read(name).decode()


class TestIsodistortAmplitudeConventions(unittest.TestCase):
    """ISODISTORT's own documented definitions of `As`, `Ap` and `normfactor`,
    verified against every real modes-details fixture in this project.

    These are not our conventions to choose -- they come from ISODISTORT's
    help text, and `decomposition`'s parent-cell normalisation is built to
    reproduce `Ap`. Pinning them here means a future misreading (this file's
    own history has two) fails a test instead of becoming prose.
    """

    def test_ap_equals_as_over_sqrt_s(self):
        """`Ap = As * sqrt(Vp/Vs) = As / sqrt(s)`, with `s` printed directly
        on ISODISTORT's own `Subgroup details` line."""
        checked = 0
        for name, text in _isodistort_texts():
            match = _SUBGROUP_S.search(text)
            if match is None:  # a Gamma-only file needs no supercell line
                continue
            s = int(match.group(1))
            for line in text.splitlines():
                row = _AMPLITUDE.match(line.strip())
                if row is None:
                    continue
                a_s, a_p = float(row.group(2)), float(row.group(3))
                if abs(a_s) < 1e-9:
                    continue
                self.assertAlmostEqual(
                    a_p, a_s / math.sqrt(s), delta=6e-5,
                    msg=f"{name}: {row.group(1)} As={a_s} Ap={a_p} s={s}",
                )
                checked += 1
        self.assertGreater(checked, 20, "expected many real modes to check")

    def test_normfactor_uses_the_unstrained_supercell(self):
        """`normfactor = 1/sqrt(sum over ALL atoms of ||B' . v||^2)`, with `v`
        the printed mode vector in superlattice FRACTIONAL coordinates and
        `B'` the UNSTRAINED supercell -- the cell printed under 'Undistorted
        superstructure', never the distorted one.

        The sum runs over the PRIMITIVE supercell. When ISODISTORT prints a
        CENTRED conventional cell it lists the centring copies too, and those
        must be divided back out -- confirmed on real M5+ data (67 Cmma, whose
        Eu mode lists 4 atoms but whose printed normfactor counts 2). That is
        independent evidence for the help text's "primitive supercell" wording,
        which a purely primitive fixture set could not have distinguished.
        """
        checked = 0
        for name, text in _isodistort_texts():
            lines = text.splitlines()
            symbol = _SUBGROUP_SYMBOL.search(text)
            centering = (
                _CENTERING_MULTIPLICITY[symbol.group(1)] if symbol else 1
            )
            cell = None
            for index, line in enumerate(lines):
                if line.strip().startswith("Undistorted superstructure"):
                    for probe in lines[index:index + 4]:
                        found = _CELL.search(probe)
                        if found:
                            cell = _lattice_matrix(*(float(v) for v in found.groups()))
                            break
                    break
            if cell is None:
                continue

            # Strain modes are macroscopic -- linear combinations of strain
            # components, not atomic displacement vectors -- so their
            # normfactor is a different quantity and this formula does not
            # apply. Parse only the displacive block.
            try:
                index = next(
                    i for i, line in enumerate(lines)
                    if line.strip().startswith("Displacive mode definitions")
                )
                stop = next(
                    (i for i in range(index + 1, len(lines))
                     if "strain mode definitions" in lines[i].strip().lower()),
                    len(lines),
                )
            except StopIteration:
                continue
            lines = lines[:stop]

            while index < len(lines):
                header = _NORMFACTOR.match(lines[index].strip())
                if header is None:
                    index += 1
                    continue
                expected = float(header.group(2))
                total = 0.0
                index += 1
                while index < len(lines):
                    row = _VECTOR_ROW.match(lines[index].rstrip())
                    if row is None:
                        break
                    vector = np.array([float(row.group(i)) for i in (4, 5, 6)])
                    total += float(np.dot(vector @ cell, vector @ cell))
                    index += 1
                if total <= 0.0:
                    continue
                self.assertAlmostEqual(
                    1.0 / math.sqrt(total / centering), expected, delta=5e-5,
                    msg=f"{name}: {header.group(1)} (centering {centering})",
                )
                checked += 1
        self.assertGreater(checked, 20, "expected many real modes to check")

    def test_as_is_the_norm_over_the_primitive_supercell(self):
        """A mode's `As` is the Frobenius norm of its Cartesian displacements,
        so it must equal `dmax * sqrt(n)` when all `n` moving atoms of the mode
        move equally -- the case for every displayed CMO mode, whose vectors
        are all `+-1` on a single axis."""
        with zipfile.ZipFile(FIXTURES / "CMO_BTO_decomposition.zip") as archive:
            name = next(
                n for n in archive.namelist()
                if "complete modes details" in n and n.endswith("a00a.txt")
            )
            text = archive.read(name).decode()
        rows = {
            row.group(1): (float(row.group(2)), float(row.group(4)))
            for row in (_AMPLITUDE.match(l.strip()) for l in text.splitlines())
            if row is not None
        }
        # Ca2 on 8d in the s=4 supercell: 8 atoms moving equally.
        a_s, dmax = next(v for k, v in rows.items() if "GM1+" in k and "Ca2" in k)
        self.assertAlmostEqual(abs(a_s), dmax * math.sqrt(8), delta=5e-5)


if __name__ == "__main__":
    unittest.main()

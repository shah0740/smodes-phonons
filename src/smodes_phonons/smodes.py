"""Building smodes input, running `smodes`, and parsing its output.

`smodes` takes a compact text description of a symmetry (space group,
lattice parameters, Wyckoff positions with any free coordinates, and a list
of k-vectors) and prints, for every irrep at each requested k-vector, a
symmetry-adapted supercell plus a set of displacement basis vectors.

the developer notes flags the legacy parser (`smodes_disp_calc.sh`) as some of the
most fragile code in the pipeline -- fragile because it locates fields by
fixed line-offset arithmetic on free-format text. The parser below instead
walks the file using named section markers (regex/substring matches), which
is what real `smodes` output turns out to need: optional annotation lines
("Raman active", "These include N translational modes", ...) shift every
downstream line number, and a fixed-offset parser breaks whenever they show
up. This was verified line-by-line against four real smodes output files
(tests/fixtures/STO_findsym_{Gamma,M,R}, tests/fixtures/TiO2_findsym_GM).

`SymmetryInfo` is a structured stand-in for a parsed findsym.log -- enough
to build smodes input. Actually parsing findsym.log text belongs to the
`symmetry` stage, which does not exist yet; this is the boundary between
that stage and this one.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import isotropy

_SPECIES_TOLERANCE = 1e-8
_STAR_SEP = re.compile(r"^\*{5,}\s*$")
_DASH_SEP = re.compile(r"^-{5,}\s*$")
_K_VECTOR_RE = re.compile(r"^k vector\s+(\S+)\s*=\s*\(([^)]*)\)")
_IRREP_RE = re.compile(r"^Irrep\s+(\S+)\s*(.*)$")
_DEGENERACY_RE = re.compile(r"^Degeneracy:\s*(\d+)")
_TOTAL_MODES_RE = re.compile(r"^Total number of modes:\s*(\d+)")
_TRANSLATIONAL_RE = re.compile(r"^These include (\d+) translational modes")


class SmodesOutputError(ValueError):
    """Raised when smodes output cannot be parsed unambiguously."""


# ---------------------------------------------------------------------------
# Input side: a structured symmetry description, and building smodes input
# ---------------------------------------------------------------------------


@dataclass
class WyckoffPosition:
    species: str
    letter: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    """Free-parameter value for each axis, or None if that axis is fixed.

    A fixed axis is rendered in smodes input as a literal '0', not '0.00000'
    -- confirmed against tests/fixtures/TiO2_findsym_smodes_input ("O e 0 0
    -0.41753": x and y fixed, z free at -0.41753). This mirrors
    generator_smodes_input.sh, which hardcodes "0" for any axis whose
    Wyckoff-position text doesn't contain a free-parameter letter (x/y/z),
    regardless of what the actual fixed coordinate is -- true for the
    symmorphic sites seen so far, but not a general guarantee.
    """

    @property
    def has_free_params(self) -> bool:
        return self.x is not None or self.y is not None or self.z is not None


@dataclass
class SymmetryInfo:
    title: str
    space_group_number: int
    lattice_parameters: tuple
    """(a, b, c, alpha, beta, gamma)."""
    wyckoff_positions: list
    """list[WyckoffPosition], in findsym.log order."""


def _format_wyckoff_axis(value: float | None) -> str:
    return "0" if value is None else f"{value:.5f}"


def build_smodes_input(symmetry: SymmetryInfo, k_vector: str) -> str:
    """Render a SymmetryInfo + a single requested k-vector label as smodes stdin text.

    Exactly one k-vector, never a batch -- this is deliberate, not an
    oversight. The legacy README warns that requesting multiple k-vectors in
    one smodes run causes intermittent errors ("sometimes errors occur,
    redo step 2"), and the developer notes' file-naming convention already assumes
    one smodes_input/output pair per k-vector so the k-point can be encoded
    in the filename. Taking a single string here enforces that by
    signature rather than leaving it as a convention a caller can violate.
    """
    lines = [
        symmetry.title,
        str(symmetry.space_group_number),
        " ".join(f"{x:.5f}" for x in symmetry.lattice_parameters),
        str(len(symmetry.wyckoff_positions)),
    ]
    for wyc in symmetry.wyckoff_positions:
        if not wyc.has_free_params:
            lines.append(f"{wyc.species} {wyc.letter}")
        else:
            axes = " ".join(_format_wyckoff_axis(v) for v in (wyc.x, wyc.y, wyc.z))
            lines.append(f"{wyc.species} {wyc.letter} {axes}")
    lines.append("1")
    lines.append(k_vector)
    return "\n".join(lines) + "\n"


def run_smodes(input_text: str, binary: str | None = None) -> str:
    """Run the smodes binary with `input_text` on stdin and return its stdout."""
    binary_path = binary or isotropy.find_binary("smodes")
    if binary_path is None:
        raise FileNotFoundError("smodes binary not found on PATH -- run `smodes-phonons doctor`")
    try:
        result = subprocess.run(
            [binary_path], input=input_text, capture_output=True, text=True, check=False,
            timeout=isotropy.ISOTROPY_SUBPROCESS_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise isotropy.IsotropyTimeoutError(
            f"smodes did not finish within {isotropy.ISOTROPY_SUBPROCESS_TIMEOUT_SECONDS}s and "
            "was killed -- a hang, not slow work. Under the macOS Docker wrapper the container "
            "survives the client; check `docker ps` and `docker kill` it."
        ) from exc
    if result.returncode != 0:
        raise RuntimeError(f"smodes exited with status {result.returncode}: {result.stderr}")
    return result.stdout


# ---------------------------------------------------------------------------
# Output side: parsing smodes stdout
# ---------------------------------------------------------------------------


@dataclass
class SymmetryMode:
    """One symmetry-adapted displacement basis vector for an irrep."""

    species: str
    """The single chemical species that moves in this mode."""
    displacements: np.ndarray
    """(n_atoms, 3) supercell displacement pattern; zero for atoms that don't move."""

    @property
    def normalized(self) -> np.ndarray:
        """Displacement pattern scaled to unit Frobenius norm (the convention displacement amplitude and force projection share)."""
        norm = np.linalg.norm(self.displacements)
        if norm == 0:
            raise SmodesOutputError("mode has zero displacement everywhere")
        return self.displacements / norm


@dataclass
class Irrep:
    label: str
    mulliken: str | None
    k_label: str
    k_vector: tuple
    degeneracy: int
    total_modes: int
    n_translational_modes: int
    raman_active: bool
    ir_active: bool
    superlattice: np.ndarray
    """(3, 3) supercell lattice vectors, Cartesian Angstrom."""
    atom_species: list
    """Chemical symbol per supercell atom."""
    atom_positions: np.ndarray
    """(n_atoms, 3) Cartesian positions of supercell atoms, Angstrom."""
    modes: list
    """list[SymmetryMode]. Length is `total_modes // degeneracy` normally, and
    `total_modes` for a complex pair (see `is_complex_pair`) -- there the
    entries come in consecutive groups of `degeneracy`, one group per distinct
    mode, because SMODES prints partners it cannot expect a caller to
    reconstruct by rotation."""

    @property
    def n_atoms(self) -> int:
        return len(self.atom_species)

    is_complex_pair: bool = False
    """True when SMODES labelled this irrep with a PAIR of symbols, e.g.
    `D1+D2+` or `A2A3` -- ISODISTORT's documented convention for a type-2 or
    type-3 COMPLEX irrep, whose physical (real) form is the direct sum of an
    irrep with its own complex conjugate.

    It changes how much SMODES prints. For an ordinary irrep the degenerate
    partners are reachable from one representative by a real-space rotation,
    so SMODES prints only `total_modes // degeneracy` blocks and leaves the
    rest to be constructed. For a complex pair the two real components are the
    cos and sin parts of one Bloch wave and NO rotation relates them, so
    SMODES prints both -- `total_modes` blocks, `degeneracy` per distinct
    mode, laid out consecutively.

    Confirmed on real HfO2's D point: `Irrep D1+D2+`, degeneracy 2, total 18,
    and 18 printed blocks where an ordinary irrep would print 9. Note this is
    NOT the non-self-inverse condition -- D is `(0,1/2,1/2)` in a primitive
    P2_1/c cell, so `2k` is integral and D IS self-inverse. Irrep reality (the
    Herring criterion) is a separate question from whether the star pairs
    `+k` with `-k`.
    """

    space_group_number: int | None = None
    """The parent's space group, copied from the smodes output's own header so
    a consumer holding only an `Irrep` (the v2.0 ISOTROPY-matrix basis, which
    queries `iso` by space-group number) does not need the findsym log for it."""

    @property
    def n_modes(self) -> int:
        """DISTINCT modes -- `total_modes // degeneracy` in both cases. For a
        complex pair `len(self.modes)` is `degeneracy` times this, since
        SMODES prints every partner rather than one representative."""
        return self.total_modes // self.degeneracy


@dataclass
class SmodesOutput:
    title: str
    space_group_number: int
    space_group_symbol: str
    lattice_parameters: tuple
    wyckoff_positions: list
    """list[WyckoffPosition], echoed back from the input."""
    irreps: list
    """list[Irrep], across every k-vector section in the file."""

    def irrep(self, label: str) -> Irrep:
        matches = [ir for ir in self.irreps if ir.label == label]
        if not matches:
            if not self.irreps:
                # This project only ever requests one k-vector per smodes
                # run (see build_smodes_input's own docstring), so an
                # entirely empty irrep list here means smodes literally
                # printed no irrep text at all for whichever k-point this
                # run asked for -- that k-point itself doesn't exist for
                # this material's space group, a categorically different
                # problem from "wrong label at a real k-point" below, so
                # it gets its own, unambiguous message rather than an
                # empty "(have: )" list.
                m = _IRREP_LABEL_RE.match(label)
                k_label = m.group(1) if m else label
                raise KeyError(
                    f"no such k-point {k_label!r} exists for this material -- smodes printed "
                    f"no irreps at all here, so the requested irrep {label!r} cannot exist "
                    "either."
                )
            available = ", ".join(sorted({ir.label for ir in self.irreps}))
            raise KeyError(
                f"{label!r} does not exist. Available irrep(s) at this k-point: {available}. "
                "(Irrep labels depend on the Wyckoff setting findsym chose for this structure "
                "-- a label from ISODISTORT or a differently-built POSCAR may not appear "
                "here.)"
            )
        if len(matches) > 1:
            raise SmodesOutputError(f"irrep label {label!r} is ambiguous across k-vectors")
        return matches[0]


_IRREP_LABEL_RE = re.compile(r"^([A-Za-z]+)\d+[+-]?$")


def k_vector_label_from_irrep(label: str) -> str:
    """Derive the k-vector label smodes uses (e.g. 'GM', 'X') from an irrep label.

    'GM5+' -> 'GM', 'X2-' -> 'X', 'R4+' -> 'R', 'M3-' -> 'M': strip the
    trailing digit and optional +/- sign. This is what lets `irrep GM5+`
    figure out it needs the Gamma-point smodes output without being told.
    """
    m = _IRREP_LABEL_RE.match(label)
    if not m:
        raise ValueError(f"{label!r} doesn't look like an irrep label (expected e.g. 'GM5+', 'X2-')")
    return m.group(1)


def _single_species(atom_species, displacements, context: str) -> str:
    moving = {
        sp
        for sp, row in zip(atom_species, displacements)
        if np.linalg.norm(row) > _SPECIES_TOLERANCE
    }
    if len(moving) != 1:
        raise SmodesOutputError(
            f"{context}: displacement spans species {sorted(moving)}, expected exactly one -- "
            "SMODES guarantees a single species per mode, so something upstream is broken"
        )
    return moving.pop()


def _parse_wyckoff_axis(token: str) -> float | None:
    """A fixed axis is echoed back as a bare '0'; a free axis has a decimal point
    (e.g. '-0.41753'). See WyckoffPosition for why this distinction matters."""
    return float(token) if "." in token else None


def _parse_wyckoff_line(line: str, line_no: int) -> WyckoffPosition:
    tokens = line.split()
    if len(tokens) == 2:
        return WyckoffPosition(species=tokens[0], letter=tokens[1])
    if len(tokens) == 5:
        x, y, z = (_parse_wyckoff_axis(t) for t in tokens[2:5])
        return WyckoffPosition(species=tokens[0], letter=tokens[1], x=x, y=y, z=z)
    raise SmodesOutputError(f"line {line_no}: malformed Wyckoff-position row {line!r}")


def _parse_irrep(lines: list, i: int, k_label: str | None, k_vector: tuple | None):
    if k_label is None:
        raise SmodesOutputError(f"line {i + 1}: Irrep header before any 'k vector' section")

    m = _IRREP_RE.match(lines[i].rstrip())
    if not m:
        raise SmodesOutputError(f"line {i + 1}: could not parse Irrep header {lines[i]!r}")
    label = m.group(1)
    mulliken = m.group(2).strip() or None
    i += 1

    deg_m = _DEGENERACY_RE.match(lines[i].strip())
    if not deg_m:
        raise SmodesOutputError(f"line {i + 1}: expected 'Degeneracy:', got {lines[i]!r}")
    degeneracy = int(deg_m.group(1))
    i += 1

    tot_m = _TOTAL_MODES_RE.match(lines[i].strip())
    if not tot_m:
        raise SmodesOutputError(f"line {i + 1}: expected 'Total number of modes:', got {lines[i]!r}")
    total_modes = int(tot_m.group(1))
    i += 1

    n_translational = 0
    raman_active = False
    ir_active = False
    while True:
        s = lines[i].strip()
        trans_m = _TRANSLATIONAL_RE.match(s)
        if trans_m:
            n_translational = int(trans_m.group(1))
            i += 1
        elif s == "Raman active":
            raman_active = True
            i += 1
        elif s == "IR active":
            ir_active = True
            i += 1
        else:
            break

    if lines[i].strip() != "Vectors defining superlattice":
        raise SmodesOutputError(
            f"line {i + 1}: expected 'Vectors defining superlattice', got {lines[i]!r}"
        )
    i += 1
    try:
        superlattice = np.array([[float(x) for x in lines[i + k].split()] for k in range(3)])
    except ValueError as exc:
        raise SmodesOutputError(f"line {i + 1}: malformed superlattice vectors") from exc
    i += 3

    if lines[i].strip() != "atom  type   position":
        raise SmodesOutputError(f"line {i + 1}: expected atom/type/position header, got {lines[i]!r}")
    i += 1

    atom_species = []
    atom_positions = []
    while i < len(lines) and lines[i].strip() != "Symmetry modes:":
        tokens = lines[i].split()
        if len(tokens) < 5:
            raise SmodesOutputError(f"line {i + 1}: malformed atom position row {lines[i]!r}")
        atom_species.append(tokens[1])
        atom_positions.append([float(x) for x in tokens[2:5]])
        i += 1
    if i >= len(lines):
        raise SmodesOutputError(f"irrep {label}: reached end of file while reading atom positions")
    atom_positions = np.array(atom_positions)
    n_atoms = len(atom_species)
    i += 1  # past "Symmetry modes:"

    if lines[i].strip().startswith("The following include"):
        i += 1

    if lines[i].strip() != "atom  type   displacement":
        raise SmodesOutputError(
            f"line {i + 1}: expected atom/type/displacement header, got {lines[i]!r}"
        )
    i += 1

    if degeneracy <= 0 or total_modes % degeneracy != 0:
        raise SmodesOutputError(
            f"irrep {label}: total_modes={total_modes} not divisible by degeneracy={degeneracy}"
        )
    # A COMPLEX PAIR prints every partner, not one representative per distinct
    # mode -- so the block count is `total_modes`, not `total_modes //
    # degeneracy`. Reading the smaller number leaves real blocks unconsumed
    # and the irrep then fails to close: real HfO2 D and E reported
    # "expected '****' to close irrep D1+D2+" at the tenth of eighteen blocks.
    is_complex_pair = _COMPLEX_PAIR_LABEL.match(label) is not None
    n_blocks = total_modes if is_complex_pair else total_modes // degeneracy

    modes = []
    for mode_idx in range(n_blocks):
        if not _DASH_SEP.match(lines[i]):
            raise SmodesOutputError(
                f"line {i + 1}: expected '----' mode separator in irrep {label}, got {lines[i]!r}"
            )
        i += 1
        disp = np.zeros((n_atoms, 3))
        while i < len(lines) and not _DASH_SEP.match(lines[i]) and not _STAR_SEP.match(lines[i]):
            tokens = lines[i].split()
            if len(tokens) < 5:
                raise SmodesOutputError(f"line {i + 1}: malformed displacement row {lines[i]!r}")
            atom_idx = int(tokens[0]) - 1
            if not (0 <= atom_idx < n_atoms):
                raise SmodesOutputError(
                    f"line {i + 1}: atom index {tokens[0]} out of range for {n_atoms} supercell atoms"
                )
            disp[atom_idx] = [float(x) for x in tokens[2:5]]
            i += 1
        species = _single_species(atom_species, disp, f"irrep {label} mode {mode_idx}")
        modes.append(SymmetryMode(species=species, displacements=disp))

    if i >= len(lines) or not _STAR_SEP.match(lines[i]):
        raise SmodesOutputError(f"line {i + 1}: expected '****' to close irrep {label}")
    i += 1

    irrep = Irrep(
        label=label,
        mulliken=mulliken,
        k_label=k_label,
        k_vector=k_vector,
        degeneracy=degeneracy,
        total_modes=total_modes,
        is_complex_pair=is_complex_pair,
        n_translational_modes=n_translational,
        raman_active=raman_active,
        ir_active=ir_active,
        superlattice=superlattice,
        atom_species=atom_species,
        atom_positions=atom_positions,
        modes=modes,
    )
    return irrep, i


_COMPLEX_PAIR_LABEL = re.compile(r"^([A-Z]+\d+[+-]?)([A-Z]+\d+[+-]?)$")
"""A label made of TWO irrep symbols, e.g. `D1+D2+`, `D1-D2-`, `A2A3`, `P1P1`
-- ISODISTORT's convention for a complex (type-2/3) irrep. Both halves must
carry letters AND digits, so an ordinary label cannot match: `GM4-` has only
one symbol, and `GM12` cannot split because the trailing piece would have no
letters."""


def parse_smodes_output(text: str) -> SmodesOutput:
    lines = text.splitlines()
    if len(lines) < 9:
        raise SmodesOutputError("output too short to be smodes output")
    if "smodes" not in lines[0].lower():
        raise SmodesOutputError(f"line 1 does not look like an smodes banner: {lines[0]!r}")

    title = lines[3].strip()
    i = 4

    if lines[i].strip() != "Space Group:":
        raise SmodesOutputError(f"line {i + 1}: expected 'Space Group:', got {lines[i]!r}")
    i += 1
    sg_tokens = lines[i].split()
    if len(sg_tokens) < 2:
        raise SmodesOutputError(f"line {i + 1}: could not parse space group line {lines[i]!r}")
    space_group_number = int(sg_tokens[0])
    space_group_symbol = " ".join(sg_tokens[1:])
    i += 1

    if not lines[i].strip().startswith("Lattice Parameters"):
        raise SmodesOutputError(f"line {i + 1}: expected lattice-parameters header, got {lines[i]!r}")
    i += 1
    lat_tokens = lines[i].split()
    if len(lat_tokens) != 6:
        raise SmodesOutputError(f"line {i + 1}: expected 6 lattice parameters, got {lines[i]!r}")
    lattice_parameters = tuple(float(x) for x in lat_tokens)
    i += 1

    if not lines[i].strip().startswith("Wyckoff position"):
        raise SmodesOutputError(f"line {i + 1}: expected Wyckoff-position header, got {lines[i]!r}")
    i += 1

    wyckoff_positions = []
    while i < len(lines) and not _STAR_SEP.match(lines[i]):
        wyckoff_positions.append(_parse_wyckoff_line(lines[i], i + 1))
        i += 1
    if i >= len(lines):
        # Confirmed against the real smodes binary: for a k-point label
        # that doesn't exist for this material's space group, smodes
        # prints its header (title, space group, lattice parameters,
        # Wyckoff positions) and then nothing else at all -- no k-vector
        # line, no '****' separator, no error text, no "End of output".
        # There is no ambiguity about what this specific failure means,
        # so name it directly rather than describing the parser's own
        # internal expectation.
        raise SmodesOutputError(
            "smodes printed its header and then nothing else -- the requested k-point does "
            "not exist for this material's space group"
        )
    i += 1

    irreps = []
    k_label = None
    k_vector = None
    while i < len(lines):
        line = lines[i].strip()
        if line == "End of output":
            break
        if not line:
            i += 1
            continue
        k_m = _K_VECTOR_RE.match(line)
        if k_m:
            k_label = k_m.group(1)
            k_vector = tuple(Fraction(tok.strip()) for tok in k_m.group(2).split(","))
            i += 1
            if not (i < len(lines) and _STAR_SEP.match(lines[i])):
                raise SmodesOutputError(f"line {i + 1}: expected '****' after k-vector header")
            i += 1
            continue
        if line.startswith("Irrep"):
            irrep, i = _parse_irrep(lines, i, k_label, k_vector)
            irreps.append(irrep)
            continue
        raise SmodesOutputError(f"line {i + 1}: unexpected content {lines[i]!r}")

    for irrep in irreps:
        irrep.space_group_number = space_group_number
    return SmodesOutput(
        title=title,
        space_group_number=space_group_number,
        space_group_symbol=space_group_symbol,
        lattice_parameters=lattice_parameters,
        wyckoff_positions=wyckoff_positions,
        irreps=irreps,
    )

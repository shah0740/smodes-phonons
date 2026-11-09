"""Building findsym input, running `findsym`, parsing its log.

Per the architecture rule ("every parser must be separable from the
subprocess that produces its input"), `run_findsym` is a thin subprocess
wrapper and does no interpretation of findsym's output; `parse_findsym_log`
takes text (plus the original Structure -- see its docstring for why) and
never touches a subprocess, so it stays testable from saved logs even on a
machine that can't run findsym at all.

**Two findsym versions are supported, detected automatically, not assumed.**
The HPC cluster (this project's real production environment) and every
`tests/fixtures/` findsym log/input pair originally captured from it use
FINDSYM 2.5 (February 2001), which takes plain
positional stdin input. A Mac dev environment can now also run FINDSYM 7.1.6
(Oct 2023) via a local Docker setup (see `tools/setup-macos-docker.sh`) --
materially different: 7.x requires keyword-format input (`!useKeyWords` then
labeled blocks) passed as a *filename argument*, not stdin (confirmed
empirically: piping keyword text via stdin makes 7.x silently fall back to
positional parsing and error out). The old positional format itself also
changed shape between versions -- 2.5 takes one combined "accuracy of data"
tolerance; 7.x's positional mode expects three separate tolerance fields
(lattice, atomic position, magnetic moment), which is why feeding old-style
input straight to 7.x errors with "Error reading form of lattice
parameters". The output log format changed too: 7.x's is versioned in its
own first line ("FINDSYM, Version 7.1.6..." vs 2.5's lowercase "findsym,
Version 2.5..."), adds a colon after "Space Group", carries element symbols
inline in the echoed position table, adds an occupancy column and a
per-Wyckoff-block "RMS/MAX position shift" line, and appends a trailing CIF
block that didn't exist before.

`detect_findsym_version` probes the installed binary by running it with
empty stdin -- both versions print their version banner as the very first
output line regardless of whether any real input follows, so this needs no
valid input to work. `build_findsym_input` and `parse_findsym_log` both
branch on the detected major version (>=6 uses keyword format, matching
FINDSYM's own "beginning with version 6, keyword input" cutover); `run_findsym`
detects which invocation style an input needs by checking whether it starts
with `!useKeyWords`, so callers never have to track invocation style
separately from input format.

Validated against real 7.1.6 output (not just the samples in `reference/`,
which come from a different sub-version and are missing the RMS/MAX shift
lines seen in real 7.1.6 logs): STO (space group 221), TiO2 (141, confirming
the physicist's anatase fixture fix independently), and CMO (139, including
its centered-space-group free-parameter Wyckoff orbits) all reproduce their
already-known-correct space groups and Wyckoff assignments. `build_smodes_input`
and `parse_smodes_output` needed no changes at all for this -- confirmed by
running real 7.1.6-era `smodes` end to end for all three materials and
getting results identical (STO, TiO2) or physically equivalent up to a
benign per-mode sign/Wyckoff-orbit-ordering convention (CMO) to the saved
2.5-era fixtures.
"""

from __future__ import annotations

import os
import math
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import isotropy
from .smodes import SymmetryInfo, WyckoffPosition
from .structure import Structure


class FindsymLogError(ValueError):
    """Raised when a findsym log cannot be parsed unambiguously."""


_VERSION_RE = re.compile(r"findsym,\s*version\s+([\d.]+)", re.IGNORECASE)


def _parse_version(first_line: str) -> tuple:
    m = _VERSION_RE.search(first_line)
    if not m:
        raise FindsymLogError(f"could not detect a findsym version from {first_line!r}")
    return tuple(int(x) for x in m.group(1).split("."))


_VERSION_CACHE: dict = {}


def _binary_identity(binary_path: str):
    """`(path, mtime_ns, size)` -- the cache key for a probed version.

    NOT the bare path. Tests routinely point `isotropy.find_binary` at a
    freshly written stub, and several write stubs of DIFFERENT versions
    during one process; a path-only key would hand the second one the
    first one's version. Including mtime and size makes a rewritten stub a
    genuine cache miss. One `stat` is negligible against the subprocess it
    avoids. Returns `None` (meaning "do not cache") if the binary cannot be
    stat'd, so an unreadable path falls through to the real probe and its
    real error rather than being cached as a success.
    """
    try:
        info = os.stat(binary_path)
    except OSError:
        return None
    return (binary_path, info.st_mtime_ns, info.st_size)


def detect_findsym_version(binary: str | None = None) -> tuple:
    """Probe the installed findsym binary's version.

    Both FINDSYM 2.5 and 7.x print their version banner as the very first
    line of output before reading any input, so this runs the binary with
    empty stdin and needs no valid input file. Returns a tuple of ints, e.g.
    (7, 1, 6) or (2, 5).

    **Cached per binary.** A findsym binary's version cannot change while a
    command runs, but this is called on EVERY direction-basis findsym
    escalation -- so an escalation cost two subprocess invocations, the
    probe and the real run, and the probe's answer was already known.
    Measured directly: a stub-binary CLI test doing 4 escalations invoked
    findsym 8 times. That matters most exactly where it hurts most, on the
    HPC cluster and under the macOS Docker wrapper, where each call is a
    real process at ~15-20s.
    """
    binary_path = binary or isotropy.find_binary("findsym")
    if binary_path is None:
        raise FileNotFoundError("findsym binary not found on PATH -- run `smodes-phonons doctor`")

    identity = _binary_identity(binary_path)
    if identity is not None and identity in _VERSION_CACHE:
        return _VERSION_CACHE[identity]

    with tempfile.TemporaryDirectory() as tmp:
        # findsym always writes findsym.log into its cwd, even on this empty-
        # input probe -- isolate it in a throwaway directory, same as run_findsym.
        result = subprocess.run(
            [binary_path], input="", capture_output=True, text=True, check=False, cwd=tmp, timeout=30
        )
    first_line = result.stdout.splitlines()[0] if result.stdout else ""
    version = _parse_version(first_line)
    if identity is not None:
        _VERSION_CACHE[identity] = version
    return version


def _species_type_indices(structure: Structure) -> list:
    """Per-atom findsym atom-type index, one distinct integer per species.

    Legacy atom_maker.py writes every atom as type 1, which only works when
    different species happen to occupy geometrically inequivalent sites. Per
    the developer notes ("Species in findsym input"), real species indices are used
    instead -- correct by construction, including for structures where the
    legacy shortcut silently over-symmetrizes (rock salt, zinc blende).

    Only used for the pre-6.x positional format. 6.x's `!atomType` keyword
    accepts chemical symbols directly, so the keyword-format builder writes
    `structure.species` straight through instead.
    """
    counts = structure.species_counts  # also validates same-species contiguity
    index_by_symbol = {symbol: i + 1 for i, (symbol, _) in enumerate(counts)}
    return [index_by_symbol[sp] for sp in structure.species]


def _lattice_parameters(lattice) -> tuple:
    lattice = np.asarray(lattice, dtype=np.float64)
    a, b, c = (float(np.linalg.norm(v)) for v in lattice)

    def angle(u, v) -> float:
        return float(np.degrees(np.arccos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))))

    alpha = angle(lattice[1], lattice[2])
    beta = angle(lattice[0], lattice[2])
    gamma = angle(lattice[0], lattice[1])
    return a, b, c, alpha, beta, gamma


def _build_findsym_input_legacy(structure: Structure, title: str, tolerance: float) -> str:
    """Pre-6.x positional format. Mirrors the field order of legacy/sym_finder.sh's
    known-working invocation: title, tolerance ("accuracy of data"), a fixed "1"
    selecting explicit lattice vectors (as opposed to a,b,c,alpha,beta,gamma
    form), the three lattice vectors, an identity transformation-to-standard-
    setting matrix, atom count, per-atom type indices, then fractional positions.
    """
    lines = [
        title,
        f"{tolerance:g}",
        "1",
    ]
    lines += [" ".join(f"{x:.10f}" for x in row) for row in structure.lattice]
    lines += ["1 0 0", "0 1 0", "0 0 1"]
    lines.append(str(structure.n_atoms))
    lines.append(" ".join(str(t) for t in _species_type_indices(structure)))
    lines += [" ".join(f"{x:.10f}" for x in row) for row in structure.frac_coords]
    return "\n".join(lines) + "\n"


def _build_findsym_input_keyword(structure: Structure, title: str, tolerance: float) -> str:
    """6.x+ keyword format. `!atomType` takes chemical symbols directly (per
    the physicist: "legacy/atom_maker.py and its atom_index mapping are
    obsolete" for this format), so species are written straight from
    `structure.species` -- no type-index indirection needed.

    `tolerance == 0` (this project's "use findsym's own default, exact
    match" convention, matching the legacy/pre-6.x semantics) omits the
    tolerance keywords entirely rather than writing a literal 0, since 6.x's
    !latticeTolerance/!atomicPositionTolerance default to small nonzero
    values (0.00001/0.001) when omitted -- passing a literal 0 is not
    documented to mean the same thing. A nonzero tolerance is applied to
    both the lattice and atomic-position keywords, matching how the single
    pre-6.x tolerance value was used for both.
    """
    a, b, c, alpha, beta, gamma = _lattice_parameters(structure.lattice)
    lines = ["!useKeyWords", "!title", title]
    if tolerance > 0:
        lines += ["!latticeTolerance", f"{tolerance:g}", "!atomicPositionTolerance", f"{tolerance:g}"]
    lines += [
        "!latticeParameters",
        f"{a:.10f} {b:.10f} {c:.10f} {alpha:.6f} {beta:.6f} {gamma:.6f}",
        "!atomCount",
        str(structure.n_atoms),
        "!atomType",
        " ".join(structure.species),
        "!atomPosition",
    ]
    lines += [" ".join(f"{x:.10f}" for x in row) for row in structure.frac_coords]
    return "\n".join(lines) + "\n"


def build_findsym_input(structure: Structure, title: str, tolerance: float = 0.0, version: tuple = (2, 5)) -> str:
    """Render a Structure as findsym input text, in whichever format `version`
    needs (see the module docstring). Defaults to the pre-6.x positional
    format, matching every existing caller and fixture; pass the real
    detected version (see `detect_findsym_version`) to get keyword-format
    input for a 6.x+ binary.

    Two deliberate differences from legacy, both formats: `structure.lattice`
    already has the POSCAR scale factor applied (legacy read raw, unscaled
    lines 3-5), and species are never collapsed to a blanket "1".
    """
    if version[0] >= 6:
        return _build_findsym_input_keyword(structure, title, tolerance)
    return _build_findsym_input_legacy(structure, title, tolerance)


def run_findsym(input_text: str, binary: str | None = None, isodata: str | None = None) -> str:
    """Run findsym on `input_text` and return the findsym.log it wrote.

    findsym always writes findsym.log into its current working directory, so
    this always runs it in a fresh temp directory -- concurrent runs can
    never clobber each other -- and returns the log's contents instead of
    leaving the file behind.

    Invocation style is detected from `input_text` itself (whether it starts
    with `!useKeyWords`), not from a separate parameter: 6.x+ keyword input
    must be passed as a filename argument (confirmed empirically -- piping
    it via stdin makes findsym silently fall back to positional parsing and
    error out), while pre-6.x positional input is piped via stdin as before.
    """
    binary_path = binary or isotropy.find_binary("findsym")
    if binary_path is None:
        raise FileNotFoundError("findsym binary not found on PATH -- run `smodes-phonons doctor`")

    env = os.environ.copy()
    if isodata:
        env["ISODATA"] = isodata

    with tempfile.TemporaryDirectory() as tmp:
        uses_keywords = input_text.lstrip().startswith("!useKeyWords")
        if uses_keywords:
            input_filename = "findsym_input.in"
            (Path(tmp) / input_filename).write_text(input_text)
            args = [binary_path, input_filename]
            run_kwargs = {}
        else:
            args = [binary_path]
            run_kwargs = {"input": input_text}

        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=False,
                cwd=tmp,
                env=env,
                timeout=isotropy.ISOTROPY_SUBPROCESS_TIMEOUT_SECONDS,
                **run_kwargs,
            )
        except subprocess.TimeoutExpired as exc:
            raise isotropy.IsotropyTimeoutError(
                f"findsym did not finish within {isotropy.ISOTROPY_SUBPROCESS_TIMEOUT_SECONDS}s and was "
                "killed. findsym normally answers in seconds, so this is a hang rather than slow "
                "work. If you are on the macOS Docker wrapper, killing the client does NOT stop "
                "the container -- check `docker ps` and `docker kill` it, or it will keep burning "
                "CPU."
            ) from exc
        if result.returncode != 0:
            raise RuntimeError(f"findsym exited with status {result.returncode}: {result.stderr}")

        log_path = Path(tmp) / "findsym.log"
        if not log_path.is_file():
            raise RuntimeError(
                "findsym exited successfully but did not write findsym.log -- "
                f"stdout was: {result.stdout!r}"
            )
        log_text = log_path.read_text()

        # 6.x+ also writes a separate findsym.cif -- for some space-group
        # settings (e.g. hexagonal axes) its content is *not* also embedded
        # in findsym.log, so it would otherwise be silently lost. Append it
        # when present and not already inline, rather than reading a second
        # file everywhere this module's CIF-derived symmetry operations are
        # used (see parse_cif_symmetry_operations).
        cif_path = Path(tmp) / "findsym.cif"
        if cif_path.is_file() and "# CIF file created" not in log_text:
            log_text = log_text.rstrip("\n") + "\n" + cif_path.read_text()

        return log_text


# ---------------------------------------------------------------------------
# Parsing findsym.log
# ---------------------------------------------------------------------------

_NUM_ATOMS_HEADER = "Number of atoms in unit cell:"
_LATTICE_PARAMS_HEADER = "Lattice parameters, a,b,c,alpha,beta,gamma:"
_STANDARD_LATTICE_PARAMS_HEADER = "Values of a,b,c,alpha,beta,gamma:"
_WYCKOFF_AXIS_RE = re.compile(r"([xyz])\s*=\s*(-?\d+\.\d+)")

_ATOMIC_POSITIONS_HEADER_LEGACY = "Atomic positions in terms of a,b,c:"
_SPACE_GROUP_RE_LEGACY = re.compile(r"^Space Group\s+(\d+)")
_WYCKOFF_HEADER_RE_LEGACY = re.compile(r"^Wyckoff position ([^,\s]+)(?:,\s*(.*))?$")

_ATOMIC_POSITIONS_HEADER_KEYWORD = "Atomic positions and occupancies in terms of a,b,c:"
_SPACE_GROUP_RE_KEYWORD = re.compile(r"^Space Group:\s+(\d+)")
_WYCKOFF_HEADER_RE_KEYWORD = re.compile(r"^Wyckoff position (\S+)\s*\([^)]*\)(?:,\s*(.*))?$")
_RMS_SHIFT_PREFIX = "RMS position shift"
_MAX_SHIFT_RE = re.compile(r"MAX position shift\s*=\s*(-?[\d.]+)")

# Floor for the reconstruction guard below. Comfortably above any position
# shift findsym makes while idealizing at the tolerances this project uses
# (0 to ~0.01 A), and far below the 0.45 A error the guard exists to catch.
_RECONSTRUCTION_TOLERANCE_ANGSTROM = 0.1

# A reported cell of a different VOLUME is a different description of the
# cell (findsym reducing a tiled supercell, say), not a frame mismatch --
# there is no one-to-one distance comparison to make. An axis permutation,
# which is what the guard exists to catch, preserves volume exactly, so
# skipping here costs the guard nothing.
_RECONSTRUCTION_VOLUME_RELATIVE_TOLERANCE = 0.01
_SEPARATOR_LINE_RE = re.compile(r"^-+$")


def _find_value_after(lines: list, header: str) -> tuple:
    """Return (index_of_header, next_line) for the first line matching `header`."""
    for i, line in enumerate(lines):
        if line.strip() == header:
            if i + 1 >= len(lines):
                raise FindsymLogError(f"line {i + 1}: {header!r} has no value line after it")
            return i, lines[i + 1]
    raise FindsymLogError(f"could not find {header!r} in findsym log")



def _parse_lattice_parameters(lines: list) -> tuple:
    """The cell the reported Wyckoff positions are actually expressed in.

    findsym prints TWO lattice-parameter lines and they are NOT the same
    cell. The first, under `Lattice parameters, a,b,c,...:`, is a bare echo
    of the INPUT. The second, under `Values of a,b,c,...:`, is the standard
    setting findsym re-settled into -- and it sits immediately above
    `Atomic positions ... in terms of a,b,c:`, which is exactly the frame
    every Wyckoff parameter below it is given in.

    Reading the echo pairs standard-setting Wyckoff parameters with the
    INPUT's axes. That is harmless while `Vectors a,b,c:` is the identity --
    true of every parent fixture here, which is why no test caught it -- and
    catastrophic when findsym permutes axes. Real Pnma CaTiO3 and SrSnO3 are
    supplied in the Pbnm-like axis order (long axis third); findsym re-settles
    to standard Pnma (long axis second) and smodes was handed the permuted
    parameters against the unpermuted cell. The result is an EXACTLY Pnma
    structure -- so every symmetry check passes -- whose oxygens are in the
    wrong place: Ti-O 1.496 A against the relaxed 1.951 A, O-O 2.263 against
    2.737. Hence ~14 eV/A of force on an undisplaced reference and imaginary
    frequencies on every branch.

    Falls back to the echo when the standard line is absent, so a log
    truncated before the space-group section still parses.
    """
    for header in (_STANDARD_LATTICE_PARAMS_HEADER, _LATTICE_PARAMS_HEADER):
        try:
            _, line = _find_value_after(lines, header)
        except FindsymLogError:
            continue
        tokens = line.split()
        if len(tokens) != 6:
            raise FindsymLogError(f"expected 6 lattice parameters after {header!r}, got {line!r}")
        return tuple(float(x) for x in tokens)
    raise FindsymLogError(f"could not find {_LATTICE_PARAMS_HEADER!r} in findsym log")



def _lattice_from_parameters(params) -> np.ndarray:
    """Cartesian lattice matrix (rows = a, b, c) from a,b,c,alpha,beta,gamma."""
    a, b, c, alpha, beta, gamma = (float(v) for v in params)
    ca, cb, cg = (math.cos(math.radians(v)) for v in (alpha, beta, gamma))
    sg = math.sin(math.radians(gamma))
    if abs(sg) < 1e-12:
        raise FindsymLogError(f"degenerate lattice parameters (gamma = {gamma}) in findsym log")
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz2 = c * c - cx * cx - cy * cy
    return np.array([[a, 0.0, 0.0], [b * cg, b * sg, 0.0], [cx, cy, math.sqrt(max(cz2, 0.0))]])


def _distance_spectrum(lattice, frac) -> np.ndarray:
    """Sorted minimum-image interatomic distances -- a SETTING-INDEPENDENT
    fingerprint of a structure.

    Invariant to origin choice, axis permutation, rotation and atom ordering,
    because every one of those leaves the set of interatomic distances alone.
    That is exactly what makes it the right comparison here: the whole failure
    mode being guarded against produces a structure that is a perfectly valid
    crystal in its own right, just not the one that was handed in.

    Minimum image is taken by an explicit search over the 27 neighbouring
    cells rather than by rounding, which is only exact for a near-orthogonal
    cell and this must stay correct for a skewed one.
    """
    frac = np.asarray(frac, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    shifts = np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)], dtype=float)
    chunks = []
    for i in range(len(frac) - 1):
        d = frac[i + 1:] - frac[i]
        d -= np.round(d)
        cand = (d[:, None, :] + shifts[None, :, :]) @ lattice
        chunks.append(np.linalg.norm(cand, axis=2).min(axis=1))
    if not chunks:
        return np.zeros(0)
    return np.sort(np.concatenate(chunks))


def _validate_reconstruction(lattice_parameters, positions, structure, lines) -> None:
    """Assert the structure findsym reported is the structure it was given.

    findsym re-expresses the input in a standard setting: a different origin,
    often permuted or rotated axes, atoms in a different order, positions
    idealized onto exact Wyckoff parameters. All of that is legitimate and
    none of it changes the physical structure -- so the interatomic distances
    must survive it. If they do not, the parsed cell and the parsed positions
    are describing different things, and every downstream number is built on
    a structure the user never asked for.

    This guards a real, silent, whole-project-lifetime bug: the parser read
    findsym's echo of the INPUT lattice while the Wyckoff positions below it
    are in the re-settled STANDARD cell. Identical whenever `Vectors a,b,c:`
    is the identity -- every fixture here -- and catastrophic when findsym
    permutes axes. Real Pnma CaTiO3/SrSnO3, supplied in the Pbnm-like axis
    order, came back as an EXACTLY Pnma structure (so every symmetry check
    passed) with Ti-O at 1.496 A against the relaxed 1.951 A, producing ~14
    eV/A of force on an undisplaced reference and imaginary frequencies on
    every branch. Nothing errored; the forces were the only symptom.

    Skipped when the atom counts differ, or when the reported cell VOLUME
    differs from the input's -- findsym legitimately reports a REDUCED cell
    for a tiled supercell input (`Ratio of new/old primtive cell volumes`
    != 1), where there is no one-to-one comparison to make. Both skips are
    free for the defect this guards: an axis permutation changes neither the
    atom count nor the volume.
    """
    if len(positions) != structure.n_atoms:
        return

    reported_lattice = _lattice_from_parameters(lattice_parameters)
    reported_volume = abs(float(np.linalg.det(reported_lattice)))
    input_volume = abs(float(np.linalg.det(np.asarray(structure.lattice, dtype=float))))
    if input_volume <= 0.0:
        return
    if abs(reported_volume - input_volume) / input_volume > _RECONSTRUCTION_VOLUME_RELATIVE_TOLERANCE:
        return

    original = _distance_spectrum(structure.lattice, structure.frac_coords)
    reconstructed = _distance_spectrum(reported_lattice, positions)
    if original.shape != reconstructed.shape:
        return
    deviation = float(np.max(np.abs(original - reconstructed))) if original.size else 0.0

    # findsym reports how far it moved each orbit while idealizing; allow
    # generous room above that, so a deliberately loose --tolerance run does
    # not trip the guard.
    shifts = [float(m.group(1)) for line in lines for m in [_MAX_SHIFT_RE.search(line)] if m]
    tolerance = max(_RECONSTRUCTION_TOLERANCE_ANGSTROM, 4.0 * max(shifts, default=0.0))
    if deviation <= tolerance:
        return

    raise FindsymLogError(
        f"findsym's reported structure does not match the structure it was given: "
        f"interatomic distances differ by up to {deviation:.4f} A (tolerance {tolerance:.4f} A). "
        f"The reported cell is a={lattice_parameters[0]:.5f} b={lattice_parameters[1]:.5f} "
        f"c={lattice_parameters[2]:.5f} and the input cell is "
        f"{_format_cell(structure.lattice)}. Interatomic distances are independent of "
        f"origin, axis order and rotation, so a mismatch this large means the lattice and "
        f"the atomic positions parsed out of this log are describing different structures -- "
        f"most likely findsym re-settled the axes (see its 'Vectors a,b,c:' block) and the "
        f"two were read from different frames. Feeding this to smodes would give a structure "
        f"with the right symmetry and the wrong bond lengths."
    )


def _format_cell(lattice) -> str:
    a, b, c = _lattice_parameters(lattice)[:3]
    return f"a={a:.5f} b={b:.5f} c={c:.5f}"


def _find_header_index(lines: list, header: str) -> int:
    for i, line in enumerate(lines):
        if line.strip() == header:
            return i
    raise FindsymLogError(f"could not find {header!r} in findsym log")


def _validate_wyckoff_orbit(letter: str, atom_indices: list, structure: Structure) -> str:
    """Shared species-consistency check for both log formats: every atom in a
    Wyckoff orbit must belong to the same species (a mixed orbit means
    over-symmetrization somewhere upstream, per the developer notes' "Species in
    findsym input"). Returns the orbit's single species.
    """
    species_in_orbit = set()
    for idx in atom_indices:
        if not (1 <= idx <= structure.n_atoms):
            raise FindsymLogError(
                f"Wyckoff orbit {letter!r}: atom index {idx} out of range for "
                f"{structure.n_atoms} atoms in the given structure"
            )
        species_in_orbit.add(structure.species[idx - 1])
    if len(species_in_orbit) != 1:
        raise FindsymLogError(
            f"Wyckoff orbit {letter!r} spans species {sorted(species_in_orbit)}, expected "
            "exactly one -- a mixed orbit means findsym over-symmetrized the structure "
            "(were the atoms given their real species?)"
        )
    return species_in_orbit.pop()


def _parse_wyckoff_block_legacy(lines: list, i: int, structure: Structure) -> tuple:
    """Parse one 'Wyckoff position ...' header + its atom rows (pre-6.x format:
    4 tokens per row, no occupancy, no trailing shift-summary line, blocks
    separated by a blank line). Returns (WyckoffPosition, next_i).
    """
    m = _WYCKOFF_HEADER_RE_LEGACY.match(lines[i].strip())
    if not m:
        raise FindsymLogError(f"line {i + 1}: expected a 'Wyckoff position ...' header, got {lines[i]!r}")
    letter = m.group(1)
    axes = {axis: float(value) for axis, value in _WYCKOFF_AXIS_RE.findall(m.group(2) or "")}
    i += 1

    atom_indices = []
    positions = []
    while i < len(lines) and lines[i].strip():
        tokens = lines[i].split()
        if len(tokens) != 4:
            raise FindsymLogError(f"line {i + 1}: malformed Wyckoff atom row {lines[i]!r}")
        try:
            atom_indices.append(int(tokens[0]))
            positions.append([float(t) for t in tokens[1:4]])
        except ValueError as exc:
            raise FindsymLogError(f"line {i + 1}: malformed Wyckoff atom row {lines[i]!r}") from exc
        i += 1

    if not atom_indices:
        raise FindsymLogError(f"Wyckoff orbit {letter!r} (line {i}) has no atom rows")

    species = _validate_wyckoff_orbit(letter, atom_indices, structure)
    wyckoff_position = WyckoffPosition(species=species, letter=letter, x=axes.get("x"), y=axes.get("y"), z=axes.get("z"))
    return wyckoff_position, positions, i


def _parse_wyckoff_block_keyword(lines: list, i: int, structure: Structure) -> tuple:
    """Parse one 'Wyckoff position <letter> (<Species N>)[, x = ...]' header +
    its atom rows (6.x+ format: 5 tokens per row -- index, x, y, z,
    occupancy -- followed by a 'RMS position shift = ..., MAX position shift
    = ...' line that ends the block, no blank-line separator needed).
    Returns (WyckoffPosition, next_i).
    """
    m = _WYCKOFF_HEADER_RE_KEYWORD.match(lines[i].strip())
    if not m:
        raise FindsymLogError(f"line {i + 1}: expected a 'Wyckoff position ...' header, got {lines[i]!r}")
    letter = m.group(1)
    axes = {axis: float(value) for axis, value in _WYCKOFF_AXIS_RE.findall(m.group(2) or "")}
    i += 1

    atom_indices = []
    positions = []
    while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith(_RMS_SHIFT_PREFIX):
        tokens = lines[i].split()
        if len(tokens) != 5:
            raise FindsymLogError(f"line {i + 1}: malformed Wyckoff atom row {lines[i]!r}")
        try:
            atom_indices.append(int(tokens[0]))
            positions.append([float(t) for t in tokens[1:4]])
        except ValueError as exc:
            raise FindsymLogError(f"line {i + 1}: malformed Wyckoff atom row {lines[i]!r}") from exc
        i += 1

    if not atom_indices:
        raise FindsymLogError(f"Wyckoff orbit {letter!r} (line {i}) has no atom rows")

    # Skip the trailing "RMS position shift = ..., MAX position shift = ..." line.
    if i < len(lines) and lines[i].strip().startswith(_RMS_SHIFT_PREFIX):
        i += 1

    species = _validate_wyckoff_orbit(letter, atom_indices, structure)
    wyckoff_position = WyckoffPosition(species=species, letter=letter, x=axes.get("x"), y=axes.get("y"), z=axes.get("z"))
    return wyckoff_position, positions, i


def _parse_findsym_log_legacy(text: str, lines: list, structure: Structure, title: str) -> SymmetryInfo:
    space_group_number = None
    for line in lines:
        m = _SPACE_GROUP_RE_LEGACY.match(line.strip())
        if m:
            space_group_number = int(m.group(1))
            break
    if space_group_number is None:
        raise FindsymLogError("could not find a 'Space Group <N>' line in findsym log")

    lattice_parameters = _parse_lattice_parameters(lines)

    i = _find_header_index(lines, _ATOMIC_POSITIONS_HEADER_LEGACY) + 1
    wyckoff_positions = []
    reported_positions = []
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        wyckoff_position, block_positions, i = _parse_wyckoff_block_legacy(lines, i, structure)
        wyckoff_positions.append(wyckoff_position)
        reported_positions.extend(block_positions)

    _validate_reconstruction(lattice_parameters, reported_positions, structure, lines)

    return SymmetryInfo(
        title=title,
        space_group_number=space_group_number,
        lattice_parameters=lattice_parameters,
        wyckoff_positions=wyckoff_positions,
    )


def _parse_findsym_log_keyword(text: str, lines: list, structure: Structure, title: str) -> SymmetryInfo:
    space_group_number = None
    for line in lines:
        m = _SPACE_GROUP_RE_KEYWORD.match(line.strip())
        if m:
            space_group_number = int(m.group(1))
            break
    if space_group_number is None:
        raise FindsymLogError("could not find a 'Space Group: <N>' line in findsym log")

    lattice_parameters = _parse_lattice_parameters(lines)

    i = _find_header_index(lines, _ATOMIC_POSITIONS_HEADER_KEYWORD) + 1
    wyckoff_positions = []
    reported_positions = []
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        # End of the Wyckoff section, trailing CIF block follows -- the
        # dashed separator doesn't always precede "# CIF file created"
        # (confirmed absent for several real 7.1.6 logs: HfO2, graphite,
        # SiO2, BaTiO3, and generally any log generated with a nonzero
        # tolerance; present for others, e.g. STO/TiO2/CMO's original
        # tolerance=0 logs), so both must be treated as valid end markers.
        if _SEPARATOR_LINE_RE.match(stripped) or stripped.startswith("# CIF file created"):
            break
        wyckoff_position, block_positions, i = _parse_wyckoff_block_keyword(lines, i, structure)
        wyckoff_positions.append(wyckoff_position)
        reported_positions.extend(block_positions)

    _validate_reconstruction(lattice_parameters, reported_positions, structure, lines)

    return SymmetryInfo(
        title=title,
        space_group_number=space_group_number,
        lattice_parameters=lattice_parameters,
        wyckoff_positions=wyckoff_positions,
    )


def parse_findsym_log(text: str, structure: Structure, title: str) -> SymmetryInfo:
    """Parse a findsym.log into a SymmetryInfo, ready for build_smodes_input.

    Format (pre-6.x positional-log vs. 6.x+ keyword-log) is detected from the
    log's own first line -- see the module docstring for exactly what
    differs and how each was validated.

    `structure` must be the same Structure that `build_findsym_input` was
    given to produce this log. findsym.log tracks atoms only by an integer
    "type" (pre-6.x) or an inline species label used only for display
    (6.x+, deliberately not trusted -- see `_validate_wyckoff_orbit`).
    Species per Wyckoff orbit are instead recovered by looking up each
    orbit's atom indices against `structure.species`, which also lets this
    validate the developer notes' "every Wyckoff orbit must contain only one chemical
    species" rule -- something findsym.log alone can't check.

    `title` is caller-supplied rather than read from the log: the log's own
    echoed title is a fixed placeholder string regardless of material in
    pre-6.x logs (tests/fixtures/CMO_findsym's title line reads "...BSO_1uc"
    despite being Ca3Mn2O7), inherited from legacy/sym_finder.sh hardcoding
    it. The
    real downstream title -- what ends up in smodes_input and gets echoed
    back in smodes' own output -- is the findsym-log filename, e.g.
    "CMO_findsym", which the caller already knows.

    Takes text plus an in-memory Structure, never a subprocess -- satisfies
    the architecture rule that parsers must be separable from the
    subprocess that produces their input.
    """
    lines = text.splitlines()
    if not lines:
        raise FindsymLogError("findsym log is empty")

    version = _parse_version(lines[0])

    _, count_line = _find_value_after(lines, _NUM_ATOMS_HEADER)
    try:
        n_atoms = int(count_line.strip())
    except ValueError as exc:
        raise FindsymLogError(f"expected an atom count after {_NUM_ATOMS_HEADER!r}, got {count_line!r}") from exc
    if n_atoms != structure.n_atoms:
        raise FindsymLogError(
            f"findsym log declares {n_atoms} atoms but the given structure has "
            f"{structure.n_atoms} -- wrong POSCAR/log pair?"
        )

    if version[0] >= 6:
        return _parse_findsym_log_keyword(text, lines, structure, title)
    return _parse_findsym_log_legacy(text, lines, structure, title)


_SPACE_GROUP_SYMBOL_RE = re.compile(r"^Space Group:?\s*(\d+)\s+\S+\s+(\S+)")
_PRIMITIVE_RATIO_RE = re.compile(r"[Rr]atio of new/old prim\S* cell volumes\s*=\s*([\d.]+)")


def parse_space_group_symbol(log_text: str) -> str:
    """The Hermann-Mauguin space-group symbol from a findsym log's own
    'Space Group:' line (e.g. 'Pm-3m', 'P4mm', 'I4/mmm') -- the third
    whitespace-separated token on that line, after the space-group number
    and the point-group symbol (e.g. 'Oh-1', 'C4v-1'). Confirmed identical
    3-token shape across both format tiers (pre-6.x has no colon after
    'Space Group', 6.x+ does) -- one regex covers both. Raises
    FindsymLogError if no such line is found at all.
    """
    for line in log_text.splitlines():
        m = _SPACE_GROUP_SYMBOL_RE.match(line.strip())
        if m:
            return m.group(2)
    raise FindsymLogError("no 'Space Group:' line found in this findsym log")


def parse_primitive_cell_ratio(log_text: str) -> float:
    """The 'Ratio of new/old primitive cell volumes' findsym reports for a
    6.x+ log (matched against findsym's own real typo, 'primtive', not
    corrected here since matching what the tool actually prints matters
    more than matching the dictionary spelling) -- greater than 1 means
    the INPUT structure is itself a redundant supercell of a smaller true
    primitive cell findsym found; exactly 1 means the input already is
    (findsym's own idea of) the primitive cell. Confirmed directly against
    the real binary: an intentionally-doubled, exactly-periodic SrTiO3
    supercell (no real distortion at all) reports 'Space Group: 221
    ... Pm-3m' with this ratio at 2 -- findsym does reliably detect
    arbitrary redundant periodicity, not only conventional-vs-primitive
    reductions for a recognized centering type.

    Returns 1.0 if the line isn't present at all (pre-6.x logs don't have
    it, and it also never appears if findsym's own analysis never reaches
    this point) -- a safe default meaning "no reduction detected", not "no
    reduction exists".
    """
    for line in log_text.splitlines():
        m = _PRIMITIVE_RATIO_RE.search(line)
        if m:
            return float(m.group(1))
    return 1.0


_VECTORS_ABC_HEADER = "Vectors a,b,c:"

# Standard International Tables primitive-cell transforms, keyed by the
# Hermann-Mauguin centering letter (always the first character of a
# findsym-reported space-group symbol) -- `primitive_lattice = transform @
# conventional_lattice` (row-vector convention, matching every other
# transform in this codebase). "P" (primitive, no centering) deliberately
# has no entry -- there is nothing to reduce.
#
# I and R are validated against real fixture data in this project (CMO,
# I4/mmm -- reduces POSCAR_CMO's real 24-atom conventional cell to the
# true 12-atom primitive, matching the independently-validated primitive
# composition exactly; R, BaTiO3 R3m -- reduces the real 15-atom
# hexagonal-setting POSCAR_BaTiO3 to the true 5-atom rhombohedral
# primitive, matching POSCAR_BaTiO3_primitive exactly). F/A/B/C are the
# same well-known, standard International Tables conventions but have no
# real fixture in this project to validate against yet -- `reduce_to_
# primitive_cell`'s own atom-count/self-consistency check (every species
# orbit must divide by the transform's own volume ratio exactly) is what
# keeps an untested or wrong transform from ever being silently trusted:
# a transform that doesn't cleanly pair up every atom is rejected, not
# used, regardless of which centering letter triggered it.
_STANDARD_CENTERING_TRANSFORMS = {
    "I": np.array([[-0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.5, 0.5, -0.5]]),
    "F": np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]),
    "A": np.array([[1.0, 0.0, 0.0], [0.0, 0.5, -0.5], [0.0, 0.5, 0.5]]),
    "B": np.array([[0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [0.5, 0.0, 0.5]]),
    "C": np.array([[0.5, 0.5, 0.0], [-0.5, 0.5, 0.0], [0.0, 0.0, 1.0]]),
    "R": np.array([[2.0, 1.0, 1.0], [-1.0, 1.0, 1.0], [-1.0, -2.0, 1.0]]) / 3.0,
}


def parse_reduced_structure(log_text: str, structure: Structure) -> Structure | None:
    """A genuinely smaller `Structure`, built directly from a 6.x+ findsym
    log's own `Vectors a,b,c:` transform and `Atomic positions and
    occupancies in terms of a,b,c:` Wyckoff blocks -- for the real,
    hit-in-practice case of a `sub` structure that's bigger than it needs
    to be (e.g. a real DFT relaxation run on a needlessly padded
    supercell), letting `decomposition` (or any other command) work
    against the smaller, symmetrically-equivalent cell findsym itself
    already found, instead of failing outright because no candidate
    matches the oversized `stoichiometry_multiplier`.

    **Deliberately reads the transform directly, not `parse_primitive_
    cell_ratio`'s bare number.** `Vectors a,b,c:` is a real 3x3 matrix,
    each row giving one of the SMALLER cell's own lattice vectors as a
    fraction of `structure`'s own (row-vector convention, matching every
    other rotation/transform in this codebase) -- so the smaller lattice
    is just `transform @ structure.lattice`, no separate reconstruction
    from `Values of a,b,c,alpha,beta,gamma:` needed (that line is left
    unparsed here; it's redundant with the transform and would only be
    useful as an extra sanity check). Every atom's own fractional
    position is printed directly in the Wyckoff blocks, already expressed
    in the SMALLER cell's own a,b,c -- no symmetry operations need to be
    applied to expand an orbit; findsym already did that.

    **Returns `None`, not an error, whenever there's genuinely nothing to
    reduce** -- confirmed to matter for a real, common case, not just a
    defensive check: `Vectors a,b,c:` is present (and printed as the
    identity, 3x3) even when `structure` is ALREADY findsym's own
    smallest reported cell (the common case), and separately, for a
    CENTERED space group (e.g. CMO's own I4/mmm), findsym deliberately
    reports the conventional centered description as `Vectors a,b,c:`
    rather than the mathematically smaller primitive cell -- a real
    crystallographic convention choice (confirmed directly: `parse_
    primitive_cell_ratio` can report a ratio > 1 there while `Vectors
    a,b,c:` still comes back identity, meaning nothing further reduces).
    Both cases -- identity transform, or a transform that doesn't
    actually shrink the atom count -- return `None` rather than a
    same-size or larger "reduced" structure. A pre-6.x log (no `Vectors
    a,b,c:` line at all) also returns `None`.

    Validated against a real, exactly-periodic BaTiO3 2x2x2 supercell (40
    atoms, no distortion applied) via the real findsym 7.1.6 binary:
    recovers the original 5-atom cubic primitive cell exactly (`Ratio of
    new/old primtive cell volumes = 8`, `Vectors a,b,c:` a genuine
    quarter-scale transform, not identity) -- confirmed atom-for-atom
    identical to `POSCAR_BaTiO3_cubic`, the fixture this reduction was
    built from directly.
    """
    lines = log_text.splitlines()
    try:
        vectors_i = _find_header_index(lines, _VECTORS_ABC_HEADER)
    except FindsymLogError:
        return None

    transform_rows = []
    for row_line in lines[vectors_i + 1 : vectors_i + 4]:
        tokens = row_line.split()
        if len(tokens) != 3:
            raise FindsymLogError(f"expected 3 values after {_VECTORS_ABC_HEADER!r}, got {row_line!r}")
        transform_rows.append([float(x) for x in tokens])
    transform = np.array(transform_rows, dtype=np.float64)
    if np.allclose(transform, np.eye(3), atol=1e-6):
        return None

    try:
        j = _find_header_index(lines, _ATOMIC_POSITIONS_HEADER_KEYWORD) + 1
    except FindsymLogError:
        return None

    species: list = []
    frac_coords: list = []
    while j < len(lines):
        if not lines[j].strip():
            j += 1
            continue
        m = _WYCKOFF_HEADER_RE_KEYWORD.match(lines[j].strip())
        if not m:
            break
        letter = m.group(1)
        j += 1
        atom_rows = []
        while j < len(lines) and lines[j].strip() and not lines[j].strip().startswith(_RMS_SHIFT_PREFIX):
            tokens = lines[j].split()
            if len(tokens) != 5:
                raise FindsymLogError(f"line {j + 1}: malformed Wyckoff atom row {lines[j]!r}")
            atom_rows.append((int(tokens[0]), [float(tokens[1]), float(tokens[2]), float(tokens[3])]))
            j += 1
        if j < len(lines) and lines[j].strip().startswith(_RMS_SHIFT_PREFIX):
            j += 1
        if not atom_rows:
            raise FindsymLogError(f"Wyckoff orbit {letter!r} (line {j}) has no atom rows")
        sp = _validate_wyckoff_orbit(letter, [idx for idx, _frac in atom_rows], structure)
        for _idx, frac in atom_rows:
            species.append(sp)
            frac_coords.append(frac)

    if not species or len(species) >= structure.n_atoms:
        return None

    new_lattice = transform @ np.asarray(structure.lattice, dtype=np.float64)
    reduced_parameters = _lattice_parameters(new_lattice)
    try:
        _, printed_line = _find_value_after(lines, "Values of a,b,c,alpha,beta,gamma:")
    except FindsymLogError:
        printed_line = None  # unexpected log shape -- skip the cross-check rather than fail outright
    if printed_line is not None:
        printed_parameters = tuple(float(x) for x in printed_line.split())
        if not np.allclose(reduced_parameters, printed_parameters, rtol=1e-3, atol=1e-3):
            raise FindsymLogError(
                f"parsed reduced lattice parameters {reduced_parameters} don't match findsym's own "
                f"printed {printed_parameters} -- the 'Vectors a,b,c:' transform was likely misparsed"
            )

    frac_coords_arr = np.array(frac_coords, dtype=np.float64)
    try:
        space_group_symbol = parse_space_group_symbol(log_text)
    except FindsymLogError:
        space_group_symbol = None
    if space_group_symbol is not None and space_group_symbol[0] in _STANDARD_CENTERING_TRANSFORMS:
        # A centered space group in findsym's standard conventional-cell
        # setting -- `Vectors a,b,c:`/`Values of a,b,c,...` describe the
        # CONVENTIONAL, centered cell, but the plain-text Wyckoff block
        # above only ever prints the PRIMITIVE-cell-worth of atom
        # representatives. Confirmed directly against real BaTiO3 R3m
        # findsym output (a 2x2x2, 40-atom input): 5 atoms (1 Ba + 1 Ti + 3
        # O) printed here, against a `Vectors a,b,c:`-implied 196.9 A^3
        # cell -- exactly 3x the true 65.6 A^3 primitive volume the log's
        # own CIF block confirms via its atom-site multiplicities (3, 3, 9
        # = 15 total, vs. 5 printed here). Pairing `new_lattice`
        # (conventional) directly with these 5 (primitive-only) positions
        # -- what this function did before this fix -- silently built a
        # structure 3x too big for the atoms it actually contains.
        #
        # Fixed by converting `new_lattice` down to the true primitive cell
        # via the standard International Tables transform for this
        # centering type (see `_STANDARD_CENTERING_TRANSFORMS`) -- a fixed,
        # universal convention for findsym's own conventional-cell output,
        # not something derived from this one material. `frac_coords` must
        # be re-expressed in the new (smaller) basis via the transform's
        # own inverse -- reusing the OLD (conventional-basis) fractional
        # values against the NEW (primitive) lattice would silently
        # misplace every atom.
        centering_transform = _STANDARD_CENTERING_TRANSFORMS[space_group_symbol[0]]
        new_lattice = centering_transform @ new_lattice
        frac_coords_arr = (frac_coords_arr @ np.linalg.inv(centering_transform)) % 1.0

    # POSCAR requires same-species atoms contiguous; preserve first-appearance order.
    species_order = list(dict.fromkeys(species))
    rows = sorted(
        zip(species, frac_coords_arr.tolist()), key=lambda row: species_order.index(row[0])
    )
    return Structure(
        comment=f"{structure.comment} (findsym-reduced)",
        lattice=new_lattice,
        species=[r[0] for r in rows],
        frac_coords=np.array([r[1] for r in rows], dtype=np.float64),
    )


def reduce_to_primitive_cell(structure: Structure, log_text: str) -> Structure | None:
    """The smallest cell describing `structure`'s own crystal, covering both
    real cases `parse_reduced_structure` alone leaves unhandled -- always
    tries that function FIRST (unchanged), only reaching the fallback below
    when it returns `None`.

    **Case 1** (`parse_reduced_structure` succeeds): a genuinely REDUNDANT
    input -- e.g. a DFT-relaxed structure computed on a needlessly padded
    supercell -- where findsym's own `Vectors a,b,c:` directly exposes a
    real reduction transform.

    **Case 2, new here**: `structure` IS ALREADY findsym's own standard
    CONVENTIONAL-cell description for a centered space group (e.g. a real,
    user-supplied `POSCAR_BaTiO3` -- the textbook 15-atom R-centered
    hexagonal setting for R3m, not a padded supercell of anything).
    Findsym has no reason to offer a further `Vectors a,b,c:` reduction
    here at all (the conventional cell already IS its own canonical
    reference form for this space group) -- `parse_reduced_structure`
    correctly returns `None` -- yet `parse_primitive_cell_ratio` still
    reports a true primitive cell exists at a smaller size. Confirmed as a
    real, hit-in-practice gap, not a hypothetical one: this is exactly the
    real `POSCAR_BaTiO3` fixture's own situation.

    Case 2 applies the SAME `_STANDARD_CENTERING_TRANSFORMS` table
    `parse_reduced_structure` itself now uses for its own R-centering
    correction, directly to `structure`, keyed by `parse_space_group_
    symbol`'s own first character (the centering letter) -- no dependency
    on findsym exposing anything beyond the bare symbol. Self-validating,
    not merely trusting the table: the resulting atom count must divide
    the original by EXACTLY the transform's own determinant, with every
    duplicate atom found genuinely pairing up (same species, same
    fractional position mod 1, within a real tolerance) -- a transform
    that's wrong for some untested centering type (F/A/B/C, see the
    table's own docstring) would fail this check rather than silently
    produce a wrong structure, so this never trusts the table blindly.

    Returns `None` (never raises) whenever there is genuinely nothing to
    reduce -- `structure` is already primitive (`ratio <= 1`), the
    centering letter has no known transform, or the transform's own
    self-consistency check fails -- so a caller can always safely fall
    back to using `structure` exactly as given.
    """
    reduced = parse_reduced_structure(log_text, structure)
    if reduced is not None:
        return reduced

    ratio = parse_primitive_cell_ratio(log_text)
    if ratio <= 1.0 + 1e-6:
        return None
    try:
        space_group_symbol = parse_space_group_symbol(log_text)
    except FindsymLogError:
        return None
    if not space_group_symbol or space_group_symbol[0] not in _STANDARD_CENTERING_TRANSFORMS:
        return None
    transform = _STANDARD_CENTERING_TRANSFORMS[space_group_symbol[0]]

    expected_n = structure.n_atoms / round(ratio)
    if abs(expected_n - round(expected_n)) > 1e-6:
        return None
    expected_n = round(expected_n)

    new_lattice = transform @ np.asarray(structure.lattice, dtype=np.float64)
    new_frac = (np.asarray(structure.frac_coords, dtype=np.float64) @ np.linalg.inv(transform)) % 1.0

    species_out: list = []
    frac_out: list = []
    for sp, fc in zip(structure.species, new_frac):
        duplicate = False
        for sp2, fc2 in zip(species_out, frac_out):
            if sp != sp2:
                continue
            diff = fc - fc2
            diff -= np.round(diff)
            if np.max(np.abs(diff)) < 1e-3:
                duplicate = True
                break
        if not duplicate:
            species_out.append(sp)
            frac_out.append(fc)

    if len(species_out) != expected_n:
        return None

    species_order = list(dict.fromkeys(species_out))
    rows = sorted(zip(species_out, frac_out), key=lambda row: species_order.index(row[0]))
    return Structure(
        comment=f"{structure.comment} (centering-reduced)",
        lattice=new_lattice,
        species=[r[0] for r in rows],
        frac_coords=np.array([r[1] for r in rows], dtype=np.float64),
    )


# ---------------------------------------------------------------------------
# findsym's own transformation to the standard setting, and its CIF operations
# (what carries ISOTROPY's operations onto the input structure -- see
# isotropy_basis.standard_frame)
# ---------------------------------------------------------------------------

_CIF_SYMOP_HEADER = "_space_group_symop_operation_xyz"
_ORIGIN_RE = re.compile(r"^Origin at\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)")
_SYMOP_TERM_RE = re.compile(r"([+-]?)([^+-]+)")
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
_AXIS_TRANSFORM_HEADER = "Vectors a,b,c:"


def _parse_symop_string(op_str: str) -> tuple:
    """One CIF symop string like 'y+1/2,x+1/2,-z' as (R, t) -- R (3,3), t
    (3,) -- with new_frac = frac @ R + t (ROW-vector convention: a term "y"
    in output component k means R[index(y), k] = 1; the transpose would
    generate a different, still well-defined, wrong orbit)."""
    parts = [p.strip() for p in op_str.strip().split(",")]
    if len(parts) != 3:
        raise FindsymLogError(f"malformed symmetry-operation string: {op_str!r}")
    R = np.zeros((3, 3))
    t = np.zeros(3)
    for row_i, expr in enumerate(parts):
        for sign, term in _SYMOP_TERM_RE.findall(expr):
            if not term:
                continue
            s = -1.0 if sign == "-" else 1.0
            if term in _AXIS_INDEX:
                R[_AXIS_INDEX[term], row_i] = s
            elif "/" in term:
                num, den = term.split("/")
                t[row_i] += s * float(num) / float(den)
            else:
                t[row_i] += s * float(term)
    return R, t


def parse_cif_symmetry_operations(log_text: str) -> list:
    """The CIF block's `_space_group_symop_operation_xyz` loop as affine
    operations [(R, t), ...], new_frac = frac @ R + t.

    Raises FindsymLogError if the log has no CIF block (a pre-6.x log; 7.x
    always writes one, inline or as the separate `findsym.cif` that
    `run_findsym` appends)."""
    lines = log_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip() == _CIF_SYMOP_HEADER:
            start = i + 1
            break
    if start is None:
        raise FindsymLogError("no CIF symmetry-operation block found in this findsym log (a pre-6.x log?)")
    ops = []
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped:
            break
        tokens = stripped.split(None, 1)
        if len(tokens) != 2 or not tokens[0].isdigit():
            break
        ops.append(_parse_symop_string(tokens[1]))
    return ops


def parse_origin_shift(log_text: str) -> np.ndarray:
    """The 'Origin at x y z' line: findsym's standard origin, in the input
    structure's fractional coordinates. Zeros if absent (pre-6.x)."""
    for line in log_text.splitlines():
        m = _ORIGIN_RE.match(line.strip())
        if m:
            return np.array([float(m.group(i)) for i in (1, 2, 3)])
    return np.zeros(3)


def parse_axis_transform(log_text: str) -> np.ndarray:
    """The 'Vectors a,b,c:' matrix M: findsym's standard axes are
    `M @ input_lattice_rows`. Any non-singular M is returned -- a relabeling
    (|det| = 1, e.g. Pbnm-ordered input re-settled to Pnma), a standard
    centred cell bigger than the input (|det| >= 2), or one smaller than a
    tiled input (|det| < 1). The identity if absent or singular."""
    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == _AXIS_TRANSFORM_HEADER:
            try:
                rows = [[float(x) for x in lines[i + 1 + j].split()] for j in range(3)]
            except (IndexError, ValueError):
                break
            matrix = np.array(rows)
            if matrix.shape == (3, 3) and abs(np.linalg.det(matrix)) > 1e-9:
                return matrix
            break
    return np.eye(3)

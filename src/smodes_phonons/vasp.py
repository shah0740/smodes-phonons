"""Parsing/writing VASP-specific files -- the calculator seam.

Per the developer notes' architecture rule ("nothing above the calculator seam may
know which DFT code ran"), this is the only place in the codebase allowed to
mention OUTCAR, TOTAL-FORCE, KPOINTS, or INCAR. Everything above this module
works with plain (n_atoms, 3) force arrays in eV/A and a DFT-agnostic k-point
mesh description, regardless of which DFT code is actually being driven.
"""

from __future__ import annotations

import gzip
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_TOTAL_FORCE_RE = re.compile(r"TOTAL-FORCE")
_DASH_SEP_RE = re.compile(r"^-+$")


class VaspOutputError(ValueError):
    """Raised when an OUTCAR cannot be parsed unambiguously."""


def _read_text(path) -> str:
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return fh.read()
    return path.read_text()


def parse_outcar_forces(path) -> np.ndarray:
    """Parse the last TOTAL-FORCE block in an OUTCAR (plain or .gz) -> (n_atoms, 3) eV/A.

    Layout (confirmed against real OUTCARs in tests/fixtures/*_Irrep_*/): a header line
    containing "TOTAL-FORCE", one dashed separator line, then one row per atom
    (position xyz, then force xyz -- the last 3 columns), then another dashed
    line, then a "total drift:" line.

    Takes the *last* TOTAL-FORCE block rather than the first. Every real OUTCAR
    seen so far has exactly one (INCAR_fcon uses IBRION=-1, NSW=0 -- a single
    static calculation), so first and last coincide today, but a relaxation
    OUTCAR (IBRION=2, NSW>0) would have one block per ionic step, and the last
    one is the physically meaningful one.
    """
    text = _read_text(path)
    lines = text.splitlines()

    header_indices = [i for i, line in enumerate(lines) if _TOTAL_FORCE_RE.search(line)]
    if not header_indices:
        raise VaspOutputError(f"{path}: no 'TOTAL-FORCE' block found")
    header_i = header_indices[-1]

    start = header_i + 2
    forces = []
    i = start
    while i < len(lines) and lines[i].strip() and not _DASH_SEP_RE.match(lines[i].strip()):
        tokens = lines[i].split()
        if len(tokens) < 6:
            raise VaspOutputError(f"{path}: line {i + 1}: malformed force row {lines[i]!r}")
        try:
            forces.append([float(x) for x in tokens[-3:]])
        except ValueError as exc:
            raise VaspOutputError(f"{path}: line {i + 1}: non-numeric force row {lines[i]!r}") from exc
        i += 1

    if not forces:
        raise VaspOutputError(f"{path}: no atom force rows found after the TOTAL-FORCE header")

    return np.array(forces)


# ---------------------------------------------------------------------------
# Born effective charges / dielectric tensor (LO-TO splitting inputs)
# ---------------------------------------------------------------------------

_BORN_HEADER_RE = re.compile(r"BORN EFFECTIVE CHARGES")
_BORN_ION_RE = re.compile(r"^\s*ion\s+(\d+)\s*$")
_ELECTRONIC_DIELECTRIC_HEADER_RE = re.compile(
    r"MACROSCOPIC STATIC DIELECTRIC TENSOR \(including local field effects in DFT\)"
)
_IONIC_DIELECTRIC_HEADER_RE = re.compile(r"MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION")


def parse_born_effective_charges(path) -> np.ndarray:
    """Parse the last 'BORN EFFECTIVE CHARGES' block in an OUTCAR (plain or
    .gz) from a real VASP LEPSILON=.TRUE. run -> (n_atoms, 3, 3), ordered by
    ion index (same POSCAR atom order LEPSILON was run against).

    Layout: a 'BORN EFFECTIVE CHARGES (in e, ...)' header, then one 'ion N'
    block per atom, each followed by 3 rows (row index, then 3 tensor
    components) -- Z*_kappa,alpha,beta with alpha (field direction) the row
    index and beta (displacement direction) the column, matching VASP's own
    documented convention.

    NOT YET VALIDATED against a real OUTCAR (no such fixture exists in this
    project yet) -- built from VASP's own documented/standard format. Will
    be checked and adjusted, if needed, against the physicist's real
    STO/BaTiO3 LEPSILON output once available; treat this function's exact
    header-text matching as provisional until then.
    """
    text = _read_text(path)
    lines = text.splitlines()

    header_indices = [i for i, line in enumerate(lines) if _BORN_HEADER_RE.search(line)]
    if not header_indices:
        raise VaspOutputError(f"{path}: no 'BORN EFFECTIVE CHARGES' block found")
    header_i = header_indices[-1]

    # Consume "ion N" sub-blocks (each: the "ion N" line, then 3 tensor
    # rows) as long as N increases 1, 2, 3, ... in strict sequence --
    # stops cleanly the moment that sequence breaks, which is exactly
    # where this block ends (no need to separately detect blank lines,
    # a trailing dashed separator, or the start of a later, unrelated
    # OUTCAR section).
    charges = []
    i = header_i + 1
    expected_ion = 1
    while i < len(lines):
        m = _BORN_ION_RE.match(lines[i])
        if m is None:
            i += 1
            continue
        if int(m.group(1)) != expected_ion:
            break
        rows = []
        for row_i in range(3):
            row_line = lines[i + 1 + row_i]
            tokens = row_line.split()
            if len(tokens) != 4:
                raise VaspOutputError(f"{path}: ion {expected_ion} row {row_i + 1}: malformed line {row_line!r}")
            try:
                rows.append([float(x) for x in tokens[1:]])
            except ValueError as exc:
                raise VaspOutputError(
                    f"{path}: ion {expected_ion} row {row_i + 1}: non-numeric entries {row_line!r}"
                ) from exc
        charges.append(np.array(rows))
        i += 4
        expected_ion += 1

    if not charges:
        raise VaspOutputError(f"{path}: no 'ion N' Born-charge blocks found after the header")

    return np.array(charges)


def parse_dielectric_tensor(path) -> np.ndarray:
    """Parse the electronic (clamped-ion) macroscopic static dielectric
    tensor from a real VASP LEPSILON=.TRUE. run -> (3, 3).

    Specifically the block headed 'MACROSCOPIC STATIC DIELECTRIC TENSOR
    (including local field effects in DFT)' -- despite the confusing name,
    "local field effects" here refers to electronic local-field effects
    within the DFT linear response itself, not an ionic/phonon contribution
    -- this is the purely electronic tensor (epsilon_infinity) the
    non-analytic LO-TO correction needs. A run combining LEPSILON with
    IBRION=8 (full DFPT phonons) additionally prints a separate 'IONIC
    CONTRIBUTION' block; using that (or the two summed) instead would
    double-count the phonon/ionic response the Born charges already carry
    in the correction formula -- deliberately not read by this function.
    Raises VaspOutputError (rather than silently reading the wrong block)
    if only an ionic-contribution block is found without an electronic one.

    NOT YET VALIDATED against a real OUTCAR -- see parse_born_effective_
    charges's own docstring; same caveat applies here.
    """
    text = _read_text(path)

    if _ELECTRONIC_DIELECTRIC_HEADER_RE.search(text) is None:
        if _IONIC_DIELECTRIC_HEADER_RE.search(text) is not None:
            raise VaspOutputError(
                f"{path}: found an IONIC CONTRIBUTION dielectric tensor block but no electronic "
                "one -- the non-analytic LO-TO correction needs the purely electronic tensor "
                "(the block headed '...including local field effects in DFT)'), not the ionic "
                "contribution, to avoid double-counting the phonon response the Born charges "
                "already carry."
            )
        raise VaspOutputError(f"{path}: no electronic MACROSCOPIC STATIC DIELECTRIC TENSOR block found")

    lines = text.splitlines()
    header_indices = [i for i, line in enumerate(lines) if _ELECTRONIC_DIELECTRIC_HEADER_RE.search(line)]
    header_i = header_indices[-1]

    rows = []
    i = header_i + 1
    while i < len(lines) and len(rows) < 3:
        stripped = lines[i].strip()
        if stripped and not _DASH_SEP_RE.match(stripped):
            tokens = stripped.split()
            if len(tokens) != 3:
                raise VaspOutputError(f"{path}: malformed dielectric tensor row {lines[i]!r}")
            try:
                rows.append([float(x) for x in tokens])
            except ValueError as exc:
                raise VaspOutputError(f"{path}: non-numeric dielectric tensor row {lines[i]!r}") from exc
        i += 1

    if len(rows) != 3:
        raise VaspOutputError(f"{path}: expected 3 dielectric tensor rows, found {len(rows)}")

    return np.array(rows)


# ---------------------------------------------------------------------------
# KPOINTS
# ---------------------------------------------------------------------------


class KpointsError(ValueError):
    """Raised when a KPOINTS file isn't in the automatic-mesh format this supports."""


@dataclass
class KpointsMesh:
    comment: str
    scheme: str
    """e.g. 'Gamma' or 'Monkhorst-Pack' -- preserved verbatim from the source file."""
    mesh: tuple
    """(n1, n2, n3) mesh subdivisions."""
    shift: tuple | None
    """(s1, s2, s3) mesh shift, or None if the source file omitted it."""

    def write(self, path) -> None:
        lines = [self.comment, "0", self.scheme, " ".join(str(n) for n in self.mesh)]
        if self.shift is not None:
            lines.append(" ".join(f"{x:g}" for x in self.shift))
        Path(path).write_text("\n".join(lines) + "\n")


def parse_kpoints_mesh(text: str) -> KpointsMesh:
    """Parse an automatic-mesh KPOINTS file (line 2 == '0').

    Only this format is supported -- an explicit k-point list, or anything
    else, raises KpointsError rather than guessing at a scaling.
    """
    lines = text.splitlines()
    if len(lines) < 4:
        raise KpointsError("KPOINTS file too short to be an automatic-mesh file")

    comment = lines[0]
    if lines[1].strip() != "0":
        raise KpointsError(
            f"only automatic-mesh KPOINTS files (line 2 == '0') are supported, got {lines[1]!r}"
        )
    scheme = lines[2].strip()

    mesh_tokens = lines[3].split()
    if len(mesh_tokens) != 3:
        raise KpointsError(f"expected 3 mesh subdivisions on line 4, got {lines[3]!r}")
    try:
        mesh = tuple(int(x) for x in mesh_tokens)
    except ValueError as exc:
        raise KpointsError(f"mesh subdivisions must be integers, got {lines[3]!r}") from exc

    shift = None
    if len(lines) > 4 and lines[4].strip():
        shift_tokens = lines[4].split()
        if len(shift_tokens) == 3:
            try:
                shift = tuple(float(x) for x in shift_tokens)
            except ValueError:
                shift = None

    return KpointsMesh(comment=comment, scheme=scheme, mesh=mesh, shift=shift)


def scaled_kpoints_mesh(kmesh: KpointsMesh, original_lattice, new_lattice) -> KpointsMesh:
    """Scale a k-point mesh so real-space sampling density stays roughly
    constant under a change of lattice vectors (e.g. building a supercell).

    new_N_i = max(1, round(orig_N_i * |original_lattice[i]| / |new_lattice[i]|)),
    applied independently per lattice vector by length ratio -- the standard,
    practical convention (not an exact reciprocal-space folding), and it
    degrades gracefully for non-diagonal "weird" cells since it only needs
    vector lengths, not axis alignment.
    """
    original_lengths = np.linalg.norm(np.asarray(original_lattice, dtype=np.float64), axis=1)
    new_lengths = np.linalg.norm(np.asarray(new_lattice, dtype=np.float64), axis=1)
    new_mesh = tuple(
        max(1, round(n * o / new_len))
        for n, o, new_len in zip(kmesh.mesh, original_lengths, new_lengths)
    )
    return KpointsMesh(comment=kmesh.comment, scheme=kmesh.scheme, mesh=new_mesh, shift=kmesh.shift)


_INCAR_TAG_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")

SCF_INCAR_OVERRIDES = {"NSW": "0", "IBRION": "-1", "ISYM": "0"}
"""Forced onto a user-supplied INCAR for a single-point force/energy
evaluation of an already-built, fixed structure -- no ionic or cell
relaxation, matching legacy/INCAR_fcon's own NSW=0/IBRION=-1 (the template
that produced every phonon-mode force calculation this project's own
`phonons` command was validated against). Used for `irrep` (Mode_i/
Undistorted), `freeze-amplitude[-grid]`, and `landau-grid` -- every one of
these either measures forces at a small displacement, or exists
specifically to compare structures at a caller-chosen FIXED amplitude
(freeze-amplitude['s sibling]'s E(Q) sweep only means anything if the
amplitude stays exactly what was asked for, not wherever ionic relaxation
would carry it). ISIF is deliberately left untouched here: VASP ignores it
entirely once IBRION=-1/NSW=0, so there's no real value to force.

`ISYM=0` (confirmed with the physicist): every structure written under
these overrides is deliberately symmetry-broken relative to the parent by
a tiny displacement (0.001 A, per this project's own fixed amplitude
convention) -- exactly the scale VASP's own symmetry detection (its
SYMPREC tolerance) can fail to notice, silently treating the displaced
structure as if it still had the parent's own higher symmetry and
symmetrizing away (part of) the very forces a finite-displacement phonon
calculation depends on measuring accurately. Forcing ISYM=0 disables that
detection entirely, so the small symmetry-breaking displacement is always
respected rather than silently rounded away -- a well-known practical
requirement for finite-displacement/frozen-phonon methods generally, not
specific to this project's own conventions.

`freeze` is the one exception, given no overrides at all -- it's meant as a
relaxation seed (freeze in a candidate order parameter as a symmetry-
breaking kick, then let the ions and cell relax from there to the true
minimum in that lower-symmetry basin), so the user's own INCAR (their real
ISIF/IBRION/NSW/ISYM choice, e.g. ISIF=3/IBRION=2 for a full cell+ion
relax) is written through completely unchanged -- `freeze`'s whole point is
letting VASP's own symmetry machinery find (or not find) a genuinely lower-
symmetry minimum during relaxation, not fixing a single displacement's own
forces in place."""


LEPSILON_INCAR_OVERRIDES = {"LEPSILON": ".TRUE.", "IBRION": "-1"}
"""Forced onto a user-supplied INCAR for the Born-effective-charge/
dielectric-tensor (LO-TO input) calculation -- a genuinely different,
third calculation type from SCF_INCAR_OVERRIDES/freeze's plain pass-
through: a DFPT linear-response run on the plain, undistorted parent
structure (LEPSILON=.TRUE.), not a force/energy evaluation of an
already-built displaced structure. IBRION=-1 alongside it, same reasoning
as SCF_INCAR_OVERRIDES's own -- no ionic motion, just the clamped-ion
electronic response at the fixed equilibrium geometry. Born effective
charges/the dielectric tensor are per-atom, equilibrium-structure
properties (like the elastic tensor), not something computed per phonon
mode -- one BEC calculation serves every Gamma-point irrep of a given
material, never one per irrep."""


def apply_incar_overrides(template_text: str, overrides: dict) -> str:
    """Return template_text with each tag in `overrides` forced to its given
    value, replacing an existing 'TAG = ...' line in place (matched
    case-insensitively, any whitespace/no-space '=' form) or appended at the
    end if the template doesn't already set it. Every other line (comments,
    blank lines, every other tag -- ENCUT, GGA, POTCAR-matched settings,
    magnetic moments, NCORE, ...) is passed through byte-for-byte: this
    project doesn't second-guess the physicist's own material-specific INCAR
    choices, only the handful of tags that mechanically differ by
    calculation type (see SCF_INCAR_OVERRIDES). `overrides={}` is a valid,
    meaningful call (the `freeze` case) -- returns the template unchanged.
    """
    remaining = dict(overrides)
    out = []
    for line in template_text.splitlines():
        m = _INCAR_TAG_RE.match(line)
        matched_key = None
        if m:
            tag_upper = m.group(1).upper()
            for key in remaining:
                if key.upper() == tag_upper:
                    matched_key = key
                    break
        if matched_key is not None:
            out.append(f"{matched_key} = {remaining.pop(matched_key)}")
        else:
            out.append(line)
    for key, value in remaining.items():
        out.append(f"{key} = {value}")
    return "\n".join(out) + "\n"


_MAGMOM_TOKEN_RE = re.compile(r"^(?:(\d+)\*)?(-?[\d.]+(?:[eE][+-]?\d+)?)$")


def parse_magmom(incar_text: str) -> list[float] | None:
    """Extract and expand the MAGMOM tag from `incar_text` into a flat
    per-atom list, in the same atom order as whatever POSCAR this INCAR was
    written against. Returns `None` if no MAGMOM tag is present at all
    (distinct from a malformed one, which raises `VaspOutputError`) --
    `None` is the normal case for a non-magnetic material's INCAR.

    Expands VASP's own `N*value` repeat shorthand (e.g. `6*0  3 -3 3 -3
    14*0`, confirmed against the physicist's own real CMO INCAR_conv --
    AFM Mn ordering flanked by non-magnetic Ca/O) alongside plain
    space-separated values -- both forms are valid VASP syntax and this
    project's own real fixture data uses both in the same line.
    """
    for line in incar_text.splitlines():
        m = _INCAR_TAG_RE.match(line)
        if not m or m.group(1).upper() != "MAGMOM":
            continue
        value_text = line[m.end():].split("#", 1)[0].split("!", 1)[0]
        values: list[float] = []
        for token in value_text.split():
            tm = _MAGMOM_TOKEN_RE.match(token)
            if not tm:
                raise VaspOutputError(f"MAGMOM: could not parse token {token!r} in line {line!r}")
            count = int(tm.group(1)) if tm.group(1) else 1
            values.extend([float(tm.group(2))] * count)
        return values
    return None


def format_magmom(values) -> str:
    """Render a flat per-atom magnetic-moment list back into a VASP MAGMOM
    value string, using the `N*value` repeat shorthand for consecutive
    equal values (matching the physicist's own real INCAR_conv style, and
    keeping a large tiled supercell's own MAGMOM line human-readable rather
    than one value per atom on a single very long line).
    """
    values = list(values)
    if not values:
        raise ValueError("format_magmom: empty value list")
    groups = []
    run_value = values[0]
    run_len = 1
    for v in values[1:]:
        if v == run_value:
            run_len += 1
        else:
            groups.append((run_len, run_value))
            run_value, run_len = v, 1
    groups.append((run_len, run_value))

    def _fmt(v: float) -> str:
        return f"{v:g}"

    return "  ".join(f"{n}*{_fmt(v)}" if n > 1 else _fmt(v) for n, v in groups)

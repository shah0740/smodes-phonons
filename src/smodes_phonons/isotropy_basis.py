"""Order-parameter-direction (OPD) basis built from ISOTROPY's own irrep matrices.

Replaced the v1.x partner SEARCH in v2.0.0 (recorded in
the v1.x development record). SMODES prints one pattern per mode; instead of
searching for the other partners, they are CONSTRUCTED with the projection
(transfer) operator:

    partner_j = (d / |G|) * sum_g  D_jk(g) * (g . seed)

- seed: SMODES's own pattern for the mode, tiled onto the cell;
- g: every operation of the parent modulo the cell, INCLUDING the lattice and
  centring translations inside it -- they carry the arm signs and the Bloch
  phases (quadrature);
- D(g): ISOTROPY's matrix for g in this irrep, from the `iso` program.

The result IS ISOTROPY's basis, in ISOTROPY's component order, so an OPD typed
from ISOTROPY or ISODISTORT means the same thing here. Prototype measured
(2026-11-02, `OPD_DFT_Validation_v1.43.0/PROTOTYPE_ISOTROPY_BASIS.md`): STO
GM4-/R4-/M2-/M5-/X5-, NaCl L3+-, Si L3+-, graphite K5 all equal the old basis
transformed by the measured rule T; the projection also removes the old
basis's tiling noise on rotated SMODES cells (0.2 deg on Si L3-).

FRAME. ISOTROPY and FINDSYM share BYU's settings (including Fd-3m origin
choice 2), so findsym's own reported transformation of the parent POSCAR to
the standard setting (`Origin at` + `Vectors a,b,c:`) carries ISOTROPY's
operations onto our POSCAR exactly -- also for a POSCAR whose axes are
permuted relative to the standard setting (CaTiO3, SrSnO3). Every generator is
then CHECKED to be a symmetry of the parent; nothing is trusted unverified.

OPERATIONS COME FROM ISOTROPY TOO. `iso` prints each operation's rotation and
translation in Cartesian form once lattice parameters are set (`value lattice
param`, `show cartesian`). Conventions, measured on Pm-3m, Fd-3m, Pnma, P2_1/c:
the printed 3x3 is the TRANSPOSE of the matrix acting on column vectors;
translations are in Angstrom; the Cartesian frame is the standard
crystallographic one (a along x, b in the xy plane). No hand-written table of
Kovalev operation names is needed.

ARCHITECTURE. Like every other module here, this one never starts a
subprocess: `iso` output is TEXT supplied by a runner (`set_iso_runner`), which
the CLI provides (with an on-disk cache) and the tests replace with saved
fixtures.
"""

from __future__ import annotations

import hashlib
import itertools
import math
import re
from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np

from .structure import Structure

class IsotropyBasisError(ValueError):
    """The ISOTROPY-matrix construction could not produce a valid basis."""


# ---------------------------------------------------------------------------
# iso runner (subprocess lives in the CLI; tests supply saved text)
# ---------------------------------------------------------------------------

_iso_runner = None
_memory_cache: dict = {}


def set_iso_runner(runner) -> None:
    """`runner(script_text) -> output_text`. The CLI passes `iso.run_iso`
    wrapped in an on-disk cache; tests pass a fixture lookup."""
    global _iso_runner
    _iso_runner = runner
    _memory_cache.clear()


def script_key(script: str) -> str:
    """Stable name for a script's output (on-disk cache and test fixtures)."""
    return hashlib.sha1(script.encode()).hexdigest()[:16]


def _run(script: str) -> str:
    if script in _memory_cache:
        return _memory_cache[script]
    if _iso_runner is None:
        raise IsotropyBasisError("no iso runner configured (set_iso_runner)")
    text = _iso_runner(script)
    _memory_cache[script] = text
    return text


# ---------------------------------------------------------------------------
# iso input builders and parsers (text in, data out)
# ---------------------------------------------------------------------------

_END = "quit\n"


def build_operations_input(space_group: int, lattice_parameters) -> str:
    a, b, c, alpha, beta, gamma = lattice_parameters
    return (
        f"value parent {space_group}\n"
        f"value lattice param {a:.6f} {b:.6f} {c:.6f} {alpha:.6f} {beta:.6f} {gamma:.6f}\n"
        "show generators\nshow cartesian\ndisplay parent\n" + _END
    )


def build_matrices_input(space_group: int, irrep_label: str, elements: list) -> str:
    lines = [f"value parent {space_group}", f"value irrep {irrep_label.lower()}", "show matrix"]
    for element in elements:
        lines += [f"value element {element}", "display irrep"]
    return "\n".join(lines) + "\n" + _END


_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
_ELEMENT = re.compile(r"^\*?\s*\(([^|()]+)\|([^)]*)\)(.*)$")


@dataclass(frozen=True)
class IsoOperation:
    name: str
    translation_label: str
    """ISOTROPY's own fractional translation, e.g. `0,1/4,1/4`."""
    rotation_cartesian: np.ndarray
    """Acting on COLUMN vectors (already transposed from iso's printout)."""
    translation_cartesian: np.ndarray

    @property
    def element(self) -> str:
        """The `value element` string selecting this operation."""
        return f"{self.name} " + " ".join(self.translation_label.split(","))


def parse_cartesian_operations(text: str) -> list:
    ops = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _ELEMENT.match(lines[i].strip().lstrip("*").strip())
        if m:
            rows = [[float(x) for x in _NUMBER.findall(m.group(3))]]
            for j in range(1, 4):
                rows.append([float(x) for x in _NUMBER.findall(lines[i + j])])
            if [len(r) for r in rows] != [3, 3, 3, 3]:
                raise IsotropyBasisError(f"cannot parse iso operation block at line {i + 1}")
            printed = np.array(rows[:3])
            ops.append(IsoOperation(m.group(1).strip(), m.group(2).strip(), printed.T, np.array(rows[3])))
            i += 4
            continue
        i += 1
    if not ops:
        raise IsotropyBasisError("iso printed no operations:\n" + text[-400:])
    return ops


def parse_irrep_matrices(text: str, dimension: int) -> list:
    """Every `display irrep` block, in order: (element label, d x d matrix)."""
    blocks = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _ELEMENT.match(lines[i].strip().lstrip("*").strip())
        if m:
            numbers = [float(x) for x in _NUMBER.findall(m.group(3))]
            j = i + 1
            while len(numbers) < dimension * dimension and j < len(lines):
                numbers += [float(x) for x in _NUMBER.findall(lines[j])]
                j += 1
            if len(numbers) != dimension * dimension:
                raise IsotropyBasisError(
                    f"iso matrix for ({m.group(1)}|{m.group(2)}) has {len(numbers)} entries, "
                    f"expected {dimension}x{dimension} -- does the irrep label correspond?"
                )
            blocks.append((f"({m.group(1)}|{m.group(2)})", _snap(np.array(numbers).reshape(dimension, dimension))))
            i = j
            continue
        if "Error" in lines[i]:
            raise IsotropyBasisError("iso: " + lines[i].strip().lstrip("*"))
        i += 1
    return blocks


_EXACT = (0.0, 0.5, -0.5, math.sqrt(3) / 2, -math.sqrt(3) / 2, 1.0, -1.0, 1 / math.sqrt(2), -1 / math.sqrt(2))


def _snap(matrix: np.ndarray) -> np.ndarray:
    """iso prints 3 decimals; restore the exact entries before multiplying,
    or rounding grows through hundreds of products."""
    out = matrix.copy()
    for index, value in np.ndenumerate(matrix):
        best = min(_EXACT, key=lambda e: abs(e - value))
        if abs(best - value) < 5e-3:
            out[index] = best
    return out


def _fraction_text(vector) -> str:
    return " ".join(str(Fraction(float(x)).limit_denominator(48)) for x in vector)


def space_group_from_log(findsym_log_text: str) -> int:
    m = re.search(r"Space Group:?\s+(\d+)", findsym_log_text)
    if not m:
        raise IsotropyBasisError("findsym log names no space group")
    return int(m.group(1))


def runner_configured() -> bool:
    return _iso_runner is not None


# ---------------------------------------------------------------------------
# frame: ISOTROPY standard setting <-> our POSCAR
# ---------------------------------------------------------------------------

def _lattice_parameters(rows: np.ndarray) -> tuple:
    a, b, c = (np.linalg.norm(v) for v in rows)
    angle = lambda u, v: math.degrees(math.acos(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))))
    return (a, b, c, angle(rows[1], rows[2]), angle(rows[0], rows[2]), angle(rows[0], rows[1]))


def _crystallographic_columns(params) -> np.ndarray:
    """ISOTROPY's Cartesian embedding: a along x, b in the xy plane."""
    a, b, c, alpha, beta, gamma = params
    ca, cb, cg = (math.cos(math.radians(v)) for v in (alpha, beta, gamma))
    sg = math.sin(math.radians(gamma))
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    cz = math.sqrt(max(c * c - cx * cx - cy * cy, 0.0))
    return np.array([[a, b * cg, cx], [0.0, b * sg, cy], [0.0, 0.0, cz]])


@dataclass
class StandardFrame:
    """f_input = Mt @ f_standard + origin (column vectors)."""

    Mt: np.ndarray
    origin: np.ndarray
    standard_lattice_parameters: tuple
    standard_centrings: list


def _pure_translations(parent: Structure, tol: float = 1e-3) -> list:
    """Every non-zero fractional translation mapping `parent` onto itself
    (its centring vectors, in its own frame)."""
    f = np.asarray(parent.frac_coords, float) % 1.0
    species = list(parent.species)
    out = []
    for j in range(len(f)):
        if species[j] != species[0]:
            continue
        t = (f[j] - f[0]) % 1.0
        if np.allclose(t, 0.0, atol=tol) or np.allclose(t, 1.0, atol=tol):
            continue
        g = (f + t) % 1.0
        diff = g[:, None, :] - f[None, :, :]
        diff -= np.round(diff)
        if all(any(np.linalg.norm(diff[i, k]) < tol and species[i] == species[k] for k in range(len(f)))
               for i in range(len(f))):
            out.append(t)
    return out


def standard_frame(parent: Structure, findsym_log_text: str | None) -> StandardFrame:
    """The ISOTROPY-standard <-> POSCAR transformation from findsym's log. With
    no log, or one without a CIF block (pre-6.x), the POSCAR is ASSUMED to be
    in the standard setting and the origin is left to be searched
    (`origin=None`); every generator is still checked to be a symmetry, so a
    wrong assumption raises rather than passes."""
    from .findsym import FindsymLogError, parse_axis_transform, parse_cif_symmetry_operations, parse_origin_shift

    if findsym_log_text:
        try:
            parse_cif_symmetry_operations(findsym_log_text)
        except FindsymLogError:
            findsym_log_text = None
    if not findsym_log_text:
        return StandardFrame(np.eye(3), None, _lattice_parameters(np.asarray(parent.lattice, float)),
                             _pure_translations(parent))
    M = parse_axis_transform(findsym_log_text)
    origin = parse_origin_shift(findsym_log_text)
    standard_rows = M @ np.asarray(parent.lattice, float)
    centrings = []
    for R, t in parse_cif_symmetry_operations(findsym_log_text):
        t = np.asarray(t, float) % 1.0
        if np.allclose(R, np.eye(3)) and not np.allclose(t, 0.0, atol=1e-6):
            centrings.append(t)
    return StandardFrame(M.T, origin, _lattice_parameters(standard_rows), centrings)


# ---------------------------------------------------------------------------
# the irrep's representation, in OUR frame
# ---------------------------------------------------------------------------

@dataclass
class Representation:
    space_group: int
    irrep_label: str
    dimension: int
    generators: list = field(default_factory=list)
    """[(R_input (int 3x3), t_input (3,), D (d x d)), ...]"""
    translations: list = field(default_factory=list)
    """[(t_input (3,), D), ...]: the parent's lattice vectors, then centrings."""
    origin: np.ndarray | None = None
    """Where ISOTROPY's origin sits in the parent's fractional coordinates."""
    origin_from_log: bool = True
    """False when no findsym log fixed the frame and `origin` was searched."""
    origin_candidates: list = field(default_factory=list)
    """Every origin that makes ISOTROPY's operations symmetries (no-log case).
    They can be related by translations that PERMUTE irrep labels, so the
    seeds decide among them (`joint_basis`)."""


def _is_symmetry(parent: Structure, R, t, tol: float = 1e-3) -> bool:
    f = np.asarray(parent.frac_coords, float) % 1.0
    g = (f @ np.asarray(R, float).T + t) % 1.0
    diff = g[:, None, :] - f[None, :, :]
    diff -= np.round(diff)
    dist = np.linalg.norm(diff, axis=2)
    same = np.array([[a == b for b in parent.species] for a in parent.species])
    return bool(np.all(np.min(np.where(same, dist, np.inf), axis=1) < tol))


def _candidate_origins(parent: Structure, ops: list, A_iso_inv) -> list:
    """No findsym log: every origin p (POSCAR already in the standard axes)
    that makes all ISOTROPY generators symmetries of `parent`, on a 1/8 grid
    (then 1/24 if none). Several can exist; they may permute irrep labels."""
    rotations = [np.round(A_iso_inv @ op.rotation_cartesian @ np.linalg.inv(A_iso_inv)).astype(int) for op in ops]
    translations = [np.array([float(Fraction(x)) for x in op.translation_label.split(",")]) for op in ops]
    for denominator in (8, 24):
        found = [np.array(p) for p in itertools.product([i / denominator for i in range(denominator)], repeat=3)
                 if all(_is_symmetry(parent, R, t + np.array(p) - R @ np.array(p))
                        for R, t in zip(rotations, translations))]
        if found:
            return found
    raise IsotropyBasisError("no origin makes ISOTROPY's operations symmetries of this structure "
                             "(no findsym log was available to give the frame)")


def seeds_of(irrep) -> list:
    """One seed per DISTINCT mode. For a complex pair (`D1+D2+`) SMODES prints
    every partner, `degeneracy` blocks per mode, consecutively."""
    if getattr(irrep, "is_complex_pair", False):
        return [mode.displacements for mode in irrep.modes[::irrep.degeneracy]]
    return [mode.displacements for mode in irrep.modes]


def resolve_space_group(irreps, findsym_log_text: str | None) -> int:
    if findsym_log_text:
        try:
            return space_group_from_log(findsym_log_text)
        except IsotropyBasisError:
            pass
    numbers = {getattr(irrep, "space_group_number", None) for irrep in irreps}
    if len(numbers) == 1 and None not in numbers:
        return numbers.pop()
    raise IsotropyBasisError("the space group is unknown (no findsym log, and the irreps do not carry it)")


def representation(irrep, parent: Structure, space_group: int, findsym_log_text: str | None,
                   origin_override=None) -> Representation:
    """ISOTROPY's operations and matrices for `irrep`, moved into `parent`'s frame."""
    frame = standard_frame(parent, findsym_log_text)
    Mt, Mt_inv, origin = frame.Mt, np.linalg.inv(frame.Mt), frame.origin
    A_iso = _crystallographic_columns(frame.standard_lattice_parameters)
    A_iso_inv = np.linalg.inv(A_iso)

    ops = parse_cartesian_operations(_run(build_operations_input(space_group, frame.standard_lattice_parameters)))
    lattice_vectors = [np.eye(3)[i] for i in range(3)]
    centrings_input = [(Mt @ c) % 1.0 for c in frame.standard_centrings]
    translation_elements = ["E " + _fraction_text(Mt_inv @ e) for e in lattice_vectors] + \
                           ["E " + _fraction_text(c) for c in frame.standard_centrings]
    d = irrep.degeneracy
    blocks = parse_irrep_matrices(
        _run(build_matrices_input(space_group, irrep.label, [op.element for op in ops] + translation_elements)), d
    )
    if len(blocks) != len(ops) + len(translation_elements):
        raise IsotropyBasisError(
            f"{irrep.label}: iso returned {len(blocks)} matrices for "
            f"{len(ops) + len(translation_elements)} requested elements"
        )

    from_log = origin is not None
    candidates = [] if from_log else _candidate_origins(parent, ops, A_iso_inv)
    if origin_override is not None:
        origin = np.asarray(origin_override, float)
    elif origin is None:
        origin = candidates[0]
    rep = Representation(space_group, irrep.label, d, origin=origin, origin_from_log=from_log,
                         origin_candidates=candidates)
    for op, (_label, D) in zip(ops, blocks):
        R_std = A_iso_inv @ op.rotation_cartesian @ A_iso
        # The EXACT fractional translation from iso's own label: its Cartesian
        # printout has 5 decimals in Angstrom, ~1e-5 off an exact quarter,
        # which is enough to stop the group closure from ever closing.
        t_std = np.array([float(Fraction(x)) for x in op.translation_label.split(",")])
        R_in = Mt @ R_std @ Mt_inv
        R_int = np.round(R_in).astype(int)
        if not np.allclose(R_in, R_int, atol=1e-3):
            raise IsotropyBasisError(
                f"ISOTROPY operation {op.element} is not an integer matrix in the parent's frame -- "
                "the findsym log and ISOTROPY's setting disagree"
            )
        t_in = Mt @ t_std + origin - R_int @ origin
        rep.generators.append((R_int, t_in, D))
    for t, (_label, D) in zip(lattice_vectors + centrings_input, blocks[len(ops):]):
        rep.translations.append((t, D))
    return rep


def point_group_rotations(parent: Structure, space_group: int, findsym_log_text: str | None) -> list:
    """Every rotation of the crystal's point group, as integer matrices in
    `parent`'s fractional frame (acting on column vectors) -- the closure of
    ISOTROPY's generators, carried into the parent's frame exactly as
    `representation` does. For questions about the point group alone (LO-TO's
    equivalent q-directions); no irrep, translation or origin is involved."""
    frame = standard_frame(parent, findsym_log_text)
    Mt, Mt_inv = frame.Mt, np.linalg.inv(frame.Mt)
    A_iso = _crystallographic_columns(frame.standard_lattice_parameters)
    A_iso_inv = np.linalg.inv(A_iso)
    ops = parse_cartesian_operations(_run(build_operations_input(space_group, frame.standard_lattice_parameters)))
    generators = []
    for op in ops:
        R_in = Mt @ (A_iso_inv @ op.rotation_cartesian @ A_iso) @ Mt_inv
        R_int = np.round(R_in).astype(int)
        if not np.allclose(R_in, R_int, atol=1e-3):
            raise IsotropyBasisError(
                f"ISOTROPY operation {op.element} is not an integer matrix in the parent's frame -- "
                "the findsym log and ISOTROPY's setting disagree"
            )
        generators.append(R_int)
    group = {tuple(np.eye(3, dtype=int).ravel()): np.eye(3, dtype=int)}
    frontier = list(group.values())
    while frontier:
        new = []
        for R in frontier:
            for G in generators:
                P = G @ R
                key = tuple(P.ravel())
                if key not in group:
                    group[key] = P
                    new.append(P)
        frontier = new
    return list(group.values())


# ---------------------------------------------------------------------------
# cell, seeds, group, projection
# ---------------------------------------------------------------------------

def cell_multipliers(rep: Representation, limit: int = 24) -> tuple:
    """Smallest n_i with D(lattice vector i)^n_i = 1: the diagonal multiple of
    the parent cell on which every component of this irrep is periodic."""
    out = []
    for t, D in rep.translations[:3]:
        power = np.eye(rep.dimension)
        for n in range(1, limit + 1):
            power = D @ power
            if np.allclose(power, np.eye(rep.dimension), atol=1e-6):
                out.append(n)
                break
        else:
            raise IsotropyBasisError(f"{rep.irrep_label}: no period up to {limit} along axis {len(out)}")
    return tuple(out)


def cell_matrix(multipliers) -> np.ndarray:
    """A cell as a matrix M: rows are its lattice vectors in units of the
    parent's (new_lattice = M @ parent.lattice). A 3-tuple is diagonal. Rows
    may be RATIONAL for a centred parent -- real CMO X2+'s own cell has rows
    like (-1/2, 1/2, -1/2) of the I-centred conventional cell."""
    M = np.asarray(multipliers, dtype=float)
    if M.shape != (3, 3):
        M = np.diag(M)
    if abs(np.linalg.det(M)) < 1e-6:
        raise IsotropyBasisError(f"cell {M.tolist()} is singular")
    return M


def _cell_offsets(M: np.ndarray) -> np.ndarray:
    """Integer parent translations reaching every coset of an integer cell M."""
    M_inv = np.linalg.inv(M)
    n = int(round(abs(np.linalg.det(M))))
    reach = int(np.ceil(np.abs(M).sum(axis=0).max())) + 1
    found = {}
    for c in itertools.product(range(-reach, reach + 1), repeat=3):
        key = tuple(round(x, 6) % 1.0 for x in np.round(np.asarray(c, float) @ M_inv, 6) % 1.0)
        if key not in found:
            found[key] = c
            if len(found) == n:
                break
    return np.array(sorted(found.values()), dtype=float)


def tiled_parent_with_index(parent: Structure, multipliers):
    """The parent tiled onto cell M (rational rows allowed), atom-major: each
    parent atom, then its copies, so every species stays one contiguous POSCAR
    block. Also returns, per atom, the index of the parent atom it copies --
    which is how per-atom parent data (MAGMOM) is carried, including onto a
    cell SMALLER than a centred conventional parent, where atoms that are
    centring images of each other merge."""
    M = cell_matrix(multipliers)
    M_inv = np.linalg.inv(M)
    expected = int(round(parent.n_atoms * abs(np.linalg.det(M))))
    reach = int(np.ceil(np.abs(M).sum(axis=0).max())) + 1
    offsets = sorted(itertools.product(range(-reach, reach + 1), repeat=3), key=lambda c: (sum(map(abs, c)), c))
    species, frac, index, seen = [], [], [], set()
    for j, (s, f) in enumerate(zip(parent.species, parent.frac_coords)):
        x = np.asarray(f, float) % 1.0
        for c in offsets:
            g = ((x + np.asarray(c, float)) @ M_inv) % 1.0
            key = (s, tuple(round(v, 5) % 1.0 for v in np.round(g, 5)))
            if key in seen:
                continue
            seen.add(key)
            species.append(s)
            frac.append(g)
            index.append(j)
    if len(species) != expected:
        raise IsotropyBasisError(f"tiling onto cell {np.round(M, 4).tolist()} gave {len(species)} atoms, "
                                 f"expected {expected} -- is it a lattice of the parent's translations?")
    lattice = M @ np.asarray(parent.lattice, float)
    structure = Structure(comment=f"{parent.comment} cell {np.round(M, 4).tolist()}", lattice=lattice,
                          species=species, frac_coords=np.array(frac))
    return structure, np.array(index)


def enlarge_exactly(structure: Structure, K) -> Structure:
    """`structure` repeated onto the cell K @ structure.lattice (K an integer
    matrix), exactly, in fractional coordinates -- keeping whatever strain the
    structure carries. Atom-major, so species stay contiguous."""
    K = np.round(np.asarray(K, float))
    K_inv = np.linalg.inv(K)
    offsets = _cell_offsets(K)
    species, frac = [], []
    for s, f in zip(structure.species, structure.frac_coords):
        for c in offsets:
            species.append(s)
            frac.append(((np.asarray(f, float) + c) @ K_inv) % 1.0)
    return Structure(comment=f"{structure.comment} x{K.astype(int).tolist()}",
                     lattice=K @ np.asarray(structure.lattice, float), species=species, frac_coords=np.array(frac))


def tiled_parent(parent: Structure, multipliers) -> Structure:
    return tiled_parent_with_index(parent, multipliers)[0]


def _permutation(reference: Structure, multipliers, R, t, tol: float = 1e-3) -> np.ndarray:
    """perm[i] = index of the atom that (R, t) -- column convention, parent
    fractional -- carries atom i onto, on cell M."""
    M = cell_matrix(multipliers).astype(float)
    f = np.asarray(reference.frac_coords, float) % 1.0
    x = f @ M                                                     # parent-fractional rows
    y = ((x @ np.asarray(R, float).T + t) @ np.linalg.inv(M)) % 1.0
    diff = y[:, None, :] - f[None, :, :]
    diff -= np.round(diff)
    dist = np.linalg.norm(diff @ np.asarray(reference.lattice, float), axis=2)
    perm = np.argmin(dist, axis=1)
    if np.max(dist[np.arange(len(perm)), perm]) > 0.05 or len(set(perm)) != len(perm):
        raise IsotropyBasisError("an ISOTROPY operation is not a symmetry of the parent in this frame")
    species = reference.species
    if any(species[i] != species[perm[i]] for i in range(len(perm))):
        raise IsotropyBasisError("an ISOTROPY operation maps one species onto another")
    return perm


def translation_matrix(rep: "Representation", vector) -> np.ndarray:
    """ISOTROPY's matrix for a parent translation -- an integer lattice vector
    or one plus a centring vector -- from the lattice-vector and centring
    matrices (translations commute)."""
    v = np.asarray(vector, float)
    for t_c, D_c in [(np.zeros(3), np.eye(rep.dimension))] + list(rep.translations[3:]):
        n = v - np.asarray(t_c, float)
        if np.allclose(n, np.round(n), atol=1e-6):
            D = D_c.copy()
            for (_t, Dk), k in zip(rep.translations[:3], np.round(n).astype(int)):
                D = D @ np.linalg.matrix_power(Dk if k >= 0 else np.linalg.inv(Dk), abs(int(k)))
            return D
    raise IsotropyBasisError(f"{np.round(v, 4).tolist()} is not a translation of the parent")


def hosts(rep: "Representation", multipliers) -> bool:
    """True when every lattice vector of cell M acts as the identity on this
    irrep -- i.e. every component is periodic on M."""
    M = cell_matrix(multipliers)
    return all(np.allclose(translation_matrix(rep, row), np.eye(rep.dimension), atol=1e-6) for row in M)


def _lattice_basis(generators: list) -> np.ndarray:
    """A 3-row basis of the lattice the rational row vectors `generators`
    generate (integer row reduction after clearing denominators)."""
    den = 1
    for g in generators:
        for x in g:
            den = math.lcm(den, Fraction(float(x)).limit_denominator(24).denominator)
    rows = [[int(round(float(x) * den)) for x in g] for g in generators]
    rows = [r for r in rows if any(r)]
    basis = []
    for col in range(3):
        while True:
            nonzero = [r for r in rows if r[col] != 0]
            if len(nonzero) <= 1:
                break
            pivot = min(nonzero, key=lambda r: abs(r[col]))
            for r in nonzero:
                if r is not pivot:
                    q = r[col] // pivot[col]
                    r[:] = [a - q * b for a, b in zip(r, pivot)]
            rows = [r for r in rows if any(r)]
        nonzero = [r for r in rows if r[col] != 0]
        if nonzero:
            basis.append(nonzero[0])
            rows = [r for r in rows if r is not nonzero[0]]
    if len(basis) != 3:
        raise IsotropyBasisError("the invariant translations do not span three dimensions")
    return np.array(basis, dtype=float) / den


def reduce_cell(cell, lattice) -> np.ndarray:
    """The same lattice as `cell` (rows, in `lattice`'s fractional
    coordinates), on its shortest near-orthogonal basis, ordered and signed to
    follow the parent's own axes, right-handed.

    `_lattice_basis` returns a valid but arbitrary (echelon) basis: real CMO
    P5 at (0,-a,-a,0) came out [[1,-3,-2],[0,1,-1],[0,0,2]] -- 40 A vectors on
    a 48-atom cell -- unusable as DFT input and slow to decompose against."""
    B = np.array(cell_matrix(cell), float)
    L = np.asarray(lattice, float)
    changed = True
    while changed:
        changed = False
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                ci, cj = B[i] @ L, B[j] @ L
                mu = round(float(ci @ cj / (cj @ cj)))
                if mu and np.linalg.norm(ci - mu * cj) < np.linalg.norm(ci) - 1e-9:
                    B[i] = B[i] - mu * B[j]
                    changed = True
    cart = B @ L
    axes = L / np.linalg.norm(L, axis=1)[:, None]
    unit = cart / np.linalg.norm(cart, axis=1)[:, None]
    best = max(itertools.permutations(range(3)),
               key=lambda p: (round(sum(abs(unit[p[i]] @ axes[i]) for i in range(3)), 9), [-q for q in p]))
    B = B[list(best)]
    for i in range(3):
        if (B[i] @ L) @ axes[i] < -1e-9:
            B[i] = -B[i]
    if np.linalg.det(B) < 0:
        worst = min(range(3), key=lambda i: abs(((B[i] @ L) / np.linalg.norm(B[i] @ L)) @ axes[i]))
        B[worst] = -B[worst]
    return B


def invariant_cell(reps: dict, vectors: dict, base_cell, allow_centring: bool = True, lattice=None) -> np.ndarray:
    """The SMALLEST cell on which the requested order parameters are periodic:
    spanned by `base_cell` (where every component is periodic) plus every
    parent translation T with D_label(T) v_label = v_label for each requested
    irrep. `vectors` = {label: OPD coefficient vector}. With
    `allow_centring=False` only integer translations count (a magnetic parent
    whose moments break the centring). With `lattice` (the parent's), the
    basis is reduced by `reduce_cell`."""
    B = cell_matrix(base_cell)
    first = next(iter(reps.values()))
    centrings = [np.zeros(3)] + ([np.asarray(t_c, float) for t_c, _D in first.translations[3:]] if allow_centring else [])
    generators = [row for row in B]
    if np.allclose(B, np.round(B)):
        offsets = _cell_offsets(np.round(B))
    else:
        offsets = np.array([np.zeros(3)])
    for c in offsets:
        for t_c in centrings:
            T = c + t_c
            if all(np.allclose(translation_matrix(reps[label], T) @ v, v, atol=1e-9) for label, v in vectors.items()):
                generators.append(T)
    basis = _lattice_basis(generators)
    return basis if lattice is None else reduce_cell(basis, lattice)


def fold_onto_cell(base: Structure, base_cell, parent: Structure, cell, patterns: dict):
    """A cell SMALLER than (a superlattice of) `base`'s, on which every pattern
    in `patterns` ({key: (n_base, 3)}) is periodic: returns (structure, parent
    index per atom, {key: (n, 3)}). Raises if a pattern is not actually
    periodic on `cell`."""
    small, index = tiled_parent_with_index(parent, cell)
    B = cell_matrix(base_cell)
    x = np.asarray(small.frac_coords, float) @ cell_matrix(cell)
    f_base = (x @ np.linalg.inv(B)) % 1.0
    g = np.asarray(base.frac_coords, float) % 1.0
    diff = f_base[:, None, :] - g[None, :, :]
    diff -= np.round(diff)
    dist = np.linalg.norm(diff @ np.asarray(base.lattice, float), axis=2)
    match = np.argmin(dist, axis=1)
    if np.max(dist[np.arange(len(match)), match]) > 0.05:
        raise IsotropyBasisError("the small cell's sites do not map onto the base cell")
    folded = {key: np.asarray(P)[match] for key, P in patterns.items()}
    # periodicity check: every base atom equivalent under `cell` must carry the same displacement
    xb = np.asarray(base.frac_coords, float) @ B
    f_small = (xb @ np.linalg.inv(cell_matrix(cell))) % 1.0
    gs = np.asarray(small.frac_coords, float) % 1.0
    d2 = f_small[:, None, :] - gs[None, :, :]
    d2 -= np.round(d2)
    back = np.argmin(np.linalg.norm(d2 @ np.asarray(small.lattice, float), axis=2), axis=1)
    for key, P in patterns.items():
        if np.max(np.abs(np.asarray(P) - folded[key][back])) > 1e-9:
            raise IsotropyBasisError(f"pattern {key} is not periodic on the requested cell")
    return small, index, folded


@dataclass
class _Element:
    R: np.ndarray
    t: np.ndarray
    D: np.ndarray
    perm: np.ndarray
    Rc: np.ndarray


def group_elements(rep: Representation, reference: Structure, parent: Structure, multipliers) -> list:
    """Every operation of the parent modulo the cell (point part x translations
    inside it), each with its ISOTROPY matrix, atom permutation and Cartesian
    rotation. Closure from the generators; atom permutations of products are
    composed, not re-matched."""
    M_inv = np.linalg.inv(cell_matrix(multipliers).astype(float))
    A = np.asarray(parent.lattice, float).T
    A_inv = np.linalg.inv(A)

    def make(R, t, D):
        return _Element(np.asarray(R, int), np.asarray(t, float), D, _permutation(reference, multipliers, R, t),
                        _cartesian_rotation(A, R))

    gens = [make(R, t, D) for R, t, D in rep.generators] + \
           [make(np.eye(3, dtype=int), t, D) for t, D in rep.translations]
    def key(e):
        s = np.round((e.t @ M_inv) % 1.0, 6) % 1.0               # translation, in cell units, mod the cell
        return (tuple(e.R.ravel()), tuple(np.round(s, 6)))
    identity = _Element(np.eye(3, dtype=int), np.zeros(3), np.eye(rep.dimension), np.arange(reference.n_atoms),
                        np.eye(3))
    elements = {key(identity): identity}
    frontier = [identity]
    while frontier:
        new = []
        for e in frontier:
            for g in gens:
                c = _Element(g.R @ e.R, g.R @ e.t + g.t, g.D @ e.D, g.perm[e.perm], g.Rc @ e.Rc)
                k = key(c)
                if k not in elements:
                    elements[k] = c
                    new.append(c)
        frontier = new
    return list(elements.values())


def _cartesian_rotation(A: np.ndarray, R) -> np.ndarray:
    """The Cartesian form of fractional rotation R, made EXACTLY orthogonal
    (polar decomposition). A lattice written to 6 decimals cannot represent
    sqrt(3)/2 (real POSCAR_CsV3Sb5), so A R A^-1 is orthogonal only to ~1e-5,
    which would leak into the constructed basis; the nearest orthogonal matrix
    is the same operation without the rounding."""
    M = A @ np.asarray(R, float) @ np.linalg.inv(A)
    U, _s, Vt = np.linalg.svd(M)
    return U @ Vt


def _act(e: _Element, field: np.ndarray) -> np.ndarray:
    out = np.empty_like(field)
    out[e.perm] = field @ e.Rc.T
    return out


def project(seed: np.ndarray, elements: list, dimension: int):
    """ISOTROPY-basis partners of `seed`: (d, n_atoms, 3), plus the component
    k it was projected from. Unit Frobenius norm each."""
    images = [_act(e, seed) for e in elements]
    best_k, best_norm = 0, 0.0
    for k in range(dimension):
        norm = np.linalg.norm(sum(e.D[k, k] * img for e, img in zip(elements, images)))
        if norm > best_norm + 1e-12:
            best_k, best_norm = k, norm
    if best_norm < 1e-8 * len(elements):
        raise IsotropyBasisError("the seed has no component in this irrep -- do the labels correspond?")
    W = np.array([sum(e.D[j, best_k] * img for e, img in zip(elements, images)) for j in range(dimension)])
    W *= dimension / len(elements)
    W /= np.linalg.norm(W.reshape(dimension, -1), axis=1)[:, None, None]
    return W, best_k


def arm_structure(rep: Representation):
    """(arm number per component, quadrature pairs), read off ISOTROPY's
    translation matrices: at a self-inverse k each translation is a +-1 per
    component, and components sharing a sign pattern share an arm; at a
    non-self-inverse k translations rotate components into each other, and
    those pairs are the quadrature partners (one arm in ISOTROPY's notation)."""
    d = rep.dimension
    mats = [D for _t, D in rep.translations]
    if all(np.allclose(D, np.diag(np.diag(D)), atol=1e-6) for D in mats):
        signature = [tuple(int(round(D[j, j])) for D in mats) for j in range(d)]
        order = list(dict.fromkeys(signature))
        return [order.index(s) + 1 for s in signature], []
    pairs = sorted({(j, l) for D in mats for j in range(d) for l in range(d) if j < l and abs(D[j, l]) > 1e-6})
    return [1] * d, pairs


def arm_k_vectors(rep: Representation, k_vector, rotations: list) -> list:
    """One wavevector per arm of `arm_structure(rep)`, in arm order, in the
    parent's fractional reciprocal basis -- the `Q^(1), Q^(2), ...` a paper
    states an order parameter in.

    Read off ISOTROPY's own translation matrices, so arm N's Q is the
    wavevector of exactly the components ISOTROPY's letters put on arm N: a
    component on wavevector q picks up the phase exp(2 pi i q.T) under a
    translation T, so D(T) is cos(2 pi q.T) on that component (and
    +-sin(2 pi q.T) into its quadrature partner). Each arm is matched against
    the star of `k_vector` under `rotations` (`point_group_rotations`; a
    wavevector transforms by inv(R).T). Testing the CENTRING translations as
    well as the lattice vectors is what separates members that differ by a
    reciprocal vector of the conventional cell but not of the crystal -- real
    TiO2 X's (1/2,1/2,0) and (1/2,-1/2,0). Members are kept UNWRAPPED, as
    the star generates them. Raises if an arm matches no member."""
    k = np.asarray(k_vector, dtype=float)
    orbit: list = []
    for q in [k] + [np.linalg.inv(np.asarray(R, float)).T @ k for R in rotations]:
        if not any(np.allclose(q, seen, atol=1e-9) for seen in orbit):
            orbit.append(q)
    arms, pairs = arm_structure(rep)
    partner = {j: l for j, l in pairs} | {l: j for j, l in pairs}
    def matches(q, j0):
        for t, D in rep.translations:
            phase = 2 * np.pi * float(np.asarray(t, float) @ q)
            if abs(np.cos(phase) - D[j0, j0]) > 1e-6:
                return False
            if j0 in partner and abs(abs(np.sin(phase)) - abs(D[partner[j0], j0])) > 1e-6:
                return False
        return True

    out = []
    for arm in sorted(set(arms)):
        j0 = arms.index(arm)
        # q and -q are the same real wave and pass the same test; of every
        # matching representative, print the most readable (fewest negative
        # components), keeping the star's own order on ties.
        hits = [s * q for q in orbit for s in (1, -1) if matches(s * q, j0)]
        if not hits:
            raise IsotropyBasisError(f"{rep.irrep_label}: arm {arm} matches no member of the star of {k.tolist()}")
        out.append(min(hits, key=lambda q: int(np.sum(q < -1e-9))) + 0.0)
    return out


def symmetrized_parent(parent: Structure, rep: Representation) -> Structure:
    """`parent` with every atom averaged over the space group's operations, so
    it is EXACTLY symmetric. A real POSCAR is symmetric only to its written
    precision (CMO_smodes-phonons: centring partners differ by up to 4e-7 A),
    and on a cell smaller than a centred conventional one those partners MERGE
    into one site -- which copy wins can then differ from cell to cell, leaving
    a 1e-6 A residual between two descriptions of one structure. Moves no atom
    by more than that noise; species and order unchanged."""
    I3 = np.eye(3)
    elements = group_elements(rep, tiled_parent(parent, I3), parent, I3)
    f = np.asarray(parent.frac_coords, float) % 1.0
    total = np.zeros_like(f)
    for e in elements:
        # atom i is carried onto atom perm[i]: x_perm[i] = R x_i + t (mod 1)
        estimate = (np.linalg.inv(e.R.astype(float)) @ (f[e.perm] - e.t).T).T
        total += f + ((estimate - f + 0.5) % 1.0 - 0.5)
    sym = (total / len(elements)) % 1.0
    return Structure(comment=parent.comment, lattice=np.asarray(parent.lattice, float),
                     species=list(parent.species), frac_coords=sym)


@dataclass
class JointBasis:
    """Every requested irrep's ISOTROPY basis on ONE shared cell."""

    reference: Structure
    """The parent tiled onto cell `multipliers`, atom-major (atom i is parent
    atom i // n_cells) -- so a per-atom parent quantity such as MAGMOM tiles by
    `np.repeat(values, n_cells)`."""
    multipliers: np.ndarray
    """The cell as an integer matrix (rows in parent lattice units)."""

    @property
    def n_cells(self) -> int:
        return int(round(abs(np.linalg.det(self.multipliers))))
    bases: dict
    """{label: [W per seed]}, each W (d, n_atoms, 3), rows in ISOTROPY's
    component order, unit Frobenius norm."""
    representations: dict
    """{label: Representation}"""
    parent: Structure = None
    """The (exactly symmetrized) parent every cell here is tiled from."""

    def position_labels(self, label: str) -> list:
        return position_labels(self.representations[label])


def joint_basis(seeded: list, parent: Structure, space_group: int, findsym_log_text: str,
                target_multiplier=None) -> JointBasis:
    """`seeded`: [(irrep, [seed, ...]), ...], each seed an (n_native, 3) pattern
    in the irrep's own SMODES atom ordering -- a SMODES mode, or a DFT
    eigenvector expressed in that ordering. One shared cell for all of them:
    the common diagonal multiple of the parent on which every requested irrep
    is periodic (so irreps at DIFFERENT k-points combine too)."""
    from . import supercell as supercell_module

    reps = {irrep.label: representation(irrep, parent, space_group, findsym_log_text) for irrep, _s in seeded}
    parent = symmetrized_parent(parent, reps[seeded[0][0].label])
    per_irrep = [cell_multipliers(reps[irrep.label]) for irrep, _s in seeded]
    # The construction sums over the symmetry operations, so it needs a cell
    # every operation maps onto itself: the diagonal period of the irreps is
    # one (D(e_i)^n_i = 1 is symmetric under the point group). A requested
    # cell -- `opd.build_opd_bases(target_lattice=...)`, or the distorted
    # structure's own cell in `decomposition` -- need not be (STO M padded to
    # 4x2x2 is not invariant under the 4-fold), so the basis is built on the
    # symmetric cell and tiled onto the requested one afterwards: exact, since
    # every pattern is periodic on the symmetric cell.
    multipliers = cell_matrix(tuple(math.lcm(*axis) for axis in zip(*per_irrep)))
    requested = None
    if target_multiplier is not None:
        requested = cell_matrix(target_multiplier)
        bad = [irrep.label for irrep, _s in seeded if not hosts(reps[irrep.label], requested)]
        if bad:
            raise IsotropyBasisError(f"target cell {requested.tolist()} does not host {', '.join(bad)} "
                                     f"(each must be periodic on it)")
        if np.allclose(requested, multipliers):
            requested = None
    reference = tiled_parent(parent, multipliers)

    tiled_sets = {}
    for irrep, seeds in seeded:
        tiled_sets[irrep.label] = supercell_module.tile_onto_supercell(
            irrep, np.asarray(parent.lattice, float), multipliers,
            {"seeds": [np.asarray(s, float) for s in seeds]}, comment=irrep.label,
        )

    first = reps[seeded[0][0].label]
    # ONE origin for every irrep: it belongs to the frame, not to an irrep. With
    # a findsym log it is fixed; without one, the candidates are tried and the
    # first under which EVERY seed belongs to its own label is kept (origins
    # related by a non-symmetry translation permute labels, e.g. graphite H1/H2).
    origins = [first.origin] if first.origin_from_log else first.origin_candidates
    failures = []
    for origin in origins:
        bases: dict = {}
        chosen: dict = {}
        try:
            for irrep, _seeds in seeded:
                rep = reps[irrep.label] if first.origin_from_log else representation(
                    irrep, parent, space_group, findsym_log_text, origin_override=origin)
                tiled, sets = tiled_sets[irrep.label]
                bases[irrep.label] = _project_seeds(rep, reference, parent, multipliers, tiled, sets["seeds"],
                                                    origin, irrep.label)
                chosen[irrep.label] = rep
        except IsotropyBasisError as exc:
            failures.append(str(exc))
            continue
        reps.update(chosen)
        if requested is not None:
            within = requested @ np.linalg.inv(multipliers)
            if np.allclose(within, np.round(within), atol=1e-6):
                # a SUPERCELL of the symmetric period: tile the patterns up
                target_reference, index = _tile_reference(reference, multipliers, parent, requested)
                bases = {label: [_renormalised(W[:, index, :]) for W in Ws] for label, Ws in bases.items()}
            else:
                # any other cell the irreps are periodic on (e.g. smaller, with
                # centred rows): fold every pattern onto it, checking periodicity
                keyed = {(label, m, j): W[j] for label, Ws in bases.items()
                         for m, W in enumerate(Ws) for j in range(W.shape[0])}
                target_reference, _index, folded = fold_onto_cell(reference, multipliers, parent, requested, keyed)
                bases = {label: [_renormalised(np.array([folded[(label, m, j)] for j in range(W.shape[0])]))
                                 for m, W in enumerate(Ws)] for label, Ws in bases.items()}
            return JointBasis(target_reference, requested, bases, reps, parent)
        return JointBasis(reference, multipliers, bases, reps, parent)
    raise IsotropyBasisError("no consistent frame -- " + "; ".join(dict.fromkeys(failures)))


def _tile_reference(base: Structure, base_cell, parent: Structure, target_cell):
    """`target_cell`'s tiled parent, and for each of its atoms the index of the
    same site (modulo the base cell) in `base`."""
    target = tiled_parent(parent, target_cell)
    B = cell_matrix(base_cell).astype(float)
    x = np.asarray(target.frac_coords, float) @ cell_matrix(target_cell).astype(float)   # parent-fractional
    f_base = (x @ np.linalg.inv(B)) % 1.0
    g = np.asarray(base.frac_coords, float) % 1.0
    diff = f_base[:, None, :] - g[None, :, :]
    diff -= np.round(diff)
    dist = np.linalg.norm(diff @ np.asarray(base.lattice, float), axis=2)
    index = np.argmin(dist, axis=1)
    if np.max(dist[np.arange(len(index)), index]) > 0.05:
        raise IsotropyBasisError("target cell sites do not map onto the base cell")
    return target, index


def _renormalised(W: np.ndarray) -> np.ndarray:
    return W / np.linalg.norm(W.reshape(W.shape[0], -1), axis=1)[:, None, None]


def _project_seeds(rep, reference, parent, multipliers, tiled, seeds, origin, label) -> list:
    elements = None
    failures = []
    for perm in _seed_placements(reference, tiled, multipliers, origin):
        if elements is None:
            elements = group_elements(rep, reference, parent, multipliers)
        try:
            out = []
            for seed_index, seed in enumerate(seeds):
                W, _k = project(np.asarray(seed)[perm], elements, rep.dimension)
                _check_representation(W, rep, reference, parent, multipliers, label, seed_index)
                out.append(W)
            return out
        except IsotropyBasisError as exc:
            failures.append(str(exc))
    raise IsotropyBasisError(f"{label}: " + ("; ".join(dict.fromkeys(failures)) or "SMODES's tiled cell does not align"))


def _seed_placements(reference: Structure, tiled: Structure, multipliers, origin):
    """Atom permutations placing SMODES's tiled cell (standard-setting origin)
    on `reference` (POSCAR origin). First the exact one, shifting by ISOTROPY's
    origin; then whatever a free translation search finds. A free search alone
    can land on a translation that maps the structure onto itself without
    being a symmetry -- which would put the seed in the wrong irrep or swap
    which Wyckoff site a mode lives on -- so it is only the fallback, and the
    projection test decides."""
    from .freeze import FreezeError, align_by_nearest_neighbor, search_alignment_shift

    seen = []
    shifted = (np.asarray(tiled.frac_coords, float)
               + np.asarray(origin, float) @ np.linalg.inv(cell_matrix(multipliers).astype(float))) % 1.0
    for frac in (shifted, np.asarray(tiled.frac_coords, float)):
        try:
            perm = list(align_by_nearest_neighbor(reference.species, reference.frac_coords, tiled.species,
                                                  frac, reference.lattice))
        except FreezeError:
            continue
        if perm not in seen:
            seen.append(perm)
            yield np.asarray(perm)
    _shift, perm = search_alignment_shift(reference, tiled, 0.3)
    if perm is not None and list(perm) not in seen:
        yield np.asarray(perm)


def position_labels(rep: Representation) -> list:
    """Each OPD position's relation to the others, in the vocabulary every
    report already uses -- `native`, `within`, `armN`, `within(armN)`,
    `quadrature`, `quad+within` -- read off ISOTROPY's translation matrices
    (`arm_structure`). Under this basis `native` means the first component on
    star arm 1, not SMODES's own pattern."""
    arms, pairs = arm_structure(rep)
    d = rep.dimension
    half = d // 2
    if pairs and d % 2 == 0 and set(pairs) == {(j, j + half) for j in range(half)}:
        return [("native" if j == 0 else "within") if j < half else
                ("quadrature" if j == half else "quad+within") for j in range(d)]
    labels, seen = [], {}
    for arm in arms:
        w = seen.get(arm, 0)
        seen[arm] = w + 1
        if arm == 1:
            labels.append("native" if w == 0 else "within")
        else:
            labels.append(f"arm{arm}" if w == 0 else f"within(arm{arm})")
    return labels


def build_joint_opd_cell(irreps: list, parent: Structure, space_group: int, findsym_log_text: str,
                         target_multiplier=None):
    """Drop-in for `decomposition.build_joint_opd_cell`, same return shape:
    `(reference_structure, {(label, arm, name): (n_atoms, 3)}, component_count)`,
    seeded with every SMODES mode of every irrep.

    Positions are ISOTROPY's component order. Within an arm the name carries
    `__w<index>` exactly like the old construction, so `_opd_sort_key` orders
    them identically. A quadrature k-point is one arm in ISOTROPY's notation,
    so all its components share arm 1.
    """
    joint = joint_basis([(irrep, seeds_of(irrep)) for irrep in irreps], parent, space_group, findsym_log_text,
                        target_multiplier=target_multiplier)
    patterns: dict = {}
    for irrep in irreps:
        arms, pairs = arm_structure(joint.representations[irrep.label])
        d = len(arms)
        # Quadrature naming (`_quad`) only when ISOTROPY pairs component j with
        # j + d/2 -- then `_opd_sort_key`'s (arm, quadrature, within) order is
        # still ISOTROPY's order. Any other pairing keeps plain within-arm
        # names, so positions can never be reordered by a label.
        half = d // 2
        quad_split = bool(pairs) and d % 2 == 0 and set(pairs) == {(j, j + half) for j in range(half)}
        for mode_index, W in enumerate(joint.bases[irrep.label]):
            within: dict = {}
            for j in range(d):
                if quad_split:
                    w, quad = j % half, j >= half
                else:
                    w = within.get(arms[j], 0)
                    within[arms[j]] = w + 1
                    quad = False
                name = f"m{mode_index}" + ("_quad" if quad else "") + (f"__w{w}" if w else "")
                patterns[(irrep.label, arms[j], name)] = W[j]
    return joint.reference, patterns, len(patterns)


def _check_representation(W, rep, reference, parent, multipliers, label, mode_index, tol: float = 1e-6) -> None:
    """The constructed basis must transform EXACTLY by ISOTROPY's matrices.
    This is what makes a wrong frame, a wrong operation or a label mismatch
    loud rather than a plausible-looking wrong basis."""
    d = rep.dimension
    flat = W.reshape(d, -1)
    gram = flat @ flat.T
    if np.max(np.abs(gram - np.eye(d))) > tol:
        raise IsotropyBasisError(f"{label} mode {mode_index}: constructed basis is not orthonormal")
    A = np.asarray(parent.lattice, float).T
    A_inv = np.linalg.inv(A)
    for R, t, D in list(rep.generators) + [(np.eye(3, dtype=int), t, D) for t, D in rep.translations]:
        e = _Element(np.asarray(R, int), np.asarray(t, float), D, _permutation(reference, multipliers, R, t),
                     _cartesian_rotation(A, R))
        got = np.array([[flat[a] @ _act(e, W[b]).ravel() for b in range(d)] for a in range(d)])
        if np.max(np.abs(got - D)) > 1e-5:
            raise IsotropyBasisError(
                f"{label} mode {mode_index}: constructed basis does not transform by ISOTROPY's matrices "
                f"(max error {np.max(np.abs(got - D)):.1e})"
            )

"""One order-parameter-direction basis, built once and validated at birth.

`build_opd_bases` builds every OPD of the requested irreps on ONE cell
(`decomposition.build_joint_opd_cell`, i.e. the ISOTROPY-matrix basis of
`isotropy_basis`), and wraps each mode's directions in an `OPDBasis` with its
measured quality; `format_opd_basis` renders the report `irrep --opd` and
`phonons --generate-opd` write.

THE LOAD-BEARING DESIGN DECISION. An `OPDBasis` owns exactly ONE structure,
and every pattern is validated against it at construction. Comparing patterns
that live on different cells is meaningless -- atom 5 of one cell is not atom
5 of another -- and that mistake was made three separate times in this
project. Bundling the patterns with their cell makes it structurally
impossible rather than something to remember.

`rank` and `conditioning` are measured FIELDS, not methods, so they appear in
every repr and every log line without anyone choosing to ask.

COMPOSITION, NOT NEW SYMMETRY. Positions, their mechanism labels (arm,
within-arm, quadrature) and each arm's Q-vector all come from ISOTROPY's own
matrices; this module adds typing, validation and naming -- nothing else.
"""

from __future__ import annotations

import collections
import re
import sys
from dataclasses import dataclass, field

import numpy as np

from . import isotropy_basis
from .freeze import opd_basis_quality
from .smodes import Irrep
from .structure import Structure

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def _format_q(component: float) -> str:
    """A k-vector component as a physicist writes it: `1/2`, `-1/3`, `0`.
    Zone-boundary components are always low-denominator rationals, so a
    decimal like `0.3333` is strictly less readable and no more exact."""
    from fractions import Fraction

    exact = Fraction(component).limit_denominator(12)
    if abs(float(exact) - component) > 1e-6:
        return f"{component:.4f}"
    if exact.denominator == 1:
        return str(exact.numerator)
    return f"{exact.numerator}/{exact.denominator}"


def format_direction_spec(position: int, arms) -> str:
    """The `--direction` string selecting `position` alone.

    `arms[i]` is the star arm hosting position `i`. Semicolons separate
    STAR ARMS and commas separate positions within one arm -- `iso`'s own
    convention, verified against `iso` 9.6.1 (TiO2 X1, self-inverse,
    comes back `(a,0;0,0)`).

    Shared by `OPDBasis.direction_spec` and `decomposition`'s own
    per-direction totals table so the two cannot describe the same order
    parameter differently. They did: `irrep --opd` printed real CMO X4-
    as `a;0` while `decomposition` printed `a,0` for that same irrep,
    which hides the fact that those two positions are different star arms
    rather than within-arm partners.

    A single arm (or no arm information at all) gives the flat comma
    form, which is what every non-multi-arm irrep should show.
    """
    arms = list(arms)
    letters = ["a" if i == position else "0" for i in range(len(arms))]
    distinct = sorted(set(arms))
    if len(distinct) <= 1:
        return ",".join(letters)
    groups: dict = collections.defaultdict(list)
    for letter, arm in zip(letters, arms):
        groups[arm].append(letter)
    return ";".join(",".join(groups[arm]) for arm in distinct)


class OPDError(ValueError):
    """Raised when an OPD basis cannot be built or fails its own validation."""


@dataclass(frozen=True)
class OPD:
    """One order-parameter direction of one mode of one irrep."""

    position: int
    """0-based index into the irrep's order parameter. Positions are a property
    of the IRREP and shared by all its modes, so `--direction` is written once
    per irrep -- writing it per mode would imply a distinction that does not
    exist."""

    arm: int
    """1-based star arm hosting this direction."""

    within_index: int
    """0-based index among the directions this arm contributes."""

    mechanism: str
    """How it was found: `native`, `armN`, `within`, `quadrature`,
    `quad+within`, or `orbit`. Descriptive only -- a label names an OPD
    regardless of how it was reached, per the physicist's own framing. Kept
    because `orbit` in particular means "no mechanism carrying physical
    identity could reach this", which a reader deserves to see."""

    pattern: np.ndarray
    """(n_atoms, 3) displacement, on the OWNING BASIS's structure. Never
    meaningful apart from it."""


@dataclass(frozen=True)
class OPDBasis:
    """Every order-parameter direction of ONE mode of ONE irrep, on one cell."""

    irrep_label: str
    mode_index: int
    degeneracy: int
    n_arms: int
    within_arm: int
    structure: Structure
    """THE cell. Every `OPD.pattern` is validated against it at construction."""
    opds: tuple[OPD, ...]
    arm_k_vectors: tuple = ()
    """One reciprocal-space wave-vector per star arm, in the SAME order the
    arms are numbered -- `Q_M^(1), Q_M^(2), ...` in the notation a CDW paper
    uses. This is the only bridge between our internal arm bookkeeping and the
    literature: a paper states its order parameter as Q-vectors, and without
    these you cannot check that "arm 2" and "Q = (1/2, 0, 1/2)" are the same
    claim. Empty when the arm search could not run -- a cosmetic label must
    never take the report down with it."""
    rank: int = field(init=False)
    conditioning: float = field(init=False)

    def __post_init__(self) -> None:
        expected = (self.structure.n_atoms, 3)
        for opd in self.opds:
            shape = np.asarray(opd.pattern).shape
            if shape != expected:
                raise OPDError(
                    f"{self.irrep_label} mode {self.mode_index} position "
                    f"{opd.position}: pattern is {shape}, but this basis's cell has "
                    f"{self.structure.n_atoms} atoms (expected {expected}). A pattern "
                    "belongs to exactly one cell; mixing cells silently compares "
                    "atom i of one against atom i of another."
                )
        rank, conditioning = opd_basis_quality([o.pattern for o in self.opds])
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "conditioning", conditioning)

    @property
    def is_complete(self) -> bool:
        """Spans the degenerate space. NOT the same as having `degeneracy`
        entries -- real STO M5- once had 6 entries spanning 3 dimensions."""
        return self.rank == self.degeneracy

    def direction_spec(self, position: int) -> str:
        """The `--direction` string selecting `position` alone, in `iso`'s own
        notation: semicolons separate STAR ARMS at a self-inverse k-point, and
        a non-self-inverse one is flat because `+k` and `-k` form a single
        physically irreducible representation with no arm split to mark.

        Verified against real `iso` 9.6.1 -- TiO2 X1 (self-inverse) comes back
        `(a,0;0,0)` while TiO2 P1 (not) comes back `(a,0,a,0)`.
        """
        if not 0 <= position < len(self.opds):
            raise OPDError(
                f"{self.irrep_label}: position {position} out of range for "
                f"degeneracy {self.degeneracy}"
            )
        if self.n_arms > 1 and self.within_arm >= 1 and self._grouped_by_arm:
            return format_direction_spec(position, [o.arm for o in self.opds])
        return format_direction_spec(position, [0] * len(self.opds))

    @property
    def _grouped_by_arm(self) -> bool:
        """Semicolons are only meaningful when the entries really do partition
        into star arms. A quadrature substitution collapses the star onto one
        arm, and then a flat spec is the honest description."""
        return len({o.arm for o in self.opds}) == self.n_arms and self.n_arms > 1

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        state = "complete" if self.is_complete else f"SHORT {self.rank}/{self.degeneracy}"
        return (
            f"<OPDBasis {self.irrep_label} mode {self.mode_index} "
            f"{self.n_arms}x{self.within_arm} {state} cond={self.conditioning:.3g} "
            f"cell={self.structure.n_atoms} atoms>"
        )


_MODE_INDEX = re.compile(r"m(\d+)")






def target_multiplier(parent: Structure, target_lattice, tol: float = 1e-6):
    """`target_lattice` as an integer multiplier of `parent.lattice`, or
    `None` when no target was requested.

    THE THIRD CELL. An `OPDBasis` owns exactly one cell, and the three
    consumers need different ones: `irrep --opd` wants the irrep's own
    (smallest, most readable), `freeze` wants the smallest that hosts the
    requested directions (DFT cost is atoms), and `decomposition` gets NO
    choice at all -- the user hands it a relaxed CONTCAR and the analysis
    must happen on that cell, strain and all. Passing the lattice through to
    the existing `force_joint_multiplier` seam is what lets one construction
    serve all three, rather than `decomposition` re-deriving directions by a
    second route. Two implementations of the same physics is the failure this
    module exists to prevent.

    REQUIRES AN EXACT INTEGER RELATIONSHIP, and says so plainly when there
    isn't one. A target that is not a whole multiple of the parent cannot
    host the parent's own periodicity, so there is no correct answer to
    round toward -- guessing here would produce a cell that tiles
    arithmetically and means nothing physically.

    NOTE FOR A STRAINED `sub`. A real relaxed structure is strained relative
    to the parent by real physics, so its ABSOLUTE lattice will not clear
    this check. The multiplier is a property of `(parent, k-vector)` alone,
    so the caller derives it against the UNSTRAINED parent and applies it to
    the strained lattice -- exactly what `decomposition.enlarge_onto_lattice`
    already does via its own `reference_lattice` fallback. Do not relax the
    tolerance here to accommodate strain; that would silently accept a
    genuinely non-commensurate target.
    """
    if target_lattice is None:
        return None
    target = np.asarray(target_lattice, dtype=np.float64)
    if target.shape != (3, 3):
        raise OPDError(f"target_lattice must be 3x3, got {target.shape}")
    parent_lattice = np.asarray(parent.lattice, dtype=np.float64)
    multiplier = target @ np.linalg.inv(parent_lattice)
    volume_ratio = abs(float(np.linalg.det(multiplier)))

    if volume_ratio < 1e-9:
        raise OPDError("target_lattice is singular (zero volume)")

    # SMALLER THAN THE PARENT is a genuinely different situation from a
    # rounding failure, and must not be reported as one. For a CENTRED
    # setting the irrep's own native cell really is smaller -- real CMO
    # Gamma's is 12 atoms against a 24-atom conventional parent, a true
    # det = 1/2 relationship, not numerical noise. It is refused anyway,
    # because hosting it would mean reducing the PARENT.
    if volume_ratio < 1.0 - 1e-6:
        raise OPDError(
            f"target_lattice is SMALLER than the parent cell (volume ratio "
            f"{volume_ratio:.4g}). This is refused deliberately, not for lack of a "
            "representation: building there would mean reducing `--parent`, which is a "
            "documented do-not-retry -- `irrep.k_vector` and `irrep.superlattice` are "
            "tied to the basis smodes was run against, and an I-centring reduction is a "
            "genuine basis change.\n"
            "  The supported route is the opposite one: ENLARGE the smaller structure "
            "onto a common lattice first (`decomposition.enlarge_onto_lattice`), then "
            "pass THAT as target_lattice. A `--sub` smaller than `--parent` is a real, "
            "already-handled case; it is handled by growing sub, never by shrinking "
            "the parent."
        )

    rounded = np.round(multiplier)
    error = float(np.abs(multiplier - rounded).max())
    if error > tol:
        raise OPDError(
            "target_lattice is not an integer multiple of the parent lattice "
            f"(largest deviation {error:.3g}). A cell that is not a whole multiple of "
            "the parent cannot host its periodicity, so there is no correct rounding. "
            "If this is a strained relaxed structure, derive the multiplier against the "
            "UNSTRAINED parent and apply it to the strained lattice -- see "
            "`decomposition.enlarge_onto_lattice`'s own `reference_lattice` fallback. "
            "Do not loosen this tolerance to accommodate strain; that would silently "
            "accept a genuinely non-commensurate target."
        )
    return rounded.astype(int)


OPD_MAX_CROSS_IRREP_OVERLAP = 1e-3
"""Distinct irreps are ORTHOGONAL (Schur), so the physical bound is zero --
but the MEASUREMENT is not exact, and this sits between the two scales that
actually occur.

Set at 1e-6 initially, which was below the known noise floor and flagged
ordinary tiling residue as a defect: real graphite K and H come in at 4e-6
and 5e-6, and CsV3Sb5 K/L/M at 7e-5 to 9e-5 -- all the documented ~1e-5
cross-talk floor from per-irrep resolution (see known issue 3 in the developer notes),
none of it structural.

The genuine failures are FIVE ORDERS OF MAGNITUDE larger: 0.707107 and
1.000000. Nothing real has ever been observed between 1e-4 and 0.28, so
1e-3 separates the two populations with a wide margin on both sides rather
than splitting a continuum."""


def cross_irrep_overlap(bases: dict) -> tuple:
    """`(worst_overlap, description)` across every pair of OPDs belonging to
    DIFFERENT irreps. `description` is `None` when nothing overlaps.

    THE CHECK THAT TESTS IDENTITY RATHER THAN A PROPERTY. Six cheap
    validity tests have failed in this project -- count, rank, conditioning,
    tileability, folded-cosine, and rank-with-conditioning -- and every one
    measures something ABOUT a basis instead of asking whether its vectors
    are the ones they claim to be. A basis can be three mutually orthogonal
    vectors of perfect conditioning and still contain a direction belonging
    to a neighbouring irrep: real STO GM4- reports `rank 3/3, cond 1.0`
    while its third direction IS a GM5- one, overlap exactly 0.707107.

    Only valid for bases that SHARE A CELL -- pass the dict from a single
    `build_opd_bases` call. Patterns on different cells cannot be compared
    index-wise, and doing so anyway is a mistake this project has made
    repeatedly. Bases of differing atom counts are skipped rather than
    silently compared.
    """
    entries = []
    for (label, mode), basis in sorted(bases.items()):
        for opd in basis.opds:
            vector = np.asarray(opd.pattern, dtype=np.float64).ravel()
            norm = float(np.linalg.norm(vector))
            if norm > 0.0:
                entries.append((label, mode, opd.position, basis.structure.n_atoms,
                                vector / norm))

    worst = 0.0
    description = None
    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            if entries[i][0] == entries[j][0] or entries[i][3] != entries[j][3]:
                continue
            overlap = abs(float(entries[i][4] @ entries[j][4]))
            if overlap > worst:
                worst = overlap
                description = (
                    f"{entries[i][0]}.m{entries[i][1]}.{LETTERS[entries[i][2]]} . "
                    f"{entries[j][0]}.m{entries[j][1]}.{LETTERS[entries[j][2]]}"
                )
    return worst, description


def _warn_on_cross_irrep_overlap(bases: dict) -> None:
    """Report, once per build, when a constructed direction does not belong
    to the irrep it is filed under.

    A WARNING RATHER THAN AN ERROR, deliberately. This currently fires on
    real STO Gamma, STO R and CMO Gamma -- 3 of 6 surveyed k-points -- and
    raising would take working commands (`irrep --opd`) offline for a defect
    that has been present since v1.3.0 and that users may be tolerating
    unknowingly. The amplitudes for OPDs that ARE correct remain correct.
    What is unacceptable is that it was silent: a 70%-wrong direction was
    being reported as `rank 3/3, cond 1.0`.
    """
    if len(bases) < 2:
        return
    worst, description = cross_irrep_overlap(bases)
    if worst > OPD_MAX_CROSS_IRREP_OVERLAP:
        print(
            f"Warning: order-parameter directions overlap ACROSS irreps "
            f"(worst |dot| = {worst:.6f}, {description}). Distinct irreps are "
            "orthogonal, so a direction this large belongs to a different irrep "
            "than the one it is listed under -- freezing along it would produce "
            "part of another irrep's distortion, and decomposing would attribute "
            "amplitude to the wrong one. Directions that do NOT appear here are "
            "unaffected.",
            file=sys.stderr,
        )


def build_opd_bases(
    irreps: list,
    parent: Structure,
    *,
    target_lattice=None,
    findsym_log_text: str | None = None,
    tol: float = 1e-2,
) -> dict:
    """Every mode of every requested irrep, keyed `(label, mode_index)`.

    ONE `build_joint_opd_cell` call serves them all: the cell, the group and
    the projection are built once, not once per mode.

    THE IRREP LIST IS THE SCOPE, and the two useful scopes are exactly the
    two the CLI already distinguishes:

    - `[irrep]` -- that irrep's own cell. `irrep GM5+`.
    - every irrep at a k-point -- the shared k-point cell. `irrep GM`.

    All modes of one irrep genuinely DO share a cell (a physical fact, not
    a convention), and every basis returned from a single call shares one
    structure, so patterns from this dict may be combined with each other
    freely. That is what `phonons --generate_opd` needs: its contraction
    sums over MODES, so per-mode cells would make the sum meaningless.
    """
    from .decomposition import _opd_sort_key, build_joint_opd_cell

    structure, patterns, _count = build_joint_opd_cell(
        irreps, parent, tol=tol, findsym_log_text=findsym_log_text,
        force_joint_multiplier=target_multiplier(parent, target_lattice),
    )

    bases: dict = {}
    for irrep in irreps:
        # Positions are ISOTROPY's components; their mechanism labels and each
        # arm's Q-vector are read off ISOTROPY's own matrices (cached).
        # Q-vectors are cosmetic and failure-tolerant: a failed match costs the
        # label and nothing else.
        rep = isotropy_basis.representation(
            irrep, parent, isotropy_basis.resolve_space_group([irrep], findsym_log_text), findsym_log_text)
        v2_labels = isotropy_basis.position_labels(rep)
        try:
            frame = isotropy_basis.standard_frame(parent, findsym_log_text)
            k_parent = np.linalg.inv(frame.Mt).T @ np.array([float(x) for x in irrep.k_vector])
            rotations = isotropy_basis.point_group_rotations(parent, rep.space_group, findsym_log_text)
            arm_qs = tuple(tuple(float(c) for c in q)
                           for q in isotropy_basis.arm_k_vectors(rep, k_parent, rotations))
        except (isotropy_basis.IsotropyBasisError, ValueError, np.linalg.LinAlgError):
            arm_qs = ()

        keys = sorted((k for k in patterns if k[0] == irrep.label), key=_opd_sort_key)
        by_mode: dict = collections.defaultdict(list)
        for key in keys:
            match = _MODE_INDEX.match(key[2])
            if match:
                by_mode[int(match.group(1))].append(key)

        for mode_index, ordered in sorted(by_mode.items()):
            arms = sorted({key[1] for key in ordered})
            per_arm_seen: dict = collections.defaultdict(int)
            opds = []
            for position, key in enumerate(ordered):
                arm = key[1]
                opds.append(OPD(
                    position=position,
                    arm=arm,
                    within_index=per_arm_seen[arm],
                    mechanism=v2_labels[position],
                    pattern=np.asarray(patterns[key], dtype=np.float64),
                ))
                per_arm_seen[arm] += 1

            bases[(irrep.label, mode_index)] = OPDBasis(
                irrep_label=irrep.label,
                mode_index=mode_index,
                degeneracy=irrep.degeneracy,
                n_arms=len(arms),
                within_arm=max(1, irrep.degeneracy // max(len(arms), 1)),
                structure=structure,
                opds=tuple(opds),
                arm_k_vectors=arm_qs,
            )
    _warn_on_cross_irrep_overlap(bases)
    return bases


_MECHANISM_NOTES = {
    "native": "ISOTROPY's first component on star arm 1",
    "within": "within-arm partner, from this arm's own little group",
    "quadrature": "Bloch quadrature partner -- a 90-degree PHASE shift, not "
                  "reachable by any rotation or permutation of atoms",
    "quad+within": "within-arm partner of the Bloch quadrature partner",
    "orbit": "orbit completion -- no mechanism carrying physical identity "
             "reached this direction; it was recovered from the orbit of the "
             "pattern under the crystal's verified operations",
}


def _mechanism_note(mechanism: str) -> str:
    """One sentence explaining a mechanism, for a reader who is a physicist
    rather than a developer. `armN`/`within(armN)` are formatted rather than
    looked up, since N varies."""
    if mechanism in _MECHANISM_NOTES:
        return _MECHANISM_NOTES[mechanism]
    # v2.0 (ISOTROPY-matrix basis): positions are ISOTROPY's own components,
    # described by structure rather than by how they were found.
    if mechanism.startswith("quadrature with "):
        return (f"Bloch quadrature partner of position(s) {mechanism[len('quadrature with '):]} -- "
                "a lattice translation rotates one into the other")
    if mechanism.startswith(("arm ", "within-arm ")) or mechanism == "single":
        return f"ISOTROPY component ({mechanism})"
    if mechanism.startswith("within(arm"):
        arm = mechanism[len("within(arm"):-1]
        return f"within-arm partner on star arm {arm}"
    if mechanism.startswith("arm"):
        return (f"star arm {mechanism[3:]} -- reached by a crystal symmetry "
                "operation (a rotation, or a centring translation)")
    return mechanism


def format_opd_basis(basis: OPDBasis, *, show_patterns: bool = True) -> list[str]:
    """Render one `OPDBasis` as report lines. Pure -- no I/O, no files.

    THE single place an OPD basis is described in words, so `irrep`,
    `decomposition` and the freeze family cannot describe the same irrep
    differently. That divergence is not hypothetical: the acceptance
    criterion was independently reinvented at four call sites before it was
    consolidated, each version weaker than the real one.

    Two decisions settled with the physicist, both visible here:

    - **The full per-atom pattern is shown**, not a summary with the data
      hidden in a `.npz`. Reading what each order-parameter direction
      actually does to the atoms -- and comparing it against SMODES's own
      native pattern by eye -- is the reason the report exists. Atoms the
      mode does not move are omitted, which is a readability choice, not a
      summary: a mode typically touches one species out of several.

    - **Provenance is stated INLINE, not in a separate file.** How each OPD
      was found belongs next to that OPD's own pattern, because it is a
      property of it. One file also means the two can never disagree about
      how many OPDs there are or what order they sit in.

    The mechanism is DIAGNOSTIC in both directions: it never appears in the
    `--direction` grammar (a label names an OPD regardless of how it was
    reached), and no computation consumes it -- `phonons --generate_opd`
    contracts the PATTERNS.
    """
    state = "complete" if basis.is_complete else f"INCOMPLETE ({basis.rank} of {basis.degeneracy})"
    lines = [
        f"Order-parameter directions for {basis.irrep_label}, mode {basis.mode_index}",
        "",
        f"  degeneracy    {basis.degeneracy}  =  {basis.n_arms} star arm(s) "
        f"x {basis.within_arm} within-arm direction(s)",
        f"  cell          {basis.structure.n_atoms} atoms",
        f"  basis         {state}, conditioning {basis.conditioning:.4g}",
    ]
    if len(basis.arm_k_vectors) > 1:
        # Single-arm irreps are skipped: the one Q-vector IS the k-point, which
        # the report's own header already names, and at Gamma it reads
        # `Q^(1) = (0, 0, 0)` -- pure noise.
        # The arm's own Q-vector, in the SAME order the arms are numbered, so
        # `arm 2` in the table below and `Q^(2)` here are the same arm. This is
        # what a CDW paper quotes; see `direction.arm_k_vectors`.
        lines.append(
            "  arm Q-vectors "
            + "   ".join(
                f"Q^({i + 1}) = ({', '.join(_format_q(c) for c in q)})"
                for i, q in enumerate(basis.arm_k_vectors)
            )
        )
    if not basis.is_complete:
        lines.append(
            f"  WARNING       only {basis.rank} independent direction(s) were found where "
            f"{basis.degeneracy} are expected. Having {basis.degeneracy} entries is not the "
            "same as spanning them; this basis does NOT span the degenerate subspace, so a "
            "distortion along a missing direction cannot be represented or detected."
        )
    lines.append("")

    for opd in basis.opds:
        letter = LETTERS[opd.position] if opd.position < len(LETTERS) else f"?{opd.position}"
        lines.append(
            f"  {letter}  --direction {basis.direction_spec(opd.position)}"
            f"   [arm {opd.arm}, {opd.mechanism}]"
        )
        lines.append(f"       {_mechanism_note(opd.mechanism)}")
    lines.append("")

    if basis.n_arms > 1:
        lines.append(
            "  Positions sharing a star arm sit on the same cell -- a rotation of the pattern, "
            "nothing structural changes. Positions on different arms are genuinely different "
            "star members. Semicolons in --direction separate arms."
        )
    lines.append(
        "  A letter names a direction, never the mechanism that found it: --direction is "
        "written the same way whether a position is a within-arm partner, on another star "
        "arm, or a Bloch-phase (quadrature) partner."
    )

    if show_patterns:
        lines.append("")
        for opd in basis.opds:
            letter = LETTERS[opd.position] if opd.position < len(LETTERS) else f"?{opd.position}"
            lines.append(f"  Position {letter} ({opd.mechanism}) -- displacement per atom:")
            # ATOM INDEX, not just the species. Only atoms that actually move
            # are listed, so without an index a material with several atoms of
            # one species (BaSnO3's three O) gives no way to tell WHICH oxygen
            # a row refers to. 1-based and global, matching smodes's own
            # `atom  type   displacement` table so the two can be read side by
            # side against the same structure.
            lines.append("    atom  type   displacement")
            moved = 0
            for index, (species, vector) in enumerate(
                zip(basis.structure.species, opd.pattern), start=1
            ):
                if float(np.linalg.norm(vector)) > 1e-6:
                    lines.append(
                        f"    {index:>4}   {species:<4} {vector[0]: .6f}  "
                        f"{vector[1]: .6f}  {vector[2]: .6f}"
                    )
                    moved += 1
            if not moved:
                lines.append("    (no atom moves by more than 1e-6 -- this pattern is numerically zero)")
            lines.append("")
    return lines


def opd_bases_for_eigenvectors(bases_by_mode: dict, coefficients: np.ndarray) -> dict:
    """Re-express OPDs in terms of DFT eigenvectors instead of SMODES basis
    modes. Keyed by eigenvector index, same shape as the input.

    This is a CHANGE OF BASIS, not a new symmetry search -- the whole reason
    `phonons --generate_opd` needs no findsym, no subprocess, and no symmetry
    code at runtime. The OPD operation `R_p` is linear, so for an eigenvector
    `v_i = sum_j c_ji b_j` over the SMODES basis modes `b_j`:

        R_p v_i = R_p sum_j c_ji b_j = sum_j c_ji (R_p b_j)

    The OPD of an eigenvector is the SAME linear combination of the OPDs of
    the basis modes, which `irrep` already computed. `R_p` is a genuine
    crystal symmetry and so COMMUTES with the dynamical matrix, making
    `R_p v_i` an eigenvector at the same eigenvalue -- a real degenerate
    partner, not merely a rotated pattern.

    ORTHOGONALITY IS NOT USED, only linearity. SMODES basis modes are
    mutually orthogonal and eigenvectors are MIXTURES of them; that is fine.
    The eigenvectors stay mutually orthogonal in the mass-weighted metric
    (`eigh` on a real symmetric matrix guarantees it), so mixing is not
    cross-talk and nothing leaks between eigenvectors.

    `coefficients[j, i]` is mode `j`'s contribution to eigenvector `i` --
    pass `dyn_vec / sqrt(masses)[:, None]`, EXACTLY what
    `phonons._reconstruct_eigenvectors` is given for
    `atomic_displacement_eigenvectors`. Two normalization details make that
    the right input rather than an approximation of it, both verified rather
    than assumed:

    - The mass factor is required. `eigh` runs on the MASS-WEIGHTED matrix,
      so its raw eigenvectors are not displacement amplitudes until divided
      by `sqrt(m_j)`.
    - `build_joint_opd_cell` builds OPD patterns from `mode.displacements`
      -- the RAW, un-Frobenius-normalized vector -- which is the same source
      `_reconstruct_eigenvectors` uses. Checked directly, because if one used
      `.normalized` and the other `.displacements` the contraction would
      still run and would silently weight the modes wrongly relative to each
      other. It does not; no per-mode scale correction is needed.

    Each result is unit-normalized, matching `_reconstruct_eigenvectors`.

    Two properties change character here and must not be assumed downstream:
    a mode's OPD partners can be linearly DEPENDENT (already true of the
    SMODES-basis OPDs -- real TiO2 X1 has Gram off-diagonals of exactly
    +-1.0), and ONE-SPECIES-PER-MODE IS LOST, since an eigenvector mixes
    modes and therefore mixes species.
    """
    if not bases_by_mode:
        raise OPDError("no OPD bases given")

    modes = sorted(bases_by_mode)
    reference = bases_by_mode[modes[0]]
    coefficients = np.asarray(coefficients, dtype=np.float64)

    n_modes = len(modes)
    if coefficients.shape != (n_modes, n_modes):
        raise OPDError(
            f"{reference.irrep_label}: coefficients are {coefficients.shape}, expected "
            f"({n_modes}, {n_modes}) for {n_modes} mode(s). Every mode of the irrep must be "
            "present -- the contraction sums over modes, so a missing one silently omits "
            "part of each eigenvector."
        )
    for mode in modes:
        basis = bases_by_mode[mode]
        if basis.structure is not reference.structure:
            raise OPDError(
                f"{reference.irrep_label}: mode {mode} sits on a different cell than mode "
                f"{modes[0]}. Summing over modes requires ONE shared cell -- atom i of one "
                "cell is not atom i of another. Build every mode in a single "
                "`build_opd_bases` call."
            )
        if len(basis.opds) != len(reference.opds):
            raise OPDError(
                f"{reference.irrep_label}: mode {mode} has {len(basis.opds)} OPD(s), mode "
                f"{modes[0]} has {len(reference.opds)}. Positions must correspond across modes."
            )

    out: dict = {}
    for eigen_index in range(n_modes):
        opds = []
        for position, template in enumerate(reference.opds):
            accumulated = np.zeros_like(np.asarray(template.pattern, dtype=np.float64))
            for row, mode in enumerate(modes):
                accumulated = accumulated + (
                    coefficients[row, eigen_index]
                    * np.asarray(bases_by_mode[mode].opds[position].pattern, dtype=np.float64)
                )
            norm = float(np.linalg.norm(accumulated))
            if norm == 0.0:
                raise OPDError(
                    f"{reference.irrep_label}: eigenvector {eigen_index}, position "
                    f"{template.position} contracted to exactly zero -- degenerate or "
                    "malformed input."
                )
            opds.append(OPD(
                position=template.position,
                arm=template.arm,
                within_index=template.within_index,
                mechanism=template.mechanism,
                pattern=accumulated / norm,
            ))
        out[eigen_index] = OPDBasis(
            irrep_label=reference.irrep_label,
            mode_index=eigen_index,
            degeneracy=reference.degeneracy,
            n_arms=reference.n_arms,
            within_arm=reference.within_arm,
            structure=reference.structure,
            opds=tuple(opds),
        )
    return out


def format_native_only(irrep: Irrep, reason: str) -> list[str]:
    """Last-resort report for an irrep whose OPD basis could not be BUILT.

    Distinct from an INCOMPLETE basis, and the difference is worth keeping
    straight: an incomplete basis was constructed successfully and simply
    does not span its own degenerate subspace (it gets the ordinary report,
    with a WARNING). A hard failure means construction raised -- there is no
    cell and no constructed pattern at all.

    What survives either way is SMODES's own printed pattern per mode, which
    needs no symmetry construction, no findsym and no joint cell: it is read
    straight out of the smodes output and is exactly what `Mode_i/POSCAR`
    already contains. Reporting it is strictly better than reporting
    nothing -- the reader still learns what the calculation they are about
    to run actually displaces, and learns precisely which extra directions
    are unavailable and why.
    """
    lines = [
        f"Order-parameter directions for {irrep.label} -- COULD NOT BE CONSTRUCTED",
        "",
        f"  degeneracy    {irrep.degeneracy}",
        f"  reason        {reason}",
        "",
        f"  Only SMODES's own native direction is available below. The other "
        f"{max(irrep.degeneracy - 1, 0)} direction(s) of this "
        f"{irrep.degeneracy}-fold degenerate irrep could not be constructed, so this "
        "irrep cannot currently be frozen along, or decomposed onto, anything but its "
        "native direction.",
        "",
        "  The native pattern below is exactly what each Mode_i/POSCAR already "
        "contains -- those structures are unaffected and ready to run.",
        "",
    ]
    for mode_index, mode in enumerate(irrep.modes):
        lines.append(f"  Mode {mode_index}, native direction -- displacement per atom:")
        # See the note at the other print site: the index is what identifies
        # WHICH atom of a repeated species moved.
        lines.append("    atom  type   displacement")
        moved = 0
        for index, (species, vector) in enumerate(
            zip(irrep.atom_species, np.asarray(mode.normalized)), start=1
        ):
            if float(np.linalg.norm(vector)) > 1e-6:
                lines.append(
                    f"    {index:>4}   {species:<4} {vector[0]: .6f}  "
                    f"{vector[1]: .6f}  {vector[2]: .6f}"
                )
                moved += 1
        if not moved:
            lines.append("    (no atom moves by more than 1e-6)")
        lines.append("")
    return lines



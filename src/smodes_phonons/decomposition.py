"""Mode decomposition: given a parent structure and an already-distorted
("sub") structure -- typically a real relaxed CONTCAR -- recover which
symmetry modes, at which k-vector(s), and along which order-parameter
direction(s), are actually present, plus the lattice strain relating the two.

Not a wrapper around ISODISTORT (its "Method 4" mode decomposition is a
web-only feature, not part of the downloadable ISOTROPY package) -- this is a
from-scratch implementation on data and machinery validated elsewhere in this
codebase (findsym.py, smodes.py, supercell.py, isotropy_basis.py).

- The parent and sub need not share a lattice: `sub` is validated as a
  genuine supercell of the SAME material (`check_stoichiometry`), the strain
  tensor is reported as its own output (`compute_strain`), and the INTERNAL
  displacement (`internal_displacement`) is strain- and rotation-invariant.
- Several k-vectors per call, each decomposed independently -- needed for
  e.g. Ca3Mn2O7's hybrid improper ferroelectricity, with real contributions at
  a zone-boundary k-point and at Gamma at once.
- Per k-vector, ONE joint cell hosts every order-parameter direction of every
  irrep there, constructed from ISOTROPY's irrep matrices
  (`isotropy_basis.build_joint_opd_cell`), so positions are ISOTROPY's
  components. `sub` is aligned to it once and each irrep's amplitudes are
  solved by least squares against the same displacement: distinct irreps are
  exactly orthogonal on the shared cell (Schur), while one irrep's components
  can be oblique, where independent dot products would double-count.

The v1.x construction -- per-irrep target resolution, multi-arm blend
detection, the within-arm direction sweep -- is recorded in
the v1.x development record.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import re
import sys

import numpy as np

from . import isotropy_basis as isotropy_basis_module
from . import opd as opd_module
from . import findsym as findsym_module
from .freeze import (
    FreezeError,
    align_by_nearest_neighbor,
    search_alignment_shift,
)
from .smodes import k_vector_label_from_irrep
from .freeze import _SHIFT_SNAP_DENOMINATORS
from .structure import Structure
from .supercell import (
    enlarge_onto_lattice,
)


class DecompositionError(ValueError):
    """Raised when a sub structure can't be decomposed against a set of irreps."""


# ---------------------------------------------------------------------------
# Stoichiometry
# ---------------------------------------------------------------------------


def check_stoichiometry(parent: Structure, sub: Structure) -> int:
    """Verify `sub` is a genuine supercell of `parent`'s own material --
    every chemical species must scale by the *identical* integer multiple
    `N`, not just "some integer each" (different per-species multiples
    would mean a different chemical formula entirely, not a bigger cell of
    the same one). Returns `N`. Raises DecompositionError naming exactly
    which species disagree otherwise -- a wrong-material error should
    never look like a supercell-shape error.
    """
    parent_counts = dict(parent.species_counts)
    sub_counts = dict(sub.species_counts)

    if set(parent_counts) != set(sub_counts):
        raise DecompositionError(
            f"distorted structure's elements {sorted(sub_counts)} do not match parent's "
            f"{sorted(parent_counts)} -- not the same material"
        )

    ratios = {}
    for species, n_parent in parent_counts.items():
        n_sub = sub_counts[species]
        if n_sub % n_parent != 0:
            ratios[species] = n_sub / n_parent
        else:
            ratios[species] = n_sub // n_parent

    distinct = set(ratios.values())
    if len(distinct) != 1 or not isinstance(next(iter(distinct)), int):
        detail = ", ".join(f"{sp}×{r:g}" for sp, r in sorted(ratios.items()))
        raise DecompositionError(
            f"distorted structure's per-species atom-count ratios are not a single consistent "
            f"integer multiple of the parent's (found {detail}) -- not a valid supercell "
            "of this material"
        )

    return next(iter(distinct))


# ---------------------------------------------------------------------------
# Strain
# ---------------------------------------------------------------------------


@dataclass
class StrainResult:
    deformation_gradient: np.ndarray
    """F, such that sub_lattice ~= reference_lattice @ F (row-vector convention,
    matching every other lattice operation in this codebase)."""
    rotation: np.ndarray
    """R, the proper (det = +1) orthogonal part of F -- a pure re-orientation,
    zero physical strain. Real relaxed cells routinely come back with a real,
    nonzero R even when no *physical* rotation occurred, purely because the
    DFT code's own Cartesian embedding doesn't have to match the reference's
    -- the same class of issue already documented for SMODES's own
    superlattice embedding (see supercell.py's module docstring)."""
    strain: np.ndarray
    """eta = U - I, the symmetric part (U = the right stretch tensor,
    F = R @ U) -- the genuine physical deformation, with any pure
    re-orientation already factored out."""
    determinant: float
    """det(F) -- the volume ratio between sub and reference. Negative would
    mean sub is a mirror image of reference (a genuinely different chirality,
    not a strain at all); never silently corrected, just reported."""


def compute_strain(reference_lattice, sub_lattice) -> StrainResult:
    """Deformation gradient F = reference_lattice^{-1} @ sub_lattice, then
    the right polar decomposition F = R @ U, U symmetric positive
    (semi-)definite, R proper-orthogonal.

    U is computed directly from the eigendecomposition of the right
    Cauchy-Green tensor C = F^T F (always symmetric PSD, with no sign
    ambiguity at all -- U = C^{1/2} unconditionally), rather than via the
    more common SVD route, specifically to sidestep the SVD's own sign-flip
    edge case when det(F) < 0: U's eigenvalues (the principal stretches)
    must always be non-negative regardless of whether R itself needs a
    sign correction, and computing U first, independently, guarantees that
    without any extra casework. R = F @ U^{-1} follows directly and is
    automatically proper whenever det(F) > 0 (the ordinary case for a real
    relaxation); det(F) is reported as its own field precisely so a caller
    can notice, rather than silently paper over, the unusual det(F) < 0
    case.
    """
    L0 = np.asarray(reference_lattice, dtype=np.float64)
    L1 = np.asarray(sub_lattice, dtype=np.float64)
    F = np.linalg.solve(L0, L1)

    C = F.T @ F
    eigvals, eigvecs = np.linalg.eigh(C)
    U = eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None))) @ eigvecs.T
    R = F @ np.linalg.inv(U)

    return StrainResult(
        deformation_gradient=F,
        rotation=R,
        strain=U - np.eye(3),
        determinant=float(np.linalg.det(F)),
    )


# ---------------------------------------------------------------------------
# Internal (strain-invariant) displacement
# ---------------------------------------------------------------------------


def _smaller_alignment(reference: Structure, sub: Structure, tol: float, direct_perm):
    """A shifted alignment better than the direct one, or `None`.

    Returns `(origin_shift, perm, shifted_sub_frac)` only when a uniform
    origin shift produces a STRICTLY smaller total displacement than the
    permutation already found. Any failure of the search is simply "no
    better candidate" -- the direct alignment already succeeded, so there is
    nothing to recover from and nothing to report.
    """
    def norm_for(perm, shift):
        frac = np.asarray(sub.frac_coords, dtype=np.float64)
        if shift is not None:
            frac = (frac - shift) % 1.0
        delta = frac[perm] - reference.frac_coords
        delta -= np.round(delta)
        return float(np.linalg.norm(delta @ reference.lattice)), frac

    direct_norm, _ = norm_for(direct_perm, None)
    try:
        shift, perm = search_alignment_shift(reference, sub, tol)
    except Exception:  # noqa: BLE001 -- absence of a better option, not an error
        return None
    if shift is None or perm is None:
        return None
    shifted_norm, shifted_frac = norm_for(perm, shift)
    if shifted_norm >= direct_norm:
        return None
    return shift, perm, shifted_frac


def _is_clean_origin_shift(shift) -> bool:
    """True when every component of an origin shift is a low-denominator
    fraction -- the only values a genuine origin re-setting can take.

    `search_alignment_shift` derives its shift from ONE anchor atom and snaps
    it to such a fraction only within 1e-3. If the anchor atom is itself
    displaced by more than that (routine at a 0.05 A freeze amplitude), the
    shift keeps the anchor's own displacement and comes out OFF this grid.
    """
    for x in np.asarray(shift, dtype=np.float64):
        if not any(abs(x * d - round(x * d)) < 1e-6 for d in _SHIFT_SNAP_DENOMINATORS):
            return False
    return True


def _anchor_sublattice(reference: Structure) -> list[int]:
    """Indices of the atoms on the origin-shift ANCHOR's own sublattice.

    `search_alignment_shift` anchors on the first atom of `reference`'s
    least-common species. Its sublattice is every same-species atom that is
    a pure lattice translation of it -- found from `reference` itself (the
    undistorted, tiled parent): a translation maps a sublattice onto itself
    exactly when it maps the WHOLE undistorted structure onto itself. This
    includes centring translations, which are genuine lattice translations of
    a centred parent.
    """
    counts: dict = {}
    for s in reference.species:
        counts[s] = counts.get(s, 0) + 1
    anchor_species = min(counts, key=counts.get)
    anchor = reference.species.index(anchor_species)
    frac = np.asarray(reference.frac_coords, dtype=np.float64)
    members = []
    for j, s in enumerate(reference.species):
        if s != anchor_species:
            continue
        t = frac[j] - frac[anchor]
        try:
            align_by_nearest_neighbor(reference.species, frac, reference.species, (frac + t) % 1.0,
                                      reference.lattice, tol=1e-3)
        except FreezeError:
            continue
        members.append(j)
    return members


def _remove_anchor_contamination(displacement, reference, origin_shift):
    """Undo the rigid shift an off-grid origin estimate leaves behind, and
    return the origin shift that reproduces the corrected displacement.

    Called only when the origin shift came out OFF the clean-fraction grid.
    The shift was defined to make the anchor atom's own displacement exactly
    zero; if the anchor was itself displaced by a zone-boundary (non-Γ)
    distortion, that motion d was mistaken for a shift of the whole crystal
    and every atom carries the same spurious -d.

    Remove it by subtracting the AVERAGE displacement of the anchor's own
    sublattice (every lattice-translation copy of it). That average is
    exactly the anchor's non-Γ contamination: a zone-boundary pattern
    averages to zero over a sublattice, while a Γ displacement moves every
    copy identically -- so for a purely Γ structure the average is already
    zero and NOTHING changes. Every Γ result therefore keeps today's
    convention; only the zone-boundary contamination is removed.

    (A first version subtracted the centre-of-mass displacement instead. The
    suite's arbitrary-origin test caught that it moved a pure-Γ result that
    was already exactly right -- recorded in the validation campaign's
    issue list, Issue 1.)
    """
    members = _anchor_sublattice(reference)
    if not members:
        return displacement, origin_shift
    shift_cart = displacement[members].mean(axis=0)
    corrected = displacement - shift_cart
    shift_frac = shift_cart @ np.linalg.inv(np.asarray(reference.lattice, dtype=np.float64))
    effective = np.asarray(origin_shift, dtype=np.float64) + shift_frac
    effective = effective - np.round(effective)
    return corrected, effective


def internal_displacement(reference: Structure, sub: Structure, tol: float = 0.3,
                          prefer_smallest_displacement: bool = False):
    """The internal (non-affine) atomic displacement of `sub` relative to
    `reference`, and the atom permutation aligning them.

    This is, deliberately, the *same* formula v1 used --
    `delta_frac = sub.frac_coords[perm] - reference.frac_coords`, converted
    to Cartesian via `reference.lattice` -- because it turns out to already
    be exactly the right, strain-and-rotation-invariant quantity, not
    something that needed fixing for real (strained) CONTCAR input.

    The reason: fractional coordinates are exactly invariant under *any*
    purely affine response -- if every lattice vector and every atomic
    position is hit by the same linear map (whether a genuine strain, a
    pure rotation, or both), the same (x,y,z) triple still describes each
    atom's position relative to the new lattice vectors. So a structure
    that responds *purely* affinely to the parent-to-sub deformation (zero
    internal relaxation) has, by this formula, exactly zero displacement
    here, regardless of how large or anisotropic the strain is. What's
    left over after taking this difference is, by construction, precisely
    the non-affine part -- exactly what a phonon eigenvector basis can
    meaningfully be projected against, since phonon eigenvectors say
    nothing about strain at all.

    Returns (raw_displacement, perm, origin_shift_frac) -- perm re-indexes
    sub's own atoms to match reference's ordering, via
    `align_by_nearest_neighbor` (itself already strain-tolerant for the
    same reason: it compares atoms by fractional-coordinate proximity
    through one consistent lattice, not raw Cartesian distance between two
    differently-shaped cells). `origin_shift_frac` is `None` when `sub`
    already shares `reference`'s own origin convention (the common case --
    direct alignment succeeds with no search needed); otherwise it's the
    uniform fractional shift `freeze.search_alignment_shift` found and
    applied, reported so a caller can surface it rather than leaving the
    correction invisible.
    """
    if reference.n_atoms != sub.n_atoms:
        raise DecompositionError(
            f"reference structure has {reference.n_atoms} atoms, distorted structure has "
            f"{sub.n_atoms} -- these must match exactly by this point (stoichiometry/"
            "target-lattice resolution should already have caught a real mismatch)"
        )
    try:
        perm = align_by_nearest_neighbor(
            reference.species, reference.frac_coords, sub.species, sub.frac_coords,
            reference.lattice, tol=tol,
        )
        sub_frac = np.asarray(sub.frac_coords, dtype=np.float64)
        origin_shift = None
        if prefer_smallest_displacement:
            # A DIRECT ALIGNMENT THAT SUCCEEDS IS NOT NECESSARILY THE RIGHT
            # ONE. `align_by_nearest_neighbor` pairs each atom with the
            # closest same-species candidate within `tol`, and at the 0.3 A
            # default a whole COSET offset can still clear that bar -- every
            # atom finds *a* partner, just not its own. Real graphite K4:
            # the joint OPD cell and `tile_onto_supercell` place atoms 1/3
            # apart along a, the direct match succeeded anyway, and the
            # measured displacement came out 0.001006 for an injected
            # 0.001000 -- 0.6% of error injected BEFORE any projection, on
            # a basis that is provably exact (cos 1.000000, condition 1.0).
            #
            # The shift search finds that offset (-1/3, 0, 0) and gives
            # 0.001000, but it only ever ran as a fallback for an alignment
            # that RAISED, so it never got the chance. Here both are
            # computed and the smaller total displacement wins -- the same
            # criterion `_search_origin_shift` already uses to choose among
            # its own candidates.
            #
            # Opt-in, because it costs a second search on every call and
            # every shipped caller is already correct without it.
            best = _smaller_alignment(reference, sub, tol, perm)
            if best is not None:
                origin_shift, perm, sub_frac = best
    except FreezeError as direct_exc:
        origin_shift, perm = search_alignment_shift(reference, sub, tol)
        if origin_shift is None:
            raise DecompositionError(
                f"{direct_exc} (also tried an automatic origin-shift search -- no uniform "
                "translation of the distorted structure produces a full alignment either)"
            ) from direct_exc
        sub_frac = (np.asarray(sub.frac_coords, dtype=np.float64) - origin_shift) % 1.0

    delta_frac = sub_frac[perm] - reference.frac_coords
    delta_frac -= np.round(delta_frac)
    raw_displacement = delta_frac @ reference.lattice
    if origin_shift is not None and not _is_clean_origin_shift(origin_shift):
        raw_displacement, origin_shift = _remove_anchor_contamination(
            raw_displacement, reference, origin_shift)
    return raw_displacement, perm, origin_shift


# ---------------------------------------------------------------------------
# Target lattice / arm resolution
# ---------------------------------------------------------------------------


class _StructureAsIrrep:
    """Duplicate of `cli._StructureAsIrrep` -- decomposition.py must not
    import from cli.py (wrong dependency direction per this project's
    architecture rules), so this tiny (4-attribute) adapter is kept here
    too rather than shared. Adapts a plain `Structure` to the
    `(superlattice, atom_species, atom_positions, n_atoms)` shape
    `tile_onto_supercell` actually reads off an `Irrep`."""

    def __init__(self, structure: Structure):
        self.superlattice = structure.lattice
        self.atom_species = structure.species
        self.atom_positions = structure.cart_coords
        self.n_atoms = structure.n_atoms












"""Largest rotation between `sub` and a candidate arm still treated as the
SAME arm. Half the 60-degree minimum separation between real star arms, so
it can never reach the next one.

Raised from 10.0, which was calibrated on cubic/tetragonal settings where a
correctly-matched arm sits at essentially 0 degrees. That does not hold for a
hexagonal one: real graphite M and L put the CORRECT arm at 10.89 degrees --
the OPD cell's frame is genuinely reoriented from the parent's -- with the
other two at 70.89 and 130.89. The right arm was already being chosen by
`argmin` and then discarded by this guard, failing 8 real (material, k,
irrep) combinations outright."""

"""How far the runner-up must sit beyond the winner for the choice to be
unambiguous. Also half the 60-degree arm separation: two candidates closer
together than this are genuinely indistinguishable and must not be guessed
between -- which is the case the original absolute threshold existed to
catch. Real graphite M separates 10.89 from 70.89, a full 60 degrees clear."""




def _opd_mechanism_token(arm_number: int, key_name: str) -> str:
    """The terse mechanism token the PER-IRREP path reports, from a joint
    cell's own `(label, arm, name)` key.

    The two paths must describe an order parameter identically or a reader
    -- and every test, report and downstream parser -- sees two different
    vocabularies for one thing. The per-irrep path takes this straight from
    `OPD.mechanism` (`native`, `within`, `arm2`, `within(arm2)`,
    `quadrature`, `quad+within`); the joint cell carries only raw pattern
    keys, so the same token is reconstructed here.

    This is the machine-comparable form; reports render the same fact as
    prose.
    """
    within = "__w" in key_name
    quad = "_quad" in key_name
    if quad:
        return "quad+within" if within else "quadrature"
    if arm_number <= 1:
        return "within" if within else "native"
    return f"within(arm{arm_number})" if within else f"arm{arm_number}"




def _opd_sort_key(key):
    """Arm-major, native-first, then within-arm partners, then quadrature --
    ISOTROPY's component order, so a position index here is the
    `--direction` position."""
    _label, arm_number, name = key
    within = re.search(r"__w(\d+)", name)
    return (
        arm_number,
        1 if "_quad" in name else 0,
        int(within.group(1)) if within else 0,
    )


def build_joint_opd_cell(
    irreps: list,
    parent: Structure,
    tol: float = 1e-2,
    findsym_log_text: str | None = None,
    force_joint_multiplier=None,
):
    """ONE joint cell per k-point, hosting every OPD of every irrep there.

    Returns `(reference_structure, patterns, component_count)` where
    `patterns` is `{(irrep_label, arm_number, name): (n_atoms, 3) array}`
    on that one shared cell. Every pattern is constructed from ISOTROPY's
    irrep matrices (`isotropy_basis.build_joint_opd_cell`), so positions are
    ISOTROPY's components in ISOTROPY's order.

    Deliberately ONE cell per k-point, never one across several requested
    k-points: modes at different k carry different Bloch wavevectors and are
    orthogonal anyway, so a cross-k cell would only cost the LCM of every
    k-point's cell. `force_joint_multiplier` builds on a given cell instead
    (decomposition's own cell for `sub`). `tol` is accepted for
    compatibility and unused.
    """
    return isotropy_basis_module.build_joint_opd_cell(
        irreps, parent, isotropy_basis_module.resolve_space_group(irreps, findsym_log_text), findsym_log_text,
        target_multiplier=force_joint_multiplier,
    )


def _hnf_supercells(index: int) -> list:
    """Every distinct supercell SHAPE of the given index, exactly once.

    Hermite Normal Form, UPPER triangular -- the correct convention when a
    lattice matrix's ROWS are its lattice vectors, so that `M @ lattice`
    combines rows. Verified: upper-triangular with these index ranges has
    zero internal duplicates at index 2, 3 and 4 (7, 13 and 35 matrices),
    while the lower-triangular form with the same ranges double-counts
    badly (7, 39 and 140 duplicate pairs) -- it is not an alternative
    convention to fall back on, it is simply wrong here.
    """
    out = []
    for a in range(1, index + 1):
        if index % a:
            continue
        for c in range(1, index // a + 1):
            if (index // a) % c:
                continue
            f = index // (a * c)
            for b in range(c):
                for d in range(f):
                    for e in range(f):
                        out.append(np.array([[a, b, d], [0, c, e], [0, 0, f]], dtype=np.float64))
    return out


def _reduce_lattice_basis(lattice, rounds: int = 8):
    """Greedy Minkowski-style reduction toward the shortest basis.

    Needed because a Gram matrix describes a BASIS, not a lattice: the same
    lattice written in two different bases has two different Grams, so
    comparing them directly answers the wrong question. Real, hit directly:
    HNF's canonical basis for STO's M2- cell is `(a,a,0), (0,2a,0), (0,0,a)`
    with Gram diagonal (30.44, 60.87, 15.22), while ISODISTORT writes the
    IDENTICAL lattice as `(a*sqrt2,0,0), (0,a*sqrt2,0), (0,0,a)` with
    diagonal (30.44, 30.44, 15.22). Reducing both first makes the
    comparison basis-independent.
    """
    reduced = np.array(lattice, dtype=np.float64)
    for _ in range(rounds):
        changed = False
        for i in range(3):
            for j in range(3):
                if i == j:
                    continue
                for k in (-2, -1, 1, 2):
                    candidate = reduced[i] + k * reduced[j]
                    if np.linalg.norm(candidate) < np.linalg.norm(reduced[i]) - 1e-9:
                        reduced[i] = candidate
                        changed = True
        if not changed:
            break
    return reduced


def _lattice_fingerprint(lattice):
    """Basis-independent shape fingerprint: sorted reduced edge lengths and
    sorted |dot products|."""
    reduced = _reduce_lattice_basis(lattice)
    gram = reduced @ reduced.T
    lengths = np.sort(np.sqrt(np.abs(np.diag(gram))))
    angles = np.sort([abs(gram[0, 1]), abs(gram[0, 2]), abs(gram[1, 2])])
    return np.concatenate([lengths, angles])


def align_sub_frame_to_parent(parent: Structure, sub: Structure, tol: float = 1e-2):
    """Rotate `sub` into `parent`'s own Cartesian frame if it isn't already.

    Returns `(aligned_sub, rotation)`, or `(sub, None)` when no rotation is
    needed or none is found -- so any caller that already worked is
    untouched, and no cost is paid.

    **Why.** A structure round-tripped through an external tool arrives in
    THAT tool's standard setting. Real ISODISTORT output writes STO's M2-
    cell axis-aligned with edge `a*sqrt(2) = 5.5168`, where SMODES writes the
    same lattice on the diagonals (`+-a, a, 0`) -- same crystal, same volume,
    same 10 atoms, described 45 degrees apart. Decomposition then failed
    outright, rejecting every arm on the orientation tie-break (measured
    45.00 / 148.60 / 98.42 degrees against a 10 degree threshold) and
    reporting that the structure wasn't built on this irrep, which was false.

    Tiling cannot fix it: in one shared Cartesian frame the lattices are
    genuinely incommensurate (`target @ inv(sub_lattice)` is
    `diag(1.4142, 1.4142, 1)`, a sqrt(2) scale, not an integer matrix). The
    rotation has to come out first.

    **Method** (the physicist's own framing: the sub is N times the parent,
    so enumerate the ways to build an N-times supercell and see which shape
    fits). For index 2 that is three axis doublings plus the sqrt2 x sqrt2 x 1
    cell in each of three planes; `_hnf_supercells` generalises it to any N.
    Each candidate is compared by REDUCED fingerprint, not raw Gram, and the
    recovered transform must be a proper rotation. On real ISODISTORT STO
    M2- this yields exactly 3 fingerprint matches -- the sqrt2 cell in the
    three planes, as predicted -- of which exactly one is orthogonal, giving
    45.00 degrees and an aligned lattice identical to M2-'s own SMODES
    native superlattice.

    Atom fractional coordinates are deliberately unchanged: a rigid rotation
    of the Cartesian frame leaves them invariant.
    """
    parent_lattice = np.asarray(parent.lattice, dtype=np.float64)
    sub_lattice = np.asarray(sub.lattice, dtype=np.float64)

    direct = sub_lattice @ np.linalg.inv(parent_lattice)
    if np.abs(direct - np.round(direct)).max() < tol:
        return sub, None

    ratio = abs(np.linalg.det(sub_lattice) / np.linalg.det(parent_lattice))
    index = int(round(ratio))
    if index < 1 or abs(ratio - index) > 1e-2 or index > 64:
        return sub, None

    target_fingerprint = _lattice_fingerprint(sub_lattice)
    reduced_sub = _reduce_lattice_basis(sub_lattice)
    for matrix in _hnf_supercells(index):
        candidate_lattice = matrix @ parent_lattice
        if not np.allclose(
            _lattice_fingerprint(candidate_lattice), target_fingerprint, rtol=3e-3, atol=3e-2
        ):
            continue
        try:
            rotation = np.linalg.inv(reduced_sub) @ _reduce_lattice_basis(candidate_lattice)
        except np.linalg.LinAlgError:
            continue
        # A genuine frame change is a PROPER rotation. Several candidates can
        # share a fingerprint (the same shape in different planes); requiring
        # orthogonality with positive determinant is what selects the real
        # one -- confirmed on STO M2-, where 3 shapes match and exactly 1 is
        # orthogonal.
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=5e-3):
            continue
        if np.linalg.det(rotation) < 0.0:
            continue
        aligned_lattice = sub_lattice @ rotation
        check = aligned_lattice @ np.linalg.inv(parent_lattice)
        if np.abs(check - np.round(check)).max() > tol:
            continue
        return (
            Structure(
                comment=sub.comment,
                lattice=aligned_lattice,
                species=list(sub.species),
                frac_coords=np.array(sub.frac_coords, dtype=np.float64),
            ),
            rotation,
        )
    return sub, None


def parent_cell_normalisation(parent: Structure, sub: Structure,
                              findsym_log_text: str | None = None):
    """`sqrt(N)`, the divisor that makes a reported amplitude cell-size
    INDEPENDENT -- with `N` the sub's volume in PRIMITIVE parent cells.

    Returns `(sqrt_N, N)`, or `(None, None)` if `N` isn't a clean integer.

    **Why.** Every amplitude this project reports is a Frobenius norm over
    whatever cell the decomposition resolved against, so the SAME physical
    distortion described in a 2x larger cell comes back `sqrt(2)` larger.
    That is the long-documented "a reported amplitude is NOT invariant to
    the size of the cell being decomposed against", and it was confirmed
    against an external reference for the first time here: real CMO's GM1+
    comes back 0.001796 on the four 48-atom subs and 0.002539 on the
    96-atom one -- exactly `sqrt(2)` apart -- while ISODISTORT reports
    0.00162 for both, unchanged. The cells are NOT the difference (ours and
    ISODISTORT's are identical, 48 and 96 atoms respectively, verified); the
    normalisation convention is.

    Dividing by `sqrt(N)` reproduces ISODISTORT's own cell-independent `Ap`
    column. Validated on real STO M2- against ISODISTORT's `Ap = 0.00998`:
    our 10-atom-cell amplitude 0.014123 gives 0.009986, and our 20-atom-cell
    amplitude 0.019960 gives 0.009980 -- the same answer from two different
    cells, both matching ISODISTORT to 3 significant figures.

    `N` counts PRIMITIVE parent cells, not conventional ones. For a centred
    parent the two differ: real CMO is I-centred, so its 24-atom
    conventional cell holds 2 primitive cells, and a 48-atom sub is N = 4,
    not 2. The centring multiplicity comes from the parent's own
    findsym-reported space-group symbol via
    `findsym._STANDARD_CENTERING_TRANSFORMS` (`1 / |det|`; "P" has no entry
    and means 1). Without a findsym log the conventional count is used and
    a centred parent would be under-counted, so the log is preferred.
    """
    volume_ratio = abs(
        np.linalg.det(np.asarray(sub.lattice, dtype=np.float64))
        / np.linalg.det(np.asarray(parent.lattice, dtype=np.float64))
    )
    multiplicity = 1.0
    if findsym_log_text:
        match = re.search(r"Space Group:?\s+\d+\s+\S+\s+(\S+)", findsym_log_text)
        if match:
            letter = match.group(1)[0].upper()
            transform = findsym_module._STANDARD_CENTERING_TRANSFORMS.get(letter)
            if transform is not None:
                determinant = abs(float(np.linalg.det(transform)))
                if determinant > 1e-9:
                    multiplicity = 1.0 / determinant
    n_primitive = volume_ratio * multiplicity
    rounded = int(round(n_primitive))
    # 2% tolerance, not machine epsilon: a real relaxed `sub` is STRAINED
    # relative to the parent, so the volume ratio is never exactly integral.
    # Real CMO comes out at 4.0018 rather than 4, and a 1e-3 cutoff rejected
    # it outright. Strain is physics here, not error -- `compute_strain`
    # already reports it separately.
    if rounded < 1 or abs(n_primitive - rounded) > 0.02 * max(rounded, 1):
        return None, None
    return float(np.sqrt(rounded)), rounded












def _k_point_wide_failure_note(reference_irrep) -> str:
    """Appended to every per-k-point failure message: arm geometry
    (native cell volume, arm count, which arms exist) is a property of
    `(parent, k-vector)` alone -- confirmed directly across every real
    multi-arm fixture in this codebase (STO M2-/M3-/M5-, CMO's own
    X-point irreps all sharing identical native volume/arm count despite
    being different irreps at the same k-point; see the developer notes'
    "arm geometry depends only on (parent, k-vector), never on which
    irrep" finding). This means a resolution failure for `reference_
    irrep` (`irreps[0]`, whichever irrep smodes happened to list first at
    this k-point) is NOT a "this one irrep failed, the others might have
    worked" situation -- every other irrep sharing this same k-vector
    needs the IDENTICAL target lattice geometry, so it would fail
    identically too. Named explicitly so a bare-k-point `decomposition
    --k_vector <K>` failure (which bundles every irrep at that k-point
    into one joint cell) isn't misread as "only the first
    irrep was checked and the rest were silently skipped" -- confirmed
    directly this session as a real point of confusion, not hypothetical.
    """
    return (
        f" -- this applies to EVERY irrep at k = {k_vector_label_from_irrep(reference_irrep.label)} "
        f"(not just {reference_irrep.label}), since arm/target-cell geometry depends only on the "
        "parent structure and k-vector, never on which specific irrep is being considered"
    )




# ---------------------------------------------------------------------------
# Direction sweep
# ---------------------------------------------------------------------------



"""Relative slack before independently-decomposed groups are called
over-claiming. Quadrature over orthogonal groups lands on `norm` to floating
point, so anything above this is a real precondition failure, not rounding."""




_OPD_ORTHOGONAL_TOLERANCE = 1e-6






# ---------------------------------------------------------------------------
# Top-level per-k-vector decomposition
# ---------------------------------------------------------------------------


def isodistort_gauge_factor(irrep, mode_index: int) -> float | None:
    """Issue 9. At GAMMA, a polar irrep's rigid sublattice translation is
    gauge-dependent: ISODISTORT pins the origin on one species, we do not.
    If the frozen mode IS that species' own translation, our gauge moves
    `n_moved` atoms by `d` and reports `sqrt(n_moved)*d`, while ISODISTORT's
    holds them still and moves the other `n_other` by `-d`, reporting
    `sqrt(n_other)*d`. Returns that ratio, `sqrt(n_moved / n_other)`, or
    `None` when the case does not apply.

    Reproduces all three measured cases exactly: real STO GM4- (Sr, 5-atom
    primitive) sqrt(1/4) = 0.5000, real TiO2 GM5- (Ti, 6-atom) sqrt(2/4) =
    0.7071, real CsV3Sb5 GM6- (Cs, 9-atom) sqrt(1/8) = 0.3536.

    **NOT a defect and not a correction to apply.** Adopting ISODISTORT's
    gauge was considered and rejected with the physicist; this only NAMES the
    one place where `Ap` is genuinely not comparable across the two programs,
    so a reader spot-checking a single number does not read a factor of 2.83
    as a broken amplitude.
    """
    k = np.asarray([float(x) for x in irrep.k_vector], dtype=np.float64)
    if not np.allclose(k, 0.0, atol=1e-9):
        return None                      # only Gamma has this freedom
    try:
        mode = irrep.modes[mode_index]
        pattern = np.asarray(mode.displacements, dtype=np.float64)
        species = list(irrep.atom_species)
    except Exception:
        return None
    if pattern.shape[0] != len(species):
        return None
    moved = [i for i, sp in enumerate(species) if sp == mode.species]
    others = [i for i in range(len(species)) if i not in set(moved)]
    if not moved or not others:
        return None
    scale = float(np.max(np.abs(pattern))) or 1.0
    # every atom of the moving species displaced by the SAME vector ...
    reference = pattern[moved[0]]
    if np.linalg.norm(reference) < 1e-12 * scale:
        return None
    if not np.allclose(pattern[moved], reference, atol=1e-6 * scale):
        return None
    # ... and nothing else displaced at all
    if not np.allclose(pattern[others], 0.0, atol=1e-6 * scale):
        return None
    return float(np.sqrt(len(moved) / len(others)))


@dataclass
class ModeAmplitude:
    label: str
    mode_index: int
    species: str
    degeneracy: int
    direction: str
    """The OPD POSITION LETTER ('a', 'b', ...) -- the same letter
    `irrep --opd` prints and `freeze --direction` accepts (ISOTROPY's
    component order), so a direction round-trips without translation."""
    amplitude_angstrom: float
    # Issue 9: sqrt(n_moved / n_other) when this mode is a GAMMA sublattice
    # translation whose amplitude is therefore origin-gauge dependent, else
    # None. Set from `isodistort_gauge_factor`; see it for why this is
    # reported and never applied.
    isodistort_gauge_factor: float | None = None
    mechanism: str = ""
    # Which STAR ARM hosts this direction, so the report can group
    # positions the way `iso` and `irrep --opd` both do (semicolons
    # between arms). 0 means "no arm information" -- the multi-arm blend
    # path does not record one -- and then the flat comma form is used,
    # exactly as before.
    arm: int = 0
    """How the direction was found: 'native', 'armN', 'within', 'quadrature',
    'orbit'. Carried separately BECAUSE the direction letter deliberately
    does not encode it -- a position names a direction regardless of the
    mechanism that reached it. The reporting layer used to infer the kind by
    string-matching the direction name (`n == "native" or
    n.startswith("(")`), which silently stops working once names are
    letters; it reads this instead."""


@dataclass
class KVectorResult:
    k_vector: str
    arm_index: int
    arm_count: int
    multiplier: np.ndarray
    strain: "StrainResult"
    """Computed relative to whichever target lattice actually won arm
    resolution -- deliberately not a single, once-per-structure quantity,
    since a different arm choice means a different (rotated) reference
    lattice, and the strain is only physically meaningful relative to the
    correct one."""
    displacement_norm_angstrom: float
    residual_norm_angstrom: float
    parent_cell_divisor: float | None = None
    """`sqrt(N)` with `N` the resolved cell's volume in PRIMITIVE parent
    cells -- divide any amplitude by this for the cell-size INDEPENDENT
    value, which is what ISODISTORT's own `Ap` column reports. See
    `parent_cell_normalisation`. `None` when `N` isn't a clean integer."""
    parent_cell_count: int | None = None
    amplitudes: list = field(default_factory=list)
    origin_shift_frac: np.ndarray | None = None
    """Set only when `sub` did NOT already share `resolved.structure`'s own
    origin convention -- the uniform fractional-coordinate shift
    `internal_displacement`'s automatic origin-shift search found and
    applied before computing the displacement (see that function's own
    docstring). `None` in the common case (no shift needed)."""
    is_multi_arm_blend: bool = False
    """True when more than one star arm of one irrep came back with
    amplitude -- a multi-arm state."""
    multi_arm_gram_off_diagonal_max: float | None = None
    """Largest off-diagonal magnitude of the design matrix's own Gram
    matrix (`A.T @ A`, `A`'s columns being each blended arm's own
    unit-normalized pattern for one mode), maximized over every mode
    checked -- a diagnostic for how non-orthogonal the blended arms' own
    patterns are to each other; `None` unless `is_multi_arm_blend`. Large
    values don't necessarily mean the fit is wrong (the least-squares
    solve accounts for non-orthogonality correctly either way), just that
    per-arm amplitudes should be read with more care."""






"""Whether `decompose_at_k_vector` tries the joint OPD cell before the
per-irrep path.

ON by default since v1.26.0, with nine known failing tests, enabled at the
physicist's explicit direction after the trade was laid out: the joint cell
fixes real HfO2 Z outright (4 clean / 32 wrong -> 36 clean / 0 wrong) and
restores exact cross-irrep orthogonality (~1e-15 against 2.00e-05), and the
outstanding regressions are small and diagnosed. See the developer notes' known-issues list for the
list. Flip this to False to restore the previous behaviour exactly -- the
per-irrep path is untouched behind it, and is still used automatically
whenever the joint cell cannot be built or `sub` cannot be aligned to it.
"""


def _block_gram_off_diagonal_max(block_patterns: list):
    """Worst |cos| between distinct patterns in one irrep's block, or None.

    The same obliqueness `_joint_block_amplitudes` measures to decide
    whether a joint solve is needed, exposed so the report can carry it.
    Real TiO2 X1 comes back at exactly 1.0 -- two of its four labelled
    components are the same direction -- which is precisely the number a
    reader needs in order to interpret a minimum-norm split.
    """
    if len(block_patterns) < 2:
        return None
    matrix = np.array(
        [np.asarray(pattern, dtype=np.float64).ravel() for pattern in block_patterns]
    )
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(norms > 0):
        return None
    unit = matrix / norms[:, np.newaxis]
    gram = np.abs(unit @ unit.T)
    return float((gram - np.diag(np.diag(gram))).max())


def _joint_block_amplitudes(block_patterns: list, displacement):
    """One irrep's hosted patterns solved together, or None for dot products.

    Returns None when the patterns are already orthogonal: the solve would
    agree with plain dot products to machine precision, so declining keeps
    them bit-identical. Each mode's components are orthonormal by
    construction (checked against ISOTROPY's matrices); the solve matters
    when patterns of DIFFERENT modes of one irrep overlap on the shared cell,
    where independent dot products would double-count.

    Solved per IRREP rather than per k-point deliberately. The two are
    numerically identical -- distinct irreps come back exactly orthogonal
    on this shared cell (measured 0.000000 at every multi-irrep k-point) --
    but a block confines a rank-deficiency to the irrep that has it,
    instead of declining or min-norm-splitting the whole k-point.
    """
    if len(block_patterns) < 2:
        return None
    matrix = np.array(
        [np.asarray(pattern, dtype=np.float64).ravel() for pattern in block_patterns]
    )
    norms = np.linalg.norm(matrix, axis=1)
    if not np.all(norms > 0):
        return None
    unit = matrix / norms[:, np.newaxis]
    gram = unit @ unit.T
    if np.abs(gram - np.diag(np.diag(gram))).max() <= _OPD_ORTHOGONAL_TOLERANCE:
        return None
    # No rank guard: where patterns are linearly dependent, lstsq's
    # minimum-norm solution reconstructs the displacement exactly, and dot
    # products against an oblique basis would double-count instead.
    target = np.asarray(displacement, dtype=np.float64).ravel()
    if target.shape[0] != matrix.shape[1]:
        return None
    solution, *_ = np.linalg.lstsq(matrix.T, target, rcond=None)
    return [float(value) for value in solution]


def _integer_cell_from_metric(parent, sub, rel_tol: float = 0.05, reach: int = 4):
    """Every integer M (rows in parent lattice units) with M G_parent M^T
    matching sub's metric G_sub to within `rel_tol` (strain), det M equal to
    the atom-count ratio, rows in sub's own order. Several can match: they are
    related by symmetries of the LATTICE, which need not be symmetries of the
    crystal (polar R3m BaTiO3 lacks the rhombohedral lattice's inversion), so
    the caller keeps the one the atoms actually align with."""
    found = []
    if parent.n_atoms == 0 or sub.n_atoms % parent.n_atoms:
        return found
    volume = sub.n_atoms // parent.n_atoms
    Gp = np.asarray(parent.lattice, float) @ np.asarray(parent.lattice, float).T
    Ls = np.asarray(sub.lattice, float)
    Gs = Ls @ Ls.T
    vectors = np.array([v for v in itertools.product(range(-reach, reach + 1), repeat=3) if any(v)], float)
    lengths = np.einsum("ij,jk,ik->i", vectors, Gp, vectors)
    rows = [vectors[np.abs(lengths - Gs[i, i]) <= rel_tol * Gs[i, i]] for i in range(3)]
    scale = np.sqrt(np.outer(np.diag(Gs), np.diag(Gs)))
    for a in rows[0]:
        for b in rows[1]:
            if abs(a @ Gp @ b - Gs[0, 1]) > rel_tol * scale[0, 1]:
                continue
            for c in rows[2]:
                if (abs(a @ Gp @ c - Gs[0, 2]) <= rel_tol * scale[0, 2]
                        and abs(b @ Gp @ c - Gs[1, 2]) <= rel_tol * scale[1, 2]):
                    M = np.array([a, b, c])
                    if round(np.linalg.det(M)) == volume:
                        found.append(M.astype(int))
    return found


def _isotropy_cell_for_sub(irreps, parent, sub, findsym_log_text, strain_tol: float = 0.1,
                           align_tol: float = 0.3):
    """v2.0: `(cell, K)` -- the cell to build the OPD basis on for this `sub`,
    and the integer matrix K enlarging `sub` onto it (identity when `sub`'s own
    cell hosts every requested irrep) -- or None for the irreps' own minimal
    cell (then `sub` is enlarged onto it by the generic route).

    `sub`'s own cell, as a multiple of the parent, is found in this order:

    1. RATIONAL rows that are lattice-plus-centring translations of the parent
       (a centred parent's sublattices: real CMO X2+'s own cell has rows like
       (-1/2, 1/2, -1/2)). Checked first because an integer search can find a
       same-shaped WRONG cell for these whose atoms still line up (42% residual).
    2. Every INTEGER cell matching sub's METRIC (rotation-invariant), the plain
       lattice ratio among them, keeping the one that leaves the LEAST
       displacement. A rigid rotation of the frame is physics-free: by a
       symmetry angle it is still an integer lattice relation, and taking it at
       face value relabels the OPD positions (STO M2- at 90 deg: arm 2 read as
       arm 3); the least-displacement cell is the un-rotated correspondence.
    3. None found: enlarge `sub` onto the irreps' own cell (when no larger).

    Building on sub's own cell is what lets a padded, rotated-setting or
    hexagonal-setting `sub` be decomposed directly."""
    sg = isotropy_basis_module.resolve_space_group(irreps, findsym_log_text)
    reps = [isotropy_basis_module.representation(irrep, parent, sg, findsym_log_text) for irrep in irreps]
    raw = np.asarray(sub.lattice, float) @ np.linalg.inv(np.asarray(parent.lattice, float))
    volume_ok = lambda m: abs(abs(np.linalg.det(m)) * parent.n_atoms - sub.n_atoms) < 0.5

    M = None
    rational = np.round(raw * 6) / 6
    # Tight tolerance: a rational ratio is only meaningful when sub shares the
    # parent's axes; for a ROTATED frame almost any matrix sits within 1/12 of
    # the 1/6 grid (and an R-centred parent's thirds then look like centrings).
    if (np.max(np.abs(raw - rational)) <= 0.02 and not np.allclose(rational, np.round(rational))
            and abs(np.linalg.det(rational)) > 1e-6 and volume_ok(rational)):
        try:
            for row in rational:
                isotropy_basis_module.translation_matrix(reps[0], row)
            M = rational
        except isotropy_basis_module.IsotropyBasisError:
            M = None
    if M is None:
        candidates = list(_integer_cell_from_metric(parent, sub))
        plain = np.round(raw)
        if np.max(np.abs(raw - plain)) <= strain_tol and volume_ok(plain):
            # the plain lattice ratio goes FIRST, so it wins every tie: a
            # symmetry-related cell can leave exactly the same displacement norm
            # while flipping the OPD's sign or permuting its positions
            candidates = [plain.astype(int)] + [c for c in candidates if not np.array_equal(plain, c)]
        plain = np.round(raw)

        def row_alignment(m):
            # 0 for a cell whose rows line up one-to-one with the parent's (a
            # positive diagonal); larger for permuted or sign-flipped rows
            m = np.asarray(m, float)
            return float(np.abs(m - np.diag(np.diag(m))).sum() + 2 * np.sum(np.diag(m) < 0))

        scored = []
        for candidate in candidates:
            try:
                displacement, _perm, _shift = internal_displacement(
                    isotropy_basis_module.tiled_parent(parent, candidate), sub, tol=align_tol)
            except DecompositionError:
                continue
            scored.append((float(np.linalg.norm(displacement)), row_alignment(candidate), candidate))
        if scored:
            least = min(norm for norm, _a, _c in scored)
            # among cells leaving the least displacement (ties are symmetry-
            # equivalent readings), take the one whose axes line up with the
            # parent's: sub's fractional coordinates then mean what they say
            tied = [entry for entry in scored if entry[0] <= least * (1 + 1e-6) + 1e-12]
            plain_tied = [entry for entry in tied if np.array_equal(np.asarray(entry[2]), plain)]
            if plain_tied and plain_tied[0][1] > 0 and len(tied) > 1:
                # Agreed frame convention (physicist, 2026-11-03): read sub's
                # Cartesian coordinates as given, in the parent's frame -- and
                # SAY so when its axes are a symmetry-rotated copy of the
                # parent's, since the OPD letters then name the equivalent
                # domain rather than the one in sub's own axes.
                print(
                    "  note: the distorted structure's axes are a symmetry-rotated copy of the parent's; "
                    "its OPD is reported in the PARENT's frame, as the symmetry-equivalent domain "
                    "(same amplitude and space group -- position letters can differ from those in "
                    "the structure's own axes).",
                    file=sys.stderr,
                )
            # Ties are symmetry-equivalent readings of one structure. The PLAIN
            # lattice ratio takes the structure's Cartesian coordinates as
            # given, in the parent's frame (a CONTCAR next to its parent); only
            # when it is not among them (a genuinely rotated setting) does the
            # row-aligned cell decide.
            M = np.asarray((plain_tied[0] if plain_tied else min(tied, key=lambda entry: entry[1]))[2], float)
    if M is None:
        return None

    best = None
    for factors in itertools.product(range(1, 5), repeat=3):
        candidate = np.diag(factors) @ M
        if all(isotropy_basis_module.hosts(rep, candidate) for rep in reps):
            volume = abs(float(np.linalg.det(candidate)))
            if best is None or volume < best[0]:
                best = (volume, candidate, np.diag(factors))
    return None if best is None else (best[1], best[2])


def _decompose_via_joint_opd_cell(
    k_vector, irreps, parent, sub, stoichiometry_multiplier, tol,
    findsym_log_text,
):
    """Decompose against ONE cell per k-point hosting every OPD there.

    The cell is chosen for `sub` by `_isotropy_cell_for_sub` (centred rows,
    integer metric candidates, or an enlargement), the ISOTROPY-matrix basis
    is built on it, `sub` is aligned ONCE (one origin-shift search, one
    strain), and each irrep's amplitudes are solved by least squares against
    the same displacement -- distinct irreps are exactly orthogonal on the
    shared cell, while one irrep's components can be oblique.

    Raises if the cell cannot be built or `sub` cannot be aligned to it.
    """
    target = None
    exact_enlargement = None
    found = _isotropy_cell_for_sub(irreps, parent, sub, findsym_log_text, align_tol=tol)
    if found is not None:
        target, exact_enlargement = found
    reference, patterns, _n_components = build_joint_opd_cell(
        irreps, parent, tol=1e-2, findsym_log_text=findsym_log_text, force_joint_multiplier=target,
    )
    if exact_enlargement is not None and not np.allclose(exact_enlargement, np.eye(3)):
        # v2.0: the integer relation sub -> target is known, so enlarge EXACTLY
        # (the generic route below recovers it through a rotation fit, good to
        # only ~1e-6 A on an oblique cell -- a 0.13% residual on real CMO X2+).
        sub = isotropy_basis_module.enlarge_exactly(sub, exact_enlargement)
        repeats_from_exact = int(round(abs(np.linalg.det(exact_enlargement))))
    else:
        repeats_from_exact = 1

    # ENLARGE `sub` onto the joint cell, and undo OUR OWN repetition.
    #
    # Enlarging repeats the distortion atom-for-atom, so its norm grows by
    # sqrt(repeats) while the patterns stay unit-normalised on the joint
    # cell -- real STO M2- frozen at 0.001 on a 20-atom cell reports
    # 0.001414 against the 40-atom joint cell. Dividing by sqrt(repeats)
    # recovers it exactly: verified 8 of 8 on every case that needs an
    # enlargement here (STO M/X, CMO GM, TiO2 GM, graphite M/L, CsV3Sb5
    # M/L), each an exact periodic repeat and each returning 0.001000000.
    #
    # THIS IS NOT THE REJECTED sqrt(N) CORRECTION, and the distinction is
    # the one the development log itself draws. That correction was rejected
    # because two cases are "genuinely indistinguishable from `sub` alone":
    # a literal periodic repeat (where it is "needed and correct" -- real
    # CMO GM1+, 12 -> 24) and a structure built FRESH at full size (where
    # it is wrong -- real STO GM4- in a combine_modes-built 20-atom joint
    # structure, turned -0.001000 into -0.000500). Nothing is inferred
    # here: THIS function performs the repetition itself, immediately
    # below, so it is case (a) by construction.
    #
    # The rejected regression is also structurally unreachable, not merely
    # avoided: `repeats` is 1 whenever no enlargement happened, so a sub
    # built fresh at the joint cell's own size is multiplied by exactly 1.
    # That is the STO GM4- case, pinned by
    # `TestJointOPDCellAmplitudeScale.test_a_fresh_built_sub_is_never_rescaled`.
    #
    # Only OUR repetition is undone. Whatever convention applied to the
    # user's own `sub` at its own size still applies -- if they hand us an
    # already-padded structure, that padding is theirs and is left alone,
    # exactly as the documented default requires.
    effective_sub = sub
    repeats = repeats_from_exact
    if sub.n_atoms != reference.n_atoms:
        effective_sub = enlarge_onto_lattice(
            sub, reference.lattice, reference_lattice=parent.lattice
        )
        if sub.n_atoms and effective_sub.n_atoms % sub.n_atoms == 0:
            repeats = effective_sub.n_atoms // sub.n_atoms
    repeat_scale = float(np.sqrt(repeats))
    # DO NOT RETRY `prefer_smallest_displacement=True` here. It is exactly
    # the wrong criterion, measured: on real graphite K4 it does find the
    # (-1/3, 0, 0) coset offset and does lower the measured displacement
    # (0.001006174 -> 0.001000370, against an injected 0.001000000), and
    # then the reported amplitude gets WORSE, 0.001005577 -> 0.000872149.
    # A shift can reduce the total norm while pairing atoms with the wrong
    # partners, so "smallest displacement" does not imply "correct
    # alignment". The 0.6% error this was meant to remove is real and still
    # open -- see the developer notes' known-issues list -- but this is not its fix.
    raw_displacement, _perm, origin_shift = internal_displacement(
        reference, effective_sub, tol=tol
    )
    strain = compute_strain(reference.lattice, effective_sub.lattice)

    amplitudes = []
    explained = np.zeros_like(raw_displacement)
    gram_off_diagonal_max = None
    for irrep in irreps:
        keys = sorted(
            (key for key in patterns if key[0] == irrep.label), key=_opd_sort_key
        )
        # Group by mode: every mode of an irrep shares the SAME set of
        # order-parameter directions (that is what "same irrep" means), so
        # positions are numbered within one mode's own key list.
        by_mode = {}
        for key in keys:
            name = key[2]
            mode_index = int(re.match(r"m(\d+)", name).group(1))
            by_mode.setdefault(mode_index, []).append(key)

        # SOLVE THIS IRREP'S COLUMNS JOINTLY, not as independent dot
        # products. A dot product is the expansion coefficient only against
        # an ORTHOGONAL basis; within one irrep the OPDs are routinely
        # oblique -- measured on the joint cell at |cos| 0.912948 for real
        # STO M and 0.825293 for real TiO2 X. Summing them independently
        # double-counts, which is what made this path report one injected
        # amplitude twice at 100% residual.
        #
        # PER-IRREP, not per-k-point, though the two are numerically
        # identical here: distinct irreps come back EXACTLY orthogonal on
        # this shared cell (0.000000 for every multi-irrep k-point
        # measured -- Schur, and the shared grid is what makes it exact
        # rather than approximate). Blocks are preferred because they
        # degrade better: one irrep going rank-deficient on some future
        # material would otherwise decline, or min-norm-split, the entire
        # k-point.
        block = [patterns[key] for key in keys]
        solved = _joint_block_amplitudes(block, raw_displacement)
        # The obliqueness of this irrep's own basis, reported the same way
        # the per-irrep blend path reports it -- a diagnostic a reader needs
        # in order to judge a min-norm split, and previously left as None.
        block_gram = _block_gram_off_diagonal_max(block)
        if block_gram is not None:
            gram_off_diagonal_max = (
                block_gram if gram_off_diagonal_max is None
                else max(gram_off_diagonal_max, block_gram)
            )
        amplitude_by_key = dict(zip(keys, solved)) if solved is not None else None

        for mode_index, mode_keys in sorted(by_mode.items()):
            count = len(mode_keys)
            for position, key in enumerate(mode_keys):
                pattern = patterns[key]
                amplitude = (
                    amplitude_by_key[key] if amplitude_by_key is not None
                    else float(np.sum(pattern * raw_displacement))
                )
                # `explained` accumulates in the JOINT cell's own scale, since
                # it is subtracted from `raw_displacement` to form the
                # residual. The REPORTED amplitude carries the repeat
                # correction instead.
                explained = explained + amplitude * pattern
                amplitude /= repeat_scale
                amplitudes.append(
                    ModeAmplitude(
                        label=irrep.label,
                        mode_index=mode_index,
                        species=irrep.modes[mode_index].species,
                        isodistort_gauge_factor=isodistort_gauge_factor(irrep, mode_index),
                        degeneracy=irrep.degeneracy,
                        # SAME SHAPE AS THE PER-IRREP PATH: a bare position
                        # letter, with provenance in its own `mechanism`
                        # field. This used to be a positional spec with the
                        # provenance folded into the string
                        # ("a,0,0 [arm 1 (SMODES native)]"), leaving
                        # `mechanism` and `arm` empty -- so the two paths
                        # described one order parameter two different ways
                        # and anything parsing a direction broke on the
                        # switch.
                        direction=opd_module.LETTERS[position],
                        amplitude_angstrom=amplitude,
                        mechanism=_opd_mechanism_token(key[1], key[2]),
                        arm=key[1],
                    )
                )

    # BLEND DETECTION IS A COUNT, not a search. The joint cell hosts every arm
    # simultaneously as its own
    # position, so a blend is simply "more than one arm came back with
    # amplitude", already measured. Confirmed on a real STO M2- three-arm
    # blend: injected +0.020000 / -0.010000 / +0.015000 recovered exactly,
    # residual 0.0000% -- the amplitudes were always right here and only
    # this flag was missing, which is what failed seven blend tests.
    #
    # Same two floors the report itself uses (`cli`'s own absolute 1e-7 and
    # 2.5% relative), so "carries amplitude" means the same thing in both.
    blend_arms: dict = {}
    for entry in amplitudes:
        blend_arms.setdefault(entry.label, set())
        peak = max(
            (abs(other.amplitude_angstrom) for other in amplitudes
             if other.label == entry.label),
            default=0.0,
        )
        value = abs(entry.amplitude_angstrom)
        if value >= 1e-7 and (peak <= 0.0 or value / peak >= 0.025):
            blend_arms[entry.label].add(entry.arm)
    is_blend = any(len(arms) > 1 for arms in blend_arms.values())
    # BLEND-ONLY by contract: a non-blend result reports None here, not a
    # near-zero float. `test_gram_off_diagonal_is_none_for_a_non_blend_result`
    # pins that, and it is the right shape -- the number describes how
    # oblique a set of BLENDED components is, so it has no meaning when
    # there is only one.
    if not is_blend:
        gram_off_diagonal_max = None

    residual = raw_displacement - explained
    # `reference` IS the cell every amplitude above was measured on. This
    # read `resolved.structure`, which does not exist in this function --
    # a NameError, and the proof this path had never once executed.
    divisor, n_primitive = parent_cell_normalisation(
        parent, reference, findsym_log_text
    )
    # THE Ap DIVISOR MUST MATCH THE CELL `As` IS REPORTED ON. Every
    # amplitude above was divided by sqrt(repeats) to bring it back to the
    # caller's own cell, so dividing again by the JOINT cell's own
    # primitive count double-corrects: real CMO GM came back at Ap =
    # 0.000500 where the same physics on the un-enlarged cell gives
    # 0.001000, a clean factor of 2. `n_primitive` scales with volume, so
    # the cell `As` actually refers to holds `n_primitive / repeats` of
    # them.
    #
    # This is exactly the failure mode `Ap` exists to prevent -- it is
    # supposed to be cell-size INDEPENDENT -- so getting it wrong here
    # would have been quietly corrosive: the number most likely to be
    # quoted or compared externally, wrong by a whole factor, on a value
    # that looks entirely plausible.
    if repeats > 1 and n_primitive % repeats == 0:
        n_primitive //= repeats
        divisor = float(np.sqrt(n_primitive))
    return KVectorResult(
        parent_cell_divisor=divisor,
        parent_cell_count=n_primitive,
        k_vector=k_vector,
        arm_index=1,
        # STAR SIZE, not how many arms happen to carry amplitude. `blend_arms`
        # answers the second question (it is what `is_multi_arm_blend` needs);
        # `arm_count` has always meant the first.
        arm_count=max((len({k[1] for k in patterns if k[0] == label})
                       for label in {k[0] for k in patterns}), default=1),
        is_multi_arm_blend=is_blend,
        multi_arm_gram_off_diagonal_max=gram_off_diagonal_max,
        multiplier=reference.lattice @ np.linalg.inv(parent.lattice),
        strain=strain,
        # Same repeat correction as the amplitudes, so every reported
        # number is in the caller's own scale. The residual PERCENTAGE is
        # a ratio, so it is unaffected either way.
        displacement_norm_angstrom=float(np.linalg.norm(raw_displacement)) / repeat_scale,
        residual_norm_angstrom=float(np.linalg.norm(residual)) / repeat_scale,
        amplitudes=amplitudes,
        origin_shift_frac=origin_shift,
    )




def decompose_at_k_vector(
    k_vector: str,
    irreps: list,
    parent: Structure,
    sub: Structure,
    stoichiometry_multiplier: int,
    tol: float = 0.3,
    findsym_log_text: str | None = None,
) -> KVectorResult:
    """Decompose `sub` against every irrep at one k-vector, on ONE joint cell
    hosting every OPD of every irrep there (`_decompose_via_joint_opd_cell`),
    built from ISOTROPY's irrep matrices (`isotropy_basis`).

    `findsym_log_text` is the PARENT's log: it supplies the frame carrying
    ISOTROPY's operations onto the parent (`Origin at`, `Vectors a,b,c:`).
    """
    if not irreps:
        raise DecompositionError(f"no irreps given to decompose against at k = {k_vector}")

    # A `sub` from an external tool arrives in that tool's own standard
    # setting, which need not share the parent's axes -- real ISODISTORT
    # output for STO's M2- is the same lattice written 45 degrees round.
    # Fallback only: a sub already commensurate with the parent is returned
    # untouched.
    sub, frame_rotation = align_sub_frame_to_parent(parent, sub, tol=1e-2)
    if frame_rotation is not None:
        angle = float(np.degrees(np.arccos(
            np.clip((np.trace(frame_rotation) - 1.0) / 2.0, -1.0, 1.0)
        )))
        print(
            f"  note: the distorted structure's own axes are rotated {angle:.2f} deg relative to "
            "the parent's -- an external tool's standard setting, not a physical "
            "difference. Rotated into the parent's frame before decomposing; atom "
            "fractional coordinates are unchanged by this.",
            file=sys.stderr,
        )

    try:
        return _decompose_via_joint_opd_cell(
            k_vector, irreps, parent, sub, stoichiometry_multiplier,
            tol, findsym_log_text,
        )
    except Exception as exc:  # noqa: BLE001 -- reported, with the k-point-wide note
        raise DecompositionError(
            f"{k_vector}: the OPD cell could not be used ({type(exc).__name__}: {exc})"
            + _k_point_wide_failure_note(irreps[0])
        ) from exc

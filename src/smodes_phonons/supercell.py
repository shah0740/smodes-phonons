"""Mapping an irrep's SMODES-native supercell onto a human-readable diagonal one.

SMODES picks the *smallest* cell that can host a given k-point's displacement
pattern, which is generally not aligned with the crystallographic axes the
physicist actually recognizes. This module finds the smallest diagonal
supercell (in the original POSCAR's own a/b/c directions) that contains the
SMODES supercell as a whole number of copies, and tiles a mode's displacement
pattern onto it.

Validated numerically against real fixtures (see the plan this was built
from): STO Gamma/M/R give multipliers [1,1,1]/[2,2,1]/[2,2,2], exactly what
each k-vector's own periodicity requires. Ca3Mn2O7 (space group 139, I4/mmm,
body-centered) exposed that the naive "the relationship is already an
integer matrix" assumption breaks for centered space groups -- the
relationship there is *half*-integer, the signature of I-centering -- fixed
by finding the smallest common integer denominator per row via
fractions.Fraction rather than assuming an integer relationship outright.
Re-validated CMO X3- gives [2,2,1] (doubles along a,b; not c) with this
generalization, and it doesn't disturb the simpler STO cases.

Tiling needs no phase corrections: a k-point's superlattice is by
definition the smallest cell where the pattern is exactly periodic, so
filling the bigger cell is just copying the same atoms and the same
displacement vectors at each lattice-offset copy.

**Monoclinic and triclinic cells need more than a diagonal multiplier.**
Their own conventional cell already has an off-diagonal shear (a lattice
angle other than 90 degrees), so a k-point's native supercell generally
isn't a purely diagonal integer multiple of the parent lattice -- confirmed
for HfO2 (P2_1/c): `nice_diagonal_multipliers` genuinely cannot find a
diagonal relationship for its Z/A/B/C points (all real, non-Gamma
irreps), where STO/CMO/TiO2/BaTiO3/graphite's own centered or hexagonal
cells never needed more than the fraction-based generalization above.
Worse, comparing HfO2's parent lattice and SMODES's own superlattice
directly as raw Cartesian matrices doesn't even get as far as "needs
off-diagonal terms" -- the two disagree by an actual rigid ROTATION, not
just a relabeling: SMODES's own Cartesian embedding of a low-symmetry cell
apparently doesn't always share the same axis handedness/orientation as
whatever tool produced the original POSCAR. Confirmed via the metric
tensor (G = L @ L.T, invariant under any rotation of L): HfO2's parent and
superlattice metrics match almost exactly once the right integer relation
is found, even though the raw Cartesian matrices don't come close.

`nice_supercell_matrix` handles both problems together: it tries the fast,
diagonal-only path first (identical to `nice_diagonal_multipliers`, correct
and sufficient for every crystal system tested here except monoclinic/
triclinic), and falls back to `_find_integer_representation`, which
searches for an integer matrix using ONLY the metric tensors of both
lattices -- sidestepping the rotation problem entirely, since lengths and
angles (unlike raw coordinates) don't care how a lattice happens to be
oriented in Cartesian space. `tile_onto_supercell` then recovers the
specific rotation needed (via the Kabsch/orthogonal-Procrustes solution)
and applies it to the irrep's own atom positions before tiling. Validated
against all four of HfO2's real non-Gamma irreps: residuals ~1e-6 after
recovering the rotation, where comparing raw Cartesian matrices directly
was off by 15-35%.
"""

from __future__ import annotations

import itertools
from fractions import Fraction
from functools import reduce
from math import gcd

import numpy as np

from .smodes import Irrep
from .structure import Structure


class SupercellError(ValueError):
    """Raised when a nice supercell can't be found or a tiling is inconsistent."""


def _multi_gcd(*values) -> int:
    return reduce(gcd, (abs(int(v)) for v in values))


def nice_diagonal_multipliers(irrep_superlattice, reference_lattice, max_denominator: int = 24) -> tuple:
    """Smallest (n1, n2, n3) such that diag(n1,n2,n3) @ reference_lattice is an
    integer supercell of irrep_superlattice.

    B = reference_lattice @ inv(irrep_superlattice) expresses the reference
    lattice's own axes in units of the irrep's supercell. Row i of B may be
    rational (not integer -- centered space groups give half-integers, etc.);
    n_i is the smallest positive integer that clears row i's denominators.
    """
    irrep_superlattice = np.asarray(irrep_superlattice, dtype=np.float64)
    reference_lattice = np.asarray(reference_lattice, dtype=np.float64)

    try:
        b = reference_lattice @ np.linalg.inv(irrep_superlattice)
    except np.linalg.LinAlgError as exc:
        raise SupercellError("irrep superlattice is singular") from exc

    multipliers = []
    for row in b:
        fracs = [Fraction(x).limit_denominator(max_denominator) for x in row]
        denom = 1
        for f in fracs:
            denom = denom * f.denominator // gcd(denom, f.denominator)
        row_int = [round(x * denom) for x in row]
        # Relative, not absolute: real (DFT-relaxed, independently-fitted)
        # lattice constants routinely disagree at the 0.1%-ish level between
        # a POSCAR and a separately-generated smodes output for the "same"
        # material -- this must tolerate that while still catching a
        # genuinely different structure (which shows up as a much larger
        # mismatch, not a rounding-sized one).
        residual = max(abs(x * denom - r) for x, r in zip(row, row_int))
        if residual / denom > 1e-2:
            raise SupercellError(
                f"reference lattice row {row.tolist()} is not commensurate with the "
                "irrep superlattice within the allowed denominator -- check that both "
                "describe the same structure"
            )
        g = _multi_gcd(denom, *row_int)
        multipliers.append(denom // g)

    return tuple(multipliers)


def _find_integer_representation_candidates(
    G_ref: np.ndarray, G_target: np.ndarray, max_component: int = 8, tol: float = 2e-2,
    max_candidates: int = 500,
) -> list:
    """Every integer 3x3 matrix M (up to `max_candidates`) such that
    `M @ G_ref @ M.T ~= G_target` (element-wise, within relative tolerance
    `tol`) -- i.e. M @ reference is related to the target lattice by SOME
    rigid rotation, without needing to know that rotation in advance (see
    module docstring for why the metric tensor sidesteps the rotation
    problem entirely). Sorted by full-matrix Gram error (`||M @ G_ref @ M.T
    - G_target||`, relative, smallest first) -- ties (see below) keep their
    search-discovery order.

    Searches row by row: for each target row, tries integer vectors (each
    component in [-max_component, max_component]) whose length and pairwise
    dot products with already-chosen rows (both computed via G_ref, so
    "length"/"angle" here mean the REAL lattice's own metric, not raw
    component values) match the target's own Gram-matrix entries.

    **Explores every row-by-row branch within `tol`, not just one.** This
    matters whenever `tol` is loose enough (e.g. matching a real,
    strained/relaxed structure rather than an exact crystallographic
    relationship) for more than one row-0 (or row-1) candidate to satisfy
    the length/dot-product constraints: a real, confirmed case (BaTiO3's
    own R3m phase) has
    `[-1,-1,0]` and `[1,-1,0]` BOTH leading to a complete, tolerance-
    satisfying, positive-determinant matrix, with EXACTLY (to floating-
    point precision) the same full-matrix Gram error -- an intrinsic,
    Gram-matrix-blind ambiguity, not a search weakness: the two candidate
    lattices are related by a genuine rotation that is NOT itself a
    symmetry of the reference lattice (confirmed directly: `inv(M1) @ M2`
    is not an integer matrix), so no amount of Gram-tensor scoring can ever
    tell them apart -- only one of them happens to already share `sub`'s
    own actual real-space ORIENTATION (confirmed by actually tiling and
    testing atom alignment against real data), which is exactly why this
    returns every tied candidate for the caller to test directly, rather
    than silently picking one. Degrades to exactly a single answer whenever
    only one branch is viable (every already-validated tight-tolerance
    case, e.g. HfO2's non-Gamma k-points), so no already-passing result
    changes for callers that only look at the first.

    For the LAST row specifically, the length + dot-product constraints
    generically leave a 2-fold reflection ambiguity (mirror images through
    the plane of the already-chosen rows) on top of whatever earlier-row
    ambiguity exists; only a positive-determinant completion is ever
    considered, rather than "fixing" a negative determinant by swapping rows
    afterward -- that would silently break the row-to-target-row
    correspondence the search already established (confirmed as a real bug
    during validation: swapping rows on a negative-determinant result gave a
    completely wrong M that happened to still have a valid-looking positive
    determinant).

    `max_candidates` bounds the total number of COMPLETE matrices collected
    across the whole search (not per row) -- generous for every real
    crystallographic case tested (a handful at most), just a safety cap
    against pathological blowup at a very loose `tol`.

    Returns `[]` if no consistent M is found within `max_component`.
    """
    n = G_ref.shape[0]
    candidates_per_component = list(range(-max_component, max_component + 1))
    complete: list = []

    def matches(v, row_idx, chosen):
        target_len_sq = G_target[row_idx, row_idx]
        v = np.asarray(v, dtype=np.float64)
        len_sq = v @ G_ref @ v
        if len_sq < 1e-9:
            return False
        if abs(len_sq - target_len_sq) > tol * max(1.0, target_len_sq):
            return False
        for j, prior in enumerate(chosen):
            target_dot = G_target[row_idx, j]
            dot = v @ G_ref @ np.asarray(prior, dtype=np.float64)
            if abs(dot - target_dot) > tol * max(1.0, abs(target_dot)):
                return False
        return True

    def search(row_idx, chosen):
        if len(complete) >= max_candidates:
            return
        found = []
        for v in itertools.product(candidates_per_component, repeat=n):
            if all(c == 0 for c in v):
                continue
            if matches(v, row_idx, chosen):
                found.append(v)

        if row_idx < n - 1:
            for v in found:
                search(row_idx + 1, chosen + [v])
                if len(complete) >= max_candidates:
                    return
            return

        # Last row: only a positive-determinant completion is a candidate at
        # all -- see docstring for why fixing a negative determinant
        # afterward (e.g. by swapping rows) is wrong.
        for v in found:
            candidate_M = np.array(chosen + [v], dtype=np.float64)
            if np.linalg.det(candidate_M) > 0.5:
                complete.append(candidate_M)
                if len(complete) >= max_candidates:
                    return

    search(0, [])
    if not complete:
        return []

    def full_error(M: np.ndarray) -> float:
        G_candidate = M @ G_ref @ M.T
        return float(np.linalg.norm(G_candidate - G_target) / max(np.linalg.norm(G_target), 1e-12))

    return [M.astype(int) for M in sorted(complete, key=full_error)]


def _find_integer_representation(
    G_ref: np.ndarray, G_target: np.ndarray, max_component: int = 8, tol: float = 2e-2,
    max_candidates: int = 500,
):
    """The single best (smallest full-matrix Gram error) integer matrix
    from `_find_integer_representation_candidates` -- see that function's
    own docstring for the full search description, including why a
    genuine, Gram-matrix-blind tie between multiple equally-valid
    candidates can occur. Every existing caller of this function (`nice_
    supercell_matrix`) only ever needs a single answer and has no
    orientation ambiguity to resolve (an exact crystallographic
    relationship, not a loosely-toleranced match against separately-given
    real data), so picking the first-ranked candidate reproduces this
    function's previous behavior exactly for every already-validated case.
    A caller that DOES need to resolve a genuine tie by testing candidates
    directly should call `_find_integer_representation_candidates` instead.

    Returns None if no consistent M is found within `max_component`.
    """
    candidates = _find_integer_representation_candidates(G_ref, G_target, max_component, tol, max_candidates)
    return candidates[0] if candidates else None


def nice_supercell_matrix(
    irrep_superlattice, reference_lattice, max_denominator: int = 24, max_admixture_component: int = 8,
    fast_path_max_denominator: int = 4,
) -> np.ndarray:
    """Smallest integer 3x3 matrix M such that M @ reference_lattice is,
    up to a rigid rotation, an integer supercell of irrep_superlattice.

    Tries `nice_diagonal_multipliers` first (returned as `diag(n1,n2,n3)`)
    -- correct and sufficient for every crystal system this project has
    real fixtures for except monoclinic/triclinic. Falls back to
    `_find_integer_representation` (metric-tensor-based, handles both the
    genuine off-diagonal shear AND a possible Cartesian-embedding rotation
    mismatch between SMODES's own superlattice and the parent POSCAR --
    see module docstring) only when the fast path fails.

    Unlike `nice_diagonal_multipliers`, which raises `SupercellError` the
    moment a purely diagonal relationship doesn't exist, this is the
    function to use when a genuinely non-diagonal (or rotated) relationship
    might be needed -- `tile_onto_supercell` accepts either this function's
    result or `nice_diagonal_multipliers`'s own, and recovers/applies
    whatever rotation is needed automatically.

    **The fast-path attempt deliberately uses a SMALLER denominator budget
    (`fast_path_max_denominator`, default 4) than `nice_diagonal_multipliers`'s
    own default (24), not the same value.** `Fraction.limit_denominator`
    always returns *some* rational approximation, and for a row that has no
    clean diagonal relationship at all (needs genuine admixture),
    that approximation can land within the 1% relative tolerance purely by
    coincidence at a large, physically meaningless denominator -- confirmed
    for HfO2's C point, where `nice_diagonal_multipliers(..., max_denominator=24)`
    "succeeds" with multiplier (2, 2, 23) (volume ratio 92!) and even
    `max_denominator=6` still gives a spurious (2, 2, 6) (volume ratio 24) --
    only `max_denominator=4` correctly fails and reaches the fallback, which
    finds the true relationship (volume ratio 2, matching
    `irrep.n_atoms / N_reference` exactly). Every genuine diagonal
    relationship in this project's real fixtures needs a denominator of at
    most 2 (confirmed: STO, CMO, TiO2, BaTiO3, graphite, SiO2), so 4 is
    generous for real cases while still well short of where spurious
    coincidences start appearing; the fallback handles anything larger
    correctly regardless; `max_denominator` (still 24 by default) only
    applies if the fallback itself needs it -- currently unused there, kept
    for API symmetry with `nice_diagonal_multipliers`.
    """
    reference_lattice = np.asarray(reference_lattice, dtype=np.float64)
    irrep_superlattice = np.asarray(irrep_superlattice, dtype=np.float64)

    try:
        n = nice_diagonal_multipliers(irrep_superlattice, reference_lattice, fast_path_max_denominator)
        return np.diag(n).astype(int)
    except SupercellError:
        pass

    G_ref = reference_lattice @ reference_lattice.T
    G_target = irrep_superlattice @ irrep_superlattice.T
    M = _find_integer_representation(G_ref, G_target, max_admixture_component)
    if M is None:
        raise SupercellError(
            "no integer supercell matrix (diagonal or otherwise) relates reference_lattice "
            "to irrep_superlattice within the allowed search range -- check that both "
            "describe the same structure"
        )
    return M










def _target_lattice(reference_lattice, multipliers) -> np.ndarray:
    reference_lattice = np.asarray(reference_lattice, dtype=np.float64)
    multipliers = np.asarray(multipliers)
    if multipliers.shape == (3, 3):
        return multipliers.astype(np.float64) @ reference_lattice
    return np.diag(multipliers) @ reference_lattice


def _aligning_rotation(source_lattice: np.ndarray, target_lattice: np.ndarray) -> np.ndarray:
    """Orthogonal rotation R such that source_lattice @ R ~= target_lattice
    (the Kabsch/orthogonal-Procrustes solution) -- used when two lattices
    describe the same abstract structure but were embedded in Cartesian
    space differently by two independent tools (see module docstring:
    confirmed for HfO2, where SMODES's own superlattice needed an actual
    rotation, not just an integer relabeling, to align with the parent
    POSCAR's own embedding).
    """
    U, _, Vt = np.linalg.svd(source_lattice.T @ target_lattice)
    if np.linalg.det(U @ Vt) < 0:
        U = U.copy()
        U[:, -1] *= -1
    return U @ Vt


def _integer_tiling_matrix(target_lattice, irrep_superlattice) -> np.ndarray:
    m = target_lattice @ np.linalg.inv(np.asarray(irrep_superlattice, dtype=np.float64))
    m_int = np.round(m).astype(int)
    # Relative tolerance, same reasoning as nice_diagonal_multipliers: real,
    # independently-sourced fixtures for "the same" material can disagree by
    # ~0.1% in lattice constant.
    if not np.allclose(m, m_int, rtol=1e-2, atol=1e-2):
        raise SupercellError(
            "target diagonal lattice is not an integer supercell of the irrep's own "
            "superlattice -- this should never happen for a real SMODES result, "
            "something upstream is inconsistent"
        )
    return m_int


def _coset_offsets(m_int: np.ndarray) -> list:
    """det(M) integer offset vectors (in the irrep superlattice's own units),
    one per coset of Z^3 / (M @ Z^3) -- found by an expanding brute-force
    search, deduped by fractional position within the target cell.

    THE KEY MUST BE ROUNDED BEFORE THE FINAL WRAP, NOT AFTER. `x % 1.0`
    returns `1.0` for an `x` a hair below 1 (and for a hair-below-zero `x`:
    `(-1e-16) % 1.0` is `1.0`, not `0.0`), so rounding afterwards yields
    `1.0` where the true coset coordinate is `0.0` -- two different keys for
    the SAME coset. `seen` then holds more than `det_m` entries and the
    `[:det_m]` truncation keeps two representatives of one coset while
    dropping a real one entirely. The tiled structure comes out with the
    right atom COUNT and duplicate atoms in it, which is why this survived:
    nothing downstream counts distinct positions.

    Invisible for a multiplier whose inverse has only halves or quarters --
    exactly representable in binary, so the boundary is never approached
    inexactly. It bites when `m_inv` carries THIRDS, which is what an
    R-centred parent produces. Confirmed on real BaTiO3 L1 (`det(M) = 6`):
    offsets keyed `(0.5, 1.0, 0.5)` and `(0.5, 0.0, 0.5)` are one coset, so
    a 60-atom tiling contained only 50 distinct sites. Downstream, arm
    alignment then failed outright -- a one-to-one match is impossible when
    the target has fewer distinct sites than atoms -- surfacing as a
    "no matching Ba atom" error two steps away from the cause.

    Rounding first and wrapping second (`round(x, 6) % 1.0`) sends
    `0.9999999 -> 1.0 -> 0.0` and `-0.0 -> 0.0`, so each coset has exactly
    one key.
    """
    det_m = round(abs(np.linalg.det(m_int)))
    if det_m == 0:
        raise SupercellError("tiling matrix is singular")
    m_inv = np.linalg.inv(m_int)

    bound = 1
    while True:
        seen = {}
        for i in range(-bound, bound + 1):
            for j in range(-bound, bound + 1):
                for k in range(-bound, bound + 1):
                    v = np.array([i, j, k])
                    key = tuple(np.mod(np.round((v @ m_inv) % 1.0, 6), 1.0))
                    if key not in seen:
                        seen[key] = v
        if len(seen) > det_m:
            # Cannot happen once the key is canonical: there are exactly
            # det_m cosets. Raise rather than truncate -- truncation is what
            # silently produced duplicate atoms before.
            raise SupercellError(
                f"found {len(seen)} distinct cosets for a tiling matrix of determinant "
                f"{det_m} -- the coset key is not canonical, which would silently produce "
                "a supercell containing duplicate atoms"
            )
        if len(seen) == det_m:
            return list(seen.values())
        bound *= 2


def _atom_tiling_order(irrep: Irrep, reference_lattice, multipliers):
    """The deterministic (offset, native atom index) ordering every
    supercell-writing command's tiled structure follows -- factored out of
    `tile_onto_supercell`.

    Returns `(target_lattice, irrep_superlattice, native_atom_order,
    offset_vectors)`, both of the last two length-matched to the final
    tiled structure's own atom count: row `i` of the tiled structure is
    native atom `native_atom_order[i]`, translated by the (still-fractional,
    in units of `irrep_superlattice`) lattice vector `offset_vectors[i]`.
    `irrep_superlattice` is returned alongside since it may differ from
    `irrep.superlattice` itself when the rotation-recovery fallback below
    fires -- a caller that also needs Cartesian positions (`tile_onto_
    supercell`) needs this SAME possibly-rotated version to convert
    `offset_vectors[i] @ irrep_superlattice` correctly.
    """
    target_lattice = _target_lattice(reference_lattice, multipliers)
    irrep_superlattice = np.asarray(irrep.superlattice, dtype=np.float64)

    try:
        m_int = _integer_tiling_matrix(target_lattice, irrep_superlattice)
    except SupercellError:
        # Fall back to rotation recovery -- see module docstring. Only
        # triggers when the fast, no-rotation path fails (every crystal
        # system tested here except monoclinic/triclinic uses the fast
        # path and never reaches this branch). Confirmed for HfO2: SMODES's
        # own superlattice can be embedded in Cartesian space with a
        # genuinely different rotation than the parent POSCAR, even when
        # `multipliers` (from `nice_supercell_matrix`) already correctly
        # relates the two lattices up to that rotation.
        R = _aligning_rotation(irrep_superlattice, target_lattice)
        try:
            m_int = _integer_tiling_matrix(target_lattice, irrep_superlattice @ R)
            irrep_superlattice = irrep_superlattice @ R
        except SupercellError:
            # Kabsch above aligns the superlattice ONTO the target, which is
            # only meaningful when the two are the same lattice up to a
            # rotation -- i.e. when `multipliers` came from
            # `nice_supercell_matrix` and both sides have the same volume.
            # That is the case every previously-validated HfO2 result used.
            #
            # It breaks when the target is a genuine SUPERCELL of the native
            # cell: real HfO2's joint path passes diag(2,2,1) (det 4) against
            # a det-2 native cell, so the two differ by volume as well as
            # orientation and no rotation can align them. Kabsch then returns
            # a meaningless R and the retry fails, which is why all four of
            # HfO2's zone-boundary k-points could not be built at all.
            #
            # Derive the rotation from the SAME-VOLUME relationship instead --
            # `nice_supercell_matrix` already finds it, rotation-invariantly,
            # via its Gram-tensor search -- and only then apply it to the
            # larger target. Second fallback only: it runs after both the
            # direct path and Kabsch have failed, so no case that already
            # worked can reach it.
            same_volume = nice_supercell_matrix(irrep_superlattice, reference_lattice)
            R = _aligning_rotation(
                irrep_superlattice,
                np.asarray(same_volume, dtype=np.float64) @ np.asarray(reference_lattice, dtype=np.float64),
            )
            irrep_superlattice = irrep_superlattice @ R
            m_int = _integer_tiling_matrix(target_lattice, irrep_superlattice)

    offsets = _coset_offsets(m_int)

    # (species, native_atom_idx, offset) per pre-sort row, in [offset-major,
    # native-atom-minor] order.
    rows = [
        (irrep.atom_species[atom_idx], atom_idx, v) for v in offsets for atom_idx in range(irrep.n_atoms)
    ]
    # POSCAR requires same-species atoms contiguous; naive tiling interleaves
    # them (one full native-cell copy at a time), so group by species,
    # preserving the native cell's own first-appearance order. Python's sort
    # is stable, so within one species the original [offset-major,
    # atom-minor] relative order survives.
    species_order = list(dict.fromkeys(irrep.atom_species))
    rows.sort(key=lambda r: species_order.index(r[0]))
    native_atom_order = [atom_idx for _species, atom_idx, _v in rows]
    offset_vectors = [v for _species, _atom_idx, v in rows]

    return target_lattice, irrep_superlattice, native_atom_order, offset_vectors


def tile_onto_supercell(
    irrep: Irrep, reference_lattice, multipliers, displacement_sets: dict, comment: str
) -> tuple:
    """Build the undistorted nice-diagonal-cell Structure, and tile each named
    set of per-mode displacement patterns onto it.

    displacement_sets: {name: list[(n_atoms, 3) ndarray]}, one array per mode,
    all in the irrep's own native atom ordering (e.g. result.atomic_displacement_eigenvectors).

    Returns (undistorted_structure, {name: list[(n_atoms_big, 3) ndarray]}),
    with each returned displacement array re-normalized to unit Frobenius
    norm (the same convention used everywhere else in this codebase for mode
    basis vectors).
    """
    target_lattice, irrep_superlattice, native_atom_order, offset_vectors = _atom_tiling_order(
        irrep, reference_lattice, multipliers
    )
    atom_positions = np.asarray(irrep.atom_positions, dtype=np.float64)
    displacement_sets = {name: [np.asarray(m, dtype=np.float64) for m in modes] for name, modes in displacement_sets.items()}

    if not np.array_equal(irrep_superlattice, np.asarray(irrep.superlattice, dtype=np.float64)):
        # _atom_tiling_order hit the rotation-recovery fallback -- apply the
        # same rotation to atom positions and displacement patterns (never
        # to the ordering itself, which doesn't depend on orientation).
        #
        # FIXED: a real, confirmed bug -- `irrep_superlattice` here already
        # IS `irrep.superlattice @ R` (that's exactly what `_atom_tiling_
        # order`'s own rotation-recovery fallback computed), so recovering
        # R from the two matrices needs `R = inv(irrep.superlattice) @
        # irrep_superlattice` (left-multiply, matching this codebase's own
        # row-vector convention `A @ R = B` established in `_aligning_
        # rotation`'s own docstring) -- NOT `irrep_superlattice @ inv(irrep.
        # superlattice)` (right-multiply), which is a SIMILARITY TRANSFORM
        # of R by irrep.superlattice, not R itself, and is only equal to R
        # when irrep.superlattice happens to commute with R (true for some
        # simple axis-permutation-type rotations, e.g. TiO2's real GM5+/
        # HfO2 fixtures already validated elsewhere -- confirmed those
        # stay byte-identical after this fix, see the regression run this
        # fix shipped with). For a genuinely oblique lattice (confirmed
        # directly for BaTiO3's real rhombohedral primitive cell, R3m,
        # tiling F1 onto a det=2 supercell of the true 5-atom primitive
        # parent) the two formulas diverge completely -- the old, wrong
        # formula produced a structure that findsym reports as P1 (no
        # symmetry at all) for every one of 6 independently-tried
        # Gram-tensor-tied candidate matrices, even though F1's own native
        # cell is a genuine, full-symmetry space-group-160 (R3m) structure
        # on its own; the corrected formula recovers exactly that same
        # space group (160) once properly re-embedded.
        R = np.linalg.inv(irrep.superlattice) @ irrep_superlattice
        atom_positions = atom_positions @ R
        displacement_sets = {name: [m @ R for m in modes] for name, modes in displacement_sets.items()}

    n_modes_by_set = {name: len(modes) for name, modes in displacement_sets.items()}

    # FIXED: atom positions are now converted to fractional coordinates in
    # SMODES's OWN lattice frame first (`atom_positions @ inv(irrep_
    # superlattice)` -- both quantities parsed from the SAME SMODES text
    # output, sharing the same limited decimal precision, so their RATIO
    # cancels most of that rounding rather than compounding it), and only
    # THEN moved into `target_lattice`'s own frame via the exact INTEGER
    # matrix `m_int` (`target_lattice = m_int @ irrep_superlattice`,
    # already found above by `_atom_tiling_order`/`_integer_tiling_matrix`)
    # -- never by dividing SMODES's own Cartesian numbers directly by the
    # PARENT POSCAR's own, independently-rounded absolute lattice constant.
    # That direct division was a real, confirmed source of small but
    # nonzero numerical noise on every generated structure: real BaTiO3
    # GM4- data showed atoms whose true fractional coordinate is EXACTLY
    # 0.5 or 0.0 (a symmetry requirement) coming out as `0.5000001`/
    # `0.0000001` instead -- confirmed to be a single, uniform
    # multiplicative scale factor (not rotation/alignment noise: the
    # residual on every atom was explained EXACTLY, to 15+ significant
    # figures, by one scalar ratio), consistent with SMODES's own printed
    # lattice constant differing from the parent's true value in its last
    # few decimal places. This was always harmless for every downstream
    # symmetry check (orders of magnitude below even the tightest findsym
    # tolerance used anywhere in this codebase) but visibly wrong to a
    # physicist reading a written POSCAR directly, especially on an axis
    # with NO real physical displacement, where this artifact was the
    # ONLY thing occupying that coordinate. `m_int` is an exact integer
    # matrix, so this transform introduces no new lattice-constant-scale
    # error at all -- confirmed directly: the same real BaTiO3 GM4- atoms
    # come out EXACTLY 0.5/0.0 with this fix, not just closer to it.
    m_int = _integer_tiling_matrix(target_lattice, irrep_superlattice)
    atom_frac_native = atom_positions @ np.linalg.inv(irrep_superlattice)

    big_species = [irrep.atom_species[atom_idx] for atom_idx in native_atom_order]
    big_frac_native = np.array(
        [
            atom_frac_native[atom_idx] + v
            for atom_idx, v in zip(native_atom_order, offset_vectors)
        ]
    )
    big_frac = (big_frac_native @ np.linalg.inv(m_int)) % 1.0

    undistorted = Structure(
        comment=comment, lattice=target_lattice, species=big_species, frac_coords=big_frac
    )

    big_displacement_sets = {}
    for name, modes in displacement_sets.items():
        tiled_modes = []
        for mode_idx in range(n_modes_by_set[name]):
            arr = np.array([modes[mode_idx][atom_idx] for atom_idx in native_atom_order])
            norm = np.linalg.norm(arr)
            if norm == 0:
                raise SupercellError(f"{name} mode {mode_idx}: tiled displacement is exactly zero")
            tiled_modes.append(arr / norm)
        big_displacement_sets[name] = tiled_modes

    return undistorted, big_displacement_sets




def map_parent_values_to_native(irrep: Irrep, parent: Structure, parent_values, tol: float = 0.3) -> list:
    """Given a per-atom scalar (e.g. a magnetic moment) in `parent`'s own
    POSCAR atom order, return the same values reordered/duplicated into
    `irrep.atom_species`/`atom_positions`'s own native-cell order -- what
    `irrep --magmom` writes for each mode's native cell.

    Confirmed empirically (not just assumed) that this can be a plain
    Cartesian fold-and-match against `parent.lattice` directly, with NO
    rotation correction needed at this step: `irrep.atom_positions`/
    `irrep.superlattice` are parsed straight from SMODES's own text output,
    which describes the SAME real-space structure as the findsym log
    (and, in turn, the parent POSCAR) it was generated from -- there is no
    extra ad-hoc rotation introduced by SMODES itself the way there can be
    between its native cell's SHAPE and some separately-built target
    lattice (`tile_onto_supercell`'s own rotation-recovery fallback, a
    genuinely different problem: reconciling `irrep.superlattice`'s own
    shape against a diagonal multiple of `reference_lattice`, not relating
    `irrep.atom_positions` back to `parent`'s own atoms in the first
    place). Verified directly against real CMO X3- data -- whose own
    native superlattice IS a genuine rotation relative to the parent's
    axes (see the developer notes) -- folding native atom positions into the
    parent's own fractional frame lands within 1e-4 A of the true parent
    atom, not a large, rotation-sized residual.

    For a k-point whose native cell is a genuine multiple of the parent
    (e.g. STO's R-point, 2x, before any further `tile_onto_supercell`
    coset tiling even happens), several native atoms correctly map to the
    SAME parent atom -- each is a distinct periodic image of it, and
    physically must carry the identical value.

    Raises `SupercellError` naming the unmatched native atom if no
    same-species parent atom is found within `tol` Angstrom (even after
    the origin-shift fallback below) -- mirrors `freeze.align_by_nearest_
    neighbor`'s own error contract, but allows reuse (a `Structure`-
    comparison permutation must be 1:1; this mapping is deliberately
    many-to-one).

    **FIXED: a genuine origin-convention mismatch between `irrep.atom_
    positions` and `parent`'s own atoms is now searched for and corrected,
    not just assumed away.** This function's own zero-shift assumption
    (direct Cartesian fold-and-match, no origin correction) was validated
    against real CMO X3- data and confirmed to hold there -- but a real
    HPC bug report (Born effective charges from a Gamma-point DFPT run on
    TiO2's real 12-atom conventional parent, needing reduction onto a
    Gamma-point irrep's own smaller 6-atom native cell) surfaced a case
    where it does NOT hold: TiO2's own GM3-/GM5- native atom
    reconstruction sits at a real, nonzero, fixed fractional shift
    relative to the parent POSCAR's own atoms (confirmed directly: a
    shift of `(-0.5, 0.25, 0.125)` reconciles every one of GM3-'s 6 native
    atoms to a distinct parent atom, residual ~1e-5 A -- a genuine origin
    convention difference, not a coincidence or numerical noise; the
    UNSHIFTED direct match fails at ~1.5 A for the very first atom, far
    beyond any noise floor). This is the SAME class of origin-shift
    mismatch already solved elsewhere in this codebase for structurally
    different reasons (`freeze.search_alignment_shift`, for a 1:1 atom
    count) -- but that function's own `align_by_nearest_neighbor` call
    requires a BIJECTIVE permutation, which does not apply here (this
    mapping is deliberately many-to-one, e.g. STO's own R-point case
    documented above), so a dedicated, many-to-one-aware search is used
    instead, applying the identical underlying principle: try the direct
    (zero-shift) match FIRST (unchanged, zero cost, for every already-
    validated case, e.g. CMO X3-), and only search for a shift when that
    fails for at least one atom. Candidate shifts are derived directly
    from the data (anchor on `irrep`'s own fewest-count species, try every
    same-species PARENT atom as that first native atom's true partner --
    each pairing implies exactly one candidate shift), never a hardcoded
    "common" value -- the same data-derived principle `search_alignment_
    shift` already established, just re-implemented for a many-to-one
    mapping. Among every candidate giving a FULL alignment (every native
    atom matched within `tol`), the one with the smallest total residual
    wins.
    """
    parent_values = list(parent_values)
    if len(parent_values) != parent.n_atoms:
        raise SupercellError(
            f"map_parent_values_to_native: {len(parent_values)} parent values given, but the "
            f"parent structure has {parent.n_atoms} atoms"
        )
    lattice = np.asarray(parent.lattice, dtype=np.float64)
    parent_frac = np.asarray(parent.frac_coords, dtype=np.float64)
    native_frac_in_parent = (np.asarray(irrep.atom_positions, dtype=np.float64) @ np.linalg.inv(lattice)) % 1.0

    def try_match(shift: np.ndarray):
        shifted = (native_frac_in_parent + shift) % 1.0
        matches = []
        total = 0.0
        for i in range(irrep.n_atoms):
            species = irrep.atom_species[i]
            best_j, best_dist = None, None
            for j in range(parent.n_atoms):
                if parent.species[j] != species:
                    continue
                delta = parent_frac[j] - shifted[i]
                delta -= np.round(delta)
                dist = float(np.linalg.norm(delta @ lattice))
                if best_dist is None or dist < best_dist:
                    best_dist, best_j = dist, j
            if best_j is None or best_dist > tol:
                return None
            matches.append(best_j)
            total += best_dist
        return matches, total

    direct = try_match(np.zeros(3))
    if direct is not None:
        matches, _total = direct
        return [parent_values[j] for j in matches]

    species_counts: dict = {}
    for s in irrep.atom_species:
        species_counts[s] = species_counts.get(s, 0) + 1
    anchor_species = min(species_counts, key=species_counts.get)
    anchor_index = irrep.atom_species.index(anchor_species)
    anchor_native_frac = native_frac_in_parent[anchor_index]
    candidate_js = [j for j in range(parent.n_atoms) if parent.species[j] == anchor_species]

    best_matches, best_total = None, None
    for j in candidate_js:
        shift = parent_frac[j] - anchor_native_frac
        shift -= np.round(shift)
        result = try_match(shift)
        if result is None:
            continue
        matches, total = result
        if best_total is None or total < best_total:
            best_matches, best_total = matches, total

    if best_matches is None:
        raise SupercellError(
            f"map_parent_values_to_native: no matching parent atom found for one or more of "
            f"{irrep.label}'s own native atoms within {tol} A, even after searching for a data-"
            "derived origin shift -- check that the parent structure and this irrep describe the "
            "same material"
        )
    return [parent_values[j] for j in best_matches]


class _StructureAsIrrep:
    """Adapts a plain `Structure` to the `(superlattice, atom_species,
    atom_positions, n_atoms)` shape `nice_supercell_matrix`/
    `tile_onto_supercell` actually read off an `Irrep` -- letting
    `enlarge_onto_lattice` (and any other caller) reuse that exact,
    already-validated tiling machinery on an arbitrary `Structure`, rather
    than duplicating it. Neither function reads anything else off `irrep`
    (no `.label`, no mode data), so this is a complete adapter for both.
    """

    def __init__(self, structure: Structure):
        self.superlattice = structure.lattice
        self.atom_species = structure.species
        self.atom_positions = structure.cart_coords
        self.n_atoms = structure.n_atoms


def enlarge_onto_lattice(small: Structure, target_lattice, reference_lattice=None) -> Structure:
    """Periodically repeat `small`'s own real atoms onto `target_lattice`
    (a bigger, commensurate lattice) and return the resulting `Structure`.

    **`reference_lattice` makes this work for a genuinely STRAINED `small`**
    -- the real, hit-in-practice case of a DFT-relaxed `decomposition --sub`,
    whose cell shape differs from the parent's by a real physical strain
    rather than only by an integer repetition. Without it, the direct path
    below matches `small`'s own ABSOLUTE lattice against `target_lattice`,
    which is built from the (unstrained) parent -- so a real relaxation of
    even a couple of percent puts the Gram match outside tolerance and this
    raises `SupercellError`, blocking the whole decomposition. Confirmed
    against real BaTiO3 data: a 2% tetragonal strain (a=b=c=3.98610 parent
    vs. a=c=3.97448, b=4.07213 sub) is enough, and it fails for every
    non-Gamma k-point at once since they all need an enlarged sub.

    The fix rests on the multiplier being a property of `(parent, k-vector)`
    ALONE, never of the strained sub: when `reference_lattice` is given, the
    integer multiplier is derived by relating `target_lattice` to it (both
    unstrained, so they match cleanly) and then applied to `small`'s OWN
    lattice, so the enlarged cell keeps `small`'s real strain instead of
    being forced onto the parent's cell shape. That is the physically
    correct thing to do anyway: `internal_displacement` is exactly
    strain-invariant by construction (fractional-coordinate differencing),
    and `compute_strain` reports the strain separately as its own
    first-class output -- so preserving it here costs the projection
    nothing and keeps the reported strain honest.

    Tried only as a FALLBACK, after the direct path already failed, so
    every already-validated unstrained case takes the identical code path
    it always did.

    A general "is `small` a genuinely reduced/smaller version of a bigger
    reference lattice" primitive -- factored out of what was originally
    `cli.py`'s own `_try_enlarge_sub_to_match_parent`/`_StructureAsIrrep`
    pair (built for reconciling a `decomposition --sub` against a `--parent`
    that wasn't given in its own most-reduced form). `_try_enlarge_sub_to_match_parent`'s
    own docstring, and the developer notes' writeup of this fix, cover why
    ENLARGING the smaller structure (never reducing/re-deriving the bigger
    one) is the safe direction: reducing a structure needs real symmetry
    detection (findsym) on a structure whose actual displacement isn't
    known yet in the blend-candidate case (the amplitudes are exactly what
    the caller is trying to solve for), while enlarging `small` is a purely
    geometric "repeat these already-known real atoms" operation with no
    such circularity.

    Reuses `nice_supercell_matrix` (rotation-aware, so `small`'s own
    embedding need not already share `target_lattice`'s axis convention)
    and `tile_onto_supercell` (with an empty displacement-sets dict --
    matching `_try_enlarge_sub_to_match_parent`'s own "just tile the atoms,
    no pattern needed" convention, since `tile_onto_supercell` rejects an
    all-zero displacement outright but is perfectly happy tiling zero
    displacement SETS at all).

    Raises `SupercellError` (never silently returns something wrong) if no
    valid integer relationship between `small.lattice` and `target_lattice`
    exists at all -- the caller decides what "no relationship found" means
    for its own purpose (a genuine mismatch vs. just "this enlargement
    doesn't apply here").
    """
    try:
        multipliers = nice_supercell_matrix(small.lattice, target_lattice)
    except SupercellError:
        if reference_lattice is None:
            raise
        # Strain-tolerant fallback: the multiplier is a property of the
        # (unstrained) reference/target pair, never of `small` itself, so
        # derive it there and apply it to `small`'s own cell -- preserving
        # whatever real strain `small` carries. See the docstring above.
        multipliers = nice_supercell_matrix(target_lattice, reference_lattice)
        enlarged, _disp_sets = tile_onto_supercell(
            _StructureAsIrrep(small), small.lattice, multipliers, {},
            comment=f"{small.comment} enlarged",
        )
        return enlarged
    enlarged, _disp_sets = tile_onto_supercell(
        _StructureAsIrrep(small), target_lattice, multipliers, {},
        comment=f"{small.comment} enlarged",
    )
    return enlarged

"""Atom alignment between independently built structures, and an OPD-basis
quality measure.

Structures that describe the same physical atoms -- a pattern tiled onto a
cell, a `sub` from DFT or ISODISTORT, two tilings of one parent -- do not
produce identical fractional coordinates (small numerical noise, ~2 mA on real
STO), and may differ by an origin choice. `align_by_nearest_neighbor` matches
atoms by species under the minimum-image convention, with a tolerance well
above that noise and well below a bond length; `search_alignment_shift` finds a
single uniform translation where direct matching fails.

The freeze-family commands themselves build their structures from the
ISOTROPY-matrix basis (`isotropy_basis`, via `cli`); the v1.x joint-cell
tiling that used to live here is recorded in the v1.x development record.
"""

from __future__ import annotations


import numpy as np

from .structure import Structure


class FreezeError(ValueError):
    """Raised when requested modes can't be combined onto a shared supercell."""






def align_by_nearest_neighbor(
    reference_species: list, reference_frac, other_species: list, other_frac, lattice, tol: float = 0.3
) -> list:
    """Species-matched nearest-neighbor permutation (minimum-image convention)
    so that `other`'s rows can be reindexed to match `reference`'s ordering:
    `other_frac[perm[i]]` is the atom corresponding to `reference_frac[i]`.

    Raises FreezeError naming the unmatched atom if no same-species candidate
    is within `tol` Angstrom -- this should only fire if the two structures
    don't actually describe the same crystal.
    """
    lattice = np.asarray(lattice, dtype=np.float64)
    reference_frac = np.asarray(reference_frac, dtype=np.float64)
    other_frac = np.asarray(other_frac, dtype=np.float64)

    n = len(reference_species)
    if len(other_species) != n:
        raise FreezeError(
            f"atom count mismatch: reference has {n} atoms, other has {len(other_species)}"
        )

    used = set()
    perm = []
    for i in range(n):
        best_j = None
        best_dist = None
        for j in range(n):
            if j in used or other_species[j] != reference_species[i]:
                continue
            delta_frac = other_frac[j] - reference_frac[i]
            delta_frac -= np.round(delta_frac)
            dist = float(np.linalg.norm(delta_frac @ lattice))
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j is None or best_dist > tol:
            raise FreezeError(
                f"no matching {reference_species[i]!r} atom found for reference atom {i} "
                f"within {tol} A (closest candidate: {best_dist} A) -- check that all "
                "requested irreps describe the same structure"
            )
        used.add(best_j)
        perm.append(best_j)
    return perm






# Denominators of the low-order rational fractions a real crystallographic
# origin/coset shift is always built from (see `search_alignment_shift`'s
# own comment on `_snap_to_nice_fraction` for the full rationale and real
# examples -- 0.5, and TiO2's real (0.5, 0.75, 0.875)).
_SHIFT_SNAP_DENOMINATORS = (1, 2, 3, 4, 6, 8, 12)
# Deliberately tight -- see `search_alignment_shift`'s own comment for why
# this must NOT be loose enough to touch a deliberately-not-nice test shift
# like (0.13, 0.27, 0.61).
_SHIFT_SNAP_TOLERANCE = 1e-3


def _snap_to_nice_fraction(value: float, tolerance: float) -> float:
    """Round `value` to the nearest low-denominator fraction
    (`_SHIFT_SNAP_DENOMINATORS`) if one is within `tolerance` -- otherwise
    returns `value` unchanged."""
    best = value
    best_diff = None
    for denominator in _SHIFT_SNAP_DENOMINATORS:
        candidate = round(value * denominator) / denominator
        diff = abs(candidate - value)
        if diff <= tolerance and (best_diff is None or diff < best_diff):
            best, best_diff = candidate, diff
    return best


def search_alignment_shift(reference: Structure, sub: Structure, tol: float):
    """Find a uniform fractional-coordinate shift `t` such that
    `sub.frac_coords - t` (mod 1) aligns against `reference` --
    `align_by_nearest_neighbor` compares raw fractional coordinates
    directly, which silently assumes `reference` and `sub` already share
    the same ORIGIN convention (which atom sits at (0,0,0)).

    Built for `decomposition.py`'s `internal_displacement`: an
    externally-sourced `sub` (ISODISTORT, a published structure, a
    colleague's POSCAR) can sit at a genuinely different Wyckoff-setting
    origin (first confirmed on real BaTiO3 R3m). Only a pure TRANSLATION is
    searched: a genuine axis re-orientation shows up as a different lattice
    shape and is handled before this is called.

    Candidate shifts are derived directly from the data, not a hardcoded
    list of "common" centering vectors (0.5, 1/3, etc.) -- generalizing to
    ANY origin/coset convention difference: anchor on whichever species
    has the FEWEST atoms in `reference` (to keep the candidate count
    small), take its first atom, and try every same-species `sub` atom as
    that atom's true partner -- each pairing implies exactly one candidate
    shift. A candidate that doesn't extend to a full, consistent,
    every-atom alignment (not just the one anchor pair) is rejected
    immediately by `align_by_nearest_neighbor`'s own existing tolerance
    check, so a spurious single-pair coincidence essentially never survives
    to be considered a match. Among every candidate that DOES fully align,
    the one giving the smallest total displacement norm wins -- the
    physically most plausible interpretation.

    Returns (shift, perm) for the winning candidate, or (None, None) if no
    candidate produces a full alignment.
    """
    species_counts: dict = {}
    for s in reference.species:
        species_counts[s] = species_counts.get(s, 0) + 1
    anchor_species = min(species_counts, key=species_counts.get)
    anchor_index = reference.species.index(anchor_species)
    anchor_ref_frac = np.asarray(reference.frac_coords)[anchor_index]

    sub_frac = np.asarray(sub.frac_coords, dtype=np.float64)
    candidate_js = [j for j, s in enumerate(sub.species) if s == anchor_species]

    best_shift, best_perm, best_norm = None, None, None
    for j in candidate_js:
        shift = sub_frac[j] - anchor_ref_frac
        shift -= np.round(shift)
        shifted_frac = (sub_frac - shift) % 1.0
        try:
            perm = align_by_nearest_neighbor(
                reference.species, reference.frac_coords, sub.species, shifted_frac,
                reference.lattice, tol=tol,
            )
        except FreezeError:
            continue

        # FIXED: the shift used to be taken literally from this one anchor
        # pair's own raw coordinate difference -- correct only when the
        # anchor atom itself carries zero real physical displacement (the
        # common case: `min(species_counts)` tends to pick a low-count,
        # often-undisplaced species like a perovskite's own A-site).
        # Confirmed wrong directly on real graphite GM3+ data: with every
        # atom the same species (the anchor has no way to avoid being one
        # of the atoms the mode itself displaces), the raw anchor-pair
        # shift came out 0.49990906 instead of the true, exact 0.5 -- a
        # ~1e-4 error that then got applied UNIFORMLY to every atom,
        # smearing a ~0.0007 A phantom displacement onto atoms the real
        # mode never touches at all, inflating decomposition's own
        # residual to 81.65% for what should have been a clean, near-
        # zero-residual single-mode injection.
        #
        # Two earlier attempts were tried and rejected, each confirmed
        # wrong directly against the already-validated real BaTiO3 R3m
        # fixture (5 atoms: 1 Ba, 1 Ti, 3 O; Ba -- the anchor, the single
        # least-common species -- has EXACTLY zero real displacement for
        # this real distortion):
        # 1. Averaging the shift across every matched atom pair: Ti/O/O's
        #    own genuinely large, unequal real displacements are the
        #    MAJORITY of the atoms here (unlike graphite's 50/50 split),
        #    so the average is dominated by real signal, not the single
        #    correct (already-exact) anchor value -- wrongly reports a
        #    nonzero Ba amplitude.
        # 2. Picking whichever of (raw, averaged) gives the smaller TOTAL
        #    displacement norm: wrong for a different reason -- BaTiO3's
        #    own real distortion is large (~0.2 A total), so minimizing
        #    the norm just finds whatever shift best explains away as
        #    much of that real, LARGE physical signal as possible, which
        #    is never the correct answer once the real distortion is
        #    bigger than the shift error itself. This coincidentally
        #    "worked" for graphite only because its real signal (~0.001 A)
        #    happens to be the SAME tiny scale as the shift error, not
        #    because the criterion is actually sound.
        #
        # The fix that's actually safe in both directions: a real
        # crystallographic origin/coset shift is always a low-denominator
        # RATIONAL fraction (halves, thirds, quarters, sixths, eighths,
        # twelfths -- the same set of centering/Wyckoff-driven values this
        # project already documents seeing everywhere: STO/CMO/graphite's
        # own 0.5, TiO2's own real (0.5, 0.75, 0.875)). If the raw anchor-
        # derived shift is within `_SHIFT_SNAP_TOLERANCE` of such a
        # fraction, snap to it -- this only ever corrects genuine
        # numerical noise in the ANCHOR'S OWN shift value (graphite:
        # ~9.1e-5 off; BaTiO3: ~1.5e-4 off, both well inside the snap
        # tolerance), and is blind to every OTHER atom's own real
        # displacement, so it can never absorb real physical signal from
        # Ti/O/O the way both earlier attempts did. `_SHIFT_SNAP_TOLERANCE`
        # is deliberately tight (1e-3, not the far looser 1e-2 first
        # tried) specifically so it does NOT touch a real fixture's own
        # deliberately-NOT-nice test shift (STO's own arbitrary
        # (0.13, 0.27, 0.61), chosen specifically to prove this search is
        # data-derived, not a hardcoded centering-vector guess -- 0.13 is
        # 0.005 from the nearest nice fraction, 1/8, safely outside 1e-3).
        snapped_shift = np.array([_snap_to_nice_fraction(v, _SHIFT_SNAP_TOLERANCE) for v in shift])
        if not np.allclose(snapped_shift, shift):
            snapped_shifted_frac = (sub_frac - snapped_shift) % 1.0
            try:
                snapped_perm = align_by_nearest_neighbor(
                    reference.species, reference.frac_coords, sub.species, snapped_shifted_frac,
                    reference.lattice, tol=tol,
                )
                shift, perm, shifted_frac = snapped_shift, snapped_perm, snapped_shifted_frac
            except FreezeError:
                pass

        delta_frac = shifted_frac[perm] - reference.frac_coords
        delta_frac -= np.round(delta_frac)
        norm = float(np.linalg.norm(delta_frac @ reference.lattice))
        if best_norm is None or norm < best_norm:
            best_shift, best_perm, best_norm = shift, perm, norm
    return best_shift, best_perm










































# ---------------------------------------------------------------------------
# OPD BASIS QUALITY -- the single definition of "is this basis acceptable".
#
# Written after the same defect was found FOUR times in one session, once per
# call site, because each had invented its own weaker notion of "good enough":
#
#   per-arm within-arm gate    counted partners      missed a cross-arm duplicate
#   build_joint_opd_cell       counted hosted modes  count right, rank wrong
#   orbit-completion trigger   rank at tol=1e-8      missed noise and near-parallel
#   its own greedy keep        matrix_rank           kept a vector 2e-7 from parallel
#
# Every one of those was a rediscovery of the same thing. Anything deciding
# whether an OPD basis is good enough MUST call these, so a future weakening
# happens in one place or not at all.
# ---------------------------------------------------------------------------

OPD_RANK_TOLERANCE = 1e-8
"""Singular-value cut, RELATIVE to the largest, for counting dimensions."""

"""Below this ratio of smallest to largest singular value a basis counts as
deficient even at full rank. Separates real graphite L2 modes 0/1 (2e-7, two
pairs essentially parallel, six entries spanning four dimensions) from a
legitimately OBLIQUE basis -- real TiO2 X1 mode 3 sits at 0.072 with its two
directions 44 degrees apart, which is perfectly usable. Three orders clear of
both."""

"""A candidate direction must survive projection against everything already
accepted by more than this FRACTION OF ITS OWN NORM. Relative, not absolute:
an absolute cut is meaningless because patterns are normalised at different
scales, and `matrix_rank` is not a substitute -- a vector 2e-7 from parallel
still increments the rank."""

"""An orbit residual below this fraction of the pattern's own norm is
FLOATING-POINT NOISE, not a direction. When every operation maps a pattern to
+-itself the projection cancels catastrophically, leaving ~1e-16 against a
unit-norm pattern; real HfO2 Z1 fabricated all 18 of its partners this way,
some landing at |cos| = 0.000000 and looking perfect."""


def opd_basis_quality(vectors) -> tuple:
    """`(rank, conditioning)` for a set of OPD patterns, flattened.

    `conditioning` is the smallest singular value over the largest: 1.0 for an
    orthonormal set, ~0 for a collapsed one. Rank alone is a yes/no verdict at
    an arbitrary tolerance and cannot distinguish "independent" from "2e-7 from
    parallel"; conditioning is the continuous measure that can.
    """
    columns = np.array([np.asarray(v, dtype=np.float64).ravel() for v in vectors]).T
    if columns.size == 0:
        return 0, 0.0
    singular = np.linalg.svd(columns, compute_uv=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return 0, 0.0
    rank = int((singular > OPD_RANK_TOLERANCE * singular[0]).sum())
    return rank, float(singular[-1] / singular[0])











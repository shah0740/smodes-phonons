"""Non-analytic (LO-TO splitting) correction to a Gamma-point dynamical matrix.

Long-wavelength optical phonons in a polar crystal couple to the
macroscopic electric field their own displacement pattern induces -- this
makes the dynamical matrix at q=0 direction-dependent (non-analytic),
splitting otherwise-degenerate zone-center optical modes into a
longitudinal (LO, shifted) and transverse (TO, unshifted) branch along
whichever direction q approaches Gamma from. Standard reference: Gonze &
Lee, Phys. Rev. B 55, 10355 (1997); this is the same correction Phonopy,
QE, and ABINIT all implement.

Calculator-agnostic by design (plain numpy arrays only -- Born effective
charges, a dielectric tensor, masses, a cell volume), per this project's
architecture rule that nothing above the calculator seam (vasp.py) may
know which DFT code produced its inputs. vasp.py owns parsing Born
charges/the dielectric tensor out of a real OUTCAR; phonons.py owns adding
the resulting correction matrix into its own already-built force-constant
matrix and diagonalizing, exactly as it already does without this
correction (see diagonalize_phonons's own additional_force_constant_
correction parameter).

Only ever meaningful at Gamma (q -> 0). Every non-Gamma irrep's own
k-point is a genuinely different, finite q -- the non-analytic,
direction-dependent treatment below doesn't apply there at all, and
nothing in this module should be called for one.

The correction matrix built here (`non_analytic_correction`) is a rank-1
matrix by construction: prefactor * outer(s, s), where s is one scalar per
SMODES basis mode (`mode_polarization`) measuring how strongly that mode's
own displacement pattern couples to a macroscopic field along the
requested direction. This is what makes the physics come out right without
any extra bookkeeping: a mode with zero net dipole (every acoustic mode,
by the Born-charge sum rule sum_kappa Z*_kappa = 0; any IR-silent optical
mode, by symmetry) has s_mode = 0, so outer(s, s) contributes nothing to
that mode's own row/column -- diagonalizing the corrected matrix returns
that mode's frequency completely unchanged, with no separate case-by-case
"is this mode polar" logic needed anywhere in this module.
"""

from __future__ import annotations

import numpy as np

from .masses import mass_for_species


def _cart_operation(R_frac, lattice) -> np.ndarray:
    """A fractional rotation acting on row vectors (`f @ R_frac`) as the
    Cartesian rotation acting the same way (`x @ R_cart`)."""
    lattice = np.asarray(lattice, dtype=np.float64)
    return np.linalg.inv(lattice) @ R_frac @ lattice

_COULOMB_EV_ANGSTROM = 14.399645
"""e^2 / (4 pi eps_0), in eV*Angstrom -- the standard "practical units"
Coulomb constant (Ashcroft & Mermin and every other solid-state text use
the same ~14.4 eV*Angstrom figure). Converts a Born-charge/dielectric-
tensor-built quantity into eV/Angstrom^2, the same force-constant
convention phonons.py's own _build_force_constant_matrix already uses.
"""


class LoToError(ValueError):
    """Raised when LO-TO inputs (direction, dielectric tensor, shapes) are invalid."""


def _unit_vector(direction) -> np.ndarray:
    v = np.asarray(direction, dtype=np.float64)
    if v.shape != (3,):
        raise LoToError(f"direction must be a 3-vector, got shape {v.shape}")
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise LoToError("direction must be nonzero")
    return v / norm


def mode_polarization(mode_normalized: np.ndarray, born_charges: np.ndarray, direction) -> float:
    """s_mode(q_hat) = sum_{kappa,beta} mode_normalized[kappa,beta] * (q_hat . Z*_kappa)[beta].

    How strongly one SMODES basis mode's own (unit-Frobenius-norm)
    displacement pattern couples to a macroscopic field along `direction`
    -- exactly zero for a mode with no net dipole (acoustic modes; any
    IR-silent mode by symmetry), regardless of direction; nonzero only for
    a genuinely IR-active (polar) mode. `born_charges` must be indexed
    the same way as `mode_normalized`'s own (n_atoms, 3) array -- same
    atom count, same order (see non_analytic_correction's own docstring
    for why this holds automatically for a Gamma-point irrep's own
    untiled supercell).
    """
    q_hat = _unit_vector(direction)
    p = np.einsum("a,kab->kb", q_hat, born_charges)  # (n_atoms, 3): per-atom field-response vector
    if p.shape != mode_normalized.shape:
        raise LoToError(
            f"born_charges covers {p.shape[0]} atom(s) but the mode's own displacement "
            f"pattern covers {mode_normalized.shape[0]} -- Born charges must be given per "
            "atom of this irrep's own (Gamma-point, untiled) supercell, in the same order."
        )
    return float(np.sum(mode_normalized * p))


def non_analytic_correction(
    modes: list, born_charges: np.ndarray, dielectric_tensor: np.ndarray, direction, volume: float
) -> np.ndarray:
    """The (n_modes, n_modes) non-analytic correction to add directly to
    phonons._build_force_constant_matrix's own output (same SMODES-basis,
    same eV/Angstrom^2 convention), before mass-weighting.

    `modes` is an Irrep's own `.modes` list (SymmetryMode, in the same
    order phonons.py already builds its force-constant matrix in) --
    `born_charges` must be a (n_atoms, 3, 3) array in that SAME atom
    order. This only ever holds automatically for a Gamma-point irrep:
    SMODES's own superlattice multiplier is always [1,1,1] at Gamma (no
    supercell folding needed at q=0, a firm consequence of what a
    superlattice needs to host a k-point's phase pattern at all -- see
    the developer notes' supercell-multiplier table), so a Gamma-point irrep's
    atoms are exactly the parent POSCAR's own atoms, in whatever order a
    Born-charge DFPT run on that same parent structure would report them.
    Calling this for anything but a Gamma-point irrep is a caller error --
    the whole non-analytic treatment is only physically meaningful as
    q -> 0, not at a genuinely finite q.
    """
    q_hat = _unit_vector(direction)
    denom = float(q_hat @ np.asarray(dielectric_tensor, dtype=np.float64) @ q_hat)
    if denom <= 0:
        raise LoToError(
            f"direction . dielectric_tensor . direction must be positive, got {denom} -- "
            "check the dielectric tensor is the real, physical (positive-definite) one"
        )
    if volume <= 0:
        raise LoToError(f"volume must be positive, got {volume}")

    s = np.array([mode_polarization(mode.normalized, born_charges, q_hat) for mode in modes])
    prefactor = _COULOMB_EV_ANGSTROM * 4.0 * np.pi / volume / denom
    return prefactor * np.outer(s, s)


def mode_effective_charges(modes: list, dyn_vec: np.ndarray, born_charges: np.ndarray) -> np.ndarray:
    """The mode effective charge Z_bar_m (a 3-vector, one component per
    Cartesian field direction) for every true phonon eigenmode m -- (n_modes, 3).

    `dyn_vec` is phonons.PhononResult's own raw `mode_basis_eigenvectors`
    (eigh's direct output on the mass-weighted dynamical matrix, in the
    orthonormal reduced SMODES-mode basis) -- NOT
    `dynamical_matrix_eigenvectors`/`atomic_displacement_eigenvectors`,
    which are reconstructed into real space and then re-normalized to unit
    Frobenius norm for display (see phonons.py's own module docstring); that
    display normalization is NOT the physically correct one for this
    formula.

    Derivation: `dyn_vec[:, m]` satisfies sum_i dyn_vec[i,m]^2 = 1 by
    construction (eigh's own orthonormality). Writing the real eigenmode
    displacement as u(m) = sum_i c_i(m) * modes[i].normalized, the standard
    kinetic-energy normalization sum_kappa M_kappa |u_kappa(m)|^2 = 1 --
    combined with modes[i].normalized already being mutually orthogonal in
    this SAME mass-weighted metric (guaranteed by each mode moving exactly
    one species, per the developer notes -- this is exactly what makes the scalar
    sqrt(m_i * m_j) mass-weighting in `phonons._mass_matrix` valid at all)
    -- gives c_i(m) = dyn_vec[i,m] / sqrt(mass_i) directly (the same
    quantity phonons.diagonalize_phonons computes internally as
    `dyn_vec_unweighted`, just not previously exposed on its own). The mode
    effective charge is then the same per-basis-mode coupling
    `mode_polarization` already computes (once per Cartesian axis, not one
    q_hat direction), linearly combined with these c_i(m) coefficients --
    exactly analogous to how non_analytic_correction's own rank-1 structure
    already combines mode_polarization's per-basis-mode output for a single
    direction, just done here for all three axes independently, into
    the true eigenbasis rather than staying in the raw SMODES basis.
    """
    sqrt_masses = np.sqrt([mass_for_species(mode.species) for mode in modes])
    weighted = np.asarray(dyn_vec, dtype=np.float64) / sqrt_masses[:, np.newaxis]  # (n_modes, n_modes)

    axes = (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    s = np.array(
        [[mode_polarization(mode.normalized, born_charges, axis) for mode in modes] for axis in axes]
    )  # (3, n_modes)
    return weighted.T @ s.T  # (n_modes, 3)


def static_dielectric_tensor(
    mode_charges: np.ndarray, dynamical_matrix_eigenvalues: np.ndarray, dielectric_tensor: np.ndarray, volume: float
) -> tuple:
    """The static (zero-frequency) dielectric tensor epsilon(0), via the
    generalized Lyddane-Sachs-Teller sum (Gonze & Lee 1997):

        epsilon(0) = epsilon_inf + 14.399645 * (4*pi/Omega) *
                     sum_m outer(Z_bar_m, Z_bar_m) / omega_m^2

    `dynamical_matrix_eigenvalues` is phonons.PhononResult's own
    `dynamical_matrix_eigenvalues` (signed, natural-units eV/A^2/amu,
    proportional to omega_m^2) -- kept in these SAME natural units
    throughout (never converted to cm^-1/THz/meV) so this stays
    dimensionally consistent with non_analytic_correction's own eV/A^2
    force-constant-matrix convention, without any extra unit-conversion
    factors to get wrong.

    A mode with (numerically) zero mode effective charge contributes
    nothing regardless of its own eigenvalue -- true acoustic modes
    (protected exactly by the Born-charge sum rule, see
    non_analytic_correction's own docstring) never contribute at all, even
    though their eigenvalue is itself near zero.

    Returns (epsilon0, unstable_excluded). A contributing (nonzero mode
    effective charge) mode with a non-positive eigenvalue is EXCLUDED from
    the sum, not included with its literal (negative or huge, near-zero-
    denominator) 1/omega^2 value: the LST-type sum is derived by treating
    each mode as a bound harmonic oscillator and evaluating its static
    (zero-driving-frequency) response, 1/omega_m^2 -- for an unstable mode
    that premise doesn't hold at all (there is no bound oscillator to
    respond), so its own literal contribution isn't a real physical
    correction to include, just a number that happens to fall out of
    misapplying the formula outside where it's valid. Excluding it (rather
    than either including a physically meaningless value, or refusing to
    compute epsilon0 at all) keeps the reported number from silently
    looking like a real, trustworthy static dielectric constant.
    `unstable_excluded` is True whenever this happened, so the caller can
    flag it explicitly (this project doesn't silently hide a real
    situation, per the developer notes): a polar, unstable mode means the reference
    structure isn't a true energy minimum -- a real, textbook-known
    situation for a material with an incipient ferroelectric instability
    (e.g. real STO's own GM4- soft mode), not a bug, but a sign the
    reported epsilon(0) is not the true equilibrium answer (which would
    need relaxing to the actual minimum first).
    """
    if volume <= 0:
        raise LoToError(f"volume must be positive, got {volume}")

    eps0 = np.array(dielectric_tensor, dtype=np.float64).copy()
    prefactor = _COULOMB_EV_ANGSTROM * 4.0 * np.pi / volume
    unstable_excluded = False
    for m in range(len(dynamical_matrix_eigenvalues)):
        zbar = mode_charges[m]
        if np.allclose(zbar, 0.0):
            continue
        omega2 = float(dynamical_matrix_eigenvalues[m])
        if omega2 <= 0:
            unstable_excluded = True
            continue
        eps0 += prefactor * np.outer(zbar, zbar) / omega2

    return eps0, unstable_excluded


def group_equivalent_directions(directions: list, verified_rotations: list, lattice, tol: float = 1e-6) -> list:
    """Partition `directions` (each a 3-vector, Cartesian, not required to
    already be a unit vector) into symmetry-equivalence classes under
    `verified_rotations` -- a list of `(R_frac, _)` pairs, `R_frac` a
    fractional-coordinate rotation acting on ROW vectors (`f @ R_frac`).
    The CLI passes the transposes of `isotropy_basis.point_group_rotations`
    (ISOTROPY's point group in the parent's frame, column-vector form); the
    second element of each pair is ignored, since this is purely a question
    about the crystal's point group acting on q-space, not about which atom
    maps to which.

    Two directions are equivalent if some symmetry operation of the crystal
    maps one onto the other -- up to an overall sign, since q_hat and
    -q_hat give identical LO-TO physics (non_analytic_correction only ever
    depends on outer products of q_hat with itself). Each R_frac becomes a
    Cartesian rotation via `_cart_operation`, applied as `direction @ R_cart`.

    Returns a list of groups, each group a list of indices into
    `directions`, covering every index exactly once. Used to avoid
    computing (and reporting) the identical non_analytic_correction result
    multiple times for directions a genuine symmetry operation already
    guarantees will agree -- e.g. a cubic crystal's x/y/z all collapse to
    one group (compute once); a tetragonal crystal's z stays in its own
    group, separate from x/y (compute twice, honestly).
    """
    unit_dirs = []
    for d in directions:
        d = np.asarray(d, dtype=np.float64)
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            raise LoToError(f"direction {d.tolist()} must be nonzero")
        unit_dirs.append(d / norm)

    # The identity is always included explicitly, regardless of whether
    # it happens to already be in `verified_rotations` (in practice it
    # always is -- every real crystal is trivially symmetric under doing
    # nothing -- but this function shouldn't silently depend on that): q_hat
    # and -q_hat give identical LO-TO physics purely as a property of the
    # correction formula itself (non_analytic_correction only ever depends
    # on outer products of q_hat with itself), independent of any crystal
    # symmetry at all.
    cart_rotations = [np.eye(3)] + [_cart_operation(R, lattice) for R, _ in verified_rotations]

    n = len(unit_dirs)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if find(i) == find(j):
                continue
            for R_cart in cart_rotations:
                mapped = unit_dirs[i] @ R_cart
                if np.allclose(mapped, unit_dirs[j], atol=tol) or np.allclose(mapped, -unit_dirs[j], atol=tol):
                    union(i, j)
                    break

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())

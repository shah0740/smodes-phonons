"""Force constants, the dynamical matrix, and phonon frequencies/eigenvectors.

Validated numerically against three complete real VASP calculations
(tests/fixtures/STO_Irrep_GM4-, tests/fixtures/TiO2_Irrep_GM5+,
tests/fixtures/CMO_Irrep_X3-) before this module was written -- see the
plan this was built from. Computed
frequencies match legacy's real Phonon_frequencies output to within ~0.1%,
the residual fully explained by the deliberate `c` constant fix below.

Deliberate deviations from legacy (flagged per the developer notes, not silent):
  - numpy.linalg.eigh, not eig: real eigenvalues, no complex dtype. An
    unstable mode is reported as a negative cm^-1 value directly (e.g.
    -66.5), not "0.00+66.50j".
  - c = 2.99792458e10, not 3E10: shifts every frequency by ~0.07%.
  - force_constant_eigenvalues are the actual eV/A^2 eigenvalues, not their
    square root. Legacy's Eigenvalues_fcon reports sqrt(eigenvalue) under a
    header that claims eV/A^2 -- confirmed by reproducing legacy's own
    numbers (STO GM4-: legacy's 4.11/2.42/0.69j/0.03 are exactly
    sqrt(|16.92|)/sqrt(|5.84|)/sqrt(|-0.48|)/sqrt(|0.001|)).
  - Modes are ordered by ascending eigenvalue (eigh's natural order,
    negative/unstable first), not legacy's unspecified eig order.

One thing deliberately *not* changed, despite looking at first like the same
kind of inconsistency: eigenvector reconstruction (all three kinds, below)
uses each mode's *raw* SymmetryMode.displacements, not `.normalized`, even
though the force-constant matrix itself is built from normalized
displacements. This was tested both ways against the real
STO_Irrep_GM4-/Eigenvectors_Atom_disp reference -- raw matches to 6 decimal
places, normalized does not. the developer notes' normalization convention applies to
the force-projection step (see _build_force_constant_matrix), not to
reconstructing eigenvectors back into real atomic-displacement space.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .displace import DISPLACEMENT_AMPLITUDE_ANGSTROM
from .masses import mass_for_species
from .smodes import Irrep

_ELECTRON_CHARGE_C = 1.602177e-19
_ATOMIC_MASS_UNIT_KG = 1.66053e-27
_SPEED_OF_LIGHT_CM_PER_S = 2.99792458e10
_FORCE_CONST_TO_SI = _ELECTRON_CHARGE_C / (1e-10) ** 2  # eV/A^2 -> J/m^2


class PhononError(ValueError):
    """Raised when phonon inputs (forces, irrep) are inconsistent."""


@dataclass
class PhononResult:
    frequencies_cm: np.ndarray
    """Signed frequency per mode, cm^-1. Ascending; unstable modes negative."""
    frequencies_thz: np.ndarray
    frequencies_mev: np.ndarray
    force_constant_eigenvalues: np.ndarray
    """Signed eigenvalues of the (mass-independent) force-constant matrix, eV/A^2."""
    force_constant_eigenvectors: list
    """list[(n_atoms, 3)] -- force-constant matrix eigenvectors in real space."""
    dynamical_matrix_eigenvectors: list
    """list[(n_atoms, 3)] -- dynamical matrix eigenvectors in real space (mass-weighted)."""
    atomic_displacement_eigenvectors: list
    """list[(n_atoms, 3)] -- true atomic displacement pattern per mode (mass-unweighted)."""
    dynamical_matrix_eigenvalues: np.ndarray
    """Signed eigenvalues of the mass-weighted dynamical matrix, eV/(A^2*amu) --
    the SAME quantity frequencies_cm is derived from, kept here in its raw
    "natural units" form (proportional to omega^2) for anything needing to
    recombine it algebraically (e.g. lo_to.static_dielectric_tensor's own
    1/omega^2 sum) without re-deriving it from frequencies_cm's own unit
    conversion chain."""
    mode_basis_eigenvectors: np.ndarray
    """(n_modes, n_modes) -- the dynamical matrix's own raw eigenvectors, in
    the orthonormal reduced SMODES-mode basis (eigh's direct output, before
    any reconstruction into real atomic-displacement space). NOT the same
    normalization as dynamical_matrix_eigenvectors/atomic_displacement_
    eigenvectors above (those are re-normalized to unit Frobenius norm
    after reconstructing into real space, for display -- see this module's
    own docstring on why raw, not normalized, displacements are used for
    that reconstruction). This raw form is what a caller needs to correctly
    project a per-basis-mode quantity (e.g. lo_to.mode_polarization applied
    to each irrep.modes[i] separately) onto the true, mass-weighted phonon
    eigenmodes via a plain linear combination."""


def _build_force_constant_matrix(irrep: Irrep, mode_forces: list, undistorted_forces) -> np.ndarray:
    n = len(irrep.modes)
    if len(mode_forces) != n:
        raise PhononError(f"expected forces for {n} modes, got {len(mode_forces)}")

    fff = np.zeros((n, n))
    for i in range(n):
        forces_i = mode_forces[i]
        if undistorted_forces is not None:
            forces_i = forces_i - undistorted_forces
        for j in range(n):
            fff[i, j] = np.sum(irrep.modes[j].normalized * forces_i)

    phi = -fff / DISPLACEMENT_AMPLITUDE_ANGSTROM
    return (phi + phi.T) / 2


def _mass_matrix(irrep: Irrep) -> np.ndarray:
    masses = np.array([mass_for_species(mode.species) for mode in irrep.modes])
    sqrt_masses = np.sqrt(masses)
    return np.outer(sqrt_masses, sqrt_masses)


def _reconstruct_eigenvectors(irrep: Irrep, eigvecs: np.ndarray) -> list:
    """sum_j eigvecs[j, i] * mode_j.displacements (raw, not normalized), unit-normalized."""
    n = len(irrep.modes)
    results = []
    for i in range(n):
        acc = np.zeros_like(irrep.modes[0].displacements)
        for j in range(n):
            acc = acc + eigvecs[j, i] * irrep.modes[j].displacements
        norm = np.linalg.norm(acc)
        if norm == 0:
            raise PhononError(f"reconstructed eigenvector {i} is exactly zero -- degenerate/malformed input")
        results.append(acc / norm)
    return results


def diagonalize_phonons(
    irrep: Irrep, mode_forces: list, undistorted_forces=None, additional_force_constant_correction=None
) -> PhononResult:
    """Build the force-constant matrix from per-mode forces, diagonalize, and return
    frequencies plus eigenvectors.

    `mode_forces[i]` is the (n_atoms, 3) force array (eV/A) for irrep.modes[i]'s
    displaced structure. If `undistorted_forces` is given, it's subtracted from
    every mode's forces first -- needed whenever the parent structure has free
    Wyckoff parameters and its own reference forces aren't exactly zero.

    `additional_force_constant_correction`, if given, is an (n_modes, n_modes)
    matrix added directly to the force-constant matrix before mass-weighting
    -- this is the seam `lo_to.non_analytic_correction` plugs into for
    Gamma-point LO-TO splitting, kept as a plain optional array here rather
    than anything LO-TO-specific so this module stays agnostic to why the
    correction exists (matches this project's calculator-seam separation:
    phonons.py doesn't need to know about Born charges or a dielectric
    tensor, only that some caller wants an extra term added in).
    """
    force_mat = _build_force_constant_matrix(irrep, mode_forces, undistorted_forces)
    if additional_force_constant_correction is not None:
        if additional_force_constant_correction.shape != force_mat.shape:
            raise PhononError(
                f"additional_force_constant_correction has shape "
                f"{additional_force_constant_correction.shape}, expected {force_mat.shape}"
            )
        force_mat = force_mat + additional_force_constant_correction
    mass_mat = _mass_matrix(irrep)
    dyn_mat = force_mat / mass_mat

    force_eig, force_vec = np.linalg.eigh(force_mat)
    dyn_eig, dyn_vec = np.linalg.eigh(dyn_mat)

    frequencies_cm = (
        np.sign(dyn_eig)
        * np.sqrt(np.abs(dyn_eig) * _FORCE_CONST_TO_SI / _ATOMIC_MASS_UNIT_KG)
        / (2 * np.pi * _SPEED_OF_LIGHT_CM_PER_S)
    )
    frequencies_thz = frequencies_cm / 33.356
    frequencies_mev = frequencies_cm / 8.066

    force_constant_eigenvectors = _reconstruct_eigenvectors(irrep, force_vec)
    dynamical_matrix_eigenvectors = _reconstruct_eigenvectors(irrep, dyn_vec)

    sqrt_masses = np.sqrt([mass_for_species(mode.species) for mode in irrep.modes])
    dyn_vec_unweighted = dyn_vec / sqrt_masses[:, np.newaxis]
    atomic_displacement_eigenvectors = _reconstruct_eigenvectors(irrep, dyn_vec_unweighted)

    return PhononResult(
        frequencies_cm=frequencies_cm,
        frequencies_thz=frequencies_thz,
        frequencies_mev=frequencies_mev,
        force_constant_eigenvalues=force_eig,
        force_constant_eigenvectors=force_constant_eigenvectors,
        dynamical_matrix_eigenvectors=dynamical_matrix_eigenvectors,
        atomic_displacement_eigenvectors=atomic_displacement_eigenvectors,
        dynamical_matrix_eigenvalues=dyn_eig,
        mode_basis_eigenvectors=dyn_vec,
    )


def save_phonon_result(result: PhononResult, path) -> None:
    """Save a PhononResult to a single .npz file.

    The three eigenvector lists (each a list of (n_atoms, 3) arrays, one per
    mode) are stacked into (n_modes, n_atoms, 3) arrays for storage.
    """
    np.savez(
        path,
        frequencies_cm=result.frequencies_cm,
        frequencies_thz=result.frequencies_thz,
        frequencies_mev=result.frequencies_mev,
        force_constant_eigenvalues=result.force_constant_eigenvalues,
        force_constant_eigenvectors=np.stack(result.force_constant_eigenvectors),
        dynamical_matrix_eigenvectors=np.stack(result.dynamical_matrix_eigenvectors),
        atomic_displacement_eigenvectors=np.stack(result.atomic_displacement_eigenvectors),
        dynamical_matrix_eigenvalues=result.dynamical_matrix_eigenvalues,
        mode_basis_eigenvectors=result.mode_basis_eigenvectors,
    )


def load_phonon_result(path) -> PhononResult:
    """Inverse of save_phonon_result."""
    with np.load(Path(path)) as data:
        return PhononResult(
            frequencies_cm=data["frequencies_cm"],
            frequencies_thz=data["frequencies_thz"],
            frequencies_mev=data["frequencies_mev"],
            force_constant_eigenvalues=data["force_constant_eigenvalues"],
            force_constant_eigenvectors=list(data["force_constant_eigenvectors"]),
            dynamical_matrix_eigenvectors=list(data["dynamical_matrix_eigenvectors"]),
            atomic_displacement_eigenvectors=list(data["atomic_displacement_eigenvectors"]),
            dynamical_matrix_eigenvalues=data["dynamical_matrix_eigenvalues"],
            mode_basis_eigenvectors=data["mode_basis_eigenvectors"],
        )

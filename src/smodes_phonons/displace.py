"""Turning a parsed smodes Irrep + mode into a displaced supercell POSCAR.

DISPLACEMENT_AMPLITUDE_ANGSTROM is the one place this constant is defined
(the developer notes: legacy defines it twice, independently, in disp_irrep_calc.py
and phonon_modes.py, and if they ever disagree force constants are wrong by
their ratio with no error). Any future force-analysis code must import it
from here, never redefine it.

Matches legacy/disp_irrep_calc.py's math exactly:
    positions_disp = positions + 0.001 * (irrep_disp / norm(irrep_disp))
converted back to fractional via cart @ inv(lattice). The nested
Irrep_<label>/Mode_i/POSCAR layout here replaces legacy's flat POSCAR_i
naming deliberately, per the developer notes' CLI design.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .smodes import Irrep
from .structure import Structure

DISPLACEMENT_AMPLITUDE_ANGSTROM = 0.001


def supercell_structure(irrep: Irrep, comment: str) -> Structure:
    """The undistorted symmetry-adapted supercell an irrep's modes displace from."""
    frac_coords = irrep.atom_positions @ np.linalg.inv(irrep.superlattice)
    return Structure(
        comment=comment,
        lattice=irrep.superlattice.copy(),
        species=list(irrep.atom_species),
        frac_coords=frac_coords,
    )


def write_mode_poscars(irrep: Irrep, irrep_dir: Path, comment: str) -> list:
    """Write one Mode_i/POSCAR per distinct mode, displaced at the fixed amplitude,
    plus an Undistorted/POSCAR (the same supercell, no displacement applied).

    The undistorted reference is always written, whether or not this material
    has free Wyckoff parameters: it's cheap, and its presence on disk is what
    later force analysis uses to decide whether to subtract a reference force
    -- checking a file, rather than asking a question with no shared state
    (legacy's "degrees of freedom?" prompt, asked twice, independently, in
    sub.sh and force_analysis.sh).

    Reruns silently overwrite -- auto-creation is meant to be silent and
    non-interactive (the developer notes), so no --force guard here.
    """
    undistorted = supercell_structure(irrep, comment)

    undistorted_dir = Path(irrep_dir) / "Undistorted"
    undistorted_dir.mkdir(parents=True, exist_ok=True)
    undistorted.to_poscar(undistorted_dir / "POSCAR", comment=f"{comment} Undistorted")

    paths = []
    for i, mode in enumerate(irrep.modes):
        displaced = undistorted.displaced(mode.normalized * DISPLACEMENT_AMPLITUDE_ANGSTROM)
        mode_dir = Path(irrep_dir) / f"Mode_{i}"
        mode_dir.mkdir(parents=True, exist_ok=True)
        out_path = mode_dir / "POSCAR"
        displaced.to_poscar(out_path, comment=f"{comment} Mode_{i}")
        paths.append(out_path)
    return paths

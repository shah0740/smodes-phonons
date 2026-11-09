"""Shared checks for the v2.0 (ISOTROPY-matrix) OPD basis tests."""

import numpy as np

from smodes_phonons.decomposition import check_stoichiometry, decompose_at_k_vector
from smodes_phonons.opd import LETTERS




def assert_round_trip(case, parent, findsym_log_text, structure, k_vector, irreps, requested):
    """Decompose `structure` (what `freeze` wrote) and require EXACTLY the
    requested OPD back. `requested` = {irrep label: {position: coefficient}}.

    The frozen pattern may be a DFT eigenvector -- a fixed mix of the irrep's
    SMODES modes -- so per irrep the (mode x position) amplitude table must
    factor as (mode weights) x (requested coefficients): rank one, with the
    position profile proportional to the request, and zero on every position
    not requested. Residual must vanish."""
    result = decompose_at_k_vector(
        k_vector, irreps, parent, structure,
        stoichiometry_multiplier=check_stoichiometry(parent, structure), findsym_log_text=findsym_log_text,
    )
    case.assertLess(result.residual_norm_angstrom, 1e-6)
    for label, coefficients in requested.items():
        rows = [ma for ma in result.amplitudes if ma.label == label]
        modes = sorted({ma.mode_index for ma in rows})
        positions = sorted({LETTERS.index(ma.direction) for ma in rows})
        table = np.zeros((len(modes), len(positions)))
        for ma in rows:
            table[modes.index(ma.mode_index), positions.index(LETTERS.index(ma.direction))] = ma.amplitude_angstrom
        u, s, vt = np.linalg.svd(table)
        case.assertLess(s[1] if len(s) > 1 else 0.0, 1e-8 * max(s[0], 1e-30) + 1e-12, f"{label}: not one direction")
        profile = vt[0] * np.sign(vt[0][np.argmax(np.abs(vt[0]))])
        wanted = np.array([coefficients.get(p, 0.0) for p in positions], float)
        wanted = wanted / np.linalg.norm(wanted) * np.sign(wanted[np.argmax(np.abs(wanted))])
        np.testing.assert_allclose(profile, wanted, atol=1e-7, err_msg=f"{label}: position profile")
    return result

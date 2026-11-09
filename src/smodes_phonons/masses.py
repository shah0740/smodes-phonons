"""Standard atomic weights, keyed by chemical symbol.

Hardcoded rather than read from a POTCAR's POMASS lines: a POTCAR isn't
always available alongside an OUTCAR the way this pipeline is used (e.g. a
physicist reusing an old calculation, or reorganizing directories), and the
values VASP prints there track standard atomic weights closely enough
(confirmed against real fixtures: Sr 87.620 vs. 87.62 here, Ti 47.880 vs.
47.867 here, O 16.000 vs. 15.999 here) that a fixed table is both simpler
and independent of what happens to be sitting on disk for a given run.

IUPAC 2021 standard atomic weights (conventional single values, not ranges).
Radioactive elements with no stable isotope use the mass of their longest-
lived known isotope, in brackets in most tables -- listed here without
brackets since this is a plain float table.
"""

from __future__ import annotations

ATOMIC_MASSES = {
    "H": 1.008, "He": 4.002602,
    "Li": 6.94, "Be": 9.0121831, "B": 10.81, "C": 12.011, "N": 14.007,
    "O": 15.999, "F": 18.998403163, "Ne": 20.1797,
    "Na": 22.98976928, "Mg": 24.305, "Al": 26.9815384, "Si": 28.085,
    "P": 30.973761998, "S": 32.06, "Cl": 35.45, "Ar": 39.95,
    "K": 39.0983, "Ca": 40.078, "Sc": 44.955908, "Ti": 47.867,
    "V": 50.9415, "Cr": 51.9961, "Mn": 54.938043, "Fe": 55.845,
    "Co": 58.933194, "Ni": 58.6934, "Cu": 63.546, "Zn": 65.38,
    "Ga": 69.723, "Ge": 72.630, "As": 74.921595, "Se": 78.971,
    "Br": 79.904, "Kr": 83.798,
    "Rb": 85.4678, "Sr": 87.62, "Y": 88.905838, "Zr": 91.224,
    "Nb": 92.90637, "Mo": 95.95, "Tc": 97.0, "Ru": 101.07,
    "Rh": 102.90549, "Pd": 106.42, "Ag": 107.8682, "Cd": 112.414,
    "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60,
    "I": 126.90447, "Xe": 131.293,
    "Cs": 132.90545196, "Ba": 137.327, "La": 138.90547, "Ce": 140.116,
    "Pr": 140.90766, "Nd": 144.242, "Pm": 145.0, "Sm": 150.36,
    "Eu": 151.964, "Gd": 157.25, "Tb": 158.925354, "Dy": 162.500,
    "Ho": 164.930329, "Er": 167.259, "Tm": 168.934219, "Yb": 173.045,
    "Lu": 174.9668, "Hf": 178.486, "Ta": 180.94788, "W": 183.84,
    "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084,
    "Au": 196.966570, "Hg": 200.592, "Tl": 204.38, "Pb": 207.2,
    "Bi": 208.98040, "Po": 209.0, "At": 210.0, "Rn": 222.0,
    "Fr": 223.0, "Ra": 226.0, "Ac": 227.0, "Th": 232.0377,
    "Pa": 231.03588, "U": 238.02891, "Np": 237.0, "Pu": 244.0,
    "Am": 243.0, "Cm": 247.0, "Bk": 247.0, "Cf": 251.0, "Es": 252.0,
    "Fm": 257.0, "Md": 258.0, "No": 259.0, "Lr": 262.0, "Rf": 267.0,
    "Db": 268.0, "Sg": 271.0, "Bh": 272.0, "Hs": 270.0, "Mt": 276.0,
    "Ds": 281.0, "Rg": 280.0, "Cn": 285.0, "Nh": 284.0, "Fl": 289.0,
    "Mc": 288.0, "Lv": 293.0, "Ts": 294.0, "Og": 294.0,
}


def mass_for_species(symbol: str) -> float:
    """Standard atomic weight (amu) for a chemical symbol.

    Raises KeyError with a clear message for an unknown symbol -- never
    guess or silently average, matching this codebase's rule that a mode
    mixing species (or, here, naming one we don't recognize) means
    something upstream is broken.
    """
    try:
        return ATOMIC_MASSES[symbol]
    except KeyError:
        raise KeyError(
            f"no atomic mass on file for species {symbol!r} -- check the symbol is a "
            "standard chemical element symbol (case-sensitive, e.g. 'Sr' not 'sr')"
        ) from None

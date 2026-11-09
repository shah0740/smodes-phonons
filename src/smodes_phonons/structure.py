"""POSCAR parsing, writing, and fractional/Cartesian conversion.

This is the only place in the codebase that is allowed to know the POSCAR
line layout. Everything else -- symmetry, displacement generation, force
constants -- goes through a `Structure` instance.

Fixes a legacy bug: the scale factor on line 2 was ignored everywhere, with
lattice vectors read raw from lines 3-5. VASP's format allows two things on
that line:
  - a positive number: a uniform multiplier applied to the raw lattice
    vectors (and, if coordinates are Cartesian, to the coordinates too).
  - a negative number: minus the target cell volume in A^3. The raw lattice
    vectors are uniformly rescaled so the resulting cell has that volume.
`Structure.lattice` always holds the real, fully-scaled lattice in Angstrom,
and `Structure.to_poscar` always writes a scale factor of 1.0 -- so nothing
downstream ever has to think about the scale factor again.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


class PoscarFormatError(ValueError):
    """Raised when a POSCAR cannot be parsed unambiguously."""


def _looks_numeric(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def _resolve_scale_factor(raw_lattice: np.ndarray, scale: float, path) -> float:
    if scale > 0:
        return scale
    if scale == 0:
        raise PoscarFormatError(f"{path}: scale factor on line 2 is 0, which is not valid")

    raw_volume = float(np.linalg.det(raw_lattice))
    if raw_volume <= 0:
        raise PoscarFormatError(
            f"{path}: negative scale factor requests a target volume, but the raw lattice "
            f"vectors on lines 3-5 are left-handed or degenerate (det={raw_volume})"
        )
    target_volume = -scale
    return (target_volume / raw_volume) ** (1.0 / 3.0)


@dataclass
class Structure:
    comment: str
    lattice: np.ndarray
    """(3, 3) real-space lattice vectors in Angstrom, as rows. Scale factor already applied."""
    species: list
    """Chemical symbol per atom, length n_atoms. Same species must be contiguous, as in POSCAR."""
    frac_coords: np.ndarray
    """(n_atoms, 3) fractional coordinates."""

    def __post_init__(self) -> None:
        self.lattice = np.array(self.lattice, dtype=np.float64)
        self.frac_coords = np.array(self.frac_coords, dtype=np.float64)
        self.species = list(self.species)

        if self.lattice.shape != (3, 3):
            raise ValueError(f"lattice must be shape (3, 3), got {self.lattice.shape}")
        if self.frac_coords.shape != (len(self.species), 3):
            raise ValueError(
                f"frac_coords shape {self.frac_coords.shape} does not match "
                f"{len(self.species)} species"
            )

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def cart_coords(self) -> np.ndarray:
        return self.frac_to_cart(self.frac_coords)

    def frac_to_cart(self, frac_coords) -> np.ndarray:
        return np.asarray(frac_coords, dtype=np.float64) @ self.lattice

    def cart_to_frac(self, cart_coords) -> np.ndarray:
        return np.asarray(cart_coords, dtype=np.float64) @ np.linalg.inv(self.lattice)

    @property
    def species_counts(self) -> list:
        """[(symbol, count), ...] in file order, one entry per contiguous block.

        Raises if a species appears in more than one block -- POSCAR requires
        same-species atoms to be grouped together, and writing such a
        structure back out would silently corrupt the species/count lines.
        """
        counts: list = []
        seen = set()
        for sp in self.species:
            if counts and counts[-1][0] == sp:
                counts[-1][1] += 1
                continue
            if sp in seen:
                raise ValueError(
                    f"species {sp!r} appears in more than one block; POSCAR requires "
                    "atoms of the same species to be contiguous"
                )
            seen.add(sp)
            counts.append([sp, 1])
        return [(sp, n) for sp, n in counts]

    def displaced(self, cart_displacement) -> "Structure":
        """Return a new Structure with atoms shifted by a Cartesian (n_atoms, 3) displacement."""
        cart_displacement = np.asarray(cart_displacement, dtype=np.float64)
        if cart_displacement.shape != (self.n_atoms, 3):
            raise ValueError(
                f"displacement shape {cart_displacement.shape} does not match "
                f"({self.n_atoms}, 3)"
            )
        new_frac = self.cart_to_frac(self.cart_coords + cart_displacement)
        return Structure(
            comment=self.comment,
            lattice=self.lattice.copy(),
            species=list(self.species),
            frac_coords=new_frac,
        )

    @classmethod
    def from_poscar(cls, path) -> "Structure":
        path = Path(path)
        lines = path.read_text().splitlines()
        if len(lines) < 8:
            raise PoscarFormatError(f"{path}: too few lines to be a POSCAR")

        comment = lines[0]

        scale_tokens = lines[1].split()
        if len(scale_tokens) != 1:
            raise PoscarFormatError(
                f"{path}: line 2 (scale factor) must be a single number, got {lines[1]!r}"
            )
        try:
            scale = float(scale_tokens[0])
        except ValueError as exc:
            raise PoscarFormatError(f"{path}: line 2 (scale factor) is not a number: {lines[1]!r}") from exc

        try:
            raw_lattice = np.array([[float(x) for x in lines[i].split()] for i in (2, 3, 4)])
        except ValueError as exc:
            raise PoscarFormatError(f"{path}: lines 3-5 (lattice vectors) are not all numeric") from exc

        factor = _resolve_scale_factor(raw_lattice, scale, path)
        lattice = raw_lattice * factor

        symbol_tokens = lines[5].split()
        count_tokens = lines[6].split()
        if not symbol_tokens or any(_looks_numeric(t) for t in symbol_tokens):
            raise PoscarFormatError(
                f"{path}: line 6 must list chemical species symbols (VASP5 format), got "
                f"{lines[5]!r}. POSCARs without an explicit species line are not supported -- "
                "species are never inferred."
            )
        if len(symbol_tokens) != len(count_tokens):
            raise PoscarFormatError(
                f"{path}: line 6 has {len(symbol_tokens)} species but line 7 has "
                f"{len(count_tokens)} counts"
            )
        try:
            counts = [int(c) for c in count_tokens]
        except ValueError as exc:
            raise PoscarFormatError(f"{path}: line 7 (species counts) is not all integers: {lines[6]!r}") from exc
        species = [sym for sym, n in zip(symbol_tokens, counts) for _ in range(n)]
        n_atoms = len(species)

        header_idx = 7
        header = lines[header_idx].strip()
        if header[:1].lower() == "s":
            # "Selective dynamics" line -- skip it, coordinate-type line follows.
            header_idx += 1
            header = lines[header_idx].strip()

        mode = header[:1].lower()
        if mode not in ("d", "c"):
            raise PoscarFormatError(
                f"{path}: line {header_idx + 1} must be 'Direct' or 'Cartesian', got {lines[header_idx]!r}"
            )

        coord_start = header_idx + 1
        coord_lines = lines[coord_start:coord_start + n_atoms]
        if len(coord_lines) < n_atoms:
            raise PoscarFormatError(
                f"{path}: expected {n_atoms} atom coordinate lines after line {coord_start + 1}, "
                f"found {len(coord_lines)}"
            )
        try:
            raw_coords = np.array([[float(x) for x in line.split()[:3]] for line in coord_lines])
        except ValueError as exc:
            raise PoscarFormatError(f"{path}: atom coordinate lines are not all numeric") from exc

        if mode == "d":
            frac_coords = raw_coords
        else:
            frac_coords = (raw_coords * factor) @ np.linalg.inv(lattice)

        return cls(comment=comment, lattice=lattice, species=species, frac_coords=frac_coords)

    def to_poscar(self, path, comment=None) -> None:
        counts = self.species_counts
        lines = [
            comment if comment is not None else self.comment,
            "1.0",
        ]
        lines += [" ".join(f"{x:.15f}" for x in row) for row in self.lattice]
        lines.append(" ".join(sym for sym, _ in counts))
        lines.append(" ".join(str(n) for _, n in counts))
        lines.append("Direct")
        lines += [" ".join(f"{x:.15f}" for x in row) for row in self.frac_coords]
        Path(path).write_text("\n".join(lines) + "\n")

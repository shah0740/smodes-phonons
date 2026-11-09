# Command reference

*Written by Claude (Anthropic), from the smodes-phonons v2.0.0 source and its
`--help` text.*

Every command is run as `smodes-phonons <command> ...`. `smodes-phonons
<command> --help` prints the same information from the program itself.

**Conventions shared by every command**

- **Material tag (`--name`).** Generated files are named from a short tag:
  `STO_findsym`, `STO_smodes_GM`, … The tag is taken from `--name`, else from
  a project config saved in the working directory, else from the first line
  of the POSCAR (cleaned up). It is saved on first use, so later commands in
  the same directory agree.
- **Irrep labels** are case-insensitive. A **bare k-point label** (letters
  only, e.g. `GM`, `X`, `M`, `R`) means every irrep at that k-point; labels
  can be mixed (`GM X2+`).
- **Directory layout.** Each irrep lives in `<K>/Irrep_<label>/`
  (e.g. `GM/Irrep_GM4-/`). A flat `Irrep_<label>/` from older versions is
  still read.
- **`--isodata PATH`** points at the ISOTROPY data directory (must end in
  `/`). Order of precedence: flag, project config, user config, `ISODATA`
  environment variable, `findsym` on `PATH`.
- **Never interactive in scripts.** Prompts (e.g. to choose a mode) appear
  only when stdin is a terminal; otherwise the command stops with a message
  naming the flag to use.

| command | what it does |
|---|---|
| [`doctor`](#doctor) | check the environment |
| [`symmetry`](#symmetry) | run findsym on a POSCAR |
| [`run-smodes`](#run-smodes) | run smodes at a k-point |
| [`smodes`](#smodes) | summarise an smodes output |
| [`irrep`](#irrep) | write the displaced structures for DFT |
| [`phonons`](#phonons) | frequencies and eigenvectors from DFT forces (incl. LO-TO) |
| [`bec`](#bec) | set up the Born-charge calculation for every IR-active Γ irrep |
| [`bec-analysis`](#bec-analysis) | Born charges, ε∞, LO-TO and the total ε(0) |
| [`freeze`](#freeze) | freeze modes along an order-parameter direction |
| [`freeze-amplitude`](#freeze-amplitude) | sweep the amplitude of 1–3 frozen modes |
| [`decomposition`](#decomposition) | decompose a distorted structure into mode amplitudes |
| [`landau-fit`](#landau-fit) | fit Landau coefficients to DFT energies |
| [`landau-grid`](#landau-grid) | generate the structures for a Landau fit |

---

## `doctor`

Checks that NumPy, `findsym` and `smodes` are found and work, reports the
detected findsym version and input/log format, and how `ISODATA` was
resolved. Run it once after installing.

```
smodes-phonons doctor [--isodata PATH]
```

| flag | default | meaning |
|---|---|---|
| `--isodata PATH` | resolved as above | ISOTROPY data directory (trailing `/`) |

---

## `symmetry`

Builds findsym input from a POSCAR (real species, scale factor applied), runs
findsym, and saves the log as `<tag>_findsym`.

```
smodes-phonons symmetry POSCAR [--name TAG] [--tolerance TOL] [--isodata PATH]
```

| argument / flag | default | meaning |
|---|---|---|
| `POSCAR` | — | VASP5-format structure |
| `--name TAG` | see conventions | material tag |
| `--tolerance TOL` | `0` (exact) | findsym position tolerance (Å). A DFT-relaxed or frozen structure needs a non-zero value, e.g. `0.005`, or it reads as P1 |
| `--isodata PATH` | | ISOTROPY data directory |

The log also carries findsym's CIF block; cut it out with
`sed -n '/^# CIF file created by FINDSYM/,/^# end of cif/p' <tag>_findsym > file.cif`.

---

## `run-smodes`

Builds smodes input from a findsym log and runs smodes for one k-point,
saving `<tag>_smodes_<K>`. Usually not needed directly: `irrep` does this
automatically.

```
smodes-phonons run-smodes FINDSYM_LOG K [--poscar POSCAR] [--name TAG]
```

| argument / flag | default | meaning |
|---|---|---|
| `FINDSYM_LOG` | — | e.g. `STO_findsym` |
| `K` | — | k-point label, e.g. `GM`, `X`, `M`, `R` |
| `--poscar POSCAR` | `POSCAR` | the structure the log was built from (supplies species names) |
| `--name TAG` | | material tag |

---

## `smodes`

Prints a summary of an smodes output: space group, lattice, Wyckoff site of
each species, and every irrep with its Mulliken label, dimension, number of
distinct modes, translational modes and IR/Raman activity.

```
smodes-phonons smodes SMODES_OUTPUT
```

Irrep labels depend on the Wyckoff setting findsym chose; if a label you
expect (e.g. from ISODISTORT) is missing, compare the printed Wyckoff sites.

---

## `irrep`

Writes the displaced structures for the DFT force calculations: for each
requested irrep, `<K>/Irrep_<label>/Mode_i/POSCAR` (one per distinct mode,
displaced 0.001 Å along that SMODES mode) and `Undistorted/POSCAR`. The
findsym log and smodes output are created first if missing.

```
smodes-phonons irrep IRREP [IRREP ...] [--poscar POSCAR] [--name TAG]
    [--tolerance TOL] [--isodata PATH] [--kpoints KPOINTS] [--incar INCAR]
    [--opd] [--bec] [--magmom]
```

| argument / flag | default | meaning |
|---|---|---|
| `IRREP ...` | — | labels (`GM4-`, `X2+`) or bare k-points (`GM`) |
| `--poscar POSCAR` | `POSCAR` | parent structure |
| `--name TAG` | | material tag |
| `--tolerance TOL` | `0` | findsym tolerance, only used if the log does not exist yet |
| `--isodata PATH` | | ISOTROPY data directory |
| `--kpoints KPOINTS` | `KPOINTS` | mesh to rescale to each supercell, keeping the real-space k-point density; skipped with a warning if absent |
| `--incar INCAR` | `INCAR` | template written into every `Mode_i/` and `Undistorted/` with `NSW=0`, `IBRION=-1`, `ISYM=0` forced (a static force calculation); every other tag copied unchanged; skipped with a warning if absent |
| `--opd` | off | also write `opd.txt`, the order-parameter-direction report for each degenerate irrep (see below) |
| `--bec` | off | for a Γ-point IR-active irrep, also write `BEC/` (input for the Born-charge calculation, see [LO-TO](#lo-to-splitting)) |
| `--magmom` | off | rewrite `MAGMOM` from the template (parent atom order) for each structure's own atom order; error if its length does not match the parent |

**`opd.txt`** lists every direction of each degenerate mode: its position
letter (the `--direction` string `freeze` takes), its star arm, how it relates
to the others (within-arm, other arm, quadrature), each arm's wavevector Q,
and the full displacement pattern. Built from ISOTROPY's matrices; no DFT.
See `OPD_CONSTRUCTION.md`.

**Without `--magmom`,** a `MAGMOM` in the template is copied unchanged, which
is wrong for any structure not on the parent's cell; a warning says so.

---

## `phonons`

Reads `<K>/Irrep_<label>/Mode_i/OUTCAR[.gz]` (and `Undistorted/OUTCAR` if
present, whose forces are subtracted), builds the force-constant matrix in the
symmetry-adapted basis, mass-weights it, and diagonalises it. Unstable modes
are reported as negative frequencies.

```
smodes-phonons phonons IRREP [IRREP ...] [--generate-opd] [--poscar POSCAR]
    [--name TAG] [--born-charges OUTCAR] [--direction X,Y,Z]
    [--tolerance TOL] [--isodata PATH]
```

| argument / flag | default | meaning |
|---|---|---|
| `IRREP ...` | — | labels or bare k-points; a failing label does not stop the others |
| `--generate-opd` | off | also write `opd_eigenvectors.txt`: every order-parameter direction of each DFT eigenvector (same letters as `irrep --opd`) |
| `--poscar POSCAR` | `POSCAR` | parent structure |
| `--name TAG` | | material tag |
| `--born-charges OUTCAR` | `BEC/OUTCAR` if it exists | VASP `LEPSILON=.TRUE.` output with Born charges and ε∞, for [LO-TO splitting](#lo-to-splitting) |
| `--direction X,Y,Z` | x, y, z | extra q-direction for LO-TO splitting; repeatable |
| `--tolerance TOL`, `--isodata PATH` | | findsym settings, only used for LO-TO direction grouping if the log does not exist |

**Prints** frequencies (cm⁻¹, THz, meV) and force-constant eigenvalues
(eV/Å²). **Writes**, in `<K>/Irrep_<label>/`:

| file | contents |
|---|---|
| `phonons.txt` | frequencies and the force-constant, dynamical-matrix and atomic-displacement eigenvectors |
| `phonons.npz` | the same in machine-readable form; what `freeze` reads |
| `POSCAR_superlattice`, `phonons_superlattice.txt` | the eigenvectors on a diagonal supercell aligned with the parent's axes, for figures |

### LO-TO splitting

In a polar insulator, a long-wavelength optical phonon creates a macroscopic
electric field that shifts the longitudinal (LO) branch away from the
transverse (TO) one, by an amount that depends on the direction q approaches
Γ from. `phonons` adds this non-analytic correction (Gonze & Lee, Phys.
Rev. B 55, 10355 (1997)) when:

- the irrep is at **Γ** and **IR-active** (other k-points: not applicable;
  `--born-charges` there is an error), and
- Born effective charges and the electronic dielectric tensor ε∞ are
  available from a VASP `LEPSILON=.TRUE.` calculation on the undistorted
  parent: `BEC/OUTCAR` is used automatically, or give `--born-charges PATH`.

It is only meaningful for an **insulator or semiconductor**: free carriers in
a metal screen the field, and Born charges are not well defined there. The
program cannot tell from a POSCAR, so the `BEC/` calculation is only set up
when asked (`irrep --bec` or `bec`).

The corrected spectrum is computed along x, y, z and any `--direction`;
directions related by the crystal's point group are grouped and computed once
(for cubic STO, x/y/z are one group and [111] another). The report, written
to `BEC/report.txt`, also lists the mode effective charges Z\* and this
irrep's contribution to the static dielectric constant ε(0) (generalised
Lyddane–Sachs–Teller sum). An unstable mode is left out of ε(0) and flagged.
`phonons` covers one irrep at a time; use `bec-analysis` for the total ε(0).

---

## `bec`

Finds every IR-active Γ-point irrep and runs `irrep --bec` for all of them:
the displaced structures for each, plus one `BEC/` directory per material
(the plain parent, `BEC/INCAR` with `LEPSILON=.TRUE.` and `IBRION=-1` forced,
`BEC/KPOINTS` copied unscaled). Re-running is safe; nothing already present is
changed. Leave the resulting `BEC/OUTCAR` where VASP wrote it.

```
smodes-phonons bec [--poscar POSCAR] [--name TAG] [--tolerance TOL]
    [--isodata PATH] [--kpoints KPOINTS] [--incar INCAR] [--magmom]
```

Flags as for [`irrep`](#irrep).

---

## `bec-analysis`

The complete Born-charge analysis, no irrep needed: prints the Born effective
charges Z\* and ε∞ from the `LEPSILON=.TRUE.` OUTCAR, then, for every
IR-active Γ irrep whose phonon calculation is done, runs `phonons
--born-charges` (LO-TO splitting) and adds that irrep's contribution to the
material's total ε(0). Irreps not computed yet are skipped and the total is
labelled a partial sum. The combined result is written to
`BEC/combined_report.txt`.

```
smodes-phonons bec-analysis [--born-charges OUTCAR] [--poscar POSCAR]
    [--name TAG] [--tolerance TOL] [--isodata PATH]
```

| flag | default | meaning |
|---|---|---|
| `--born-charges OUTCAR` | `BEC/OUTCAR` | the `LEPSILON=.TRUE.` output |
| `--poscar POSCAR` | `POSCAR` | labels Z\* by species and finds the IR-active irreps |
| others | | as for `irrep` |

---

## `freeze`

Freezes one or more modes into one structure, each along a chosen
order-parameter direction, and writes `Freeze_<...>/POSCAR`: a starting
structure for a relaxation, or one point for an energy-surface study.

```
smodes-phonons freeze IRREP [IRREP ...] [--smodes-basis] [--poscar POSCAR]
    [--name TAG] [--mode [LABEL=]INDEX] [--amplitude [LABEL=]VALUE]
    [--direction [LABEL=]SPEC] [--isodata PATH] [--kpoints KPOINTS]
    [--incar INCAR] [--magmom]
```

| argument / flag | default | meaning |
|---|---|---|
| `IRREP ...` | — | one or more labels; the same irrep twice is refused |
| `--smodes-basis` | off | use the SMODES mode instead of a DFT eigenvector (no `phonons` run needed) |
| `--mode [LABEL=]INDEX` | automatic if the irrep has one mode | which mode (eigenvector or SMODES mode); otherwise prompted for at a terminal |
| `--amplitude [LABEL=]VALUE` | 0.001 Å (0.01 Å with `--smodes-basis`) | amplitude per unit coefficient |
| `--direction [LABEL=]SPEC` | `a,0,...,0` | order-parameter direction, e.g. `a,0,0`, `a,a,0`, `0.5a,-a,0` |
| `--kpoints`, `--incar` | | as for `irrep`, except the INCAR is copied **unchanged**: a frozen structure is a relaxation seed, so your own `ISIF`/`IBRION`/`NSW` are kept |
| `--magmom` | off | `MAGMOM` rewritten for the written structure's atoms; also keeps the cell a whole multiple of the parent |

`LABEL=` can be left off when only one irrep is requested; with several, each
needs its own entry.

**Direction specs.** One position per component of the irrep, in ISOTROPY's
(and ISODISTORT's) order; each position is a coefficient on `a` (`a`, `-a`,
`2a`, `0.577a`) or `0`. `irrep --opd` lists what each position is. The written
displacement norm is amplitude × √(Σ coefficient²): `a,0,0` at 0.05 Å writes
0.05 Å, `a,a,0` writes 0.0707 Å. Start a spec with `-` as
`--direction=-a,0,0` so it is not read as a flag.

**Output** in `Freeze_<irrep>-m<mode>-sym-<...>/`: `POSCAR` (the smallest cell
the distortion needs), `direction.txt` (irrep, mode, amplitude, spec, and the
combined displacement norm), and, if findsym finds an even smaller equivalent
cell, the larger one as `POSCAR_superlattice` with `findsym_reduction.log`.
Magnetic structures (`--magmom`) are never reduced.

---

## `freeze-amplitude`

Like `freeze`, but sweeps the amplitude: one structure per point of a 1-D
sweep (one irrep) or a 2-D/3-D grid (two or three irreps, every combination).

```
smodes-phonons freeze-amplitude IRREP [IRREP [IRREP]] --start S --end E
    --steps N [--poscar POSCAR] [--name TAG] [--mode LABEL=INDEX]
    [--direction LABEL=SPEC] [--isodata PATH] [--kpoints KPOINTS]
    [--incar INCAR] [--magmom]
```

| flag | meaning |
|---|---|
| `--start S`, `--end E` | amplitude range in Å, the same for every irrep |
| `--steps N` | points per irrep including both ends (`--steps 5` with two irreps gives 25 structures) |
| `--incar INCAR` | static calculation forced (`NSW=0`, `IBRION=-1`, `ISYM=0`): each point is a single-point energy |
| others | as for `freeze` |

Writes `Phonon_<label>_<mode>[...]_Amplitude/Point_<i>_.../POSCAR` and a
`grid.txt` table of the amplitudes, ready for an energy column and
`landau-fit`.

---

## `decomposition`

Decomposes a distorted structure (e.g. a relaxed CONTCAR, or one written by
ISODISTORT or `freeze`) into the amplitude of every order-parameter direction
of every irrep at the requested k-points, plus the strain relating it to the
parent and a residual for what those irreps do not explain.

```
smodes-phonons decomposition --parent POSCAR --sub POSCAR --k_vector K [K ...]
    [--name TAG] [--tolerance TOL] [--findsym-tolerance TOL] [--isodata PATH]
```

| flag | default | meaning |
|---|---|---|
| `--parent POSCAR` | — | the high-symmetry parent (give the plain parent, not a supercell; a supercell is warned about) |
| `--sub POSCAR` | — | the distorted structure; any supercell of the parent, strained or not, at any origin or symmetry-rotated orientation |
| `--k_vector K ...` | — | one or more k-points (`GM X R`); every irrep at each is checked |
| `--name TAG` | | material tag |
| `--tolerance TOL` | 0.3 Å | atom-matching tolerance; widen to ~0.5 for a large (ferroelectric-scale) distortion |
| `--findsym-tolerance TOL` | `0` | findsym tolerance for the parent and the distorted structure |

**The screen** shows one line per k-point per irrep detected (largest first,
with its OPD and `Ap`), or `no irrep amplitudes detected`. **The full report**
is in `decomposition_mode_<K>`, one per k-point: space groups, strain and
det(F), the origin shift if one was applied, the total displacement and the
residual, a `Mode decomposition:` table (OPD and amplitude per mode) and a
`Per-direction decomposition:` table with `As` and `Ap` per direction and for
all components together. `decomposition_summary` collects every k-point.

`As` is the norm over the cell used; `Ap = As/√N` (N = cell volume in
primitive parent cells) is independent of the cell and is the value to
compare with ISODISTORT. See `OPD_CONSTRUCTION.md` for the Γ-point polar
caveat.

---

## `landau-fit`

Fits the coefficients of a Landau free-energy expansion to DFT total energies
by linear least squares (each invariant is a fixed polynomial in the order
parameters). A constant offset is fitted automatically.

```
smodes-phonons landau-fit (--invariants FILE | --terms TERMS) --data FILE
    [--predict] [--bound B] [--n-starts N]
```

| flag | default | meaning |
|---|---|---|
| `--invariants FILE` | — | INVARIANTS text copied from ISOTROPY/ISODISTORT |
| `--terms TERMS` | — | instead, a term list such as `a2,b2,abc,a4,a2b2` |
| `--data FILE` | — | tab- or comma-separated table: one column per order parameter plus `energy` |
| `--predict` | off | also report the order-parameter values that minimise the fitted energy |
| `--bound B` | 1.0 | random starting points for `--predict` are drawn from [−B, B] |
| `--n-starts N` | 25 | number of starting points |

---

## `landau-grid`

Generates the `freeze-amplitude` grid for the irreps named in an INVARIANTS
file, with `grid.txt` columns named by the order-parameter symbols
(`a`, `b`, …), so that adding an `energy` column makes it a `landau-fit --data`
file.

```
smodes-phonons landau-grid --invariants FILE --start S --end E --steps N
    [--poscar POSCAR] [--name TAG] [--mode LABEL=INDEX]
    [--direction LABEL=SPEC] [--isodata PATH] [--kpoints KPOINTS]
    [--incar INCAR] [--magmom]
```

`--mode` and `--direction` are keyed by the irrep **label**
(`--direction GM4-=a,0,0`), not the order-parameter symbol; the printed
`Order parameters: a=GM5-, ...` line gives the mapping. Every irrep needs
`phonons` first. Other flags as for `freeze-amplitude`.

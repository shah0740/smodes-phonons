# Tutorial: the ferroelectric soft mode of SrTiO3

*Written by Claude (Anthropic). Every output below is real, from a run of
smodes-phonons v2.0.0 on cubic SrTiO3; the VASP force calculations themselves
are not shown.*

This walks through the whole workflow for one irrep, GM4- (T1u) of cubic
SrTiO3 (Pm-3m, a = 3.901 Å):

    symmetry → irrep → (VASP) → phonons → freeze → decomposition → CIF

Start in an empty directory containing only the parent `POSCAR`:

```
SrTiO3 Cubic Perovskite
3.901000
  1.000000  0.000000  0.000000
  0.000000  1.000000  0.000000
  0.000000  0.000000  1.000000
Sr Ti O
  1  1  3
Direct
  0.000000  0.000000  0.000000
  0.500000  0.500000  0.500000
  0.500000  0.500000  0.000000
  0.500000  0.000000  0.500000
  0.000000  0.500000  0.500000
```

## 1. Symmetry and the irreps at Γ

```
$ smodes-phonons symmetry POSCAR --name STO
Wrote STO_findsym
```

`irrep` would create the Γ-point smodes output itself; to see it first:

```
$ smodes-phonons run-smodes STO_findsym GM --name STO
Wrote STO_smodes_GM
$ smodes-phonons smodes STO_smodes_GM
STO_findsym  (space group 221 Pm-3m)
a,b,c,alpha,beta,gamma = 3.901, 3.901, 3.901, 90, 90, 90
Wyckoff: Sr a, Ti b, O c
2 irrep(s):
  GM4- T1u  k=GM  degeneracy=3  modes=4 (total 12)  atoms=5  [3 translational, IR active]
  GM5- T2u  k=GM  degeneracy=3  modes=1 (total 3)  atoms=5
```

GM4- is three-dimensional (x, y and z components) with four distinct modes,
one of them the acoustic translation.

## 2. The displaced structures for DFT

```
$ smodes-phonons irrep GM4- --opd --name STO
GM4-: wrote 4 mode(s) to GM/Irrep_GM4-/
Wrote GM/Irrep_GM4-/opd.txt
GM4-: this is an IR-active Gamma-point mode -- LO-TO (non-analytic) splitting is symmetry-allowed. ...
```

```
GM/Irrep_GM4-/
├── Mode_0/POSCAR   Mode_1/POSCAR   Mode_2/POSCAR   Mode_3/POSCAR
├── Undistorted/POSCAR
└── opd.txt
```

Each `Mode_i/POSCAR` is the parent displaced by 0.001 Å along one SMODES mode.
Pass `--incar INCAR` and `--kpoints KPOINTS` to have a static-calculation
INCAR and a correctly scaled mesh written next to each one.

`opd.txt` lists the three components of each mode, which are the positions
`--direction` takes later:

```
  a  --direction a,0,0   [arm 1, native]
  b  --direction 0,a,0   [arm 1, within]
  c  --direction 0,0,a   [arm 1, within]
```

## 3. Run VASP

Run a static force calculation in every `Mode_i/` (and in `Undistorted/` if the
parent has free Wyckoff parameters; it does not here), leaving each `OUTCAR`
where it is.

## 4. Frequencies and eigenvectors

```
$ smodes-phonons phonons GM4- --name STO
GM4- (4 modes):
  mode        cm-1       THz       meV    fc eigval (eV/A^2)
     0     -66.547    -1.995    -8.250               -0.4777
     1       2.838     0.085     0.352                0.0010
     2     157.022     4.707    19.467                5.8380
     3     534.759    16.032    66.298               16.9207
Wrote GM/Irrep_GM4-/phonons.txt
Wrote GM/Irrep_GM4-/phonons.npz
Wrote GM/Irrep_GM4-/POSCAR_superlattice (1x1x1, 5 atoms)
Wrote GM/Irrep_GM4-/phonons_superlattice.txt
```

Mode 0 is unstable (−66.5 cm⁻¹, printed as a negative frequency): the
ferroelectric soft mode. Mode 1 is the acoustic mode (≈ 0).

**LO-TO splitting.** GM4- is IR-active, so if the material is an insulator
(SrTiO3 is) you can add the LO-TO correction: `irrep GM4- --bec` (or `bec`)
writes `BEC/`; run VASP with `LEPSILON = .TRUE.` there and leave the
`OUTCAR` in `BEC/`; `phonons GM4-` then picks it up automatically and writes
`BEC/report.txt`. See [the command reference](COMMANDS.md#lo-to-splitting).

## 5. Freeze the soft mode along (a,a,a)

```
$ smodes-phonons freeze GM4- --mode 0 --direction=a,a,a --amplitude 0.05 --name STO
Wrote Freeze_GM4--m0-sym-1a-2a-3a/POSCAR (5 atoms)
```

`--mode 0` takes the DFT eigenvector of mode 0 from `phonons.npz`; use
`--smodes-basis` instead to freeze a SMODES mode before any DFT exists. The
amplitude applies to each component, so the written displacement is
0.05 × √3 Å, as recorded in `direction.txt`:

```
irrep       GM4-
mode        0
amplitude   0.05 A
direction   a,a,a

position  coefficient  mechanism
       1            1  native
       2            1  within
       3            1  within

combined pattern norm   0.0866025404 A
```

## 6. Decompose it back

```
$ smodes-phonons decomposition --parent POSCAR \
      --sub Freeze_GM4--m0-sym-1a-2a-3a/POSCAR --k_vector GM --name STO
GM: GM4-  OPD a,a,a  Ap 0.086603 A
Full reports: decomposition_mode_<K>; summary: decomposition_summary
```

The full report, `decomposition_mode_GM`:

```
Parent:    space group 221 (Pm-3m), a=3.90100 b=3.90100 c=3.90100 alpha=90.000 beta=90.000 gamma=90.000
Distorted: space group 160 (R3m), a=5.51685 b=5.51685 c=6.75673 alpha=90.000 beta=90.000 gamma=120.000

=== k = GM ===
Origin: same as the parent

Det(F) = 1.000000   (volume ratio)
...
Total displacement norm: 0.086603 A
Residual (not explained by the irreps below): 0.000000 A (0.00%)

Mode decomposition:
  Irrep  Mode  Species  Degeneracy  OPD    OPD construction  As, per component (A)  Ap, per component (A)
  GM4-      0  Sr                3  a,a,a  3 x within-arm    a = -0.005474 A        a = -0.005474 A
  GM4-      1  Ti                3  a,a,a  3 x within-arm    a = -0.012539 A        a = -0.012539 A
  GM4-      2  O                 3  a,a,a  3 x within-arm    a = 0.019594 A         a = 0.019594 A
  GM4-      3  O                 3  a,a,a  3 x within-arm    a = 0.043919 A         a = 0.043919 A
  GM5-      0  O                 3  0,0,0  3 x within-arm    (below noise floor)

Per-direction decomposition:
  Irrep      OPD                   As (A)       Ap (A)
  GM4-       a,0,0               0.050000     0.050000
  GM4-       0,a,0               0.050000     0.050000
  GM4-       0,0,a               0.050000     0.050000
  GM4-       all components      0.086603     0.086603
```

What to read from it:

- The distorted structure is **R3m (160)**, the rhombohedral ferroelectric
  phase that (a,a,a) of T1u produces.
- The residual is 0: GM4- explains the whole distortion.
- Each SMODES mode carries the same OPD, `a,a,a`: the DFT eigenvector is a
  fixed mix of the four SMODES modes (Sr, Ti and two O patterns), all along
  the same direction.
- 0.05 Å per component, 0.0866 Å in total: exactly what was frozen.

## 7. A CIF for ISODISTORT

```
$ mkdir cif && cp Freeze_GM4--m0-sym-1a-2a-3a/POSCAR cif/ && cd cif
$ smodes-phonons symmetry POSCAR --tolerance 0.005 --name frozen
Wrote frozen_findsym
$ grep "Space Group" frozen_findsym
Space Group: 160  C3v-5     R3m
$ sed -n '/^# CIF file created by FINDSYM/,/^# end of cif/p' frozen_findsym > STO_GM4-_m0_aaa.cif
```

A non-zero tolerance is needed for a frozen structure. Check the space group
is not the parent's: at too small an amplitude or too loose a tolerance,
findsym "sees through" the distortion. Upload the CIF to ISODISTORT
(Method 4, against the parent) to compare: expect the same subgroup and the
GM4- order parameter `(a,a,a)`, or a symmetry-equivalent domain of it. For
a polar Γ mode like this one, ISODISTORT's `Ap` differs from ours because it
holds its first-listed atom (Sr) fixed; see `OPD_CONSTRUCTION.md`.

## Where next

- Other directions: `--direction=a,0,0` (P4mm, 99), `--direction=a,a,0`
  (Amm2, 38).
- Zone-boundary irreps work the same way (`irrep M`, `irrep R`); the frozen
  cell is then a supercell, chosen automatically.
- Coupling two irreps (e.g. X2+ and X3- in Ca3Mn2O7):
  `freeze X2+ X3- --direction X2+=a,0 --direction X3-=0,a ...`, and
  energy surfaces with `freeze-amplitude`, `landau-grid` and `landau-fit`.

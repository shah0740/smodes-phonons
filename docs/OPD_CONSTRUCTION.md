# How order-parameter directions are calculated

*Written by Claude (Anthropic), from the smodes-phonons v2.0.0 source.*

A degenerate irrep of dimension *d* has *d* independent components: *d*
orthogonal displacement patterns that transform into one another under the
crystal's symmetry. Any distortion that transforms as that irrep is a
combination of them, written as an **order-parameter direction (OPD)** such as
`a,0,0`, `a,a,0` or `a,a,a`: one coefficient per component. This file explains
how smodes-phonons builds those components, and what the letters therefore
mean.

## The starting point: one pattern per mode

SMODES (ISOTROPY Software Suite) gives, for each irrep at a k-point, a
symmetry-adapted basis: one displacement pattern per *distinct mode*, which is
one component of that mode. A phonon calculation then mixes the modes of an
irrep into eigenvectors, but each eigenvector is still a single component. The
other *d − 1* components are not printed, and the operation matrices that
would generate them are not in the findsym or SMODES output.

## The construction: a projection with ISOTROPY's own matrices

smodes-phonons constructs every component with the projection (transfer)
operator of group theory:

```math
\mathbf{w}_j \;=\; \frac{d}{|G|}\sum_{g\in G} D_{jk}(g)\;\big(g\cdot\mathbf{s}\big)
```

| symbol | meaning |
|---|---|
| $\mathbf{s}$ | the **seed**: the SMODES pattern of the mode, or a DFT eigenvector (a fixed combination of the modes), tiled onto the working cell |
| $G$ | every symmetry operation of the parent crystal *modulo the cell*: rotations and screws/glides, **and** the lattice and centring translations inside the cell |
| $g\cdot\mathbf{s}$ | the seed after operation $g$: atoms permuted, and each atom's displacement rotated |
| $D(g)$ | ISOTROPY's own $d\times d$ matrix for $g$ in this irrep |
| $k$ | the component the seed belongs to; $j$ runs over all $d$ components |

The translations in $G$ matter: at a zone-boundary k-point a lattice
translation multiplies a component by a sign or a phase, and that is what
separates components on different star arms and what produces the Bloch
quadrature partners at a k-point such as the hexagonal K.

Because the matrices are ISOTROPY's, **the resulting basis is ISOTROPY's, in
ISOTROPY's component order.** An OPD read off ISOTROPY or ISODISTORT means the
same distortion when typed into `freeze --direction`, and `decomposition`
reports in the same letters.

## Where each ingredient comes from

| ingredient | source |
|---|---|
| symmetry operations | the `iso` program: `value lattice param`, `show generators`, `show cartesian`. The printed 3×3 matrix is the transpose of the matrix acting on column vectors; the exact fractional translation is read from iso's own label (its Ångström printout is rounded, which would stop the group from closing). The full group is the closure of the generators. |
| irrep matrices $D(g)$ | `iso`: one session per irrep, displaying the matrix of each generator and of each lattice and centring translation; entries are exact values (0, ±1/2, ±√3/2, ±1/√2, ±1). |
| frame: ISOTROPY's setting → your POSCAR | findsym's own `Origin at` and `Vectors a,b,c:` lines, f<sub>input</sub> = Mᵀ f<sub>standard</sub> + origin. findsym and ISOTROPY share BYU's settings, so this is exact, including a POSCAR whose axes are permuted relative to the standard setting (e.g. Pnma supplied as Pbnm). Every generator is then checked to be a symmetry of your structure. |
| origin | one origin for every irrep at a k-point, chosen so that each seed projects onto a non-zero component |

`iso` output is cached the first time it is used, under
`~/.cache/smodes_phonons/iso` (or the directory in `SMODES_ISO_CACHE`): it
depends only on the space group and irrep, never on your structure. Only
genuine ISOTROPY output is cached, and deleting the folder is harmless.

## The safety check

Every constructed basis must be orthonormal **and** transform exactly by
ISOTROPY's matrices: applying each operation $g$ to the components must
reproduce $D(g)$ to 10⁻⁵. If it does not (a wrong frame, a wrong operation, an
irrep label in a different setting) the construction stops with an error. It
cannot quietly produce a plausible-looking wrong basis.

## Arms, within-arm partners and quadrature

These are read off ISOTROPY's translation matrices, not searched for:

- **Star arms.** At a k-point whose star has several members (e.g. the three M
  points of a cubic crystal), each lattice translation multiplies a component
  by +1 or −1. Components with the same sign pattern lie on the same arm.
- **Within-arm partners.** Several components with the same sign pattern:
  directions on one arm, related by the arm's own little group (e.g. the x, y
  and z components of a polar Γ mode).
- **Quadrature partners.** Where k and −k are distinct (e.g. hexagonal K), a
  translation *rotates* a pair of components into each other instead of
  changing a sign. The pair is one arm in ISOTROPY's notation: the second
  member is the 90° Bloch-phase partner of the first.

The `irrep --opd` and `phonons --generate-opd` reports label every position
this way (`arm 1, native`, `within`, `arm2`, `quadrature`, …) and print each
arm's wavevector `Q^(1), Q^(2), …`, also taken from ISOTROPY's translation
matrices. Of the two equivalent forms q and −q (the same real wave), the one
with fewer negative components is printed.

## Cells

- **`freeze`** writes the smallest cell on which the requested OPD is
  periodic: the cell spanned by every parent translation T with
  D(T)·v = v, where v is the OPD's coefficient vector. For a centred parent
  this can be smaller than your conventional POSCAR. **With `--magmom`, only
  whole lattice translations are used** (moments can break the centring), so a
  magnetic structure's cell is always a whole multiple of the parent cell. The
  cell is then written on its shortest, most nearly orthogonal basis, ordered
  and signed to follow your parent's axes.
- **`decomposition`** builds one cell per k-point hosting every OPD of every
  irrep there, matched to the distorted structure you give it (enlarged or
  folded as needed). Your distorted structure is aligned to it once, with one
  origin-shift search and one strain, and each irrep's amplitudes are solved
  by least squares against the same displacement. Different irreps are
  exactly orthogonal on that shared cell (Schur orthogonality), so one irrep's
  amplitude never leaks into another's.

## What the letters mean

- One position per component, in ISOTROPY's order; the spec length must equal
  the irrep's dimension (a 1-dimensional irrep takes a bare `a`).
- Each position is a coefficient on the letter `a`: `a`, `-a`, `2a`,
  `0.577a`; `0` leaves that component out.
- Coefficients are relative weights; `--amplitude` scales each component, so
  the written structure's displacement norm is amplitude × √(Σ coefficient²).
  `a,0,0` at 0.05 Å writes 0.05 Å; `a,a,0` writes 0.0707 Å.
- With no `--direction`, `freeze` uses ISOTROPY's first component,
  `a,0,…,0`. That need not be SMODES's printed pattern, which is generally a
  different direction inside the same degenerate subspace.
- `decomposition` reports in your parent's frame. A structure that is a
  symmetry-rotated copy of the parent is the symmetry-equivalent domain: same
  amplitude, same space group, possibly different letters. The report says so
  when it detects this.

## Comparing with ISODISTORT

- **Space group and OPD.** Expect the same subgroup, and the same OPD or a
  symmetry-equivalent domain. ISODISTORT reports its own standard domain, so a
  structure frozen along `0,a,0` can come back as `a,0,0`; that is the same
  physical state.
- **`As` and `Ap`.** `decomposition` prints both of ISODISTORT's
  normalisations: `As`, the norm over the cell used, and
  `Ap = As / √N` with N the cell's volume in primitive parent cells. Compare
  `Ap` across programs.
- **Γ-point polar irreps.** A uniform translation transforms like a polar Γ
  mode, so the split of amplitude between modes depends on which atom is held
  fixed. ISODISTORT holds the first site of its parent atom list fixed;
  smodes-phonons does not. Pin the same site before comparing `Ap` for these
  irreps.
- **A parameter that appears in several components** (e.g. `a` in
  `(a,b,-a)`) carries one `As` covering all of them; the value per component
  is As × coefficient / √(Σ coefficient²).

## Validation (v2.0.0)

Before release the construction was checked against independent data:

- 520 frozen structures from an earlier campaign (STO, NaCl, Si, CaTiO3,
  Ca3Mn2O7, BaTiO3, CsV3Sb5, graphite) decompose to ISODISTORT's OPD, either
  identically or as a symmetry-equivalent domain;
- 269 structures frozen at ISODISTORT's own letters reach ISODISTORT's
  subgroup (findsym) and decompose back to exactly those letters;
- for monoclinic HfO2, every primary direction in ISOTROPY's subgroup table
  reaches ISOTROPY's subgroup (144 of 144).

## Further reading

- H. T. Stokes, D. M. Hatch and B. J. Campbell, ISOTROPY Software Suite,
  iso.byu.edu.
- B. J. Campbell, H. T. Stokes, D. E. Tanner and D. M. Hatch, "ISODISPLACE: a
  web-based tool for exploring structural distortions", J. Appl. Cryst. 39,
  607 (2006).
- `docs/PHONON_THEORY.pdf` for the phonon and group-theory background.

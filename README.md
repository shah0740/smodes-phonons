# smodes-phonons

*This README was written by Claude (Anthropic).*

Symmetry-adapted frozen-phonon calculations driven by the ISOTROPY Software
Suite.

smodes-phonons takes a crystal structure, uses ISOTROPY's `findsym` and
`smodes` to build the symmetry-adapted displacement basis of any irrep at any
k-point, writes the displaced structures for a DFT force calculation, and
turns the forces into phonon frequencies and eigenvectors, one irrep (one
block of the dynamical matrix) at a time. From there it builds distorted
structures along any order-parameter direction of an irrep, decomposes real
distorted or relaxed structures back into symmetry-mode amplitudes, and fits
Landau free-energy expansions to DFT energies.

Order-parameter directions are constructed from ISOTROPY's own irrep
matrices, so the letters (`a,0,0`, `a,a,0`, …) mean the same as in ISOTROPY
and ISODISTORT.

**What it does**

- **Phonons by symmetry block.** Displaced structures for one irrep, then
  frequencies (unstable modes as negative cm⁻¹) and eigenvectors from the
  forces.
- **LO-TO splitting** at Γ for polar insulators, with Born effective charges,
  mode effective charges and the static dielectric constant ε(0).
- **Frozen distortions** (`freeze`) of one or several modes, along any
  order-parameter direction, on the smallest cell the distortion needs;
  amplitude sweeps for energy surfaces.
- **Mode decomposition** (`decomposition`) of a relaxed or external
  structure into irrep amplitudes (`As` and `Ap`, as in ISODISTORT), with the
  strain and any origin shift reported.
- **Landau fitting** of invariant-polynomial coefficients to DFT energies,
  with a prediction of the equilibrium distortion.

The DFT calculator is VASP. Running the DFT calculations is up to you;
smodes-phonons writes their inputs and reads their outputs.

## Requirements

- Python 3.10 or later, and NumPy.
- **The ISOTROPY Software Suite** command-line programs `findsym`, `smodes`
  and `iso`, from Brigham Young University. They are not included here and
  are distributed by BYU under their own terms. smodes-phonons is developed
  against FINDSYM 7.1.6.
- VASP, to run the force calculations.

## Installation

### 1. smodes-phonons

```
git clone https://github.com/shah0740/smodes-phonons.git
cd smodes-phonons
pip install -e .
```

This installs the `smodes-phonons` command.

### 2. The ISOTROPY suite on Linux

1. Download the Linux command-line suite from
   [iso.byu.edu](https://iso.byu.edu/iso/isolinux.php) and unzip it, e.g. to
   `~/software/isobyu/`.
2. Put the directory holding `findsym`, `smodes` and `iso` on your `PATH`.
3. Point `ISODATA` at the data directory, **with a trailing slash** (findsym
   joins it directly to file names):

   ```
   export PATH="$HOME/software/isobyu:$PATH"
   export ISODATA="$HOME/software/isobyu/"
   ```

   (`--isodata PATH` on any command overrides the environment variable.)

### 3. The ISOTROPY suite on macOS

BYU's binaries are Linux x86-64 only. `tools/setup-macos-docker.sh` installs
small wrappers in `~/.local/bin` that run them inside a Docker container, so
`findsym`, `smodes` and `iso` work from an ordinary macOS shell:

1. Install Docker Desktop, and enable *Settings → General → Use Rosetta for
   x86_64/amd64 emulation* on Apple Silicon.
2. Download and unzip the Linux suite as above.
3. Run, with `ISOBYU_DIR` pointing at the unzipped suite:

   ```
   ISOBYU_DIR=~/software/isobyu ./tools/setup-macos-docker.sh
   ```

4. Make sure `~/.local/bin` is on your `PATH`, and set `ISODATA` as above.

Docker Desktop must be running whenever smodes-phonons calls the ISOTROPY
programs.

### 4. Check

```
smodes-phonons doctor
```

reports whether NumPy, `findsym` and `smodes` are found, the findsym version,
and how `ISODATA` was resolved.

## Quick start

From a directory containing your parent `POSCAR`:

```
smodes-phonons irrep GM4-              # displaced structures for one irrep
#   ... run VASP in each GM/Irrep_GM4-/Mode_i/ ...
smodes-phonons phonons GM4-            # frequencies and eigenvectors
smodes-phonons freeze GM4- --mode 0 --direction=a,a,a --amplitude 0.05
smodes-phonons decomposition --parent POSCAR \
    --sub Freeze_GM4--m0-sym-1a-2a-3a/POSCAR --k_vector GM
```

A bare k-point (`irrep GM`, `irrep M`) means every irrep at that k-point.
`irrep GM4- --opd` writes a report of every order-parameter direction of the
irrep and what each `--direction` position is.

## Documentation

| file | contents |
|---|---|
| [`docs/TUTORIAL.md`](docs/TUTORIAL.md) | a complete worked example: the ferroelectric soft mode of SrTiO3 |
| [`docs/COMMANDS.md`](docs/COMMANDS.md) | every command and flag |
| [`docs/OPD_CONSTRUCTION.md`](docs/OPD_CONSTRUCTION.md) | how order-parameter directions are calculated, and comparing with ISODISTORT |
| [`docs/PHONON_THEORY.md`](docs/PHONON_THEORY.md) | phonons and phonon symmetry: the theory behind the method (Jay Shah) |

## Citing

If you use smodes-phonons, please cite the ISOTROPY Software Suite, which it
depends on (see [iso.byu.edu](https://iso.byu.edu) for BYU's current
citation requests):

- H. T. Stokes, D. M. Hatch, and B. J. Campbell, *ISOTROPY Software Suite*,
  iso.byu.edu.
- H. T. Stokes and D. M. Hatch, "FINDSYM: program for identifying the
  space-group symmetry of a crystal", J. Appl. Cryst. **38**, 237 (2005).
- H. T. Stokes, D. M. Hatch, and J. D. Wells, "Group-theoretical methods for
  obtaining distortions in crystals: Applications to vibrational modes and
  phase transitions", Phys. Rev. B **43**, 11010 (1991) (SMODES).

## Authors

Jay Shah (design, physics, validation; `legacy/` and `docs/PHONON_THEORY.md`).
The code, tests and the other documentation were written with Claude
(Anthropic), which also wrote this README.

## License

MIT; see [`LICENSE`](LICENSE). The ISOTROPY Software Suite is not part of
this repository and is covered by BYU's own terms.

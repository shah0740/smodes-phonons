"""Validated against real STO GM4-/GM5- data (see decomposition.py's module
docstring): injecting a known combination of two GM4- modes and decomposing
recovers those exact amplitudes with ~1e-16 residual, the other GM4- modes
and the orthogonal GM5- irrep both come back ~0, and the method stays
accurate within the injected noise level when the distorted structure's
atom ordering is shuffled and small synthetic noise is added.

v1 -> v2: `decompose_structure` (one call, every k-vector at once) no
longer exists -- the per-k-vector orchestration now lives in the CLI
(`_run_decomposition` in cli.py), which calls `decompose_at_k_vector` once
per requested k-vector with that k-vector's own already-parsed irrep list
and an already-computed `stoichiometry_multiplier` (`check_stoichiometry`).
Every test here mirrors that same calling convention directly.

Also new in v2: a triply-degenerate mode's amplitude is reported once per
swept direction (`native`, plus `(1,0,0)`/`(0,1,0)`/`(0,0,1)` for
degeneracy 3 -- see `mode_direction_sweep`), not just once. Tests ported
from v1 that only care about the single, single-direction amplitude filter
on `direction == "native"` to recover the old semantics exactly (`native`
and `(1,0,0)` are numerically identical for STO GM4-, whose own native
direction already lies along x -- see the developer notes).
"""

import unittest
from pathlib import Path

import numpy as np

from smodes_phonons import Structure, parse_smodes_output
from smodes_phonons.decomposition import (
    DecompositionError,
    check_stoichiometry,
    decompose_at_k_vector,
)
from smodes_phonons.supercell import (
    SupercellError,
    nice_supercell_matrix,
    enlarge_onto_lattice,
    nice_diagonal_multipliers,
    tile_onto_supercell,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def load(name: str) -> str:
    return (FIXTURES / name).read_text()




def _inject_single_irrep(parent, irrep, amplitude, mode_index=0, position=0):
    """A structure distorted by `amplitude` (A) along one OPD position of one
    mode, built the way the ACTIVE basis builds it.

    v2.0 (ISOTROPY-matrix basis): the parent tiled onto the irrep's cell plus
    the constructed pattern -- referenced to the PARENT POSCAR, as a real
    `freeze` output or relaxed CONTCAR is. The old route tiled SMODES's own
    idealised coordinates, which differ from the POSCAR by up to 5.6e-4 A
    (real CMO, free Wyckoff parameters re-derived by SMODES) and so leave a
    residual against the parent. Old basis: SMODES's own pattern, tiled."""
    from smodes_phonons import isotropy_basis as ib
    joint = ib.joint_basis([(irrep, ib.seeds_of(irrep))], parent, irrep.space_group_number, None)
    return joint.reference.displaced(amplitude * joint.bases[irrep.label][mode_index][position])
    multipliers = nice_supercell_matrix(irrep.superlattice, parent.lattice)
    structure, sets = tile_onto_supercell(
        irrep, parent.lattice, multipliers,
        {irrep.label: [mode.displacements for mode in irrep.modes]}, comment="frozen single irrep",
    )
    return structure.displaced(amplitude * sets[irrep.label][mode_index])


def _native_amplitudes(result, label: str) -> dict:
    """Each mode's amplitude along its NATIVE direction.

    Selects on `mechanism`, not on the direction NAME. Since the OPD
    unification a direction is named by its position letter (`a`, `b`, ...)
    -- the same letter `irrep --opd` prints and `freeze --direction`
    accepts -- so `direction == "native"` matches nothing and every caller
    of this helper silently receives an empty dict. The mechanism is what
    actually carries "this is the native one", and it survives the rename
    by design: a position names a direction regardless of how it was
    reached.

    Falls back to the old name for any decomposition that still reports the
    older strings (no OPD basis available for that k-point).
    """
    by_mechanism = {
        ma.mode_index: ma.amplitude_angstrom
        for ma in result.amplitudes
        if ma.label == label and getattr(ma, "mechanism", "") == "native"
    }
    if by_mechanism:
        return by_mechanism
    return {
        ma.mode_index: ma.amplitude_angstrom
        for ma in result.amplitudes
        if ma.label == label and ma.direction == "native"
    }


class TestDecomposeAtKVector(unittest.TestCase):
    def setUp(self):
        self.parent = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        out = parse_smodes_output(load("STO_findsym_Gamma"))
        self.gm4 = out.irrep("GM4-")
        self.gm5 = out.irrep("GM5-")
        self.multiplier = nice_diagonal_multipliers(self.gm4.superlattice, self.parent.lattice)
        disp_sets = {"gm4": [m.displacements for m in self.gm4.modes]}
        self.undistorted, big_sets = tile_onto_supercell(
            self.gm4, self.parent.lattice, self.multiplier, disp_sets, comment="undistorted"
        )
        self.tiled_gm4_modes = big_sets["gm4"]

    def _distorted(self, injected: dict) -> Structure:
        cart_disp = sum(amp * self.tiled_gm4_modes[i] for i, amp in injected.items())
        return self.undistorted.displaced(cart_disp)

    def test_exact_recovery_of_injected_amplitudes(self):
        injected = {2: 0.003, 3: -0.001}
        distorted = self._distorted(injected)

        result = decompose_at_k_vector("GM", [self.gm4, self.gm5], self.parent, distorted, stoichiometry_multiplier=1)

        by_index = _native_amplitudes(result, "GM4-")
        for i in range(4):
            expected = injected.get(i, 0.0)
            self.assertAlmostEqual(by_index[i], expected, places=9)

        gm5_amplitudes = [ma.amplitude_angstrom for ma in result.amplitudes if ma.label == "GM5-"]
        for amp in gm5_amplitudes:
            self.assertAlmostEqual(amp, 0.0, places=9)

        self.assertAlmostEqual(result.residual_norm_angstrom, 0.0, places=9)
        np.testing.assert_array_almost_equal(result.multiplier, np.diag(self.multiplier))
        self.assertEqual(result.arm_index, 1)
        self.assertEqual(result.arm_count, 1)

    def test_recovers_exact_amplitudes_despite_an_arbitrary_origin_shift(self):
        """`internal_displacement`'s automatic origin-shift search
        (`_search_origin_shift`) should recover the SAME exact amplitudes
        as the zero-shift case above, given a `sub` whose origin doesn't
        match the parent's own convention at all -- a generic, non-
        symmetry-related shift (0.13, 0.27, 0.61), not one of the
        "obvious" half-integer values a hardcoded centering-vector list
        would need, to prove the search really is data-derived rather than
        a lookup table."""
        injected = {2: 0.003, 3: -0.001}
        distorted = self._distorted(injected)

        shift = np.array([0.13, 0.27, 0.61])
        shifted_frac = (distorted.frac_coords + shift) % 1.0
        shifted = Structure(
            comment="origin-shifted", lattice=distorted.lattice, species=distorted.species, frac_coords=shifted_frac
        )

        result = decompose_at_k_vector("GM", [self.gm4, self.gm5], self.parent, shifted, stoichiometry_multiplier=1)

        self.assertIsNotNone(result.origin_shift_frac)
        np.testing.assert_allclose(result.origin_shift_frac % 1.0, shift, atol=1e-6)

        by_index = _native_amplitudes(result, "GM4-")
        for i in range(4):
            expected = injected.get(i, 0.0)
            self.assertAlmostEqual(by_index[i], expected, places=6)
        self.assertAlmostEqual(result.residual_norm_angstrom, 0.0, places=6)

    def test_robust_to_shuffled_atom_order_and_noise(self):
        injected = {2: 0.003, 3: -0.001}
        distorted = self._distorted(injected)

        rng = np.random.default_rng(1)
        noise = rng.normal(scale=0.0001, size=(distorted.n_atoms, 3))
        noisy = distorted.displaced(noise)

        shuffle = rng.permutation(noisy.n_atoms)
        shuffled = Structure(
            comment="shuffled",
            lattice=noisy.lattice,
            species=[noisy.species[i] for i in shuffle],
            frac_coords=noisy.frac_coords[shuffle],
        )

        result = decompose_at_k_vector("GM", [self.gm4, self.gm5], self.parent, shuffled, stoichiometry_multiplier=1)
        by_index = _native_amplitudes(result, "GM4-")

        # Noise is ~0.1 mA against a 1-3 mA signal -- recovered amplitudes
        # should land close to the injected values, well within the noise.
        self.assertAlmostEqual(by_index[2], 0.003, delta=0.0005)
        self.assertAlmostEqual(by_index[3], -0.001, delta=0.0005)
        self.assertAlmostEqual(by_index[0], 0.0, delta=0.0005)
        self.assertAlmostEqual(by_index[1], 0.0, delta=0.0005)

    def test_zero_distortion_gives_zero_amplitudes(self):
        result = decompose_at_k_vector("GM", [self.gm4], self.parent, self.undistorted, stoichiometry_multiplier=1)
        # Every swept direction, not just native -- a zero distortion has
        # zero projection onto every candidate direction.
        for ma in result.amplitudes:
            self.assertAlmostEqual(ma.amplitude_angstrom, 0.0, places=9)
        self.assertAlmostEqual(result.displacement_norm_angstrom, 0.0, places=9)

    def test_mode_amplitudes_carry_species_and_degeneracy(self):
        distorted = self._distorted({2: 0.003})
        result = decompose_at_k_vector("GM", [self.gm4], self.parent, distorted, stoichiometry_multiplier=1)
        for ma in result.amplitudes:
            self.assertEqual(ma.degeneracy, 3)
        species_by_index = {
            ma.mode_index: ma.species for ma in result.amplitudes
            if (getattr(ma, "mechanism", "") == "native" or ma.direction == "native")
        }
        self.assertEqual(species_by_index, {0: "Sr", 1: "Ti", 2: "O", 3: "O"})

    def test_degeneracy_three_sweeps_cartesian_axes(self):
        """New in v2: a degeneracy-3 mode gets one amplitude per swept
        direction (native + its within-arm partners), not just one -- see
        `mode_direction_sweep`'s own docstring for why this is needed to
        tell a pure (a,0,0) distortion apart from a real (a,a,a) one.

        GM4-'s own native direction lies exactly along x (documented in
        the developer notes) -- `mode_direction_sweep` no longer also reports that
        SAME direction a second time under its own canonical Cartesian
        label "(1,0,0)" (a previously-intentional duplicate, dropped once
        nothing needed it for label-set backward compatibility anymore),
        so a degeneracy-3 mode now reports exactly 3 distinct directions:
        native, (0,1,0), (0,0,1)."""
        distorted = self._distorted({2: 0.003})
        result = decompose_at_k_vector("GM", [self.gm4], self.parent, distorted, stoichiometry_multiplier=1)
        directions = {ma.direction for ma in result.amplitudes if ma.mode_index == 2}
        # OPD position letters now, one per degenerate direction. The old
        # set {"native", "(0,1,0)", "(0,0,1)"} named the same three
        # directions under `mode_direction_sweep`'s own scheme.
        self.assertEqual(directions, {"a", "b", "c"})

        by_direction = {
            ma.direction: ma.amplitude_angstrom for ma in result.amplitudes if ma.mode_index == 2
        }
        # The two orthogonal axes must come back ~0 for a pure
        # single-axis injected distortion; native itself carries the
        # whole real signal.
        # Shape, not names: exactly one direction carries the injected
        # single-axis distortion. Directions are OPD position letters now
        # and no letter is pinned to a Cartesian axis.
        ordered = sorted((abs(v) for v in by_direction.values()), reverse=True)
        self.assertNotAlmostEqual(ordered[0], 0.0, places=9)
        for other in ordered[1:]:
            self.assertAlmostEqual(other, 0.0, places=9)

    def test_mismatched_lattice_no_longer_an_error_reports_strain_instead(self):
        """v1 -> v2: an arbitrary lattice mismatch is no longer a hard
        error -- the whole point of v2 is handling a real relaxed CONTCAR,
        whose lattice routinely differs from any hand-picked target. A
        uniformly-scaled (isotropic strain) lattice, same atom count, is
        now accepted and its strain reported rather than rejected."""
        bogus = Structure(
            comment="bogus",
            lattice=self.undistorted.lattice * 2.0,
            species=self.undistorted.species,
            frac_coords=self.undistorted.frac_coords,
        )
        result = decompose_at_k_vector("GM", [self.gm4], self.parent, bogus, stoichiometry_multiplier=1)
        self.assertAlmostEqual(result.strain.determinant, 8.0, places=6)
        np.testing.assert_array_almost_equal(np.diag(result.strain.strain), [1.0, 1.0, 1.0])

    def test_wrong_atom_count_raises(self):
        truncated = Structure(
            comment="truncated",
            lattice=self.undistorted.lattice,
            species=self.undistorted.species[:-1],
            frac_coords=self.undistorted.frac_coords[:-1],
        )
        with self.assertRaises(DecompositionError):
            decompose_at_k_vector("GM", [self.gm4], self.parent, truncated, stoichiometry_multiplier=1)

    def test_far_displaced_atom_beyond_tolerance_raises(self):
        distorted = self._distorted({2: 0.003})
        corrupted_frac = distorted.frac_coords.copy()
        corrupted_frac[2] = corrupted_frac[2] + 0.4
        corrupted = Structure(
            comment="corrupted", lattice=distorted.lattice, species=distorted.species, frac_coords=corrupted_frac
        )
        with self.assertRaises(DecompositionError):
            decompose_at_k_vector("GM", [self.gm4], self.parent, corrupted, stoichiometry_multiplier=1)

    def test_empty_irrep_list_raises(self):
        with self.assertRaises(DecompositionError):
            decompose_at_k_vector("GM", [], self.parent, self.undistorted, stoichiometry_multiplier=1)








class TestRealBaTiO3FerroelectricDecomposition(unittest.TestCase):
    """Real, non-synthetic decomposition validation (the STO tests above
    inject a KNOWN synthetic amplitude and recover it; this one decomposes
    a genuinely different, independently-obtained real structure).

    No cubic (Pm-3m) BaTiO3 POSCAR existed in this project's fixtures --
    only two already-distorted real polymorphs (`POSCAR_BaTiO3`, R3m, and
    `POSCAR_BaTiO3_ferroelectric`, P4mm), which are NOT a parent/child pair
    (R3m is not a supergroup of P4mm; they're two different real subgroups
    of the same cubic parent). `POSCAR_BaTiO3_cubic` was derived directly
    from `POSCAR_BaTiO3_ferroelectric`'s own real lattice/Wyckoff data
    (confirmed with the physicist): cubic a = the real fixture's own a = b
    (already exactly equal, the in-plane lattice parameter the P4mm
    distortion does NOT affect -- c is left out of the average since c
    itself carries part of the physical distortion), Ti at the ideal
    origin (0,0,0) and Ba at the ideal body center (0.5,0.5,0.5) (already
    exactly the real fixture's own x,y -- only z, which carries the
    ferroelectric shift, is idealized to 0/0.5), and the three O atoms at
    the ideal face-center sites (0,0.5,0)/(0.5,0,0)/(0,0,1)/(0,0,0.5)
    surrounding Ti's own octahedron (again matching the real fixture's own
    x,y exactly; only z is idealized). `BaTiO3_cubic_findsym`/
    `BaTiO3_cubic_findsym_GM` are real findsym 7.1.6 / smodes 1.2.5 output
    for this derived structure (confirmed space group 221, Pm-3m; GM4-
    T1u degeneracy 3, GM5- T2u degeneracy 3 -- the same IR-active/IR-silent
    pair STO's own real GM4-/GM5- already validate).

    The real ferroelectric distortion is large (~0.3-0.4 A per atom, a
    genuine phase transition, not a small frozen-phonon-scale one), so
    needs `tol` widened well past the 0.3 A default (same guidance
    `--tolerance`'s own CLI help text already gives) -- 0.5 A here.

    Confirmed against real, independent BaTiO3 ferroelectricity physics:
    GM4- (the polar T1u soft mode) carries its ENTIRE amplitude along the
    swept (0,0,1) direction for every one of its 4 modes (Ba, Ti, and both
    O modes), with native/(1,0,0)/(0,1,0) all ~0 -- exactly the textbook
    c-axis (z-polarized) ferroelectric distortion of tetragonal P4mm
    BaTiO3, recovered here with NO prior knowledge of which axis baked
    into the test (the swept-direction machinery finds it). GM5- (IR-
    silent, T2u) comes back exactly 0 on every mode and every direction --
    correctly contributing nothing to a real polar distortion. Cross-
    checked independently by decomposition.py's real `iso` module elsewhere
    (`smodes-phonons decomposition`'s own iso cross-check reports
    `GM4- direction (a,0,0) (P4mm)` for parent 221 -> subgroup 99, the same
    physical axis under iso's own labeling convention -- not reproduced as
    an automated test here since it needs a live `iso` binary, unlike
    everything else in this class)."""

    def setUp(self):
        self.parent = Structure.from_poscar(FIXTURES / "POSCAR_BaTiO3_cubic")
        self.sub = Structure.from_poscar(FIXTURES / "POSCAR_BaTiO3_ferroelectric")
        out = parse_smodes_output(load("BaTiO3_cubic_findsym_GM"))
        self.gm4 = out.irrep("GM4-")
        self.gm5 = out.irrep("GM5-")

    def test_polar_soft_mode_recovers_the_real_z_polarized_distortion(self):
        result = decompose_at_k_vector(
            "GM", [self.gm4, self.gm5], self.parent, self.sub, stoichiometry_multiplier=1, tol=0.5
        )
        by_mode_direction = {
            (ma.label, ma.mode_index, ma.direction): ma.amplitude_angstrom for ma in result.amplitudes
        }

        # GM4- (T1u, IR-active): every mode's real amplitude lives almost
        # entirely along ONE direction, with the others ~0 -- the textbook
        # c-axis (z-polarized) P4mm distortion.
        #
        # ASSERTED AS A SHAPE, not against a named direction. Directions are
        # now OPD position letters, and no letter is pinned to a Cartesian
        # axis -- position order is arm-major/native-first, which is a
        # property of the irrep, not of the lab frame. "Exactly one
        # direction carries the amplitude" is the physics this test exists
        # for and it holds in any labelling; requiring it to land on the
        # string "(0,0,1)" was only ever true of one naming scheme.
        for mode_index, min_expected in [(0, 0.25), (1, 0.3), (2, 0.1), (3, 0.25)]:
            values = sorted(
                (abs(v) for (lab, m, _d), v in by_mode_direction.items()
                 if lab == "GM4-" and m == mode_index),
                reverse=True,
            )
            self.assertGreater(values[0], min_expected)
            for other in values[1:]:
                self.assertAlmostEqual(other, 0.0, delta=0.001)

        # GM5- (T2u, IR-silent): contributes nothing to a real polar
        # distortion, on every swept direction. GM5-'s own native
        # direction coincides with z (confirmed real BaTiO3 data: native
        # direction (-0,-0,-1)), so its own dropped duplicate is
        # "(0,0,1)", not "(1,0,0)" -- the remaining 3 entries are native,
        # (1,0,0), (0,1,0).
        # Every direction of the IR-silent irrep, whatever they are named.
        for (lab, mode, _d), value in by_mode_direction.items():
            if lab == "GM5-" and mode == 0:
                self.assertAlmostEqual(value, 0.0, delta=0.001)

        self.assertEqual(result.arm_index, 1)
        self.assertEqual(result.arm_count, 1)


class TestRealBaTiO3RhombohedralDecomposition(unittest.TestCase):
    """The other real BaTiO3 polymorph in this project's fixtures,
    `POSCAR_BaTiO3` (space group 160, R3m) -- famously the SAME T1u GM4-
    soft mode as the P4mm case above, just condensed along a different
    order-parameter direction: (a,0,0) gives tetragonal P4mm (the class
    above), (a,a,0) gives orthorhombic Amm2, (a,a,a) gives rhombohedral
    R3m -- BaTiO3's own textbook three real ferroelectric phases, all one
    irrep. `smodes-phonons decomposition`'s own real `iso` cross-check
    (needs a live `iso` binary, not reproduced as an automated test)
    independently confirms this: `GM4- direction (a,a,a) (R3m)` for
    parent 221 -> subgroup 160.

    `POSCAR_BaTiO3` itself is the conventional HEXAGONAL SETTING (15
    atoms, 3 formula units, R-centered) -- confirmed via findsym's own
    primitive-cell-ratio note that the true primitive cell is 3x smaller.
    `resolve_target`'s single-arm "further-scaled" branch now handles this
    too, via `_search_target_matrix` (the same class of off-diagonal
    relationship `nice_supercell_matrix` was built to handle for HfO2
    elsewhere in this codebase, generalized into `resolve_target` directly
    -- see `test_hexagonal_setting_cell_now_resolves_via_non_diagonal_
    target_matrix` below for the fix and its own, still-real, follow-on
    limitation). `POSCAR_BaTiO3_primitive` (this test's own sub structure,
    used by every OTHER test in this class) is the TRUE 5-atom rhombohedral
    primitive cell, derived directly from
    `POSCAR_BaTiO3`'s own real Cartesian atom positions via the standard
    obverse hexagonal-to-rhombohedral transformation
    (`a_rh = (2a1+a2+a3)/3`, etc.) -- confirmed correct two ways: the
    transformed lattice's own volume is exactly 1/3 of the hexagonal
    cell's, and running real findsym independently on the derived 5-atom
    cell alone reports the identical space group 160 (R3m).

    **Also exercises `internal_displacement`'s automatic origin-shift
    search for real, not synthetically** -- this fixture is deliberately
    committed in its raw, UNSHIFTED form. The natural R-centering copy
    nearest the hexagonal cell's own origin puts Ba, not Ti, near (0,0,0),
    a genuine origin-convention mismatch against this project's own cubic
    parent's convention (Ti at (0,0,0), Ba at (0.5,0.5,0.5)) -- exactly
    the real, hit-in-practice scenario `_search_origin_shift` exists to
    solve automatically (see its own docstring): a `--sub` sourced
    independently of `--parent` (here, derived from a real different
    polymorph's own data) with a different, equally-valid Wyckoff-setting
    origin choice. Direct (unshifted) alignment fails outright (closest
    candidate 3.45 A away); `decompose_at_k_vector` recovers automatically
    and reports the found shift via `KVectorResult.origin_shift_frac`.
    """

    def setUp(self):
        self.parent = Structure.from_poscar(FIXTURES / "POSCAR_BaTiO3_cubic")
        self.sub = Structure.from_poscar(FIXTURES / "POSCAR_BaTiO3_primitive")
        out = parse_smodes_output(load("BaTiO3_cubic_findsym_GM"))
        self.gm4 = out.irrep("GM4-")
        self.gm5 = out.irrep("GM5-")

    def test_polar_soft_mode_recovers_the_real_body_diagonal_distortion(self):
        result = decompose_at_k_vector(
            "GM", [self.gm4, self.gm5], self.parent, self.sub, stoichiometry_multiplier=1, tol=0.5
        )

        # This fixture's real origin genuinely doesn't match the parent's
        # own convention -- confirms the automatic search fired and found
        # a real, non-trivial shift (not that direct alignment happened to
        # already work).
        self.assertIsNotNone(result.origin_shift_frac)
        np.testing.assert_allclose(np.abs(result.origin_shift_frac) % 1.0, [0.5, 0.5, 0.5], atol=0.01)

        by_mode_direction = {
            (ma.label, ma.mode_index, ma.direction): ma.amplitude_angstrom for ma in result.amplitudes
        }

        # GM4- (T1u, IR-active): a real (a,a,a) distortion -- every swept
        # direction (native and its 2 within-arm partners) comes back with
        # the SAME amplitude per mode, unlike the P4mm case's single
        # dominant axis. This is exactly what the direction sweep exists
        # to show (see mode_direction_sweep's own docstring). GM4-'s own
        # native direction coincides with x, so -- since `mode_direction_
        # sweep` no longer also reports that same direction a second time
        # under its own canonical "(1,0,0)" label -- only "native",
        # "(0,1,0)", and "(0,0,1)" remain (3 entries, matching the true
        # degeneracy, not 4).
        for mode_index, min_abs in [(1, 0.03), (2, 0.05), (3, 0.03)]:
            amps = [v for (lab, m, _d), v in by_mode_direction.items()
                    if lab == "GM4-" and m == mode_index]
            for amp in amps:
                self.assertGreater(abs(amp), min_abs)
            # MAGNITUDES, not signed values. An individual mode's amplitude
            # SIGN for a given direction is an artifact of which symmetry
            # operation the search happened to find -- which is exactly why
            # per-direction totals are summed in quadrature and never raw
            # (the developer notes, "decomposition's per-direction totals"). Measured
            # here: mode 3 comes back a=-0.062356 b=+0.062356 c=-0.062356,
            # equal in magnitude with one sign flipped, so a signed max-min
            # reports 0.125 for a distortion that is textbook (a,a,a). The
            # older direction sweep happened to yield one consistent sign;
            # the OPD basis does not, and both are the same physics.
            magnitudes = [abs(v) for v in amps]
            self.assertAlmostEqual(max(magnitudes) - min(magnitudes), 0.0, delta=1e-6)

        # GM5- (T2u, IR-silent): contributes nothing to a real polar
        # distortion, on every swept direction. GM5-'s own native
        # direction coincides with z (confirmed real BaTiO3 data: native
        # direction (-0,-0,-1)), so its own dropped duplicate is
        # "(0,0,1)", not "(1,0,0)" -- the remaining 3 entries are native,
        # (1,0,0), (0,1,0).
        # Every direction of the IR-silent irrep, whatever they are named.
        for (lab, mode, _d), value in by_mode_direction.items():
            if lab == "GM5-" and mode == 0:
                self.assertAlmostEqual(value, 0.0, delta=0.001)

        self.assertEqual(result.arm_index, 1)
        self.assertEqual(result.arm_count, 1)

    def test_hexagonal_setting_cell_now_resolves_via_non_diagonal_target_matrix(self):
        """The 15-atom conventional hexagonal-setting cell (3x the cubic
        parent) is NOT a diagonal supercell of a cubic lattice --
        `_search_target_multiplier` alone (diagonal-only) cannot represent
        it and `resolve_target` used to raise `DecompositionError` outright
        for this real, originally-supplied fixture (not a synthetic
        construction). `_search_target_matrix` now finds the genuinely
        non-diagonal target matrix via the same Gram-tensor search
        `nice_supercell_matrix` already uses for HfO2, disambiguating a
        REAL, exact tie between two Gram-equivalent candidates (confirmed:
        both give identical full-matrix Gram error, since they're related
        by a rotation that isn't itself a symmetry of the cubic lattice) by
        actually tiling and testing atom alignment against `sub` -- only
        one of the two tied candidates aligns at all.

        Confirms the ORIGINAL blocking error is gone, and that the
        resolved structure/strain/origin-shift are physically correct --
        matching the values independently established elsewhere (the
        `decomposition` auto-reduction CLI test on the same real R3m
        physics, and this class's own already-validated
        `POSCAR_BaTiO3_primitive` strain).

        **Does NOT yet assert a near-zero residual** -- that's a real,
        separate, still-open limitation this fix does not close: the
        within-arm direction sweep (`enumerate_within_arm_directions`)
        only finds 2 of GM4-'s 3 needed partner directions on this bigger,
        non-diagonally-admixed 15-atom resolved structure (confirmed
        directly, independent of tolerance -- not a noise issue), where it
        finds all 3 cleanly on the smaller 5-atom primitive cell this
        class's other tests use. So only the "native" direction is
        reported here, leaving real (a,a,a) amplitude unexplained in the
        residual -- a known gap in the symmetry-operation search on a
        non-canonically-oriented cell, not a wrong answer from this fix.
        """
        hexagonal = Structure.from_poscar(FIXTURES / "POSCAR_BaTiO3")
        result = decompose_at_k_vector(
            "GM", [self.gm4, self.gm5], self.parent, hexagonal, stoichiometry_multiplier=3, tol=0.3
        )

        self.assertEqual(result.arm_count, 1)
        # Same real strain this class's own primary test's structure
        # carries relative to the idealized cubic parent (a small, genuine
        # rhombohedral distortion) -- not a coincidence, both describe the
        # identical real crystal.
        self.assertAlmostEqual(result.strain.strain[0, 0], 0.009439, delta=1e-4)
        self.assertAlmostEqual(result.strain.strain[2, 2], 0.013766, delta=1e-4)

        # The real GM4- Ti-mode amplitude is present (confirms the
        # resolved target and internal displacement are physically
        # meaningful, not zeroed out by a bad alignment), even though the
        # within-arm sweep gap above means only "native" is reported.
        by_mode_direction = {
            (ma.label, ma.mode_index, ma.direction): ma.amplitude_angstrom for ma in result.amplitudes
        }
        # v2.0 CLOSES the gap described above: all three GM4- components
        # are built from ISOTROPY's matrices on this 15-atom cell too, so
        # the residual vanishes and the (a,a,a) signature is exact. The
        # SIGN of a component is a basis convention (SMODES's printed
        # pattern vs ISOTROPY's), so magnitudes are compared.
        self.assertLess(result.residual_norm_angstrom, 1e-6)
        ti = [v for (lab, m, d), v in by_mode_direction.items() if lab == "GM4-" and m == 1]
        self.assertEqual(len(ti), 3)
        self.assertAlmostEqual(max(ti) - min(ti), 0.0, delta=1e-6)
        self.assertGreater(abs(ti[0]), 0.05)




if __name__ == "__main__":
    unittest.main()


class TestStrainedSubEnlargement(unittest.TestCase):
    """A DFT-relaxed `--sub` is strained relative to `--parent` by real
    physics, not just repeated -- and `enlarge_onto_lattice` used to match
    `sub`'s own ABSOLUTE lattice against a target built from the unstrained
    parent, so a real relaxation of a couple of percent put the Gram match
    outside tolerance and blocked every non-Gamma k-point at once.

    Found from real physicist usage on BaTiO3 (a GM4- frozen structure
    relaxed to P4mm: parent a=b=c=3.98610, sub a=c=3.97448, b=4.07213 --
    a 2% tetragonal strain), where `--k_vector X` and `--k_vector M` both
    hard-failed with "sub is too small to be enlarged onto this irrep's own
    native cell either" while `--k_vector GM` succeeded. The earlier
    enlargement fix was validated only against `freeze`-generated
    structures, which never change the cell, so this case was never
    exercised.
    """

    def setUp(self):
        self.parent = Structure.from_poscar(FIXTURES / "POSCAR_BaTiO3_cubic")
        gm = parse_smodes_output((FIXTURES / "BaTiO3_cubic_findsym_GM").read_text()).irrep("GM4-")
        pattern = np.array(gm.modes[1].displacements)
        strain = np.diag([1 - 0.003984, 1 + 0.020488, 1 - 0.003984])
        lattice = self.parent.lattice @ strain
        disp = pattern / np.linalg.norm(pattern) * 0.146072
        self.strained_sub = Structure(
            comment="BaTiO3 GM4- relaxed",
            lattice=lattice,
            species=self.parent.species,
            frac_coords=self.parent.frac_coords + disp @ np.linalg.inv(lattice),
        )

    def test_strained_sub_enlarges_onto_a_bigger_irrep_cell(self):
        # The strained sub is 1x the parent; X1+'s own native cell needs 2x.
        x1 = parse_smodes_output((FIXTURES / "BaTiO3_cubic_smodes_X").read_text()).irrep("X1+")

        # Direct matching genuinely cannot work here -- the shapes differ by
        # a real 2% strain, which is exactly what makes the fallback needed.
        with self.assertRaises(SupercellError):
            enlarge_onto_lattice(self.strained_sub, x1.superlattice)

        enlarged = enlarge_onto_lattice(
            self.strained_sub, x1.superlattice, reference_lattice=self.parent.lattice
        )
        self.assertEqual(enlarged.n_atoms, 2 * self.strained_sub.n_atoms)

        # The sub's own real strain must survive -- it is reported separately
        # by compute_strain, and internal_displacement is invariant to it.
        lengths = [float(np.linalg.norm(enlarged.lattice[i])) for i in range(3)]
        self.assertAlmostEqual(lengths[0], 3.97448, places=4)
        self.assertAlmostEqual(lengths[1], 2 * 4.07213, places=4)
        self.assertAlmostEqual(lengths[2], 3.97448, places=4)

    def test_unstrained_sub_takes_the_direct_path_unchanged(self):
        # Regression guard: the fallback must not alter any already-working
        # case. With no strain, direct matching succeeds on its own and the
        # reference_lattice argument changes nothing.
        x1 = parse_smodes_output((FIXTURES / "BaTiO3_cubic_smodes_X").read_text()).irrep("X1+")
        unstrained = Structure(
            comment="unstrained",
            lattice=self.parent.lattice,
            species=self.parent.species,
            frac_coords=self.parent.frac_coords,
        )
        direct = enlarge_onto_lattice(unstrained, x1.superlattice)
        with_ref = enlarge_onto_lattice(unstrained, x1.superlattice, reference_lattice=self.parent.lattice)
        self.assertEqual(direct.n_atoms, with_ref.n_atoms)
        np.testing.assert_allclose(direct.lattice, with_ref.lattice, atol=1e-9)

    def test_strained_graphite_recovers_its_real_signal_and_zeros_elsewhere(self):
        # Same strain gap, reached through a genuinely different code path
        # (the multi-arm blend branch, `_try_blend_components`) and on a
        # hexagonal lattice rather than cubic -- confirming the fix is not
        # specific to one branch or one crystal family. A K1-frozen
        # structure, strained, decomposed against its OWN k-point must
        # still recover the real amplitude exactly: the fix has to unblock
        # resolution without corrupting genuine signal.
        parent = Structure.from_poscar(FIXTURES / "POSCAR_graphite")
        log = (FIXTURES / "findsym_7.1.6_graphite.log").read_text()
        out = parse_smodes_output((FIXTURES / "graphite_findsym_K").read_text())
        k1 = out.irrep("K1")

        from smodes_phonons import isotropy_basis as ib
        # Built from the PARENT, as a real strained CONTCAR would be: the
        # old route tiles SMODES's own coordinates, 3.4e-5 A from this
        # POSCAR, which then shows up as residual against the parent.
        joint = ib.joint_basis([(k1, ib.seeds_of(k1))], parent, ib.space_group_from_log(log), log)
        ref, pattern = joint.reference, joint.bases["K1"][0][0]
        strain = np.diag([1.015, 1.015, 0.99])
        lattice = ref.lattice @ strain
        amplitude = 0.05
        disp = pattern / np.linalg.norm(pattern) * amplitude
        strained = Structure(
            comment="graphite K1 strained",
            lattice=lattice,
            species=ref.species,
            frac_coords=ref.frac_coords + disp @ np.linalg.inv(lattice),
        )

        result = decompose_at_k_vector(
            "K", list(out.irreps), parent, strained,
            stoichiometry_multiplier=strained.n_atoms // parent.n_atoms,
            findsym_log_text=log,
        )
        recovered = max(abs(ma.amplitude_angstrom) for ma in result.amplitudes if ma.label == "K1")
        # Recovered 0.049261 A against an injected 0.05 A. The ~1.5%
        # shortfall is the strain itself, not an error: the displacement is
        # applied in fractional coordinates against the STRAINED lattice, so
        # its Cartesian magnitude is rescaled by exactly that much. The
        # assertion that matters is that the real signal survives at full
        # magnitude (rather than being lost, or the whole k-vector
        # erroring), and that nothing is left unexplained.
        self.assertAlmostEqual(recovered, amplitude, delta=0.02 * amplitude)
        self.assertLess(result.residual_norm_angstrom, 1e-6)

        # Every other K irrep must stay at the noise floor -- the strain
        # must not smear real signal across irreps that carry none.
        others = [
            abs(ma.amplitude_angstrom) for ma in result.amplitudes if ma.label != "K1"
        ]
        # 1e-4, not the old 1e-6: per-irrep resolution (see
        # `_group_irreps_by_native_cell`) gives up the exact numerical
        # orthogonality the old single-shared-frame tiling had, since each
        # irrep's pattern now carries its own tiling noise rather than being
        # realized on one common grid. Measured cross-talk here is 1.67e-05,
        # and it is a fixed floor, NOT strain-induced -- confirmed identical
        # (1.673e-05) at applied strains of 0, 0.5%, 1%, 2% and 4%. The
        # assertion that matters is unchanged: real signal must not be
        # smeared across irreps carrying none, and 1e-4 is still ~60x below
        # the injected amplitude.
        self.assertLess(max(others), 1e-4)

    def test_strained_one_times_sub_enlarges_onto_a_hexagonal_k_point_cell(self):
        # The complementary half: a 1x-parent structure, strained, must
        # still enlarge onto a bigger k-point cell. Before the fix this
        # raised SupercellError and the whole k-vector hard-failed.
        parent = Structure.from_poscar(FIXTURES / "POSCAR_graphite")
        log = (FIXTURES / "findsym_7.1.6_graphite.log").read_text()
        k1 = parse_smodes_output((FIXTURES / "graphite_findsym_K").read_text()).irrep("K1")
        from smodes_phonons import isotropy_basis
        ref, _patterns, _n = isotropy_basis.build_joint_opd_cell(
            [k1], parent, isotropy_basis.space_group_from_log(log), log,
        )
        small = Structure(
            comment="graphite 1x strained",
            lattice=parent.lattice @ np.diag([1.015, 1.015, 0.99]),
            species=parent.species,
            frac_coords=parent.frac_coords,
        )
        with self.assertRaises(SupercellError):
            enlarge_onto_lattice(small, ref.lattice)
        enlarged = enlarge_onto_lattice(small, ref.lattice, reference_lattice=parent.lattice)
        self.assertEqual(enlarged.n_atoms, ref.n_atoms)


class TestQuadratureRoundTrip(unittest.TestCase):
    """A native+quadrature state round-trips at a time-reversal-paired
    k-point.

    THE CONCERN THIS RETIRES. Where `2k` is not integral (graphite and
    CsV3Sb5's K and H), the star is a time-reversal pair and the two arms
    are exactly PARALLEL -- arm 2 carries nothing new. The genuinely
    distinct second order parameter is the Bloch QUADRATURE, a 90-degree
    phase shift rather than a rotation of anything in real space.
    the developer notes' known-issues list recorded that `decomposition`'s blend path fits in terms
    of `armN` components, which at these k-points cannot express a
    native+quadrature state, and that whether a real structure round-trips
    here had never been checked.

    It does. `enumerate_direction_basis` substitutes the quadrature FOR the
    redundant arm before decomposition ever sees it, so the components
    being fitted are already `native` and `quadrature` -- the `armN` path
    the concern was about is not the one taken.
    """

    CASES = [
        ("graphite", "POSCAR_graphite", "graphite_findsym_K",
         "findsym_7.1.6_graphite.log", "K", "K4", 3e-4),
        ("graphite", "POSCAR_graphite", "graphite_findsym_H",
         "findsym_7.1.6_graphite.log", "H", "H1", 1e-6),
        ("CsV3Sb5", "POSCAR_CsV3Sb5", "CsV3Sb5_findsym_K",
         "findsym_7.1.6_CsV3Sb5.log", "K", "K1", 1e-6),
    ]
    NATIVE, QUAD = 0.010, 0.006

    def test_native_quadrature_and_their_blend_all_recover(self):
        from smodes_phonons.opd import LETTERS, build_opd_bases
        for material, poscar, smodes_file, log_file, k, label, tol in self.CASES:
            with self.subTest(material=material, irrep=label):
                parent = Structure.from_poscar(FIXTURES / poscar)
                log = (FIXTURES / log_file).read_text()
                irreps = list(parse_smodes_output((FIXTURES / smodes_file).read_text()).irreps)
                irrep = next(i for i in irreps if i.label == label)
                basis = build_opd_bases([irrep], parent, findsym_log_text=log)[(label, 0)]
                mechanisms = [o.mechanism for o in basis.opds]
                self.assertIn("quadrature", mechanisms,
                              f"{material} {label} has no quadrature position")
                native_i = mechanisms.index("native")
                quad_i = mechanisms.index("quadrature")
                patterns = [np.asarray(o.pattern, dtype=np.float64) for o in basis.opds]

                # The two are genuinely independent, not a relabelling.
                unit = [p.ravel() / np.linalg.norm(p) for p in patterns]
                self.assertAlmostEqual(
                    abs(float(unit[native_i] @ unit[quad_i])), 0.0, places=6
                )

                def recovered(displacement):
                    """Largest amplitude per direction ACROSS MODES.

                    Keyed per direction alone this silently reads the LAST
                    mode instead of the populated one -- which produced an
                    all-zero result for CsV3Sb5 K1 (2 modes) while graphite
                    K4 (1 mode) looked fine, and briefly read as a real bug.
                    """
                    sub = basis.structure.displaced(displacement)
                    result = decompose_at_k_vector(
                        k, irreps, parent, sub, check_stoichiometry(parent, sub),
                        findsym_log_text=log,
                    )
                    best: dict = {}
                    for amplitude in result.amplitudes:
                        if amplitude.label != label:
                            continue
                        current = best.get(amplitude.direction, 0.0)
                        if abs(amplitude.amplitude_angstrom) > abs(current):
                            best[amplitude.direction] = amplitude.amplitude_angstrom
                    return best

                native_letter, quad_letter = LETTERS[native_i], LETTERS[quad_i]

                pure_native = recovered(self.NATIVE * patterns[native_i])
                self.assertAlmostEqual(abs(pure_native[native_letter]), self.NATIVE, delta=tol)
                self.assertAlmostEqual(abs(pure_native.get(quad_letter, 0.0)), 0.0, delta=tol)

                pure_quad = recovered(self.QUAD * patterns[quad_i] / self.QUAD * self.NATIVE)
                self.assertAlmostEqual(abs(pure_quad[quad_letter]), self.NATIVE, delta=tol)
                self.assertAlmostEqual(abs(pure_quad.get(native_letter, 0.0)), 0.0, delta=tol)

                blend = recovered(self.NATIVE * patterns[native_i]
                                  + self.QUAD * patterns[quad_i])
                self.assertAlmostEqual(abs(blend[native_letter]), self.NATIVE, delta=tol)
                self.assertAlmostEqual(abs(blend[quad_letter]), self.QUAD, delta=tol)


class TestRigidRotationInvariance(unittest.TestCase):
    """A sub in a rotated frame decomposes identically -- including when the
    rotation is a symmetry of the parent and the star arms become
    indistinguishable.

    WHY THIS MATTERS. A structure from an external tool (ISODISTORT, a
    published CIF, a collaborator) arrives in THAT tool's standard setting.
    Real ISODISTORT output for STO M2- writes the cell axis-aligned where
    SMODES writes it on the diagonals -- the same crystal, 45 degrees
    apart. A rigid rotation is physics-free, so a correct decomposition
    must be invariant to it, and these tests are self-validating: the
    injected amplitude is known, so no external reference file is needed.

    The symmetry-angle cases (90 degrees for cubic/tetragonal, 120 for
    hexagonal) used to RAISE, on the reasoning that indistinguishable arms
    meant the sub might not be built on this irrep at all. Measured, that
    was over-conservative: arms of a star are symmetry-equivalent by
    construction, so there is no unique answer AND it does not matter.
    """

    ROTATION_CASES = [
        ("STO", "POSCAR_STO", "STO_findsym_M", "findsym_7.1.6_STO.log", "M", "M2-", 45),
        ("STO", "POSCAR_STO", "STO_findsym_M", "findsym_7.1.6_STO.log", "M", "M2-", 90),
        ("TiO2", "POSCAR_TiO2", "TiO2_findsym_X", "findsym_7.1.6_TiO2.log", "X", "X1", 45),
        ("TiO2", "POSCAR_TiO2", "TiO2_findsym_X", "findsym_7.1.6_TiO2.log", "X", "X1", 90),
        ("graphite", "POSCAR_graphite", "graphite_findsym_K",
         "findsym_7.1.6_graphite.log", "K", "K4", 60),
        ("graphite", "POSCAR_graphite", "graphite_findsym_K",
         "findsym_7.1.6_graphite.log", "K", "K4", 120),
    ]
    AMPLITUDE = 0.01

    @staticmethod
    def _rotated(structure, degrees):
        """Rigidly rotate about z: rotate the LATTICE and keep fractional
        coordinates, so Cartesian positions turn with it and the crystal
        moves as one body. (Rotating the lattice while pinning CARTESIAN
        coordinates would be a shear, not a rotation -- the wrong test.)"""
        theta = np.radians(degrees)
        rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                             [np.sin(theta), np.cos(theta), 0.0],
                             [0.0, 0.0, 1.0]])
        return Structure(structure.comment,
                         np.asarray(structure.lattice, dtype=np.float64) @ rotation.T,
                         structure.species, structure.frac_coords)

    def test_every_opd_survives_a_rigid_rotation(self):
        from smodes_phonons.opd import LETTERS, build_opd_bases
        for material, poscar, smodes_file, log_file, k, label, degrees in self.ROTATION_CASES:
            with self.subTest(material=material, irrep=label, degrees=degrees):
                parent = Structure.from_poscar(FIXTURES / poscar)
                log = (FIXTURES / log_file).read_text()
                irreps = list(parse_smodes_output((FIXTURES / smodes_file).read_text()).irreps)
                irrep = next(i for i in irreps if i.label == label)
                basis = build_opd_bases([irrep], parent, findsym_log_text=log)[(label, 0)]
                for position in range(len(basis.opds)):
                    pattern = np.asarray(basis.opds[position].pattern, dtype=np.float64)
                    sub = basis.structure.displaced(self.AMPLITUDE * pattern)
                    result = decompose_at_k_vector(
                        k, irreps, parent, self._rotated(sub, degrees),
                        check_stoichiometry(parent, sub),
                        findsym_log_text=log,
                    )
                    own = [a for a in result.amplitudes if a.label == label]
                    top = max(own, key=lambda a: abs(a.amplitude_angstrom))
                    symmetry_angle = (material in ("STO", "TiO2") and degrees % 90 == 0) or \
                                     (material == "graphite" and degrees % 60 == 0)
                    if symmetry_angle:
                        # v2.0 reads a structure's Cartesian coordinates as given,
                        # in the parent's frame. Rotated by a SYMMETRY angle, the
                        # crystal is the symmetry-equivalent domain, so the
                        # position letter may change (STO M2-: arm 2 -> arm 3);
                        # what must hold is ONE position carrying the amplitude.
                        # (At a quadrature k-point a symmetry rotation can turn
                        # one position into a mix of it and its partner, so the
                        # invariant is the irrep's total, with nothing left over.)
                        # (graphite's 6_3 also swaps the two carbon sites, moving
                        # amplitude between modes: the total is over all of them.)
                        total = float(np.sqrt(sum(a.amplitude_angstrom ** 2 for a in own)))
                        self.assertAlmostEqual(total, self.AMPLITUDE, delta=3e-5)
                        self.assertLess(result.residual_norm_angstrom, 1e-6)
                    else:
                        self.assertEqual(
                            top.direction, LETTERS[position],
                            f"{material} {label} at {degrees} deg: froze position "
                            f"{LETTERS[position]}, recovered {top.direction}",
                        )
                    if not symmetry_angle:
                        # Same measured construction floor as everywhere else.
                        self.assertAlmostEqual(
                            abs(top.amplitude_angstrom), self.AMPLITUDE, delta=3e-5
                        )

    def test_a_distortion_with_no_content_at_this_k_point_is_exposed(self):
        """Taking arm 1 rather than raising must not turn wrong input into a
        plausible answer. The RESIDUAL is what catches it, and does so far
        more sharply than the lattice-orientation check it replaced."""
        from smodes_phonons.opd import build_opd_bases
        parent = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        log = (FIXTURES / "findsym_7.1.6_STO.log").read_text()
        irreps = list(parse_smodes_output((FIXTURES / "STO_findsym_M").read_text()).irreps)
        irrep = next(i for i in irreps if i.label == "M2-")
        basis = build_opd_bases([irrep], parent, findsym_log_text=log)[("M2-", 0)]
        cell = basis.structure

        # Gamma-periodic by construction: every atom of a species moves
        # identically, so there is no M-point content at all.
        uniform = np.zeros((cell.n_atoms, 3))
        for i, species in enumerate(cell.species):
            uniform[i] = ([1.0, 0.0, 0.0] if species == "Sr"
                          else [-1.0, 0.0, 0.0] if species == "Ti" else [0.0, 0.5, 0.0])
        uniform /= np.linalg.norm(uniform)

        sub = self._rotated(cell.displaced(self.AMPLITUDE * uniform), 90)
        result = decompose_at_k_vector(
            "M", irreps, parent, sub, check_stoichiometry(parent, sub),
            findsym_log_text=log,
        )
        fraction = result.residual_norm_angstrom / result.displacement_norm_angstrom
        self.assertGreater(fraction, 0.99)      # measured: 100.000%
        self.assertLess(
            max(abs(a.amplitude_angstrom) for a in result.amplitudes), 1e-6
        )


class TestJointOPDCellAmplitudeScale(unittest.TestCase):
    """The joint OPD cell enlarges `sub` onto its own grid; undoing exactly
    that repetition -- and nothing else -- is what these pin.

    the development log records a `sqrt(N)` amplitude correction that was
    implemented and REJECTED, because two cases are "genuinely
    indistinguishable from `sub` alone": a literal periodic repeat (where
    the correction is needed and correct) and a structure built fresh at
    full size (where it is wrong, and measurably so -- real STO GM4- in a
    combine_modes-built joint cell went from -0.001000 to -0.000500).

    The joint path is neither, because it performs the repetition itself
    and so never has to infer which case it is in. These tests exist to
    keep that property true rather than merely argued.
    """

    FIXTURES = FIXTURES

    def _joint(self, poscar, smodes_file, findsym_file):
        from smodes_phonons.decomposition import build_joint_opd_cell
        parent = Structure.from_poscar(self.FIXTURES / poscar)
        log = (self.FIXTURES / findsym_file).read_text()
        irreps = list(parse_smodes_output((self.FIXTURES / smodes_file).read_text()).irreps)
        reference, patterns, _n = build_joint_opd_cell(
            irreps, parent, tol=1e-2, findsym_log_text=log
        )
        return parent, irreps, log, None, reference, patterns

    def test_an_enlarged_sub_reports_its_own_scale_not_the_joint_cell_s(self):
        """Real STO M2- frozen on its own smaller cell, decomposed against
        the larger joint cell, must report the amplitude it was frozen at.

        Without the correction this is 0.001414 = 0.001 * sqrt(2) -- the
        joint cell holds twice as many atoms, so the repeated displacement's
        norm grows against a pattern renormalised over the bigger cell.
        """
        parent, irreps, log, runner, reference, _p = self._joint(
            "POSCAR_STO", "STO_findsym_M", "findsym_7.1.6_STO.log"
        )
        m2 = next(i for i in irreps if i.label == "M2-")
        multipliers = nice_supercell_matrix(m2.superlattice, parent.lattice)
        cell, sets = tile_onto_supercell(
            m2, parent.lattice, multipliers,
            {"p": [m2.modes[0].displacements]}, comment="frozen",
        )
        pattern = np.asarray(sets["p"][0], dtype=np.float64)
        pattern /= np.linalg.norm(pattern)
        sub = cell.displaced(0.001 * pattern)
        self.assertLess(sub.n_atoms, reference.n_atoms)  # an enlargement really happens

        result = decompose_at_k_vector(
            "M", irreps, parent, sub, check_stoichiometry(parent, sub),
            findsym_log_text=log,
        )
        recovered = max(
            abs(a.amplitude_angstrom) for a in result.amplitudes if a.label == "M2-"
        )
        self.assertAlmostEqual(recovered, 0.001, places=6)

    def test_a_fresh_built_sub_is_never_rescaled(self):
        """The REJECTED correction's own failure case, pinned.

        Real STO GM4- inside a `combine_modes`-built joint structure is
        built fresh AT its full size -- nothing is periodically repeated,
        so there is nothing to undo. Applying a blanket `sqrt(N)` here is
        exactly what turned -0.001000 into -0.000500. The joint path cannot
        do that, because `repeats` is 1 when no enlargement happened, but
        that is a property worth holding rather than reasoning about.
        """
        parent, irreps, log, runner, reference, patterns = self._joint(
            "POSCAR_STO", "STO_findsym_M", "findsym_7.1.6_STO.log"
        )
        key = sorted(k for k in patterns if k[0] == "M2-")[0]
        # Built directly AT the joint cell's own size -- no repetition.
        sub = reference.displaced(0.001 * np.asarray(patterns[key], dtype=np.float64))
        self.assertEqual(sub.n_atoms, reference.n_atoms)

        result = decompose_at_k_vector(
            "M", irreps, parent, sub, check_stoichiometry(parent, sub),
            findsym_log_text=log,
        )
        recovered = max(
            abs(a.amplitude_angstrom) for a in result.amplitudes if a.label == "M2-"
        )
        self.assertAlmostEqual(recovered, 0.001, places=9)

    def test_an_oversized_padded_sub_now_RESOLVES_with_a_correct_Ap(self):
        """A sub padded beyond the joint cell used to be REFUSED; it now
        resolves exactly, and `Ap` is unchanged by the padding.

        This test previously asserted the refusal, on the reasoning that a
        clear error beats a silently mis-scaled amplitude. Measured, that
        premise does not hold -- nothing is mis-scaled:

            un-padded  40 atoms  (8x)  resid 0.000%  As 0.001000  Ap 0.000354
            padded     80 atoms (16x)  resid 0.000%  As 0.001414  Ap 0.000354

        `As` is BY DEFINITION a norm over the cell it was measured on, so a
        2x repetition inflating it by exactly sqrt(2) is that definition
        being obeyed, not a defect -- and `Ap`, the cell-size-independent
        column and the one to quote or compare, is IDENTICAL across the two.
        The user's own padding stays theirs in `As` and normalises out of
        `Ap`, which is precisely the split the two columns exist for.

        What changed: `resolve_target`'s multi-arm branch used to raise
        whenever `sub` was a further MULTIPLE of the irrep's own base volume,
        because it only ever tried ENLARGING `sub` onto the native cell and
        never TILING the reference up to meet it. The further-tiling fallback
        that does exactly that already sat below the raise, unreached.
        """
        parent, irreps, log, runner, reference, patterns = self._joint(
            "POSCAR_STO", "STO_findsym_M", "findsym_7.1.6_STO.log"
        )
        key = sorted(k for k in patterns if k[0] == "M2-")[0]
        at_joint_size = reference.displaced(
            0.001 * np.asarray(patterns[key], dtype=np.float64)
        )
        padded_lattice = np.asarray(reference.lattice, dtype=np.float64).copy()
        padded_lattice[0] *= 2.0
        padded = enlarge_onto_lattice(
            at_joint_size, padded_lattice, reference_lattice=parent.lattice,
        )
        self.assertEqual(padded.n_atoms, 2 * reference.n_atoms)

        def _resolve(structure):
            result = decompose_at_k_vector(
                "M", irreps, parent, structure,
                check_stoichiometry(parent, structure),
                findsym_log_text=log,
            )
            best = max(
                (a for a in result.amplitudes if a.label == "M2-"),
                key=lambda a: abs(a.amplitude_angstrom),
            )
            return result, best

        plain_result, plain_best = _resolve(at_joint_size)
        padded_result, padded_best = _resolve(padded)

        # Both fit exactly -- the padding costs no accuracy.
        for result in (plain_result, padded_result):
            self.assertLess(
                result.residual_norm_angstrom / result.displacement_norm_angstrom, 1e-6
            )
        # `As` follows the cell it was measured on: sqrt(2) for a 2x repeat.
        self.assertAlmostEqual(plain_best.amplitude_angstrom, 0.001, places=9)
        self.assertAlmostEqual(
            abs(padded_best.amplitude_angstrom), 0.001 * np.sqrt(2.0), places=9
        )
        # `Ap` does not -- it is the same number either way, which is the
        # whole point of reporting it.
        plain_ap = plain_best.amplitude_angstrom / plain_result.parent_cell_divisor
        padded_ap = abs(padded_best.amplitude_angstrom) / padded_result.parent_cell_divisor
        self.assertAlmostEqual(abs(plain_ap), padded_ap, places=9)


class TestFrozenIrrepLeavesOtherIrrepsNearZero(unittest.TestCase):
    """Freeze ONE irrep; every OTHER irrep at that k-point must come back
    near zero -- not merely the frozen one's own amplitude being right.

    This gap is why a real bug went unnoticed for a long time. The
    long-standing `--k_vector GM M R` check on STO asserted that M2- came
    back at its injected 0.001000 A, and never asserted that M5- was zero --
    so it passed cleanly while M5- was simultaneously reporting the SAME
    full amplitude as a phantom, because `resolve_target` was laying every
    irrep's displacements into `irreps[0]`'s own atom ordering (see
    `_group_irreps_by_native_cell`). Checking only the injected irrep cannot
    see that failure mode at all: the injected irrep's own number stays
    correct throughout.

    Deliberately swept across several materials and k-points rather than one
    case, since the three confirmed real variants of this bug each surface at
    a different k-point -- cell ORIENTATION (STO's M, smodes 1.2.4), atom
    DECORATION (graphite's K, 1.2.5), and permuted lattice-vector ORDER
    (STO's Gamma).
    """

    # Per-irrep resolution's own measured cross-talk floor is 2.00% of the
    # injected amplitude at worst (real TiO2 X); see this file's own
    # `test_strained_graphite...` and the developer notes for the full table. 2.5%
    # matches `cli._NEGLIGIBLE_AMPLITUDE_RELATIVE_FLOOR`, the threshold
    # below which the report itself declines to assign a direction letter.
    RELATIVE_FLOOR = 0.025

    CASES = [
        ("STO Gamma", "POSCAR_STO", "STO_findsym_Gamma", "GM", ["GM4-", "GM5-"]),
        ("STO M", "POSCAR_STO", "STO_findsym_M", "M", ["M1+", "M3+", "M2-", "M5-"]),
        ("STO R", "POSCAR_STO", "STO_findsym_R", "R", ["R5+", "R4-"]),
        ("CMO X", "POSCAR_CMO", "CMO_findsym_X", "X", ["X2+", "X3-"]),
        ("TiO2 X", "POSCAR_TiO2", "TiO2_findsym_X", "X", ["X1", "X2"]),
        ("graphite K", "POSCAR_graphite", "graphite_findsym_K", "K", ["K1", "K4", "K5", "K6"]),
    ]

    def test_freezing_one_irrep_leaves_every_other_near_zero(self):
        amplitude = 0.001
        for name, poscar, smodes_file, k_vector, labels in self.CASES:
            parent = Structure.from_poscar(FIXTURES / poscar)
            out = parse_smodes_output(load(smodes_file))
            by_label = {irrep.label: irrep for irrep in out.irreps}
            every = [irrep.label for irrep in out.irreps]

            for label in labels:
                with self.subTest(material=name, frozen=label):
                    irrep = by_label[label]
                    sub = _inject_single_irrep(parent, irrep, amplitude)

                    result = decompose_at_k_vector(
                        k_vector, [by_label[other] for other in every], parent, sub,
                        stoichiometry_multiplier=check_stoichiometry(parent, sub),
                    )

                    # The frozen irrep itself must be recovered at full
                    # magnitude. Sign is not asserted: an individual mode's
                    # amplitude sign is an artifact of which symmetry
                    # operation the search happened to find (documented in
                    # the developer notes' per-direction-totals section).
                    frozen = [
                        abs(ma.amplitude_angstrom)
                        for ma in result.amplitudes if ma.label == label
                    ]
                    # `delta`, not `places=6`, and the size is measured
                    # rather than chosen. Building the joint OPD cell and
                    # `tile_onto_supercell` are two independent routes to
                    # the same cell, and they agree only to ~1e-5 A. That
                    # shows up as a FIXED ABSOLUTE offset, entirely
                    # independent of the injected amplitude -- measured on
                    # real graphite K4 at 5.577e-06 A and real TiO2 X2 at
                    # 2.000e-05 A across injections of 0.001, 0.01, 0.05
                    # and 0.1 A, identical at every one.
                    #
                    # Being ABSOLUTE is what makes it acceptable. The
                    # 0.001 A here is this project's frozen-phonon
                    # amplitude, deliberately tiny -- a symmetry-breaking
                    # kick to start a relaxation, not a physical
                    # distortion -- so the floor is 2% of it. The
                    # amplitude that relaxation converges to is far
                    # larger, and at a realistic 0.1 A the same floor is
                    # 0.02%. Confirmed with the physicist as acceptable
                    # for that reason.
                    self.assertAlmostEqual(max(frozen), amplitude, delta=3e-5)

                    # ...and every OTHER irrep must stay at the noise floor.
                    # This is the assertion the old tests were missing.
                    others = [
                        abs(ma.amplitude_angstrom)
                        for ma in result.amplitudes if ma.label != label
                    ]
                    if others:
                        self.assertLess(
                            max(others), self.RELATIVE_FLOOR * amplitude,
                            f"{name}: freezing {label} leaked "
                            f"{max(others):.3e} A into another irrep",
                        )

                    # Nothing may be left unexplained either: a single frozen
                    # irrep is fully described by the irreps at its own
                    # k-point, so the residual must vanish -- to the same
                    # measured ~1e-5 A construction floor as above, which
                    # shows up here as residual rather than as amplitude.
                    #
                    # WHAT THIS STILL CATCHES, which is the point of the
                    # test: the bug it was written for gave a FULL 0.001 A
                    # phantom amplitude on a neighbouring irrep (real STO
                    # M5-, from laying every irrep's displacements into
                    # irreps[0]'s atom ordering). That is 20x this
                    # threshold, so the test's own purpose survives the
                    # loosening intact -- it is no longer sensitive to a
                    # leak BELOW the floor the construction itself has.
                    self.assertLess(result.residual_norm_angstrom, 5e-5)




class TestFrameConventionNote(unittest.TestCase):
    """v2.0 frame convention (physicist, 2026-11-03): a distorted structure's
    Cartesian coordinates are read as given, in the parent's frame. When its
    axes are a symmetry-rotated copy of the parent's, the OPD is reported as
    the equivalent domain -- and a note says so, since position letters can
    then differ from those in the structure's own axes."""

    def _stderr_for(self, degrees):
        import contextlib
        import io
        from smodes_phonons.opd import build_opd_bases
        parent = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        log = load("findsym_7.1.6_STO.log")
        irreps = list(parse_smodes_output(load("STO_findsym_M")).irreps)
        irrep = next(i for i in irreps if i.label == "M2-")
        basis = build_opd_bases([irrep], parent, findsym_log_text=log)[("M2-", 0)]
        sub = TestRigidRotationInvariance._rotated(basis.structure.displaced(0.01 * np.asarray(basis.opds[1].pattern)),
                                                   degrees)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            decompose_at_k_vector("M", irreps, parent, sub, check_stoichiometry(parent, sub), findsym_log_text=log)
        return err.getvalue()

    def test_a_symmetry_rotated_frame_is_named(self):
        self.assertIn("symmetry-rotated copy of the parent's", self._stderr_for(90))

    def test_the_parent_frame_gets_no_note(self):
        self.assertNotIn("symmetry-rotated copy", self._stderr_for(0))

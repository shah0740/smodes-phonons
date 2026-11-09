"""`opd.py` -- the OPD objects, the shared report formatter, and
`build_opd_bases` on real fixtures (ISOTROPY-matrix basis, saved `iso`
output, so subprocess-free).
"""

import unittest
from pathlib import Path

import numpy as np

from smodes_phonons.opd import (
    LETTERS,
    OPD,
    OPDBasis,
    OPDError,
    format_opd_basis,
    opd_bases_for_eigenvectors,
)
from smodes_phonons.opd import (
    OPD_MAX_CROSS_IRREP_OVERLAP,
    build_opd_bases,
    cross_irrep_overlap,
)
from smodes_phonons.smodes import parse_smodes_output
from smodes_phonons.decomposition import check_stoichiometry, decompose_at_k_vector
from smodes_phonons.structure import Structure

FIXTURES = Path(__file__).resolve().parent / "fixtures"




def _structure(n_atoms: int = 2) -> Structure:
    return Structure(
        comment="synthetic",
        lattice=np.eye(3) * 4.0,
        species=["Sr"] * n_atoms,
        frac_coords=np.array([[i / n_atoms, 0.0, 0.0] for i in range(n_atoms)]),
    )


def _basis(patterns, *, degeneracy=None, n_arms=1, within_arm=None, arms=None,
           mechanisms=None, n_atoms=2) -> OPDBasis:
    patterns = [np.asarray(p, dtype=np.float64) for p in patterns]
    arms = arms or [1] * len(patterns)
    mechanisms = mechanisms or ["native"] + [f"arm{a}" for a in arms[1:]]
    seen: dict = {}
    opds = []
    for position, (pattern, arm, mechanism) in enumerate(zip(patterns, arms, mechanisms)):
        opds.append(OPD(position=position, arm=arm, within_index=seen.get(arm, 0),
                        mechanism=mechanism, pattern=pattern))
        seen[arm] = seen.get(arm, 0) + 1
    return OPDBasis(
        irrep_label="X3-", mode_index=0,
        degeneracy=degeneracy if degeneracy is not None else len(patterns),
        n_arms=n_arms,
        within_arm=within_arm if within_arm is not None else max(1, len(patterns) // n_arms),
        structure=_structure(n_atoms), opds=tuple(opds),
    )


class TestOPDBasisValidatesItsOwnCell(unittest.TestCase):
    """The load-bearing invariant: a pattern belongs to exactly ONE cell."""

    def test_a_pattern_shaped_for_a_different_cell_is_rejected(self):
        """Comparing atom i of one cell against atom i of another is the
        mistake this project made three times, once while building the very
        baseline meant to detect it. It must be structurally impossible, not
        something to remember."""
        with self.assertRaises(OPDError) as caught:
            _basis([np.zeros((2, 3)), np.zeros((5, 3))])
        message = str(caught.exception)
        self.assertIn("(5, 3)", message)
        self.assertIn("2 atoms", message)

    def test_rank_and_conditioning_are_measured_at_construction(self):
        rank_2 = _basis([[[1.0, 0, 0], [0, 0, 0]], [[0, 1.0, 0], [0, 0, 0]]])
        self.assertEqual(rank_2.rank, 2)
        self.assertTrue(rank_2.is_complete)
        self.assertAlmostEqual(rank_2.conditioning, 1.0, places=10)

    def test_duplicate_directions_are_reported_as_incomplete_not_as_two_entries(self):
        """Having `degeneracy` entries is NOT the same as spanning them --
        real STO M5- once had 6 entries spanning 3 dimensions."""
        duplicated = _basis([[[1.0, 0, 0], [0, 0, 0]], [[1.0, 0, 0], [0, 0, 0]]])
        self.assertEqual(duplicated.rank, 1)
        self.assertEqual(duplicated.degeneracy, 2)
        self.assertFalse(duplicated.is_complete)


class TestDirectionSpec(unittest.TestCase):
    def test_single_arm_is_flat(self):
        basis = _basis([[[1.0, 0, 0], [0, 0, 0]], [[0, 1.0, 0], [0, 0, 0]]], n_arms=1)
        self.assertEqual(basis.direction_spec(0), "a,0")
        self.assertEqual(basis.direction_spec(1), "0,a")

    def test_multi_arm_is_grouped_by_semicolon(self):
        """`iso`'s own notation: semicolons separate star arms. Verified
        against real iso 9.6.1 -- TiO2 X1 comes back `(a,0;0,0)`."""
        basis = _basis([[[1.0, 0, 0], [0, 0, 0]], [[0, 1.0, 0], [0, 0, 0]]],
                       n_arms=2, within_arm=1, arms=[1, 2])
        self.assertEqual(basis.direction_spec(0), "a;0")
        self.assertEqual(basis.direction_spec(1), "0;a")

    def test_out_of_range_position_is_a_clear_error(self):
        basis = _basis([[[1.0, 0, 0], [0, 0, 0]]])
        with self.assertRaises(OPDError):
            basis.direction_spec(3)


class TestFormatOPDBasis(unittest.TestCase):
    """The report is the ONLY place an OPD basis is described in words, so
    `irrep`/`decomposition`/freeze cannot describe the same irrep
    differently."""

    def _report(self, basis, **kwargs) -> str:
        return "\n".join(format_opd_basis(basis, **kwargs))

    def test_states_provenance_inline_for_every_position(self):
        """Settled with the physicist: how each OPD was found belongs next to
        that OPD, in the same file, not in a separate operations file."""
        basis = _basis([[[1.0, 0, 0], [0, 0, 0]], [[0, 1.0, 0], [0, 0, 0]]],
                       n_arms=2, within_arm=1, arms=[1, 2])
        text = self._report(basis)
        self.assertIn("native", text)
        self.assertIn("arm2", text)
        self.assertIn("ISOTROPY's first component", text)
        self.assertIn("centring translation", text)

    def test_explains_quadrature_as_a_phase_shift_not_a_rotation(self):
        """A Bloch quadrature partner is unreachable by any rotation, and a
        reader needs to know that rather than assume it is another arm."""
        basis = _basis([[[1.0, 0, 0], [0, 0, 0]], [[0, 1.0, 0], [0, 0, 0]]],
                       n_arms=2, within_arm=1, arms=[1, 2],
                       mechanisms=["native", "quadrature"])
        text = self._report(basis)
        self.assertIn("PHASE shift", text)
        self.assertIn("not reachable by any rotation", text)

    def test_shows_the_full_per_atom_pattern(self):
        """Settled with the physicist: full patterns, not a summary with the
        numbers hidden in the .npz."""
        basis = _basis([[[0.5, -0.25, 0.125], [0, 0, 0]]])
        text = self._report(basis)
        self.assertIn(" 0.500000", text)
        self.assertIn("-0.250000", text)
        self.assertIn(" 0.125000", text)

    def test_patterns_can_be_suppressed_for_a_compact_listing(self):
        basis = _basis([[[0.5, -0.25, 0.125], [0, 0, 0]]])
        self.assertNotIn("0.500000", self._report(basis, show_patterns=False))

    def test_an_all_zero_pattern_says_so_rather_than_printing_nothing(self):
        """Silence would read as 'no atoms listed here' rather than 'this
        pattern is numerically zero', which are very different facts."""
        basis = _basis([np.zeros((2, 3))])
        self.assertIn("numerically zero", self._report(basis))

    def test_an_incomplete_basis_is_flagged_loudly_with_the_consequence(self):
        """Rank deficiency must not be a number the reader has to notice --
        it means a real distortion cannot be represented or detected."""
        basis = _basis([[[1.0, 0, 0], [0, 0, 0]], [[1.0, 0, 0], [0, 0, 0]]])
        text = self._report(basis)
        self.assertIn("INCOMPLETE", text)
        self.assertIn("WARNING", text)
        self.assertIn("cannot be represented or detected", text)

    def test_a_complete_basis_carries_no_warning(self):
        basis = _basis([[[1.0, 0, 0], [0, 0, 0]], [[0, 1.0, 0], [0, 0, 0]]])
        self.assertNotIn("WARNING", self._report(basis))

    def test_says_a_letter_names_a_direction_not_a_mechanism(self):
        """The whole point of the label being mechanism-agnostic -- stated in
        the report so a reader does not infer that `--direction` depends on
        how the OPD happened to be found."""
        basis = _basis([[[1.0, 0, 0], [0, 0, 0]]])
        self.assertIn("A letter names a direction, never the mechanism", self._report(basis))

    def test_reports_the_degeneracy_split_and_the_cell(self):
        basis = _basis([[[1.0, 0, 0], [0, 0, 0]], [[0, 1.0, 0], [0, 0, 0]]],
                       n_arms=2, within_arm=1, arms=[1, 2])
        text = self._report(basis)
        self.assertIn("2 star arm(s) x 1 within-arm direction(s)", text)
        self.assertIn("2 atoms", text)

    def test_every_position_gets_its_own_direction_spec(self):
        basis = _basis([[[1.0, 0, 0], [0, 0, 0]], [[0, 1.0, 0], [0, 0, 0]]],
                       n_arms=2, within_arm=1, arms=[1, 2])
        text = self._report(basis)
        self.assertIn("--direction a;0", text)
        self.assertIn("--direction 0;a", text)

    def test_letters_follow_position_order(self):
        basis = _basis([[[1.0, 0, 0], [0, 0, 0]], [[0, 1.0, 0], [0, 0, 0]],
                        [[0, 0, 1.0], [0, 0, 0]]], n_arms=1)
        text = self._report(basis, show_patterns=False)
        for letter in LETTERS[:3]:
            self.assertIn(f"  {letter}  --direction", text)


if __name__ == "__main__":
    unittest.main()


class TestOPDBasesForEigenvectors(unittest.TestCase):
    """The `phonons --generate_opd` contraction: a CHANGE OF BASIS from
    SMODES basis modes to DFT eigenvectors, using linearity alone."""

    def _two_mode_bases(self, patterns_by_mode):
        """Two modes sharing ONE structure, as `build_opd_bases` returns."""
        structure = _structure(2)
        out = {}
        for mode, patterns in patterns_by_mode.items():
            opds = tuple(
                OPD(position=i, arm=1, within_index=i, mechanism="native" if i == 0 else "within",
                    pattern=np.asarray(p, dtype=np.float64))
                for i, p in enumerate(patterns)
            )
            out[mode] = OPDBasis(
                irrep_label="X3-", mode_index=mode, degeneracy=len(opds),
                n_arms=1, within_arm=len(opds), structure=structure, opds=opds,
            )
        return out

    def test_identity_coefficients_reproduce_the_input_directions(self):
        """With the identity, eigenvector i IS basis mode i -- so every OPD
        must come back parallel to the one it started from."""
        bases = self._two_mode_bases({
            0: [[[1.0, 0, 0], [0, 0, 0]], [[0, 1.0, 0], [0, 0, 0]]],
            1: [[[0, 0, 1.0], [0, 0, 0]], [[0, 0, 0], [1.0, 0, 0]]],
        })
        out = opd_bases_for_eigenvectors(bases, np.eye(2))

        for mode in (0, 1):
            for position in (0, 1):
                got = out[mode].opds[position].pattern
                want = bases[mode].opds[position].pattern
                want = want / np.linalg.norm(want)
                np.testing.assert_allclose(got, want, atol=1e-12)

    def test_a_mixture_is_the_same_mixture_of_each_position(self):
        """The load-bearing identity: R_p(sum_j c_j b_j) == sum_j c_j R_p(b_j).
        Checked at EVERY position independently, since that is what makes a
        per-position label meaningful after the change of basis."""
        bases = self._two_mode_bases({
            0: [[[1.0, 0, 0], [0, 0, 0]], [[0, 1.0, 0], [0, 0, 0]]],
            1: [[[0, 0, 1.0], [0, 0, 0]], [[0, 0, 0], [1.0, 0, 0]]],
        })
        coefficients = np.array([[0.6, -0.8], [0.8, 0.6]])  # orthogonal mixing
        out = opd_bases_for_eigenvectors(bases, coefficients)

        for eigen in (0, 1):
            for position in (0, 1):
                expected = sum(
                    coefficients[mode, eigen] * bases[mode].opds[position].pattern
                    for mode in (0, 1)
                )
                expected = expected / np.linalg.norm(expected)
                np.testing.assert_allclose(out[eigen].opds[position].pattern, expected, atol=1e-12)

    def test_position_labels_and_provenance_survive_the_change_of_basis(self):
        """A position names a DIRECTION, so it must mean the same thing
        before and after -- and its mechanism travels with it."""
        bases = self._two_mode_bases({
            0: [[[1.0, 0, 0], [0, 0, 0]], [[0, 1.0, 0], [0, 0, 0]]],
            1: [[[0, 0, 1.0], [0, 0, 0]], [[0, 0, 0], [1.0, 0, 0]]],
        })
        out = opd_bases_for_eigenvectors(bases, np.eye(2))
        self.assertEqual([o.position for o in out[0].opds], [0, 1])
        self.assertEqual([o.mechanism for o in out[0].opds], ["native", "within"])
        self.assertEqual(out[0].direction_spec(0), bases[0].direction_spec(0))

    def test_results_share_the_input_cell(self):
        bases = self._two_mode_bases({
            0: [[[1.0, 0, 0], [0, 0, 0]], [[0, 1.0, 0], [0, 0, 0]]],
            1: [[[0, 0, 1.0], [0, 0, 0]], [[0, 0, 0], [1.0, 0, 0]]],
        })
        out = opd_bases_for_eigenvectors(bases, np.eye(2))
        self.assertIs(out[0].structure, bases[0].structure)

    def test_mismatched_coefficient_shape_is_rejected(self):
        """The contraction sums over modes, so a wrong-sized matrix would
        silently omit part of every eigenvector."""
        bases = self._two_mode_bases({
            0: [[[1.0, 0, 0], [0, 0, 0]]],
            1: [[[0, 1.0, 0], [0, 0, 0]]],
        })
        with self.assertRaises(OPDError) as caught:
            opd_bases_for_eigenvectors(bases, np.eye(3))
        self.assertIn("(2, 2)", str(caught.exception))

    def test_modes_on_different_cells_are_rejected(self):
        """Summing over modes requires ONE shared cell -- the cross-cell trap
        this whole module exists to make impossible."""
        bases = self._two_mode_bases({
            0: [[[1.0, 0, 0], [0, 0, 0]]],
            1: [[[0, 1.0, 0], [0, 0, 0]]],
        })
        rebuilt = OPDBasis(
            irrep_label="X3-", mode_index=1, degeneracy=1, n_arms=1, within_arm=1,
            structure=_structure(2),  # equal, but NOT the same object
            opds=bases[1].opds,
        )
        with self.assertRaises(OPDError) as caught:
            opd_bases_for_eigenvectors({0: bases[0], 1: rebuilt}, np.eye(2))
        self.assertIn("different cell", str(caught.exception))

    def test_empty_input_is_rejected(self):
        with self.assertRaises(OPDError):
            opd_bases_for_eigenvectors({}, np.eye(1))


class TestOPDIdentityIsCellIndependent(unittest.TestCase):
    """Position `a` must name the SAME physical direction on every cell.

    THE INVARIANT THIS PINS. The same irrep gets built on genuinely
    different-sized cells depending on what it is being built WITH: its own
    native cell alone, or a joint cell shared with other irreps. Real CMO
    Gamma is 24 atoms per-irrep against 192 k-point-wide, TiO2 12 against
    96, BaTiO3 15 against 405. A user reads `--direction a,b` off a report
    and passes it to `freeze` or `decomposition`. If the two orderings ever
    disagreed, the same string would silently mean two different
    distortions, and NOTHING downstream would error -- the amplitudes would
    simply be wrong.

    This used to be reachable straight from the CLI: `irrep GM5+ --opd`
    reported the own-cell ordering and `irrep GM --opd` the shared one, so
    the two reports had to agree. `irrep --opd` is per-irrep in BOTH forms
    now (v1.23.0), which removes that particular collision but not the
    invariant: `freeze X2+ X3-` still builds a joint cell across irreps,
    a multi-arm `--direction` spec across arms, and the amplitude-baseline script (development)'s
    JOINT-CELL SECTION still measures the shared construction directly.
    Every one of those reads position letters that must keep meaning what
    the per-irrep report said they mean.

    Until this test existed the ordering was only a convention
    (`decomposition._opd_sort_key`, arm-major then within-arm), never a
    checked property. The per-irrep centring-inflation fix widened the gap
    between the two cells considerably, which is what made pinning it
    urgent rather than tidy.

    Two levels, because either alone is insufficient:

    - STRUCTURAL: position `i` has the same `(arm, within_index, mechanism)`
      on both cells. This is what actually decides the label, and it holds
      at any k-point.
    - PHYSICAL: the displacement patterns agree. Checked at GAMMA only, and
      deliberately so -- at `q = 0` the pattern is periodic with the small
      cell, so every periodic image carries the same displacement and
      folding is exact. At a non-Gamma k-point images differ by a Bloch
      phase and a folded comparison would be meaningless, not merely noisy.
    """

    MATERIALS = [("CMO", "POSCAR_CMO", "CMO_findsym_GM", "findsym_7.1.6_CMO.log"),
                 ("TiO2", "POSCAR_TiO2", "TiO2_findsym_GM", "findsym_7.1.6_TiO2.log")]

    def _bases(self, poscar, smodes_file, findsym_file):
        """Both cells built the SAME way -- with a findsym log, without a
        findsym runner.

        NO RUNNER IS DELIBERATE, and it has a visible consequence worth
        stating so a reader is not alarmed by it. Real TiO2 GM5+ needs
        findsym ESCALATION to reach its within-arm partner (measured: log
        alone gives rank 1/2, log plus runner gives 2/2), so in this
        configuration its basis is rank-deficient and
        `build_joint_opd_cell` correctly warns that the cell hosts fewer
        OPDs than the mode-count sum rule declares. That is the warning
        doing its job, not a defect.

        It does not weaken what is under test. The invariant is that the
        two cells AGREE, and both are built identically, so a partner
        missing from one is missing from the other. Running the escalation
        would mean real findsym subprocesses in the suite -- the cost this
        project already paid once, when an unconditional findsym call took
        the suite from ~20s to ~157s. CMO needs no escalation at all and
        carries the comparison on its own: 24 (irrep, mode) pairs across 36
        positions, every one with genuinely different cell sizes."""
        parent = Structure.from_poscar(FIXTURES / poscar)
        log_text = (FIXTURES / findsym_file).read_text()
        irreps = list(parse_smodes_output((FIXTURES / smodes_file).read_text()).irreps)
        per_irrep = {ir.label: build_opd_bases([ir], parent, findsym_log_text=log_text)
                     for ir in irreps}
        k_wide = build_opd_bases(irreps, parent, findsym_log_text=log_text)
        return parent, irreps, per_irrep, k_wide

    def test_position_ordering_is_identical_on_both_cells(self):
        for name, poscar, smodes_file, findsym_file in self.MATERIALS:
            with self.subTest(material=name):
                _parent, irreps, per_irrep, k_wide = self._bases(poscar, smodes_file, findsym_file)
                compared = 0
                for irrep in irreps:
                    for (label, mode), wide in k_wide.items():
                        if label != irrep.label:
                            continue
                        own = per_irrep[irrep.label].get((label, mode))
                        if own is None:
                            continue
                        self.assertEqual(
                            [(o.position, o.arm, o.within_index, o.mechanism) for o in own.opds],
                            [(o.position, o.arm, o.within_index, o.mechanism) for o in wide.opds],
                            f"{name} {label} mode {mode}: position ordering differs between the "
                            f"{own.structure.n_atoms}-atom per-irrep cell and the "
                            f"{wide.structure.n_atoms}-atom k-point cell",
                        )
                        compared += 1
                self.assertGreater(compared, 0, "no (irrep, mode) pairs were actually compared")

    def test_direction_spec_strings_are_identical_on_both_cells(self):
        """The user-facing consequence of the ordering invariant: the exact
        `--direction` string must not depend on which report it was read
        from."""
        for name, poscar, smodes_file, findsym_file in self.MATERIALS:
            with self.subTest(material=name):
                _parent, irreps, per_irrep, k_wide = self._bases(poscar, smodes_file, findsym_file)
                for (label, mode), wide in k_wide.items():
                    own = per_irrep[label].get((label, mode))
                    if own is None:
                        continue
                    self.assertEqual(
                        [own.direction_spec(i) for i in range(len(own.opds))],
                        [wide.direction_spec(i) for i in range(len(wide.opds))],
                        f"{name} {label} mode {mode}: --direction strings differ between cells",
                    )

    def test_patterns_agree_after_folding_at_gamma(self):
        """The physical half. At Gamma the big cell is a plain supercell of
        the small one with no Bloch phase, so each small-cell atom's
        displacement must equal that of every one of its periodic images."""
        for name, poscar, smodes_file, findsym_file in self.MATERIALS:
            with self.subTest(material=name):
                _parent, irreps, per_irrep, k_wide = self._bases(poscar, smodes_file, findsym_file)
                checked = 0
                for (label, mode), wide in k_wide.items():
                    own = per_irrep[label].get((label, mode))
                    if own is None or own.structure.n_atoms == wide.structure.n_atoms:
                        continue
                    mapping = self._fold_map(own.structure, wide.structure)
                    for position, (small_opd, wide_opd) in enumerate(zip(own.opds, wide.opds)):
                        folded = np.array([wide_opd.pattern[j] for j in mapping], dtype=np.float64)
                        small = np.asarray(small_opd.pattern, dtype=np.float64)
                        norms = np.linalg.norm(folded) * np.linalg.norm(small)
                        if norms == 0.0:
                            continue
                        cosine = abs(float(np.sum(folded * small) / norms))
                        self.assertGreater(
                            cosine, 0.99,
                            f"{name} {label} mode {mode} position {position}: the same labelled "
                            f"direction is a DIFFERENT distortion on the two cells (|cos| "
                            f"{cosine:.4f})",
                        )
                        checked += 1
                if checked == 0:
                    # Under v2.0 a Gamma irrep's own cell and the k-point cell
                    # are the same size, so compare against an explicitly
                    # requested 2x2x1 supercell instead -- the property (one
                    # label, one distortion, whatever the cell) is unchanged.
                    wide_lattice = np.diag([2.0, 2.0, 1.0]) @ np.asarray(_parent.lattice, dtype=np.float64)
                    log = (FIXTURES / findsym_file).read_text()
                    for label, bases in per_irrep.items():
                        wider = build_opd_bases([b for b in irreps if b.label == label], _parent,
                                                target_lattice=wide_lattice, findsym_log_text=log)
                        for key, own in bases.items():
                            mapping = self._fold_map(own.structure, wider[key].structure)
                            for small_opd, wide_opd in zip(own.opds, wider[key].opds):
                                folded = np.array([wide_opd.pattern[j] for j in mapping], dtype=np.float64)
                                small = np.asarray(small_opd.pattern, dtype=np.float64)
                                norms = np.linalg.norm(folded) * np.linalg.norm(small)
                                if norms:
                                    self.assertGreater(abs(float(np.sum(folded * small) / norms)), 0.99)
                                    checked += 1
                self.assertGreater(checked, 0, "no differing-size cells were actually compared")

    @staticmethod
    def _fold_map(small, big):
        """One `big` atom index per `small` atom, matched by fractional
        position in the SMALL cell's own frame under minimum image."""
        inverse = np.linalg.inv(np.asarray(small.lattice, dtype=np.float64))
        big_frac = np.mod(np.asarray(big.cart_coords, dtype=np.float64) @ inverse, 1.0)
        mapping = []
        for position, species in zip(np.mod(small.frac_coords, 1.0), small.species):
            best = None
            for j in range(big.n_atoms):
                if big.species[j] != species:
                    continue
                delta = big_frac[j] - position
                delta -= np.round(delta)
                distance = float(np.linalg.norm(delta))
                if best is None or distance < best[0]:
                    best = (distance, j)
            assert best is not None and best[0] < 1e-3, f"no folded match for {species}"
            mapping.append(best[1])
        return mapping

    def test_a_swapped_ordering_would_be_caught(self):
        """Negative control. The three tests above pass, which is only
        meaningful if they could fail -- a comparison that accepts anything
        would pass them too. Swapping two positions on one cell must break
        the ordering comparison."""
        parent = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        irreps = list(parse_smodes_output((FIXTURES / "CMO_findsym_GM").read_text()).irreps)
        degenerate = next(ir for ir in irreps if ir.degeneracy > 1)
        own = build_opd_bases([degenerate], parent)[(degenerate.label, 0)]

        swapped = list(own.opds)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        corrupted = OPDBasis(
            irrep_label=own.irrep_label, mode_index=own.mode_index,
            degeneracy=own.degeneracy, n_arms=own.n_arms, within_arm=own.within_arm,
            structure=own.structure, opds=tuple(swapped),
        )

        self.assertNotEqual(
            [(o.position, o.arm, o.within_index, o.mechanism) for o in own.opds],
            [(o.position, o.arm, o.within_index, o.mechanism) for o in corrupted.opds],
            "the ordering comparison must distinguish a swapped basis, or it proves nothing",
        )


class TestTargetLattice(unittest.TestCase):
    """`build_opd_bases(..., target_lattice=...)` -- the third cell.

    `irrep --opd` wants the irrep's own cell, `freeze` the smallest hosting
    the requested directions, and `decomposition` gets NO choice: the user
    hands it a relaxed CONTCAR and the analysis must happen on that cell.
    Routing all three through one construction is what stops `decomposition`
    re-deriving directions by a second path -- two implementations of the
    same physics being the failure this module exists to prevent."""

    def setUp(self):
        self.parent = Structure.from_poscar(FIXTURES / "POSCAR_CMO")
        self.log = (FIXTURES / "findsym_7.1.6_CMO.log").read_text()
        irreps = list(parse_smodes_output((FIXTURES / "CMO_findsym_GM").read_text()).irreps)
        self.irrep = next(ir for ir in irreps if ir.label == "GM5-")
        self.lattice = np.asarray(self.parent.lattice, dtype=np.float64)

    def _build(self, target=None):
        return build_opd_bases(
            [self.irrep], self.parent, target_lattice=target, findsym_log_text=self.log,
        )[(self.irrep.label, 0)]

    def test_requested_cell_is_the_cell_produced(self):
        self.assertEqual(self._build().structure.n_atoms, 24)
        self.assertEqual(self._build(self.lattice).structure.n_atoms, 24)
        self.assertEqual(self._build(np.diag([2, 2, 1]) @ self.lattice).structure.n_atoms, 96)
        self.assertEqual(self._build(np.diag([2, 2, 2]) @ self.lattice).structure.n_atoms, 192)

    def test_a_target_smaller_than_the_parent_is_refused_with_the_right_reason(self):
        """A `--sub` smaller than `--parent` is a REAL, already-handled case,
        and the answer is to enlarge sub -- never to reduce the parent.

        This must not be reported as a rounding failure. For a centred
        setting the irrep's own native cell genuinely IS smaller (real CMO
        Gamma: 12 atoms against a 24-atom conventional parent, an exact
        det = 1/2 relationship, not numerical noise). It is refused because
        hosting it would mean reducing the parent, which the developer notes records
        as a do-not-retry: `irrep.k_vector` and `irrep.superlattice` are tied
        to the basis smodes was run against, and an I-centring reduction
        broke multi-arm blend detection for real CMO.
        """
        with self.assertRaises(OPDError) as caught:
            self._build(np.asarray(self.irrep.superlattice))
        message = str(caught.exception)
        self.assertIn("SMALLER than the parent", message)
        self.assertIn("enlarge_onto_lattice", message)
        # Must NOT be misreported as the strain/rounding case, which would
        # send the caller to fix the wrong thing.
        self.assertNotIn("not an integer multiple", message)

    def test_a_non_integer_target_is_rejected_rather_than_rounded(self):
        """A cell that is not a whole multiple of the parent cannot host its
        periodicity, so there is no correct rounding -- guessing would give a
        cell that tiles arithmetically and means nothing physically."""
        with self.assertRaises(OPDError) as caught:
            self._build(self.lattice * 1.37)
        self.assertIn("integer multiple", str(caught.exception))

    def test_opd_identity_survives_the_move(self):
        """THE GATE. Moving to another cell must not change what a position
        MEANS -- otherwise `decomposition` on `sub`'s cell would report
        amplitudes against different directions than `irrep --opd` named,
        with nothing downstream able to notice."""
        small = self._build()
        big = self._build(np.diag([2, 2, 1]) @ self.lattice)
        self.assertLess(small.structure.n_atoms, big.structure.n_atoms)

        self.assertEqual(
            [(o.position, o.arm, o.within_index, o.mechanism) for o in small.opds],
            [(o.position, o.arm, o.within_index, o.mechanism) for o in big.opds],
        )
        self.assertEqual(
            [small.direction_spec(i) for i in range(len(small.opds))],
            [big.direction_spec(i) for i in range(len(big.opds))],
        )

        mapping = TestOPDIdentityIsCellIndependent._fold_map(small.structure, big.structure)
        compared = 0
        for position, (small_opd, big_opd) in enumerate(zip(small.opds, big.opds)):
            folded = np.array([big_opd.pattern[j] for j in mapping], dtype=np.float64)
            reference = np.asarray(small_opd.pattern, dtype=np.float64)
            norms = np.linalg.norm(folded) * np.linalg.norm(reference)
            if norms == 0.0:
                continue
            cosine = abs(float(np.sum(folded * reference) / norms))
            self.assertGreater(
                cosine, 0.99,
                f"position {position} is a DIFFERENT distortion on the target cell "
                f"(|cos| {cosine:.4f})",
            )
            compared += 1
        self.assertGreater(compared, 0)

    def test_fold_direction_matters_and_is_asserted(self):
        """A guard against the mistake made while developing this: folding
        was run with the larger cell as the reference, which produced 33
        spurious "violations" out of 36. The comparison is only meaningful
        smaller-into-larger, so the ordering is asserted rather than assumed."""
        small = self._build()
        big = self._build(np.diag([2, 2, 1]) @ self.lattice)
        with self.assertRaises(AssertionError):
            TestOPDIdentityIsCellIndependent._fold_map(big.structure, small.structure)


class TestCrossIrrepOverlap(unittest.TestCase):
    """Distinct irreps are ORTHOGONAL (Schur), and real SMODES modes honour
    it exactly -- 0.00e+00 across every fixture. A constructed OPD that
    overlaps another irrep is therefore not that irrep's direction at all.

    THIS IS THE CHECK THAT TESTS IDENTITY, NOT A PROPERTY. Six cheap
    validity tests have failed in this project -- count, rank, conditioning,
    tileability, folded-cosine, rank-with-conditioning -- and every one asks
    something ABOUT a basis rather than whether its vectors are the ones
    they claim to be. Real STO GM4- reports `rank 3/3, cond 1.0` while its
    third direction IS a GM5- one.
    """

    # Includes the TIME-REVERSAL-PAIRED k-points, which are the only place
    # the `quadrature` mechanism is exercised at all -- graphite K uses it 5
    # times plus one `quad+within`, CsV3Sb5 K nine times. Without them a
    # whole construction route goes untested.
    CLEAN = [("CMO", "POSCAR_CMO", "CMO_findsym_X"),
             ("TiO2", "POSCAR_TiO2", "TiO2_findsym_GM"),
             ("STO", "POSCAR_STO", "STO_findsym_M"),
             ("STO", "POSCAR_STO", "STO_findsym_Gamma"),
             ("STO", "POSCAR_STO", "STO_findsym_R"),
             ("graphite", "POSCAR_graphite", "graphite_findsym_K"),
             ("graphite", "POSCAR_graphite", "graphite_findsym_H"),
             ("graphite", "POSCAR_graphite", "graphite_findsym_L"),
             ("graphite", "POSCAR_graphite", "graphite_findsym_M"),
             ("CsV3Sb5", "POSCAR_CsV3Sb5", "CsV3Sb5_findsym_K"),
             ("CsV3Sb5", "POSCAR_CsV3Sb5", "CsV3Sb5_findsym_L"),
             ("CsV3Sb5", "POSCAR_CsV3Sb5", "CsV3Sb5_findsym_M")]
    # STO Gamma and STO R moved to CLEAN when the inverse-permutation bug
    # was fixed (0.707107 and 1.000000 both -> 0.000000). CMO Gamma remains,
    # from a DIFFERENT cause: the primitive cell of a centred setting admits
    # spurious symmetries, so the native-cell search there sees operations
    # the real crystal does not have.
    # NOT SHIPPED BEHAVIOUR -- this measures the SOURCE 1 FLOOR. The suite is
    # subprocess-free, so it supplies no `findsym_runner`, and the within-arm
    # search falls back to Source 1, which on a centred cell substitutes
    # accidental signed-permutation self-maps for the screws and glides it
    # cannot represent. One of those maps GM5+ onto ~GM1+. WITH a runner --
    # which the CLI always supplies -- this same overlap is 0.000000. See
    # known issue 11 in the developer notes. Kept as a ratchet on the Source-1 path, NOT as a
    # statement that CMO Gamma leaks.
    #
    # TIGHTENED 2026-09-20: was 1.000000, the value when the leak was first
    # recorded. Measured today it is 0.981412, so the old bound was loose by
    # 0.019 -- it would have let the leak grow most of the way back to a full
    # collapse without failing. Pinned to the observed value so this is a real
    # ratchet. Confirmed NOT a Source-1 artifact: identical with no findsym
    # log, with the parent's log, and with a real findsym runner (see
    # the developer notes, "What the suite can and cannot reach").
    LEAKING = [("CMO", "POSCAR_CMO", "CMO_findsym_GM", 0.981412)]

    def _worst(self, poscar, smodes_file):
        parent = Structure.from_poscar(FIXTURES / poscar)
        irreps = list(parse_smodes_output((FIXTURES / smodes_file).read_text()).irreps)
        return cross_irrep_overlap(build_opd_bases(irreps, parent))[0]

    def test_clean_k_points_stay_clean(self):
        """The regression guard that matters. Two attempts to fix the leak
        broke one of these -- real STO M went from 0.000000 to 0.707107 --
        and both were caught only by re-surveying by hand."""
        for name, poscar, smodes_file in self.CLEAN:
            with self.subTest(material=name, k=smodes_file):
                # Against OPD_MAX_CROSS_IRREP_OVERLAP, not zero. The
                # quadrature k-points carry the documented ~1e-5 per-irrep
                # cross-talk floor; the structural failures this guards
                # against are 0.707 and 1.0.
                self.assertLess(self._worst(poscar, smodes_file),
                                OPD_MAX_CROSS_IRREP_OVERLAP)

    def test_known_leaks_do_not_get_worse(self):
        """Pins the CURRENT scope of the open defect. These are expected to
        FAIL-to-improve, not to pass -- when the construction is fixed these
        assertions become trivially true and the case moves to CLEAN above.
        Until then this stops the leak silently spreading."""
        for name, poscar, smodes_file, known in self.LEAKING:
            with self.subTest(material=name, k=smodes_file):
                self.assertLessEqual(self._worst(poscar, smodes_file), known + 1e-6)

    def test_raw_smodes_modes_are_exactly_orthogonal(self):
        """The inputs are sound, so the defect is in the CONSTRUCTION. Worth
        pinning: if this ever fails, the whole diagnosis changes.

        Compared against OPD_MAX_CROSS_IRREP_OVERLAP, not zero: real
        POSCAR_CsV3Sb5 carries only 6 decimal places and cannot represent
        its hexagonal `b_y = a*sqrt(3)/2` exactly (documented in the developer notes),
        so its raw modes sit at 7e-5 to 9e-5 -- precision loss in the input
        file, not a symmetry failure.

        COMPARED ONLY WITHIN MATCHING SUPERLATTICES, and that restriction is
        load-bearing rather than cautious. A raw SMODES pattern is expressed
        in its own irrep's cell and atom ordering; at a multi-arm k-point
        different irreps need not share either, so comparing the arrays
        index-wise is meaningless. Written without this guard, the test
        reported real STO M's raw modes at 0.9999999999999998 -- which says
        nothing about orthogonality and everything about comparing two
        different frames, the same trap `OPDBasis` exists to prevent."""
        for name, poscar, smodes_file in self.CLEAN + [(n, p, s) for n, p, s, _ in self.LEAKING]:
            with self.subTest(material=name, k=smodes_file):
                irreps = list(parse_smodes_output((FIXTURES / smodes_file).read_text()).irreps)
                groups: dict = {}
                for irrep in irreps:
                    key = np.asarray(irrep.superlattice, dtype=np.float64).round(6).tobytes()
                    groups.setdefault(key, []).append(irrep)

                compared = 0
                for group in groups.values():
                    if len({ir.label for ir in group}) < 2:
                        continue
                    vectors = []
                    for irrep in group:
                        for mode in irrep.modes:
                            v = np.asarray(mode.displacements, dtype=np.float64).ravel()
                            norm = np.linalg.norm(v)
                            if norm > 0:
                                vectors.append((irrep.label, v / norm))
                    for i in range(len(vectors)):
                        for j in range(i + 1, len(vectors)):
                            if vectors[i][0] == vectors[j][0]:
                                continue
                            compared += 1
                            self.assertLess(
                                abs(float(vectors[i][1] @ vectors[j][1])),
                                OPD_MAX_CROSS_IRREP_OVERLAP,
                                f"{name}: raw {vectors[i][0]} and {vectors[j][0]} are not "
                                "orthogonal -- the INPUTS would then be at fault, not the "
                                "construction, and the whole diagnosis changes",
                            )
                # NOT asserted to be non-zero. Sibling irreps at one k-point
                # routinely have DIFFERENT superlattices -- real STO Gamma's
                # GM4- and GM5- do, as do CMO Gamma's GM5+/GM5-, STO R's R2-
                # and TiO2 Gamma's GM5+/GM5-. That is the documented
                # orientation-split class (the developer notes records it for BaTiO3:
                # same lattice, different basis, atom sets not corresponding
                # row-wise). Where no two irreps share a frame there is simply
                # nothing this test can compare, and demanding a comparison
                # would only tempt a future reader to remove the guard and
                # compare across frames anyway -- which is what produced the
                # meaningless 0.9999999999999998 this test was written after.
                del compared

    def test_overlap_is_reported_with_the_offending_pair_named(self):
        """Uses real CMO Gamma, the one case still leaking. It previously
        used STO Gamma, which is now clean -- the inverse-permutation fix
        took it from 0.707107 to 0.000000, so the assertion it encoded is
        obsolete rather than broken."""
        # v2.0 has no real leaking case left (CMO Gamma: 0.98 -> 5e-17), so
        # the REPORTING is exercised on two deliberately overlapping bases.
        shared = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        first = _basis([shared])
        second = _basis([shared])
        object.__setattr__(second, "irrep_label", "X2+")
        worst, description = cross_irrep_overlap({("X3-", 0): first, ("X2+", 0): second})
        self.assertGreater(worst, 0.7)
        self.assertIn("X3-", description)
        self.assertIn("X2+", description)

    def test_a_single_irrep_has_nothing_to_compare(self):
        parent = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        irrep = parse_smodes_output((FIXTURES / "STO_findsym_Gamma").read_text()).irrep("GM4-")
        worst, description = cross_irrep_overlap(build_opd_bases([irrep], parent))
        self.assertEqual(worst, 0.0)
        self.assertIsNone(description)


class TestOPDRoundTrip(unittest.TestCase):
    """Freeze along one OPD, decompose, and check the amplitude comes back.

    STRICTLY STRONGER THAN AN ORTHOGONALITY CHECK, and it must stay in the
    suite for that reason. `cross_irrep_overlap` compares patterns pairwise
    on one cell; this drives the whole real path -- construct the OPD, build
    a displaced structure, decompose it, read the amplitude back -- so it
    also catches wrong tiling, wrong amplitude scaling, and a pattern that is
    fine in isolation but mis-assembled.

    It is also a CROSS-CHECK BETWEEN TWO INDEPENDENT IMPLEMENTATIONS.
    `decomposition` derives its own directions through `mode_direction_sweep`,
    not through `build_opd_bases`, so agreement here is real evidence rather
    than one implementation confirming itself.

    THE FAILURE IT CATCHES THAT ORTHOGONALITY CANNOT. An OPD orthogonal to
    EVERYTHING -- including its own irrep's decomposition directions --
    recovers zero amplitude and leaves its whole displacement unexplained.
    A Gram check rewards exactly that: real STO R reports 0.000000
    cross-irrep overlap while an OPD frozen there round-trips to 0.0000.
    the developer notes records the same mode for graphite K's quadrature columns.
    """

    AMPLITUDE = 0.001

    def _round_trip(self, poscar, smodes_file, k_vector):
        """(min fraction recovered on the OWN irrep, max fraction leaked)."""
        parent = Structure.from_poscar(FIXTURES / poscar)
        irreps = list(parse_smodes_output((FIXTURES / smodes_file).read_text()).irreps)
        bases = build_opd_bases(irreps, parent)

        own_worst, leak_worst, checked = 1.0, 0.0, 0
        for (label, mode), basis in sorted(bases.items()):
            if mode != 0:
                continue
            for opd in basis.opds:
                sub = basis.structure.displaced(
                    self.AMPLITUDE * np.asarray(opd.pattern, dtype=np.float64)
                )
                result = decompose_at_k_vector(
                    k_vector, irreps, parent, sub, check_stoichiometry(parent, sub)
                )
                own = max((abs(a.amplitude_angstrom)
                           for a in result.amplitudes if a.label == label), default=0.0)
                other = max((abs(a.amplitude_angstrom)
                             for a in result.amplitudes if a.label != label), default=0.0)
                own_worst = min(own_worst, own / self.AMPLITUDE)
                leak_worst = max(leak_worst, other / self.AMPLITUDE)
                checked += 1
        self.assertGreater(checked, 0, "no OPDs were round-tripped")
        return own_worst, leak_worst

    def test_sto_gamma_round_trips_exactly(self):
        """Real STO Gamma, the case the inverse-permutation fix repaired.
        Every OPD must return its full injected amplitude on its own irrep
        and nothing anywhere else."""
        own, leak = self._round_trip("POSCAR_STO", "STO_findsym_Gamma", "GM")
        self.assertGreater(own, 0.99, "an OPD failed to recover its own amplitude")
        self.assertLess(leak, 0.025, "an OPD leaked into another irrep")

    # ONLY STO Gamma FOR NOW, and the reason is a decomposition limit rather
    # than an OPD one. `build_opd_bases` returns the k-point JOINT cell, and
    # `decomposition` cannot resolve one: real CMO Gamma's 96-atom joint cell
    # against a 24-atom parent raises
    #     "no supercell (diagonal or otherwise) of GM1+'s own native cell, at
    #      the required 4x volume, matches the sub structure's own lattice"
    # and TiO2 Gamma's 48-atom joint cell fails the same way. STO Gamma is
    # the case where the joint cell IS the parent cell, so nothing needs
    # resolving.
    #
    # Measured on PER-IRREP cells instead, which decomposition can resolve:
    #     TiO2 Gamma   own 1.0000   leak 0.0283
    #     CMO Gamma    own 0.0000   leak 1.0000   (the known GM5+/GM1+ defect)
    #     STO R        own 0.0000   leak 0.0000   (recovers NOTHING -- and the
    #                                              Gram check calls it clean)
    # Those belong here once either the joint cells shrink to something
    # decomposition can resolve, or this drives per-irrep cells directly.




class TestMultiPositionOPDRoundTrip(unittest.TestCase):
    """Freeze SEVERAL OPD positions at once and require every one back.

    THE NINTH CHEAP VALIDITY TEST, and the first that freezes more than one
    position. the amplitude baseline (development data) -- 716 rows, 0 wrong, and the
    project's main OPD correctness reference -- only ever freezes ONE. That is
    structurally blind to a wrong basis: the image basis and the SVD basis are
    both orthonormal bases of the SAME subspace, so freezing basis vector `i`
    and projecting recovers vector `i` whichever one was chosen. Regenerating
    that whole baseline after the v1.33.0 fix changed not a single row beyond
    signed zeros.

    Freezing a COMBINATION is what separates them, and it is what the
    physicist's own real runs do (`Freeze_X5--m2-sym-1a-2a`). Measured against
    the pre-fix code with the same committed log this test uses:

        STO X5- m0, all six equal   +0.001084 -0.002288 +0.010341
                                    -0.001453 +0.001084 +0.002822
        STO M5- m1, all six equal   two positions exactly ZERO, four at 0.010000

    THE RESIDUAL CANNOT SEE THIS -- it is 0.000% for both of those, and for the
    correct answer too. The fit really does reconstruct the structure; it just
    distributes it over the wrong directions. So this test asserts on the
    per-position AMPLITUDES and never on the residual.

    Real BaSnO3 confirms both cases outside this codebase
    (`tests/fixtures/BSO smodes phonons/`): R4- frozen `1a-2a-3a` came back
    `a,b,b` (-0.005029 / +0.004961), and X5- frozen `1a-2a` came back
    `a,0,0,b,0,c`, leaking onto positions 4 and 6 with a reported Gram
    off-diagonal of 0.9460.

    SUBPROCESS-FREE, deliberately, like the rest of this suite -- but it uses a
    COMMITTED findsym log rather than no log at all. That distinction is
    load-bearing: the defect is invisible under Source 1's exact integer signed
    permutations and only appears under Source 2's CIF-derived floats, so a
    log-less version of this test would pass on the broken code. It runs in
    ~7.5s.
    """

    AMPLITUDE = 0.01

    # The three irreps whose real BSO runs failed, all on the committed STO
    # stand-in (both cubic Pm-3m perovskites). R4- is degeneracy 3; X5- and
    # M5- are degeneracy 6, where there is room to misattribute.
    CASES = [
        ("STO_findsym_R", "R", "R4-"),
        ("STO_findsym_X", "X", "X5-"),
        ("STO_findsym_M", "M", "M5-"),
    ]

    def test_freezing_every_position_returns_every_position(self):
        parent = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        log = (FIXTURES / "findsym_7.1.6_STO.log").read_text()
        for smodes_file, k_vector, label in self.CASES:
            if not (FIXTURES / smodes_file).is_file():
                continue
            irreps = list(parse_smodes_output((FIXTURES / smodes_file).read_text()).irreps)
            target = [i for i in irreps if i.label == label]
            self.assertTrue(target, f"{label} missing from {smodes_file}")
            bases = build_opd_bases(target, parent, findsym_log_text=log)

            for (found, mode_index), basis in sorted(bases.items()):
                if len(basis.opds) < 2:
                    continue
                pattern = sum(np.asarray(o.pattern, dtype=np.float64) for o in basis.opds)
                sub = basis.structure.displaced(self.AMPLITUDE * pattern)
                result = decompose_at_k_vector(
                    k_vector, irreps, parent, sub,
                    check_stoichiometry(parent, sub), findsym_log_text=log,
                )
                amplitudes = [
                    a.amplitude_angstrom for a in result.amplitudes
                    if a.label == found and a.mode_index == mode_index
                ]
                with self.subTest(irrep=found, mode=mode_index):
                    self.assertEqual(len(amplitudes), len(basis.opds))
                    for position, value in enumerate(amplitudes):
                        # MAGNITUDE only. An individual position's overall SIGN
                        # is an artifact of which symmetry operation the search
                        # found -- real STO X5- returns +,-,+,-,+,- for an
                        # all-positive freeze -- and the developer notes already documents
                        # that a per-mode sign is not physically comparable.
                        #
                        # 5% of the injected amplitude, against a documented
                        # ~1e-5 A absolute construction floor (0.2% here). The
                        # failures this guards against are 10x to 100x larger:
                        # a lost position reads 0.000000, and a misattributed
                        # one landed at 0.001084 for an injected 0.010000.
                        self.assertAlmostEqual(
                            abs(value), self.AMPLITUDE,
                            delta=0.05 * self.AMPLITUDE,
                            msg=(f"{found} mode {mode_index} position "
                                 f"{LETTERS[position]}: froze every position at "
                                 f"{self.AMPLITUDE} and this one came back "
                                 f"{value:+.6f}. The residual will NOT show this "
                                 f"-- it stays 0.000% while the amplitude is "
                                 f"distributed over the wrong directions."),
                        )


class TestOPDReportCarriesArmQVectors(unittest.TestCase):
    """The Q-vectors moved here from the removed `star` command.

    A CDW paper states its order parameter as Q-vectors; this project's own
    bookkeeping says "arm 2". Without these printed somewhere, the two cannot
    be checked against each other -- which is the whole reason `arm_k_vectors`
    survived `star`'s removal.

    **UNWRAPPED, deliberately.** `arm_k_vectors`'s cosmetic `(-0.5, 0.5]` wrap
    maps -0.5 onto +0.5, so real TiO2 X's and real CMO X's two genuinely
    distinct arms -- (1/2,1/2,0) and (1/2,-1/2,0) -- both printed as
    (1/2,1/2,0). An irrep whose arms carry duplicate Q-vectors reads as a bug.
    """

    def test_csv3sb5_m1_plus_matches_the_published_star(self):
        # Real DFT literature gives the star of M as (1/2,0,0), (0,1/2,0),
        # (-1/2,-1/2,0). The third differs from ours by (0,1,0), a reciprocal
        # lattice vector, so it is the same physical k-point.
        parent = Structure.from_poscar(FIXTURES / "POSCAR_CsV3Sb5_exact")
        irrep = next(
            i for i in parse_smodes_output(
                (FIXTURES / "CsV3Sb5_findsym_M").read_text()
            ).irreps if i.label == "M1+"
        )
        bases = build_opd_bases(
            [irrep], parent,
            findsym_log_text=(FIXTURES / "findsym_7.1.6_CsV3Sb5.log").read_text(),
        )
        basis = bases[("M1+", 0)]
        self.assertEqual(len(basis.arm_k_vectors), 3, "M1+ has a 3-arm star")

        published = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (-0.5, -0.5, 0.0)]
        for q in basis.arm_k_vectors:
            self.assertTrue(
                any(
                    np.allclose(np.round(np.array(q) - np.array(p)), np.array(q) - np.array(p),
                                atol=1e-6)
                    for p in published
                ),
                f"{q} is not any published star member up to a reciprocal lattice vector",
            )

    def test_arms_of_one_irrep_never_share_a_q_vector(self):
        # The exact defect the wrapped form has on a body-centred lattice.
        for poscar, smodes_file, log, label in (
            ("POSCAR_TiO2", "TiO2_findsym_X", "findsym_7.1.6_TiO2.log", "X1"),
            ("POSCAR_CMO", "CMO_findsym_X", "findsym_7.1.6_CMO.log", "X1+"),
        ):
            with self.subTest(label=label):
                parent = Structure.from_poscar(FIXTURES / poscar)
                irrep = next(
                    i for i in parse_smodes_output(
                        (FIXTURES / smodes_file).read_text()
                    ).irreps if i.label == label
                )
                basis = build_opd_bases(
                    [irrep], parent,
                    findsym_log_text=(FIXTURES / log).read_text(),
                )[(label, 0)]
                unique = {tuple(round(c, 6) for c in q) for q in basis.arm_k_vectors}
                self.assertEqual(
                    len(unique), len(basis.arm_k_vectors),
                    f"{label} printed duplicate Q-vectors: {basis.arm_k_vectors}",
                )

    def test_single_arm_irrep_prints_no_q_vector_line(self):
        # At Gamma the one Q-vector is (0,0,0) and the header already names
        # the k-point -- printing it is pure noise.
        parent = Structure.from_poscar(FIXTURES / "POSCAR_STO")
        irrep = next(
            i for i in parse_smodes_output(
                (FIXTURES / "STO_findsym_Gamma").read_text()
            ).irreps if i.label == "GM4-"
        )
        basis = build_opd_bases([irrep], parent)[("GM4-", 0)]
        self.assertNotIn(
            "arm Q-vectors", "\n".join(format_opd_basis(basis, show_patterns=False))
        )

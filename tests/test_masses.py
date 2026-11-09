import unittest

from smodes_phonons.masses import ATOMIC_MASSES, mass_for_species


class TestMassForSpecies(unittest.TestCase):
    def test_species_seen_in_fixtures(self):
        # Values legacy actually used (Irrep_mode_i files), for comparison:
        # Sr 87.62, Ti 47.867, O 16, Ca 40.078, Mn 54.94.
        self.assertAlmostEqual(mass_for_species("Sr"), 87.62, places=2)
        self.assertAlmostEqual(mass_for_species("Ti"), 47.867, places=2)
        self.assertAlmostEqual(mass_for_species("O"), 16.0, places=0)
        self.assertAlmostEqual(mass_for_species("Ca"), 40.078, places=2)
        self.assertAlmostEqual(mass_for_species("Mn"), 54.94, places=1)

    def test_unknown_symbol_raises_key_error(self):
        with self.assertRaises(KeyError):
            mass_for_species("Xx")

    def test_case_sensitive(self):
        with self.assertRaises(KeyError):
            mass_for_species("sr")

    def test_table_covers_full_periodic_table(self):
        # Spot-check first, a middle, and the last few known elements.
        self.assertIn("H", ATOMIC_MASSES)
        self.assertIn("Fe", ATOMIC_MASSES)
        self.assertIn("U", ATOMIC_MASSES)
        self.assertIn("Og", ATOMIC_MASSES)
        self.assertGreaterEqual(len(ATOMIC_MASSES), 118)


if __name__ == "__main__":
    unittest.main()

"""`doctor`'s check for the `iso` program, which v2.0's order-parameter
directions depend on. Subprocess-free: the binary lookup and the run are
replaced."""

import unittest
from unittest import mock

from smodes_phonons import doctor, iso


class TestIsoCheck(unittest.TestCase):
    def test_missing_iso_fails_and_says_what_needs_it(self):
        with mock.patch.object(doctor.isotropy, "find_binary", return_value=None):
            check = doctor._check_iso()
        self.assertFalse(check.ok)
        self.assertIn("freeze", check.detail)

    def test_iso_that_runs_reports_its_version(self):
        banner = "Isotropy, Version 9.6.1, Jan 2022\nHarold T. Stokes ...\n*\n"
        with mock.patch.object(doctor.isotropy, "find_binary", return_value="/usr/bin/iso"), \
                mock.patch.object(iso, "run_iso", return_value=banner):
            check = doctor._check_iso()
        self.assertTrue(check.ok)
        self.assertIn("Isotropy, Version 9.6.1", check.detail)

    def test_iso_on_path_that_fails_to_run_fails(self):
        with mock.patch.object(doctor.isotropy, "find_binary", return_value="/usr/bin/iso"), \
                mock.patch.object(iso, "run_iso", side_effect=OSError("exec format error")):
            check = doctor._check_iso()
        self.assertFalse(check.ok)
        self.assertIn("exec format error", check.detail)

    def test_output_without_the_banner_fails(self):
        with mock.patch.object(doctor.isotropy, "find_binary", return_value="/usr/bin/iso"), \
                mock.patch.object(iso, "run_iso", return_value="Cannot connect to the Docker daemon"):
            check = doctor._check_iso()
        self.assertFalse(check.ok)


if __name__ == "__main__":
    unittest.main()

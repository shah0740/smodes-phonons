"""The package's two declared versions must agree.

Written because they silently did not, for eight releases. `__version__`
was last bumped at v0.90.0 and stayed there through v0.91-v0.98 while every
commit message claimed a new version; `pyproject.toml` was stranded further
back still, at 0.40.0. Nothing checked, so `smodes-phonons --version`
reported 0.90.0 on code documented as v0.98.0 -- which is exactly how it was
found, by the physicist pulling and not seeing what the commit messages
promised.

A version claimed only in a commit message is not a version. This test makes
the two files the single source of truth and makes disagreement a failure
rather than something to notice months later.
"""

import re
import unittest
from pathlib import Path

import smodes_phonons

ROOT = Path(__file__).resolve().parent.parent


class TestVersionConsistency(unittest.TestCase):
    def test_pyproject_and_package_versions_agree(self):
        text = (ROOT / "pyproject.toml").read_text()
        match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
        self.assertIsNotNone(match, "pyproject.toml has no top-level version")
        self.assertEqual(
            match.group(1),
            smodes_phonons.__version__,
            "pyproject.toml and smodes_phonons.__version__ disagree -- bump both, "
            "or `smodes-phonons --version` reports something the code is not",
        )

    def test_version_is_a_plain_three_part_number(self):
        self.assertRegex(smodes_phonons.__version__, r"^\d+\.\d+\.\d+$")


if __name__ == "__main__":
    unittest.main()

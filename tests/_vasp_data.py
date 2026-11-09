"""The real VASP OUTCARs behind the phonon and Born-charge tests are research
data that is not distributed with this repository. Tests that need them are
marked `@requires_vasp_data` and skip when the files are absent."""

import unittest
from pathlib import Path

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
VASP_DATA_PRESENT = any(_FIXTURES.rglob("OUTCAR*"))

requires_vasp_data = unittest.skipUnless(
    VASP_DATA_PRESENT, "needs the VASP OUTCAR fixtures, which are not distributed"
)

"""`smodes-phonons doctor` -- sanity-check the environment before running a pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

from . import isotropy
from . import findsym as findsym_module


class Check:
    def __init__(self, name: str, ok: bool, detail: str):
        self.name = name
        self.ok = ok
        self.detail = detail


def _check_numpy() -> Check:
    try:
        import numpy

        return Check("numpy", True, f"found (version {numpy.__version__})")
    except ImportError:
        return Check("numpy", False, "not importable -- reinstall the smodes-phonons package")


def _check_binary(name: str) -> Check:
    path = isotropy.find_binary(name)
    if path:
        return Check(name, True, f"found on PATH ({path})")
    return Check(name, False, "not found on PATH -- install the ISOTROPY suite and/or update PATH")


def _check_findsym_version() -> Check:
    """Report the installed findsym's version and which input/log format this
    package will use for it -- 6.x+ gets keyword-format input (see
    findsym.py's module docstring), pre-6.x gets the older positional
    format. Both are supported and auto-detected, not a hard requirement,
    so this never fails the overall doctor run by itself: an undetectable
    version just means the (safe, HPC-matching) positional-format fallback
    will be used.
    """
    if isotropy.find_binary("findsym") is None:
        return Check("findsym version", True, "skipped -- findsym binary not found")

    try:
        version = findsym_module.detect_findsym_version()
    except FileNotFoundError:
        return Check("findsym version", True, "skipped -- findsym binary not found")
    except findsym_module.FindsymLogError as exc:
        return Check(
            "findsym version",
            True,
            f"could not detect ({exc}) -- will assume pre-6.x and use positional-format input",
        )

    if version[0] >= 6:
        return Check(
            "findsym version",
            True,
            f"{'.'.join(str(v) for v in version)} -- keyword-format input/log (validated against 7.1.6)",
        )
    return Check(
        "findsym version",
        True,
        f"{'.'.join(str(v) for v in version)} -- positional-format input/log (validated against 2.5, "
        "the HPC/legacy-fixture version)",
    )


def _check_iso() -> Check:
    """`iso` builds every order-parameter direction (the ISOTROPY-matrix basis
    of `isotropy_basis`), so `irrep --opd`, `freeze` and `decomposition` need
    it. Checked by RUNNING it, not only by finding it: a wrapper (e.g. the
    macOS Docker one) can be on PATH and still fail, and iso's own banner
    names the version."""
    if isotropy.find_binary("iso") is None:
        return Check(
            "iso",
            False,
            "not found on PATH -- install the ISOTROPY suite; `freeze`, `decomposition` and "
            "`irrep --opd` need it to build order-parameter directions",
        )
    from . import iso as iso_module

    try:
        output = iso_module.run_iso("quit\n")
    except Exception as exc:  # noqa: BLE001 -- any failure to run is the finding
        return Check("iso", False, f"found on PATH but did not run ({type(exc).__name__}: {exc})")
    for line in output.splitlines():
        if line.strip().startswith("Isotropy, Version"):
            return Check("iso", True, f"found and runs ({line.strip()})")
    return Check("iso", False, "found on PATH but its output has no 'Isotropy, Version' banner")


def _check_isodata(cli_value: str | None) -> Check:
    resolution = isotropy.resolve_isodata(cli_value)

    if resolution.value is None:
        return Check(
            "ISODATA",
            False,
            "not set (checked --isodata flag, project config, user config, ISODATA env var) "
            "-- findsym will fail unless it has its own default data path",
        )

    if not resolution.value.endswith("/"):
        return Check(
            "ISODATA",
            False,
            f"resolved to {resolution.value!r} via {resolution.source}, but does not end in a "
            "trailing slash -- findsym concatenates this directly onto data filenames and will "
            "fail with a cryptic error",
        )

    if not Path(resolution.value).is_dir():
        return Check(
            "ISODATA",
            False,
            f"resolved to {resolution.value!r} via {resolution.source}, but that directory does "
            "not exist",
        )

    return Check("ISODATA", True, f"resolved to {resolution.value!r} via {resolution.source}")


def run(isodata: str | None = None) -> int:
    """Run all environment checks, print a report, and return a process exit code."""
    checks = [
        _check_numpy(),
        _check_binary("findsym"),
        _check_findsym_version(),
        _check_binary("smodes"),
        _check_iso(),
        _check_isodata(isodata),
    ]

    width = max(len(c.name) for c in checks)
    for check in checks:
        status = "OK  " if check.ok else "FAIL"
        print(f"[{status}] {check.name:<{width}}  {check.detail}", file=sys.stdout)

    if all(c.ok for c in checks):
        print("\nAll checks passed.")
        return 0

    print("\nSome checks failed -- see above.")
    return 1

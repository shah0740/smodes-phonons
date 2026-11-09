"""Suite-wide: the `iso` runner for the v2.0 ISOTROPY-matrix OPD basis.

Every `iso` output the suite needs is a saved REAL one in
fixtures/isotropy_basis/iso_cache (named by `isotropy_basis.script_key`), so
the suite stays subprocess-free. A missing one is produced once by the real
`iso` binary and RECORDED there -- commit it with the test that needed it.
"""

from pathlib import Path

from smodes_phonons import isotropy, isotropy_basis

# Resolved ONCE, before any test stubs `isotropy.find_binary` with a fake
# findsym/smodes (the CLI tests do), so a recording run reaches the real iso.
_ISO_BINARY = isotropy.find_binary("iso")
_CACHE = Path(__file__).resolve().parent / "fixtures" / "isotropy_basis" / "iso_cache"


def _saved_or_recorded(script: str) -> str:
    path = _CACHE / f"{isotropy_basis.script_key(script)}.txt"
    if path.is_file():
        return path.read_text()
    from smodes_phonons import iso

    if _ISO_BINARY is None:
        raise FileNotFoundError(f"no saved iso output {path.name} and no iso binary to record it")
    out = iso.run_iso(script + "quit\n" * 3, binary=_ISO_BINARY)
    if "Isotropy, Version" in out:            # never record anything but real iso output
        _CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(out)
    return out


isotropy_basis.set_iso_runner(_saved_or_recorded)

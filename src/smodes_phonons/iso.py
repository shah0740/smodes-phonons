"""Driving the ISOTROPY suite's `iso` program (invariant/subgroup theory) --
specifically its `SHOW SUBGROUP` / `SHOW DIRECTION VECTOR` report: given a
parent space group and an observed (lower-symmetry) subgroup, which
irrep(s) and which order-parameter direction(s) can produce that exact
subgroup from that exact parent.

This is a genuinely different `iso` feature from `landau.parse_invariants`
(that one parses `iso`'s own `DISPLAY INVARIANTS` text, pasted in by hand,
for Landau free-energy fitting) -- this module drives `iso` itself as a
subprocess and parses a different report format, the same
parser/subprocess separation this project uses everywhere else
(`findsym.py`, `smodes.py`).

Confirmed directly against the real `iso` binary (not assumed from
documentation): commands are typed WITHOUT the leading `*` shown in an
interactive session -- that `*` is `iso`'s own prompt character, echoed
back, not part of the command syntax. Real example, parent 221 (Pm-3m),
subgroup 99 (P4mm):

    v par 221
    v sub 99
    sh irrep
    sh sub
    sh dir vec
    d iso
    quit

    Irrep (ML) Subgroup Dir
    GM4-       99 P4mm  P1  (a,0,0)

The `P1`/`P2`/`P3`/... token is NOT a crystallographic point group --
confirmed by comparing three known cases (parent 221, subgroups 99/38/160,
matching the same triple the developer notes' direction-vector work already
cites from earlier manual `iso` validation): GM4-'s (a,0,0)/(a,a,0)/(a,a,a)
directions come back P1/P2/P3 respectively, and a 6-component M5- order
parameter comes back P6. It's just the sequential index of that direction
among however many the irrep has -- `iso`'s own internal numbering, not an
independently meaningful quantity.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass

from . import isotropy


class IsoOutputError(ValueError):
    """Raised when `iso`'s subgroup/direction-vector output can't be parsed."""


def build_iso_input(parent_space_group: int, subgroup_space_group: int) -> str:
    """Render the command script `run_iso` needs to report every
    (irrep, direction) pair taking `parent_space_group` down to
    `subgroup_space_group` -- exactly the `iso` commands validated above,
    minus the `*` prompt character (that's `iso`'s own echo, never part of
    the command text itself)."""
    return (
        f"v par {parent_space_group}\n"
        f"v sub {subgroup_space_group}\n"
        "sh irrep\n"
        "sh sub\n"
        "sh dir vec\n"
        "d iso\n"
        "quit\n"
    )


def run_iso(input_text: str, binary: str | None = None) -> str:
    """Run the `iso` binary with `input_text` on stdin, return its stdout.

    Unlike the interactive `iso` shim some setups install (which requests a
    TTY), a plain non-interactive stdin pipe works fine for scripted
    command sequences like `build_iso_input`'s -- confirmed directly
    against the real binary, not assumed.
    """
    binary_path = binary or isotropy.find_binary("iso")
    if binary_path is None:
        raise FileNotFoundError("iso binary not found on PATH -- run `smodes-phonons doctor`")
    # RUN IN A TEMP DIRECTORY. `iso` writes its own `iso.log` into whatever
    # directory it is invoked from, exactly as `findsym` writes `findsym.log`
    # -- and the developer notes' architecture rule already requires a temp `cwd` for
    # that reason. This function did not, and it was caught the hard way: a
    # `--map` run (a flag removed in v2.0) from the repository root modified an
    # `iso.log` that had itself been committed by accident the same way (a
    # sweep run from the root, in 632b7f3). Nothing reads the log -- stdout
    # carries the answer -- so it is simply discarded with the directory, and
    # the stray committed copy was deleted in v1.44.0. `.gitignore` now covers
    # both `iso.log` and `findsym.log` so a manual run in the root cannot
    # recreate the problem.
    with tempfile.TemporaryDirectory() as workdir:
        try:
            result = subprocess.run(
                [binary_path], input=input_text, capture_output=True, text=True,
                check=False, cwd=workdir,
                timeout=isotropy.ISOTROPY_SUBPROCESS_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            # Same hole as run_findsym/run_smodes had (known issue 6 in the developer notes), and
            # iso can genuinely wedge: it PAGINATES, so a script that runs out
            # of input leaves it waiting on stdin rather than exiting.
            raise isotropy.IsotropyTimeoutError(
                f"iso did not finish within {isotropy.ISOTROPY_SUBPROCESS_TIMEOUT_SECONDS}s and "
                "was killed. A likely cause is iso waiting for more input than the command "
                "script supplies -- it paginates long tables. Under the macOS Docker wrapper the "
                "container survives the client; check `docker ps` and `docker kill` it."
            ) from exc
    if result.returncode != 0:
        raise RuntimeError(f"iso exited with status {result.returncode}: {result.stderr}")
    return result.stdout


@dataclass
class IsoSubgroupMatch:
    irrep: str
    subgroup_number: int
    subgroup_symbol: str
    direction_index: str
    """iso's own sequential direction-index label for this irrep, e.g. 'P1',
    'P2' -- bookkeeping only, not a crystallographic point group (see
    module docstring)."""
    direction: str
    """Raw direction-vector text as iso printed it, e.g. '(a,0,0)' or, for a
    higher-dimensional order parameter, '(a,a;a,a;a,a)' -- kept as text
    rather than parsed into a tuple, since the multi-component syntax isn't
    the same shape as this project's own `--direction` spec."""


_ROW_RE = re.compile(
    r"^(\S+)\s+(\d+)\s+(\S+)\s+(P\d+)\s+(\(.+\))\s*$"
)


def parse_iso_subgroup_output(text: str) -> list[IsoSubgroupMatch]:
    """Parse the `Irrep (ML) Subgroup Dir` table `run_iso` (with
    `build_iso_input`'s own command script) prints.

    Returns an empty list if the table has a header but no data rows (a
    genuinely valid, if perhaps surprising, outcome -- not every
    parent/subgroup pair necessarily has *iso*-recognized irrep/direction
    matches). Raises IsoOutputError only if the expected header itself
    never appears at all, which means the command script didn't run as
    expected (e.g. an invalid space group number).
    """
    lines = text.splitlines()
    header_index = None
    for i, line in enumerate(lines):
        if "Irrep" in line and "Subgroup" in line and "Dir" in line:
            header_index = i
            break
    if header_index is None:
        raise IsoOutputError(
            "no 'Irrep ... Subgroup ... Dir' table found in iso output -- check that both "
            "space group numbers are valid"
        )

    matches = []
    for line in lines[header_index + 1 :]:
        stripped = line.strip()
        if not stripped or stripped == "*":
            break
        m = _ROW_RE.match(stripped)
        if not m:
            break
        irrep, subgroup_number, subgroup_symbol, direction_index, direction = m.groups()
        matches.append(
            IsoSubgroupMatch(
                irrep=irrep,
                subgroup_number=int(subgroup_number),
                subgroup_symbol=subgroup_symbol,
                direction_index=direction_index,
                direction=direction,
            )
        )
    return matches

@dataclass
class IsoStratum:
    """One row of `iso`'s own (subgroup, direction) table for a SINGLE irrep,
    as `build_irrep_directions_input` queries it."""

    subgroup_number: int
    subgroup_symbol: str
    direction_index: str
    """`iso`'s own sequential label: `P<n>` for a ONE-parameter (primary)
    stratum, `C<n>` for a multi-parameter coupled one, `<d>D<n>` for the
    generic d-dimensional cell. Bookkeeping, not a point group."""
    direction: str
    """Raw text as `iso` printed it, e.g. `(a,1.414a,a,0)`."""

    @property
    def n_components(self) -> int:
        """Component count of the direction vector -- the irrep's dimension
        as `iso` sees it. Used to GATE a comparison: `iso` is queried by
        space-group number and irrep LABEL, and labels depend on the Wyckoff
        setting (the developer notes), so `iso`'s `P1` need not be our `P1`. If this
        disagrees with our own `irrep.degeneracy` the two are different
        irreps and no mapping between them is meaningful."""
        inner = self.direction.strip("()")
        return len([t for t in inner.replace(";", ",").split(",")])

    @property
    def is_primary(self) -> bool:
        """True for a ONE-parameter stratum (`P<n>`), the only kind our
        `--direction` grammar can express. A `C<n>` row puts a LINEAR
        COMBINATION of two free order parameters in one position -- real
        TiO2 P1's `(a+b,1.414b,a-b,1.414a)` -- which the developer notes records as
        deliberately out of scope: each position would have to be an
        expression over several parameters, not a coefficient."""
        return bool(re.match(r"^P\d+$", self.direction_index))

    def as_direction_spec(self) -> str:
        """`iso`'s direction as a string our own `--direction` accepts.

        Only the parentheses come off: `iso` writes coefficients exactly the
        way Issue 10's grammar reads them (`1.414a`, `0.707a`), and its
        semicolons between star arms are the convention our grammar already
        follows. That is not a coincidence -- the grammar was specified
        against this output."""
        return self.direction.strip("()").replace(" ", "")


def build_irrep_directions_input(parent_space_group: int, irrep_label: str) -> str:
    """Command script asking `iso` for every (subgroup, direction) pair ONE
    irrep of `parent_space_group` can produce -- its complete stratum table.

    Distinct from `build_iso_input`, which asks the inverse question (which
    irreps reach a GIVEN subgroup) and prints an extra `Irrep` column.

    **Driving `iso` for THIS query is safe, unlike the cross-check that was
    removed from `decomposition`.** That one looked up (parent -> subgroup),
    where the setting and the order of interpretation matter. This is a
    self-contained lookup with a fixed script, and it was verified to run
    non-interactively and deterministically -- real parent 141 irrep P1 and
    real parent 139 irrep P5 both returned in seconds, the latter matching
    the table a physicist had previously supplied by hand, row for row.
    """
    # FULL WORDS, not iso's abbreviations. Both forms work -- verified
    # identical output for real parent 141 irrep P1 and parent 139 irrep P5 --
    # and the long form is self-documenting, so a user reading the script (or
    # a saved fixture of its output) can see what was asked without knowing
    # iso's command language. `v par`/`sh dir vec` mean the same thing and
    # read like noise.
    return (
        f"value parent {parent_space_group}\n"
        f"value irrep {irrep_label.lower()}\n"
        "show subgroup\n"
        "show direction vector\n"
        "display isotropy\n"
        # SEVERAL quits, not one. `iso` paginates a long table and asks for
        # another keypress, so a single `quit` runs out of input and the
        # Fortran runtime dies with "End of file" -- which is what a
        # degeneracy-6 irrep (real graphite L1 at parent 194, 8 primary
        # strata) does where degeneracy-4 ones got away with one. Extra
        # quits are inert when they are not needed: verified that real
        # parent 141 irrep P1 and parent 139 irrep P5 return identical
        # output with one quit and with four.
        + "quit\n" * 4
    )


_STRATUM_ROW_RE = re.compile(r"^(\d+)\s+(\S+)\s+(\S+)\s+(\(.+\))\s*$")


def parse_irrep_directions_output(text: str) -> list[IsoStratum]:
    """Parse the `Subgroup  Dir` table `build_irrep_directions_input` asks
    for. Note there is NO `Irrep` column -- the irrep is fixed by the query,
    which is exactly why `parse_iso_subgroup_output`'s regex cannot be
    reused.

    Raises IsoOutputError when the header never appears (an invalid space
    group or irrep label), and returns an empty list for a header with no
    rows.
    """
    lines = text.splitlines()
    header = None
    for i, line in enumerate(lines):
        if "Subgroup" in line and "Dir" in line:
            header = i
            break
    if header is None:
        raise IsoOutputError(
            "no 'Subgroup ... Dir' table in iso output -- check the parent space group "
            "number and that the irrep label exists in iso's own setting for it"
        )
    out = []
    for line in lines[header + 1 :]:
        stripped = line.strip()
        if not stripped or stripped == "*":
            break
        m = _STRATUM_ROW_RE.match(stripped)
        if not m:
            continue
        number, symbol, index, direction = m.groups()
        out.append(IsoStratum(int(number), symbol, index, direction))
    return out

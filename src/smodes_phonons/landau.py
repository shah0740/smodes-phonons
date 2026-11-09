"""Fitting a Landau free-energy expansion to DFT total energies, given the
invariant-polynomial FORM as text -- e.g. copy-pasted directly from the real
ISOTROPY/ISODISTORT INVARIANTS output (see `parse_invariants`'s own
docstring for the exact expected format; validated against two real
examples for Ca3Mn2O7 (space group 139, I4/mmm; irreps GM5-, X2+, X3-),
differing only in which order-parameter *direction* was chosen for the
degenerate X-point irreps -- the resulting invariant polynomial form itself
is identical in both, as expected: the set of allowed terms depends on the
irreps and how many components of each are active, not the specific
direction chosen for a single active component).

This project doesn't run/wrap `iso` itself (the ISOTROPY suite's own
invariant-theory program) -- this module only ever needs its TEXT OUTPUT,
never the program itself, matching this project's usual parser/subprocess
separation rule (`findsym.py`/`smodes.py` are the precedent: the parser
takes text, never a subprocess). The text can come from running `iso`
directly or from ISODISTORT's web-based invariants tool; either way, the
format parsed here is the same.

Fitting is plain linear least squares, not anything requiring a new
dependency: every term in a Landau expansion is a FIXED monomial in the
order parameters (e.g. a^2, a*b*c, a^2*b^2), so the energy is linear IN THE
COEFFICIENTS even though it is nonlinear in the order parameters
themselves. Given (a, b, c, ...) amplitude combinations and their DFT total
energies (e.g. from `freeze-amplitude`'s own generated structures),
finding the coefficients is `numpy.linalg.lstsq` against a design matrix of
monomial values -- no scipy needed, matching this project's numpy-only
dependency policy.

`predict_equilibrium` (finding the order-parameter values that minimize
the fitted energy) similarly avoids a new dependency: the fitted
polynomial's exact gradient is cheap and exact to compute analytically
(each term is a monomial, so its own gradient is a one-line product rule),
so a self-contained multi-start gradient descent with backtracking line
search is implemented directly rather than reaching for scipy.optimize.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np


class LandauError(ValueError):
    """Raised for malformed invariants text or fitting input."""


@dataclass
class IrrepOrderParameter:
    """One `Irrep N: ...` block from the INVARIANTS header -- purely
    informational (cross-reference between an order-parameter symbol like
    "a" and the irrep it belongs to); not needed for fitting itself, which
    only uses the "Invariant Polynomials" section."""

    index: int
    irrep_label: str
    variable: str | None


@dataclass
class LandauInvariants:
    """The parsed form of a Landau expansion: which order-parameter symbols
    exist (`variables`, e.g. `['a', 'b', 'c']`) and which monomials in them
    are symmetry-allowed (`terms`, e.g. `(2,0,0)` for `a^2`, `(1,1,1)` for
    `a*b*c`), each exponent tuple aligned to `variables`' own order.
    `irreps`, if the header block was present and parseable, cross-references
    each order-parameter symbol back to its irrep for readability."""

    variables: list[str]
    terms: list[tuple]
    irreps: list = field(default_factory=list)
    space_group_number: int | None = None
    space_group_symbol: str | None = None

    def term_label(self, term: tuple) -> str:
        if not any(term):
            return "1"
        return " ".join(
            (f"{v}^{e}" if e != 1 else v) for v, e in zip(self.variables, term) if e
        )


_SPACE_GROUP_RE = re.compile(r"^Space Group:\s*(\d+)\s+(\S+)")
_IRREP_HEADER_RE = re.compile(r"^Irrep\s+(\d+):")
_IRREDUCIBLE_REP_RE = re.compile(r"^Irreducible Representation:\s*([^\s,]+)")
_ORDER_PARAMETER_RE = re.compile(r"Order Parameter:\s*\S+\s*\(([^)]*)\)")
_DEGREE_HEADER_RE = re.compile(r"^Degree\s+(\d+)\s*$")
_TERM_TOKEN_RE = re.compile(r"^([a-zA-Z]+)(\d*)$")
_BARE_VARIABLE_RE = re.compile(r"^[a-z]\d*$")


def parse_invariants(text: str) -> LandauInvariants:
    """Parse real ISOTROPY/ISODISTORT INVARIANTS output, e.g.:

        Space Group: 139 I4/mmm D4h-17
        Irrep 1: Point 1 in first Brillouin zone: GM, k14 (0,0,0)
        Irreducible Representation: GM5-, k14t10
        Order Parameter: P1(1) (a,0) 44 Imm2 C2v-20 , basis=..., origin=...
        ...
        Invariant Polynomials
        Degree 2

        a2

        b2
        ...
        Degree 3

        a b c
        ...

    Each term line under a `Degree N` header is one invariant monomial:
    whitespace-separated tokens, each `<variable><exponent>` (exponent
    omitted means 1, e.g. the degree-3 trilinear term is written `a b c`,
    not `a1 b1 c1`). Cross-checks each parsed term's own exponents sum to
    the stated degree, raising `LandauError` immediately (not silently
    continuing) if a term doesn't match its own header -- this is a strong
    correctness signal for the token parser itself, not just the input.

    The header's `Order Parameter: ... (a,0)` / `(b;0)` / `(b;b)` line
    varies in punctuation depending on which order-parameter direction was
    chosen (confirmed across two real examples for the same material) --
    only the first token is used, and only when it looks like a bare
    variable symbol (a single lowercase letter, optionally with a trailing
    digit); this is best-effort and purely for `irreps`' own cross-reference
    display, never required for the returned `terms`/`variables`, which
    come entirely from the "Invariant Polynomials" section.
    """
    lines = text.splitlines()

    space_group_number = None
    space_group_symbol = None
    irreps: list[IrrepOrderParameter] = []
    current_index = None
    current_irrep_label = None
    poly_start = None

    for i, line in enumerate(lines):
        stripped = line.strip()

        m = _SPACE_GROUP_RE.match(stripped)
        if m:
            space_group_number = int(m.group(1))
            space_group_symbol = m.group(2)
            continue

        m = _IRREP_HEADER_RE.match(stripped)
        if m:
            current_index = int(m.group(1))
            continue

        m = _IRREDUCIBLE_REP_RE.match(stripped)
        if m:
            current_irrep_label = m.group(1).rstrip(",")
            continue

        if stripped.startswith("Order Parameter:"):
            variable = None
            m = _ORDER_PARAMETER_RE.search(stripped)
            if m:
                first_token = re.split(r"[,;]", m.group(1))[0].strip()
                if _BARE_VARIABLE_RE.fullmatch(first_token):
                    variable = first_token
            if current_index is not None and current_irrep_label is not None:
                irreps.append(IrrepOrderParameter(current_index, current_irrep_label, variable))
            current_index = None
            current_irrep_label = None
            continue

        if stripped == "Invariant Polynomials":
            poly_start = i + 1
            break

    if poly_start is None:
        raise LandauError("no 'Invariant Polynomials' section found in the given text")

    raw_terms: list[tuple[int, dict]] = []
    current_degree = None
    for line in lines[poly_start:]:
        stripped = line.strip()
        if not stripped:
            continue
        m = _DEGREE_HEADER_RE.match(stripped)
        if m:
            current_degree = int(m.group(1))
            continue
        if current_degree is None:
            continue

        exponents: dict = {}
        for token in stripped.split():
            tm = _TERM_TOKEN_RE.match(token)
            if not tm:
                raise LandauError(f"could not parse invariant term token {token!r} in line {stripped!r}")
            var, exp_str = tm.group(1), tm.group(2)
            exponents[var] = exponents.get(var, 0) + (int(exp_str) if exp_str else 1)

        total_degree = sum(exponents.values())
        if total_degree != current_degree:
            raise LandauError(
                f"term {stripped!r} has total degree {total_degree}, expected {current_degree} "
                f"(under the current 'Degree {current_degree}' header) -- check the pasted text"
            )
        raw_terms.append((current_degree, exponents))

    if not raw_terms:
        raise LandauError("no invariant polynomial terms found after 'Invariant Polynomials'")

    variables = sorted({v for _, exps in raw_terms for v in exps})
    terms = [tuple(exps.get(v, 0) for v in variables) for _, exps in raw_terms]

    return LandauInvariants(
        variables=variables,
        terms=terms,
        irreps=irreps,
        space_group_number=space_group_number,
        space_group_symbol=space_group_symbol,
    )


def parse_inline_terms(text: str) -> LandauInvariants:
    """Lighter-weight alternative to `parse_invariants` for quick, one-off
    use without a saved INVARIANTS text file: a comma- or whitespace-
    separated list of terms in the exact same per-term token syntax
    (`<variable><exponent>`, exponent omitted means 1), e.g.
    `"a2,b2,c2,abc,a4,a2b2,b4,a2c2,b2c2,c4"` or `"a2 b2 c2 abc a4"`. No
    degree-header cross-check is possible here (there's no stated degree
    to compare against, unlike `parse_invariants`), and no irrep/space-
    group provenance -- this is a strictly reduced version of
    `LandauInvariants` for when you already know the term list by heart
    (e.g. from a paper) and don't have a saved INVARIANTS file at hand.
    """
    tokens = [t for t in re.split(r"[,\s]+", text.strip()) if t]
    if not tokens:
        raise LandauError("no terms found in the given text")

    exponent_dicts = []
    for token in tokens:
        # Each token may itself be a multi-variable monomial like "a2b2"
        # (no separator between variables) -- split on variable boundaries
        # (a letter run followed by an optional digit run) rather than
        # requiring one variable per whitespace-separated piece, so both
        # "a2 b2" and "a2b2" (single combined token) parse the same way.
        exponents: dict = {}
        remaining = token
        sub_pattern = re.compile(r"([a-zA-Z])(\d*)")
        pos = 0
        for m in sub_pattern.finditer(remaining):
            if m.start() != pos:
                raise LandauError(f"could not parse term token {token!r}")
            var, exp_str = m.group(1), m.group(2)
            exponents[var] = exponents.get(var, 0) + (int(exp_str) if exp_str else 1)
            pos = m.end()
        if pos != len(remaining) or not exponents:
            raise LandauError(f"could not parse term token {token!r}")
        exponent_dicts.append(exponents)

    variables = sorted({v for exps in exponent_dicts for v in exps})
    terms = [tuple(exps.get(v, 0) for v in variables) for exps in exponent_dicts]

    return LandauInvariants(variables=variables, terms=terms)


@dataclass
class LandauFit:
    """A fitted Landau expansion: `terms[0]` is always the constant offset
    `()` (all-zero exponents, representing E(0,0,...,0) -- the reference
    structure's own total energy, which real DFT energies never have
    exactly at zero, unlike the bare symmetry-derived invariant list),
    followed by `invariants.terms` in the same order. `coefficients` is
    aligned 1:1 with `terms`."""

    variables: list[str]
    terms: list[tuple]
    coefficients: np.ndarray
    r_squared: float
    residuals: np.ndarray

    def term_label(self, term: tuple) -> str:
        if not any(term):
            return "1"
        return " ".join(
            (f"{v}^{e}" if e != 1 else v) for v, e in zip(self.variables, term) if e
        )


def _resolve_values(values, variables: list[str]) -> np.ndarray:
    if isinstance(values, dict):
        missing = [v for v in variables if v not in values]
        if missing:
            raise LandauError(f"data point is missing value(s) for {missing}")
        return np.array([float(values[v]) for v in variables], dtype=np.float64)
    resolved = np.asarray(values, dtype=np.float64)
    if resolved.shape != (len(variables),):
        raise LandauError(
            f"data point has {resolved.size} value(s), expected {len(variables)} (order: {variables})"
        )
    return resolved


def _monomial_value(term: tuple, resolved: np.ndarray) -> float:
    value = 1.0
    for exponent, v in zip(term, resolved):
        if exponent:
            value *= v**exponent
    return value


def _monomial_gradient(term: tuple, resolved: np.ndarray) -> np.ndarray:
    grad = np.zeros(len(term))
    for j, exponent in enumerate(term):
        if exponent == 0:
            continue
        g = exponent * resolved[j] ** (exponent - 1)
        for k, e2 in enumerate(term):
            if k != j and e2:
                g *= resolved[k] ** e2
        grad[j] = g
    return grad


def fit_landau_coefficients(invariants: LandauInvariants, data_points: list) -> LandauFit:
    """Fit Landau coefficients to `data_points`, a list of
    `(values, energy)` pairs -- `values` either `{variable: amplitude}` or
    a sequence in `invariants.variables`' own order, `energy` the DFT total
    energy (any consistent units) for that amplitude combination. A
    constant-offset term is added automatically (see `LandauFit`'s own
    docstring) so real DFT energies (never zero at the reference structure)
    don't need to be pre-shifted by the caller.

    Plain linear least squares (`numpy.linalg.lstsq`): every term is a
    fixed monomial, so energy is linear in the coefficients regardless of
    how nonlinear it is in the order parameters themselves. Needs at least
    as many data points as terms (including the constant), or the system
    is underdetermined; raises `LandauError` naming the shortfall rather
    than silently returning an underdetermined (infinitely many exact
    solutions, `lstsq` picks the minimum-norm one) fit.
    """
    if len(data_points) < 2:
        raise LandauError("fit_landau_coefficients needs at least 2 data points")

    terms = [tuple(0 for _ in invariants.variables)] + list(invariants.terms)
    if len(data_points) < len(terms):
        raise LandauError(
            f"fit_landau_coefficients has {len(data_points)} data point(s) but {len(terms)} "
            "term(s) (including the constant offset) -- underdetermined, add more grid points"
        )

    X = np.zeros((len(data_points), len(terms)))
    y = np.zeros(len(data_points))
    for row, (values, energy) in enumerate(data_points):
        resolved = _resolve_values(values, invariants.variables)
        for col, term in enumerate(terms):
            X[row, col] = _monomial_value(term, resolved)
        y[row] = energy

    coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    residuals = y - X @ coefficients
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0

    return LandauFit(
        variables=list(invariants.variables),
        terms=terms,
        coefficients=coefficients,
        r_squared=r_squared,
        residuals=residuals,
    )


def polynomial_energy(fit: LandauFit, values) -> float:
    resolved = _resolve_values(values, fit.variables)
    return float(sum(c * _monomial_value(t, resolved) for c, t in zip(fit.coefficients, fit.terms)))


def polynomial_gradient(fit: LandauFit, values) -> np.ndarray:
    resolved = _resolve_values(values, fit.variables)
    grad = np.zeros(len(fit.variables))
    for c, t in zip(fit.coefficients, fit.terms):
        grad += c * _monomial_gradient(t, resolved)
    return grad


def _gradient_descent(fit: LandauFit, start: np.ndarray, max_iterations: int) -> np.ndarray:
    point = start.copy()
    step = 0.1
    for _ in range(max_iterations):
        grad = polynomial_gradient(fit, point)
        grad_norm = float(np.linalg.norm(grad))
        if grad_norm < 1e-10:
            break
        direction = grad / grad_norm
        current_energy = polynomial_energy(fit, point)

        trial_step = step
        improved = False
        for _ in range(30):
            trial_point = point - trial_step * direction
            if polynomial_energy(fit, trial_point) < current_energy:
                improved = True
                break
            trial_step *= 0.5
        if not improved:
            break

        point = trial_point
        step = trial_step * 1.5
    return point


def predict_equilibrium(fit: LandauFit, bound: float = 1.0, n_starts: int = 25, max_iterations: int = 500, seed: int = 0) -> dict:
    """Find the order-parameter values minimizing the fitted Landau energy
    (the constant offset doesn't affect where the minimum is, so it's
    ignored here), via multi-start gradient descent with backtracking line
    search -- self-contained (no scipy dependency): the fitted polynomial's
    exact gradient is cheap and exact (each term is a monomial, so its own
    gradient is a one-line product-rule expression, see
    `polynomial_gradient`).

    `bound`: starting points are the origin (always tried first -- "no
    distortion" is always a real candidate equilibrium) plus `n_starts - 1`
    points drawn uniformly from `[-bound, bound]^n`. This is NOT a hard
    constraint on the returned result -- a real minimum can lie outside the
    starting range, gradient descent is free to walk there.

    `n_starts`: guards against a genuinely multi-well landscape (e.g. the
    classic anisotropic-quartic case where the true minimum sits along one
    of several symmetry-equivalent directions, not at the origin) where
    descending from a single start could land on the wrong local minimum;
    the lowest-energy point found across all starts is returned.
    """
    n = len(fit.variables)
    rng = np.random.default_rng(seed)
    starts = [np.zeros(n)] + [rng.uniform(-bound, bound, size=n) for _ in range(max(0, n_starts - 1))]

    best_point = None
    best_energy = None
    for start in starts:
        point = _gradient_descent(fit, start, max_iterations)
        energy = polynomial_energy(fit, point)
        if best_energy is None or energy < best_energy - 1e-12:
            best_energy = energy
            best_point = point

    return {v: float(x) for v, x in zip(fit.variables, best_point)}


def parse_data_table(text: str, variables: list[str]) -> list:
    """Parse a simple tab- or comma-delimited data table: a header row
    naming columns (must include every entry in `variables` plus
    'energy'), then one data row per line. Column order doesn't need to
    match `variables`' own order -- resolved by name. Blank lines and
    lines starting with '#' are skipped (comments)."""
    rows = [line for line in text.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not rows:
        raise LandauError("data table is empty")

    delimiter = "\t" if "\t" in rows[0] else ","
    header = [h.strip() for h in rows[0].split(delimiter)]
    missing = [v for v in [*variables, "energy"] if v not in header]
    if missing:
        raise LandauError(f"data table is missing column(s) {missing} (found: {header})")

    data_points = []
    for line_number, row in enumerate(rows[1:], start=2):
        cells = [c.strip() for c in row.split(delimiter)]
        if len(cells) != len(header):
            raise LandauError(f"data table line {line_number} has {len(cells)} column(s), expected {len(header)}")
        try:
            record = {name: float(cell) for name, cell in zip(header, cells)}
        except ValueError as exc:
            raise LandauError(f"data table line {line_number}: {exc}") from exc
        energy = record.pop("energy")
        data_points.append(({v: record[v] for v in variables}, energy))

    return data_points

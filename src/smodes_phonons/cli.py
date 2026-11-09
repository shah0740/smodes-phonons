"""Console entry point for smodes-phonons.

Stages (symmetry, smodes, displace, ...) are added here as individually
callable subcommands, kept exposed permanently for debugging even once
higher-level orchestration commands (e.g. `irrep X5+`) exist.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import contextlib
import io
import os
import itertools
import re
import sys
from pathlib import Path

import numpy as np

from . import __version__, config, displace, doctor, isotropy
from . import decomposition as decomposition_module
from . import findsym as findsym_module
from . import freeze as freeze_module
from . import iso as iso_module
from . import isotropy_basis as isotropy_basis_module
from . import landau as landau_module
from . import lo_to as lo_to_module
from . import opd as opd_module
from . import phonons as phonons_module
from . import smodes as smodes_module
from . import supercell as supercell_module
from . import vasp as vasp_module
from .structure import PoscarFormatError, Structure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smodes-phonons",
        description="Symmetry-adapted frozen-phonon calculations driven by the ISOTROPY suite.",
    )
    parser.add_argument("--version", action="version", version=f"smodes-phonons {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor_parser = subparsers.add_parser(
        "doctor", help="Check that the environment (numpy, findsym, smodes, ISODATA) is set up correctly."
    )
    doctor_parser.add_argument(
        "--isodata",
        default=None,
        help="Path to the ISOTROPY data directory (must end in a trailing slash). "
        "Overrides project config, user config, and the ISODATA environment variable.",
    )

    smodes_parser = subparsers.add_parser(
        "smodes", help="Parse an smodes output file and print a summary of its irreps."
    )
    smodes_parser.add_argument("output_file", help="Path to a file containing smodes stdout.")

    symmetry_parser = subparsers.add_parser(
        "symmetry", help="Build findsym input from a POSCAR, run findsym, and save the log."
    )
    symmetry_parser.add_argument("poscar", help="Path to a POSCAR file (VASP5 format).")
    symmetry_parser.add_argument(
        "--name",
        default=None,
        help="Material tag driving the output filename ('<name>_findsym'). "
        "Overrides the project config; falls back to an interactive prompt "
        "(saved for next time) only if stdin is a terminal.",
    )
    symmetry_parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="findsym's position-matching tolerance ('accuracy of data'). Default 0 "
        "(exact match), matching legacy.",
    )
    symmetry_parser.add_argument(
        "--isodata",
        default=None,
        help="Path to the ISOTROPY data directory (must end in a trailing slash). "
        "Overrides project config, user config, and the ISODATA environment variable.",
    )

    run_smodes_parser = subparsers.add_parser(
        "run-smodes", help="Build smodes input from a findsym log, run smodes, and save the output."
    )
    run_smodes_parser.add_argument("findsym_log", help="Path to a findsym log file (e.g. from `symmetry`).")
    run_smodes_parser.add_argument("k_vector", help="k-vector label to request (e.g. 'GM', 'X', 'R', 'M').")
    run_smodes_parser.add_argument(
        "--poscar",
        default="POSCAR",
        help="Path to the POSCAR the findsym log was built from (needed to recover species "
        "names for Wyckoff orbits). Default: POSCAR in the current directory.",
    )
    run_smodes_parser.add_argument(
        "--name",
        default=None,
        help="Material tag driving the output filename ('<name>_smodes_<k_vector>'). "
        "Overrides the project config and the POSCAR-title fallback.",
    )

    irrep_parser = subparsers.add_parser(
        "irrep",
        help="Create <K>/Irrep_<label>/Mode_i/POSCAR displacements for one or more irreps, "
        "auto-creating the findsym log and smodes output as needed. A bare k-point label "
        "(letters only, no digit -- e.g. 'GM', 'X') expands to every irrep found at that "
        "k-point, so 'irrep GM' computes every Gamma-point phonon in one invocation.",
    )
    irrep_parser.add_argument(
        "irreps",
        nargs="+",
        help="Irrep labels, e.g. 'GM5+' 'X2-' (case-insensitive). A bare k-point label with no "
        "digit (e.g. 'GM', 'X', 'M', 'R') is a catch-all: it expands to every irrep smodes "
        "finds at that k-point, so 'irrep GM' alone computes every Gamma-point phonon mode. "
        "Mixing both is fine, e.g. 'irrep GM X2+' -- every irrep at Gamma, plus X2+ "
        "specifically. Duplicates (e.g. an explicit label also reachable via a bare k-point "
        "expansion) are processed once.",
    )
    irrep_parser.add_argument(
        "--poscar", default="POSCAR", help="Path to a POSCAR file. Default: POSCAR in the current directory."
    )
    irrep_parser.add_argument(
        "--name",
        default=None,
        help="Material tag driving generated filenames. Overrides the project config and "
        "the POSCAR-title fallback.",
    )
    irrep_parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="findsym's position-matching tolerance, used only if the findsym log doesn't "
        "already exist. Default 0 (exact match), matching legacy.",
    )
    irrep_parser.add_argument(
        "--isodata",
        default=None,
        help="Path to the ISOTROPY data directory (must end in a trailing slash), used only "
        "if the findsym log doesn't already exist.",
    )
    irrep_parser.add_argument(
        "--kpoints",
        default="KPOINTS",
        help="Path to a KPOINTS file to scale per irrep's supercell shape. Optional -- "
        "prints a warning and skips KPOINTS generation if the file doesn't exist. "
        "Default: KPOINTS in the current directory.",
    )
    irrep_parser.add_argument(
        "--incar",
        default="INCAR",
        help="Path to a user-supplied INCAR template (real ENCUT/GGA/POTCAR-matched settings, "
        "magnetic moments, etc.) written into every Mode_i/ and Undistorted/ with NSW/IBRION/ISYM "
        "forced to a static single-point force calculation (NSW=0, IBRION=-1, ISYM=0), every other tag "
        "unchanged. Optional -- prints a warning and skips INCAR generation if the file doesn't "
        "exist. Default: INCAR in the current directory. Independent of --kpoints. Also used, "
        "with LEPSILON=.TRUE./IBRION=-1 forced instead, for BEC/INCAR when --bec is given "
        "and this command creates a BEC/ directory (see the `bec` command).",
    )
    irrep_parser.add_argument(
        "--opd",
        action="store_true",
        help="Also write the order-parameter-direction (OPD) report, "
        "<K>/Irrep_<label>/opd.txt -- one per irrep, each on that irrep's OWN native cell "
        "(enlarged only as far as hosting its own directions requires). A bare k-point gets "
        "one per irrep there, identical to what naming that irrep directly would give. It "
        "lists every order-parameter direction of each degenerate mode -- ISOTROPY's components, "
        "in ISOTROPY's order, the positions `freeze --direction` takes and `decomposition` "
        "reports -- with its star arm, how it relates to the others (arm, within-arm, Bloch "
        "quadrature), each arm's Q-vector, and its full displacement pattern. Built from "
        "ISOTROPY's irrep matrices (no DFT): the first time an irrep is used, the `iso` program "
        "is run and its text output saved under ~/.cache/smodes_phonons/iso (or "
        "$SMODES_ISO_CACHE), so later runs reuse it. Purely informational -- it changes no generated "
        "structure, so the displaced POSCARs are byte-identical with or without it.",
    )
    irrep_parser.add_argument(
        "--bec",
        action="store_true",
        help="For a Gamma-point, IR-active irrep, also write BEC/ (LEPSILON=.TRUE. input for "
        "the Born-effective-charge/dielectric-tensor calculation LO-TO splitting needs) and "
        "report whether that specific mode is symmetry-allowed to show LO-TO splitting. Off by "
        "default: whether LO-TO splitting is physically meaningful at all depends on the "
        "material being an insulator or semiconductor (free carriers in a metal screen out the "
        "macroscopic field it depends on, and Born charges/LEPSILON aren't well-defined for a "
        "metallic ground state) -- something this tool cannot determine from a bare POSCAR, so "
        "it is never assumed. Without --bec, an IR-active Gamma-point irrep just prints a note "
        "suggesting this flag rather than silently running the extra calculation.",
    )
    irrep_parser.add_argument(
        "--magmom",
        action="store_true",
        help="Correctly tile the MAGMOM tag in --incar (a per-atom magnetic-moment list, given in "
        "the parent POSCAR's own atom order) onto each generated structure's own, generally "
        "different, atom count/order -- without this flag, MAGMOM passes through unmodified from "
        "--incar into every Mode_i/Undistorted/BEC INCAR, which is wrong for anything but a "
        "non-magnetic material's Gamma-point-only, untiled case. Off by default: an explicit "
        "opt-in, matching --bec, since this changes real physics settings rather than mechanical "
        "boilerplate like NSW/IBRION/ISYM. Errors out (writing nothing) if --incar's own MAGMOM "
        "doesn't have exactly as many values as the parent POSCAR has atoms, rather than guessing.",
    )

    bec_parser = subparsers.add_parser(
        "bec",
        help="Discover every IR-active Gamma-point irrep (auto-creating the findsym log/Gamma "
        "smodes output if needed) and set up <K>/Irrep_<label>/ for each -- a complete LO-TO/"
        "epsilon(0) analysis needs every one of them, not just one, since epsilon(0) sums "
        "across every IR-active irrep at Gamma. Each one's own BEC/ (LEPSILON=.TRUE. input, "
        "one per material) is created automatically as a side effect, exactly as running "
        "`irrep` directly on each label already does -- `bec` is purely that discovery step.",
    )
    bec_parser.add_argument(
        "--poscar", default="POSCAR", help="Path to a POSCAR file. Default: POSCAR in the current directory."
    )
    bec_parser.add_argument(
        "--name",
        default=None,
        help="Material tag driving generated filenames. Overrides the project config and "
        "the POSCAR-title fallback.",
    )
    bec_parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="findsym's position-matching tolerance, used only if the findsym log doesn't "
        "already exist. Default 0 (exact match), matching legacy.",
    )
    bec_parser.add_argument(
        "--isodata",
        default=None,
        help="Path to the ISOTROPY data directory (must end in a trailing slash), used only "
        "if the findsym log doesn't already exist.",
    )
    bec_parser.add_argument(
        "--kpoints",
        default="KPOINTS",
        help="Path to a KPOINTS file, scaled per irrep's supercell shape (and copied unscaled "
        "into each BEC/KPOINTS, since that calculation runs on the plain parent cell). "
        "Optional -- prints a warning and skips KPOINTS generation if the file doesn't exist. "
        "Default: KPOINTS in the current directory.",
    )
    bec_parser.add_argument(
        "--incar",
        default="INCAR",
        help="Path to a user-supplied INCAR template, written into every Mode_i//Undistorted/ "
        "(NSW=0/IBRION=-1 forced) and BEC/INCAR (LEPSILON=.TRUE./IBRION=-1 forced), every "
        "other tag unchanged. Optional -- prints a warning and skips INCAR generation if the "
        "file doesn't exist. Default: INCAR in the current directory.",
    )
    bec_parser.add_argument(
        "--magmom",
        action="store_true",
        help="Correctly tile the MAGMOM tag in --incar onto each generated structure's own atom "
        "count/order, same as irrep --magmom (bec delegates to the identical underlying logic). "
        "Off by default -- an explicit opt-in, since this changes real physics settings.",
    )

    bec_analysis_parser = subparsers.add_parser(
        "bec-analysis",
        help="Complete Born-charge-derived analysis, no irrep argument needed: reports Born "
        "effective charges (Z*) and epsilon_infinity from a real LEPSILON=.TRUE. OUTCAR, then "
        "discovers every IR-active Gamma-point irrep and runs `phonons --born-charges` for "
        "each one already computed (LO-TO splitting + that irrep's own epsilon(0) "
        "contribution), summing every available contribution into the material's own total "
        "epsilon(0). An irrep whose phonon data isn't computed yet is skipped with a note, "
        "not a hard failure -- the reported epsilon(0) is then a clearly-labeled partial sum.",
    )
    bec_analysis_parser.add_argument(
        "--born-charges",
        default=None,
        metavar="OUTCAR",
        help="Path to the LEPSILON=.TRUE. OUTCAR. Optional -- if omitted, auto-detects "
        "BEC/OUTCAR (the calculation's own output, left in place, never moved -- same "
        "convention `phonons` uses).",
    )
    bec_analysis_parser.add_argument(
        "--poscar",
        default="POSCAR",
        help="Path to a POSCAR file -- used to label Z* by species (cosmetic; falls back to "
        "bare atom-index labels if missing/mismatched) and to discover IR-active Gamma-point "
        "irreps. Default: POSCAR in the current directory.",
    )
    bec_analysis_parser.add_argument(
        "--name",
        default=None,
        help="Material tag driving generated filenames. Overrides the project config and "
        "the POSCAR-title fallback.",
    )
    bec_analysis_parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="findsym's position-matching tolerance, used only if the findsym log doesn't "
        "already exist (needed to discover IR-active irreps). Default 0 (exact match), "
        "matching legacy.",
    )
    bec_analysis_parser.add_argument(
        "--isodata",
        default=None,
        help="Path to the ISOTROPY data directory (must end in a trailing slash), used only "
        "if the findsym log doesn't already exist.",
    )


    phonons_parser = subparsers.add_parser(
        "phonons",
        help="Compute phonon frequencies and eigenvectors for one or more irreps from completed "
        "VASP force calculations (<K>/Irrep_<label>/Mode_i/OUTCAR[.gz], or a flat "
        "Irrep_<label>/ from before that layout existed), with Gamma-point LO-TO splitting "
        "when Born charges are available.",
    )
    phonons_parser.add_argument(
        "irreps",
        nargs="+",
        help="One or more irrep labels, e.g. 'GM5+'. A bare k-point label "
        "(letters only, e.g. 'GM', 'X', 'M') is a catch-all meaning every irrep "
        "at that k-point -- the same expansion `irrep` accepts. A label that fails "
        "(no OUTCARs yet, say) does not abort the others; the exit code is 0 only "
        "if every requested label succeeded.",
    )
    phonons_parser.add_argument(
        "--generate-opd",
        action="store_true",
        help="Also write <K>/Irrep_<label>/opd_eigenvectors.txt: every order-parameter direction of "
        "each mode, expressed in terms of the real DFT eigenvectors instead of SMODES's own "
        "basis modes. Position labels match `irrep --opd` and the `--direction` argument "
        "exactly (ISOTROPY's components): the construction is linear, so an eigenvector's OPD "
        "is the same linear combination of the basis modes' OPDs. Built from ISOTROPY's irrep "
        "matrices, running `iso` once per irrep and saving its output under "
        "~/.cache/smodes_phonons/iso (or $SMODES_ISO_CACHE). Only meaningful for a "
        "degenerate irrep. Informational: it changes no frequency or eigenvector.",
    )
    phonons_parser.add_argument(
        "--poscar", default="POSCAR", help="Path to a POSCAR file. Default: POSCAR in the current directory."
    )
    phonons_parser.add_argument(
        "--name",
        default=None,
        help="Material tag identifying generated filenames. Overrides the project config and "
        "the POSCAR-title fallback.",
    )
    phonons_parser.add_argument(
        "--born-charges",
        default=None,
        metavar="OUTCAR",
        help="Path to an OUTCAR from a real VASP LEPSILON=.TRUE. run (Born effective charges + "
        "the electronic dielectric tensor), to include Gamma-point LO-TO splitting plus a "
        "Z*/epsilon(0) report. Optional to specify explicitly -- if omitted and BEC/OUTCAR "
        "exists (the calculation's own output, left in place -- see `bec`/`irrep --bec`), it's "
        "used automatically with no flag needed; whether that BEC/ was worth creating at all "
        "was already decided at `irrep --bec`/`bec` time (LO-TO splitting only applies to an "
        "insulator/semiconductor, never a metal), so this command trusts that a populated "
        "BEC/OUTCAR means the analysis was already judged worthwhile. If BEC/ exists but "
        "BEC/OUTCAR doesn't, a clear message is printed (ordinary frequencies still reported); "
        "if neither BEC/ nor an explicit path is given, LO-TO is silently skipped for a "
        "non-Gamma or IR-inactive irrep, or noted (Gamma, IR-active) as available via "
        "`--bec`/`bec`. --born-charges given explicitly for a non-Gamma irrep is a hard error. "
        "When used, also writes BEC/report.txt.",
    )
    phonons_parser.add_argument(
        "--direction",
        action="append",
        default=[],
        metavar="X,Y,Z",
        help="An additional propagation direction (q_hat) to compute LO-TO splitting along, "
        "beyond the x/y/z axes always checked automatically when --born-charges is given. "
        "Repeatable. Directions related by a real, verified symmetry operation of the crystal "
        "(e.g. x/y/z in a cubic material) are automatically grouped and only computed once.",
    )
    phonons_parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0,
        help="findsym's position-matching tolerance, used only if --born-charges is given and "
        "the findsym log doesn't already exist (needed to find the crystal's real symmetry "
        "operations for direction-grouping). Default 0 (exact match), matching legacy.",
    )
    phonons_parser.add_argument(
        "--isodata",
        default=None,
        help="Path to the ISOTROPY data directory (must end in a trailing slash), used only if "
        "--born-charges is given and the findsym log doesn't already exist.",
    )

    freeze_parser = subparsers.add_parser(
        "freeze",
        help="Freeze one or more modes (DFT eigenvectors from <K>/Irrep_<label>/phonons.npz, or "
        "SMODES modes with --smodes-basis), each along an order-parameter direction, onto the "
        "smallest cell the distortion needs, and write the frozen POSCAR.",
    )
    freeze_parser.add_argument(
        "--smodes-basis",
        action="store_true",
        help="Freeze along SMODES's own symmetry-adapted basis mode instead of a DFT "
        "eigenvector, so no `phonons` run (and no VASP) is needed first. That mode (the pattern "
        "`irrep` writes into each Mode_i/POSCAR) is the seed ISOTROPY's components are built "
        "from, and the only basis available before a phonon calculation exists -- without it "
        "the irrep -> freeze -> decomposition loop cannot be exercised at all pre-DFT. Requires "
        "--mode when the irrep "
        "has more than one mode, since there are no frequencies to choose between them. Never "
        "applied automatically: if phonons.npz is simply missing, that is an error, because "
        "silently freezing along a different basis would be a bad surprise for someone who does "
        "have phonon data.",
    )
    freeze_parser.add_argument(
        "irreps", nargs="+",
        help="Irrep labels, e.g. 'GM4-' 'M2-'. `phonons` must already have run for each, "
             "unless --smodes-basis is given."
    )
    freeze_parser.add_argument(
        "--poscar", default="POSCAR", help="Path to a POSCAR file. Default: POSCAR in the current directory."
    )
    freeze_parser.add_argument(
        "--name",
        default=None,
        help="Material tag identifying generated filenames. Overrides the project config and "
        "the POSCAR-title fallback.",
    )
    freeze_parser.add_argument(
        "--mode",
        action="append",
        default=[],
        metavar="[LABEL=]INDEX",
        help="Choose a specific mode index for an irrep, e.g. 'X2-=1'. The 'LABEL=' can be "
        "dropped (just '1') when exactly one irrep was requested. Otherwise prompts "
        "interactively (if stdin is a terminal) when an irrep has more than one mode.",
    )
    freeze_parser.add_argument(
        "--amplitude",
        action="append",
        default=[],
        metavar="[LABEL=]VALUE",
        help="Displacement amplitude in Angstrom for an irrep's chosen mode, e.g. 'X2-=0.002'. "
        "The 'LABEL=' can be dropped (just '0.002') when exactly one irrep was requested. "
        "Default: 0.001 A (same as `irrep`'s single-mode displacements), or 0.01 A with "
        "--smodes-basis. It scales each --direction component, so the written displacement norm "
        "is amplitude x sqrt(sum of squared coefficients): 'a,0,0' at 0.05 writes 0.05 A, "
        "'a,a,0' writes 0.0707 A.",
    )
    freeze_parser.add_argument(
        "--direction",
        action="append",
        default=[],
        metavar="[LABEL=]a,b,c",
        help="Order-parameter direction for an irrep, in ISOTROPY's (and ISODISTORT's) component "
        "order: one position per component of the irrep -- e.g. 'GM4-=a,0,0', 'GM4-=a,a,0', "
        "'X2+=a,a' (both star arms), 'K5=a,0,0.577a,0'. Each position is a coefficient on the "
        "letter a (a, -a, 2a, 0.5a); '0' omits it. Coefficients are relative weights -- "
        "--amplitude sets the size. Default: ISOTROPY's first component, a,0,...,0. The "
        "'LABEL=' can be dropped when exactly one irrep was requested; with more than one, "
        "every requested irrep needs its own --direction entry. `irrep --opd` lists what each "
        "position is.",
    )
    freeze_parser.add_argument(
        "--isodata",
        default=None,
        help="Path to the ISOTROPY data directory (must end in a trailing slash), used for "
        "findsym runs (the parent's log, and the reduction check on each written structure).",
    )
    freeze_parser.add_argument(
        "--kpoints",
        default="KPOINTS",
        help="Path to a KPOINTS file to scale to the combined supercell's shape. Optional -- "
        "prints a warning and skips KPOINTS generation if the file doesn't exist. "
        "Default: KPOINTS in the current directory.",
    )
    freeze_parser.add_argument(
        "--incar",
        default="INCAR",
        help="Path to a user-supplied INCAR template, written unchanged -- freeze is meant as "
        "a relaxation seed (freeze in a candidate order parameter, then let the ions/cell relax "
        "from there), so your own ISIF/IBRION/NSW choice (e.g. ISIF=3, IBRION=2) is used as-is, "
        "not overridden. Optional -- prints a warning and skips INCAR generation if the file "
        "doesn't exist. Default: INCAR in the current directory. Independent of --kpoints.",
    )
    freeze_parser.add_argument(
        "--magmom",
        action="store_true",
        help="Correctly tile the MAGMOM tag in --incar (a per-atom magnetic-moment list, given "
        "in the parent POSCAR's own atom order) onto the combined joint supercell's own, "
        "generally different, atom count/order -- the one exception to freeze's own unchanged "
        "pass-through (everything else in --incar, including ISIF/IBRION/NSW, is still written "
        "as-is). Off by default: an explicit opt-in, matching irrep --magmom. Errors out (writing "
        "nothing) if --incar's own MAGMOM doesn't have exactly as many values as the parent "
        "POSCAR has atoms.",
    )
    freeze_amplitude_parser = subparsers.add_parser(
        "freeze-amplitude",
        help="Sweep the amplitude of 1-3 irreps' chosen modes over the same range, writing one "
        "frozen POSCAR per point: a 1-D curve for one irrep, the full Cartesian-product grid for "
        "two or three.",
    )
    freeze_amplitude_parser.add_argument(
        "irreps",
        nargs="+",
        help="1-3 irrep labels, e.g. 'GM5+' or 'GM5+ GM4-'. One irrep sweeps a plain 1-D "
        "E(Q) curve; two or three sweep the full `steps**n` Cartesian-product grid, the same "
        "(start, end, steps) range applied to every irrep's own amplitude axis. More than 3 "
        "is hard to visualize, so it isn't allowed.",
    )
    freeze_amplitude_parser.add_argument(
        "--start", type=float, required=True, help="Starting displacement amplitude, Angstrom (shared by every irrep)."
    )
    freeze_amplitude_parser.add_argument(
        "--end", type=float, required=True, help="Ending displacement amplitude, Angstrom (shared by every irrep)."
    )
    freeze_amplitude_parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Number of amplitude points per irrep, including both endpoints (e.g. --steps 5 with two irreps gives 25 POSCARs).",
    )
    freeze_amplitude_parser.add_argument(
        "--poscar", default="POSCAR", help="Path to a POSCAR file. Default: POSCAR in the current directory."
    )
    freeze_amplitude_parser.add_argument(
        "--name",
        default=None,
        help="Material tag identifying generated filenames. Overrides the project config and "
        "the POSCAR-title fallback.",
    )
    freeze_amplitude_parser.add_argument(
        "--mode",
        action="append",
        default=[],
        metavar="LABEL=INDEX",
        help="Choose a specific mode index for an irrep, e.g. 'X2-=1'. The 'LABEL=' prefix "
        "can be dropped when exactly one irrep was requested (matching `freeze`'s own "
        "convention); with two or more, a bare value is rejected as ambiguous rather than "
        "guessed at. Otherwise prompts interactively (if stdin is a terminal) when an irrep "
        "has more than one mode.",
    )
    freeze_amplitude_parser.add_argument(
        "--direction",
        action="append",
        default=[],
        metavar="LABEL=a,b,c",
        help="Order-parameter direction for an irrep, in ISOTROPY's component order (see "
        "`freeze --direction`), e.g. 'GM4-=a,0,0' or (both star arms) 'X2+=a,a'; the 'LABEL=' "
        "prefix can be dropped when exactly one irrep was requested. If given, every requested "
        "irrep needs its own entry. Coefficients are relative weights; each grid point's own "
        "swept amplitude supplies the absolute size for that irrep.",
    )
    freeze_amplitude_parser.add_argument(
        "--isodata",
        default=None,
        help="Path to the ISOTROPY data directory (must end in a trailing slash), used for "
        "findsym runs (the parent's log, and the reduction check on each written structure).",
    )
    freeze_amplitude_parser.add_argument(
        "--kpoints",
        default="KPOINTS",
        help="Path to a KPOINTS file to scale to each grid point's supercell shape. Optional -- "
        "prints a warning and skips KPOINTS generation if the file doesn't exist. "
        "Default: KPOINTS in the current directory.",
    )
    freeze_amplitude_parser.add_argument(
        "--incar",
        default="INCAR",
        help="Path to a user-supplied INCAR template, written with NSW/IBRION/ISYM forced to a "
        "static single-point calculation (NSW=0, IBRION=-1, ISYM=0) at every grid point -- each "
        "point is a single-point energy at a fixed amplitude. Optional -- prints a warning and skips INCAR "
        "generation if the file doesn't exist. Default: INCAR in the current directory. "
        "Independent of --kpoints.",
    )
    freeze_amplitude_parser.add_argument(
        "--magmom",
        action="store_true",
        help="Correctly tile the MAGMOM tag in --incar onto each grid point's own supercell atom "
        "count/order (computed once -- identical across every point). Off by default -- an "
        "explicit opt-in, matching irrep --magmom. Errors out (writing nothing) if --incar's own "
        "MAGMOM doesn't have exactly as many values as the parent POSCAR has atoms.",
    )

    decomposition_parser = subparsers.add_parser(
        "decomposition",
        help="Decompose a real distorted/relaxed structure (e.g. a CONTCAR) into symmetry-mode "
        "amplitudes at one or more k-points, plus the lattice strain relating it to a parent "
        "structure. Not a wrapper around ISODISTORT (which is web-only, no command-line binary) "
        "-- computed directly. Run the real `iso` program yourself for an independent subgroup "
        "cross-check if you want one; this command doesn't run it for you.",
    )
    decomposition_parser.add_argument("--parent", required=True, metavar="POSCAR", help="Path to the high-symmetry parent POSCAR.")
    decomposition_parser.add_argument(
        "--sub",
        required=True,
        metavar="POSCAR",
        help="Path to the distorted/relaxed structure to decompose (e.g. a real CONTCAR). Its "
        "lattice does NOT need to match the parent's exactly -- real strain is measured and "
        "reported, not required to be absent. Its atom counts must be a consistent integer "
        "multiple of the parent's own, per species (see the stoichiometry check below).",
    )
    decomposition_parser.add_argument(
        "--k_vector",
        action="append",
        nargs="+",
        required=True,
        metavar="K",
        help="One or more k-vector labels to check, e.g. 'GM', 'X', 'R'. Both "
        "'--k_vector GM --k_vector X' and '--k_vector GM X' work identically (and so does any "
        "mix of the two) -- every value from every occurrence is flattened into one list, "
        "deduplicated. Every irrep found at each requested k-point is checked, not just one you "
        "have to already know the label of.",
    )
    decomposition_parser.add_argument(
        "--name",
        default=None,
        help="Material tag driving generated filenames. Overrides the project config and the "
        "parent POSCAR-title fallback.",
    )
    decomposition_parser.add_argument(
        "--tolerance",
        type=float,
        default=0.3,
        help="Atom-matching tolerance (Angstrom) used both for aligning the distorted structure's own "
        "atom ordering and for accepting a candidate cell for it. Default 0.3 A -- "
        "above typical numerical noise, below a bond length; a genuinely large real distortion "
        "(e.g. a full ferroelectric transition, not a small frozen-phonon-scale one) may need "
        "this widened.",
    )
    decomposition_parser.add_argument(
        "--findsym-tolerance",
        type=float,
        default=0.0,
        help="findsym's position-matching tolerance, used only if a findsym log doesn't already "
        "exist -- for BOTH the parent's own findsym run and the distorted structure's (a real relaxed "
        "structure routinely needs this loosened to resolve its true, lower symmetry rather "
        "than reporting P1). Default 0 (exact match), matching legacy.",
    )
    decomposition_parser.add_argument(
        "--isodata",
        default=None,
        help="Path to the ISOTROPY data directory (must end in a trailing slash), used only if "
        "a findsym log doesn't already exist.",
    )

    landau_fit_parser = subparsers.add_parser(
        "landau-fit",
        help="Fit a Landau free-energy expansion's coefficients to DFT total energies, given "
        "the invariant-polynomial form (e.g. copy-pasted from real ISOTROPY/ISODISTORT "
        "INVARIANTS output) and a data table of order-parameter amplitudes + energies -- "
        "e.g. from freeze-amplitude's own generated structures. Optionally predicts the "
        "equilibrium (energy-minimizing) order-parameter values from the fit.",
    )
    landau_form_group = landau_fit_parser.add_mutually_exclusive_group(required=True)
    landau_form_group.add_argument(
        "--invariants",
        default=None,
        metavar="FILE",
        help="Path to a text file containing real ISOTROPY/ISODISTORT INVARIANTS output "
        "(copy-pasted as-is). Preserves irrep provenance in the report. Mutually exclusive "
        "with --terms.",
    )
    landau_form_group.add_argument(
        "--terms",
        default=None,
        metavar="TERMS",
        help="A concise, comma- or space-separated term list instead of a saved INVARIANTS "
        "file, e.g. 'a2,b2,c2,abc,a4,a2b2,b4,a2c2,b2c2,c4' (same <variable><exponent> token "
        "syntax, exponent omitted means 1). No irrep provenance. Mutually exclusive with "
        "--invariants.",
    )
    landau_fit_parser.add_argument(
        "--data",
        required=True,
        metavar="FILE",
        help="Path to a tab- or comma-delimited data table: a header row naming each order-"
        "parameter variable plus 'energy', then one row per data point. Column order doesn't "
        "need to match the invariants' own variable order.",
    )
    landau_fit_parser.add_argument(
        "--predict",
        action="store_true",
        help="Also report the order-parameter values that minimize the fitted energy "
        "(multi-start gradient descent -- see --bound/--n-starts).",
    )
    landau_fit_parser.add_argument(
        "--bound",
        type=float,
        default=1.0,
        help="Range [-bound, bound] to draw --predict's random starting points from (not a "
        "hard constraint on the result). Default 1.0.",
    )
    landau_fit_parser.add_argument(
        "--n-starts",
        type=int,
        default=25,
        help="Number of --predict starting points (guards against landing on the wrong local "
        "minimum in a multi-well landscape). Default 25.",
    )

    landau_grid_parser = subparsers.add_parser(
        "landau-grid",
        help="Generate the amplitude-grid POSCARs for a Landau analysis directly from an "
        "INVARIANTS file's own irrep assignments -- the structure-generation counterpart to "
        "landau-fit, so the same file drives both ends. Thin wrapper around "
        "freeze-amplitude: same Cartesian-product sweep, same phonons.npz dependency per "
        "irrep, but grid.txt's columns are named by the invariants' own order-parameter "
        "symbols (a, b, c, ...) instead of irrep labels, so it's a drop-in target for "
        "landau-fit --data once you add an 'energy' column.",
    )
    landau_grid_parser.add_argument(
        "--invariants",
        required=True,
        metavar="FILE",
        help="Path to a text file containing real ISOTROPY/ISODISTORT INVARIANTS output "
        "(copy-pasted as-is) -- its 'Irrep N: ...'/'Irreducible Representation: ...' blocks "
        "supply the irrep label for each order-parameter symbol. Every irrep must already "
        "have `phonons` run for it (<K>/Irrep_<label>/phonons.npz).",
    )
    landau_grid_parser.add_argument(
        "--start", type=float, required=True, help="Starting displacement amplitude, Angstrom."
    )
    landau_grid_parser.add_argument(
        "--end", type=float, required=True, help="Ending displacement amplitude, Angstrom."
    )
    landau_grid_parser.add_argument(
        "--steps",
        type=int,
        required=True,
        help="Number of amplitude points per order parameter, including both endpoints.",
    )
    landau_grid_parser.add_argument(
        "--poscar", default="POSCAR", help="Path to a POSCAR file. Default: POSCAR in the current directory."
    )
    landau_grid_parser.add_argument(
        "--name",
        default=None,
        help="Material tag identifying generated filenames. Overrides the project config and "
        "the POSCAR-title fallback.",
    )
    landau_grid_parser.add_argument(
        "--mode",
        action="append",
        default=[],
        metavar="LABEL=INDEX",
        help="Choose a specific mode index for an irrep (keyed by its own irrep label, e.g. "
        "'GM5-=1' -- see the printed 'Order parameters:' line for the label each symbol maps "
        "to). Otherwise prompts interactively (if stdin is a terminal) when an irrep has more "
        "than one mode.",
    )
    landau_grid_parser.add_argument(
        "--direction",
        action="append",
        default=[],
        metavar="LABEL=a,b,c",
        help="Order-parameter direction for an irrep, in ISOTROPY's component order (see "
        "`freeze --direction`), e.g. 'GM4-=a,0,0' -- keyed by the irrep's own LABEL (see the "
        "printed 'Order parameters:' line for the label each order-parameter symbol maps to), "
        "the same convention --mode already uses, NOT the order-parameter symbol itself. If "
        "given, every irrep needs its own entry. Coefficients are relative weights; each grid "
        "point's own swept amplitude supplies the absolute size.",
    )
    landau_grid_parser.add_argument(
        "--isodata",
        default=None,
        help="Path to the ISOTROPY data directory (must end in a trailing slash), used for "
        "findsym runs (the parent's log, and the reduction check on each written structure).",
    )
    landau_grid_parser.add_argument(
        "--kpoints",
        default="KPOINTS",
        help="Path to a KPOINTS file to scale to each grid point's supercell shape. Optional -- "
        "prints a warning and skips KPOINTS generation if the file doesn't exist. "
        "Default: KPOINTS in the current directory.",
    )
    landau_grid_parser.add_argument(
        "--incar",
        default="INCAR",
        help="Path to a user-supplied INCAR template, written with NSW/IBRION/ISYM forced to a "
        "static single-point calculation (NSW=0, IBRION=-1, ISYM=0) at every grid point -- an E(Q) "
        "sweep only means something at each requested fixed amplitude. Optional -- prints a "
        "warning and skips INCAR generation if the file doesn't exist. Default: INCAR in the "
        "current directory. Independent of --kpoints.",
    )
    landau_grid_parser.add_argument(
        "--magmom",
        action="store_true",
        help="Correctly tile the MAGMOM tag in --incar onto each grid point's own supercell atom "
        "count/order (computed once -- identical across every point). Off by default -- an "
        "explicit opt-in, matching irrep --magmom. Errors out (writing nothing) if --incar's own "
        "MAGMOM doesn't have exactly as many values as the parent POSCAR has atoms.",
    )


    return parser


def _run_smodes_parse(path: str) -> int:
    text = Path(path).read_text()
    try:
        parsed = smodes_module.parse_smodes_output(text)
    except smodes_module.SmodesOutputError as exc:
        print(f"Failed to parse {path}: {exc}", file=sys.stderr)
        return 1

    print(f"{parsed.title}  (space group {parsed.space_group_number} {parsed.space_group_symbol})")
    print("a,b,c,alpha,beta,gamma = " + ", ".join(f"{x:g}" for x in parsed.lattice_parameters))
    print("Wyckoff: " + ", ".join(f"{w.species} {w.letter}" for w in parsed.wyckoff_positions))
    print(f"{len(parsed.irreps)} irrep(s):")
    for irrep in parsed.irreps:
        flags = []
        if irrep.n_translational_modes:
            flags.append(f"{irrep.n_translational_modes} translational")
        if irrep.raman_active:
            flags.append("Raman active")
        if irrep.ir_active:
            flags.append("IR active")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        mulliken = f" {irrep.mulliken}" if irrep.mulliken else ""
        print(
            f"  {irrep.label}{mulliken}  k={irrep.k_label}  degeneracy={irrep.degeneracy}  "
            f"modes={irrep.n_modes} (total {irrep.total_modes})  atoms={irrep.n_atoms}{flag_str}"
        )
    return 0


def _build_and_save_findsym_log_at(
    structure: Structure, out_path: Path, title: str, tolerance: float, isodata: str | None
) -> Path:
    resolution = isotropy.resolve_isodata(isodata)
    try:
        version = findsym_module.detect_findsym_version()
    except findsym_module.FindsymLogError:
        # Binary exists but its output isn't a recognizable version banner
        # (e.g. a non-standard build) -- fall back to the pre-6.x positional
        # format, matching build_findsym_input's own default.
        version = (2, 5)
    input_text = findsym_module.build_findsym_input(structure, title=title, tolerance=tolerance, version=version)
    log_text = findsym_module.run_findsym(input_text, isodata=resolution.value)
    out_path.write_text(log_text)
    return out_path


def _build_and_save_findsym_log(structure: Structure, tag: str, tolerance: float, isodata: str | None) -> Path:
    return _build_and_save_findsym_log_at(structure, Path(f"{tag}_findsym"), f"{tag} structure", tolerance, isodata)


def _read_or_build_findsym_log(
    path: Path, structure: Structure, title: str, tolerance: float, isodata: str | None
) -> tuple[str, "findsym_module.SymmetryInfo"]:
    """Always runs findsym fresh against `structure` and (over)writes the
    result at `path` -- never reads an existing file at `path` and trusts
    it, even if one is already there.

    `decomposition`'s own `--parent`/`--sub` are both given fresh, by path,
    on every invocation -- unlike `irrep`'s implicit "the POSCAR in this
    project directory," where a cached `{tag}_findsym` genuinely does
    describe a structure that stays fixed for the whole project. Trusting
    a same-named cache here is a real correctness risk, not just a stale-
    data inconvenience: a leftover `parent_findsym`/`subgroup_findsym` in
    the working directory could be for a structure the caller ISN'T
    currently pointing `--parent`/`--sub` at (a different project reused
    the same directory, or --sub simply changed), and the only guard a
    cache-then-validate approach can offer is an atom-COUNT check
    (`parse_findsym_log`'s own `n_atoms != structure.n_atoms`) -- silently
    wrong for two structures that coincidentally share an atom count but
    are genuinely different materials/distortions. Confirmed with the
    physicist directly: since findsym is a cheap, no-DFT symmetry search
    (the same cost this project already accepts paying elsewhere, e.g. the
    `--direction` findsym-escalation paths), there's no real benefit to
    caching here worth that risk -- always rebuild instead.

    Still writes to `path` (so `parent_findsym`/`subgroup_findsym` remain
    real, inspectable files after the run), just never reads one back in.
    Returns `(log_text, symmetry_info)` so callers never need to parse the
    same text a second time.
    """
    _build_and_save_findsym_log_at(structure, path, title, tolerance, isodata)
    log_text = path.read_text()
    symmetry_info = findsym_module.parse_findsym_log(log_text, structure, title=path.name)
    return log_text, symmetry_info


# Findsym tolerance for checking whether a FRESHLY-BUILT symbolic
# --direction result (an already-DISTORTED structure) can be reduced to a
# smaller, symmetrically-equivalent cell -- deliberately much TIGHTER than
# the 0.01 used to absorb ~1-2 mA of NUMERICAL tiling noise elsewhere.
# Applying that same looseness here would instead "see through" the real,
# deliberate ~0.001 A symbolic-direction displacement and incorrectly
# report the UNDISTORTED parent's own higher symmetry -- confirmed
# directly this session: a real generated CMO X3- structure at tolerance
# 0.001 (already 10x tighter than 0.01) still falsely reduced all the way
# back to the parent's own full I4/mmm symmetry, discarding the real
# physics, while 0.0001 (10x tighter than the displacement itself)
# correctly found the true, lower, physically genuine symmetry instead.
_SYMBOLIC_DIRECTION_REDUCE_TOLERANCE = displace.DISPLACEMENT_AMPLITUDE_ANGSTROM / 10


def _reduce_freeze_structure(structure: Structure, title: str, isodata: str | None) -> tuple:
    """After building ANY freeze-family output structure (`freeze` with or
    without `--direction`, `freeze-amplitude`/
    `landau-grid`), check
    whether its own real symmetry allows a smaller, equivalent cell than
    the joint-cell construction gave -- via a fresh findsym run at
    `_SYMBOLIC_DIRECTION_REDUCE_TOLERANCE` (see that constant's own
    docstring for why this needs a tighter tolerance than every other
    findsym check in this codebase, not the same 0.01 convention). Reuses
    `findsym.reduce_to_primitive_cell` -- the same self-validating (atom-
    count/pairing-consistent, never blindly trusted) reduction
    `decomposition --sub`'s own auto-reduction already uses.

    Returns `(structure_to_write, note | None, superlattice_and_log |
    None)`. `structure_to_write` is the ORIGINAL `structure` unchanged
    whenever no smaller cell is found (or the check itself couldn't run
    at all, e.g. no findsym binary) -- never a hard failure: the already-
    correct (if possibly bigger-than-necessary) structure is always still
    written regardless. `note` is `None` only in the single common case
    where nothing was found to report (findsym ran cleanly and confirmed
    no smaller cell exists) -- every other outcome (reduced, or couldn't
    check) gets an explicit, named note, never silent. `superlattice_and_
    log` is `(original_structure, findsym_log_text)` whenever a reduction
    actually happened -- the ORIGINAL, bigger structure and the exact
    findsym log that found the reduction, both meant to be saved
    alongside the new, smaller POSCAR (see `_write_freeze_reduction_
    byproducts`) so a physicist can inspect the bigger, more familiar-
    looking cell and the symmetry evidence for the reduction directly,
    not just take the smaller POSCAR on faith; `None` in every other case
    (nothing saved when nothing was reduced).
    """
    try:
        version = findsym_module.detect_findsym_version()
    except findsym_module.FindsymLogError:
        version = (2, 5)
    try:
        input_text = findsym_module.build_findsym_input(
            structure, title=title, tolerance=_SYMBOLIC_DIRECTION_REDUCE_TOLERANCE, version=version
        )
        resolution = isotropy.resolve_isodata(isodata)
        log = findsym_module.run_findsym(input_text, isodata=resolution.value)
    except (FileNotFoundError, RuntimeError, findsym_module.FindsymLogError) as exc:
        return structure, (
            f"  note: could not check for a smaller symmetrically-equivalent cell ({exc}) -- "
            "writing the structure at its current size."
        ), None

    try:
        reduced = findsym_module.reduce_to_primitive_cell(structure, log)
    except findsym_module.FindsymLogError as exc:
        return structure, (
            f"  note: findsym ran but the reduction couldn't be parsed ({exc}) -- "
            "writing the structure at its current size."
        ), None

    if reduced is None:
        return structure, None, None

    return reduced, (
        f"  note: this structure's own real symmetry allows a smaller, equivalent cell -- "
        f"reduced from {structure.n_atoms} to {reduced.n_atoms} atoms (same physics, "
        "less redundant padding). The original, bigger cell is saved as POSCAR_superlattice "
        "alongside it, with the findsym log that found the reduction as findsym_reduction.log."
        "\n  To read this structure's SPACE GROUP, run `symmetry` on POSCAR_superlattice "
        "rather than on the reduced POSCAR. Measured on real graphite K1-K4 and H1: the "
        "superlattice gives the correct group at every tolerance, while the reduced cell "
        "reports a proper SUBGROUP at findsym's tight tolerances (63 Cmcm for 193 P6_3/mcm, "
        "12 C2/m for 162 P-31m, and so on) -- the reduction is exact but leaves the smaller "
        "cell closer to findsym's own precision limit. See Issue 11."
    ), (structure, log)




def _write_direction_manifest(freeze_dir: Path, blocks: list) -> None:
    """Write `direction.txt`: what was actually built, in this folder.

    The directory NAME is a convenience and is lossy by design (see
    `_symbolic_coefficient_tag`). This file is the record. It exists because
    the single hardest question in this project's OPD work has been "which
    physical direction did position 3 mean" -- the whole of Issues 7, 12d and
    18. Writing the mechanism alongside the coefficient makes each structure
    self-describing, so the answer does not have to be re-derived later from
    the code that produced it.

    One block per irrep: `freeze X2+ X3-` writes a single joint directory, so
    a single manifest carries both.
    """
    lines = []
    for block in blocks:
        lines.append(f"irrep       {block['label']}")
        lines.append(f"mode        {block['mode']}")
        lines.append(f"amplitude   {block['amplitude']:g} A")
        lines.append(f"direction   {block['spec']}")
        lines.append("")
        lines.append("position  coefficient  mechanism")
        for position, coefficient, mechanism in block["rows"]:
            lines.append(f"{position:>8d}  {coefficient:>11g}  {mechanism}")
        lines.append("")
        lines.append(f"combined pattern norm   {block['norm']:.9g} A")
        lines.append("")
    (freeze_dir / "direction.txt").write_text("\n".join(lines))


def _write_freeze_reduction_byproducts(freeze_dir: Path, superlattice_and_log) -> None:
    """Saves the pre-reduction structure and the findsym log that found
    the reduction into `freeze_dir`, alongside the new, smaller `POSCAR`
    -- called only when `_reduce_freeze_structure` actually found a
    reduction (`superlattice_and_log` is `None` otherwise, a no-op).
    `POSCAR_superlattice` deliberately reuses the same filename `phonons`
    already uses for "the bigger, more familiar-looking version of this
    structure" -- the same semantic meaning here, even though the mechanism
    (a genuine symmetry reduction, not an axis-alignment remapping) differs.
    """
    if superlattice_and_log is None:
        return
    original, log_text = superlattice_and_log
    original.to_poscar(freeze_dir / "POSCAR_superlattice")
    (freeze_dir / "findsym_reduction.log").write_text(log_text)


def _reduce_freeze_structure_with_magmom(
    combined: Structure, title: str, isodata: str | None, tiled_magmom, magmom_overrides: dict,
) -> tuple:
    """`_reduce_freeze_structure`, except that a magnetic structure is never
    reduced. Returns `(structure_to_write, note | None, superlattice_and_log
    | None, superlattice_magmom_overrides | None)`; the last is always None
    (kept for the callers' shape).

    With `tiled_magmom` set, `freeze` has already written the smallest cell
    the distortion needs using only translations the MOMENTS respect.
    findsym ignores magnetism, so its reduction would merge centring partners
    whose moments differ -- and at frozen-phonon amplitudes it also erases
    part of the distortion (known issue 5d in the developer notes). Nothing to gain, something to
    lose, so it is skipped and `magmom_overrides` is left as given.
    """
    if tiled_magmom is not None:
        return combined, None, None, None
    combined, reduce_note, superlattice_and_log = _reduce_freeze_structure(combined, title, isodata)
    return combined, reduce_note, superlattice_and_log, None


def _build_and_save_smodes_output(
    symmetry_info: smodes_module.SymmetryInfo, k_label: str, tag: str, out_path: Path | None = None
) -> Path:
    """`out_path` overrides the default `{tag}_smodes_{k_label}` naming --
    used by `decomposition`, whose own `--name`/POSCAR-title-derived `tag`
    can get unwieldy for a real material (confirmed directly: real CMO's
    own tag makes `{tag}_smodes_X`-style names rather long) -- it uses a
    short, fixed `parent_smodes_{k_label}` name instead. Every other
    caller leaves this at the default, unaffected."""
    input_text = smodes_module.build_smodes_input(symmetry_info, k_label)
    output_text = smodes_module.run_smodes(input_text)
    if out_path is None:
        out_path = Path(f"{tag}_smodes_{k_label}")
    out_path.write_text(output_text)
    return out_path


def _run_symmetry(poscar_path: str, name: str | None, tolerance: float, isodata: str | None) -> int:
    try:
        structure = Structure.from_poscar(poscar_path)
    except (FileNotFoundError, PoscarFormatError) as exc:
        print(f"Failed to read {poscar_path}: {exc}", file=sys.stderr)
        return 1

    try:
        tag = config.resolve_tag(name, poscar_title=structure.comment)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        out_path = _build_and_save_findsym_log(structure, tag, tolerance, isodata)
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Wrote {out_path}")
    return 0


def _run_run_smodes(findsym_log_path: str, k_vector: str, poscar_path: str, name: str | None) -> int:
    try:
        structure = Structure.from_poscar(poscar_path)
    except (FileNotFoundError, PoscarFormatError) as exc:
        print(f"Failed to read {poscar_path}: {exc}", file=sys.stderr)
        return 1

    try:
        log_text = Path(findsym_log_path).read_text()
    except OSError as exc:
        print(f"Failed to read {findsym_log_path}: {exc}", file=sys.stderr)
        return 1

    try:
        tag = config.resolve_tag(name, poscar_title=structure.comment)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        symmetry_info = findsym_module.parse_findsym_log(log_text, structure, title=f"{tag}_findsym")
    except findsym_module.FindsymLogError as exc:
        print(f"Failed to parse {findsym_log_path}: {exc}", file=sys.stderr)
        return 1

    try:
        out_path = _build_and_save_smodes_output(symmetry_info, k_vector, tag)
    except (FileNotFoundError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Wrote {out_path}")
    return 0


def _irrep_dir_for_write(label: str) -> Path:
    """Where `irrep` writes this label's own directory: `<K>/Irrep_<label>/`.

    Grouping every irrep of one k-point under a folder named for that
    k-point keeps a bare `irrep GM` (which can expand to a dozen irreps)
    from scattering a dozen sibling directories across the working
    directory. A single named `irrep GM1+` nests identically -- one rule,
    no special case -- so `phonons GM1+` never has to work out which
    invocation produced the directory it is about to read.
    """
    return Path(smodes_module.k_vector_label_from_irrep(label)) / f"Irrep_{label}"


def _resolve_irrep_dir(label: str) -> Path:
    """Find this label's existing directory, preferring the nested layout.

    Returns the nested `<K>/Irrep_<label>/` when it exists, the flat
    `Irrep_<label>/` when only that does, and otherwise the nested path
    (the layout a fresh run would create).

    The flat fallback is load-bearing, not politeness: every VASP working
    directory computed before this layout existed is flat, including the
    physicist's own real `CMO_smodes-phonons/Irrep_X2+/` and every
    committed `*_Irrep_*` fixture. Commands that WRITE back into the
    resolved directory (`phonons`'s own `phonons.npz`) must use this same
    function, so a legacy directory gets updated in place rather than
    gaining a second, nested copy that `freeze` could later read while the
    real forces sit in the other one.
    """
    try:
        nested = _irrep_dir_for_write(label)
    except ValueError:
        return Path(f"Irrep_{label}")
    flat = Path(f"Irrep_{label}")
    if nested.is_dir():
        if flat.is_dir():
            # Half-migrated: two directories for one irrep. Nested wins,
            # but say so -- the flat one may hold the OUTCARs a stale
            # phonons.npz was actually computed from, and silently
            # preferring one would report results derived from data the
            # command never read.
            print(
                f"{label}: both {nested}/ and {flat}/ exist -- using {nested}/. "
                f"Move or delete {flat}/ so there is only one.",
                file=sys.stderr,
            )
        return nested
    if flat.is_dir():
        return flat
    return nested


def _expand_bare_k_point_labels(requested_labels, resolve_smodes):
    """Expand every letters-only label into each irrep at that k-point.

    `GM` means "every irrep at Gamma"; `GM4-` means that one irrep. The
    discriminator is exact, not a heuristic: under ISOTROPY's own
    convention a real irrep label ALWAYS carries a trailing digit
    (optionally signed), so `label.isalpha()` can never confuse the two.

    `resolve_smodes(k_label)` supplies that k-point's parsed smodes output
    and is the whole reason this is shared rather than duplicated --
    `irrep` may legitimately BUILD a missing one (running smodes), while
    `phonons` must only ever READ an existing one, since it is required
    never to spawn a symmetry subprocess (an earlier version that did took
    the suite from ~20s to ~157s). Both need identical expansion; only
    that one step differs.

    Returns `(labels, ok)`: order-preserving and deduplicated, so a label
    reachable both explicitly and via an expansion (`phonons GM GM4-`) is
    processed once. A bad entry never aborts the others -- `ok` is False
    if any failed, matching the "one bad label doesn't abort the rest"
    contract `irrep` already established.
    """
    resolved: list[str] = []
    ok = True
    for requested in requested_labels:
        label = requested.upper()
        if not label.isalpha():
            resolved.append(label)
            continue
        try:
            parsed = resolve_smodes(label)
        except Exception as exc:  # noqa: BLE001 -- reported, never fatal
            print(f"{requested}: {exc}", file=sys.stderr)
            ok = False
            continue
        found = [irrep.label for irrep in parsed.irreps] if parsed is not None else []
        if not found:
            print(
                f"{requested}: no such k-point {label!r} exists for this material -- smodes "
                "printed no irreps at all here.",
                file=sys.stderr,
            )
            ok = False
            continue
        print(
            f"{requested}: {label} has {len(found)} irrep(s) -- expanding to all of them: "
            + ", ".join(found)
        )
        resolved.extend(found)
    return list(dict.fromkeys(resolved)), ok


def _load_kpoints_mesh(kpoints_path: str) -> vasp_module.KpointsMesh | None:
    """Load an automatic-mesh KPOINTS file for later per-supercell scaling.

    Same "optional, warn and skip rather than error" contract everywhere
    it's used (`irrep`, `freeze`, `freeze-amplitude[-grid]`,
    `landau-grid`): a missing or unparseable
    KPOINTS file just means no KPOINTS get written alongside the generated
    POSCARs, never a reason to fail the whole command.
    """
    kpoints_file = Path(kpoints_path)
    if not kpoints_file.is_file():
        print(
            f"Warning: {kpoints_path} not found -- per-structure KPOINTS files were not generated. "
            "Each generated structure sits on a different-sized (often much bigger, rotated/skewed) "
            "supercell than the parent POSCAR, so its own mesh needs to be scaled down to keep the "
            "same real-space reciprocal-lattice-vector spacing -- it can't just be reused as-is. "
            "Pass your own converged KPOINTS via --kpoints (or place one named KPOINTS in the "
            "current directory) and this tool will write the correctly scaled version alongside "
            "every generated structure.",
            file=sys.stderr,
        )
        return None
    try:
        return vasp_module.parse_kpoints_mesh(kpoints_file.read_text())
    except vasp_module.KpointsError as exc:
        print(f"Warning: could not parse {kpoints_path}, skipping KPOINTS scaling: {exc}", file=sys.stderr)
        return None


def _write_scaled_kpoints(
    kmesh: vasp_module.KpointsMesh | None, original_lattice, new_lattice, dest_dir: Path, filename: str = "KPOINTS"
) -> None:
    if kmesh is None:
        return
    scaled = vasp_module.scaled_kpoints_mesh(kmesh, original_lattice, new_lattice)
    scaled.write(dest_dir / filename)


def _write_superlattice_kpoints(
    kmesh: vasp_module.KpointsMesh | None, original_lattice, superlattice_and_log, dest_dir: Path
) -> None:
    """Companion to `_write_freeze_reduction_byproducts`/`INCAR_superlattice`
    -- writes a correspondingly-scaled `KPOINTS_superlattice` for the
    saved, un-reduced structure whenever a reduction actually happened
    (`superlattice_and_log` is `None` otherwise, a no-op, matching every
    other reduction-byproduct writer's own contract).

    FIXED: a real, confirmed gap -- found directly from a physicist's own
    workflow question, not a defensive-programming guess. `INCAR_
    superlattice` was already written for exactly this reason (the bigger,
    un-reduced cell needs its own usable INCAR too), but no matching
    `KPOINTS_superlattice` ever was, across every freeze-family command
    with this feature.

    **CORRECTED (2026-09-18).** This docstring argued that relaxing the
    auto-reduced `POSCAR` instead of `POSCAR_superlattice` means "a DFT
    relaxation ... can never discover a state that genuinely needs the
    bigger cell's own periodicity", and therefore that testing a multi-arm
    blend REQUIRES relaxing the superlattice. The REPRESENTABILITY half is
    true -- a structure needing a bigger periodicity genuinely cannot exist
    on a smaller cell -- but the DISCOVERY half is wrong, and that is the
    half the conclusion rested on.

    A relaxation follows forces downhill from where it starts, and at a
    symmetric structure the symmetry-breaking forces are ZERO BY SYMMETRY,
    so it does not wander into a lower-symmetry state however large the
    cell is. Cubic BaTiO3 relaxes to cubic BaTiO3; the ferroelectric state
    is reached only by freezing GM4- first. A big cell alone buys nothing,
    and the superlattice is NOT the file that tests a multi-arm hypothesis.

    **A multi-arm blend is tested by FREEZING it** -- `freeze M2- --direction
    a,a,a`, putting a nonzero coefficient on each arm's own position (the
    `freeze-star` command this used to name was removed as redundant with
    exactly that spelling) -- and relaxing freeze's own
    output, already sized for the requested combination (real STO M2-: one
    arm 10 atoms, all three 20). That structure carries the broken symmetry
    the relaxation needs.

    `KPOINTS_superlattice` is still worth writing: the superlattice is
    offered to the user and has real uses (read the space group off it per
    Issue 11; decompose against it when reduction erases the distortion per
    known issue 5d/2 in the developer notes), and anyone running DFT on it needs a correctly
    scaled mesh exactly as `POSCAR` gets. Only the physics justification was
    wrong, not the feature.
    """
    if superlattice_and_log is None:
        return
    original, _log_text = superlattice_and_log
    _write_scaled_kpoints(kmesh, original_lattice, original.lattice, dest_dir, filename="KPOINTS_superlattice")


_INCAR_HINT_SCF = (
    "This is a static single-point (SCF) calculation -- no ionic or cell relaxation, since the "
    "whole point is evaluating the structure exactly as generated. Your INCAR needs NSW=0, "
    "IBRION=-1 (ISIF doesn't matter at NSW=0)."
)
_INCAR_HINT_RELAX = (
    "Freeze is meant as a relaxation seed -- freeze in a candidate order parameter as a "
    "symmetry-breaking kick, then let the ions and cell relax from there to the true minimum in "
    "that lower-symmetry basin. Your INCAR needs real relaxation settings (e.g. ISIF=3, "
    "IBRION=2, and a nonzero NSW), not a static run."
)
_INCAR_HINT_LEPSILON = (
    "This is a Born-effective-charge/dielectric-tensor (DFPT linear-response) calculation on "
    "the plain undistorted structure, not a force/energy evaluation of a displaced one. Your "
    "INCAR needs LEPSILON=.TRUE., IBRION=-1."
)


def _load_incar_template(incar_path: str, calc_type_hint: str) -> str | None:
    """Load a user-supplied INCAR template for later per-calculation-type
    tag overriding (see vasp.SCF_INCAR_OVERRIDES). Same "optional, warn and
    skip rather than error" contract as _load_kpoints_mesh, and independent
    of --kpoints -- each optional input file gets its own flag, not a
    combined one, matching --poscar/--kpoints already being separate.

    `calc_type_hint` (`_INCAR_HINT_SCF` or `_INCAR_HINT_RELAX`) explains, in
    the missing-file warning, which INCAR tags this particular command
    actually needs right -- callers get told the CORRECT physics, not just
    "a file is missing".
    """
    incar_file = Path(incar_path)
    if not incar_file.is_file():
        print(
            f"Warning: {incar_path} not found -- per-structure INCAR files were not generated. "
            f"{calc_type_hint} Pass your own INCAR template via --incar (or place one named INCAR "
            "in the current directory) and this tool will write the correctly adjusted version "
            "alongside every generated structure.",
            file=sys.stderr,
        )
        return None
    return incar_file.read_text()


def _write_incar(template_text: str | None, overrides: dict, dest_dir: Path, filename: str = "INCAR") -> None:
    if template_text is None:
        return
    (dest_dir / filename).write_text(vasp_module.apply_incar_overrides(template_text, overrides))


_MAGMOM_ERROR = object()
"""Sentinel returned by `_parse_and_validate_magmom` on a parse error or
atom-count mismatch (an error message has already been printed) -- distinct
from `None`, which means "nothing to do" (--magmom not requested, or the
template has no MAGMOM tag at all), so a caller can tell the two apart with
`is _MAGMOM_ERROR` / `is None` rather than risk conflating "no error" with
"no magnetic moments" (falsy-but-different failure modes)."""


def _parse_and_validate_magmom(
    magmom_flag: bool, incar_template: str | None, structure: Structure, incar_path: str, poscar_path: str
):
    """Shared by every --magmom-accepting command (irrep, bec, freeze,
    freeze-amplitude[-grid], landau-grid):
    parses and validates the parent structure's own MAGMOM ONCE, before any
    POSCAR/INCAR gets written, so a mismatch fails the whole command
    cleanly rather than leaving a partial, inconsistent output directory.

    Returns `None` if `--magmom` wasn't given or the template has no
    MAGMOM tag (both are "nothing to tile" -- not an error); the parsed
    per-atom list (in `structure`'s -- the PARENT's -- own POSCAR atom
    order) on success; or `_MAGMOM_ERROR` (after printing a clear message)
    if `--incar`'s own MAGMOM is malformed or doesn't have exactly as many
    values as the parent structure has atoms.
    """
    if incar_template is None:
        return None
    if not magmom_flag:
        # Without --magmom the template is copied verbatim, so a parent-order
        # MAGMOM lands on every generated structure as-is -- the wrong moment
        # on the wrong atom wherever the cell differs from the parent's (real
        # CMO GM5-: a 12-atom primitive cell given the parent's 24 values).
        try:
            has_magmom = vasp_module.parse_magmom(incar_template) is not None
        except vasp_module.VaspOutputError:
            has_magmom = True
        if has_magmom:
            print(
                f"Warning: {incar_path} sets MAGMOM, and without --magmom it is copied unchanged into every "
                "generated INCAR. MAGMOM is one value per atom in POSCAR order, so it is only right for a "
                "structure on the parent's own cell; generated structures are generally on a different "
                "(larger, or for a centred parent smaller) cell. Pass --magmom to tile it onto each "
                "structure (this also keeps every cell a whole multiple of the parent's).",
                file=sys.stderr,
            )
        return None
    try:
        parent_magmom = vasp_module.parse_magmom(incar_template)
    except vasp_module.VaspOutputError as exc:
        print(f"--magmom: {exc}", file=sys.stderr)
        return _MAGMOM_ERROR
    if parent_magmom is None:
        print(
            f"--magmom given but no MAGMOM tag found in {incar_path} -- nothing to tile. Add a "
            "MAGMOM line (in the parent POSCAR's own atom order) to your INCAR template, or drop "
            "--magmom.",
            file=sys.stderr,
        )
        return _MAGMOM_ERROR
    if len(parent_magmom) != structure.n_atoms:
        print(
            f"--magmom: MAGMOM in {incar_path} has {len(parent_magmom)} values but the parent "
            f"structure ({poscar_path}) has {structure.n_atoms} atoms -- check that MAGMOM matches "
            "the parent POSCAR's own atom order exactly.",
            file=sys.stderr,
        )
        return _MAGMOM_ERROR
    return parent_magmom


def _write_bec_directory(
    structure: Structure,
    kmesh: vasp_module.KpointsMesh | None,
    incar_template: str | None,
    parent_magmom: list | None = None,
) -> Path | None:
    """Write BEC/POSCAR (+ optional KPOINTS/INCAR) -- input for the
    LEPSILON=.TRUE. Born-charge/dielectric-tensor calculation, run on the
    plain undistorted parent structure. One per material, not one per
    irrep or k-point (Born charges are an equilibrium-structure property,
    see vasp.LEPSILON_INCAR_OVERRIDES's own docstring) -- shared by both
    the standalone `bec` command and `irrep --bec`'s own invocation for a
    Gamma-point, IR-active irrep, so the two can never disagree about what
    gets written. Idempotent like every other auto-created file in this
    project ({tag}_findsym, {tag}_smodes_{k}): does nothing and returns
    None if BEC/ already exists, rather than overwriting it.

    KPOINTS is written unscaled (a direct copy of the parent's own mesh) --
    the BEC calculation runs on the plain parent cell, no supercell folding
    involved at all, unlike Mode_i/KPOINTS's scaled mesh.
    """
    bec_dir = Path("BEC")
    if bec_dir.is_dir():
        return None
    bec_dir.mkdir(parents=True)
    structure.to_poscar(bec_dir / "POSCAR")
    if kmesh is not None:
        kmesh.write(bec_dir / "KPOINTS")
    if incar_template is not None:
        # BEC/POSCAR is `structure` written out directly, with no tiling at
        # all (same atom count and order as the parent) -- parent_magmom
        # therefore needs no mapping or tiling, just reformatting.
        lepsilon_overrides = vasp_module.LEPSILON_INCAR_OVERRIDES
        if parent_magmom is not None:
            lepsilon_overrides = {**lepsilon_overrides, "MAGMOM": vasp_module.format_magmom(parent_magmom)}
        _write_incar(incar_template, lepsilon_overrides, bec_dir)
    return bec_dir


def _discover_ir_active_gamma_irreps(
    structure: Structure, tag: str, tolerance: float, isodata: str | None
) -> list[str] | None:
    """Auto-creates the findsym log + Gamma-point smodes output if needed
    (the same helpers `irrep` itself uses), and returns the labels of
    every IR-active irrep found there -- the FULL set `bec`/`bec-analysis`
    need for a complete LO-TO/epsilon(0) analysis. `phonons <IRREP>` only
    ever diagonalizes one irrep's own block at a time (the entire point of
    symmetry block-diagonalization -- different irreps never mix), so
    epsilon(0) only comes out complete once every IR-active Gamma-point
    irrep's own contribution has been included, not just whichever one a
    user happened to pick first (see the developer notes' documented
    limitation).

    Returns None (after printing a clear error) on any failure to build or
    parse the findsym log/smodes output; an empty list (not None) if
    parsing succeeds but the material genuinely has no IR-active irrep at
    Gamma.
    """
    findsym_path = Path(f"{tag}_findsym")
    try:
        if not findsym_path.is_file():
            _build_and_save_findsym_log(structure, tag, tolerance, isodata)
        symmetry_info = findsym_module.parse_findsym_log(
            findsym_path.read_text(), structure, title=f"{tag}_findsym"
        )
    except (FileNotFoundError, RuntimeError, findsym_module.FindsymLogError) as exc:
        print(str(exc), file=sys.stderr)
        return None

    smodes_path = Path(f"{tag}_smodes_GM")
    try:
        if not smodes_path.is_file():
            _build_and_save_smodes_output(symmetry_info, "GM", tag)
        smodes_output = smodes_module.parse_smodes_output(smodes_path.read_text())
    except (FileNotFoundError, RuntimeError, smodes_module.SmodesOutputError) as exc:
        print(str(exc), file=sys.stderr)
        return None

    return [irrep.label for irrep in smodes_output.irreps if irrep.ir_active]


def _run_bec(
    poscar_path: str,
    name: str | None,
    tolerance: float,
    isodata: str | None,
    kpoints_path: str,
    incar_path: str,
    magmom: bool = False,
) -> int:
    """Discovers every IR-active Gamma-point irrep (auto-creating the
    findsym log/Gamma smodes output if needed) and runs `irrep` for all of
    them at once -- a complete LO-TO/epsilon(0) analysis needs every one
    of them, not just whichever single irrep a user happened to pick
    first (see _discover_ir_active_gamma_irreps's own docstring). Each
    gets its own Irrep_<label>/Mode_i/POSCAR (for computing forces on) via
    `_run_irrep`'s own existing, idempotent logic, called with
    write_bec=True -- exactly as if `irrep --bec` had been called
    directly on each label. Running `bec` at all is itself the user's
    explicit statement that this material is worth a Born-charge/LO-TO
    analysis (see `irrep --bec`'s own help text for why that's never
    assumed automatically -- it isn't meaningful for a metal). `bec` no
    longer writes BEC/ itself at all -- it's purely a discovery +
    delegation step onto that same shared logic, so the two can never
    disagree about what gets written, and rerunning `bec` (e.g. after a
    new irrep becomes IR-active-relevant) is always a safe, idempotent
    no-op for anything already there.
    """
    try:
        structure = Structure.from_poscar(poscar_path)
    except (FileNotFoundError, PoscarFormatError) as exc:
        print(f"Failed to read {poscar_path}: {exc}", file=sys.stderr)
        return 1

    try:
        tag = config.resolve_tag(name, poscar_title=structure.comment)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    ir_active_labels = _discover_ir_active_gamma_irreps(structure, tag, tolerance, isodata)
    if ir_active_labels is None:
        return 1
    if not ir_active_labels:
        print(
            "No IR-active irreps found at Gamma -- LO-TO splitting cannot occur for this "
            "material by symmetry, so there is nothing to set up.",
            file=sys.stderr,
        )
        return 1

    print(
        "IR-active Gamma-point irrep(s) found: "
        + ", ".join(ir_active_labels)
        + " -- creating Irrep_<label>/ for each (a complete LO-TO/epsilon(0) analysis needs "
        "every one of them, not just one)."
    )
    return _run_irrep(
        ir_active_labels, poscar_path, name, tolerance, isodata, kpoints_path, incar_path,
        write_bec=True, magmom=magmom,
    )


def _run_irrep(
    irrep_labels: list[str],
    poscar_path: str,
    name: str | None,
    tolerance: float,
    isodata: str | None,
    kpoints_path: str = "KPOINTS",
    incar_path: str = "INCAR",
    write_bec: bool = False,
    magmom: bool = False,
    write_opd: bool = False,
) -> int:
    try:
        structure = Structure.from_poscar(poscar_path)
    except (FileNotFoundError, PoscarFormatError) as exc:
        print(f"Failed to read {poscar_path}: {exc}", file=sys.stderr)
        return 1

    kmesh = _load_kpoints_mesh(kpoints_path)
    incar_template = _load_incar_template(incar_path, _INCAR_HINT_SCF)

    # Parsed and validated ONCE, up front -- a mismatch is a property of the
    # parent structure + INCAR template, not of any one requested label, so
    # there's nothing label-specific to learn by discovering it deep inside
    # the per-label loop below. Failing before any POSCAR gets written also
    # avoids a partial, inconsistent Irrep_<label>/ directory.
    parent_magmom = _parse_and_validate_magmom(magmom, incar_template, structure, incar_path, poscar_path)
    if parent_magmom is _MAGMOM_ERROR:
        return 1

    try:
        tag = config.resolve_tag(name, poscar_title=structure.comment)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    findsym_path = Path(f"{tag}_findsym")
    try:
        if not findsym_path.is_file():
            _build_and_save_findsym_log(structure, tag, tolerance, isodata)
        findsym_log_text = findsym_path.read_text()
        symmetry_info = findsym_module.parse_findsym_log(findsym_log_text, structure, title=f"{tag}_findsym")
    except (FileNotFoundError, RuntimeError, findsym_module.FindsymLogError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    smodes_cache: dict = {}
    overall_ok = True

    # `--opd` no longer varies with how the request was TYPED. A bare
    # `irrep GM` and an explicit `irrep GM1+` both report each irrep on its
    # own native cell; a bare request just does it for more of them. The
    # old split (bare => one shared k-point-wide cell, explicit => own
    # cell) meant the same irrep got a different answer depending on how
    # the user asked for it -- see the note at the end of this function.
    def resolve_smodes_building_if_missing(k_label: str):
        """`irrep`'s resolver: build the smodes output when absent.

        `irrep` is the command that CREATES this material's per-k-point
        smodes output in the first place, so a missing one is an ordinary
        first-run state here, not an error.
        """
        if k_label not in smodes_cache:
            smodes_path = Path(f"{tag}_smodes_{k_label}")
            if not smodes_path.is_file():
                _build_and_save_smodes_output(symmetry_info, k_label, tag)
            smodes_cache[k_label] = smodes_module.parse_smodes_output(smodes_path.read_text())
        return smodes_cache[k_label]

    resolved_labels, expansion_ok = _expand_bare_k_point_labels(
        irrep_labels, resolve_smodes_building_if_missing
    )
    overall_ok = overall_ok and expansion_ok

    for label in resolved_labels:
        try:
            k_label = smodes_module.k_vector_label_from_irrep(label)
        except ValueError as exc:
            print(f"{label}: {exc}", file=sys.stderr)
            overall_ok = False
            continue

        if k_label not in smodes_cache:
            smodes_path = Path(f"{tag}_smodes_{k_label}")
            try:
                if not smodes_path.is_file():
                    _build_and_save_smodes_output(symmetry_info, k_label, tag)
                smodes_cache[k_label] = smodes_module.parse_smodes_output(smodes_path.read_text())
            except (FileNotFoundError, RuntimeError, smodes_module.SmodesOutputError) as exc:
                print(f"{label}: {exc}", file=sys.stderr)
                overall_ok = False
                continue

        try:
            irrep = smodes_cache[k_label].irrep(label)
        except KeyError as exc:
            print(f"{label}: {exc}", file=sys.stderr)
            overall_ok = False
            continue

        irrep_dir = _resolve_irrep_dir(label)
        paths = displace.write_mode_poscars(irrep, irrep_dir, comment=f"{tag} {label}")
        print(f"{label}: wrote {len(paths)} mode(s) to {irrep_dir}/")

        # `--opd` is the single direction report (matching `phonons
        # --generate-opd`); `irrep` is pre-DFT, so `opd.txt` IS the
        # symmetry-only prediction.

        # Per-irrep OPD report, on this irrep's OWN cell. Skipped when this
        # label came from a bare k-point expansion -- the k-point-wide
        # report written after the loop covers it on the shared cell, and
        # building both would pay the joint-cell cost twice for the same
        # information.
        # `degeneracy > 1` matches the guard on the predicted-directions
        # report just above, and for the same reason: a degeneracy-1 irrep
        # has exactly ONE order-parameter direction, which is SMODES's own
        # native pattern and is already what `Mode_i/POSCAR` contains.
        # There is nothing to construct, so building a basis for it would
        # be pure cost -- and not free cost: it can trigger real findsym
        # subprocess calls. That is the documented ~20s-to-~157s failure
        # mode this project already hit once, and a real regression test
        # (`test_multi_irrep_multi_kpoint_end_to_end`, which pins findsym to
        # exactly 2 invocations per command) caught it here again.
        if write_opd and irrep.degeneracy > 1:
            _write_opd_report(
                [irrep], structure, irrep_dir / "opd.txt",
                heading=f"Irrep {label} (k-point {k_label}) -- {irrep.n_modes} mode(s), "
                        f"degeneracy {irrep.degeneracy}, on this irrep's own cell",
                findsym_log_text=findsym_log_text, isodata=isodata,
            )

        if kmesh is not None:
            scaled = vasp_module.scaled_kpoints_mesh(kmesh, structure.lattice, irrep.superlattice)
            scaled.write(irrep_dir / "Undistorted" / "KPOINTS")
            for i in range(irrep.n_modes):
                scaled.write(irrep_dir / f"Mode_{i}" / "KPOINTS")

        if incar_template is not None:
            scf_overrides = vasp_module.SCF_INCAR_OVERRIDES
            if parent_magmom is not None:
                # Mode_i/Undistorted's own POSCAR is the irrep's NATIVE cell
                # directly (displace.supercell_structure -- no further
                # coset/multiplier tiling beyond that), so mapping parent's
                # MAGMOM onto irrep.atom_species's own native order is the
                # whole job here.
                try:
                    native_magmom = supercell_module.map_parent_values_to_native(irrep, structure, parent_magmom)
                except supercell_module.SupercellError as exc:
                    print(f"{label}: --magmom: {exc}", file=sys.stderr)
                    overall_ok = False
                    continue
                scf_overrides = {**scf_overrides, "MAGMOM": vasp_module.format_magmom(native_magmom)}
            _write_incar(incar_template, scf_overrides, irrep_dir / "Undistorted")
            for i in range(irrep.n_modes):
                _write_incar(incar_template, scf_overrides, irrep_dir / f"Mode_{i}")

        if k_label == "GM" and irrep.ir_active:
            if write_bec:
                bec_dir = _write_bec_directory(structure, kmesh, incar_template, parent_magmom)
                if bec_dir is not None:
                    print(
                        f"{label}: symmetry analysis confirms this mode CAN undergo LO-TO "
                        f"splitting (IR-active, Gamma-point) -- wrote {bec_dir}/ (LEPSILON=.TRUE. "
                        f"input; run it, leaving the resulting OUTCAR in {bec_dir}/ -- "
                        f"`phonons {label} --born-charges` will pick it up automatically)"
                    )
            else:
                print(
                    f"{label}: this is an IR-active Gamma-point mode -- LO-TO (non-analytic) "
                    "splitting is symmetry-allowed. Whether it's physically meaningful depends "
                    "on this material being an insulator or semiconductor: LO-TO splitting "
                    "relies on a macroscopic electric field that free carriers in a metal would "
                    "completely screen out, and Born effective charges/LEPSILON=.TRUE. aren't "
                    "well-defined for a metallic ground state. If this material is insulating, "
                    f"re-run with `irrep {label} --bec` to generate BEC/ (the LEPSILON=.TRUE. "
                    "input LO-TO splitting needs)."
                )
        elif write_bec:
            if k_label != "GM":
                print(
                    f"{label}: --bec requested, but this irrep is at {k_label}, not Gamma -- "
                    "LO-TO splitting only ever applies to zone-center (q -> 0) modes, so no "
                    "BEC/ was generated for this label."
                )
            else:
                print(
                    f"{label}: --bec requested, but symmetry analysis shows this Gamma-point "
                    "mode is NOT IR-active -- it cannot couple to a macroscopic electric field, "
                    "so LO-TO splitting does not apply and no BEC/ was generated for this label."
                )

    # NO shared k-point-wide OPD report. `irrep GM --opd` used to write one
    # `opd_GM.txt` covering every irrep at once, on a single shared cell,
    # INSTEAD of the per-irrep reports above. That was wrong on both counts.
    #
    # It does not match the software. Nothing builds that way: `freeze M5-`
    # puts M5- on its own native cell, `phonons --generate-opd` reports one
    # irrep on its own cell, and `decomposition` has resolved each irrep
    # independently since v0.92.0. A report on a cell no command uses
    # describes directions the user cannot then address.
    #
    # It is also actively misleading at a k-point with an ORIENTATION SPLIT
    # (smodes placing different irreps of one k-point on differently-
    # oriented native cells -- real STO M and X, real BaTiO3 GM and F). One
    # shared frame cannot host those without misaligning atoms, so the
    # shared report permuted position labels that are correct per-irrep.
    # Measured directly: the amplitude baseline built this way reported 6
    # STO rows as failures which come back exactly right per-irrep
    # (113 clean/6 wrong -> 119 clean/0 wrong).
    #
    # The shared cell is still worth measuring -- it is the only frame in
    # which different irreps are directly comparable, which is what the
    # cross-irrep orthogonality check needs (that check is what caught real
    # CMO Gamma at |dot| = 1.000000). That measurement now lives in
    # the amplitude-baseline script (development)'s own explicitly-labelled
    # JOINT-CELL SECTION, where an orientation split reads as the known
    # property it is, rather than in a user command that would pay a real
    # findsym cost per invocation for a diagnostic -- and could fail
    # outright doing it (real BaTiO3 L raises today).

    return 0 if overall_ok else 1






def _find_outcar(directory: Path) -> Path:
    plain = directory / "OUTCAR"
    gz = directory / "OUTCAR.gz"
    if plain.is_file():
        return plain
    if gz.is_file():
        return gz
    raise FileNotFoundError(f"no OUTCAR or OUTCAR.gz found in {directory}")


def _describe_unready_forces(label: str, irrep_dir: Path, n_modes: int) -> str | None:
    """Say WHY an irrep has no phonons yet, or `None` if it is ready.

    Matters most for `phonons GM`, where a bare k-point expands to several
    irreps that are legitimately at different stages -- some computed, some
    submitted, some never set up. The bare `no OUTCAR or OUTCAR.gz found in
    GM/Irrep_GM5-/Mode_0` this replaced named a path but not the situation,
    and did not say which of three genuinely different things to do next.
    """
    if not irrep_dir.is_dir():
        return (f"{label}: not calculated -- {irrep_dir}/ does not exist. Run "
                f"`smodes-phonons irrep {label}` to write its displacements, run VASP on "
                "each Mode_i/, then re-run this command.")

    missing = [i for i in range(n_modes)
               if not (irrep_dir / f"Mode_{i}" / "OUTCAR").is_file()
               and not (irrep_dir / f"Mode_{i}" / "OUTCAR.gz").is_file()]
    if not missing:
        return None
    if len(missing) == n_modes:
        return (f"{label}: not calculated -- {irrep_dir}/ has its displacements but no forces "
                f"yet (0 of {n_modes} Mode_i/OUTCAR). Run VASP in each Mode_i/, then re-run "
                "this command.")
    return (f"{label}: incomplete -- {n_modes - len(missing)} of {n_modes} Mode_i/OUTCAR "
            f"present in {irrep_dir}/ (missing: "
            + ", ".join(f"Mode_{i}" for i in missing)
            + "). All modes are needed to build the force-constant matrix.")


def _format_phonon_summary(label: str, result: phonons_module.PhononResult) -> str:
    lines = [f"{label} ({len(result.frequencies_cm)} modes):"]
    lines.append(f"  {'mode':>4}  {'cm-1':>10}  {'THz':>8}  {'meV':>8}  {'fc eigval (eV/A^2)':>20}")
    for i in range(len(result.frequencies_cm)):
        lines.append(
            f"  {i:>4}  {result.frequencies_cm[i]:>10.3f}  {result.frequencies_thz[i]:>8.3f}  "
            f"{result.frequencies_mev[i]:>8.3f}  {result.force_constant_eigenvalues[i]:>20.4f}"
        )
    return "\n".join(lines)


_EIGENVECTOR_SET_TITLES = {
    "force_constant": "Force Constant Matrix Eigenvectors",
    "dynamical_matrix": "Dynamical Matrix Eigenvectors",
    "atomic_displacement": "Atomic Displacement Eigenvectors (mass-unweighted -- how atoms actually move)",
}


def _render_eigenvector_sets(titled_sets: list) -> list:
    lines = []
    for title, vectors in titled_sets:
        lines.append(title)
        for i, vec in enumerate(vectors):
            lines.append(f"Mode {i}:")
            for row in vec:
                lines.append(" ".join(f"{x:.7f}" for x in row))
        lines.append("")
    return lines


def _format_phonon_report(
    label: str,
    result: phonons_module.PhononResult,
    lo_groups: list | None = None,
) -> str:
    """The irrep's own report. With Born charges, BOTH branches appear here.

    `result` is the UNCORRECTED spectrum. In a polar crystal at Gamma that is
    the TRANSVERSE (TO) branch: the transverse modes carry no macroscopic
    field, so the non-analytic term leaves them alone and the plain
    force-constant matrix already describes them.

    `lo_groups` is `[(direction_labels, corrected_result), ...]`, one entry per
    symmetry-equivalent q_hat group -- the LONGITUDINAL (LO) branch. It used to
    be written only to `BEC/report.txt`, which split one irrep's physics across
    two files and two directories; the physicist asked for both in one place.

    **LO IS DIRECTION-DEPENDENT AND TO IS NOT**, which is why this is one TO
    block and one LO block PER DIRECTION GROUP rather than two columns. Real
    STO: x, y and z collapse into a single group while [1,1,1] gives a
    genuinely different corrected spectrum.

    **The LO eigenvectors are NOT the TO eigenvectors.** The correction is a
    rank-1 term added to the force-constant matrix and the whole matrix is
    re-diagonalized, so the modes MIX -- an eigenvector survives only if it is
    orthogonal to the correction, and then its eigenvalue survives too.
    Measured on real STO GM4-, the two spectra interlace strictly
    (-66.547 < 2.234 < 2.838 < 154.267 < 157.022 < 513.867 < 534.759 < 680.481)
    and NO eigenvalue is shared, so no eigenvector is preserved. Each LO block
    therefore carries its own eigenvectors, not a note pointing at the TO ones.
    """
    lo_groups = lo_groups or []
    heading = "TRANSVERSE (TO) branch -- uncorrected" if lo_groups else None
    lines = [f"Phonon results for Irrep_{label}", ""]
    if heading:
        lines += [heading,
                  "  No macroscopic field acts on a transverse mode, so the plain",
                  "  force-constant matrix already describes this branch.", ""]
    lines += [_format_phonon_summary(label, result), ""]
    lines += _render_eigenvector_sets(
        [
            (_EIGENVECTOR_SET_TITLES["force_constant"], result.force_constant_eigenvectors),
            (_EIGENVECTOR_SET_TITLES["dynamical_matrix"], result.dynamical_matrix_eigenvectors),
            (_EIGENVECTOR_SET_TITLES["atomic_displacement"], result.atomic_displacement_eigenvectors),
        ]
    )
    for labels, corrected in lo_groups:
        lines += ["", "=" * 72, "",
                  f"LONGITUDINAL (LO) branch -- q_hat along {labels}",
                  "  Non-analytic (LO-TO) correction applied, per Gonze & Lee",
                  "  PRB 55, 10355 (1997). The modes MIX under it: these",
                  "  eigenvectors are NOT the TO eigenvectors above.",
                  "",
                  "  NOTE: a periodic supercell carries no macroscopic field, so",
                  "  freezing an LO pattern and running ordinary DFT returns the",
                  "  ZERO-FIELD energy -- the right displacement, the wrong energy",
                  "  surface. `freeze` uses the TO eigenvectors in phonons.npz.",
                  ""]
        lines += [_format_phonon_summary(label, corrected), ""]
        lines += _render_eigenvector_sets(
            [
                (_EIGENVECTOR_SET_TITLES["force_constant"], corrected.force_constant_eigenvectors),
                (_EIGENVECTOR_SET_TITLES["dynamical_matrix"], corrected.dynamical_matrix_eigenvectors),
                (_EIGENVECTOR_SET_TITLES["atomic_displacement"], corrected.atomic_displacement_eigenvectors),
            ]
        )
    return "\n".join(lines) + "\n"


def _parse_lo_to_direction(spec: str) -> np.ndarray:
    """'1,1,1' -> array([1.0, 1.0, 1.0]) -- a plain Cartesian propagation
    direction (q_hat, not yet normalized), unrelated to the freeze-family
    commands' own symbolic a/b/c direction notation (that convention is
    for combining degenerate-partner MODES; LO-TO's --direction is a
    single physical q-vector, not a sum of separately-constructed
    partners)."""
    parts = spec.split(",")
    if len(parts) != 3:
        raise ValueError(f"--direction {spec!r} must be three comma-separated numbers, e.g. '1,1,1'")
    try:
        values = [float(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"--direction {spec!r}: {exc}") from exc
    if not any(values):
        raise ValueError(f"--direction {spec!r} is the zero vector -- no propagation direction to compute along")
    return np.array(values)


def _format_lo_to_report(irrep_label: str, baseline: phonons_module.PhononResult, groups_report: list) -> str:
    lines = [f"LO-TO splitting for {irrep_label} (IR-active):", ""]
    lines.append("  uncorrected: " + ", ".join(f"{f:.3f}" for f in sorted(baseline.frequencies_cm)) + " cm^-1")
    lines.append("")
    for labels, corrected in groups_report:
        lines.append(f"  direction {labels}:")
        lines.append("    corrected:   " + ", ".join(f"{f:.3f}" for f in sorted(corrected.frequencies_cm)) + " cm^-1")
    return "\n".join(lines)


def _format_born_charges_report(atom_species: list, born_charges: np.ndarray) -> str:
    lines = ["Born effective charges (Z*, in e):"]
    for sp, z in zip(atom_species, born_charges):
        lines.append(f"  {sp}:")
        for row in z:
            lines.append("    " + "  ".join(f"{x: .4f}" for x in row))
    return "\n".join(lines)


def _format_static_dielectric_report(eps0: np.ndarray, unstable_excluded: bool) -> str:
    lines = ["Static dielectric tensor epsilon(0):"]
    for row in eps0:
        lines.append("  " + "  ".join(f"{x: .4f}" for x in row))
    if unstable_excluded:
        lines.append(
            "  Warning: at least one polar, UNSTABLE mode was excluded from this sum -- the "
            "reference structure isn't a true energy minimum (a real ferroelectric-type "
            "instability, not a bug), so this is NOT the true equilibrium static dielectric "
            "constant."
        )
    return "\n".join(lines)


def _format_electronic_dielectric_report(dielectric_tensor: np.ndarray) -> str:
    lines = ["Electronic (clamped-ion) dielectric tensor epsilon(infinity):"]
    for row in dielectric_tensor:
        lines.append("  " + "  ".join(f"{x: .4f}" for x in row))
    return "\n".join(lines)


def _run_bec_analysis(
    born_charges_path: str | None, poscar_path: str, name: str | None, tolerance: float, isodata: str | None
) -> int:
    """The complete Born-charge-derived analysis, no irrep argument needed
    up front: Z*/epsilon_infinity come directly from the OUTCAR (no phonon
    data involved), then every IR-active Gamma-point irrep is discovered
    (`_discover_ir_active_gamma_irreps`) and run through `_run_phonons`
    (its own full LO-TO/epsilon(0) report, reused as-is -- not
    reimplemented here) for whichever ones already have computed phonon
    data; an irrep whose Mode_i/OUTCAR data isn't there yet is skipped
    with a note, not a hard failure, so a partial analysis is still useful
    while DFT calculations for the rest are still running.

    Each per-irrep call already prints/writes its OWN complete report
    (including that irrep's own epsilon(0), computed AS IF it were the
    only irrep, same as running `phonons <label> --born-charges` directly
    -- and each one also overwrites BEC/report.txt in turn, so only the
    LAST processed irrep's own report ends up persisted there). This
    function's own added value is the material's TRUE total epsilon(0) --
    every included irrep's own contribution, properly summed on top of
    epsilon_infinity once (not each irrep's own report re-adding
    epsilon_infinity redundantly) -- computed directly via
    lo_to.mode_effective_charges/static_dielectric_tensor on each
    included irrep's own saved phonons.npz, and written separately to
    BEC/combined_report.txt so it can never be clobbered by any single
    irrep's own per-irrep report.
    """
    bec_dir = Path(_DEFAULT_BEC_DIR)
    if born_charges_path is None:
        try:
            born_charges_path = str(_find_outcar(bec_dir))
        except FileNotFoundError:
            print(
                f"No OUTCAR or OUTCAR.gz found in {bec_dir}/ -- run the LEPSILON=.TRUE. "
                f"calculation in {bec_dir}/ (see `bec`) first, leaving the resulting OUTCAR "
                "there, or pass --born-charges PATH to point at data elsewhere.",
                file=sys.stderr,
            )
            return 1

    try:
        born_charges = vasp_module.parse_born_effective_charges(born_charges_path)
        dielectric_tensor = vasp_module.parse_dielectric_tensor(born_charges_path)
    except (FileNotFoundError, vasp_module.VaspOutputError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        structure = Structure.from_poscar(poscar_path)
    except (FileNotFoundError, PoscarFormatError) as exc:
        print(f"Failed to read {poscar_path}: {exc}", file=sys.stderr)
        return 1

    atom_labels = [f"atom {i + 1}" for i in range(born_charges.shape[0])]
    if structure.n_atoms == born_charges.shape[0]:
        atom_labels = structure.species

    born_charges_report = _format_born_charges_report(atom_labels, born_charges)
    electronic_report = _format_electronic_dielectric_report(dielectric_tensor)
    print(born_charges_report)
    print(electronic_report)

    try:
        tag = config.resolve_tag(name, poscar_title=structure.comment)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    ir_active_labels = _discover_ir_active_gamma_irreps(structure, tag, tolerance, isodata)
    if ir_active_labels is None:
        return 1

    if not ir_active_labels:
        print(
            "No IR-active irreps found at Gamma -- nothing beyond Z*/epsilon_infinity to report.",
            file=sys.stderr,
        )
        if bec_dir.is_dir():
            report_path = bec_dir / "report.txt"
            report_path.write_text("\n\n".join([born_charges_report, electronic_report]) + "\n")
            print(f"Wrote {report_path}")
        return 0

    print(
        "\nIR-active Gamma-point irrep(s) found: "
        + ", ".join(ir_active_labels)
        + " -- running the full LO-TO/epsilon(0) analysis for each one already computed.\n"
    )

    smodes_path = Path(f"{tag}_smodes_GM")
    eps0_total = np.array(dielectric_tensor, dtype=np.float64).copy()
    any_unstable = False
    included = []
    skipped = []
    for label in ir_active_labels:
        if _run_phonons(label, poscar_path, name, born_charges_path, None, tolerance, isodata) != 0:
            skipped.append(label)
            continue

        try:
            irrep = smodes_module.parse_smodes_output(smodes_path.read_text()).irrep(label)
        except (smodes_module.SmodesOutputError, KeyError) as exc:
            print(f"{label}: {exc}", file=sys.stderr)
            skipped.append(label)
            continue

        result = phonons_module.load_phonon_result(_resolve_irrep_dir(label) / "phonons.npz")
        volume = float(abs(np.linalg.det(irrep.superlattice)))
        mode_charges = lo_to_module.mode_effective_charges(irrep.modes, result.mode_basis_eigenvectors, born_charges)
        contribution, unstable_excluded = lo_to_module.static_dielectric_tensor(
            mode_charges, result.dynamical_matrix_eigenvalues, np.zeros((3, 3)), volume
        )
        eps0_total += contribution
        any_unstable = any_unstable or unstable_excluded
        included.append(label)

    summary_lines = [
        "Combined static dielectric tensor epsilon(0), summed across every included "
        f"IR-active Gamma-point irrep ({', '.join(included) if included else 'none'}):"
    ]
    for row in eps0_total:
        summary_lines.append("  " + "  ".join(f"{x: .4f}" for x in row))
    if any_unstable:
        summary_lines.append(
            "  Warning: at least one included irrep had a polar, UNSTABLE mode excluded from "
            "its own contribution -- see that irrep's own report above for which one."
        )
    if skipped:
        summary_lines.append(
            f"  Note: {', '.join(skipped)} not included -- phonon data not ready yet (compute "
            "forces on its Mode_i/POSCAR files, run `phonons`, then rerun `bec-analysis`). "
            "The epsilon(0) above is a PARTIAL sum, not the material's true total."
        )
    summary = "\n".join(summary_lines)
    print(summary)

    if bec_dir.is_dir():
        combined_report_path = bec_dir / "combined_report.txt"
        combined_report_path.write_text(
            "\n\n".join([born_charges_report, electronic_report, summary]) + "\n"
        )
        print(f"Wrote {combined_report_path}")
    return 0


def _format_supercell_matrix(multipliers) -> str:
    """Human-readable label for a nice_supercell_matrix result -- "n1xn2xn3"
    for the common diagonal case (STO/CMO/TiO2/BaTiO3/graphite/SiO2 never
    need anything else), or the full matrix for a genuinely non-diagonal
    one (needed for monoclinic/triclinic materials, e.g. HfO2's non-Gamma
    irreps -- see supercell.py's module docstring)."""
    m = np.asarray(multipliers)
    if m.shape == (3, 3) and np.array_equal(m, np.diag(np.diag(m))):
        n1, n2, n3 = np.diag(m)
        return f"{n1}x{n2}x{n3}"
    if m.shape == (3,):
        n1, n2, n3 = m
        return f"{n1}x{n2}x{n3}"
    rows = ", ".join("[" + " ".join(str(int(x)) for x in row) + "]" for row in m)
    return f"non-diagonal supercell matrix {rows}"


def _format_supercell_report(
    label: str, result: phonons_module.PhononResult, big_displacement_sets: dict, multipliers: tuple
) -> str:
    lines = [
        f"Phonon results for Irrep_{label}, mapped onto the {_format_supercell_matrix(multipliers)} "
        "supercell (POSCAR_superlattice)",
        "",
        _format_phonon_summary(label, result),
        "",
    ]
    lines += _render_eigenvector_sets(
        [(_EIGENVECTOR_SET_TITLES[name], big_displacement_sets[name]) for name in _EIGENVECTOR_SET_TITLES]
    )
    return "\n".join(lines) + "\n"


_DEFAULT_BEC_DIR = "BEC"


def _run_phonons_labels(
    irrep_labels: list[str],
    poscar_path: str,
    name: str | None,
    born_charges_path: str | None = None,
    direction_args: list[str] | None = None,
    tolerance: float = 0.0,
    isodata: str | None = None,
    generate_opd: bool = False,
) -> int:
    """`phonons` for one or more labels, expanding a bare k-point.

    `phonons GM` solves every irrep at Gamma; `phonons GM4-` solves that
    one -- the same catch-all `irrep` has always had, which previously
    stopped at `irrep`: after `irrep M` and eight VASP runs, a user had to
    type `phonons M1+` ... `phonons M5-` by hand.

    Deliberately READ-ONLY when expanding. `phonons` must never spawn a
    symmetry subprocess, so a missing `{tag}_smodes_{K}` is reported (with
    the `irrep` command that would create it) rather than built here.

    Each label is run independently through `_run_phonons`, which is
    unchanged -- one irrep's missing OUTCARs, or an irrep that legitimately
    fails, must not cost the user every other irrep's results. Exit code is
    0 only if every requested label succeeded, matching `irrep`.

    `--generate-opd` needs no special handling: it writes each irrep's own
    `opd_eigenvectors.txt` into that irrep's own directory, so running it
    over several irreps just produces several per-irrep reports. There is
    no k-point-wide file, matching `irrep --opd` (see the note at the end
    of `_run_irrep` for why a shared k-point cell is wrong here).
    """
    try:
        structure = Structure.from_poscar(poscar_path)
        tag = config.resolve_tag(name, poscar_title=structure.comment)
    except (FileNotFoundError, PoscarFormatError, RuntimeError) as exc:
        print(f"Failed to read {poscar_path}: {exc}"
              if not isinstance(exc, RuntimeError) else str(exc), file=sys.stderr)
        return 1

    def resolve_smodes_read_only(k_label: str):
        smodes_path = Path(f"{tag}_smodes_{k_label}")
        if not smodes_path.is_file():
            raise FileNotFoundError(
                f"{smodes_path} not found -- run `smodes-phonons irrep {k_label}` first "
                "to create the displacements and the smodes output."
            )
        return smodes_module.parse_smodes_output(smodes_path.read_text())

    labels, ok = _expand_bare_k_point_labels(irrep_labels, resolve_smodes_read_only)
    for label in labels:
        if _run_phonons(label, poscar_path, name, born_charges_path,
                        direction_args, tolerance, isodata, generate_opd) != 0:
            ok = False
    return 0 if ok else 1


def _run_phonons(
    irrep_label: str,
    poscar_path: str,
    name: str | None,
    born_charges_path: str | None = None,
    direction_args: list[str] | None = None,
    tolerance: float = 0.0,
    isodata: str | None = None,
    generate_opd: bool = False,
) -> int:
    try:
        structure = Structure.from_poscar(poscar_path)
    except (FileNotFoundError, PoscarFormatError) as exc:
        print(f"Failed to read {poscar_path}: {exc}", file=sys.stderr)
        return 1

    try:
        tag = config.resolve_tag(name, poscar_title=structure.comment)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        k_label = smodes_module.k_vector_label_from_irrep(irrep_label)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    smodes_path = Path(f"{tag}_smodes_{k_label}")
    if not smodes_path.is_file():
        print(
            f"{smodes_path} not found -- run `smodes-phonons irrep {irrep_label}` first "
            "to create the displacements and the smodes output.",
            file=sys.stderr,
        )
        return 1

    try:
        irrep = smodes_module.parse_smodes_output(smodes_path.read_text()).irrep(irrep_label)
    except (smodes_module.SmodesOutputError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if born_charges_path is not None and irrep.k_label != "GM":
        print(
            f"--born-charges only applies to Gamma-point irreps (k = {irrep.k_label}) -- LO-TO "
            "splitting is only meaningful as q -> 0.",
            file=sys.stderr,
        )
        return 1

    # Same auto-detection convention as --kpoints/--incar: an explicit
    # --born-charges always wins, but if it wasn't given, BEC/OUTCAR (where
    # the LEPSILON=.TRUE. calculation's own output is meant to stay -- it's
    # never moved out of BEC/, matching `bec`'s/`irrep --bec`'s own written
    # BEC/POSCAR sitting there) is picked up with no flag needed at all.
    # WHETHER to create BEC/ in the first place is the physics judgment
    # call gated behind `--bec`/`bec` (LO-TO splitting only applies to an
    # insulator/semiconductor, never a metal, and this tool can't tell the
    # two apart from a bare POSCAR) -- but once BEC/OUTCAR actually exists,
    # that judgment call has already been made, by the same real symmetry
    # check `irrep --bec` reports per irrep (Gamma-point, IR-active); this
    # command re-derives the identical fact for free from the same smodes
    # output it already has to parse for the force-constant matrix itself,
    # so there is nothing further to re-confirm here. Only ever attempted
    # for a Gamma-point irrep -- LO-TO doesn't apply anywhere else, and a
    # leftover BEC/ from unrelated work must never surprise a non-Gamma
    # `phonons` call.
    bec_dir = Path(_DEFAULT_BEC_DIR)
    bec_outcar_missing = False
    if born_charges_path is None and irrep.k_label == "GM":
        try:
            born_charges_path = str(_find_outcar(bec_dir))
        except FileNotFoundError:
            if bec_dir.is_dir():
                # BEC/ was set up (by `bec` or `irrep --bec`) but never
                # actually run/populated -- a clear, distinct message from
                # "nothing was ever set up at all" below.
                bec_outcar_missing = True

    irrep_dir = _resolve_irrep_dir(irrep_label)
    unready = _describe_unready_forces(irrep_label, irrep_dir, irrep.n_modes)
    if unready is not None:
        print(unready, file=sys.stderr)
        return 1

    mode_forces = []
    for i in range(irrep.n_modes):
        try:
            outcar_path = _find_outcar(irrep_dir / f"Mode_{i}")
            mode_forces.append(vasp_module.parse_outcar_forces(outcar_path))
        except (FileNotFoundError, vasp_module.VaspOutputError) as exc:
            print(f"{irrep_label}: {exc}", file=sys.stderr)
            return 1

    undistorted_forces = None
    try:
        undistorted_outcar = _find_outcar(irrep_dir / "Undistorted")
    except FileNotFoundError:
        undistorted_outcar = None
    if undistorted_outcar is not None:
        try:
            undistorted_forces = vasp_module.parse_outcar_forces(undistorted_outcar)
        except vasp_module.VaspOutputError as exc:
            print(f"{undistorted_outcar}: {exc}", file=sys.stderr)
            return 1

    # LO-corrected results, one entry per symmetry-equivalent q_hat group.
    # Stays empty when there are no Born charges, which is the common case.
    lo_groups: list = []
    try:
        result = phonons_module.diagonalize_phonons(irrep, mode_forces, undistorted_forces)
    except phonons_module.PhononError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if born_charges_path is None:
        if bec_outcar_missing:
            print(
                f"{bec_dir}/OUTCAR not found -- cannot calculate LO-TO splitting. Run the real "
                f"LEPSILON=.TRUE. calculation in {bec_dir}/ (see `bec`), leaving the resulting "
                f"OUTCAR there (don't move it) -- reporting uncorrected frequencies only for now.",
                file=sys.stderr,
            )
        elif irrep.k_label == "GM" and irrep.ir_active:
            print(
                f"note: {irrep_label} is IR-active -- if this material is an insulator or "
                f"semiconductor, run `smodes-phonons bec` (or `irrep {irrep_label} --bec`) to set "
                f"up {bec_dir}/, run LEPSILON=.TRUE. there, and leave the resulting OUTCAR in "
                f"{bec_dir}/, to include Gamma-point LO-TO splitting. Not generated automatically: "
                "LO-TO splitting isn't physically meaningful for a metal (free carriers screen "
                "the macroscopic field it depends on). Or pass --born-charges PATH to point at "
                "data elsewhere.",
                file=sys.stderr,
            )
    elif not irrep.ir_active:
        print(
            f"{irrep_label} is IR-inactive by symmetry -- LO-TO splitting cannot occur regardless "
            "of Born charges. Reporting uncorrected frequencies only.",
            file=sys.stderr,
        )
    else:
        try:
            born_charges = vasp_module.parse_born_effective_charges(born_charges_path)
            dielectric_tensor = vasp_module.parse_dielectric_tensor(born_charges_path)
        except (FileNotFoundError, vasp_module.VaspOutputError) as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if born_charges.shape[0] != irrep.n_atoms:
            # A real, confirmed HPC case: `BEC/POSCAR` is the plain PARENT
            # structure (see _write_bec_directory), but a Gamma-point
            # irrep's own native cell is not always identical to the
            # parent's -- confirmed for real TiO2 (body-centered, I4_1/amd):
            # every Gamma-point irrep's own native cell is the true 6-atom
            # PRIMITIVE cell, half of POSCAR_TiO2's own 12-atom CONVENTIONAL
            # description, contradicting this project's own long-assumed
            # "a Gamma-point irrep's superlattice multiplier is always
            # [1,1,1], so its atoms are exactly the parent's own" shortcut
            # (true for STO/BaTiO3/CMO/HfO2, all Gamma-primitive parents in
            # their own real fixtures, but not general). When the Born
            # charges instead match the PARENT's own atom count exactly,
            # reduce them onto the irrep's own native ordering via the same
            # (now origin-shift-aware) primitive `--magmom` tiling already
            # uses for the opposite (parent-smaller) direction -- a real,
            # nonzero origin shift between SMODES's own native atom
            # reconstruction and the parent POSCAR is a confirmed, real
            # possibility here too (see `map_parent_values_to_native`'s own
            # docstring), not just a same-order subset selection.
            if born_charges.shape[0] == structure.n_atoms and structure.n_atoms != irrep.n_atoms:
                try:
                    born_charges = np.stack(
                        supercell_module.map_parent_values_to_native(irrep, structure, list(born_charges))
                    )
                except supercell_module.SupercellError as exc:
                    print(f"{born_charges_path}: {exc}", file=sys.stderr)
                    return 1
                print(
                    f"note: {born_charges_path} covers the parent structure's own {structure.n_atoms} "
                    f"atom(s) -- reduced onto {irrep_label}'s own {irrep.n_atoms}-atom native cell "
                    "before use.",
                    file=sys.stderr,
                )
            else:
                print(
                    f"{born_charges_path}: Born charges cover {born_charges.shape[0]} atom(s), but "
                    f"{irrep_label} has {irrep.n_atoms} (parent has {structure.n_atoms}) -- must be "
                    "from a DFPT run on the same (Gamma-point) parent structure, in the same atom order.",
                    file=sys.stderr,
                )
                return 1

        volume = float(abs(np.linalg.det(irrep.superlattice)))
        mode_charges = lo_to_module.mode_effective_charges(irrep.modes, result.mode_basis_eigenvectors, born_charges)
        eps0, unstable_excluded = lo_to_module.static_dielectric_tensor(
            mode_charges, result.dynamical_matrix_eigenvalues, dielectric_tensor, volume
        )

        try:
            custom_directions = [_parse_lo_to_direction(d) for d in (direction_args or [])]
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1

        candidate_directions = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
        candidate_labels = ["x", "y", "z"]
        for d, raw in zip(custom_directions, direction_args or []):
            candidate_directions.append(d)
            candidate_labels.append(raw)

        findsym_path = Path(f"{tag}_findsym")
        try:
            if not findsym_path.is_file():
                _build_and_save_findsym_log(structure, tag, tolerance, isodata)
            findsym_log_text = findsym_path.read_text()
        except (FileNotFoundError, RuntimeError, findsym_module.FindsymLogError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

        # ISOTROPY's point group, in the parent's frame. Those matrices act on
        # COLUMN vectors; `group_equivalent_directions` takes the row-vector
        # form (`f @ R`), hence the transpose.
        try:
            rotations = isotropy_basis_module.point_group_rotations(
                structure, isotropy_basis_module.space_group_from_log(findsym_log_text), findsym_log_text
            )
        except (isotropy_basis_module.IsotropyBasisError, ValueError) as exc:
            print(f"LO-TO: could not get the crystal's point group from ISOTROPY ({exc})", file=sys.stderr)
            return 1
        groups = lo_to_module.group_equivalent_directions(
            candidate_directions, [(R.T, None) for R in rotations], structure.lattice
        )

        groups_report = []
        for group in sorted(groups, key=min):
            rep_direction = candidate_directions[group[0]]
            labels = ", ".join(candidate_labels[i] for i in sorted(group))
            try:
                correction = lo_to_module.non_analytic_correction(
                    irrep.modes, born_charges, dielectric_tensor, rep_direction, volume
                )
            except lo_to_module.LoToError as exc:
                print(f"  direction {labels}: {exc}", file=sys.stderr)
                continue
            corrected = phonons_module.diagonalize_phonons(
                irrep, mode_forces, undistorted_forces, additional_force_constant_correction=correction
            )
            groups_report.append((labels, corrected))
        lo_groups = groups_report

        bec_report = "\n\n".join(
            [
                _format_born_charges_report(irrep.atom_species, born_charges),
                _format_static_dielectric_report(eps0, unstable_excluded),
                _format_lo_to_report(irrep_label, result, groups_report),
            ]
        )
        print(bec_report)
        if bec_dir.is_dir():
            bec_report_path = bec_dir / "report.txt"
            bec_report_path.write_text(bec_report + "\n")
            print(f"Wrote {bec_report_path}")

    if generate_opd and irrep.degeneracy > 1:
        _write_eigenvector_opd_report(
            irrep, irrep_label, structure, result, irrep_dir, tag, isodata,
        )

    # The post-DFT direction report is `--generate-opd` (opd_eigenvectors.txt).
    print(_format_phonon_summary(irrep_label, result))
    report_path = irrep_dir / "phonons.txt"
    report_path.write_text(
        _format_phonon_report(irrep_label, result, lo_groups)
    )
    print(f"Wrote {report_path}")

    npz_path = irrep_dir / "phonons.npz"
    phonons_module.save_phonon_result(result, npz_path)
    print(f"Wrote {npz_path}")

    try:
        multipliers = supercell_module.nice_supercell_matrix(irrep.superlattice, structure.lattice)
        displacement_sets = {
            "force_constant": result.force_constant_eigenvectors,
            "dynamical_matrix": result.dynamical_matrix_eigenvectors,
            "atomic_displacement": result.atomic_displacement_eigenvectors,
        }
        big_structure, big_displacement_sets = supercell_module.tile_onto_supercell(
            irrep, structure.lattice, multipliers, displacement_sets, comment=f"{tag} {irrep_label} superlattice"
        )
    except supercell_module.SupercellError as exc:
        print(f"Warning: could not map results onto a nice supercell: {exc}", file=sys.stderr)
        return 0

    superlattice_path = irrep_dir / "POSCAR_superlattice"
    big_structure.to_poscar(superlattice_path)
    print(f"Wrote {superlattice_path} ({_format_supercell_matrix(multipliers)}, {big_structure.n_atoms} atoms)")

    superlattice_report_path = irrep_dir / "phonons_superlattice.txt"
    superlattice_report_path.write_text(
        _format_supercell_report(irrep_label, result, big_displacement_sets, multipliers)
    )
    print(f"Wrote {superlattice_report_path}")

    return 0




def _write_eigenvector_opd_report(
    irrep,
    irrep_label: str,
    parent: Structure,
    result,
    irrep_dir: Path,
    tag: str,
    isodata: str | None,
) -> Path | None:
    """Write `Irrep_<label>/opd_eigenvectors.txt` -- the OPDs expressed in
    terms of the real DFT eigenvectors rather than SMODES's basis modes.

    A CHANGE OF BASIS. `opd.opd_bases_for_eigenvectors` needs no symmetry
    search of its own: the OPD operation is linear, so an eigenvector's OPD
    is the same linear combination of the basis modes' OPDs. All this
    function does is build the basis, hand over the right coefficients, and
    format the result.

    THE COEFFICIENTS ARE `dyn_vec / sqrt(masses)`, not `dyn_vec`. `eigh`
    runs on the MASS-WEIGHTED force-constant matrix, so its raw eigenvectors
    are not displacement amplitudes until the mass factor is divided out --
    exactly what `phonons._reconstruct_eigenvectors` is given to build
    `atomic_displacement_eigenvectors`. Reusing that same input is what
    keeps this consistent with the eigenvectors already in `phonons.txt`
    rather than a second, subtly different reconstruction.

    THIS IS WHY --generate_opd EXISTS AS A FLAG. `phonons` must never
    trigger a findsym subprocess silently -- an earlier version that did
    took the suite from ~20s to ~157s. Building the OPD basis can escalate
    to findsym (fallback-only, but it can), and since the basis is
    deliberately NOT cached between commands, it must be rebuilt here. An
    explicit flag is what makes that cost requested rather than imposed,
    the same argument `irrep --opd` already makes.

    Non-fatal, like every other supplementary report: the frequencies and
    eigenvectors are the primary output and are already written by the time
    this runs.
    """
    findsym_path = Path(f"{tag}_findsym")
    findsym_log_text = findsym_path.read_text() if findsym_path.is_file() else None

    try:
        bases = opd_module.build_opd_bases(
            [irrep], parent, findsym_log_text=findsym_log_text,
        )
        by_mode = {mode: basis for (_label, mode), basis in bases.items()}
        sqrt_masses = np.sqrt([
            phonons_module.mass_for_species(mode.species) for mode in irrep.modes
        ])
        coefficients = result.mode_basis_eigenvectors / sqrt_masses[:, np.newaxis]
        eigen_bases = opd_module.opd_bases_for_eigenvectors(by_mode, coefficients)
    except Exception as exc:  # noqa: BLE001 -- same broad guard as _write_opd_report
        print(
            f"{irrep_label}: could not express the order-parameter directions in terms of the "
            f"DFT eigenvectors: {type(exc).__name__}: {exc}\n"
            "  The frequencies and eigenvectors above are unaffected.",
            file=sys.stderr,
        )
        return None

    lines = [
        f"Order-parameter directions for Irrep_{irrep_label}, in terms of the real DFT "
        "eigenvectors",
        "",
        "Each block below is ONE phonon eigenvector (not one SMODES basis mode), with every",
        "order-parameter direction that eigenvector has. Position labels match",
        f"Irrep_{irrep_label}/opd.txt and the `--direction` argument exactly -- a position names",
        "a direction, never the mechanism or the basis it was expressed in.",
        "",
        "These are UNIT vectors (Frobenius norm 1 over the whole n_atoms x 3 array), the same",
        "convention as phonons.txt's own eigenvectors -- they give the SHAPE of the distortion,",
        "not a displacement in Angstrom. Multiply by an amplitude to get real displacements;",
        "`freeze --amplitude` does exactly that.",
        "",
    ]
    for eigen_index in sorted(eigen_bases):
        frequency = result.frequencies_cm[eigen_index]
        lines.append(f"=== Eigenvector {eigen_index}  ({frequency:.4f} cm^-1) ===")
        lines.append("")
        lines.extend(opd_module.format_opd_basis(eigen_bases[eigen_index]))
        lines.append("")

    out_path = irrep_dir / "opd_eigenvectors.txt"
    out_path.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Wrote {out_path}")
    return out_path


_OPD_REPORT_PREAMBLE = [
    "Every direction below is a real, physically realisable distortion of this",
    "crystal. SMODES prints ONE pattern per mode; every component listed here is",
    "constructed from it with ISOTROPY's own irrep matrices (a projection over the",
    "crystal's symmetry operations), so the positions are ISOTROPY's components, in",
    "ISOTROPY's order -- the same letters ISOTROPY and ISODISTORT use.",
    "",
    "No DFT data is involved -- this is pure symmetry, available before any",
    "calculation is run. Use a position's own `--direction` string to freeze or",
    "decompose along it.",
    "",
]


def _write_opd_report(
    irreps: list,
    parent: Structure,
    out_path: Path,
    heading: str,
    findsym_log_text: str,
    isodata: str | None,
) -> Path | None:
    """Write the OPD report for `irreps` -- one irrep (its own cell) or every
    irrep at a k-point (the shared cell). Returns the path, or `None` if the
    basis could not be built.

    NEVER RAISES, AND NEVER RETURNS NON-ZERO. This is a secondary report; the
    `Mode_i/POSCAR` files are the primary product and are already written by
    the time this runs. Real BaTiO3's L point raises `FreezeError` today, so
    an unguarded call here would abort `irrep L` BEFORE the POSCARs existed --
    destroying what the user actually needs in order to protect a convenience.
    Same "warn, never block the primary output" contract `--kpoints`/`--incar`/
    `--bec` already follow.

    The guard is deliberately broad (`Exception`) rather than a named list.
    The OPD construction reaches through `build_joint_opd_cell` into the
    findsym/smodes/supercell/freeze stack, which raises at least
    `FreezeError`, `SupercellError`, `IsotropyBasisError`, `SmodesOutputError`,
    `FindsymLogError`, `OSError` and `numpy.linalg.LinAlgError` -- enumerating
    them would be a list that silently goes stale, and any escape at all
    defeats the entire purpose of the guard.
    """

    def _build(subset):
        return opd_module.build_opd_bases(
            subset, parent, findsym_log_text=findsym_log_text,
        )

    bases: dict = {}
    failures: list = []
    shared_cell = True
    try:
        bases = _build(irreps)
    except Exception as exc:  # noqa: BLE001 -- see docstring
        # TIER 2: the requested set could not share one cell. Retry each
        # irrep alone -- distinct irreps are orthogonal, so a per-irrep
        # answer is still correct, it just cannot be compared ACROSS irreps
        # in one frame. Confirmed to be a genuinely different outcome from a
        # blanket failure: real BaTiO3 L fails at BOTH tiers (its arms will
        # not align even for a single irrep), while the shared-cell-only
        # failure mode is what the SMODES orientation split produces.
        shared_cell = False
        if len(irreps) > 1:
            print(
                f"Could not place all {len(irreps)} irrep(s) at this k-point on one shared "
                f"cell ({type(exc).__name__}: {exc}). Retrying each irrep on its own cell.",
                file=sys.stderr,
            )
        for irrep in irreps:
            try:
                bases.update(_build([irrep]))
            except Exception as inner:  # noqa: BLE001 -- see docstring
                failures.append((irrep, f"{type(inner).__name__}: {inner}"))

    if failures:
        print(
            "Could not construct the order-parameter directions for "
            + ", ".join(irrep.label for irrep, _ in failures)
            + ".\n  Their native SMODES direction is reported instead; every other direction "
            "of those irreps is unavailable.\n"
            "  The displaced POSCARs above are unaffected and ready to run.",
            file=sys.stderr,
        )
    if not bases and not failures:
        return None

    lines = [heading, ""] + list(_OPD_REPORT_PREAMBLE)
    if not shared_cell and bases:
        lines.append(
            "NOTE: these irreps could not be placed on ONE shared cell, so each is reported "
            "on its own. Every amplitude below is still correct -- distinct irreps are "
            "orthogonal, so one irrep's own directions never depend on another's. What is "
            "unavailable is comparing or combining them in a single frame."
        )
        lines.append("")

    for _key, basis in sorted(bases.items(), key=lambda item: (item[0][0], item[0][1])):
        lines.extend(opd_module.format_opd_basis(basis))
        lines.append("")
    for irrep, reason in failures:
        lines.extend(opd_module.format_native_only(irrep, reason))
        lines.append("")

    incomplete = [b for b in bases.values() if not b.is_complete]
    if incomplete:
        lines.append(
            f"{len(incomplete)} of {len(bases)} (irrep, mode) bases did NOT span their own "
            "degenerate subspace -- see the WARNING lines above. A distortion along a missing "
            "direction cannot be represented by `freeze` or detected by `decomposition`."
        )
    if failures:
        lines.append(
            f"{len(failures)} irrep(s) could not be constructed at all and are reported with "
            "their native SMODES direction only: "
            + ", ".join(irrep.label for irrep, _ in failures) + "."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines).rstrip() + "\n")
    print(f"Wrote {out_path}")
    return out_path


def _parse_label_value_pairs(pairs: list[str], value_type, flag_name: str, single_label: str | None = None) -> dict:
    """Parse repeated '[LABEL=]VALUE' arguments. A bare VALUE (no '=') is
    only allowed when `single_label` is given (i.e. exactly one irrep was
    requested overall) -- it then applies to that one irrep, since there's
    nothing else it could mean. With more than one irrep requested,
    `single_label` is None and a bare VALUE is rejected as ambiguous.
    """
    result = {}
    for pair in pairs:
        if "=" not in pair:
            if single_label is None:
                raise ValueError(
                    f"{flag_name} {pair!r} must be of the form LABEL=VALUE (more than one irrep was requested, "
                    "so a bare value is ambiguous)"
                )
            label, raw_value = single_label, pair
        else:
            label, _, raw_value = pair.partition("=")
            label = label.upper()  # irrep labels are matched case-insensitively everywhere
        try:
            result[label] = value_type(raw_value)
        except ValueError as exc:
            raise ValueError(f"{flag_name} {pair!r}: {exc}") from exc
    return result


#: The ONLY letter the symbolic grammar defines. It is a pure placeholder for
#: "one unit of the order parameter" -- the real Angstrom scale always comes
#: from a separate knob (`--amplitude`, or a swept grid point's own value).
#:
#: `b` and `c` USED to exist as fixed relative weights (1.2 and 0.8) and were
#: REMOVED. Two reasons. They could only express two ratios, which is what made
#: ISOTROPY's irrational strata untypeable (Issue 10: real graphite H1 needs
#: -0.577a, H2 needs 1.732a, K5 0.577a, CsV3Sb5 K6 -1.732a). And they actively
#: CONFLICTED with ISOTROPY's own notation, where `(a,b,c)` means three
#: INDEPENDENT free parameters rather than fixed multiples of one -- so our
#: `a,b,a` and their `(a,b,a)` looked alike and meant different things, which
#: muddied every label comparison in the v1.x label-comparison notes.
#:
#: An explicit coefficient says what it means: `a,2a,0.5a,-a`.
_SYMBOLIC_UNIT_LETTER = "a"

#: `[coefficient]a`, e.g. `a`, `2a`, `0.5a`, `-a`, `-1.77a`. The letter is
#: MANDATORY: a bare number is rejected, because `2` reads like a position
#: index rather than a weight.
_SYMBOLIC_TOKEN_RE = re.compile(r"^([+-]?(?:\d+\.?\d*|\.\d+)?)([a-z])$")


def _configure_iso_runner() -> None:
    """Give `isotropy_basis` its `iso` runner (v2.0 OPD basis), with an on-disk
    cache: `iso`'s output depends only on the script, so each (space group,
    irrep) query runs once per machine. A runner already set (the tests' saved
    fixtures) is left alone."""
    if isotropy_basis_module.runner_configured():
        return
    cache = Path(os.environ.get("SMODES_ISO_CACHE", Path.home() / ".cache" / "smodes_phonons" / "iso"))

    def runner(script: str) -> str:
        path = cache / f"{isotropy_basis_module.script_key(script)}.txt"
        if path.is_file():
            return path.read_text()
        out = iso_module.run_iso(script + "quit\n" * 3)
        if "Isotropy, Version" in out:        # cache only genuine iso output
            try:
                cache.mkdir(parents=True, exist_ok=True)
                path.write_text(out)
            except OSError:
                pass
        return out

    isotropy_basis_module.set_iso_runner(runner)


@dataclass(frozen=True)
class _IsotropyCell:
    """Returned in place of the old `components` list by the v2.0 builder:
    for each atom of its cell, the index of the parent atom it copies -- so
    MAGMOM maps atom by atom (`parent_magmom[parent_index]`), whatever the
    cell, including one smaller than a centred conventional parent."""

    parent_index: tuple


def _build_symbolic_irrep_patterns_isotropy(irrep_labels, symbolic_specs, loaded_by_label, structure, full_group_log,
                                            allow_centring: bool = True):
    """v2.0 path of `_build_symbolic_irrep_patterns`: each irrep's OPD basis is
    CONSTRUCTED from ISOTROPY's irrep matrices (isotropy_basis.joint_basis),
    seeded with the loaded pattern (SMODES mode or DFT eigenvector). Same
    return shape as the old path."""
    label_coefficients: dict = {}
    seeded = []
    for label in irrep_labels:
        irrep, pattern, _mode_index = loaded_by_label[label]
        try:
            label_coefficients[label] = _parse_symbolic_direction_spec(
                symbolic_specs[label], "--direction", irrep.degeneracy)
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return None
        seeded.append((irrep, [pattern]))
    try:
        joint = isotropy_basis_module.joint_basis(
            seeded, structure, isotropy_basis_module.resolve_space_group([irrep for irrep, _s in seeded], full_group_log), full_group_log)
    except (isotropy_basis_module.IsotropyBasisError, freeze_module.FreezeError,
            supercell_module.SupercellError, FileNotFoundError, RuntimeError) as exc:
        print(f"OPD basis (ISOTROPY matrices): {exc}", file=sys.stderr)
        return None
    per_irrep_pattern = {}
    for label in irrep_labels:
        W = joint.bases[label][0]
        per_irrep_pattern[label] = sum(c * W[position] for position, c in label_coefficients[label].items())
    mechanisms_by_label = {label: joint.position_labels(label) for label in irrep_labels}
    # Write the SMALLEST cell the requested distortion needs -- spanned by every
    # parent translation that leaves each requested OPD unchanged (ISOTROPY's
    # translation matrices). Building on the full period and leaving the
    # reduction to findsym loses accuracy: real CMO X2+ at 0.001 A moves 32
    # atoms by 1.8e-4 A each, at findsym's 1e-4 A reduction tolerance, and the
    # reduced POSCAR kept only 93% of the distortion (known issue 5d in the developer notes).
    vectors = {}
    for label in irrep_labels:
        v = np.zeros(loaded_by_label[label][0].degeneracy)
        for position, c in label_coefficients[label].items():
            v[position] = c
        vectors[label] = v
    try:
        cell = isotropy_basis_module.invariant_cell(
            {label: joint.representations[label] for label in irrep_labels}, vectors, joint.multipliers,
            allow_centring=allow_centring, lattice=joint.parent.lattice)
        n_base = joint.reference.n_atoms
        reference, parent_index, per_irrep_pattern = isotropy_basis_module.fold_onto_cell(
            joint.reference, joint.multipliers, joint.parent, cell, per_irrep_pattern)
        # Keep `--amplitude`'s meaning: the displacement norm of the structure
        # WRITTEN (what the old freeze did, normalising on the cell it wrote).
        # Folding onto the smaller cell scales a norm by sqrt(n_small/n_base).
        scale = float(np.sqrt(n_base / reference.n_atoms))
        per_irrep_pattern = {label: scale * pattern for label, pattern in per_irrep_pattern.items()}
    except isotropy_basis_module.IsotropyBasisError as exc:
        print(f"OPD basis (ISOTROPY matrices): {exc}", file=sys.stderr)
        return None
    return (reference, per_irrep_pattern, _IsotropyCell(tuple(int(i) for i in parent_index)), label_coefficients,
            full_group_log, mechanisms_by_label)


def _parse_symbolic_direction_spec(spec: str, flag_name: str, degeneracy: int) -> dict:
    """'a,0,0' -> {0: 1.0}; 'a,a' -> {0: 1.0, 1: 1.0}; 'a,b' -> {0: 1.0, 1: 1.2}.

    ISODISTORT-style symbolic direction notation for `freeze --direction`,
    sized to the irrep's own reported `degeneracy` (star arms, within-arm
    components, or both). Each position is one of ISOTROPY's components, in
    ISOTROPY's order -- not necessarily a literal x/y/z axis.

    A coefficient may be written on the unit letter: `a`, `2a`, `0.5a`,
    `-a`, `-1.77a`. Coefficients are RELATIVE weights only -- never an
    absolute scale; a separate, explicit knob (--amplitude, or each swept
    grid point's own value) always supplies the real Angstrom magnitude.
    `0` and `0a` both mean an unpopulated position.

    Returns `{position: coefficient}` (0-indexed, ISOTROPY's component
    order) for every NONZERO
    position -- a position given as literal `'0'` is simply omitted.
    """
    # SEMICOLONS SEPARATE STAR ARMS, commas separate positions within one
    # arm (and quadrature partners, which live on the same cell as their
    # native). That is `iso`'s own notation and it is what `irrep --opd`
    # prints -- verified against real iso 9.6.1, where TiO2 X1 comes back
    # `(a,0;0,0)`.
    #
    # Both are accepted here and mean the same thing to the parser: a
    # position is a position, and the arm grouping is presentational. Until
    # now only commas parsed, so `irrep --opd` was printing a syntax
    # (`a;0;0` for real STO M2-) that `freeze --direction` rejected outright
    # as "1 position" -- the tool telling a user a form of the flag it would
    # not itself accept.
    tokens = [t.strip() for group in spec.split(";") for t in group.split(",")]
    if len(tokens) != degeneracy:
        raise ValueError(
            f"{flag_name} {spec!r} has {len(tokens)} position(s), but this irrep's own degeneracy is "
            f"{degeneracy} -- a symbolic direction needs exactly one position per degenerate direction "
            "(see `irrep --opd` or `phonons --generate-opd` for what each position is)"
        )
    result = {}
    for i, token in enumerate(tokens):
        if token in ("0", "0a"):
            continue
        match = _SYMBOLIC_TOKEN_RE.match(token)
        if match is None or match.group(2) != _SYMBOLIC_UNIT_LETTER:
            # A bare number is the likeliest mistake and gets its own sentence,
            # since `2` reads like a position index rather than a weight.
            hint = (
                f" -- a coefficient needs its letter, so write {token}a rather than {token}"
                if re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)", token) else ""
            )
            raise ValueError(
                f"{flag_name} {spec!r}: {token!r} is not a recognized symbolic token{hint}. Expected "
                "'0', or a coefficient on the unit letter 'a' -- 'a', '2a', '0.5a', '-a', '-1.77a'. "
                "Letters are RELATIVE weights; the absolute scale comes from --amplitude."
            )
        coefficient = float(match.group(1)) if match.group(1) not in (None, "", "+", "-") else (
            -1.0 if match.group(1) == "-" else 1.0
        )
        if coefficient == 0.0:
            continue
        result[i] = coefficient
    if not result:
        raise ValueError(f"{flag_name} {spec!r} is the zero vector -- no direction to construct")
    return result


def _v2_default_direction_args(irrep_labels, loaded_by_label, direction_args):
    """No `--direction` means ISOTROPY's first component (`a,0,...,0`).
    Unchanged when `--direction` was given."""
    if direction_args:
        return direction_args
    out = []
    for label in irrep_labels:
        degeneracy = loaded_by_label[label][0].degeneracy
        spec = ",".join(["a"] + ["0"] * (degeneracy - 1))
        out.append(spec if len(irrep_labels) == 1 else f"{label}={spec}")
    return out


def _resolve_symbolic_direction_specs(irrep_labels: list[str], direction_args: list[str]) -> dict | None:
    """Resolve `--direction` arguments (always symbolic `a/b/c` notation --
    numeric `X,Y,Z` notation was removed) into `{label: raw_spec}`, one
    entry required for every requested irrep. A single requested irrep may
    give a bare spec with no `LABEL=` prefix (`--direction a,a`); with more
    than one irrep, every entry must be `LABEL=spec` and every irrep must
    be covered -- there is no well-defined "default" contribution once
    several irreps are blended onto one shared, arm-rotated joint cell
    from the start (unlike a within-arm-only rotation, which always has a
    natural "just use the native direction" fallback).

    Prints its own clear error and returns `None` on any problem -- unlike
    the old numeric-vs-symbolic detector this replaces, there is no longer
    a fallback path to silently defer to, so failures are reported
    directly here rather than deferred to a caller that no longer exists.
    """
    if len(irrep_labels) == 1 and len(direction_args) == 1 and "=" not in direction_args[0]:
        return {irrep_labels[0]: direction_args[0]}
    specs: dict = {}
    for raw in direction_args:
        if "=" not in raw:
            print(
                f"--direction {raw!r} must be of the form LABEL=a,a (more than one irrep was "
                "requested, so a bare value is ambiguous)",
                file=sys.stderr,
            )
            return None
        label, _, spec = raw.partition("=")
        label = label.upper()
        if label not in irrep_labels:
            print(f"--direction {raw!r}: {label!r} is not one of the requested irreps", file=sys.stderr)
            return None
        if label in specs:
            print(f"--direction given more than once for {label}", file=sys.stderr)
            return None
        specs[label] = spec
    missing = [label for label in irrep_labels if label not in specs]
    if missing:
        print(
            f"--direction must be given for every requested irrep -- missing: {', '.join(missing)}",
            file=sys.stderr,
        )
        return None
    return specs


def _resolve_freeze_mode_index(label: str, irrep, result, override: int | None) -> int:
    n_modes = irrep.n_modes
    if override is not None:
        if not 0 <= override < n_modes:
            raise freeze_module.FreezeError(
                f"{label}: --mode index {override} out of range (irrep has {n_modes} mode(s))"
            )
        return override

    if n_modes == 1:
        return 0

    if not sys.stdin.isatty():
        raise freeze_module.FreezeError(
            f"{label} has {n_modes} modes and stdin is not a terminal -- pass --mode {label}=INDEX"
        )

    print(f"{label}: choose a mode")
    for i in range(n_modes):
        print(f"  {i}: {result.frequencies_cm[i]:.3f} cm^-1")
    choice = input(f"Mode index for {label} [0-{n_modes - 1}]: ").strip()
    try:
        idx = int(choice)
    except ValueError as exc:
        raise freeze_module.FreezeError(f"{label}: {choice!r} is not a valid mode index") from exc
    if not 0 <= idx < n_modes:
        raise freeze_module.FreezeError(f"{label}: mode index {idx} out of range (0-{n_modes - 1})")
    return idx


def _check_no_duplicate_irreps(labels: list[str]) -> None:
    seen = set()
    for label in labels:
        if label in seen:
            raise ValueError(f"{label} was requested more than once -- an irrep cannot be combined with itself")
        seen.add(label)


def _load_freeze_component(label: str, tag: str, mode_override: int | None,
                           smodes_basis: bool = False):
    """Resolve one irrep's already-computed mode for freeze/freeze-amplitude(-grid).

    Returns (irrep, displacement, mode_index) on success. On failure, prints a
    clear error to stderr and returns None -- callers should treat that as
    "return 1".
    """
    try:
        k_label = smodes_module.k_vector_label_from_irrep(label)
    except ValueError as exc:
        print(f"{label}: {exc}", file=sys.stderr)
        return None

    irrep_dir = _resolve_irrep_dir(label)
    npz_path = irrep_dir / "phonons.npz"
    if not smodes_basis and not npz_path.is_file():
        print(
            f"{npz_path} not found -- run `smodes-phonons phonons {label}` first, "
            "or pass --smodes-basis to freeze along SMODES's own symmetry-adapted "
            "basis mode instead of a DFT eigenvector.",
            file=sys.stderr,
        )
        return None

    smodes_path = Path(f"{tag}_smodes_{k_label}")
    if not smodes_path.is_file():
        print(f"{smodes_path} not found -- run `smodes-phonons irrep {label}` first.", file=sys.stderr)
        return None

    try:
        irrep = smodes_module.parse_smodes_output(smodes_path.read_text()).irrep(label)
    except (smodes_module.SmodesOutputError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return None

    if smodes_basis:
        # SMODES's own symmetry-adapted basis mode, not a DFT eigenvector.
        #
        # A perfectly ordinary thing to freeze along, and the ONLY thing
        # available before any phonon calculation exists -- it is exactly
        # what `irrep` already writes into every `Mode_i/POSCAR`. Without
        # this, `freeze` cannot run at all until VASP has been run for every
        # irrep, which makes the whole irrep -> freeze -> decomposition loop
        # impossible to exercise or validate pre-DFT.
        #
        # `.normalized` to match what a DFT eigenvector is: the eigenvector
        # branch returns `atomic_displacement_eigenvectors`, which are unit
        # Frobenius-normalised, so `--amplitude` means the same thing on
        # either path. Using raw `.displacements` here would silently scale
        # every frozen structure by that mode's own arbitrary norm.
        #
        # EXPLICIT FLAG, never an automatic fallback: quietly switching
        # which basis is being frozen along, because a file happened to be
        # missing, would be a bad surprise for someone who does have phonon
        # data and simply mistyped a path.
        if mode_override is not None:
            idx = mode_override
        elif irrep.n_modes == 1:
            idx = 0
        else:
            print(
                f"{label}: --smodes-basis needs --mode, since this irrep has "
                f"{irrep.n_modes} modes and there are no frequencies to choose "
                "between them without phonon data.",
                file=sys.stderr,
            )
            return None
        if not 0 <= idx < irrep.n_modes:
            print(f"{label}: --mode {idx} out of range (irrep has {irrep.n_modes} mode(s))",
                  file=sys.stderr)
            return None
        return irrep, np.asarray(irrep.modes[idx].normalized, dtype=np.float64), idx

    result = phonons_module.load_phonon_result(npz_path)

    try:
        idx = _resolve_freeze_mode_index(label, irrep, result, mode_override)
    except freeze_module.FreezeError as exc:
        print(str(exc), file=sys.stderr)
        return None

    return irrep, result.atomic_displacement_eigenvectors[idx], idx


SMODES_BASIS_DEFAULT_AMPLITUDE = 0.01
"""Default amplitude for `freeze --smodes-basis`, in Angstrom.

DELIBERATELY NOT `displace.DISPLACEMENT_AMPLITUDE_ANGSTROM` (0.001), and
that constant is untouched: it is the finite-displacement amplitude every
force-constant calculation in this project is built on, defined in exactly
one place, and changing it would silently rescale every phonon frequency.

0.001 A is too small to survive a symmetry search. Fed to `decomposition`,
a structure frozen at that amplitude is reduced back to the parent cell and
reports ZERO on every irrep at 0.00% residual -- reading as "nothing is
here" rather than "I removed it" (see known issue 5d in the developer notes). Measured on real STO
M2-: findsym reports the frozen structure as space group 221, the
undistorted parent, and the displacement norm comes back 0.000000.

0.01 A is confirmed with the physicist as the right default here on physical
grounds too -- a real distortion is normally well above 0.001 A -- and real
STO round-trips cleanly at it across all 20 irreps at GM, M, R and X. It is
a default, not a floor: `--amplitude` overrides it as always.
"""


def _run_freeze(
    irrep_labels: list[str],
    poscar_path: str,
    name: str | None,
    mode_args: list[str],
    amplitude_args: list[str],
    direction_args: list[str] | None = None,
    isodata: str | None = None,
    kpoints_path: str = "KPOINTS",
    incar_path: str = "INCAR",
    magmom: bool = False,
    smodes_basis: bool = False,
) -> int:
    """`freeze <IRREP...>` -- combine one or more already-computed modes
    onto one shared, commensurate supercell.

    `--direction` (symbolic `a/b/c` notation only -- see `_parse_symbolic_
    direction_spec`) and `--amplitude` are independent knobs: the letters
    say how each irrep's own requested components are weighted RELATIVE to
    each other, `--amplitude LABEL=VALUE` (default `DISPLACEMENT_
    AMPLITUDE_ANGSTROM`) says how big that irrep's own combined pattern
    actually is. With no `--direction` at all, each irrep is frozen along
    ISOTROPY's first component (`a,0,...,0`).
    """
    try:
        structure = Structure.from_poscar(poscar_path)
    except (FileNotFoundError, PoscarFormatError) as exc:
        print(f"Failed to read {poscar_path}: {exc}", file=sys.stderr)
        return 1

    kmesh = _load_kpoints_mesh(kpoints_path)
    incar_template = _load_incar_template(incar_path, _INCAR_HINT_RELAX)

    parent_magmom = _parse_and_validate_magmom(magmom, incar_template, structure, incar_path, poscar_path)
    if parent_magmom is _MAGMOM_ERROR:
        return 1

    try:
        tag = config.resolve_tag(name, poscar_title=structure.comment)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    single_label = irrep_labels[0] if len(irrep_labels) == 1 else None
    try:
        mode_overrides = _parse_label_value_pairs(mode_args, int, "--mode", single_label)
        amplitude_overrides = _parse_label_value_pairs(amplitude_args, float, "--amplitude", single_label)
        _check_no_duplicate_irreps(irrep_labels)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    loaded_by_label = {}
    for label in irrep_labels:
        loaded = _load_freeze_component(label, tag, mode_overrides.get(label), smodes_basis=smodes_basis)
        if loaded is None:
            return 1
        loaded_by_label[label] = loaded

    default_amplitude = (
        SMODES_BASIS_DEFAULT_AMPLITUDE if smodes_basis
        else displace.DISPLACEMENT_AMPLITUDE_ANGSTROM
    )
    amplitudes = {
        label: amplitude_overrides.get(label, default_amplitude) for label in irrep_labels
    }

    direction_was_typed = bool(direction_args)
    direction_args = _v2_default_direction_args(irrep_labels, loaded_by_label, direction_args)
    symbolic_specs = _resolve_symbolic_direction_specs(irrep_labels, direction_args)
    if symbolic_specs is None:
        return 1
    built = _build_symbolic_irrep_patterns(
        irrep_labels, symbolic_specs, loaded_by_label, structure, tag, isodata,
        # Moments can break the centring, so a magnetic structure's cell uses
        # only whole lattice translations of the parent (never smaller).
        allow_smaller_anchor=parent_magmom is None,
    )
    if built is None:
        return 1
    (ref_structure, per_irrep_pattern, components, label_coefficients,
     full_group_log, mechanisms_by_label) = built

    problem = _check_symbolic_pattern_is_nonzero(
        per_irrep_pattern, label_coefficients, loaded_by_label
    )
    if problem is not None:
        print(problem, file=sys.stderr)
        return 1
    combined_pattern = sum(amplitudes[label] * per_irrep_pattern[label] for label in irrep_labels)
    combined = ref_structure.displaced(combined_pattern)

    dir_name = "Freeze_" + "_".join(
        _symbolic_dirname_part(label, loaded_by_label[label][2], label_coefficients[label])
        if direction_was_typed else f"{label}-m{loaded_by_label[label][2]}"
        for label in irrep_labels
    )
    freeze_dir = Path(dir_name)
    freeze_dir.mkdir(parents=True, exist_ok=True)

    # ISSUE 10 -- the authoritative record of what this directory holds.
    # The directory NAME is lossy by design; this is not.
    _write_direction_manifest(freeze_dir, [
        {
            "label": label,
            "mode": loaded_by_label[label][2],
            "amplitude": amplitudes[label],
            "spec": symbolic_specs[label],
            "rows": [
                (position + 1, coefficient,
                 mechanisms_by_label.get(label, [])[position]
                 if position < len(mechanisms_by_label.get(label, [])) else "?")
                for position, coefficient in sorted(label_coefficients[label].items())
            ],
            "norm": float(np.linalg.norm(
                amplitudes[label] * np.asarray(per_irrep_pattern[label], dtype=np.float64)
            )),
        }
        for label in irrep_labels
    ])

    # MAGMOM follows the written structure atom by atom: every atom of the
    # cell `freeze` wrote carries the index of the parent atom it came from.
    freeze_overrides: dict = {}
    tiled_magmom = None
    if parent_magmom is not None:
        tiled_magmom = np.asarray(parent_magmom, dtype=float)[list(components.parent_index)]
        freeze_overrides["MAGMOM"] = vasp_module.format_magmom(tiled_magmom)

    combined, reduce_note, superlattice_and_log, superlattice_magmom_overrides = (
        _reduce_freeze_structure_with_magmom(
            combined, f"{tag} {'_'.join(irrep_labels)} symbolic direction result", isodata, tiled_magmom,
            freeze_overrides,
        )
    )
    _write_freeze_reduction_byproducts(freeze_dir, superlattice_and_log)
    _write_superlattice_kpoints(kmesh, structure.lattice, superlattice_and_log, freeze_dir)
    if superlattice_and_log is not None:
        _write_incar(
            incar_template, superlattice_magmom_overrides or {}, freeze_dir, filename="INCAR_superlattice"
        )

    out_path = freeze_dir / "POSCAR"
    combined.to_poscar(out_path)
    print(f"Wrote {out_path} ({combined.n_atoms} atoms)")
    if reduce_note is not None:
        print(reduce_note)
    _write_scaled_kpoints(kmesh, structure.lattice, combined.lattice, freeze_dir)
    _write_incar(incar_template, freeze_overrides, freeze_dir)
    return 0




#: A symbolic `--direction` request whose combined pattern falls below this
#: FRACTION of what its own components carry has CANCELLED rather than
#: combined. Relative, not absolute: letters are relative weights and the
#: amplitude is a separate knob, so no absolute displacement is meaningful
#: here. 1e-6 sits far below any real physical near-cancellation and far above
#: the ~1e-16 floor of summing unit vectors.
_SYMBOLIC_CANCELLATION_FLOOR = 1e-6


def _check_symbolic_pattern_is_nonzero(per_irrep_pattern: dict, label_coefficients: dict,
                                       loaded_by_label: dict):
    """An error message if a symbolic `--direction` combined pattern CANCELLED
    to (near) zero, else None. Issue 6, and the backstop for Issue 3.

    A nonzero order parameter at a nonzero amplitude MUST lower the symmetry.
    A combined pattern of zero means the components cancelled, and freezing it
    writes the UNDISTORTED PARENT -- a structure that looks entirely normal,
    passes every downstream check, and silently answers a different question
    than the one asked.

    Not hypothetical. Real STO M5- mode 0 at `--direction a,a,a,a,a,a`
    produced exactly that: displacement norm 0.000000, space group 221, at
    both the reduced 5-atom and the pre-reduction 40-atom cell, with
    `decomposition` reporting every irrep below the noise floor. It was caught
    only because two other constructions disagreed (both summed to sqrt(6))
    -- by hand, long after the structures had been generated and used.

    The root cause is fixed (Issues 12d, 5 and 13; that same request now gives
    R32 at |sum| = sqrt(6)). This guard exists so the FAILURE MODE cannot
    return silently whatever causes it next. Measured as a fraction of the
    components' own summed weight, so it fires on cancellation and never on a
    legitimately small amplitude.
    """
    for label, pattern in per_irrep_pattern.items():
        coefficients = label_coefficients.get(label, {})
        scale = sum(abs(c) for c in coefficients.values()) or 1.0
        magnitude = float(np.linalg.norm(np.asarray(pattern, dtype=np.float64)))
        if magnitude > _SYMBOLIC_CANCELLATION_FLOOR * scale:
            continue
        mode_index = loaded_by_label[label][2] if label in loaded_by_label else "?"
        positions = ", ".join(str(i + 1) for i in sorted(coefficients))
        return (
            f"{label} mode {mode_index}: the requested order-parameter components CANCEL -- "
            f"positions {positions} combine to a pattern of norm {magnitude:.3e}, which is zero "
            "to within numerical noise. Freezing this would write the UNDISTORTED PARENT rather "
            "than a distorted structure, so nothing has been written. A nonzero order parameter "
            "at a nonzero amplitude must lower the symmetry; check the letter positions against "
            f"`irrep --opd {label}` if a different combination was intended."
        )
    return None


def _symbolic_coefficient_tag(coefficient: float) -> str:
    """One position's coefficient, encoded for a DIRECTORY NAME.

    A plain `a` (coefficient 1) stays `a`, so every directory this project has
    ever written for an all-`a` request keeps exactly the name it had. Anything
    else carries its number: `2a`, `0.5a`, `n1.77a`.

    `-` becomes `n` because `-` is already the field separator in the directory
    name, so `-4--1.77a` would be genuinely ambiguous. The decimal point is
    kept: it is legal in a filename on every platform this runs on, and `0p5a`
    is harder to read than `0.5a`.

    The directory name is a CONVENIENCE, not the record. Each freeze directory
    also carries `direction.txt` with the spec exactly as typed, so a name that
    is merely awkward can never be ambiguous about what was built.
    """
    if coefficient == 1.0:
        return _SYMBOLIC_UNIT_LETTER
    text = f"{coefficient:g}".replace("-", "n")
    return f"{text}{_SYMBOLIC_UNIT_LETTER}"


def _symbolic_dirname_part(label: str, mode_index: int, coefficients: dict) -> str:
    parts = []
    for i, c in coefficients.items():
        tag = _symbolic_coefficient_tag(c)
        # An explicit coefficient is separated from the position by `_`, or
        # position 2 with coefficient 2 would render `22a` and read as
        # coefficient 22. A plain `a` needs no separator, which is what keeps
        # every existing all-`a` directory name byte-identical.
        parts.append(f"-{i + 1}{tag}" if tag == _SYMBOLIC_UNIT_LETTER else f"-{i + 1}_{tag}")
    return f"{label}-m{mode_index}-sym" + "".join(parts)


def _build_symbolic_irrep_patterns(
    irrep_labels: list[str],
    symbolic_specs: dict,
    loaded_by_label: dict,
    structure: Structure,
    tag: str,
    isodata: str | None,
    allow_smaller_anchor: bool = True,
):
    """Shared construction step behind every symbolic `--direction` command
    (`freeze`, `freeze-amplitude`, `landau-grid`): the parent's findsym log
    (built if missing), then `_build_symbolic_irrep_patterns_isotropy` --
    the ISOTROPY-matrix basis, the requested letter-weighted pattern per
    irrep, folded onto the smallest cell the distortion needs.

    Returns that function's tuple on success, or `None` (after printing a
    clear error). Letters mean relative weight only; the caller applies
    each irrep's amplitude.
    """
    findsym_path = Path(f"{tag}_findsym")
    try:
        if not findsym_path.is_file():
            _build_and_save_findsym_log(structure, tag, 0.0, isodata)
        full_group_log = findsym_path.read_text()
    except (FileNotFoundError, RuntimeError, findsym_module.FindsymLogError) as exc:
        print(str(exc), file=sys.stderr)
        return None

    return _build_symbolic_irrep_patterns_isotropy(
        irrep_labels, symbolic_specs, loaded_by_label, structure, full_group_log,
        allow_centring=allow_smaller_anchor)


_MAX_FREEZE_AMPLITUDE_GRID_IRREPS = 3


def _run_freeze_amplitude_grid(
    irrep_labels: list[str],
    start: float,
    end: float,
    steps: int,
    poscar_path: str,
    name: str | None,
    mode_overrides: dict,
    direction_args: list[str] | None = None,
    isodata: str | None = None,
    column_labels: list[str] | None = None,
    column_header_suffix: str = "_amp_A",
    kpoints_path: str = "KPOINTS",
    incar_path: str = "INCAR",
    magmom: bool = False,
) -> int:
    """Sweep 1-3 irreps' chosen modes across the same [start, end] amplitude
    range, writing one frozen POSCAR per point of the Cartesian-product grid
    (steps**len(irrep_labels) points total). len==1 is a plain 1-D sweep --
    `freeze-amplitude`'s own dispatch calls this function with exactly one
    irrep. `freeze-amplitude`'s own CLI entry point
    (`_run_freeze_amplitude_grid_cli`) re-checks the 1-3 range before
    ever reaching this function, so the two commands stay distinct rather
    than overlapping on the single-irrep case -- this function itself still
    accepts 1-3, since it's shared.

    `column_labels`, if given, is a list parallel to `irrep_labels` giving
    the DISPLAY name (grid.txt column header, directory/point naming) for
    each component instead of its own irrep label -- `landau-grid` uses
    this to name output with the same order-parameter symbols (a, b, c,
    ...) an invariants file's own `landau-fit` expects, while
    `irrep_labels` itself still drives which `Irrep_<label>/phonons.npz`
    gets loaded. Defaults to `irrep_labels` itself (today's behavior,
    unchanged for `freeze-amplitude`'s own CLI entry point).

    `column_header_suffix` controls grid.txt's own column header, e.g.
    `"GM4-_amp_A"` (the default, descriptive but not what `landau-fit`'s
    own `parse_data_table` expects -- it looks up columns by the bare
    variable name). `landau-grid` passes `""` so its own grid.txt is a
    genuine drop-in target for `landau-fit --data` (just add an `energy`
    column) rather than needing a manual rename first.

    `direction_args`, if given, is a list of raw `--direction` strings
    (always symbolic `a/b/c` notation -- numeric notation was removed).
    Letters mean relative weight only, never an absolute scale: each grid
    point's own swept amplitude supplies the real magnitude, multiplying
    that irrep's own letter-weighted shape (built once, shared across
    every point in the sweep, since the shape itself is amplitude-
    independent). With no `direction_args` at all, every irrep simply
    combines at its own native direction, exactly as before.
    """
    if not 1 <= len(irrep_labels) <= _MAX_FREEZE_AMPLITUDE_GRID_IRREPS:
        print(
            f"freeze-amplitude accepts 1-{_MAX_FREEZE_AMPLITUDE_GRID_IRREPS} irreps "
            f"(got {len(irrep_labels)}) -- more than {_MAX_FREEZE_AMPLITUDE_GRID_IRREPS} is hard to visualize.",
            file=sys.stderr,
        )
        return 1
    if steps < 1:
        print(f"steps must be >= 1, got {steps}", file=sys.stderr)
        return 1

    try:
        _check_no_duplicate_irreps(irrep_labels)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    display_labels = column_labels if column_labels is not None else irrep_labels
    if len(display_labels) != len(irrep_labels):
        print(
            f"internal error: column_labels has {len(display_labels)} entries, "
            f"expected {len(irrep_labels)}",
            file=sys.stderr,
        )
        return 1
    display_label_by_irrep = dict(zip(irrep_labels, display_labels))

    try:
        structure = Structure.from_poscar(poscar_path)
    except (FileNotFoundError, PoscarFormatError) as exc:
        print(f"Failed to read {poscar_path}: {exc}", file=sys.stderr)
        return 1

    kmesh = _load_kpoints_mesh(kpoints_path)
    incar_template = _load_incar_template(incar_path, _INCAR_HINT_SCF)

    parent_magmom = _parse_and_validate_magmom(magmom, incar_template, structure, incar_path, poscar_path)
    if parent_magmom is _MAGMOM_ERROR:
        return 1

    try:
        tag = config.resolve_tag(name, poscar_title=structure.comment)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    loaded_by_label = {}
    for label in irrep_labels:
        loaded = _load_freeze_component(label, tag, mode_overrides.get(label))
        if loaded is None:
            return 1
        loaded_by_label[label] = loaded

    amplitudes = np.linspace(start, end, steps)
    n_points = steps ** len(irrep_labels)
    width = max(1, len(str(n_points - 1)))

    direction_was_typed = bool(direction_args)
    direction_args = _v2_default_direction_args(irrep_labels, loaded_by_label, direction_args)
    label_coefficients: dict = {}
    symbolic_specs = _resolve_symbolic_direction_specs(irrep_labels, direction_args)
    if symbolic_specs is None:
        return 1
    built = _build_symbolic_irrep_patterns(
        irrep_labels, symbolic_specs, loaded_by_label, structure, tag, isodata,
        # Moments can break the centring, so a magnetic structure's cell uses
        # only whole lattice translations of the parent (never smaller).
        allow_smaller_anchor=parent_magmom is None,
    )
    if built is None:
        return 1
    (ref_structure, per_irrep_pattern, components, label_coefficients,
     full_group_log, mechanisms_by_label) = built

    # MAGMOM computed ONCE, against the cell every point shares (only the
    # amplitude changes across the sweep), atom by atom from the parent.
    grid_overrides = dict(vasp_module.SCF_INCAR_OVERRIDES)
    tiled_magmom = None
    if parent_magmom is not None:
        tiled_magmom = np.asarray(parent_magmom, dtype=float)[list(components.parent_index)]
        grid_overrides = {**grid_overrides, "MAGMOM": vasp_module.format_magmom(tiled_magmom)}

    dir_name_parts = [
        _symbolic_dirname_part(display_label_by_irrep[label], loaded_by_label[label][2], label_coefficients[label])
        if direction_was_typed else f"{display_label_by_irrep[label]}_{loaded_by_label[label][2]}"
        for label in irrep_labels
    ]
    dir_name = "Phonon_" + "_".join(dir_name_parts) + "_Amplitude"
    grid_dir = Path(dir_name)
    grid_dir.mkdir(parents=True, exist_ok=True)

    summary_lines = [
        "point\t" + "\t".join(f"{display_label_by_irrep[label]}{column_header_suffix}" for label in irrep_labels)
        + "\tpath"
    ]

    flat_idx = 0
    for index_combo in itertools.product(range(steps), repeat=len(irrep_labels)):
        amp_values = {label: float(amplitudes[i]) for label, i in zip(irrep_labels, index_combo)}

        combined_pattern = sum(amp_values[label] * per_irrep_pattern[label] for label in irrep_labels)
        combined = ref_structure.displaced(combined_pattern)

        point_name = f"Point_{flat_idx:0{width}d}_" + "_".join(
            f"{display_label_by_irrep[label]}_{amp_values[label]:.6f}" for label in irrep_labels
        )
        point_dir = grid_dir / point_name
        point_dir.mkdir(parents=True, exist_ok=True)

        # `grid_overrides["MAGMOM"]` was computed ONCE, before this loop,
        # against the UN-reduced joint cell every point shares -- a FRESH
        # copy is passed into the reducer for each point (mutated in place
        # there for that point's own re-derived MAGMOM), so one point's
        # own reduction can never leak into another's overrides. Runs one
        # extra findsym call PER GRID POINT -- a real, known cost for a
        # large sweep (steps**n_irreps points), not just per call.
        point_overrides = dict(grid_overrides)
        combined, point_reduce_note, superlattice_and_log, superlattice_overrides = (
            _reduce_freeze_structure_with_magmom(
                combined, f"{tag} freeze-amplitude {' '.join(irrep_labels)} point {flat_idx}",
                isodata, tiled_magmom, point_overrides,
            )
        )
        _write_freeze_reduction_byproducts(point_dir, superlattice_and_log)
        _write_superlattice_kpoints(kmesh, structure.lattice, superlattice_and_log, point_dir)
        if superlattice_and_log is not None:
            _write_incar(
                incar_template,
                superlattice_overrides if superlattice_overrides is not None else grid_overrides,
                point_dir, filename="INCAR_superlattice",
            )

        combined.to_poscar(point_dir / "POSCAR")
        if point_reduce_note is not None:
            print(f"point {flat_idx}:")
            print(point_reduce_note)
        _write_scaled_kpoints(kmesh, structure.lattice, combined.lattice, point_dir)
        _write_incar(incar_template, point_overrides, point_dir)
        summary_lines.append(
            f"{flat_idx}\t" + "\t".join(f"{amp_values[label]:.6f}" for label in irrep_labels) + f"\t{point_name}/POSCAR"
        )
        flat_idx += 1

    summary_path = grid_dir / "grid.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n")
    print(f"Wrote {flat_idx} POSCAR(s) to {grid_dir}/")
    print(f"Wrote {summary_path}")
    return 0


def _run_freeze_amplitude_grid_cli(
    irrep_labels: list[str],
    start: float,
    end: float,
    steps: int,
    poscar_path: str,
    name: str | None,
    mode_args: list[str],
    direction_args: list[str] | None = None,
    isodata: str | None = None,
    kpoints_path: str = "KPOINTS",
    incar_path: str = "INCAR",
    magmom: bool = False,
) -> int:
    # 1 irrep is a plain 1-D E(Q) sweep, 2-3 the Cartesian-product grid --
    # ONE command since v1.41.0. They were split before that, which was pure
    # CLI surface: `_run_freeze_amplitude_grid` has always accepted 1-3, and
    # the old `freeze-amplitude` dispatch simply called it with a one-element
    # list. The upper bound is a readability judgement, not a limitation.
    if not 1 <= len(irrep_labels) <= _MAX_FREEZE_AMPLITUDE_GRID_IRREPS:
        print(
            f"freeze-amplitude accepts 1-{_MAX_FREEZE_AMPLITUDE_GRID_IRREPS} irreps "
            f"(got {len(irrep_labels)}: {irrep_labels}) -- more than "
            f"{_MAX_FREEZE_AMPLITUDE_GRID_IRREPS} is hard to visualize.",
            file=sys.stderr,
        )
        return 1

    # A bare `--mode 2` / `--direction a,0,0` is only unambiguous with exactly
    # one irrep; with more, `_parse_label_value_pairs` rejects it rather than
    # guessing. Same rule `freeze` already uses.
    single_label = irrep_labels[0] if len(irrep_labels) == 1 else None
    try:
        mode_overrides = _parse_label_value_pairs(mode_args, int, "--mode", single_label)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return _run_freeze_amplitude_grid(
        irrep_labels,
        start,
        end,
        steps,
        poscar_path,
        name,
        mode_overrides,
        direction_args,
        isodata,
        kpoints_path=kpoints_path,
        incar_path=incar_path,
        magmom=magmom,
    )


def _format_lattice_parameters(params) -> str:
    a, b, c, alpha, beta, gamma = params
    return f"a={a:.5f} b={b:.5f} c={c:.5f} alpha={alpha:.3f} beta={beta:.3f} gamma={gamma:.3f}"


# Thresholds for the per-mode consolidated direction-vector report below.
# `_ZERO_AMPLITUDE_THRESHOLD_ANGSTROM` was originally 1e-4, chosen from the
# physicist's own judgment before any real noise-floor measurement existed.
# Tightened to 1e-7 once one was found and measured directly: `supercell.
# enlarge_onto_lattice`'s own rotation-recovery tiling (needed whenever an
# irrep's native cell relates to the parent by a genuine embedding
# rotation, e.g. TiO2's and CMO's own Gamma-point irreps) leaves a real,
# already-documented residual around 1e-5 A (confirmed directly:
# amplitude-independent -- injecting 0.001, 0.01, and 0.1 A all left
# exactly the same ~2e-5 A residual on uninvolved irreps/directions,
# ruling out a real physics leak -- and present even decomposing a
# structure against its own native k-point, so it's provably a tiling
# artifact, not a resolution bug). The physicist's own explicit call once
# this was found: values at this level are real (not exactly zero) but
# obviously negligible, and should be shown as an actual small NUMBER for
# the physicist to judge themselves, never silently folded into "(below
# noise floor)" the way a genuine 1e-4-scale coincidence would be. Only
# truly exact-to-machine-precision zeros (confirmed elsewhere at 1e-15 to
# 1e-19 for materials with no rotation involved, e.g. STO, BaTiO3) still
# read as "(below noise floor)" at this tighter threshold.
_ZERO_AMPLITUDE_THRESHOLD_ANGSTROM = 1e-7
_EQUAL_AMPLITUDE_RELATIVE_TOLERANCE = 0.01

# A position is also treated as unpopulated when it is this small a FRACTION
# of its own mode's largest component -- a second, scale-free floor that the
# absolute threshold above structurally cannot provide.
#
# Why both are needed, from real data. Decomposing a real relaxed BaTiO3
# P4mm structure (a genuine single-axis (0,0,1) distortion) reported GM4-
# mode 1 as `a,a,b` with a = 5e-6 A and b = -0.105126 A: the two noise
# positions clear the 1e-7 absolute threshold twenty-fold and so earn their
# own letter, making a pure one-axis condensation read as three populated
# directions. The noise there is real (rotation-recovery tiling residual,
# documented under `tile_onto_supercell`'s own precision section) and its
# ABSOLUTE size is meaningless on its own: it scales with the structure, so
# a 0.001 A frozen phonon and a 0.15 A relaxation sit three orders of
# magnitude apart. Only the ratio to that mode's own peak separates them --
# 5e-6/0.105 = 0.005% here, against 100% for a genuine (a,a,a) case.
#
# The absolute threshold still does real work and is NOT superseded: a mode
# that is nothing BUT noise has no meaningful peak, so a relative test alone
# would promote its own largest noise value to letter 'a' and invent a shape
# for it (real GM5- in that same report would go from a correct `0,0,0` to a
# spurious `a,b,c`). A position must clear BOTH floors to be given a letter.
#
# RAISED 1% -> 2.5% when v0.92 decomposition moved to per-irrep resolution
# (confirmed with the physicist), which gave up the exact numerical
# orthogonality of one shared frame. The v2.0 joint cell restores it, but the
# floor was kept: it governs only the compact letter label. Measured cross-talk into genuinely
# unpopulated irreps, injecting 0.001 A at one irrep:
#
#     k-point       shared frame      per-irrep
#     STO M            3.5e-22        3.5e-22   (unchanged)
#     CMO X            1.5e-15        7.1e-06   (0.71%)
#     graphite K       8.7e-16        1.67e-05  (1.67%)
#     TiO2 X           2.7e-16        2.00e-05  (2.00%)
#
# At 1% that cross-talk would clear this floor and invent a spurious letter
# for an irrep carrying no real signal. 2.5% sits above the worst measured
# case (2.00%) with margin. Nothing is hidden by the raise: every
# amplitude is printed in full in the per-direction totals section
# regardless, so this only governs the compact a/b/c shape label.
_NEGLIGIBLE_AMPLITUDE_RELATIVE_FLOOR = 0.025


def _assign_direction_letters(values: list) -> tuple[list[str], dict]:
    """Assign a compact ISODISTORT-style a/b/c label to each position in
    one mode's own ordered list of direction amplitudes (Angstrom),
    mirroring `freeze --direction`'s own symbolic vocabulary so a
    decomposition report and a `--direction` argument can share labels
    directly.

    A position is labeled `'0'` when it fails EITHER floor: within
    `_ZERO_AMPLITUDE_THRESHOLD_ANGSTROM` of 0 in absolute terms, or below
    `_NEGLIGIBLE_AMPLITUDE_RELATIVE_FLOOR` as a fraction of its own mode's
    largest component. Both are needed -- see those constants' own comment
    for the real BaTiO3 case that showed a pure single-axis distortion as
    `a,a,b`, and for why a relative test alone invents a shape for a mode
    that is nothing but noise. Two nonzero positions get the SAME letter
    when their magnitudes
    agree within `_EQUAL_AMPLITUDE_RELATIVE_TOLERANCE` (relative) AND share
    a sign; the same magnitude with the OPPOSITE sign reuses the letter
    negated (`'-a'`) rather than allocating a new one -- an overall sign
    flip is the same physical relationship between two positions, not a
    new, unrelated one.
    Otherwise a fresh letter is assigned in position order (a, b, c, ...).

    Returns `(labels, legend)` -- `legend` maps each bare (never negated)
    letter to the real signed Angstrom value it was first assigned from.
    """
    labels: list[str] = []
    legend: dict = {}
    peak = max((abs(v) for v in values), default=0.0)
    for value in values:
        negligible_absolutely = abs(value) < _ZERO_AMPLITUDE_THRESHOLD_ANGSTROM
        negligible_relatively = peak > 0.0 and abs(value) / peak < _NEGLIGIBLE_AMPLITUDE_RELATIVE_FLOOR
        if negligible_absolutely or negligible_relatively:
            labels.append("0")
            continue
        matched = None
        for letter, canon in legend.items():
            rel = abs(abs(value) - abs(canon)) / abs(canon)
            if rel <= _EQUAL_AMPLITUDE_RELATIVE_TOLERANCE:
                matched = letter if (value > 0) == (canon > 0) else f"-{letter}"
                break
        if matched is None:
            index = len(legend)
            letter = chr(ord("a") + index) if index < 26 else f"a{index - 25}"
            legend[letter] = value
            matched = letter
        labels.append(matched)
    return labels, legend


def _join_direction_labels(labels: list, entries: list) -> str:
    """Comma between positions of ONE arm, SEMICOLON between arms.

    Real TiO2 X1 is 2 star arms x a 2-component within-arm doublet, so a flat
    `a,0,0,a` cannot say WHICH positions belong to which arm -- a reader has
    to go back to the OPD report to find out. `a,0;0,a` says it directly.

    This is ISOTROPY's own convention, not an invention here: `iso` writes
    real CMO X2+ as `(b;b)`, and `iso.IsoStratum.as_direction_spec` already
    notes that "its semicolons between star arms are the convention our
    grammar already follows". Until now we accepted that spelling on input
    without ever producing it on output.

    **Output only, deliberately.** `--direction` keeps commas as its canonical
    form: `;` is a shell metacharacter, so `--direction a,0;0,a` unquoted is
    truncated by the shell to `a,0` and then runs a DIFFERENT, entirely valid
    calculation with no error. A silent wrong answer is too high a price for a
    nicer spelling. The parser still ACCEPTS `;` (it always has --
    `_parse_symbolic_direction_spec` splits on it), so a spec copied back out
    of a report still works.

    Single-arm irreps -- every Gamma irrep, and most others -- are joined with
    commas exactly as before, so nothing they print moves.
    """
    arms = [getattr(m, "arm", 1) or 1 for m in entries]
    if len(labels) != len(arms) or len(set(arms)) < 2:
        return ",".join(labels)
    out = [labels[0]]
    for i in range(1, len(labels)):
        out.append(";" if arms[i] != arms[i - 1] else ",")
        out.append(labels[i])
    return "".join(out)


def _consolidate_mode_direction_vectors(amplitudes: list) -> list[dict]:
    """Group `amplitudes` (`ModeAmplitude`) by (label, mode_index),
    ordering each mode's own directions the same way the per-row table
    already did (native/arm1 first, then the rest alphabetically -- which
    already puts a multi-arm blend's own 'armN'/'armNwM' entries in the
    right arm-then-within-arm order, since 'arm1' is a string prefix of
    'arm1w1'), then assigns each position a letter via
    `_assign_direction_letters`.

    Returns one dict per (label, mode_index), in first-seen order:
    `{label, mode_index, species, degeneracy, vector, legend,
    is_meaningful}` -- `vector` is the comma-joined label string (e.g.
    `'a,a,a'`), `legend` the letter->value mapping, and `is_meaningful`
    True whenever at least one position isn't `'0'`.
    """
    groups: dict = {}
    order: list = []
    for ma in amplitudes:
        key = (ma.label, ma.mode_index)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(ma)

    results = []
    for key in order:
        # Alphabetical IS position order once directions are OPD letters
        # (a, b, c ...). The old key put the literal string "native" first,
        # which is a no-op for letters and would otherwise leave the order
        # depending on a name that no longer appears.
        entries = sorted(groups[key], key=lambda m: (m.direction != "native", m.direction))
        labels, legend = _assign_direction_letters([m.amplitude_angstrom for m in entries])
        results.append(
            {
                "label": entries[0].label,
                "mode_index": entries[0].mode_index,
                "species": entries[0].species,
                "degeneracy": entries[0].degeneracy,
                "vector": _join_direction_labels(labels, entries),
                "legend": legend,
                "directions": [m.direction for m in entries],
                # Carried so `_opd_construction_text` can classify by what
                # a direction IS rather than by how its name happens to look.
                "mechanisms": [getattr(m, "mechanism", "") or "" for m in entries],
                "is_meaningful": any(label != "0" for label in labels),
            }
        )
    return results


def _opd_construction_text(row: dict) -> str:
    """How an irrep's order-parameter directions are built, in words, with
    counts -- the mode table's `OPD construction` column.

    Replaces a single legend printed under the table: different irreps at
    one k-point can be built differently, so one legend cannot describe
    them all, while a per-row statement can. Derived from the per-position
    mechanisms (`opd.py`: native, within, armN, within(armN), quadrature,
    quad+within, orbit); where a path records no mechanism, the direction
    name itself (`armN`, `armNwM`) is used.

      single                        degeneracy 1 -- no partner constructed
      N x within-arm                partners from a within-arm rotation
      M x arms                      one direction on each of M star arms
      N x within-arm, M x arms      both
      quadrature                    a Bloch-phase (90-degree) partner
      N x within-arm, quadrature    both
      orbit completion              a direction no physical mechanism reached
    """
    kinds = [m or d for m, d in zip(row.get("mechanisms", []), row.get("directions", []))]
    if len(kinds) <= 1:
        return "single"
    per_arm: dict = {}
    quadrature = orbit = False
    for kind in kinds:
        if kind == "orbit":
            orbit = True
            continue
        if "quad" in kind:
            quadrature = True
            continue
        match = re.match(r"(?:within\()?arm(\d+)", kind)
        arm = int(match.group(1)) if match else 1
        per_arm[arm] = per_arm.get(arm, 0) + 1
    parts = []
    within = max(per_arm.values(), default=1)
    if within > 1:
        parts.append(f"{within} x within-arm")
    if len(per_arm) > 1:
        parts.append(f"{len(per_arm)} x arms")
    if quadrature:
        parts.append("quadrature")
    if orbit:
        parts.append("orbit completion")
    return ", ".join(parts) or "single"


def _format_mode_vector_legend(legend: dict) -> str:
    if not legend:
        return "(below noise floor)"
    return ", ".join(f"{letter} = {value:.6f} A" for letter, value in legend.items())


def _irrep_summary_entries(amplitudes: list) -> dict:
    """For each irrep label present in `amplitudes`, decide whether it has
    a meaningful contribution (any mode's own consolidated vector isn't
    all `'0'`) and, if so, its shared direction vector -- or `'MIXED'` if
    its own distinct modes don't all agree on the same vector shape. Real,
    validated behavior for a genuine single-domain condensation is that
    every nonzero mode of an irrep shares the SAME vector shape (see real
    BaTiO3 P4mm/R3m data in the developer notes), so 'MIXED' is a real, worth-
    flagging anomaly, not an expected case.

    Also reports each meaningful irrep's own overall magnitude -- `sqrt`
    of the summed squared amplitude across EVERY (mode, direction) pair of
    that irrep (not just the ones the letter-threshold called nonzero, so
    this number is exact, not thresholded).

    Returns `{label: (vector_or_MIXED, magnitude_angstrom)}`, omitting any
    label with no meaningful contribution at all.
    """
    rows = _consolidate_mode_direction_vectors(amplitudes)
    by_label: dict = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)

    magnitude_by_label: dict = {}
    for ma in amplitudes:
        magnitude_by_label[ma.label] = magnitude_by_label.get(ma.label, 0.0) + ma.amplitude_angstrom**2

    entries: dict = {}
    for label, rows_for_label in by_label.items():
        vectors = {row["vector"] for row in rows_for_label if row["is_meaningful"]}
        if not vectors:
            continue
        vector = vectors.pop() if len(vectors) == 1 else "MIXED"
        entries[label] = (vector, magnitude_by_label[label] ** 0.5)
    return entries


def _format_decomposition_summary(summary_entries: dict, norms: dict | None = None) -> str:
    """`summary_entries`: `{label: (k_vector, vector_or_MIXED, magnitude_angstrom)}`,
    accumulated across every k-vector/irrep requested so far in this
    `decomposition` invocation (one shared summary file for the whole
    command, not one per k-vector -- unlike `decomposition_mode_<K>`,
    which is deliberately one file per requested k-vector/label).

    `norms`: `{k_vector: (displacement_norm, residual_norm)}`, which adds the
    COMBINED line below. Optional so existing single-k callers are unchanged.
    """
    lines = ["Summary: irreps with meaningful contribution"]
    if not summary_entries:
        lines.append("  (none found yet)")
        return "\n".join(lines)
    for label, (k_vector, vector, magnitude) in summary_entries.items():
        lines.append(f"  {label:<10} (k={k_vector})  {vector:<12}  {magnitude:.6f} A")
    lines.extend(_format_combined_residual(norms or {}))
    return "\n".join(lines)


def _format_combined_residual(norms: dict) -> list[str]:
    """How much of the structure ALL the requested k-vectors explain together.

    Each k-vector's own residual is what THAT k-point alone leaves unexplained,
    so for a genuinely coupled structure every one of them is large and none of
    them is the answer. Real STO with R4- frozen at 0.05 and M2- at 0.03 reports
    51.45% at R and 85.75% at M -- both correct, and together they explain
    100.000%. Nothing printed that until this line existed, so a perfect coupled
    result read as two alarming numbers.

    Combining in QUADRATURE is legitimate for exactly one reason: distinct
    irreps are orthogonal (Schur, under the group-invariant inner product), so
    their explained parts add in squares. The same fact the whole per-irrep
    design rests on.
    """
    if len(norms) < 2:
        return []
    explained_sq, disp = 0.0, {}
    for k, (d, r) in norms.items():
        if d <= 0.0:
            return []
        explained_sq += max(0.0, d * d - r * r)
        disp[k] = d
    explained = float(np.sqrt(explained_sq))
    total = max(disp.values())
    out = ["", f"Combined across {', '.join(norms)}:",
           f"  explained {explained:.6f} A of {total:.6f} A"
           f"   residual {100.0 * max(0.0, 1.0 - explained / total):.4f}%"]
    # The per-k displacement norms measure the SAME structure, so they must
    # agree. When they do not, the combined number is built on an
    # inconsistency and saying so is more useful than hiding it.
    if max(disp.values()) - min(disp.values()) > 1e-6 * max(disp.values()):
        out.append("  NOTE: the k-points disagree about this structure's own total "
                   "displacement (" + ", ".join(f"{k} {v:.6f}" for k, v in disp.items())
                   + "). They measure one structure, so they should agree; the "
                     "largest is used above. Treat the combined figure as "
                     "approximate and check the alignment at the outlying k-point.")
    return out


def _format_direction_totals(amplitudes: list, divisor: float | None = None) -> list[str]:
    """One aggregate row per (irrep, direction) -- sqrt of the summed
    squared amplitude across every one of that irrep's own distinct
    modes, e.g. real BaTiO3 R3m's own (a,a,a) body-diagonal distortion:
    GM4-'s own per-mode breakdown already shows each INDIVIDUAL mode
    (Sr, Ti, and both O's, 4 separate rows) landing at the same amplitude
    on x/y/z, but nothing in the plain per-row listing states the
    OVERALL-irrep answer ("x, y, and z are equally populated") as a
    single, directly comparable number per direction -- a real, reported
    gap (this function didn't exist before), not a change to what's
    already computed.

    Sum of SQUARES, not a raw sum: an individual mode's own amplitude
    SIGN for a given direction is an artifact of which rotation/arm
    operation happened to be found (a labeling convention -- flipping
    which of two antiparallel unit vectors gets called "+x" flips the
    sign with no physical meaning attached), not a physically comparable
    quantity across different modes -- squaring before summing keeps the
    aggregate sign-convention-independent, while still reducing to
    exactly the per-mode value whenever only one mode is involved.

    Direction order is first-seen per irrep (matching `ModeAmplitude`'s
    own generation order -- native/arm1 first), not sorted, so a
    within-arm sweep's native entry stays first and a multi-arm irrep's
    arm1/arm2/... stays in arm-number order.

    Each direction is NAMED by its own position in the irrep's order
    parameter -- `a,0,0` / `0,a,0` / `0,0,a` for a degeneracy-3 irrep --
    rather than by the internal mechanism that produced it ("native",
    "armN", "quad:..."). That is the same vocabulary `freeze --direction`
    takes as input, so a direction read off this table can be typed
    straight back in. The mechanism is still shown alongside, since it is
    what distinguishes a within-arm rotation from a different star arm
    from a Bloch quadrature -- see `_opd_construction_text`.
    """
    totals: dict = {}
    # What KIND each direction is -- a within-arm rotation, a different star
    # arm, a Bloch quadrature. Since the OPD unification `direction` is the
    # POSITION letter, so a bracket showing it just repeats the column beside
    # it; the mechanism is the part a reader cannot infer from the letter,
    # and is exactly what this table's own docstring promises to show.
    #
    # Falls back to the direction name where no mechanism was recorded --
    # the multi-arm BLEND path does not set one, and there `armN` is itself
    # the informative label.
    mechanisms: dict = {}
    for ma in amplitudes:
        key = (ma.label, ma.direction)
        totals[key] = totals.get(key, 0.0) + ma.amplitude_angstrom**2
        mechanism = (getattr(ma, "mechanism", "") or "").strip()
        if mechanism and key not in mechanisms:
            mechanisms[key] = mechanism

    def _row(name: str, squared: float, trailer: str = "") -> str:
        """One row carrying BOTH normalisations, matching ISODISTORT's own
        two columns: `As` over the resolved supercell, `Ap` per primitive
        parent cell. `Ap` is blank when `N` wasn't a clean integer."""
        a_s = squared**0.5
        a_p = f"{a_s / divisor:>12.6f}" if divisor else f"{'--':>12}"
        return f"  {name:<26} {a_s:>12.6f} {a_p}{trailer}"

    lines = []
    # Same irrep order as the mode table above it (sorted by label), with a
    # blank line between irreps, so the two tables read in step.
    labels = sorted(dict.fromkeys(ma.label for ma in amplitudes))
    degeneracies = {ma.label: ma.degeneracy for ma in amplitudes}
    for label in labels:
        if lines:
            lines.append("")
        directions = list(dict.fromkeys(ma.direction for ma in amplitudes if ma.label == label))
        degeneracy = degeneracies.get(label, len(directions))
        # Positional naming is only well defined when this irrep's distinct
        # directions actually correspond to its order-parameter components.
        # They may not: this table aggregates across every MODE of the irrep,
        # and two modes can return different direction labels for the same
        # position (a documented property of the within-arm search, whose
        # label comes from a reference atom that differs mode to mode), so
        # the key count can exceed the degeneracy. Naming positions in that
        # case would invent an order-parameter component that does not exist,
        # so fall back to the raw mechanism name and say why.
        if len(directions) != degeneracy:
            for direction in directions:
                lines.append(_row(f"{label:<10} {direction}", totals[(label, direction)]))
            # Deliberately NO `all components` row here. This branch means the
            # direction set does not correspond to the order parameter's own
            # components (an incomplete sweep, or two modes labelling one
            # position differently), so a sum across them is not "the whole
            # order parameter" and printing it as such would overstate what
            # was actually resolved.
            if len(directions) > degeneracy:
                # Genuinely surprising and worth saying: more distinct
                # directions than the order parameter has components. Fewer
                # is ordinary (an incomplete sweep) and is left unremarked,
                # exactly as before this naming change.
                lines.append(
                    f"  {'':<10} (^ {label} produced {len(directions)} distinct directions for a "
                    f"degeneracy of {degeneracy} -- different modes labelled the same position "
                    "differently, so these are shown by mechanism rather than by order-parameter "
                    "position)"
                )
            continue
        # Semicolons between star arms, commas within one -- `iso`'s own
        # convention, and the same string `irrep --opd` prints, via the
        # one shared formatter so the two cannot disagree. They did:
        # real CMO X4- came back `a;0` from `irrep --opd` and `a,0` here,
        # which hid that those positions are different star arms.
        arms_by_position = [
            next((getattr(ma, "arm", 0) for ma in amplitudes
                  if ma.label == label and ma.direction == d), 0)
            for d in directions
        ]
        for position, direction in enumerate(directions):
            opd = opd_module.format_direction_spec(position, arms_by_position)
            # No per-position mechanism tag: the mode table's `OPD construction`
            # column states how this irrep's OPD is built.
            lines.append(_row(f"{label:<10} {opd}", totals[(label, direction)]))
        # The whole order parameter at once, across every component -- the
        # directly comparable counterpart to ISODISTORT's own per-irrep
        # `all` line, which reports one amplitude per mode covering ALL
        # active components rather than one per component as the rows above
        # do. For a single-component OPD the two coincide; for a genuine
        # multi-component one they differ by sqrt(n_active), e.g. real
        # BaTiO3 GM4- at (0,a,a) where ISODISTORT's Ti 0.07746 is this row
        # and 0.05477 is each component row.
        if degeneracy > 1:
            all_squared = sum(totals[(label, d)] for d in directions)
            lines.append(_row(f"{label:<10} all components", all_squared))
    return lines


def _format_k_vector_decomposition_report(result: decomposition_module.KVectorResult) -> str:
    lines = [f"=== k = {result.k_vector} ==="]
    if result.is_multi_arm_blend:
        blend_irrep_labels = sorted({ma.label for ma in result.amplitudes})
        if len(blend_irrep_labels) > 1:
            lines.append(
                f"multi-arm BLEND -- distorted structure matches the joint cell hosting {result.arm_count} "
                f"total arm-components across {len(blend_irrep_labels)} irreps ({', '.join(blend_irrep_labels)}) "
                "simultaneously (not a single resolved arm); each component's own amplitude below is "
                "fitted by one joint least-squares projection; each is named by its own order-parameter position below, with the star arm it belongs to shown alongside"
            )
        else:
            lines.append(
                f"multi-arm BLEND -- distorted structure matches the joint cell hosting all {result.arm_count} "
                f"arms of this irrep's star simultaneously (not a single resolved arm); each arm's own "
                "amplitude below is fitted by a joint least-squares projection; each is named by its own order-parameter position, with the star arm it belongs to shown alongside"
            )
    elif result.arm_count > 1:
        lines.append(
            f"arm {result.arm_index} of {result.arm_count} -- multi-arm irrep, matched against "
            "the distorted structure's own lattice shape (not necessarily the SMODES-native arm)"
        )
    # OBLIQUENESS, reported on EVERY path that measures it -- not just the
    # multi-arm blend, where this note used to be nested. The ordinary
    # joint-OPD-cell solve computes the same number and left it unprinted, so
    # real graphite L1 mode 3 reported amplitudes fitted against a basis at
    # |cos| 0.61 with nothing saying so (Issue 8).
    #
    # This is the caveat a reader actually needs, because an oblique basis
    # makes a per-component amplitude LEGITIMATELY exceed the structure's own
    # displacement norm -- real graphite L1 m3 at 1.98x and real STO M5- m0 at
    # 1.98x, both correct. The naive `|As| <= ||d||` reading only holds for an
    # ORTHONORMAL basis; the true bound is `||d|| / sigma_min`, which every
    # measured case satisfies with room to spare.
    if (
        result.multi_arm_gram_off_diagonal_max is not None
        and result.multi_arm_gram_off_diagonal_max > 0.05
    ):
        lines.append(
            f"note: this irrep's own order-parameter directions are not mutually orthogonal for at "
            f"least one mode (largest Gram off-diagonal: {result.multi_arm_gram_off_diagonal_max:.4f}). "
            "The joint least-squares fit accounts for this correctly and the residual below is still "
            "the right measure of what is explained -- but an individual component's amplitude is a "
            "coordinate in an oblique basis, so it can legitimately exceed the structure's own total "
            "displacement norm, and per-component values should be read with that in mind. The "
            "`all components` total and Ap are unaffected"
        )
    # The Γ origin-gauge caveat, the As/Ap definitions and the meaning of each
    # OPD construction belong in `--help`/the docs, not in every report
    # (validation campaign 2026-09, Issues 3 and 4). No other program is
    # referred to in the report.
    lines.append(
        "Origin: same as the parent" if result.origin_shift_frac is None
        else "Origin: distorted structure shifted by ("
        + ", ".join(f"{v:.6f}" for v in result.origin_shift_frac) + ") to match the parent"
    )
    lines.append("")
    det_note = ""
    if result.strain.determinant < 0:
        det_note = "  WARNING: negative -- the distorted structure may be a mirror image of the target, not a strain"
    lines.append(f"Det(F) = {result.strain.determinant:.6f}   (volume ratio){det_note}")
    lines.append("Strain:")
    for row in result.strain.strain:
        lines.append("  " + "  ".join(f"{v: .6f}" for v in row))
    lines.append("")
    lines.append(f"Total displacement norm: {result.displacement_norm_angstrom:.6f} A")
    pct = (
        100.0 * result.residual_norm_angstrom / result.displacement_norm_angstrom
        if result.displacement_norm_angstrom > 0
        else 0.0
    )
    lines.append(
        f"Residual (not explained by the irreps below): "
        f"{result.residual_norm_angstrom:.6f} A ({pct:.2f}%)"
    )

    sorted_amplitudes = sorted(
        result.amplitudes, key=lambda m: (m.label, m.mode_index, m.direction != "native", m.direction)
    )
    mode_rows = _consolidate_mode_direction_vectors(sorted_amplitudes)
    construction = {}
    for row in mode_rows:
        construction.setdefault(row["label"], _opd_construction_text(row))
    w_irrep = max([len("Irrep")] + [len(r["label"]) for r in mode_rows]) + 2
    w_opd = max([len("OPD")] + [len(r["vector"]) for r in mode_rows])
    w_con = max([len("OPD construction")] + list(map(len, construction.values())))
    # Ap per mode, beside As: the cell-size-independent value. Without it a
    # reader must reconstruct the As/Ap divisor from the per-direction
    # table's 6-decimal values, where a small row's rounding gives a wrong
    # ratio (0.000002/0.000001 -> 2 for a true sqrt 2) -- the validation
    # campaign's own scorer fell into exactly that (2026-09-28).
    divisor = result.parent_cell_divisor
    as_text = {id(r): _format_mode_vector_legend(r["legend"]) for r in mode_rows}
    ap_text = {
        id(r): ("" if not r["legend"] else
                ", ".join(f"{k} = {v / divisor:.6f} A" for k, v in r["legend"].items()) if divisor else "--")
        for r in mode_rows
    }
    w_as = max([len("As, per component (A)")] + list(map(len, as_text.values())))

    def _mode_line(irrep, mode, species, degeneracy, opd, cons, a_s, a_p):
        return (f"  {irrep:<{w_irrep}}{mode:>4}  {species:<9}{degeneracy:>10}  "
                f"{opd:<{w_opd}}  {cons:<{w_con}}  {a_s:<{w_as}}  {a_p}").rstrip()

    lines.append("")
    lines.append("Mode decomposition:")
    lines.append(_mode_line("Irrep", "Mode", "Species", "Degeneracy", "OPD", "OPD construction",
                            "As, per component (A)", "Ap, per component (A)"))
    lines.append("")
    prev_label = None
    for row in mode_rows:
        if prev_label is not None and row["label"] != prev_label:
            lines.append("")
        lines.append(_mode_line(row["label"], row["mode_index"], row["species"], row["degeneracy"],
                                row["vector"], construction[row["label"]],
                                as_text[id(row)], ap_text[id(row)]))
        prev_label = row["label"]
    lines.append("")
    lines.append("Per-direction decomposition:")
    lines.append(f"  {'Irrep      OPD':<26} {'As (A)':>12} {'Ap (A)':>12}")
    lines.append("")
    lines.extend(_format_direction_totals(result.amplitudes, result.parent_cell_divisor))
    return "\n".join(lines)


def _try_enlarge_sub_to_match_parent(parent: Structure, sub: Structure) -> Structure | None:
    """Enlarge `sub` onto `parent`'s own axes and return it if that
    resolves an otherwise-fatal `check_stoichiometry` mismatch where `sub`
    is SMALLER than `parent` -- i.e. `parent` itself wasn't given in its
    own most-reduced form. A real, hit-in-practice case, not hypothetical:
    `freeze` always picks the smallest cell the distortion needs
    (`isotropy_basis.invariant_cell`), so a Gamma-point irrep's own output can
    legitimately come out SMALLER than a `--parent` given in its
    conventional, non-primitive form -- confirmed directly for real CMO
    data (I-centered, space group 139): `freeze GM1+` on the 24-atom
    conventional cell writes a 12-atom true-primitive structure, since
    GM1+'s own SMODES-derived native cell already IS that smaller
    primitive (independently confirmed physically genuine, not just
    smaller: GM1+ is the totally-symmetric irrep, so it cannot break any
    symmetry at all -- `symmetry --tolerance 0.0001` on the frozen output
    still reports the identical space group 139).

    **Enlarges `sub` rather than reducing `parent`.** Reducing `--parent` to
    its primitive cell was tried and broke downstream: `irrep.k_vector` and
    `superlattice` are tied to the basis smodes was run against, and a
    centring reduction is a genuine basis change; the reduced cell's very
    oblique shape (alpha=beta~158, gamma~31 degrees for real CMO) also
    destabilised the v1.x direction search (the v1.x development record).

    Enlarging `sub` sidesteps BOTH problems by construction: `parent` is
    never touched at all (still its own original, well-conditioned axes,
    and still the exact same structure every OTHER irrep's own resolution
    already works against), and `sub`'s own real, already-displaced atom
    positions are periodically repeated onto `parent`'s bigger cell via
    `supercell.enlarge_onto_lattice` -- the SAME already-validated
    machinery the `superlattice` command uses to map an arbitrary
    (possibly small, rotated, or skewed) POSCAR onto a reference's own
    nice axes. Confirmed directly on the same real CMO GM1+ case that
    broke the parent-
    reduction approach: `big_sub`'s own lattice comes out numerically
    IDENTICAL to `parent.lattice`, and the resulting decomposition report
    gives a sane 2.60% residual (real floating-point noise) instead of
    2664%.

    **Never scoped to Gamma-only or any other k-point restriction** --
    unlike the abandoned parent-reduction approach, this never touches
    `parent`'s own interpretation or any irrep's own k-vector/basis at
    all, so the arm/k-vector-basis-dependency risk that keeps `--parent`
    itself un-reduced simply doesn't apply here; a purely geometric
    "repeat these real atoms periodically onto a bigger, already-known-
    compatible lattice" operation is well-defined for a structure from
    ANY k-point.

    Only substitutes when the enlarged `sub`'s own size EXACTLY matches
    `parent`'s (stoichiometry multiplier == 1) -- a `SupercellError` from
    `nice_supercell_matrix`/`tile_onto_supercell` (no valid relationship
    found at all) or any other stoichiometry mismatch means this genuinely
    isn't the "parent wasn't primitive" case, and `None` is returned so
    the caller falls back to the original `check_stoichiometry` error.
    Never raises.
    """
    # Try the parent's own cell first -- every case that already worked takes
    # exactly this path, unchanged. Only if that cell cannot HOST `sub`'s
    # periodicity, try the smallest diagonal multiples of it.
    #
    # Why that is needed (Issue 2 of the 2026-09 validation campaign): a
    # centred parent given in its CONVENTIONAL cell need not be a supercell of
    # a zone-boundary distortion's own cell. Real NaCl (Fm-3m, 8-atom cubic
    # cell) with an L-point distortion: the pattern flips sign under a cubic
    # lattice translation, so it does not repeat within the cubic cell at all
    # -- only 2x2x2 of it hosts it. Refusing forced users onto a
    # `POSCAR_superlattice`-as-PARENT workaround, which then mis-normalised
    # Ap by exactly 2x and doubled the direction basis (measured against
    # ISODISTORT: 20/20 affected cases wrong, 62/62 true-parent cases right).
    # Enlarging the SUB instead keeps the true parent -- the basis smodes was
    # run against and the primitive-cell count Ap is normalised by.
    per_axis = range(1, 5)
    candidates = sorted(((a, b, c) for a in per_axis for b in per_axis for c in per_axis),
                        key=lambda n: (n[0] * n[1] * n[2], n))
    for n in candidates:
        target = np.diag(np.asarray(n, dtype=np.float64)) @ np.asarray(parent.lattice, dtype=np.float64)
        try:
            big_sub = supercell_module.enlarge_onto_lattice(sub, target)
        except supercell_module.SupercellError:
            continue
        try:
            multiplier = decomposition_module.check_stoichiometry(parent, big_sub)
        except decomposition_module.DecompositionError:
            continue
        if multiplier == n[0] * n[1] * n[2]:
            return big_sub
    return None


# What the SCREEN calls "detected" (the report files still list every irrep
# in full). An irrep is shown when its amplitude clears BOTH floors:
# - absolute, 1e-4 A: five times the ~2e-5 A construction cross-talk floor
#   (known issue 9 in the developer notes), which otherwise puts phantom ~1e-5 A irreps on
#   the screen -- real CMO X2+ at 0.025 A showed X3+, X3-, X4-, GM1+ and GM5-
#   at 0.000005-0.000028 A;
# - relative, 0.5% of the total displacement norm: scales with the
#   structure, so a relaxed CONTCAR's coordinate rounding (which grows with
#   the size of the distortion) does not list every irrep either. Small
#   genuine secondary order parameters are typically a few percent and pass.
_SCREEN_ABSOLUTE_FLOOR_ANGSTROM = 1e-4
_SCREEN_RELATIVE_FLOOR = 0.005


def _split_decomposition_messages(text: str) -> tuple[list[str], list[str]]:
    """Split what a decomposition printed into (notes, warnings), each
    de-duplicated in order. A message repeated with only its decimals
    changed (the same fallback hit again with a slightly different closest
    distance) counts as one. Notes are informational and belong in the
    report file; anything else (a `Warning:` line, a Python warning) also
    goes to the screen."""
    notes, warnings_text, seen = [], [], set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        key = re.sub(r"\d+\.\d+", "#", line)
        if key in seen:
            continue
        seen.add(key)
        (notes if line.startswith("note:") else warnings_text).append(line)
    return notes, warnings_text


def _screen_lines_for_k_vector(requested: str, result) -> list[str]:
    """One screen line per irrep detected at this k-vector, or one line
    saying none was (Issue 5, as revised with the physicist: a residual
    alone does not say what was found -- 0.00% for a structure with no
    content at this k-point is meaningless). `Ap` is shown because it is
    cell-size independent, the number to quote."""
    floor = max(_SCREEN_ABSOLUTE_FLOOR_ANGSTROM, _SCREEN_RELATIVE_FLOOR * result.displacement_norm_angstrom)
    divisor = result.parent_cell_divisor or 1.0
    found = [
        (label, vector, magnitude)
        for label, (vector, magnitude) in _irrep_summary_entries(result.amplitudes).items()
        if magnitude >= floor
    ]
    if not found:
        return [f"{requested}: no irrep amplitudes detected"]
    width = max(len(label) for label, _v, _m in found)
    opd_width = max(len(vector) for _l, vector, _m in found)
    return [
        f"{requested}: {label:<{width}}  OPD {vector:<{opd_width}}  Ap {magnitude / divisor:.6f} A"
        for label, vector, magnitude in sorted(found, key=lambda entry: -entry[2])
    ]


def _parent_supercell_warning(parent: Structure, symmetry_info) -> str | None:
    """A warning when `--parent` is a SUPERCELL of the conventional cell
    findsym reports for it, else None.

    smodes then derives the irreps of that supercell -- different k-points
    and degeneracies -- so `Ap` and the direction basis refer to it, not to
    the real parent (measured on NaCl: `Ap` 2x, basis doubled). Not a single
    scale factor, so it is not corrected; the run continues with a warning
    (Issue 2, decided with the physicist: warn, not refuse).
    """
    conventional = findsym_module._lattice_from_parameters(symmetry_info.lattice_parameters)
    ratio = abs(np.linalg.det(np.asarray(parent.lattice, float))) / abs(np.linalg.det(conventional))
    if ratio < 1.5:
        return None
    return (
        f"Warning: --parent is a supercell, {ratio:.0f}x its conventional cell -- "
        "irreps, degeneracies and Ap refer to the supercell; pass the conventional "
        "cell as --parent"
    )


def _run_decomposition(
    parent_poscar_path: str,
    sub_poscar_path: str,
    k_vectors: list[str],
    name: str | None,
    tolerance: float,
    findsym_tolerance: float,
    isodata: str | None,
) -> int:
    try:
        parent = Structure.from_poscar(parent_poscar_path)
    except (FileNotFoundError, PoscarFormatError) as exc:
        print(f"Failed to read {parent_poscar_path}: {exc}", file=sys.stderr)
        return 1
    try:
        sub = Structure.from_poscar(sub_poscar_path)
    except (FileNotFoundError, PoscarFormatError) as exc:
        print(f"Failed to read {sub_poscar_path}: {exc}", file=sys.stderr)
        return 1

    try:
        tag = config.resolve_tag(name, poscar_title=parent.comment)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    enlargement_note = None
    try:
        stoichiometry_multiplier = decomposition_module.check_stoichiometry(parent, sub)
    except decomposition_module.DecompositionError as exc:
        big_sub = _try_enlarge_sub_to_match_parent(parent, sub)
        if big_sub is None:
            print(str(exc), file=sys.stderr)
            return 1
        enlargement_note = (
            f"note: --sub was enlarged from {sub.n_atoms} to {big_sub.n_atoms} atoms, onto "
            "--parent's own axes, because --sub's own cell was smaller than, or not "
            "commensurate with, --parent's cell -- --parent itself wasn't given in its own "
            "most-reduced form (a real outcome of `freeze` always picking the smallest "
            "commensurate cell it can)."
        )
        print(enlargement_note, file=sys.stderr)
        sub = big_sub
        stoichiometry_multiplier = decomposition_module.check_stoichiometry(parent, sub)

    # `decomposition`'s own findsym/smodes filenames are FIXED
    # (`parent_findsym`/`subgroup_findsym`/`parent_smodes_<K>`), not
    # tag-prefixed like every other command's -- a real material's own
    # `--name`/POSCAR-title-derived tag can make the usual `{tag}_findsym`-
    # style names unwieldy (confirmed directly for real CMO). This is
    # `decomposition`'s OWN dedicated cache, separate from `{tag}_findsym`
    # (e.g. one `irrep`/`freeze` already built) -- a deliberate, scoped
    # trade-off: `decomposition` no longer transparently reuses those
    # other commands' own cached artifacts, but its own runs (with a
    # given `--parent`) still cache and reuse normally across repeated
    # invocations, same as before, just under these short, fixed names.
    findsym_path = Path("parent_findsym")
    try:
        parent_log_text, symmetry_info = _read_or_build_findsym_log(
            findsym_path, parent, "decomposition parent structure", findsym_tolerance, isodata
        )
        parent_symbol = findsym_module.parse_space_group_symbol(parent_log_text)
    except (FileNotFoundError, RuntimeError, findsym_module.FindsymLogError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    sub_findsym_path = Path("subgroup_findsym")
    try:
        sub_log_text, sub_symmetry_info = _read_or_build_findsym_log(
            sub_findsym_path, sub, "decomposition distorted structure", findsym_tolerance, isodata
        )
        sub_symbol = findsym_module.parse_space_group_symbol(sub_log_text)
        sub_primitive_ratio = findsym_module.parse_primitive_cell_ratio(sub_log_text)
    except (FileNotFoundError, RuntimeError, findsym_module.FindsymLogError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    parent_warning = _parent_supercell_warning(parent, symmetry_info)
    if parent_warning is not None:
        print(parent_warning, file=sys.stderr)

    header = [
        f"Stoichiometry: distorted structure is {stoichiometry_multiplier}x the parent's own formula-unit count.",
        "",
        f"Parent:    space group {symmetry_info.space_group_number} ({parent_symbol}), "
        f"{_format_lattice_parameters(symmetry_info.lattice_parameters)}",
        f"Distorted: space group {sub_symmetry_info.space_group_number} ({sub_symbol}), "
        f"{_format_lattice_parameters(sub_symmetry_info.lattice_parameters)}",
    ]
    if sub_primitive_ratio > 1.0 + 1e-6:
        original_n_atoms = sub.n_atoms
        reduction_note = None
        try:
            reduced_sub = findsym_module.reduce_to_primitive_cell(sub, sub_log_text)
        except findsym_module.FindsymLogError as exc:
            reduced_sub = None
            reduction_note = (
                f"  note: findsym's own analysis of the distorted structure finds a true primitive cell "
                f"{sub_primitive_ratio:g}x SMALLER than the size given here, but it could not be "
                f"parsed into a usable structure ({exc}) -- continuing at the full given size."
            )

        if reduced_sub is not None:
            # The reduction is relative to `sub` alone -- it can shrink past
            # what's still a valid integer supercell of THIS `parent` (e.g. a
            # centered space group where `parent` itself is given in its own
            # conventional, not fully-primitive, cell: `sub`'s true primitive
            # can then come out smaller than even one parent formula unit,
            # confirmed directly against real CMO cross-irrep blend data --
            # `check_stoichiometry` correctly rejects a 0.5x ratio there).
            try:
                new_multiplier = decomposition_module.check_stoichiometry(parent, reduced_sub)
            except decomposition_module.DecompositionError:
                new_multiplier = None

            # Even when the SIZE matches, findsym's own reduction picks
            # WHATEVER lattice vectors its own algorithm naturally lands
            # on -- with no reason to share the parent POSCAR's own axis
            # choice. `align_by_nearest_neighbor` (via `internal_
            # displacement`, used throughout the rest of this command)
            # only ever searches for a TRANSLATION fix, never a full
            # ROTATION -- confirmed directly this session that this can
            # silently break: a real, genuinely-strained material (BaTiO3
            # R3m, reduced from a padded supercell) comes back with
            # completely different axes than the cubic parent uses, even
            # though it's the identical crystal (a cubic case happened to
            # get lucky -- findsym picked axes that were just a
            # permutation/sign-flip of the parent's own). Only ever
            # substitute after directly confirming alignment actually
            # works, and only when the reduction lands EXACTLY at 1x
            # parent's own size, where `parent` itself is a ready-made
            # reference to test against -- a larger multiplier has no
            # simple, k-point/irrep-independent way to build a matching
            # reference here (that needs a specific pattern, not chosen
            # yet), so it's not attempted automatically.
            alignment_ok = False
            if new_multiplier == 1:
                try:
                    freeze_module.align_by_nearest_neighbor(
                        parent.species, parent.frac_coords, reduced_sub.species, reduced_sub.frac_coords,
                        parent.lattice, tol=0.3,
                    )
                    alignment_ok = True
                except freeze_module.FreezeError:
                    _shift, perm = freeze_module.search_alignment_shift(parent, reduced_sub, 0.3)
                    alignment_ok = perm is not None

            if alignment_ok:
                sub = reduced_sub
                stoichiometry_multiplier = new_multiplier
                header[0] = (
                    f"Stoichiometry: distorted structure is {stoichiometry_multiplier}x the parent's own "
                    "formula-unit count (after automatic reduction -- see note below)."
                )
                reduction_note = (
                    f"  note: findsym's own analysis of the distorted structure found a true primitive cell "
                    f"{sub_primitive_ratio:g}x SMALLER than the {original_n_atoms}-atom structure given "
                    "-- the real distortion present doesn't need the whole given cell, so this command "
                    f"now works against the smaller, symmetrically-equivalent {sub.n_atoms}-atom cell "
                    "findsym itself found (same crystal, same physics -- just less redundant padding). "
                    "Pass a smaller --sub directly to skip this note."
                )
            elif new_multiplier is None:
                reduction_note = (
                    f"  note: findsym's own analysis of the distorted structure finds a true primitive cell "
                    f"{sub_primitive_ratio:g}x SMALLER than the size given here, but that smaller cell "
                    "isn't a valid integer supercell of the PARENT as given (likely a centered space "
                    "group where the parent's own POSCAR isn't itself in its most-reduced form) -- "
                    "continuing at the full given size."
                )
            else:
                reduction_note = (
                    f"  note: findsym's own analysis of the distorted structure found a true primitive cell "
                    f"{sub_primitive_ratio:g}x SMALLER than the size given here, but its own lattice "
                    "vector choice doesn't line up with the parent's (a real, non-crashing outcome for "
                    "a genuinely strained/rotated material -- findsym has no reason to match the "
                    "parent's own axis convention) -- continuing at the full given size."
                )
        elif reduction_note is None:
            reduction_note = (
                f"  note: findsym's own analysis of the distorted structure finds a true primitive cell "
                f"{sub_primitive_ratio:g}x SMALLER than the size given here, but it could not be "
                "reduced automatically (an unrecognized centering type, or the standard reduction "
                "transform didn't cleanly pair up every atom for this structure) -- continuing at "
                "the full given size."
            )
        header.append(reduction_note)
    if enlargement_note is not None:
        header.append(f"  {enlargement_note}")
    if parent_warning is not None:
        header.insert(0, parent_warning)
        header.insert(1, "")

    # No iso cross-check here anymore -- removed deliberately, not an
    # oversight. It ran `iso`'s own single parent/subgroup `SHOW SUBGROUP`
    # query and either printed its own duplicate of the real `iso` binary's
    # own `iso.log` session file, or (once that duplication was fixed) just
    # pointed at it -- but running `iso` at all is itself a real, separate
    # workflow step with its own real gotchas (e.g. its own order-
    # sensitivity for a genuinely COUPLED, multi-irrep query, which this
    # project's own iso.py wrapper never even attempts to replicate -- it
    # only ever drives the single-irrep-pair query). Better left as an
    # explicit, manual step the physicist runs directly when they actually
    # want it, rather than something this command runs -- and writes a
    # stray `iso.log` -- on every invocation regardless.

    # The screen gets ONE line per requested k-vector (done, with its
    # residual, or FAILED with the reason, on stderr); every report lives
    # only in its `decomposition_mode_<K>` file and `decomposition_summary`.
    # Printing every report here buried warnings in thousands of lines of
    # batch output (Issue 5, agreed with the physicist).
    overall_ok = True
    # One shared summary, accumulated across every k-vector/irrep requested
    # in THIS invocation -- {label: (k_vector, vector_or_MIXED, magnitude)}.
    # Deliberately a single file (`decomposition_summary`) for the whole
    # command, unlike `decomposition_mode_<K>` below (one file per
    # requested k-vector/label) -- rewritten after every k-vector so a
    # partial run (one bad label among several) still leaves an accurate,
    # up-to-date summary of everything processed so far.
    summary_entries: dict = {}
    # Per k-vector (displacement, residual) norms, for the COMBINED line.
    k_norms: dict = {}
    for requested in k_vectors:
        # A bare k-point (letters only, e.g. 'X') is a catch-all -- expands
        # to every irrep smodes finds there, same convention `irrep`'s own
        # bare-k-point expansion already established (`label.isalpha()` is
        # a safe, exact discriminator: a real irrep label always has a
        # trailing digit). A full irrep label (e.g. 'X2+') instead scopes
        # the decomposition to JUST that one irrep.
        k_vector = requested if requested.isalpha() else smodes_module.k_vector_label_from_irrep(requested)
        smodes_path = Path(f"parent_smodes_{k_vector}")
        try:
            # Always run smodes fresh here too, same reasoning as
            # `_read_or_build_findsym_log` above: `symmetry_info` was just
            # freshly rebuilt from the CURRENT `--parent`, and a leftover
            # `parent_smodes_<K>` on disk from an earlier run/material
            # would otherwise be silently reused with no way to tell it
            # apart from a genuinely current one.
            _build_and_save_smodes_output(symmetry_info, k_vector, tag, out_path=smodes_path)
            smodes_output = smodes_module.parse_smodes_output(smodes_path.read_text())
        except (FileNotFoundError, RuntimeError, smodes_module.SmodesOutputError) as exc:
            print(f"{requested}: FAILED -- {exc}", file=sys.stderr)
            overall_ok = False
            continue

        if requested.isalpha():
            irreps = smodes_output.irreps
            if not irreps:
                print(f"{requested}: FAILED -- no irreps found at this k-point", file=sys.stderr)
                overall_ok = False
                continue
        else:
            try:
                irreps = [smodes_output.irrep(requested)]
            except KeyError as exc:
                print(f"{requested}: FAILED -- {exc}", file=sys.stderr)
                overall_ok = False
                continue

        # Everything the decomposition prints while it runs is collected, so
        # its notes land in this k-vector's report file (once each) instead
        # of on the screen -- a fallback note could repeat once per irrep
        # (7 near-identical lines for one real CMO request). Warnings still
        # reach the screen, also once each.
        captured = io.StringIO()
        try:
            with contextlib.redirect_stderr(captured):
                result = decomposition_module.decompose_at_k_vector(
                    k_vector, irreps, parent, sub, stoichiometry_multiplier,
                    tol=tolerance, findsym_log_text=parent_log_text,
                )
        except decomposition_module.DecompositionError as exc:
            notes, warnings_text = _split_decomposition_messages(captured.getvalue())
            for line in notes + warnings_text:
                print(line, file=sys.stderr)
            print(f"{requested}: FAILED -- {exc}", file=sys.stderr)
            overall_ok = False
            continue
        notes, warnings_text = _split_decomposition_messages(captured.getvalue())
        for line in warnings_text:
            print(line, file=sys.stderr)

        report_text = _format_k_vector_decomposition_report(result)
        if notes or warnings_text:
            report_text += "\n\nNotes:\n" + "\n".join(f"  {line}" for line in warnings_text + notes)

        # Saved in addition to being printed -- one self-contained file per
        # requested k-vector/irrep label (the shared header plus this
        # specific result), named after exactly what was typed (`GM`, `X`,
        # or a full irrep label like `X2+`) so it's clear which request
        # produced it.
        report_path = Path(f"decomposition_mode_{requested}")
        report_path.write_text("\n".join(header) + "\n\n" + report_text + "\n")
        for line in _screen_lines_for_k_vector(requested, result):
            print(line)

        for irrep_label, (vector, magnitude) in _irrep_summary_entries(result.amplitudes).items():
            summary_entries[irrep_label] = (k_vector, vector, magnitude)
        k_norms[k_vector] = (result.displacement_norm_angstrom,
                             result.residual_norm_angstrom)
        summary_path = Path("decomposition_summary")
        summary_path.write_text(
            _format_decomposition_summary(summary_entries, k_norms) + "\n")

    if summary_entries or k_norms:
        print("Full reports: decomposition_mode_<K>; summary: decomposition_summary")
    return 0 if overall_ok else 1


def _format_landau_report(fit, predicted: dict | None) -> str:
    lines = [f"Fitted {len(fit.terms)} coefficient(s), R^2 = {fit.r_squared:.8f}", ""]
    for term, coeff in zip(fit.terms, fit.coefficients):
        lines.append(f"  {fit.term_label(term):>12}  {coeff: .8f}")
    if predicted is not None:
        lines.append("")
        lines.append("Predicted equilibrium (energy-minimizing order parameters):")
        for v in fit.variables:
            lines.append(f"  {v} = {predicted[v]: .8f}")
        lines.append(f"  energy there: {landau_module.polynomial_energy(fit, predicted):.8f}")
        origin = {v: 0.0 for v in fit.variables}
        lines.append(f"  energy at origin (all zero): {landau_module.polynomial_energy(fit, origin):.8f}")
    return "\n".join(lines)


def _run_landau_fit(
    invariants_path: str | None,
    terms: str | None,
    data_path: str,
    predict: bool,
    bound: float,
    n_starts: int,
) -> int:
    try:
        if invariants_path is not None:
            invariants = landau_module.parse_invariants(Path(invariants_path).read_text())
        else:
            invariants = landau_module.parse_inline_terms(terms)
    except (FileNotFoundError, landau_module.LandauError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if invariants.irreps:
        print(
            "Order parameters: "
            + ", ".join(f"{irrep.variable}={irrep.irrep_label}" for irrep in invariants.irreps if irrep.variable)
        )

    try:
        data_points = landau_module.parse_data_table(Path(data_path).read_text(), invariants.variables)
    except (FileNotFoundError, landau_module.LandauError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    try:
        fit = landau_module.fit_landau_coefficients(invariants, data_points)
    except landau_module.LandauError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    predicted = None
    if predict:
        predicted = landau_module.predict_equilibrium(fit, bound=bound, n_starts=n_starts)

    print(_format_landau_report(fit, predicted))
    return 0


def _run_landau_grid(
    invariants_path: str,
    start: float,
    end: float,
    steps: int,
    poscar_path: str,
    name: str | None,
    mode_args: list[str],
    isodata: str | None,
    kpoints_path: str = "KPOINTS",
    incar_path: str = "INCAR",
    magmom: bool = False,
    direction_args: list[str] | None = None,
) -> int:
    try:
        invariants = landau_module.parse_invariants(Path(invariants_path).read_text())
    except (FileNotFoundError, landau_module.LandauError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    resolved = [irrep for irrep in invariants.irreps if irrep.variable is not None]
    unresolved = [irrep for irrep in invariants.irreps if irrep.variable is None]
    if unresolved:
        print(
            "Warning: could not determine the order-parameter symbol for irrep(s) "
            + ", ".join(irrep.irrep_label for irrep in unresolved)
            + " from the invariants file's own header -- skipping them.",
            file=sys.stderr,
        )
    if not resolved:
        print(f"{invariants_path}: no irrep could be resolved to an order-parameter symbol", file=sys.stderr)
        return 1

    # Emit in invariants.variables' own canonical order, matching what
    # landau-fit's parse_data_table later expects for the "energy" column
    # to sit alongside.
    by_variable = {irrep.variable: irrep for irrep in resolved}
    ordered = [by_variable[v] for v in invariants.variables if v in by_variable]

    irrep_labels = [irrep.irrep_label for irrep in ordered]
    column_labels = [irrep.variable for irrep in ordered]

    try:
        _check_no_duplicate_irreps(irrep_labels)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Order parameters: " + ", ".join(f"{v}={label}" for v, label in zip(column_labels, irrep_labels)))

    try:
        mode_overrides = _parse_label_value_pairs(mode_args, int, "--mode")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    return _run_freeze_amplitude_grid(
        irrep_labels,
        start,
        end,
        steps,
        poscar_path,
        name,
        mode_overrides,
        direction_args,
        isodata,
        column_labels,
        column_header_suffix="",
        kpoints_path=kpoints_path,
        incar_path=incar_path,
        magmom=magmom,
    )


def _normalize_label_args(args: argparse.Namespace) -> None:
    """Irrep and k-point labels are matched case-insensitively everywhere
    -- every SMODES-derived label this project ever compares against is
    itself always upper-case (a real ISOTROPY/SMODES convention, not just
    a convention of this project's own fixtures), so normalizing every
    user-typed label to upper case here, once, right after parsing, means
    every downstream command sees the same canonical form regardless of
    how the user typed it (`gm4-`, `GM4-`, `Gm4-` are all the same
    request). Only touches attributes that are genuinely label-shaped
    (never a file path, a number, or an arm index) -- `hasattr` guards
    mean this is safe to call unconditionally for every command, since
    most of these attributes simply don't exist on most parsers.

    `--mode`/`--amplitude`/`--direction` are deliberately NOT handled
    here even though some commands' versions carry a 'LABEL=VALUE' irrep
    prefix -- their argparse shape varies per command (a plain int/str
    for the single-irrep commands, e.g. `star --mode`, vs. a repeatable
    'LABEL=VALUE' string list for the multi-irrep ones, e.g. `freeze
    --mode`), so blindly upper-casing here would crash or silently
    corrupt the single-irrep commands' own bare values. The 'LABEL='
    prefix case is instead normalized inside `_parse_label_value_pairs`/
    `_resolve_symbolic_direction_specs` themselves, the one place that
    shape is actually parsed.
    """
    if getattr(args, "irreps", None) is not None:
        args.irreps = [label.upper() for label in args.irreps]
    if getattr(args, "irrep", None) is not None:
        args.irrep = args.irrep.upper()
    if getattr(args, "k_vector", None) is not None:
        if isinstance(args.k_vector, str):
            # `run-smodes`'s own positional k_vector: a single label.
            args.k_vector = args.k_vector.upper()
        else:
            # `decomposition`'s own `--k_vector` (action="append", nargs="+"):
            # a list of lists, e.g. [["GM"], ["X", "R"]] for
            # `--k_vector GM --k_vector X R`. Flatten, upper-case, and
            # dedupe (preserving first-seen order) here so downstream code
            # never has to know which invocation style the user chose.
            flattened = [label.upper() for group in args.k_vector for label in group]
            args.k_vector = list(dict.fromkeys(flattened))


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _normalize_label_args(args)
    _configure_iso_runner()

    if args.command == "doctor":
        return doctor.run(isodata=args.isodata)
    if args.command == "smodes":
        return _run_smodes_parse(args.output_file)
    if args.command == "symmetry":
        return _run_symmetry(args.poscar, args.name, args.tolerance, args.isodata)
    if args.command == "run-smodes":
        return _run_run_smodes(args.findsym_log, args.k_vector, args.poscar, args.name)
    if args.command == "irrep":
        return _run_irrep(
            args.irreps,
            args.poscar,
            args.name,
            args.tolerance,
            args.isodata,
            args.kpoints,
            args.incar,
            write_bec=args.bec,
            magmom=args.magmom,
            write_opd=args.opd,
        )
    if args.command == "bec":
        return _run_bec(
            args.poscar, args.name, args.tolerance, args.isodata, args.kpoints, args.incar, magmom=args.magmom
        )
    if args.command == "bec-analysis":
        return _run_bec_analysis(args.born_charges, args.poscar, args.name, args.tolerance, args.isodata)
    if args.command == "phonons":
        return _run_phonons_labels(
            args.irreps, args.poscar, args.name, args.born_charges, args.direction,
            args.tolerance, args.isodata, generate_opd=args.generate_opd,
        )
    if args.command == "freeze":
        return _run_freeze(
            args.irreps,
            args.poscar,
            args.name,
            args.mode,
            args.amplitude,
            args.direction,
            args.isodata,
            args.kpoints,
            args.incar,
            magmom=args.magmom,
            smodes_basis=args.smodes_basis,
        )
    if args.command == "freeze-amplitude":
        return _run_freeze_amplitude_grid_cli(
            args.irreps,
            args.start,
            args.end,
            args.steps,
            args.poscar,
            args.name,
            args.mode,
            args.direction,
            args.isodata,
            args.kpoints,
            args.incar,
            magmom=args.magmom,
        )
    if args.command == "decomposition":
        return _run_decomposition(
            args.parent,
            args.sub,
            args.k_vector,
            args.name,
            args.tolerance,
            args.findsym_tolerance,
            args.isodata,
        )
    if args.command == "landau-fit":
        return _run_landau_fit(
            args.invariants,
            args.terms,
            args.data,
            args.predict,
            args.bound,
            args.n_starts,
        )
    if args.command == "landau-grid":
        return _run_landau_grid(
            args.invariants,
            args.start,
            args.end,
            args.steps,
            args.poscar,
            args.name,
            args.mode,
            args.isodata,
            args.kpoints,
            args.incar,
            magmom=args.magmom,
            direction_args=args.direction,
        )
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())

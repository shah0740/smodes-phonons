"""Detection of the external ISOTROPY suite (findsym, smodes).

findsym and smodes are compiled Fortran binaries distributed by BYU under
their own terms. This module never bundles or redistributes them -- it only
locates and validates an existing install.

ISODATA resolution order, most specific wins:
  1. explicit value passed in (e.g. from a --isodata CLI flag)
  2. project config: ./smodes_phonons.cfg in the current directory
  3. user config: ~/.config/smodes_phonons/config.cfg
  4. ISODATA environment variable
  5. none -- fall back to whatever findsym on PATH does by default
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from . import config

PROJECT_CONFIG_NAME = config.PROJECT_CONFIG_NAME
USER_CONFIG_PATH = config.USER_CONFIG_PATH


@dataclass
class IsodataResolution:
    value: str | None
    source: str  # "--isodata flag", "project config", "user config", "ISODATA env var", "none"


def resolve_isodata(cli_value: str | None = None, project_dir: Path | None = None) -> IsodataResolution:
    """Resolve the ISODATA directory using the documented precedence order."""
    if cli_value:
        return IsodataResolution(cli_value, "--isodata flag")

    project_dir = project_dir or Path.cwd()
    project_value = config.read_value(project_dir / PROJECT_CONFIG_NAME, "isotropy", "isodata")
    if project_value:
        return IsodataResolution(project_value, "project config")

    user_value = config.read_value(USER_CONFIG_PATH, "isotropy", "isodata")
    if user_value:
        return IsodataResolution(user_value, "user config")

    env_value = os.environ.get("ISODATA")
    if env_value:
        return IsodataResolution(env_value, "ISODATA env var")

    return IsodataResolution(None, "none")


def find_binary(name: str) -> str | None:
    """Locate an ISOTROPY binary (findsym, smodes) on PATH."""
    return shutil.which(name)


# Wall-clock ceiling for a real findsym/smodes run. Generous on purpose --
# MINUTES, not seconds -- because a big supercell is legitimately slow and a
# tight limit would kill correct work. What it stops is a genuine HANG:
# `detect_findsym_version` already passed `timeout=30`, but the real
# `run_findsym` and `run_smodes` calls passed none, so a wedged invocation
# blocked its caller forever. Seen in practice, not hypothetical -- a findsym
# Docker container was found at 100% CPU TEN HOURS after it started, long
# outliving the Python process that spawned it, on input findsym normally
# answers in seconds. See known issue 6 in the developer notes.
#
# NOTE, and it is why this is only half a fix: under the macOS Docker
# wrapper, killing the client does NOT kill the container. A Python-side
# timeout ends the HANG for the caller but the container keeps burning CPU
# until `docker kill`. That half needs the wrapper itself.
ISOTROPY_SUBPROCESS_TIMEOUT_SECONDS = 600


class IsotropyTimeoutError(RuntimeError):
    """A findsym or smodes invocation exceeded
    `ISOTROPY_SUBPROCESS_TIMEOUT_SECONDS`."""

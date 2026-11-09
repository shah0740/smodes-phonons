"""Shared project/user config file (smodes_phonons.cfg) resolution.

Used for both the ISODATA path (isotropy.py) and the material tag that
drives generated filenames (the developer notes' "File naming" section). Both live
in the same small config file so a project directory has one place to look,
not one config file per feature.
"""

from __future__ import annotations

import configparser
import re
import sys
from pathlib import Path

PROJECT_CONFIG_NAME = "smodes_phonons.cfg"
USER_CONFIG_PATH = Path.home() / ".config" / "smodes_phonons" / "config.cfg"

_TAG_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_]+")


def read_value(path: Path, section: str, key: str) -> str | None:
    if not path.is_file():
        return None
    parser = configparser.ConfigParser()
    parser.read(path)
    if parser.has_option(section, key):
        value = parser.get(section, key).strip()
        return value or None
    return None


def write_value(path: Path, section: str, key: str, value: str) -> None:
    parser = configparser.ConfigParser()
    if path.is_file():
        parser.read(path)
    if not parser.has_section(section):
        parser.add_section(section)
    parser.set(section, key, value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        parser.write(fh)


def _sanitize_tag(raw: str) -> str | None:
    """POSCAR title line -> filename-safe tag, or None if it has no usable content.

    "strip whitespace, replace anything that is not alphanumeric or
    underscore" (the developer notes). Runs of invalid characters collapse to a single
    underscore rather than one-per-character. A title of only punctuation/
    whitespace (e.g. "---") sanitizes to a lone "_", which isn't a usable
    tag either -- the isalnum() guard catches that and falls through to the
    next resolution tier instead of silently naming everything "_findsym".
    """
    stripped = raw.strip()
    if not any(c.isalnum() for c in stripped):
        return None
    return _TAG_SANITIZE_RE.sub("_", stripped) or None


def resolve_tag(
    cli_value: str | None = None,
    project_dir: Path | None = None,
    stdin=None,
    poscar_title: str | None = None,
) -> str:
    """Resolve the material tag that drives generated filenames (e.g. 'STO').

    Precedence: --name flag, then [project] name in the project config file,
    then the (sanitized) first line of the POSCAR, then an interactive
    prompt -- but only if stdin is a TTY. Whichever of the two *derived*
    tiers (POSCAR title, prompt) fires gets saved to the project config, so
    later commands in the same directory are consistent even if the POSCAR
    is edited afterward, and non-interactive runs find it without prompting.

    --name is deliberately never persisted: it's an explicit, same-invocation
    override, not a way to silently change what a later bare command
    resolves to. If a tag should stick, the diligent way is editing the
    POSCAR title (or leaning on the project config directly), not --name.

    Never prompts unconditionally: a non-interactive caller with nothing
    stored gets a clear error instead of a hang, per the developer notes' "irrep will
    be called from submit scripts and loops, and a blocking prompt hangs
    the job."
    """
    if cli_value:
        return cli_value

    project_dir = project_dir or Path.cwd()
    project_path = project_dir / PROJECT_CONFIG_NAME
    stored = read_value(project_path, "project", "name")
    if stored:
        return stored

    derived = _sanitize_tag(poscar_title) if poscar_title else None
    if derived:
        write_value(project_path, "project", "name", derived)
        return derived

    stdin = stdin if stdin is not None else sys.stdin
    if not stdin.isatty():
        raise RuntimeError(
            "No material tag set, and stdin is not a terminal so this won't prompt "
            f"for one (that would hang a non-interactive run). Pass --name, or set "
            f"'name' under [project] in {project_path}."
        )

    tag = input("Material tag for generated filenames (e.g. STO): ").strip()
    if not tag:
        raise RuntimeError("Material tag cannot be empty.")
    write_value(project_path, "project", "name", tag)
    return tag

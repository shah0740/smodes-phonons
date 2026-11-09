import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from smodes_phonons import config


class _FakeStdin:
    def __init__(self, is_tty: bool):
        self._is_tty = is_tty

    def isatty(self) -> bool:
        return self._is_tty


class TestReadWriteValue(unittest.TestCase):
    def test_round_trip(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "smodes_phonons.cfg"
            self.assertIsNone(config.read_value(path, "project", "name"))
            config.write_value(path, "project", "name", "STO")
            self.assertEqual(config.read_value(path, "project", "name"), "STO")

    def test_write_preserves_other_sections(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "smodes_phonons.cfg"
            config.write_value(path, "isotropy", "isodata", "/some/path/")
            config.write_value(path, "project", "name", "STO")
            self.assertEqual(config.read_value(path, "isotropy", "isodata"), "/some/path/")
            self.assertEqual(config.read_value(path, "project", "name"), "STO")


class TestResolveTag(unittest.TestCase):
    def test_cli_value_wins_without_touching_disk(self):
        with TemporaryDirectory() as tmp:
            tag = config.resolve_tag("Explicit", project_dir=Path(tmp))
        self.assertEqual(tag, "Explicit")

    def test_stored_value_used_without_prompting(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            config.write_value(project_dir / config.PROJECT_CONFIG_NAME, "project", "name", "STO")
            # Non-TTY stdin: if this tried to prompt, it would raise instead.
            tag = config.resolve_tag(None, project_dir=project_dir, stdin=_FakeStdin(False))
        self.assertEqual(tag, "STO")

    def test_noninteractive_with_nothing_stored_raises(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                config.resolve_tag(None, project_dir=Path(tmp), stdin=_FakeStdin(False))

    def test_interactive_prompt_persists_for_next_time(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            with patch("builtins.input", return_value="STO"):
                tag = config.resolve_tag(None, project_dir=project_dir, stdin=_FakeStdin(True))
            self.assertEqual(tag, "STO")

            # Second call, still non-interactive: must find the persisted value
            # rather than needing to prompt again.
            tag_again = config.resolve_tag(None, project_dir=project_dir, stdin=_FakeStdin(False))
            self.assertEqual(tag_again, "STO")

    def test_empty_prompt_response_raises(self):
        with TemporaryDirectory() as tmp:
            with patch("builtins.input", return_value="   "):
                with self.assertRaises(RuntimeError):
                    config.resolve_tag(None, project_dir=Path(tmp), stdin=_FakeStdin(True))

    def test_poscar_title_used_without_prompting(self):
        with TemporaryDirectory() as tmp:
            # Non-TTY stdin: if this fell through to the prompt, it would raise.
            tag = config.resolve_tag(
                None, project_dir=Path(tmp), stdin=_FakeStdin(False), poscar_title="STO Cubic Perovskite"
            )
        self.assertEqual(tag, "STO_Cubic_Perovskite")

    def test_poscar_title_persists_for_next_time(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            tag = config.resolve_tag(
                None, project_dir=project_dir, stdin=_FakeStdin(False), poscar_title="STO"
            )
            self.assertEqual(tag, "STO")

            # Second call, no poscar_title given at all: must find the persisted
            # value rather than needing it again.
            tag_again = config.resolve_tag(None, project_dir=project_dir, stdin=_FakeStdin(False))
            self.assertEqual(tag_again, "STO")

    def test_punctuation_only_poscar_title_falls_through_to_prompt(self):
        with TemporaryDirectory() as tmp:
            with patch("builtins.input", return_value="STO"):
                tag = config.resolve_tag(
                    None, project_dir=Path(tmp), stdin=_FakeStdin(True), poscar_title="---"
                )
        self.assertEqual(tag, "STO")

    def test_whitespace_only_poscar_title_falls_through_to_prompt(self):
        with TemporaryDirectory() as tmp:
            with patch("builtins.input", return_value="STO"):
                tag = config.resolve_tag(
                    None, project_dir=Path(tmp), stdin=_FakeStdin(True), poscar_title="   "
                )
        self.assertEqual(tag, "STO")

    def test_poscar_title_does_not_override_already_stored_tag(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            config.write_value(project_dir / config.PROJECT_CONFIG_NAME, "project", "name", "STO")
            tag = config.resolve_tag(
                None, project_dir=project_dir, stdin=_FakeStdin(False), poscar_title="SomethingElse"
            )
        self.assertEqual(tag, "STO")

    def test_cli_value_does_not_persist_even_with_poscar_title(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            tag = config.resolve_tag("Explicit", project_dir=project_dir, poscar_title="STO")
            self.assertEqual(tag, "Explicit")
            self.assertIsNone(config.read_value(project_dir / config.PROJECT_CONFIG_NAME, "project", "name"))

    def test_full_precedence_order(self):
        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            # cli_value wins even with a stored value and a poscar_title present.
            config.write_value(project_dir / config.PROJECT_CONFIG_NAME, "project", "name", "Stored")
            tag = config.resolve_tag("FromFlag", project_dir=project_dir, poscar_title="FromPoscar")
            self.assertEqual(tag, "FromFlag")

        with TemporaryDirectory() as tmp:
            project_dir = Path(tmp)
            # stored value wins over poscar_title when no cli_value given.
            config.write_value(project_dir / config.PROJECT_CONFIG_NAME, "project", "name", "Stored")
            tag = config.resolve_tag(
                None, project_dir=project_dir, stdin=_FakeStdin(False), poscar_title="FromPoscar"
            )
            self.assertEqual(tag, "Stored")


class TestSanitizeTag(unittest.TestCase):
    def test_collapses_runs_of_invalid_characters(self):
        self.assertEqual(config._sanitize_tag("SrTiO3  Cubic   Perovskite"), "SrTiO3_Cubic_Perovskite")

    def test_strips_leading_and_trailing_whitespace(self):
        self.assertEqual(config._sanitize_tag("  STO  "), "STO")

    def test_all_punctuation_returns_none(self):
        self.assertIsNone(config._sanitize_tag("---"))

    def test_all_whitespace_returns_none(self):
        self.assertIsNone(config._sanitize_tag("   "))

    def test_alphanumeric_and_underscore_pass_through_unchanged(self):
        self.assertEqual(config._sanitize_tag("STO_221_R4plus"), "STO_221_R4plus")


if __name__ == "__main__":
    unittest.main()

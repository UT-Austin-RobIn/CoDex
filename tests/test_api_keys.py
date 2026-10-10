"""Saved API keys: private file, environment precedence, masking."""

import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

from codex.utils import api_keys


class ApiKeysTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.file = Path(self.dir.name) / "keys.json"
        clean = {k: v for k, v in os.environ.items() if k not in api_keys.KEYS}
        self.env = mock.patch.dict(os.environ, dict(clean, CODEX_KEYS_FILE=str(self.file)), clear=True)
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.dir.cleanup()

    def test_saved_key_is_private_and_applied(self):
        api_keys.save_key("TRIPO_API_KEY", " tsk_abcdefghijklmnop ")
        self.assertEqual(stat.S_IMODE(self.file.stat().st_mode), 0o600)
        self.assertEqual(os.environ["TRIPO_API_KEY"], "tsk_abcdefghijklmnop")
        self.assertEqual(api_keys.saved_keys(), {"TRIPO_API_KEY": "tsk_abcdefghijklmnop"})
        self.assertEqual(api_keys.key_source("TRIPO_API_KEY"), "saved")

    def test_environment_takes_precedence_over_saved(self):
        api_keys.save_key("OPENAI_API_KEY", "sk-saved-0000000000")
        os.environ["OPENAI_API_KEY"] = "sk-env-1111111111"
        api_keys.load_saved_keys()
        self.assertEqual(api_keys.get_key("OPENAI_API_KEY"), "sk-env-1111111111")
        self.assertEqual(api_keys.key_source("OPENAI_API_KEY"), "environment")

    def test_saved_key_fills_a_missing_variable(self):
        api_keys.save_key("TRIPO_API_KEY", "tsk-saved-2222222222")
        del os.environ["TRIPO_API_KEY"]
        self.assertEqual(api_keys.get_key("TRIPO_API_KEY"), "tsk-saved-2222222222")
        api_keys.load_saved_keys()
        self.assertEqual(os.environ["TRIPO_API_KEY"], "tsk-saved-2222222222")

    def test_masking_hides_all_but_the_ends(self):
        self.assertEqual(api_keys.masked("sk-proj-abcdefghwxyz"), "sk-" + "*" * 12 + "wxyz")
        self.assertEqual(api_keys.masked("short"), "*****")
        self.assertEqual(api_keys.masked(None), "")

    def test_empty_value_removes_the_saved_key(self):
        api_keys.save_key("TRIPO_API_KEY", "tsk_abcdefghijklmnop")
        api_keys.save_key("TRIPO_API_KEY", "")
        self.assertEqual(api_keys.saved_keys(), {})

    def test_default_file_is_in_the_repo_and_ignored_by_git(self):
        root = Path(__file__).resolve().parents[1]
        self.assertEqual(api_keys.DEFAULT_FILE, root / ".api_keys.json")
        probe = subprocess.run(["git", "check-ignore", "-q", str(api_keys.DEFAULT_FILE)], cwd=root)
        if probe.returncode == 128:
            self.skipTest("not a git checkout")
        self.assertEqual(probe.returncode, 0)


if __name__ == "__main__":
    unittest.main()

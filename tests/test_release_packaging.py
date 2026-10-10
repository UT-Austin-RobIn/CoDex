"""Installer patch behavior without downloading or building FoundationPose."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts" / "apply_foundationpose_patch.sh"


class FoundationPosePatchTest(unittest.TestCase):
    def test_apply_rerun_and_reject_unknown_edits(self):
        with tempfile.TemporaryDirectory() as scratch:
            repo = Path(scratch) / "fork"
            repo.mkdir()
            env = dict(os.environ)

            def git(*args):
                return subprocess.run(
                    ["git", "-C", str(repo), *args], check=True,
                    capture_output=True, text=True, env=env,
                ).stdout.strip()

            def apply(base, patched, patch):
                return subprocess.run(
                    ["bash", str(HELPER), str(repo), base, patched, str(patch)],
                    capture_output=True, text=True, env=env,
                )

            git("init", "-q")
            git("config", "user.name", "Release test")
            git("config", "user.email", "release-test@example.invalid")
            source = repo / "foundationpose" / "estimater.py"
            source.parent.mkdir()
            source.write_text("base\n")
            git("add", ".")
            git("commit", "-qm", "base")
            base = git("rev-parse", "HEAD")
            source.write_text("base\nvalidated fix\n")
            git("commit", "-qam", "validated fix")
            patched = git("rev-parse", "HEAD")
            patch = Path(scratch) / "validated.patch"
            patch.write_text(git("diff", f"{base}..{patched}") + "\n")

            git("checkout", "-q", "--detach", base)
            self.assertEqual(apply(base, patched, patch).returncode, 0)
            self.assertEqual(git("diff", "HEAD"), patch.read_text().strip())
            self.assertEqual(apply(base, patched, patch).returncode, 0)

            git("reset", "--hard", "-q", base)
            git("checkout", "-q", "--detach", patched)
            self.assertEqual(apply(base, patched, patch).returncode, 0)
            with source.open("a") as stream:
                stream.write("unknown edit\n")
            self.assertNotEqual(apply(base, patched, patch).returncode, 0)

            git("reset", "--hard", "-q", base)
            with source.open("a") as stream:
                stream.write("unknown edit\n")
            self.assertNotEqual(apply(base, patched, patch).returncode, 0)


if __name__ == "__main__":
    unittest.main()

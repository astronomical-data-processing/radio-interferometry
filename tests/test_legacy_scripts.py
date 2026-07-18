import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LegacyScriptTests(unittest.TestCase):
    def test_python_sources_compile(self):
        for path in ROOT.rglob("*.py"):
            with self.subTest(path=path.relative_to(ROOT)):
                source = path.read_text(encoding="utf-8")
                compile(source, str(path), "exec")

    def test_shell_scripts_parse(self):
        scripts = sorted((ROOT / "data/scripts").glob("*.sh"))
        subprocess.run(["bash", "-n", *map(str, scripts)], check=True)


if __name__ == "__main__":
    unittest.main()

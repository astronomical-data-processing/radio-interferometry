import json
import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
PRACTICAL = ROOT / "9_Practical"
MANIFEST = PRACTICAL / "book_manifest.yaml"
NOTEBOOK_LINK = re.compile(r"\]\(([^)#?]+\.ipynb)(?:#[^)]+)?\)")
TYPE_LABELS = {
    "lecture": "讲授",
    "experiment": "实验",
    "reference": "参考",
    "project": "项目",
}
LEVEL_LABELS = {"core": "核心", "advanced": "提高", "course": "课程"}


def load_manifest():
    return yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))


class PracticalBookStructureTests(unittest.TestCase):
    def test_manifest_covers_practical_book(self):
        manifest = load_manifest()
        paths = [record["path"] for record in manifest["pages"]]
        compatibility = set(manifest["compatibility_notebooks"])
        expected = {
            path.name for path in PRACTICAL.glob("*.ipynb")
            if path.name != manifest["entrypoint"] and path.name not in compatibility
        }

        self.assertEqual(len(paths), len(set(paths)))
        self.assertEqual(set(paths), expected)
        self.assertTrue(all((PRACTICAL / path).is_file() for path in paths))
        self.assertTrue(all(record["type"] in manifest["page_types"] for record in manifest["pages"]))
        self.assertTrue(all(record["track"] in manifest["tracks"] for record in manifest["pages"]))
        self.assertTrue(all(record["level"] in manifest["levels"] for record in manifest["pages"]))

    def test_entrypoint_links_every_page(self):
        manifest = load_manifest()
        entrypoint = PRACTICAL / manifest["entrypoint"]
        notebook = json.loads(entrypoint.read_text(encoding="utf-8"))
        markdown = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "markdown"
        )
        linked = {
            target for target in NOTEBOOK_LINK.findall(markdown)
            if Path(target).parent == Path(".")
            and (PRACTICAL / target).is_file()
            and target != manifest["entrypoint"]
        }
        self.assertEqual(linked, {record["path"] for record in manifest["pages"]})

    def test_experiment_pages_are_executable(self):
        for record in load_manifest()["pages"]:
            if record["type"] != "experiment":
                continue
            notebook = json.loads((PRACTICAL / record["path"]).read_text(encoding="utf-8"))
            code = [
                cell for cell in notebook["cells"]
                if cell["cell_type"] == "code" and "".join(cell.get("source", [])).strip()
            ]
            with self.subTest(path=record["path"]):
                self.assertTrue(code)

    def test_entrypoint_displays_manifest_labels(self):
        manifest = load_manifest()
        lines = (PRACTICAL / manifest["entrypoint"]).read_text(encoding="utf-8").splitlines()
        for record in manifest["pages"]:
            label = f"**{TYPE_LABELS[record['type']]} · {LEVEL_LABELS[record['level']]}**"
            line = next(
                value for value in lines
                if f"]({record['path']})" in value and "**" in value
            )
            with self.subTest(path=record["path"]):
                self.assertIn(label, line)


if __name__ == "__main__":
    unittest.main()

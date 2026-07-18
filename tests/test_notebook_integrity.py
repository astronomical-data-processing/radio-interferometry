import json
import re
import unittest
from collections import Counter, defaultdict, deque
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(ROOT.rglob("*.ipynb"))
LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
CELL_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
ANCHOR_PATTERN = re.compile(r"<a\s+[^>]*(?:id|name)=['\"]([^'\"]+)['\"]", re.I)
COMPATIBILITY_NOTEBOOKS = {
    ROOT / "8_Calibration/8_x_further_reading_and_references.ipynb",
    ROOT / "9_Practical/9_3_Observing_smearing.ipynb",
    ROOT / "9_Practical/pimaging.ipynb",
}


def load_notebook(path):
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_text(path):
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in load_notebook(path)["cells"]
        if cell.get("cell_type") == "markdown"
    )


def local_target(source, target):
    if urlparse(target).scheme or target.startswith(("#", "mailto:", "data:", "//")):
        return None
    local = unquote(target.split("#", 1)[0].split("?", 1)[0])
    return (source.parent / local).resolve()


class HTMLReferenceParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []

    def handle_starttag(self, tag, attrs):
        for name, value in attrs:
            if name in {"href", "src"} and value:
                self.targets.append(value)


class NotebookIntegrityTests(unittest.TestCase):
    def test_python3_kernel_and_cell_ids(self):
        self.assertGreater(len(NOTEBOOKS), 100)
        for path in NOTEBOOKS:
            notebook = load_notebook(path)
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "python3")
                self.assertGreaterEqual(notebook.get("nbformat_minor", 0), 5)
                cell_ids = [cell.get("id", "") for cell in notebook["cells"]]
                self.assertEqual(len(cell_ids), len(set(cell_ids)))
                self.assertTrue(all(CELL_ID_PATTERN.fullmatch(value) for value in cell_ids))

    def test_no_stored_error_outputs(self):
        for path in NOTEBOOKS:
            for cell in load_notebook(path)["cells"]:
                errors = [
                    output for output in cell.get("outputs", [])
                    if output.get("output_type") == "error"
                ]
                with self.subTest(path=path.relative_to(ROOT), cell=cell["id"]):
                    self.assertEqual(errors, [])

    def test_sources_have_no_control_characters(self):
        for path in NOTEBOOKS:
            for cell in load_notebook(path)["cells"]:
                source = "".join(cell.get("source", []))
                controls = sorted({ord(char) for char in source if ord(char) < 32 and char != "\n"})
                with self.subTest(path=path.relative_to(ROOT), cell=cell["id"]):
                    self.assertEqual(controls, [])

    def test_local_markdown_links_exist(self):
        sources = [(path, path.read_text(encoding="utf-8")) for path in ROOT.rglob("*.md")]
        for path in NOTEBOOKS:
            sources.append((path, markdown_text(path)))

        for path, text in sources:
            for match in LINK_PATTERN.finditer(text):
                target = match.group(1).strip().split()[0].strip("<>")
                local = local_target(path, target)
                if local is None:
                    continue
                with self.subTest(path=path.relative_to(ROOT), target=target):
                    self.assertTrue(local.exists())

    def test_local_html_references_exist(self):
        for path in NOTEBOOKS:
            parser = HTMLReferenceParser()
            parser.feed(markdown_text(path))
            for target in parser.targets:
                local = local_target(path, target)
                if local is None:
                    continue
                with self.subTest(path=path.relative_to(ROOT), target=target):
                    self.assertTrue(local.exists())

    def test_explicit_anchors_are_unique(self):
        for path in NOTEBOOKS:
            anchors = ANCHOR_PATTERN.findall(markdown_text(path))
            duplicates = sorted(name for name, count in Counter(anchors).items() if count > 1)
            with self.subTest(path=path.relative_to(ROOT)):
                self.assertEqual(duplicates, [])

    def test_explicit_anchor_links_resolve(self):
        anchors = {
            path: set(ANCHOR_PATTERN.findall(markdown_text(path)))
            for path in NOTEBOOKS
        }
        for source in NOTEBOOKS:
            for match in LINK_PATTERN.finditer(markdown_text(source)):
                target = match.group(1).strip().split()[0].strip("<>")
                fragment = unquote(urlparse(target).fragment)
                if ":" not in fragment:
                    continue
                local = source if target.startswith("#") else local_target(source, target)
                if local not in anchors:
                    continue
                with self.subTest(path=source.relative_to(ROOT), target=target):
                    self.assertIn(fragment, anchors[local])

    def test_teaching_notebooks_are_reachable_from_contents(self):
        notebook_set = set(NOTEBOOKS)
        links = defaultdict(set)
        for path in NOTEBOOKS:
            for match in LINK_PATTERN.finditer(markdown_text(path)):
                target = match.group(1).strip().split()[0].strip("<>")
                local = local_target(path, target)
                if local in notebook_set:
                    links[path].add(local)

        start = ROOT / "0_Introduction/0_introduction.ipynb"
        reachable = {start}
        pending = deque([start])
        while pending:
            for target in links[pending.popleft()] - reachable:
                reachable.add(target)
                pending.append(target)

        expected = notebook_set - COMPATIBILITY_NOTEBOOKS
        self.assertEqual(
            sorted(path.relative_to(ROOT) for path in expected - reachable),
            [],
        )


if __name__ == "__main__":
    unittest.main()

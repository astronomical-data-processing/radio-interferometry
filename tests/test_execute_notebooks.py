import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nbformat

from tools import execute_notebooks


def write_notebook(path, cell_type, source):
    notebook = {
        "cells": [{"cell_type": cell_type, "source": [source]}],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook), encoding="utf-8")


class NotebookExecutionTests(unittest.TestCase):
    def test_discovery_selects_only_notebooks_with_code(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            code = root / "code.ipynb"
            write_notebook(code, "code", "print('ok')")
            write_notebook(root / "markdown.ipynb", "markdown", "Notes")
            write_notebook(root / "empty.ipynb", "code", "  \n")

            with patch.object(execute_notebooks, "ROOT", root):
                self.assertEqual(execute_notebooks.selected_notebooks([]), [code])

    def test_explicit_notebook_must_be_inside_repository(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repository"
            root.mkdir()
            outside = Path(temp) / "outside.ipynb"
            write_notebook(outside, "code", "print('outside')")

            with patch.object(execute_notebooks, "ROOT", root):
                with self.assertRaisesRegex(ValueError, "outside the repository"):
                    execute_notebooks.selected_notebooks([outside])

    @patch.object(execute_notebooks, "NotebookClient")
    def test_execution_treats_notebook_warnings_as_errors(self, client):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "code.ipynb"
            notebook = nbformat.v4.new_notebook(
                cells=[nbformat.v4.new_code_cell("print('ok')")]
            )
            nbformat.write(notebook, path)

            execute_notebooks.execute_one(path, "python3", 30)

        executed = client.call_args.args[0]
        self.assertEqual(
            executed.cells[0].source,
            "import warnings\nwarnings.simplefilter('error')",
        )
        client.return_value.execute.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()

"""Execute teaching notebooks in isolated, one-notebook processes."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()
KERNEL_NAME = "radio-review"


def contains_code(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    return any(
        cell.get("cell_type") == "code" and "".join(cell.get("source", [])).strip()
        for cell in notebook["cells"]
    )


def execute_one(path, kernel_name, timeout):
    notebook = nbformat.read(path, as_version=4)
    notebook.cells.insert(
        0,
        nbformat.v4.new_code_cell(
            "import warnings\nwarnings.simplefilter('error')",
            metadata={"tags": ["remove-cell"]},
        ),
    )
    NotebookClient(
        notebook,
        timeout=timeout,
        kernel_name=kernel_name,
        resources={"metadata": {"path": str(path.parent)}},
        allow_errors=False,
        record_timing=False,
        shutdown_kernel="immediate",
    ).execute()


def write_kernel_spec(jupyter_root):
    kernel_dir = jupyter_root / "kernels" / KERNEL_NAME
    kernel_dir.mkdir(parents=True)
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Radio review",
        "language": "python",
    }
    (kernel_dir / "kernel.json").write_text(json.dumps(spec), encoding="utf-8")


def selected_notebooks(values):
    if values:
        paths = [(ROOT / value).resolve() for value in values]
        outside = [path for path in paths if ROOT not in path.parents]
        if outside:
            raise ValueError(f"notebook is outside the repository: {outside[0]}")
    else:
        paths = sorted(ROOT.rglob("*.ipynb"))
    return [path for path in paths if contains_code(path)]


def execute_isolated(paths, timeout):
    with tempfile.TemporaryDirectory(prefix="radio-notebooks-") as temp:
        temp_root = Path(temp)
        run_root = temp_root / "repository"
        shutil.copytree(
            ROOT,
            run_root,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )

        jupyter_root = temp_root / "jupyter"
        write_kernel_spec(jupyter_root)
        env = os.environ.copy()
        old_path = env.get("JUPYTER_PATH")
        env["JUPYTER_PATH"] = str(jupyter_root)
        if old_path:
            env["JUPYTER_PATH"] += os.pathsep + old_path

        total = len(paths)
        for index, source in enumerate(paths, start=1):
            relative = source.relative_to(ROOT)
            target = run_root / relative
            print(f"START [{index:02d}/{total:02d}] {relative}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--single",
                    str(target),
                    "--kernel",
                    KERNEL_NAME,
                    "--timeout",
                    str(timeout),
                ],
                check=True,
                env=env,
            )
            print(f"OK    [{index:02d}/{total:02d}] {relative}", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("notebooks", nargs="*", type=Path)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--kernel", default="python3", help=argparse.SUPPRESS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.single:
        execute_one(args.notebooks[0], args.kernel, args.timeout)
        return
    paths = selected_notebooks(args.notebooks)
    execute_isolated(paths, args.timeout)
    print(f"EXECUTED={len(paths)}")


if __name__ == "__main__":
    main()

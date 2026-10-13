"""Static validation for Python and Jupyter files in this repository."""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

WINDOWS_PATH = re.compile(r"[A-Za-z]:\\|C:/Users/", re.IGNORECASE)


def validate(repo: Path) -> list[str]:
    errors: list[str] = []
    python_files = sorted(repo.rglob("*.py"))
    notebook_files = sorted(repo.rglob("*.ipynb"))

    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            errors.append(f"{path}: {exc}")
        source = path.read_text(encoding="utf-8")
        if path.name != "validate_repo.py" and WINDOWS_PATH.search(source):
            errors.append(f"{path}: contains a non-portable Windows path")

    for path in notebook_files:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"{path}: invalid notebook JSON: {exc}")
            continue
        for index, cell in enumerate(notebook.get("cells", [])):
            source = "".join(cell.get("source", []))
            if cell.get("cell_type") == "code":
                try:
                    ast.parse(source, filename=f"{path}:cell {index}")
                except SyntaxError as exc:
                    errors.append(f"{path}:cell {index}: {exc}")
                if cell.get("outputs"):
                    errors.append(
                        f"{path}:cell {index}: stale notebook outputs are present"
                    )
                if WINDOWS_PATH.search(source):
                    errors.append(f"{path}:cell {index}: contains a non-portable path")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    errors = validate(args.repo_root.resolve())
    if errors:
        print("Validation failed:")
        print("\n".join(f"- {error}" for error in errors))
        return 1
    print("Static validation passed: Python syntax, notebook syntax, paths, and outputs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

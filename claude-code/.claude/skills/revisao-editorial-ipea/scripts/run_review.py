#!/usr/bin/env python3
"""Executa a CLI editorial a partir de uma skill instalada."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def find_project_root(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    if os.environ.get("IPEA_EDITORIAL_ROOT"):
        candidates.append(Path(os.environ["IPEA_EDITORIAL_ROOT"]))
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        pyproject = root / "pyproject.toml"
        if pyproject.exists() and "lang-ipea-editorial" in pyproject.read_text(encoding="utf-8"):
            return root
    raise SystemExit(
        "Projeto lang_IPEA_editorial não encontrado. "
        "Abra o repositório, informe --project-root ou defina IPEA_EDITORIAL_ROOT."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Executa a revisão editorial Ipea.")
    parser.add_argument("input", type=Path)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--question")
    parser.add_argument("--output-docx", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-diagnostics-json", type=Path)
    parser.add_argument("--output-normalized-json", type=Path)
    parser.add_argument("--keep-history", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = find_project_root(args.project_root)
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Arquivo de entrada não encontrado: {input_path}")
    if shutil.which("uv") is None:
        raise SystemExit("O comando 'uv' não está disponível no PATH.")

    command = ["uv", "run", "editorial-docx", str(input_path)]
    for flag, value in (
        ("--question", args.question),
        ("--output-docx", args.output_docx),
        ("--output-json", args.output_json),
        ("--output-diagnostics-json", args.output_diagnostics_json),
        ("--output-normalized-json", args.output_normalized_json),
    ):
        if value is not None:
            command.extend([flag, str(value)])
    if args.keep_history:
        command.append("--keep-history")

    print("Projeto:", root)
    print("Comando:", subprocess.list2cmdline(command))
    if args.dry_run:
        return 0
    if not args.skip_preflight:
        preflight = Path(__file__).with_name("editorial_lab.py")
        preflight_command = [
            "uv",
            "run",
            "python",
            str(preflight),
            "preflight",
            "--project-root",
            str(root),
        ]
        preflight_result = subprocess.run(preflight_command, cwd=root, check=False)
        if preflight_result.returncode != 0:
            print(
                "Preflight da LLM falhou. A revisão foi interrompida para evitar "
                "que uma execução heurística parcial pareça completa.",
                file=sys.stderr,
            )
            return preflight_result.returncode
    return subprocess.run(command, cwd=root, check=False).returncode


if __name__ == "__main__":
    sys.exit(main())

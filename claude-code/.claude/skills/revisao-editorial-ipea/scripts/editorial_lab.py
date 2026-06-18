#!/usr/bin/env python3
"""Encaminha para a implementação canônica compartilhada com a skill Codex."""

from __future__ import annotations

import os
from pathlib import Path
import runpy


def find_implementation() -> Path:
    candidates: list[Path] = []
    if os.environ.get("IPEA_EDITORIAL_ROOT"):
        candidates.append(Path(os.environ["IPEA_EDITORIAL_ROOT"]))
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    for root in candidates:
        candidate = (
            root.expanduser().resolve()
            / "codex"
            / "skills"
            / "revisao-editorial-ipea"
            / "scripts"
            / "editorial_lab.py"
        )
        if candidate.exists():
            return candidate
    raise SystemExit(
        "Implementação compartilhada não encontrada. Execute dentro do repositório "
        "IPEA_editorial ou defina IPEA_EDITORIAL_ROOT."
    )


if __name__ == "__main__":
    runpy.run_path(str(find_implementation()), run_name="__main__")


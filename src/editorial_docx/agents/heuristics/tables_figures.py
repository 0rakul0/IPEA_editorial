from __future__ import annotations

import re

from ...models import AgentComment
from ...review_patterns import _folded_text, _ref_block_type


def _looks_mostly_textual_table(start_idx: int, chunks: list[str], refs: list[str]) -> bool:
    """Approximates whether nearby table cells are predominantly textual."""
    sample: list[str] = []
    idx = start_idx + 1
    while idx < len(chunks) and idx < len(refs) and len(sample) < 8:
        if _ref_block_type(refs[idx]) != "table_cell":
            break
        cell = (chunks[idx] or "").strip()
        if cell:
            sample.append(cell)
        idx += 1
    if len(sample) < 3:
        return False
    textual = 0
    for cell in sample:
        letters = len(re.findall(r"[A-Za-zÀ-ÿ]", cell))
        digits = len(re.findall(r"\d", cell))
        if letters >= max(4, digits + 2):
            textual += 1
    return textual / len(sample) >= 0.7


def heuristic_table_figure_comments(batch_indexes: list[int], chunks: list[str], refs: list[str]) -> list[AgentComment]:
    """Handles heuristic table figure comments."""
    comments: list[AgentComment] = []
    source_prefixes = ("fonte:", "elaboração:", "elaboracao:")

    for idx in batch_indexes:
        if not (0 <= idx < len(chunks)) or idx >= len(refs):
            continue
        block_type = _ref_block_type(refs[idx])
        text = (chunks[idx] or "").strip()
        if block_type != "caption":
            continue
        norm = _folded_text(text)
        if re.match(r"^(tabela|figura|quadro|grafico)\s+\d+[:\s]", norm, flags=re.IGNORECASE):
            identifier = re.match(r"^((?:Tabela|Figura|Quadro|Gráfico)\s+\d+)[:\s]+(.+)$", text)
            fix = text
            if identifier:
                fix = f"Separar em duas linhas: `{identifier.group(1).upper()}` na primeira linha e `{identifier.group(2).strip()}` na linha abaixo."
            comments.append(
                AgentComment(
                    agent="tabelas_figuras",
                    category="Legenda",
                    message="Na legenda, o identificador deve ficar na primeira linha e o título descritivo na linha abaixo.",
                    paragraph_index=idx,
                    issue_excerpt=text,
                    suggested_fix=fix,
                    action_type="auto_fix_candidate",
                )
            )
        if re.match(r"^tabela\s+\d+[:\s]", norm, flags=re.IGNORECASE) and _looks_mostly_textual_table(idx, chunks, refs):
            comments.append(
                AgentComment(
                    agent="tabelas_figuras",
                    category="Quadro/Tabela",
                    message="Pelo padrão editorial, ilustrações com células predominantemente textuais tendem a ser tratadas como quadro, não tabela.",
                    paragraph_index=idx,
                    issue_excerpt=text,
                    suggested_fix="Verificar se este bloco deve ser apresentado como `QUADRO`, e não como `TABELA`.",
                    action_type="production_request",
                )
            )
        next_idx = idx + 1
        if next_idx >= len(chunks):
            continue
        neighbor = (chunks[next_idx] or "").strip().casefold()
        if not any(neighbor.startswith(prefix) for prefix in source_prefixes) and _ref_block_type(refs[next_idx]) != "caption":
            comments.append(
                AgentComment(
                    agent="tabelas_figuras",
                    category="Fonte",
                    message="O bloco está sem uma linha de fonte ou elaboração logo abaixo da legenda.",
                    paragraph_index=idx,
                    issue_excerpt=text,
                    suggested_fix="Adicionar uma linha própria com `Fonte:` ou `Elaboração:` abaixo do bloco.",
                    action_type="production_request",
                )
            )

    return comments


__all__ = ["heuristic_table_figure_comments"]

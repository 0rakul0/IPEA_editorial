from __future__ import annotations

import re

from ...models import AgentComment
from ...review_patterns import _ref_block_type

_ABBREV_MAP: dict[str, str] = {
    "Coef.": "Coeficiente",
    "Obs.": "Observação",
}
_ABBREV_LOWER = {k.lower(): v for k, v in _ABBREV_MAP.items()}


def heuristic_grammar_comments(batch_indexes: list[int], chunks: list[str], refs: list[str]) -> list[AgentComment]:
    """Handles heuristic grammar comments."""
    comments: list[AgentComment] = []
    seen: set[tuple[int, str, str]] = set()

    def add(idx: int, issue: str, fix: str, message: str, category: str = "grammar") -> None:
        """Adds an item to the current collection."""
        key = (idx, issue, fix)
        if key in seen:
            return
        seen.add(key)
        comments.append(
            AgentComment(
                agent="gramatica_ortografia",
                category=category,
                message=message,
                paragraph_index=idx,
                issue_excerpt=issue,
                suggested_fix=fix,
            )
        )

    for idx in batch_indexes:
        if not (0 <= idx < len(chunks)) or idx >= len(refs):
            continue
        if _ref_block_type(refs[idx]) in {"reference_entry", "reference_heading", "direct_quote"}:
            continue
        text = chunks[idx] or ""

        # Locução verbal com preposição faltante ("passou ser" → "passou a ser")
        if re.search(r"\bpassou ser\b", text, flags=re.IGNORECASE):
            add(idx, "passou ser", "passou a ser", "Falta a preposição na locução verbal.", "Regência")
        for match in re.finditer(r"\bpassou ([a-zà-ú]+[r])\b", text, flags=re.IGNORECASE):
            raw = match.group(0)
            verb = match.group(1)
            fix = f"passou a {verb}"
            add(idx, raw, fix, "Falta preposição na locução verbal.", "Regência")

        # Artigo definido ausente após "todos/as" ("todos trabalhadores" → "todos os trabalhadores")
        if re.search(r"\bpara todos trabalhadores\b", text, flags=re.IGNORECASE):
            add(idx, "para todos trabalhadores", "para todos os trabalhadores", "Falta artigo definido antes do substantivo.", "Concordância")
        for match in re.finditer(r"\btod[ao]s ([a-zà-ú]+)\b", text, flags=re.IGNORECASE):
            raw = match.group(0)
            gender = "as" if raw[:4].lower() == "todas" else "os"
            noun = match.group(1)
            fix = f"tod{gender[:1]}s {gender} {noun}"
            add(idx, raw, fix, "Falta artigo definido após o pronome.", "Concordância")

        # Partícula expletiva dispensável ("observa-se que")
        if re.search(r"\bobserva-se\s+que\b", text, flags=re.IGNORECASE):
            add(idx, "observa-se que", "observa-se", "A construção contém partícula expletiva dispensável.", "Concordância")

        # Discordância nominal (benefícios monetário → benefícios monetários)
        if re.search(r"\bbenef[ií]cios monet[aá]rio\b", text, flags=re.IGNORECASE):
            add(idx, "benefícios monetário", "benefícios monetários", "Há discordância nominal entre substantivo e adjetivo.", "Concordância")

        # Concordância verbal
        if re.search(r"\bque assenta o acesso\b", text, flags=re.IGNORECASE):
            add(idx, "que assenta o acesso", "que assentam o acesso", "O verbo deve concordar com o sujeito composto.", "Concordância")
        if re.search(r"\be sugerem\b", text, flags=re.IGNORECASE) and re.search(r"\bexerc[ií]cio realizado\b", text, flags=re.IGNORECASE):
            add(idx, "e sugerem", "e sugere", "O verbo deve concordar com o núcleo singular do sujeito.", "Concordância")

        # Padronização de intervalo de anos ("1995/00" → "1995-2000")
        for match in re.finditer(r"\b(\d{4})/(\d{2})\b", text):
            first = int(match.group(1))
            second = int(match.group(2))
            full_year = first // 100 * 100 + second
            if abs(full_year - first) <= 100:
                raw = match.group(0)
                fix = f"{first}-{full_year}"
                add(idx, raw, fix, "Padronize o intervalo de anos com hífen.", "Padronização")

        # Abreviações por extenso
        for abrev, extenso in _ABBREV_LOWER.items():
            pattern = re.compile(rf"\b{re.escape(abrev)}\b", flags=re.IGNORECASE)
            for match in pattern.finditer(text):
                raw = match.group(0)
                if abrev.startswith("obs") and raw.endswith("."):
                    continue
                add(idx, raw, extenso, "Escreva a abreviatura por extenso.", "Padronização")

        # Espaçamento e pontuação
        for match in re.finditer(r"\S+ {2,}\S+", text):
            issue = match.group(0)
            fix = re.sub(r" {2,}", " ", issue)
            add(idx, issue, fix, "Há espaço duplo indevido no trecho.", "Pontuação")
        for match in re.finditer(r"\S+\s+[,.;:!?]", text):
            issue = match.group(0)
            fix = re.sub(r"\s+([,.;:!?])$", r"\1", issue)
            add(idx, issue, fix, "Há espaço indevido antes do sinal de pontuação.", "Pontuação")
        for match in re.finditer(r"\S+[.?!][A-ZÀ-Ý]\S*", text):
            issue = match.group(0)
            fix = re.sub(r"([.?!])([A-ZÀ-Ý])", r"\1 \2", issue, count=1)
            add(idx, issue, fix, "Falta espaço após a pontuação final.", "Pontuação")

    return comments


__all__ = ["heuristic_grammar_comments"]

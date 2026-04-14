from __future__ import annotations

import re

from .document_loader import Section
from .models import AgentComment, DocumentUserComment, agent_short_label
from .prompts import AGENT_ORDER
from .review_consolidation import consolidate_semantic_comments
from .review_context import PreparedReviewDocument, prepare_review_document as _prepare_review_document
from .review_heuristics import _find_reference_citation_indexes
from .review_patterns import (
    _find_metadata_like_indexes,
    _heading_word_count,
    _indexes_by_ref_type,
    _is_implicit_heading_candidate,
    _is_intro_heading,
    _normalized_text,
    _ref_block_type,
    _ref_style_name,
    _style_name_looks_explicit,
)

_USER_REFERENCE_AGENT = "comentarios_usuario_referencias"


def _build_batches(
    chunks: list[str],
    refs: list[str],
    indexes: list[int],
    max_chars: int = 12000,
    max_chunks: int = 28,
) -> list[list[int]]:
    if not chunks or not indexes:
        return []

    batches: list[list[int]] = []
    current: list[int] = []
    current_chars = 0

    for idx in indexes:
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        ref = refs[idx] if idx < len(refs) else "sem referência"
        line = f"[{idx}] ({ref}) {chunk}"
        line_len = len(line) + 1

        if current and (len(current) >= max_chunks or current_chars + line_len > max_chars):
            batches.append(current)
            current = []
            current_chars = 0

        current.append(idx)
        current_chars += line_len

    if current:
        batches.append(current)

    return batches


def _expand_neighbors(indexes: list[int], total: int, radius: int = 1) -> list[int]:
    expanded: set[int] = set()
    for idx in indexes:
        for candidate in range(max(0, idx - radius), min(total, idx + radius + 1)):
            expanded.add(candidate)
    return sorted(expanded)


def _expand_section_ranges(sections: list[Section], keywords: tuple[str, ...]) -> list[int]:
    selected: list[int] = []
    for sec in sections:
        title = sec.title.lower()
        if any(k in title for k in keywords):
            selected.extend(range(sec.start_idx, sec.end_idx + 1))
    return sorted(dict.fromkeys(selected))


def _find_content_indexes(chunks: list[str], pattern: str) -> list[int]:
    rx = re.compile(pattern, re.IGNORECASE)
    out: list[int] = []
    for idx, chunk in enumerate(chunks):
        if rx.search(chunk):
            out.append(idx)
    return out


def _agent_scope_indexes(agent: str, chunks: list[str], refs: list[str], sections: list[Section]) -> list[int]:
    """Seleciona os índices mais relevantes do documento para cada agente."""
    total = len(chunks)
    if total == 0:
        return []
    if agent == _USER_REFERENCE_AGENT:
        return []

    all_indexes = list(range(total))
    head_20 = list(range(max(1, int(total * 0.20))))
    tail_30_start = max(0, int(total * 0.70))
    tail_30 = list(range(tail_30_start, total))

    if agent == "metadados":
        sec = _expand_section_ranges(sections, ("metadad", "ficha catalogr", "capa", "titulo", "autoria"))
        head_candidates = _find_metadata_like_indexes(chunks, refs, limit=18)
        picked = sorted(dict.fromkeys([*sec, *head_candidates]))
        return picked or head_candidates or list(range(min(12, total)))

    if agent == "sinopse_abstract":
        sec = _expand_section_ranges(sections, ("sinopse", "abstract", "resumo", "summary"))
        content = _find_content_indexes(chunks, r"\b(sinopse|abstract|resumo|summary|palavras-chave|keywords|jel)\b")
        typed = _indexes_by_ref_type(refs, {"abstract_heading", "abstract_body", "keywords_label", "keywords_content", "jel_code"})
        picked = _expand_neighbors(sorted(dict.fromkeys([*sec, *content, *typed])), total=total, radius=1)
        return picked or head_20

    if agent == "estrutura":
        typed = _indexes_by_ref_type(refs, {"heading", "reference_heading"})
        section_starts = sorted(dict.fromkeys(sec.start_idx for sec in sections))
        intro_start = next(
            (
                idx
                for idx, chunk in enumerate(chunks)
                if _is_intro_heading(chunk) and _is_implicit_heading_candidate(idx, chunks, refs)
            ),
            None,
        )
        if intro_start is None:
            intro_start = next(
                (idx for idx in sorted(dict.fromkeys([*typed, *section_starts])) if 0 <= idx < len(chunks) and _is_intro_heading(chunks[idx])),
                None,
            )

        implicit = [
            idx
            for idx in range(intro_start if intro_start is not None else 0, total)
            if _is_implicit_heading_candidate(idx, chunks, refs)
        ]
        heading_candidates = sorted(dict.fromkeys([*typed, *section_starts, *implicit]))
        if not heading_candidates:
            return typed or head_20

        scoped = [idx for idx in heading_candidates if intro_start is None or idx >= intro_start]
        explicit_scoped = [idx for idx in scoped if idx in set(typed) or idx in set(section_starts)]
        implicit_short_scoped = [
            idx for idx in scoped if idx not in set(explicit_scoped) and 0 <= idx < len(chunks) and _heading_word_count(chunks[idx]) <= 4
        ]
        picked = sorted(dict.fromkeys([*explicit_scoped, *implicit_short_scoped]))
        return picked or scoped or heading_candidates

    if agent == "tabelas_figuras":
        sec = _expand_section_ranges(sections, ("tabela", "figura", "quadro", "grafico", "gráfico", "anexo"))
        content = _find_content_indexes(chunks, r"\b(tabela|figura|quadro|gr[aá]fico|imagem)\b")
        typed = _indexes_by_ref_type(refs, {"caption", "table_cell"})
        picked = _expand_neighbors(sorted(dict.fromkeys([*sec, *content, *typed])), total=total, radius=2)
        return picked or typed or all_indexes

    if agent == "referencias":
        sec = _expand_section_ranges(sections, ("refer", "bibliograf", "references", "bibliography"))
        reference_heading_idx = next((idx for idx, ref in enumerate(refs) if _ref_block_type(ref) == "reference_heading"), total)
        citation_like = _find_reference_citation_indexes(chunks, refs, body_limit=reference_heading_idx)
        if sec:
            return sorted(dict.fromkeys([*citation_like, *sec]))
        content = _find_content_indexes(chunks, r"\b(doi|http://|https://|et al\.|v\.\s*\d+|n\.\s*\d+)\b")
        typed = _indexes_by_ref_type(refs, {"reference_entry", "reference_heading"})
        picked = sorted(dict.fromkeys([*citation_like, *content, *typed]))
        if not picked:
            return tail_30
        return picked

    if agent == "tipografia":
        typed = _indexes_by_ref_type(refs, {"heading", "caption", "reference_entry", "reference_heading"})
        styled = [
            idx
            for idx, ref in enumerate(refs)
            if _ref_block_type(ref) == "paragraph" and _style_name_looks_explicit(_ref_style_name(ref)) and idx < 24
        ]
        picked = sorted(dict.fromkeys([*typed, *styled]))
        return picked or typed or head_20

    if agent == "gramatica_ortografia":
        return all_indexes

    return all_indexes


def prepare_review_batches(
    paragraphs: list[str],
    refs: list[str],
    sections: list[Section],
    selected_agents: list[str] | None = None,
    user_comments: list[DocumentUserComment] | None = None,
) -> PreparedReviewDocument:
    """Gera o documento preparado com os lotes que cada agente deve revisar."""
    agent_order = [agent for agent in (selected_agents or AGENT_ORDER) if agent in AGENT_ORDER]
    return _prepare_review_document(
        chunks=paragraphs,
        refs=refs,
        sections=sections,
        user_comments=user_comments or [],
        agent_order=agent_order,
        agent_scope_builder=_agent_scope_indexes,
    )


def _comment_priority(comment: AgentComment, refs: list[str]) -> tuple[int, int, int]:
    block_type = ""
    if isinstance(comment.paragraph_index, int) and 0 <= comment.paragraph_index < len(refs):
        block_type = _ref_block_type(refs[comment.paragraph_index])
    specificity = 0
    if comment.category == "citation_match":
        specificity += 3
    if block_type == "reference_entry":
        specificity += 2
    if block_type == "reference_heading":
        specificity -= 2
    density = len((comment.issue_excerpt or "").strip()) + len((comment.suggested_fix or "").strip())
    return specificity, density, len((comment.message or "").strip())


def _comment_sort_key(comment: AgentComment) -> tuple[int, str, str]:
    paragraph_index = comment.paragraph_index if isinstance(comment.paragraph_index, int) else 10**9
    return paragraph_index, agent_short_label(comment.agent), _normalized_text(comment.message)


def _consolidate_final_comments(comments: list[AgentComment], refs: list[str]) -> list[AgentComment]:
    """Deduplica e prioriza os comentários antes da saída final do review."""
    if not comments:
        return []

    best_by_key: dict[tuple[str, str, int | None, str, str], AgentComment] = {}
    for comment in comments:
        key = (
            comment.agent,
            comment.category,
            comment.paragraph_index if isinstance(comment.paragraph_index, int) else None,
            _normalized_text(comment.issue_excerpt),
            _normalized_text(comment.suggested_fix),
        )
        existing = best_by_key.get(key)
        if existing is None or _comment_priority(comment, refs) > _comment_priority(existing, refs):
            best_by_key[key] = comment

    deduped = list(best_by_key.values())
    deduped = consolidate_semantic_comments(deduped)
    has_reference_body_matches = any(item.agent == "referencias" and item.category == "citation_match" for item in deduped)
    filtered: list[AgentComment] = []
    suppressed_heading_messages = {_normalized_text("Há citações no corpo do texto sem correspondência clara na lista de referências.")}

    for comment in deduped:
        if (
            has_reference_body_matches
            and comment.agent == "referencias"
            and isinstance(comment.paragraph_index, int)
            and 0 <= comment.paragraph_index < len(refs)
            and _ref_block_type(refs[comment.paragraph_index]) == "reference_heading"
            and _normalized_text(comment.message) in suppressed_heading_messages
        ):
            continue
        filtered.append(comment)

    filtered.sort(key=_comment_sort_key)
    return filtered


__all__ = [
    "_agent_scope_indexes",
    "_build_batches",
    "_consolidate_final_comments",
    "prepare_review_batches",
]

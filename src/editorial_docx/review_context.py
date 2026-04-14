from __future__ import annotations

from dataclasses import dataclass, field

from .context_selector import build_excerpt
from .document_loader import Section
from .models import DocumentUserComment
from .token_utils import TokenChunkConfig, chunk_index_windows


@dataclass(slots=True)
class ReviewBatch:
    indexes: list[int]
    focus_excerpt: str
    window_excerpt: str
    headings: list[str] = field(default_factory=list)
    start_idx: int = 0
    end_idx: int = 0


@dataclass(slots=True)
class PreparedReviewDocument:
    chunks: list[str]
    refs: list[str]
    sections: list[Section]
    toc: list[str]
    user_comments: list[DocumentUserComment] = field(default_factory=list)
    agent_batches: dict[str, list[ReviewBatch]] = field(default_factory=dict)


def _build_batches(
    chunks: list[str],
    refs: list[str],
    indexes: list[int],
    max_chars: int = 12000,
    max_chunks: int = 28,
) -> list[list[int]]:
    if not chunks or not indexes:
        return []
    items: list[tuple[int, str]] = []
    for idx in indexes:
        if idx < 0 or idx >= len(chunks):
            continue
        ref = refs[idx] if idx < len(refs) else "sem referência"
        items.append((idx, f"[{idx}] ({ref}) {chunks[idx]}"))
    return chunk_index_windows(
        items,
        config=TokenChunkConfig(max_tokens=max(800, max_chars // 4), overlap_tokens=240, max_items=max_chunks),
    )


def _window_indexes(indexes: list[int], total: int, radius: int = 2) -> list[int]:
    if not indexes or total <= 0:
        return []
    start = max(0, min(indexes) - radius)
    end = min(total - 1, max(indexes) + radius)
    return list(range(start, end + 1))


def _headings_for_batch(sections: list[Section], indexes: list[int]) -> list[str]:
    if not sections or not indexes:
        return []

    start = min(indexes)
    end = max(indexes)
    headings = [section.title for section in sections if not (section.end_idx < start or section.start_idx > end)]
    if headings:
        return headings[:4]

    nearest = [section.title for section in sections if section.start_idx <= start]
    if nearest:
        return nearest[-2:]
    return []


def prepare_review_document(
    chunks: list[str],
    refs: list[str],
    sections: list[Section],
    agent_order: list[str],
    agent_scope_builder,
    user_comments: list[DocumentUserComment] | None = None,
    max_batch_chars: int = 12000,
    max_batch_chunks: int = 28,
    window_radius: int = 2,
) -> PreparedReviewDocument:
    """Prepara lotes, janelas de contexto e TOC para todos os agentes."""
    toc = [f"{section.title} [{section.start_idx}-{section.end_idx}]" for section in sections]
    prepared = PreparedReviewDocument(
        chunks=chunks,
        refs=refs,
        sections=sections,
        toc=toc,
        user_comments=list(user_comments or []),
    )

    for agent in agent_order:
        scoped_indexes = agent_scope_builder(agent, chunks, refs, sections)
        raw_batches = _build_batches(
            chunks=chunks,
            refs=refs,
            indexes=scoped_indexes,
            max_chars=max_batch_chars,
            max_chunks=max_batch_chunks,
        )
        prepared.agent_batches[agent] = [
            ReviewBatch(
                indexes=batch_indexes,
                focus_excerpt=build_excerpt(indexes=batch_indexes, chunks=chunks, refs=refs, max_chars=1_000_000),
                window_excerpt=build_excerpt(
                    indexes=_window_indexes(batch_indexes, total=len(chunks), radius=window_radius),
                    chunks=chunks,
                    refs=refs,
                    max_chars=1_000_000,
                ),
                headings=_headings_for_batch(sections, batch_indexes),
                start_idx=batch_indexes[0],
                end_idx=batch_indexes[-1],
            )
            for batch_indexes in raw_batches
            if batch_indexes
        ]

    return prepared

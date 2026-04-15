from __future__ import annotations

from dataclasses import dataclass

from .abnt_citation_parser import CitationCandidate
from .abnt_reference_parser import ParsedReferenceEntry


@dataclass(frozen=True)
class ReferenceMatchResult:
    missing_citations: tuple[CitationCandidate, ...]
    uncited_references: tuple[ParsedReferenceEntry, ...]


def compare_citations_to_references(
    citations: list[CitationCandidate],
    references: list[ParsedReferenceEntry],
) -> ReferenceMatchResult:
    reference_keys = {entry.key for entry in references}
    citation_keys = {candidate.key for candidate in citations}
    missing_citations = tuple(candidate for candidate in citations if candidate.key not in reference_keys)
    uncited_references = tuple(entry for entry in references if entry.key not in citation_keys)
    return ReferenceMatchResult(
        missing_citations=missing_citations,
        uncited_references=uncited_references,
    )


__all__ = ["ReferenceMatchResult", "compare_citations_to_references"]

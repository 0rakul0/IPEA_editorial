from __future__ import annotations

from dataclasses import dataclass
import re

from .abnt_document_types import (
    ABNT_TYPE_ARTICLE,
    ABNT_TYPE_BOOK,
    ABNT_TYPE_CHAPTER,
    ABNT_TYPE_GENERIC,
    ABNT_TYPE_INSTITUTIONAL_REPORT,
    ABNT_TYPE_LEGAL,
    ABNT_TYPE_ONLINE,
    ABNT_TYPE_THESIS,
)
from .abnt_normalizer import canonical_author_key, publication_year_from_reference


@dataclass(frozen=True)
class ParsedReferenceEntry:
    raw_text: str
    author_raw: str
    author_key: str
    publication_year: str
    label: str
    year_candidates: tuple[str, ...]
    document_type: str
    title: str
    container_title: str
    place: str
    publisher: str
    institution: str
    has_url: bool
    has_access_date: bool
    has_doi: bool
    has_in: bool
    has_volume: bool
    has_number: bool
    has_pages: bool

    @property
    def key(self) -> tuple[str, str]:
        return self.author_key, self.publication_year


def _infer_reference_document_type(text: str) -> str:
    source = text.casefold()
    if any(marker in source for marker in ("lei ", "decreto ", "portaria ", "resolução ", "resolucao ")):
        return ABNT_TYPE_LEGAL
    if any(marker in source for marker in ("tese", "dissertação", "dissertacao")):
        return ABNT_TYPE_THESIS
    if " in: " in source:
        return ABNT_TYPE_CHAPTER
    if "disponível em:" in source or "disponivel em:" in source:
        return ABNT_TYPE_ONLINE
    if "doi" in source or re.search(r"\bv\.\s*\d+", source) or re.search(r"\bn\.\s*\d+", source):
        return ABNT_TYPE_ARTICLE
    if any(marker in source for marker in ("texto para discussão", "texto para discussao", "relatório", "relatorio", "ipea")):
        return ABNT_TYPE_INSTITUTIONAL_REPORT
    if re.search(r":[^:]+,\s*(?:19|20)\d{2}", text):
        return ABNT_TYPE_BOOK
    return ABNT_TYPE_GENERIC


def _extract_reference_title(source: str, author_raw: str) -> str:
    tail = source[len(author_raw) :].lstrip(" ,.;")
    if not tail:
        return ""
    segments = [segment.strip(" .") for segment in re.split(r"\.\s+", tail) if segment.strip()]
    return segments[0] if segments else ""


def _extract_container_title(source: str, title: str) -> str:
    if " In: " in source or " in: " in source:
        match = re.search(r"\bIn:\s*([^.,]+)", source, flags=re.IGNORECASE)
        return match.group(1).strip() if match else ""
    after_title = source.split(title, 1)[-1].lstrip(" .") if title and title in source else source
    match = re.match(r"([^.,]+)", after_title)
    return match.group(1).strip() if match else ""


def _extract_place_and_publisher(source: str) -> tuple[str, str]:
    match = re.search(r"([A-ZÀ-Ý][A-Za-zÀ-ÿ\s\-]+):\s*([^,.;]+)", source)
    if not match:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()


def _extract_institution(source: str, author_raw: str, publisher: str) -> str:
    if author_raw.isupper():
        return author_raw.strip()
    match = re.search(
        r"\b(universidade|instituto|instituição|instituicao|ministério|ministerio|secretaria|ipea)\b",
        source,
        flags=re.IGNORECASE,
    )
    if match:
        return match.group(0).strip()
    return publisher


def parse_reference_entry(text: str, *, blocked_author_tokens: set[str] | None = None) -> ParsedReferenceEntry | None:
    source = (text or "").strip()
    if not source:
        return None

    publication_year = publication_year_from_reference(source)
    if not publication_year:
        return None

    if "," in source:
        author_raw = source.split(",", 1)[0].strip()
    elif "." in source:
        author_raw = source.split(".", 1)[0].strip()
    else:
        author_raw = source

    author_key = canonical_author_key(author_raw, extra_blocked_tokens=blocked_author_tokens)
    if author_key is None:
        return None

    document_type = _infer_reference_document_type(source)
    title = _extract_reference_title(source, author_raw)
    container_title = _extract_container_title(source, title)
    place, publisher = _extract_place_and_publisher(source)
    has_url = bool(re.search(r"https?://\S+", source, flags=re.IGNORECASE))
    has_access_date = bool(re.search(r"\bAcesso em\s*:", source, flags=re.IGNORECASE))
    has_doi = bool(re.search(r"\bdoi\b", source, flags=re.IGNORECASE))
    has_in = bool(re.search(r"\bIn:\s*", source, flags=re.IGNORECASE))
    has_volume = bool(re.search(r"\bv\.\s*\d+", source, flags=re.IGNORECASE))
    has_number = bool(re.search(r"\bn\.\s*\d+", source, flags=re.IGNORECASE))
    has_pages = bool(re.search(r"\bp{1,2}\.\s*\d+", source, flags=re.IGNORECASE))
    institution = _extract_institution(source, author_raw, publisher)
    year_candidates = tuple(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", source, flags=re.IGNORECASE))

    return ParsedReferenceEntry(
        raw_text=source,
        author_raw=author_raw,
        author_key=author_key,
        publication_year=publication_year,
        label=f"{author_raw} ({publication_year})",
        year_candidates=year_candidates,
        document_type=document_type,
        title=title,
        container_title=container_title,
        place=place,
        publisher=publisher,
        institution=institution,
        has_url=has_url,
        has_access_date=has_access_date,
        has_doi=has_doi,
        has_in=has_in,
        has_volume=has_volume,
        has_number=has_number,
        has_pages=has_pages,
    )


__all__ = ["ParsedReferenceEntry", "parse_reference_entry"]

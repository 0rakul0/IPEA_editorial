from __future__ import annotations

import re

from .abnt_citation_parser import extract_citation_candidates
from .abnt_matcher import compare_citations_to_references
from .abnt_normalizer import (
    canonical_author_key as _abnt_canonical_author_key,
    canonical_reference_key as _abnt_canonical_reference_key,
    citation_label as _abnt_citation_label,
    is_plausible_reference_author as _abnt_is_plausible_reference_author,
    publication_year_from_reference as _abnt_publication_year_from_reference,
)
from .abnt_reference_parser import parse_reference_entry
from .abnt_validator import validate_reference_entry
from .models import AgentComment
from .review_patterns import (
    _ascii_fold,
    _folded_text,
    _heading_word_count,
    _indexes_by_ref_type,
    _is_illustration_caption,
    _is_non_body_reference_context,
    _normalized_text,
    _ref_align,
    _ref_block_type,
    _ref_has_flag,
    _ref_has_numbering,
    _ref_style_name,
)

_NON_AUTHOR_REFERENCE_TOKENS = {
    "periodo",
    "ano",
    "anos",
    "mes",
    "meses",
    "pagina",
    "paginas",
    "secao",
    "secoes",
    "capitulo",
    "capitulos",
    "figura",
    "grafico",
    "quadro",
    "tabela",
    "nota",
    "anexo",
    "apendice",
    "parte",
    "texto",
    "documento",
    "disponivel",
    "versao",
    "edicao",
    "serie",
    "volume",
    "numero",
    "lei",
    "decreto",
    "constituicao",
    "formulario",
    "relacao",
    "tercos",
    "salarial",
    "estabelecimento",
    "sexo",
    "pis",
    "fgts",
}

_AUTHOR_PARTICLES = {"de", "da", "do", "das", "dos", "del", "della", "di"}
_LEADING_CITATION_CONTEXT_TOKENS = {"segundo", "conforme", "cf", "veja", "ver"}


def _heuristic_grammar_comments(batch_indexes: list[int], chunks: list[str], refs: list[str]) -> list[AgentComment]:
    comments: list[AgentComment] = []
    seen: set[tuple[int, str, str]] = set()

    def add(idx: int, issue: str, fix: str, message: str, category: str = "grammar") -> None:
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
        if re.search(r"\bpassou ser\b", text, flags=re.IGNORECASE):
            add(idx, "passou ser", "passou a ser", "Falta a preposição na locução verbal.", "Regência")
        if re.search(r"\bpara todos trabalhadores\b", text, flags=re.IGNORECASE):
            add(idx, "para todos trabalhadores", "para todos os trabalhadores", "Falta artigo definido antes do substantivo.", "Concordância")
        if re.search(r"\bobserva-se\s+que\b", text, flags=re.IGNORECASE):
            add(idx, "observa-se que", "observa-se", "A construção contém partícula expletiva dispensável.", "Concordância")
        if re.search(r"\bbenef[ií]cios monet[aá]rio\b", text, flags=re.IGNORECASE):
            add(idx, "benefícios monetário", "benefícios monetários", "Há discordância nominal entre substantivo e adjetivo.", "Concordância")
        if re.search(r"\bque assenta o acesso\b", text, flags=re.IGNORECASE):
            add(idx, "que assenta o acesso", "que assentam o acesso", "O verbo deve concordar com o sujeito composto.", "Concordância")
        if re.search(r"\be sugerem\b", text, flags=re.IGNORECASE) and re.search(r"\bexerc[ií]cio realizado\b", text, flags=re.IGNORECASE):
            add(idx, "e sugerem", "e sugere", "O verbo deve concordar com o núcleo singular do sujeito.", "Concordância")

    return comments


def _heuristic_synopsis_comments(batch_indexes: list[int], chunks: list[str], refs: list[str]) -> list[AgentComment]:
    comments: list[AgentComment] = []
    for idx in batch_indexes:
        if not (0 <= idx < len(chunks)) or idx >= len(refs):
            continue
        block_type = _ref_block_type(refs[idx])
        text = chunks[idx] or ""
        if block_type == "abstract_body" and _ref_align(refs[idx]) and _ref_align(refs[idx]) != "justify":
            comments.append(
                AgentComment(
                    agent="sinopse_abstract",
                    category="alignment",
                    message="O abstract deve estar justificado, mas este parágrafo está com outro alinhamento.",
                    paragraph_index=idx,
                    issue_excerpt=text,
                    suggested_fix="Justificar o parágrafo do abstract.",
                )
            )
        if block_type == "keywords_content":
            entries = [item.strip() for item in re.split(r"[;,]", text) if item.strip()]
            folded = [_folded_text(item) for item in entries]
            if len(set(folded)) != len(folded):
                comments.append(
                    AgentComment(
                        agent="sinopse_abstract",
                        category="keywords",
                        message="Há repetição de palavras-chave na lista.",
                        paragraph_index=idx,
                        issue_excerpt=text,
                        suggested_fix="Remover as repetições e manter apenas entradas únicas.",
                    )
                )
        if block_type == "abstract_body" and len(re.findall(r"[A-Za-zÀ-ÿ0-9]+", text)) > 250:
            comments.append(
                AgentComment(
                    agent="sinopse_abstract",
                    category="Extensão",
                    message="O abstract ultrapassa o limite de 250 palavras.",
                    paragraph_index=idx,
                    issue_excerpt=text,
                    suggested_fix="Reduzir o abstract para até 250 palavras.",
                )
            )
    return comments


def _heuristic_reference_comments(batch_indexes: list[int], chunks: list[str], refs: list[str]) -> list[AgentComment]:
    comments: list[AgentComment] = []
    seen: set[tuple[int, str, str]] = set()

    def add(idx: int, issue: str, fix: str, message: str, category: str = "reference_format") -> None:
        key = (idx, issue, fix)
        if key in seen:
            return
        seen.add(key)
        comments.append(
            AgentComment(
                agent="referencias",
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
        block_type = _ref_block_type(refs[idx])
        text = chunks[idx] or ""
        if block_type == "paragraph":
            for match in re.finditer(r"\b([A-ZÀ-Ý][A-Za-zÀ-ÿ'’`\-]+)\((\d{4}[a-z]?)\)", text):
                add(idx, match.group(0), f"{match.group(1)} ({match.group(2)})", "Falta um espaço antes do ano em citação autor-data.", "citation_format")
        if block_type == "reference_entry" and re.search(r"\bIn:\S", text, flags=re.IGNORECASE):
            add(idx, "In:", "In: ", "Inserir espaço após 'In:' na referência.", "reference_format")
        if block_type == "reference_entry" and re.search(r"\bDispon[ií]vel em:\s*\S+", text, flags=re.IGNORECASE) and "Acesso em:" not in text:
            add(
                idx,
                "Disponível em:",
                "Inserir `Acesso em:` com a data de consulta após a URL.",
                "A referência online informa a URL, mas não traz `Acesso em:` ao final.",
                "reference_format",
            )
        if block_type == "reference_entry" and re.search(r"([A-ZÀ-Ý][^.]*)\b(\d{4})\.(\D+[A-ZÀ-Ý])", text):
            match = re.search(r"(\d{4}\.[A-ZÀ-Ý])", text)
            if match:
                add(idx, match.group(1), match.group(1).replace(".", ". "), "Há duas referências coladas sem espaço entre elas.", "reference_format")
        if block_type == "reference_entry" and re.search(r"\bp\.(\d+[-–]\d+)", text, flags=re.IGNORECASE):
            match = re.search(r"(p\.\d+[-–]\d+)", text, flags=re.IGNORECASE)
            if match:
                add(idx, match.group(1), match.group(1).replace("p.", "p. "), "Falta espaço após a abreviatura de página.", "reference_format")
        if block_type == "reference_entry" and re.search(r"[A-Za-z0-9)\]]\s*$", text):
            add(idx, text, text.rstrip() + ".", "A referência termina sem ponto final.", "reference_format")
        duplicated_place = re.search(r"([A-ZÀ-Ý][A-Za-zÀ-ÿ\s]+):\s*\1,\s*\d{4}", text)
        if block_type == "reference_entry" and duplicated_place:
            add(idx, duplicated_place.group(0), duplicated_place.group(0), "Há duplicação de local e editora no trecho final da referência.", "reference_format")
        year_matches = re.findall(r"\b(?:19|20)\d{2}\b", text)
        if block_type == "reference_entry" and len(year_matches) >= 2 and year_matches[0] != year_matches[-1]:
            trailing = re.search(rf"\b{re.escape(year_matches[-1])}\b(?!.*\b(?:19|20)\d{{2}}\b)", text)
            if trailing:
                add(idx, trailing.group(0), year_matches[0], "O ano final da referência diverge do ano informado na abertura do registro.", "reference_format")
    return comments


def _looks_like_reference_author(author_raw: str) -> bool:
    author = (author_raw or "").strip()
    if not author:
        return False
    first_alpha = next((char for char in author if char.isalpha()), "")
    if not first_alpha or not first_alpha.isupper():
        return False
    return _canonical_author_key(author) is not None


def _canonical_author_key(author_raw: str) -> str | None:
    author = _ascii_fold((author_raw or "").strip()).casefold()
    if not author:
        return None
    author = re.sub(r"\bet\s+al\.?\b", "", author, flags=re.IGNORECASE)
    primary = re.split(r"\s+(?:e|and|&)\s+", author, maxsplit=1)[0].strip()
    tokens = re.findall(r"[a-z0-9]+", primary)
    if not tokens:
        return None

    idx = 0
    while idx < len(tokens) and tokens[idx] in _LEADING_CITATION_CONTEXT_TOKENS:
        idx += 1
    while idx < len(tokens) and tokens[idx] in _AUTHOR_PARTICLES:
        idx += 1
    if idx >= len(tokens):
        return None

    token = tokens[idx]
    if token in _NON_AUTHOR_REFERENCE_TOKENS:
        return None
    return token


def _reference_citation_key(author_raw: str, year_raw: str) -> tuple[str, str] | None:
    year = (year_raw or "").strip().casefold()
    if not year:
        return None
    author = _canonical_author_key(author_raw)
    if author is None:
        return None
    return author, year


def _reference_citation_label(author_raw: str, year_raw: str) -> str:
    author = re.sub(r"^\s*(?:Segundo|Conforme|Cf\.?|Veja|Ver)\s+", "", (author_raw or "").strip(), flags=re.IGNORECASE)
    author = re.split(r"\s+(?:et\s+al\.?|e|and|&)\b", author, maxsplit=1)[0].strip()
    if not author:
        author = (author_raw or "").strip()
    return f"{author} ({(year_raw or '').strip()})".strip()


def _reference_entry_publication_year(text: str) -> str | None:
    source = (text or "").strip()
    if not source:
        return None

    bibliographic_body = re.split(r"\b(?:Dispon[ií]vel em|Acesso em)\s*:", source, maxsplit=1, flags=re.IGNORECASE)[0]
    first_entry = re.split(r"\.\s+(?=[A-ZÀ-Ý][A-ZÀ-Ý'’`\\-]+,\s)", bibliographic_body, maxsplit=1)[0]
    year_matches = re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", first_entry, flags=re.IGNORECASE)
    if not year_matches:
        return None
    return year_matches[-1].casefold()


def _reference_body_citation_mentions(
    chunks: list[str], refs: list[str], body_limit: int
) -> list[tuple[int, str, tuple[str, str], str]]:
    mentions: list[tuple[int, str, tuple[str, str], str]] = []
    seen: set[tuple[int, str, tuple[str, str], str]] = set()

    def add_mention(idx: int, excerpt: str, author_raw: str, year_raw: str) -> None:
        key = _reference_citation_key(author_raw, year_raw)
        if key is None:
            return
        display = _reference_citation_label(author_raw, year_raw)
        clean_excerpt = re.sub(r"^\s*(?:Segundo|Conforme|Cf\.?|Veja|Ver)\s+", "", (excerpt or "").strip(), flags=re.IGNORECASE)
        mention = (idx, clean_excerpt, key, display)
        if mention in seen:
            return
        seen.add(mention)
        mentions.append(mention)

    for idx, (chunk, ref) in enumerate(zip(chunks[:body_limit], refs[:body_limit])):
        if _is_non_body_reference_context(ref, chunk, index=idx, chunks=chunks, refs=refs):
            continue
        text = chunk or ""

        for match in re.finditer(r"\b([A-ZÀ-Ý][A-Za-zÀ-ÿ'’`\-]+(?:\s+et\s+al\.?)?)\s*\((\d{4}[a-z]?)\)", text):
            author_raw = match.group(1)
            if _looks_like_reference_author(author_raw):
                add_mention(idx, match.group(0), author_raw, match.group(2))

        for parenthetical_match in re.finditer(r"\(([^)]*\d{4}[a-z]?[^)]*)\)", text):
            for segment in re.split(r";", parenthetical_match.group(1)):
                piece = segment.strip()
                match = re.search(r"([A-ZÀ-Ý][A-Za-zÀ-ÿ'’`\-]+(?:\s+et\s+al\.?)?)\s*,\s*(\d{4}[a-z]?)", piece)
                if match and _looks_like_reference_author(match.group(1)):
                    add_mention(idx, piece, match.group(1), match.group(2))

    return mentions


def _reference_body_citation_keys(chunks: list[str], refs: list[str], body_limit: int) -> set[tuple[str, str]]:
    return {key for _, _, key, _ in _reference_body_citation_mentions(chunks, refs, body_limit)}


def _reference_entry_key(text: str) -> tuple[str, str] | None:
    source = (text or "").strip()
    if not source:
        return None
    year_matches = re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", source, flags=re.IGNORECASE)
    if not year_matches:
        return None
    year = year_matches[0].casefold()

    if "," in source:
        author_part = source.split(",", 1)[0]
    elif "." in source:
        author_part = source.split(".", 1)[0]
    else:
        author_part = source

    author_part = _ascii_fold(author_part).casefold().strip()
    tokens = re.findall(r"[a-z0-9]+", author_part)
    if not tokens:
        return None
    return tokens[0], year


def _reference_entry_label(text: str) -> str:
    source = (text or "").strip()
    if not source:
        return ""
    key = _reference_entry_key(source)
    if key is None:
        return source[:80]
    author_raw = source.split(",", 1)[0].strip() if "," in source else source.split(".", 1)[0].strip()
    return f"{author_raw} ({key[1]})"


def _reference_body_citation_mentions(
    chunks: list[str], refs: list[str], body_limit: int
) -> list[tuple[int, str, tuple[str, str], str]]:
    mentions: list[tuple[int, str, tuple[str, str], str]] = []
    seen: set[tuple[int, str, tuple[str, str], str]] = set()

    def add_mention(idx: int, excerpt: str, author_raw: str, year_raw: str) -> None:
        key = _reference_citation_key(author_raw, year_raw)
        if key is None:
            return
        display = _reference_citation_label(author_raw, year_raw)
        clean_excerpt = re.sub(r"^\s*(?:Segundo|Conforme|Cf\.?|Veja|Ver)\s+", "", (excerpt or "").strip(), flags=re.IGNORECASE)
        mention = (idx, clean_excerpt, key, display)
        if mention in seen:
            return
        seen.add(mention)
        mentions.append(mention)

    narrative_pattern = re.compile(
        r"\b([A-ZÀ-Ý][A-Za-zÀ-ÿ'’`\-]+(?:\s+(?:de|da|do|das|dos)\s+[A-ZÀ-Ý][A-Za-zÀ-ÿ'’`\-]+|\s+[A-ZÀ-Ý][A-Za-zÀ-ÿ'’`\-]+|\s+et\s+al\.?)*)\s*\((\d{4}[a-z]?)\)"
    )
    parenthetical_pattern = re.compile(r"\(([^)]*\d{4}[a-z]?[^)]*)\)")
    segment_pattern = re.compile(r"([A-ZÀ-Ý][^,;()]*)\s*,\s*(\d{4}[a-z]?)")

    for idx, (chunk, ref) in enumerate(zip(chunks[:body_limit], refs[:body_limit])):
        if _is_non_body_reference_context(ref, chunk, index=idx, chunks=chunks, refs=refs):
            continue
        text = chunk or ""

        for match in narrative_pattern.finditer(text):
            author_raw = match.group(1)
            if _looks_like_reference_author(author_raw):
                add_mention(idx, match.group(0), author_raw, match.group(2))

        for parenthetical_match in parenthetical_pattern.finditer(text):
            for segment in re.split(r";", parenthetical_match.group(1)):
                piece = segment.strip()
                match = segment_pattern.search(piece)
                if match and _looks_like_reference_author(match.group(1)):
                    add_mention(idx, piece, match.group(1), match.group(2))

    return mentions


def _reference_entry_key(text: str) -> tuple[str, str] | None:
    source = (text or "").strip()
    if not source:
        return None

    year = _reference_entry_publication_year(source)
    if not year:
        return None

    if "," in source:
        author_part = source.split(",", 1)[0]
    elif "." in source:
        author_part = source.split(".", 1)[0]
    else:
        author_part = source

    author = _canonical_author_key(author_part)
    if author is None:
        return None
    return author, year


def _reference_entry_label(text: str) -> str:
    source = (text or "").strip()
    if not source:
        return ""
    key = _reference_entry_key(source)
    if key is None:
        return source[:80]
    author_raw = source.split(",", 1)[0].strip() if "," in source else source.split(".", 1)[0].strip()
    return f"{author_raw} ({key[1]})"


def _summarize_reference_labels(labels: list[str], max_items: int = 6) -> str:
    cleaned = [label.strip() for label in labels if label.strip()]
    if not cleaned:
        return ""
    if len(cleaned) <= max_items:
        return "; ".join(cleaned)
    shown = "; ".join(cleaned[:max_items])
    return f"{shown}; e mais {len(cleaned) - max_items}"


def _reference_body_format_comments(
    chunks: list[str], refs: list[str], body_limit: int, *, batch_indexes: list[int]
) -> list[AgentComment]:
    comments: list[AgentComment] = []
    batch_set = set(batch_indexes)
    for idx, text in enumerate(chunks[:body_limit]):
        if idx not in batch_set or idx >= len(refs):
            continue
        if _is_non_body_reference_context(refs[idx], text, index=idx, chunks=chunks, refs=refs):
            continue
        for match in re.finditer(r"\b([A-ZÀ-Ý][A-Za-zÀ-ÿ'’`\-]+)\((\d{4}[a-z]?)\)", text or ""):
            comments.append(
                AgentComment(
                    agent="referencias",
                    category="citation_format",
                    message="Falta um espaço antes do ano em citação autor-data.",
                    paragraph_index=idx,
                    issue_excerpt=match.group(0),
                    suggested_fix=f"{match.group(1)} ({match.group(2)})",
                )
            )
    return comments


def _find_reference_citation_indexes(chunks: list[str], refs: list[str], body_limit: int) -> list[int]:
    return sorted({idx for idx, _, _, _ in _reference_body_citation_mentions(chunks, refs, body_limit)})


def _heuristic_reference_global_comments(chunks: list[str], refs: list[str], batch_indexes: list[int]) -> list[AgentComment]:
    """Confronta corpo e lista de referências para apontar ausências e sobras."""
    reference_heading_idx = next((idx for idx, ref in enumerate(refs) if _ref_block_type(ref) == "reference_heading"), None)
    if reference_heading_idx is None:
        return []

    body_limit = reference_heading_idx
    citation_mentions = _reference_body_citation_mentions(chunks, refs, body_limit)
    citation_keys = {key for _, _, key, _ in citation_mentions}
    comments = _reference_body_format_comments(chunks, refs, body_limit, batch_indexes=batch_indexes)
    batch_set = set(batch_indexes)

    reference_entries: list[tuple[int, tuple[str, str], str]] = []
    for idx, (chunk, ref) in enumerate(zip(chunks[reference_heading_idx + 1 :], refs[reference_heading_idx + 1 :]), start=reference_heading_idx + 1):
        if _ref_block_type(ref) != "reference_entry":
            continue
        entry_key = _reference_entry_key(chunk)
        if entry_key is None:
            continue
        reference_entries.append((idx, entry_key, _reference_entry_label(chunk)))

    reference_keys = {key for _, key, _ in reference_entries}
    uncited_labels = [label for _, key, label in reference_entries if key not in citation_keys]
    missing_mentions = [(idx, excerpt, key, display) for idx, excerpt, key, display in citation_mentions if key not in reference_keys]
    missing_labels = [display for _, _, _, display in missing_mentions]

    for idx, excerpt, _, display in missing_mentions:
        if idx not in batch_set:
            continue
        comments.append(
            AgentComment(
                agent="referencias",
                category="citation_match",
                paragraph_index=idx,
                message="Esta citação no corpo do texto não tem correspondência clara na lista de referências.",
                issue_excerpt=excerpt,
                suggested_fix=f"Incluir ou revisar a referência correspondente a {display} na lista final.",
            )
        )

    if uncited_labels and reference_heading_idx in batch_set:
        comments.append(
            AgentComment(
                agent="referencias",
                category="inconsistency",
                paragraph_index=reference_heading_idx,
                message="Há referências na lista que não foram localizadas nas citações do corpo do texto.",
                issue_excerpt=chunks[reference_heading_idx],
                suggested_fix=f"Verificar estas obras: {_summarize_reference_labels(uncited_labels)}.",
            )
        )

    if missing_labels and reference_heading_idx in batch_set:
        comments.append(
            AgentComment(
                agent="referencias",
                category="inconsistency",
                paragraph_index=reference_heading_idx,
                message="Há citações no corpo do texto sem correspondência clara na lista de referências.",
                issue_excerpt=chunks[reference_heading_idx],
                suggested_fix=f"Incluir ou revisar as referências correspondentes a: {_summarize_reference_labels(missing_labels)}.",
            )
        )

    return comments


def _heuristic_table_figure_comments(batch_indexes: list[int], chunks: list[str], refs: list[str]) -> list[AgentComment]:
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
                )
            )

    return comments


def _is_same_top_level_heading(ref: str) -> bool:
    return _ref_block_type(ref) in {"heading", "reference_heading"} and "numerado=sim" in _normalized_text(ref)


def _is_final_section_heading(text: str) -> bool:
    folded = _folded_text(text)
    return folded.startswith("consideracoes finais") or folded.startswith("conclus")


def _heuristic_structure_comments(batch_indexes: list[int], chunks: list[str], refs: list[str]) -> list[AgentComment]:
    comments: list[AgentComment] = []
    heading_indexes = [idx for idx in batch_indexes if 0 <= idx < len(refs) and _ref_block_type(refs[idx]) in {"heading", "reference_heading"}]
    if not heading_indexes:
        return comments

    if any(_ref_has_numbering(refs[idx]) for idx in heading_indexes):
        next_number = 1
        for idx in heading_indexes:
            text = (chunks[idx] or "").strip()
            if _ref_has_numbering(refs[idx]):
                match = re.match(r"^\s*(\d+)", text)
                if match:
                    next_number = int(match.group(1)) + 1
                else:
                    next_number += 1
                continue
            if _heading_word_count(text) == 0:
                continue
            comments.append(
                AgentComment(
                    agent="estrutura",
                    category="numeração e hierarquia de seções",
                    message="Este título deveria seguir a sequência de numeração das seções.",
                    paragraph_index=idx,
                    issue_excerpt=text,
                    suggested_fix=f"{next_number}. {text}",
                    auto_apply=True,
                )
            )
            next_number += 1
    return comments


def _heuristic_typography_comments(batch_indexes: list[int], chunks: list[str], refs: list[str]) -> list[AgentComment]:
    comments: list[AgentComment] = []
    for idx in batch_indexes:
        if not (0 <= idx < len(refs)) or idx >= len(chunks):
            continue
        block_type = _ref_block_type(refs[idx])
        text = chunks[idx] or ""
        if block_type == "heading" and _ref_has_flag(refs[idx], "italico"):
            comments.append(
                AgentComment(
                    agent="tipografia",
                    category="heading",
                    message="Remover o itálico deste subtítulo e manter o negrito.",
                    paragraph_index=idx,
                    issue_excerpt=text,
                    suggested_fix="Remover itálico do título.",
                    auto_apply=True,
                    format_spec="italic=false",
                )
            )
        if block_type == "caption" and not _is_illustration_caption(text):
            comments.append(
                AgentComment(
                    agent="tipografia",
                    category="caption",
                    message="A legenda deve começar pelo identificador visual do elemento.",
                    paragraph_index=idx,
                    issue_excerpt=text,
                    suggested_fix=text,
                    auto_apply=False,
                )
            )
    return comments


def _heuristic_comments_for_agent(agent: str, batch_indexes: list[int], chunks: list[str], refs: list[str]) -> list[AgentComment]:
    """Centraliza as heurísticas determinísticas específicas de cada agente."""
    comments: list[AgentComment] = []
    if agent == "gramatica_ortografia":
        comments.extend(_heuristic_grammar_comments(batch_indexes=batch_indexes, chunks=chunks, refs=refs))
    if agent == "sinopse_abstract":
        comments.extend(_heuristic_synopsis_comments(batch_indexes=batch_indexes, chunks=chunks, refs=refs))
    if agent == "tabelas_figuras":
        comments.extend(_heuristic_table_figure_comments(batch_indexes=batch_indexes, chunks=chunks, refs=refs))
    if agent == "referencias":
        comments.extend(_heuristic_reference_comments(batch_indexes=batch_indexes, chunks=chunks, refs=refs))
        comments.extend(_heuristic_reference_global_comments(chunks=chunks, refs=refs, batch_indexes=batch_indexes))
    if agent == "estrutura":
        comments.extend(_heuristic_structure_comments(batch_indexes=batch_indexes, chunks=chunks, refs=refs))
    if agent == "tipografia":
        comments.extend(_heuristic_typography_comments(batch_indexes=batch_indexes, chunks=chunks, refs=refs))
    return comments

def _looks_like_reference_author(author_raw: str) -> bool:
    return _abnt_is_plausible_reference_author(author_raw, extra_blocked_tokens=_NON_AUTHOR_REFERENCE_TOKENS)


def _canonical_author_key(author_raw: str) -> str | None:
    return _abnt_canonical_author_key(author_raw, extra_blocked_tokens=_NON_AUTHOR_REFERENCE_TOKENS)


def _reference_citation_key(author_raw: str, year_raw: str) -> tuple[str, str] | None:
    return _abnt_canonical_reference_key(author_raw, year_raw, extra_blocked_tokens=_NON_AUTHOR_REFERENCE_TOKENS)


def _reference_citation_label(author_raw: str, year_raw: str) -> str:
    return _abnt_citation_label(author_raw, year_raw)


def _reference_entry_publication_year(text: str) -> str | None:
    return _abnt_publication_year_from_reference(text)


def _reference_body_citation_mentions(
    chunks: list[str], refs: list[str], body_limit: int
) -> list[tuple[int, str, tuple[str, str], str]]:
    candidates = extract_citation_candidates(
        chunks,
        refs,
        body_limit,
        is_non_body_context=_is_non_body_reference_context,
        blocked_author_tokens=_NON_AUTHOR_REFERENCE_TOKENS,
    )
    return [(item.paragraph_index, item.excerpt, item.key, item.label) for item in candidates]


def _reference_body_citation_keys(chunks: list[str], refs: list[str], body_limit: int) -> set[tuple[str, str]]:
    return {key for _, _, key, _ in _reference_body_citation_mentions(chunks, refs, body_limit)}


def _reference_entry_key(text: str) -> tuple[str, str] | None:
    parsed = parse_reference_entry(text, blocked_author_tokens=_NON_AUTHOR_REFERENCE_TOKENS)
    return parsed.key if parsed is not None else None


def _reference_entry_label(text: str) -> str:
    parsed = parse_reference_entry(text, blocked_author_tokens=_NON_AUTHOR_REFERENCE_TOKENS)
    return parsed.label if parsed is not None else (text or "").strip()[:80]


def _heuristic_reference_comments(batch_indexes: list[int], chunks: list[str], refs: list[str]) -> list[AgentComment]:
    comments: list[AgentComment] = []
    seen: set[tuple[int, str, str]] = set()

    def add(idx: int, issue: str, fix: str, message: str, category: str = "reference_format") -> None:
        key = (idx, issue, fix)
        if key in seen:
            return
        seen.add(key)
        comments.append(
            AgentComment(
                agent="referencias",
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
        block_type = _ref_block_type(refs[idx])
        text = chunks[idx] or ""
        parsed_entry = parse_reference_entry(text, blocked_author_tokens=_NON_AUTHOR_REFERENCE_TOKENS) if block_type == "reference_entry" else None

        if block_type == "paragraph":
            for match in re.finditer(r"\b([A-ZÀ-Ý][A-Za-zÀ-ÿ'’`\-]+)\((\d{4}[a-z]?)\)", text):
                add(idx, match.group(0), f"{match.group(1)} ({match.group(2)})", "Falta um espaço antes do ano em citação autor-data.", "citation_format")
        if block_type == "reference_entry" and re.search(r"\bIn:\S", text, flags=re.IGNORECASE):
            add(idx, "In:", "In: ", "Inserir espaço após 'In:' na referência.", "reference_format")
        if parsed_entry is not None and parsed_entry.document_type == "online" and "Acesso em:" not in text and "Acesso em :" not in text:
            add(
                idx,
                "Disponível em:",
                "Inserir `Acesso em:` com a data de consulta após a URL.",
                "A referência online informa a URL, mas não traz `Acesso em:` ao final.",
                "reference_format",
            )
        if block_type == "reference_entry" and re.search(r"([A-ZÀ-Ý][^.]*)\b(\d{4})\.(\D+[A-ZÀ-Ý])", text):
            match = re.search(r"(\d{4}\.[A-ZÀ-Ý])", text)
            if match:
                add(idx, match.group(1), match.group(1).replace(".", ". "), "Há duas referências coladas sem espaço entre elas.", "reference_format")
        if block_type == "reference_entry" and re.search(r"\bp\.(\d+[-–]\d+)", text, flags=re.IGNORECASE):
            match = re.search(r"(p\.\d+[-–]\d+)", text, flags=re.IGNORECASE)
            if match:
                add(idx, match.group(1), match.group(1).replace("p.", "p. "), "Falta espaço após a abreviatura de página.", "reference_format")
        if block_type == "reference_entry" and re.search(r"[A-Za-z0-9)\]]\s*$", text):
            add(idx, text, text.rstrip() + ".", "A referência termina sem ponto final.", "reference_format")
        duplicated_place = re.search(r"([A-ZÀ-Ý][A-Za-zÀ-ÿ\s]+):\s*\1,\s*\d{4}", text)
        if block_type == "reference_entry" and duplicated_place:
            add(idx, duplicated_place.group(0), duplicated_place.group(0), "Há duplicação de local e editora no trecho final da referência.", "reference_format")
        if parsed_entry is not None:
            for issue in validate_reference_entry(parsed_entry):
                add(idx, parsed_entry.raw_text, issue.suggested_fix, issue.message, issue.category)
        if block_type == "reference_entry":
            year_matches = re.findall(r"\b(?:19|20)\d{2}\b", text)
            if len(year_matches) >= 2 and year_matches[0] != year_matches[-1]:
                leading_match = re.search(r"\b(?:19|20)\d{2}\b", text)
                trailing_match = re.search(rf"\b{re.escape(year_matches[-1])}\b(?!.*\b(?:19|20)\d{{2}}\b)", text)
                if leading_match and trailing_match and leading_match.start() <= 60:
                    add(
                        idx,
                        trailing_match.group(0),
                        year_matches[0],
                        "O ano final da referência diverge do ano informado na abertura do registro.",
                        "reference_format",
                    )

    return comments


def _heuristic_reference_global_comments(chunks: list[str], refs: list[str], batch_indexes: list[int]) -> list[AgentComment]:
    reference_heading_idx = next((idx for idx, ref in enumerate(refs) if _ref_block_type(ref) == "reference_heading"), None)
    if reference_heading_idx is None:
        return []

    body_limit = reference_heading_idx
    citation_candidates = extract_citation_candidates(
        chunks,
        refs,
        body_limit,
        is_non_body_context=_is_non_body_reference_context,
        blocked_author_tokens=_NON_AUTHOR_REFERENCE_TOKENS,
    )
    comments = _reference_body_format_comments(chunks, refs, body_limit, batch_indexes=batch_indexes)
    batch_set = set(batch_indexes)

    reference_entries = [
        (idx, parsed)
        for idx, (chunk, ref) in enumerate(zip(chunks[reference_heading_idx + 1 :], refs[reference_heading_idx + 1 :]), start=reference_heading_idx + 1)
        if _ref_block_type(ref) == "reference_entry"
        for parsed in [parse_reference_entry(chunk, blocked_author_tokens=_NON_AUTHOR_REFERENCE_TOKENS)]
        if parsed is not None
    ]

    match_result = compare_citations_to_references(citation_candidates, [parsed for _, parsed in reference_entries])
    uncited_labels = [parsed.label for _, parsed in reference_entries if parsed in match_result.uncited_references]
    missing_labels = [candidate.label for candidate in match_result.missing_citations]

    for candidate in match_result.missing_citations:
        if candidate.paragraph_index not in batch_set:
            continue
        comments.append(
            AgentComment(
                agent="referencias",
                category="citation_match",
                paragraph_index=candidate.paragraph_index,
                message="Esta citação no corpo do texto não tem correspondência clara na lista de referências.",
                issue_excerpt=candidate.excerpt,
                suggested_fix=f"Incluir ou revisar a referência correspondente a {candidate.label} na lista final.",
            )
        )

    if uncited_labels and reference_heading_idx in batch_set:
        comments.append(
            AgentComment(
                agent="referencias",
                category="inconsistency",
                paragraph_index=reference_heading_idx,
                message="Há referências na lista que não foram localizadas nas citações do corpo do texto.",
                issue_excerpt=chunks[reference_heading_idx],
                suggested_fix=f"Verificar estas obras: {_summarize_reference_labels(uncited_labels)}.",
            )
        )

    if missing_labels and reference_heading_idx in batch_set:
        comments.append(
            AgentComment(
                agent="referencias",
                category="inconsistency",
                paragraph_index=reference_heading_idx,
                message="Há citações no corpo do texto sem correspondência clara na lista de referências.",
                issue_excerpt=chunks[reference_heading_idx],
                suggested_fix=f"Incluir ou revisar as referências correspondentes a: {_summarize_reference_labels(missing_labels)}.",
            )
        )

    return comments


__all__ = [
    "_find_reference_citation_indexes",
    "_heuristic_comments_for_agent",
    "_heuristic_grammar_comments",
    "_heuristic_reference_comments",
    "_heuristic_reference_global_comments",
    "_heuristic_structure_comments",
    "_heuristic_synopsis_comments",
    "_heuristic_table_figure_comments",
    "_heuristic_typography_comments",
    "_reference_body_citation_keys",
]

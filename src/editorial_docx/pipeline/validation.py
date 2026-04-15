from __future__ import annotations

import re

from ..comment_localizer import locate_comment_in_document
from ..models import AgentComment, VerificationDecision, VerificationSummary
from ..prompts import build_comment_review_prompt
from ..review_heuristics import _heuristic_comments_for_agent
from ..review_patterns import (
    _ALLOWED_TYPOGRAPHY_KEYS,
    _STYLE_BY_BLOCK_TYPE,
    _adds_coordination_comma,
    _comment_key,
    _comment_review_key,
    _contains_quote_marks,
    _count_words,
    _dedupe_comments,
    _drops_article_before_possessive,
    _extract_word_limit,
    _find_metadata_like_indexes,
    _folded_text,
    _has_repeated_keyword_entries,
    _indexes_by_ref_type,
    _introduces_plural_copula_for_singular_head,
    _is_demonstrative_swap,
    _is_grammar_rewrite_or_regency_comment,
    _is_illustration_caption,
    _is_non_body_reference_context,
    _is_reference_missing_data_speculation,
    _is_relevant_typography_spec,
    _looks_like_all_caps_title,
    _looks_like_full_reference_rewrite,
    _looks_like_quoted_excerpt,
    _normalized_text,
    _parse_format_spec,
    _punctuation_only_change,
    _quoted_terms,
    _ref_block_type,
    _ref_style_name,
    _removes_diacritic_only_word,
    _removes_terminal_period_only,
    _style_name_looks_explicit,
    _years_in_text,
)
from .runtime import (
    LLMConnectionFailure,
    _connection_error_summary,
    _invoke_with_model_fallback,
    _parse_comment_reviews,
    _sanitize_for_llm,
    _serialize_comments,
)


def _has_neighbor_with_prefix(paragraph_index: int, refs: list[str], chunks: list[str], prefixes: tuple[str, ...], radius: int = 2) -> bool:
    for candidate in range(max(0, paragraph_index - radius), min(len(chunks), paragraph_index + radius + 1)):
        text = (chunks[candidate] or "").strip().casefold()
        if any(text.startswith(prefix.casefold()) for prefix in prefixes):
            return True
    return False


def _find_excerpt_index(excerpt: str, candidate_indexes: list[int], chunks: list[str]) -> int | None:
    needle = _normalized_text(excerpt)
    if not needle:
        return None

    for idx in candidate_indexes:
        if 0 <= idx < len(chunks) and needle in _normalized_text(chunks[idx]):
            return idx

    window_chunks = [chunks[idx] for idx in candidate_indexes if 0 <= idx < len(chunks)]
    localized = locate_comment_in_document(excerpt, window_chunks)
    if localized is not None and 0 <= localized < len(candidate_indexes):
        return candidate_indexes[localized]
    return None


def _semantic_comment_key(item: AgentComment) -> tuple[str, int | None, str, str]:
    return (
        item.agent,
        item.paragraph_index if isinstance(item.paragraph_index, int) else None,
        _folded_text(item.issue_excerpt),
        _folded_text(item.suggested_fix),
    )


def _remap_comment_index(comment: AgentComment, batch_indexes: list[int], chunks: list[str]) -> AgentComment:
    paragraph_index = comment.paragraph_index

    if paragraph_index is None:
        paragraph_index = _find_excerpt_index(comment.issue_excerpt, batch_indexes, chunks)
        if paragraph_index is None and batch_indexes:
            paragraph_index = batch_indexes[0]
    elif paragraph_index not in batch_indexes and 0 <= paragraph_index < len(batch_indexes):
        paragraph_index = batch_indexes[paragraph_index]

    if paragraph_index is not None and batch_indexes and paragraph_index not in batch_indexes:
        matched = _find_excerpt_index(comment.issue_excerpt, batch_indexes, chunks)
        if matched is not None:
            paragraph_index = matched

    matched = _find_excerpt_index(comment.issue_excerpt, batch_indexes, chunks)
    if matched is not None:
        paragraph_index = matched

    return AgentComment(
        agent=comment.agent,
        category=comment.category,
        message=comment.message,
        paragraph_index=paragraph_index,
        issue_excerpt=comment.issue_excerpt,
        suggested_fix=comment.suggested_fix,
        auto_apply=comment.auto_apply,
        format_spec=comment.format_spec,
        review_status=comment.review_status,
        approved_text=comment.approved_text,
        reviewer_note=comment.reviewer_note,
    )


def _limit_auto_apply(comment: AgentComment) -> AgentComment:
    if not comment.auto_apply:
        return comment
    return AgentComment(
        agent=comment.agent,
        category=comment.category,
        message=comment.message,
        paragraph_index=comment.paragraph_index,
        issue_excerpt=comment.issue_excerpt,
        suggested_fix=comment.suggested_fix,
        auto_apply=False,
        format_spec=comment.format_spec,
        review_status=comment.review_status,
        approved_text=comment.approved_text,
        reviewer_note=comment.reviewer_note,
    )


def _tokenize_structure_text(value: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ÿ0-9]+", (value or "").casefold())


def _is_safe_structure_auto_apply(comment: AgentComment, chunks: list[str]) -> bool:
    if not isinstance(comment.paragraph_index, int) or not (0 <= comment.paragraph_index < len(chunks)):
        return False
    issue = (comment.issue_excerpt or "").strip()
    suggestion = (comment.suggested_fix or "").strip()
    source = (chunks[comment.paragraph_index] or "").strip()
    if not issue or not suggestion or not source:
        return False
    if _normalized_text(issue) != _normalized_text(source):
        return False
    return _tokenize_structure_text(issue) == _tokenize_structure_text(suggestion) == _tokenize_structure_text(source)


def _is_safe_text_normalization_auto_apply(comment: AgentComment, chunks: list[str]) -> bool:
    if not isinstance(comment.paragraph_index, int) or not (0 <= comment.paragraph_index < len(chunks)):
        return False
    issue = (comment.issue_excerpt or "").strip()
    suggestion = (comment.suggested_fix or "").strip()
    source = (chunks[comment.paragraph_index] or "").strip()
    if not issue or not suggestion or not source:
        return False
    if _normalized_text(issue) != _normalized_text(source):
        return False
    return _tokenize_structure_text(issue) == _tokenize_structure_text(suggestion) == _tokenize_structure_text(source)


def _matches_whole_paragraph(comment: AgentComment, chunks: list[str]) -> bool:
    if not isinstance(comment.paragraph_index, int) or not (0 <= comment.paragraph_index < len(chunks)):
        return False
    issue = (comment.issue_excerpt or "").strip()
    source = (chunks[comment.paragraph_index] or "").strip()
    if not issue or not source:
        return False
    return _normalized_text(issue) == _normalized_text(source)


def _should_keep_comment(comment: AgentComment, agent: str, chunks: list[str], refs: list[str]) -> bool:
    """Aplica filtros determinísticos para aceitar só comentários úteis e seguros."""
    if not (comment.message or "").strip():
        return False

    if comment.issue_excerpt and comment.suggested_fix and not comment.auto_apply:
        if _normalized_text(comment.issue_excerpt) == _normalized_text(comment.suggested_fix):
            return False

    ref = ""
    if isinstance(comment.paragraph_index, int) and 0 <= comment.paragraph_index < len(refs):
        ref = refs[comment.paragraph_index]
    block_type = _ref_block_type(ref)
    folded_message = _folded_text(comment.message)
    folded_fix = _folded_text(comment.suggested_fix)
    folded_blob = _folded_text(" ".join([comment.category or "", comment.message or "", comment.suggested_fix or ""]))
    source_text = ""
    if isinstance(comment.paragraph_index, int) and 0 <= comment.paragraph_index < len(chunks):
        source_text = chunks[comment.paragraph_index] or ""

    if agent == "estrutura":
        if "paragrafo" in folded_message:
            return False
        if block_type not in {"heading", "caption"} and any(token in folded_message for token in {"nao esta numerada", "deveria ser numerada", "numerar a secao"}):
            return False
        issue_text = comment.issue_excerpt or source_text
        if block_type == "caption" and (_is_illustration_caption(issue_text) or _is_illustration_caption(source_text)):
            if any(token in folded_blob for token in {"secao", "subsecao", "numerar a secao", "numerar"}):
                return False
        if block_type != "heading" and comment.issue_excerpt and not _matches_whole_paragraph(comment, chunks):
            if any(token in folded_blob for token in {"titulo", "secao", "subsecao", "numerada", "numerar"}):
                return False

    if agent == "estrutura" and block_type in {"direct_quote", "reference_entry", "table_cell"}:
        return False
    if agent == "estrutura" and block_type == "caption":
        source_text = ""
        if isinstance(comment.paragraph_index, int) and 0 <= comment.paragraph_index < len(chunks):
            source_text = chunks[comment.paragraph_index]
        issue_text = comment.issue_excerpt or source_text
        if _is_illustration_caption(issue_text) or _is_illustration_caption(source_text):
            structure_msg = _folded_text(comment.message)
            structure_fix = _folded_text(comment.suggested_fix)
            if any(token in structure_msg for token in {"seção", "secao", "subseção", "subsecao", "numerar a seção", "numerar a secao"}):
                return False
            if any(token in structure_fix for token in {"seção", "secao"}):
                return False
    if agent == "estrutura" and block_type not in {"heading", "caption"}:
        structure_msg = _normalized_text(comment.message)
        if any(token in structure_msg for token in {"não está numerada", "deveria ser numerada", "numerar a seção"}):
            return False
    if agent == "estrutura" and block_type == "caption":
        structure_blob = _normalized_text(" ".join([comment.message or "", comment.suggested_fix or ""]))
        if _is_illustration_caption(comment.issue_excerpt or "") and any(
            token in structure_blob for token in {"secao", "seção", "subsecao", "subseção"}
        ):
            return False
    if agent == "estrutura" and block_type == "heading" and comment.issue_excerpt:
        if not _matches_whole_paragraph(comment, chunks):
            return False
    if agent == "estrutura" and block_type != "heading":
        structure_blob = _normalized_text(" ".join([comment.message or "", comment.suggested_fix or ""]))
        title_tokens = {"titulo", "título", "secao", "seção", "subsecao", "subseção", "numerada", "numerar"}
        if comment.issue_excerpt and not _matches_whole_paragraph(comment, chunks):
            if any(token in structure_blob for token in title_tokens):
                return False
    if agent == "estrutura" and comment.auto_apply:
        if not _is_safe_structure_auto_apply(comment, chunks):
            return False

    if agent == "metadados":
        if block_type not in {"heading", "paragraph"}:
            return False
        if isinstance(comment.paragraph_index, int) and comment.paragraph_index >= 18:
            return False
        metadata_excerpt = _normalized_text(comment.issue_excerpt)
        metadata_message = _normalized_text(comment.message)
        if any(term in metadata_excerpt for term in {"não fornecido", "nao fornecido"}) and isinstance(comment.paragraph_index, int) and comment.paragraph_index > 12:
            return False
        if "placeholder" in metadata_message and "xxxxx" not in metadata_excerpt and "<td" not in metadata_excerpt:
            return False

    if agent == "tabelas_figuras":
        if not (comment.issue_excerpt or "").strip():
            return False
        issue_excerpt_folded = _folded_text(comment.issue_excerpt)
        table_blob_folded = folded_blob
        if block_type == "caption" and re.match(r"^(tabela|figura|quadro|grafico)\s+\d+\s*$", issue_excerpt_folded):
            return False
        if block_type == "caption" and comment.auto_apply:
            return False
        if block_type == "caption" and re.match(r"^(tabela|figura|quadro|grafico)\s+\d+[:\s]", issue_excerpt_folded):
            if any(token in table_blob_folded for token in {"identificador", "titulo", "subtitulo"}):
                if not any(token in table_blob_folded for token in {"mesma linha", "fundidos", "linha da legenda", "linha propria"}):
                    return False
        if re.match(r"^(tabela|figura|quadro|grafico)\s+\d+", issue_excerpt_folded) and "fonte" in table_blob_folded:
            if not any(token in table_blob_folded for token in {"abaixo do bloco", "linha propria"}):
                return False
        if "fonte" in folded_message and isinstance(comment.paragraph_index, int):
            if _has_neighbor_with_prefix(comment.paragraph_index, refs, chunks, ("Fonte:", "ElaboraÃ§Ã£o:", "Elaboracao:"), radius=2):
                return False
        if re.match(r"^(tabela|figura|quadro|grafico)\s+\d+[:\s]", issue_excerpt_folded) and any(
            token in table_blob_folded for token in {"falta identificador", "falta o identificador", "nao possui um identificador"}
        ):
            return False
        issue_excerpt = _normalized_text(comment.issue_excerpt)
        table_blob = _normalized_text(" ".join([comment.category or "", comment.message or "", comment.suggested_fix or ""]))
        if block_type == "table_cell" and any(token in table_blob for token in {"subtitulo", "subtítulo", "fonte", "identificador", "legenda"}):
            return False
        if block_type != "caption" and any(token in table_blob for token in {"subtitulo", "subtítulo", "fonte"}):
            return False
        if block_type == "caption" and re.match(r"^(tabela|figura|quadro|gr[aá]fico)\s+\d+[:\s]", issue_excerpt):
            if any(token in table_blob for token in {"identificador", "titulo", "título", "subtitulo", "subtítulo"}):
                if any(token in table_blob for token in {"mesma linha", "fundidos", "linha da legenda", "linha propria", "linha própria"}):
                    pass
                else:
                    return False
        if re.match(r"^(tabela|figura|quadro)\s+\d+", issue_excerpt):
            source_blob = _normalized_text(" ".join([comment.message or "", comment.suggested_fix or ""]))
            if "fonte" in source_blob:
                if "abaixo do bloco" in source_blob or "linha propria" in source_blob or "linha própria" in source_blob:
                    pass
                else:
                    return False
        if "fonte" in _normalized_text(comment.message) and isinstance(comment.paragraph_index, int):
            if _has_neighbor_with_prefix(comment.paragraph_index, refs, chunks, ("Fonte:", "Elaboração:"), radius=2):
                return False
        if re.match(r"^(tabela|figura|quadro|gr[aá]fico)\s+\d+[:\s]", issue_excerpt) and any(
            token in table_blob for token in {"falta o identificador", "nao possui um identificador", "não possui um identificador"}
        ):
            return False
        if comment.auto_apply and not _is_safe_text_normalization_auto_apply(comment, chunks):
            return False

    if agent == "tipografia":
        spec = _parse_format_spec(comment.format_spec)
        if not spec:
            return False
        if any(key not in _ALLOWED_TYPOGRAPHY_KEYS for key in spec):
            return False
        if not _is_relevant_typography_spec(spec):
            return False
        if comment.issue_excerpt and not _matches_whole_paragraph(comment, chunks):
            return False
        if block_type == "paragraph" and isinstance(comment.paragraph_index, int) and comment.paragraph_index >= 24:
            return False
        if block_type in {"reference_entry", "reference_heading"}:
            return False
        if block_type not in {"heading", "caption", "paragraph"}:
            return False
        if "alterar para '" in (comment.suggested_fix or "").casefold() or 'alterar para "' in (comment.suggested_fix or "").casefold():
            return False
        if any(token in _normalized_text(comment.suggested_fix) for token in {"reescrever", "substituir texto", "alterar conteúdo"}):
            return False

    if agent == "referencias" and block_type not in {"reference_entry", "reference_heading"}:
        if comment.category in {"citation_format", "citation_match"} and block_type in {"paragraph", "direct_quote", "list_item"}:
            pass
        else:
            return False
    if agent == "referencias" and comment.auto_apply:
        if not _is_safe_text_normalization_auto_apply(comment, chunks):
            return False
    if agent == "referencias" and isinstance(comment.paragraph_index, int):
        current = (chunks[comment.paragraph_index] or "").casefold()
        current_text = chunks[comment.paragraph_index] or ""
        current_ref = refs[comment.paragraph_index] if comment.paragraph_index < len(refs) else ""
        raw_message = (comment.message or "").casefold()
        if any(token in folded_blob for token in {"falta de informacoes", "adicionar informacoes", "caixa baixa", "caixa alta", "italico", "negrito", "destaque grafico"}):
            return False
        if "titulo" in folded_blob and _looks_like_all_caps_title(current_text):
            return False
        if "ponto final apos o numero" in _folded_text(raw_message):
            if re.search(r"\bn\.\s*\d+\s*,", comment.issue_excerpt or "", re.IGNORECASE):
                return False
        message_blob = _normalized_text(" ".join([comment.category or "", comment.message or "", comment.suggested_fix or ""]))
        suggestion_blob = _normalized_text(comment.suggested_fix)
        if comment.category in {"citation_format", "citation_match"} and _is_non_body_reference_context(
            current_ref,
            current_text,
            index=comment.paragraph_index,
            chunks=chunks,
            refs=refs,
        ):
            return False
        if any(token in message_blob for token in {"adicionar o titulo", "adicionar a pagina", "adicionar a paginacao", "adicionar o ano", "ano de publicacao", "verificar e corrigir o ano"}):
            return False
        if any(token in message_blob for token in {"falta de informacoes", "falta de informações", "adicionar informacoes", "adicionar informações"}):
            return False
        if "caixa baixa" in message_blob or "caixa alta" in message_blob:
            return False
        if any(token in message_blob for token in {"italico", "itálico", "negrito", "destaque grafico", "destaque gráfico"}):
            return False
        if any(token in message_blob for token in {"verificar", "confirmar", "informacoes suficientes", "informações suficientes"}) and _years_in_text(current_text):
            return False
        if any(token in message_blob for token in {"pontuacao final", "pontuação final", "ponto final", "pontuacao ao final", "pontuação ao final"}):
            if (current_text or "").rstrip().endswith((".", "!", "?")):
                return False
        if "in:" in current_text.casefold() and ("in:" in raw_message and ("uso incorreto" in raw_message or "inserir" in raw_message)):
            return False
        if "uso incorreto" in raw_message and "n." in raw_message:
            return False
        if "v." in raw_message and "espa" in raw_message and "volume" in raw_message:
            if "v." not in current_text:
                return False
        if ":" in raw_message and "espa" in raw_message:
            if not re.search(r":\S", comment.issue_excerpt or ""):
                return False
        if ("pontuação entre o título e a editora" in raw_message or "pontuacao entre o titulo e a editora" in _normalized_text(raw_message)):
            if "texto para discussão" in current_text.casefold() or "texto para discussao" in _normalized_text(current_text):
                return False
        if "titulo e a editora" in message_blob and "texto para discuss" in _normalized_text(current_text):
            return False
        if "n." in raw_message and "ponto" in raw_message:
            if re.search(r"\bn\.\s*\d+\s*,", current_text, re.IGNORECASE):
                return False
        if ("ponto final após o número" in raw_message or "ponto final apos o numero" in _normalized_text(raw_message)):
            if re.search(r"\bn\.\s*\d+\s*,", comment.issue_excerpt or "", re.IGNORECASE):
                return False
        if any(token in message_blob for token in {"titulo", "título", "autor", "ano", "periodico", "periódico"}) and _looks_like_full_reference_rewrite(current_text, comment.suggested_fix):
            return False
        if any(token in message_blob for token in {"titulo", "título"}) and re.search(r"\bpp?\.\s*\d", current_text):
            return False
        if _normalized_text(comment.suggested_fix) == _normalized_text(current_text):
            return False
        if any(token in message_blob for token in {"padrao de formatação", "padrao de formatacao", "padrão de formatação"}):
            return False
        if any(token in suggestion_blob for token in {"[ano]", "[local]", "[editora]"}) or "[" in (comment.suggested_fix or ""):
            return False
        source_text = current_text
        if "titulo" in message_blob and _looks_like_all_caps_title(source_text):
            return False
        if "ano" in _normalized_text(comment.category) or "ano" in _normalized_text(comment.message):
            current_years = _years_in_text(current_text)
            suggestion_years = _years_in_text(comment.suggested_fix)
            if current_years and suggestion_years and suggestion_years != current_years:
                return False
            if re.search(r"\b(19|20)\d{2}\b", current) and "alterar o ano" in _normalized_text(comment.suggested_fix):
                return False

    if agent == "conformidade_estilos":
        suggestion = (comment.suggested_fix or "").strip().upper()
        if not _matches_whole_paragraph(comment, chunks):
            return False
        allowed = _STYLE_BY_BLOCK_TYPE.get(block_type)
        if allowed and suggestion and suggestion not in allowed:
            return False
        if block_type == "paragraph" and suggestion in {"TITULO_1", "TÍTULO_1", "TITULO_2", "TÍTULO_2", "TITULO_3", "TÍTULO_3"}:
            return False

    if isinstance(comment.paragraph_index, int) and 0 <= comment.paragraph_index < len(chunks):
        if agent == "sinopse_abstract":
            source_text = chunks[comment.paragraph_index] or ""
            synopsis_blob = _normalized_text(" ".join([comment.message or "", comment.suggested_fix or ""]))
            if ("portugu" in synopsis_blob and "ingl" in synopsis_blob) or any(
                token in synopsis_blob for token in {"português e inglês", "portugues e ingles"}
            ):
                return False
            if any(
                token in synopsis_blob
                for token in {"nao inicia com letra maiuscula", "não inicia com letra maiúscula", "iniciar a frase com letra maiuscula", "iniciar a frase com letra maiúscula"}
            ):
                return False
            quoted_terms = _quoted_terms(" ".join([comment.message or "", comment.suggested_fix or ""]))
            issue_blob = _normalized_text(comment.issue_excerpt)
            if quoted_terms and not any(_normalized_text(term) in issue_blob for term in quoted_terms):
                return False
            word_limit = _extract_word_limit(" ".join([comment.message or "", comment.suggested_fix or ""]))
            if word_limit is not None:
                counted_text = comment.issue_excerpt or source_text
                if _count_words(counted_text) <= word_limit:
                    return False
            if block_type == "keywords_content":
                repetition_blob = _normalized_text(" ".join([comment.category or "", comment.message or "", comment.suggested_fix or ""]))
                if any(token in repetition_blob for token in {"repet", "redundan"}):
                    if not _has_repeated_keyword_entries(comment.issue_excerpt or source_text):
                        return False
        if comment.issue_excerpt:
            excerpt_ok = _find_excerpt_index(comment.issue_excerpt, [comment.paragraph_index], chunks)
            if excerpt_ok is None and agent in {"gramatica_ortografia", "referencias"}:
                return False
        if agent == "gramatica_ortografia":
            source_text = chunks[comment.paragraph_index] or ""
            grammar_blob = _normalized_text(" ".join([comment.category or "", comment.message or "", comment.suggested_fix or ""]))
            grammar_blob_folded = _folded_text(" ".join([comment.category or "", comment.message or "", comment.suggested_fix or ""]))
            if block_type == "direct_quote":
                return False
            if block_type == "reference_entry":
                return False
            if _looks_like_quoted_excerpt(comment.issue_excerpt):
                return False
            excerpt = (comment.issue_excerpt or "").strip()
            if _contains_quote_marks(source_text) and excerpt and len(excerpt) >= max(120, int(len(source_text) * 0.65)):
                return False
            if "pontua" in grammar_blob and excerpt and len(excerpt) > 120:
                return False
            if "concord" in grammar_blob and excerpt and len(excerpt) > 120:
                return False
            if any(token in grammar_blob_folded for token in {"duplicacao local", "repeticao imediata"}):
                return False
            if any(token in grammar_blob for token in {"clareza", "simplificada", "simplificar", "reestruturar", "reescr"}):
                return False
            if _adds_coordination_comma(excerpt or source_text, comment.suggested_fix):
                return False
            if _is_demonstrative_swap(excerpt or source_text, comment.suggested_fix):
                return False
            if _drops_article_before_possessive(excerpt or source_text, comment.suggested_fix):
                return False
            if _introduces_plural_copula_for_singular_head(excerpt or source_text, comment.suggested_fix):
                return False
            if "observam-se que" in _normalized_text(comment.suggested_fix):
                return False
            if _normalized_text(comment.suggested_fix) == _normalized_text(source_text):
                return False
            if _removes_terminal_period_only(comment.issue_excerpt or source_text, comment.suggested_fix):
                return False

    return True


def _basic_comment_rejection_reason(comment: AgentComment) -> str | None:
    if not (comment.message or "").strip():
        return "mensagem vazia"

    if comment.issue_excerpt and comment.suggested_fix and not comment.auto_apply:
        if _normalized_text(comment.issue_excerpt) == _normalized_text(comment.suggested_fix):
            return "sugestão idêntica ao trecho"
    return None


def _comment_rejection_reason(comment: AgentComment, agent: str, chunks: list[str], refs: list[str]) -> str | None:
    basic_reason = _basic_comment_rejection_reason(comment)
    if basic_reason is not None:
        return basic_reason

    if isinstance(comment.paragraph_index, int) and 0 <= comment.paragraph_index < len(chunks):
        source_text = chunks[comment.paragraph_index] or ""
        ref = refs[comment.paragraph_index] if comment.paragraph_index < len(refs) else ""
        block_type = _ref_block_type(ref)

        if agent == "sinopse_abstract":
            word_limit = _extract_word_limit(" ".join([comment.message or "", comment.suggested_fix or ""]))
            if word_limit is not None:
                counted_text = comment.issue_excerpt or source_text
                if _count_words(counted_text) <= word_limit:
                    return "alegação de limite de palavras não confirmada"
            if block_type == "keywords_content":
                repetition_blob = _normalized_text(" ".join([comment.category or "", comment.message or "", comment.suggested_fix or ""]))
                if any(token in repetition_blob for token in {"repet", "redundan"}):
                    if not _has_repeated_keyword_entries(comment.issue_excerpt or source_text):
                        return "alegação de repetição não confirmada"
        if agent == "gramatica_ortografia":
            excerpt = comment.issue_excerpt or source_text
            if _is_grammar_rewrite_or_regency_comment(comment.message, comment.suggested_fix):
                return "comentário gramatical de reescrita ou regência discutível"
            if _removes_diacritic_only_word(excerpt, comment.suggested_fix):
                return "remoção de acento não confirmada"
        if agent == "referencias" and block_type in {"reference_entry", "reference_heading"}:
            if _is_reference_missing_data_speculation(comment.message, comment.suggested_fix):
                return "completude bibliográfica sem evidência local"

    if not _should_keep_comment(comment, agent=agent, chunks=chunks, refs=refs):
        return "descartado por regra de verificação"
    return None


def _summarize_verification(decisions: list[VerificationDecision]) -> VerificationSummary:
    accepted_count = sum(1 for decision in decisions if decision.accepted)
    rejected_count = sum(1 for decision in decisions if not decision.accepted)
    return VerificationSummary(
        decisions=decisions[:],
        accepted_count=accepted_count,
        rejected_count=rejected_count,
    )


def _verify_batch_comments(
    comments: list[AgentComment],
    agent: str,
    batch_indexes: list[int],
    chunks: list[str],
    refs: list[str],
    existing_comments: list[AgentComment] | None = None,
    batch_index: int | None = None,
) -> tuple[list[AgentComment], list[VerificationDecision]]:
    """Combina saídas do LLM e heurísticas, removendo duplicatas e falsos positivos."""
    candidates: list[tuple[str, AgentComment]] = []
    for comment in comments:
        remapped = _limit_auto_apply(_remap_comment_index(comment, batch_indexes=batch_indexes, chunks=chunks))
        candidates.append(("llm", remapped))
    for comment in _heuristic_comments_for_agent(agent=agent, batch_indexes=batch_indexes, chunks=chunks, refs=refs):
        candidates.append(("heuristic", comment))

    accepted: list[AgentComment] = []
    decisions: list[VerificationDecision] = []
    seen_existing = {_comment_key(item) for item in (existing_comments or [])}
    seen_existing_semantic = {_semantic_comment_key(item) for item in (existing_comments or [])}
    seen_batch: set[tuple[str, str, int | None, str, str, str, bool, str]] = set()
    seen_batch_semantic: set[tuple[str, int | None, str, str]] = set()

    for source, candidate in candidates:
        key = _comment_key(candidate)
        semantic_key = _semantic_comment_key(candidate)
        if key in seen_existing or key in seen_batch or semantic_key in seen_existing_semantic or semantic_key in seen_batch_semantic:
            decisions.append(
                VerificationDecision(
                    comment=candidate,
                    accepted=False,
                    reason="comentário duplicado",
                    source=source,
                    batch_index=batch_index,
                )
            )
            continue

        reason = _basic_comment_rejection_reason(candidate)
        if reason is None and source == "llm":
            reason = _comment_rejection_reason(candidate, agent=agent, chunks=chunks, refs=refs)
        if reason is not None:
            decisions.append(
                VerificationDecision(
                    comment=candidate,
                    accepted=False,
                    reason=reason,
                    source=source,
                    batch_index=batch_index,
                )
            )
            continue

        accepted.append(candidate)
        seen_batch.add(key)
        seen_batch_semantic.add(semantic_key)
        decisions.append(
            VerificationDecision(
                comment=candidate,
                accepted=True,
                reason="aceito",
                source=source,
                batch_index=batch_index,
            )
        )

    return accepted, decisions


def _format_batch_status(status: str, decisions: list[VerificationDecision]) -> str:
    summary = _summarize_verification(decisions)
    base = (status or "").strip()
    suffix = f"verif: {summary.accepted_count} aceitos, {summary.rejected_count} rejeitados"
    return f"{base} | {suffix}" if base else suffix


def _normalize_batch_comments(
    comments: list[AgentComment],
    agent: str,
    batch_indexes: list[int],
    chunks: list[str],
    refs: list[str],
) -> list[AgentComment]:
    """Retorna apenas os comentários aprovados para um lote de revisão."""
    accepted, _ = _verify_batch_comments(
        comments=comments,
        agent=agent,
        batch_indexes=batch_indexes,
        chunks=chunks,
        refs=refs,
        existing_comments=[],
    )
    return accepted


_REVIEWER_ENABLED_AGENTS = {"sinopse_abstract"}


def _review_comments_with_llm(
    comments: list[AgentComment],
    agent: str,
    question: str,
    excerpt: str,
    profile_key: str | None,
) -> tuple[list[AgentComment], str]:
    """Passa os comentários elegíveis por um revisor LLM antes da consolidação."""
    if agent not in _REVIEWER_ENABLED_AGENTS or not comments:
        return comments, "revisor ignorado"

    prompt = build_comment_review_prompt(agent, profile_key=profile_key)
    payload = {
        "question": _sanitize_for_llm(question),
        "document_excerpt": _sanitize_for_llm(excerpt),
        "comments_json": _sanitize_for_llm(_serialize_comments(comments)),
    }
    try:
        response = _invoke_with_model_fallback(prompt, payload, operation=f"revisor {agent}")
        if response is None:
            return comments, "revisor indisponível"
    except LLMConnectionFailure as exc:
        return comments, f"revisor indisponível por conexão: {_connection_error_summary(exc.original)}"
    except Exception:
        return comments, "revisor indisponível"

    raw = response.content if isinstance(response.content, str) else str(response.content)
    reviews, status = _parse_comment_reviews(raw)
    if not reviews:
        return comments, status

    verdict_by_key = {
        _comment_review_key(
            item.get("paragraph_index"),
            str(item.get("issue_excerpt") or ""),
            str(item.get("suggested_fix") or ""),
        ): item
        for item in reviews
    }

    approved: list[AgentComment] = []
    rejected = 0
    for comment in comments:
        review = verdict_by_key.get(_comment_review_key(comment.paragraph_index, comment.issue_excerpt, comment.suggested_fix))
        if review and review.get("decision") == "reject":
            rejected += 1
            continue
        approved.append(comment)

    return approved, f"{status} | revisor: {len(approved)} aprovados, {rejected} rejeitados"


__all__ = [
    "_comment_rejection_reason",
    "_find_excerpt_index",
    "_format_batch_status",
    "_limit_auto_apply",
    "_matches_whole_paragraph",
    "_normalize_batch_comments",
    "_remap_comment_index",
    "_review_comments_with_llm",
    "_should_keep_comment",
    "_summarize_verification",
    "_verify_batch_comments",
]

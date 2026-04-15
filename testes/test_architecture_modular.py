from pathlib import Path

from editorial_docx.references.citations import extract_citation_candidates
from editorial_docx.references.matcher import compare_citations_to_references
from editorial_docx.references.parser import parse_reference_entry
from editorial_docx.references.validator import validate_reference_entry
from editorial_docx.benchmark_runner import discover_rais_documents
from editorial_docx.io.comment_localizer import locate_comment_in_document
from editorial_docx.io.document_loader import Section, load_normalized_document
from editorial_docx.models import AgentComment, DocumentUserComment
from editorial_docx.io.normalized_document import build_normalized_document
from editorial_docx.pipeline.consolidation import consolidate_semantic_comments
from editorial_docx.pipeline.context import prepare_review_document
from editorial_docx.pipeline.scope import _agent_scope_indexes
from editorial_docx.review_heuristics import _reference_entry_key
from editorial_docx.token_utils import TokenChunkConfig, chunk_index_windows


def test_build_normalized_document_persists_sections_comments_and_references():
    normalized = build_normalized_document(
        input_path=Path("amostra.docx"),
        kind="docx",
        chunks=["Introdução", "Walker e Oliveira (1991)", "Referências", "WALKER, J.; OLIVEIRA, F. Título. 1991."],
        refs=[
            "parágrafo 1 | tipo=heading | estilo=Heading 1",
            "parágrafo 2 | tipo=paragraph",
            "parágrafo 3 | tipo=reference_heading | estilo=Heading 1",
            "parágrafo 4 | tipo=reference_entry",
        ],
        sections=[Section(title="Introdução", start_idx=0, end_idx=1), Section(title="Referências", start_idx=2, end_idx=3)],
        toc=["Introdução [0-1]", "Referências [2-3]"],
        user_comments=[
            DocumentUserComment(
                comment_id=10,
                author="Editor",
                text="Buscar a referência.",
                paragraph_index=1,
                anchor_excerpt="Walker e Oliveira (1991)",
                paragraph_text="Walker e Oliveira (1991)",
            )
        ],
    )

    data = normalized.to_dict()
    assert data["metadata"]["kind"] == "docx"
    assert len(data["blocks"]) == 4
    assert data["blocks"][3]["block_type"] == "reference_entry"
    assert data["blocks"][1]["section_title"] == "Introdução"
    assert len(data["references"]) == 1
    assert len(data["user_comments"]) == 1


def test_chunk_index_windows_keeps_overlap_between_batches():
    items = [(idx, f"[{idx}] " + ("palavra " * 600)) for idx in range(4)]

    batches = chunk_index_windows(items, config=TokenChunkConfig(max_tokens=350, overlap_tokens=120, max_items=2))

    assert len(batches) >= 2
    assert batches[0][-1] == batches[1][0]


def test_prepare_review_document_batches_grammar_by_micro_batches():
    prepared = prepare_review_document(
        chunks=["Parágrafo A.", "Parágrafo B.", "Parágrafo C.", "Parágrafo D.", "Parágrafo E.", "Parágrafo F."],
        refs=[
            "parágrafo 1 | tipo=paragraph",
            "parágrafo 2 | tipo=paragraph",
            "parágrafo 3 | tipo=paragraph",
            "parágrafo 4 | tipo=paragraph",
            "parágrafo 5 | tipo=paragraph",
            "parágrafo 6 | tipo=paragraph",
        ],
        sections=[],
        agent_order=["gramatica_ortografia"],
        agent_scope_builder=lambda agent, chunks, refs, sections: [0, 1, 2, 3, 4, 5],
    )

    assert [batch.indexes for batch in prepared.agent_batches["gramatica_ortografia"]] == [[0, 1, 2, 3], [3, 4, 5]]


def test_agent_scope_indexes_grammar_uses_only_body_paragraphs_before_references():
    chunks = [
        "Título curto",
        "Parágrafo de corpo com extensão suficiente para entrar na revisão gramatical detalhada.",
        "Tabela 1",
        "Referências",
        "SILVA, J. Título. 2020.",
    ]
    refs = [
        "parágrafo 1 | tipo=heading",
        "parágrafo 2 | tipo=paragraph",
        "parágrafo 3 | tipo=table_cell",
        "parágrafo 4 | tipo=reference_heading",
        "parágrafo 5 | tipo=reference_entry",
    ]

    indexes = _agent_scope_indexes("gramatica_ortografia", chunks, refs, sections=[])

    assert indexes == [1]


def test_locate_comment_in_document_matches_fuzzy_excerpt():
    paragraphs = [
        "O texto introdutório abre a discussão.",
        "Walker e Oliveira, em 1991, apresentam a formulação original do problema.",
        "As considerações finais encerram a seção.",
    ]

    located = locate_comment_in_document("Walker e Oliveira (1991) apresentam a formulação", paragraphs)

    assert located == 1


def test_consolidate_semantic_comments_merges_near_duplicates():
    comments = [
        AgentComment(
            agent="gramatica_ortografia",
            category="Concordância",
            message="Ajustar a concordância verbal do trecho.",
            paragraph_index=4,
            issue_excerpt="e sugerem",
            suggested_fix="e sugere",
        ),
        AgentComment(
            agent="estrutura",
            category="Observação",
            message="Ajustar a concordância verbal neste ponto específico.",
            paragraph_index=4,
            issue_excerpt="e sugerem",
            suggested_fix="e sugere",
        ),
    ]

    consolidated = consolidate_semantic_comments(comments)

    assert len(consolidated) == 1


def test_load_normalized_document_restores_loaded_document_shape(tmp_path):
    normalized = build_normalized_document(
        input_path=tmp_path / "fonte.docx",
        kind="docx",
        chunks=["Introdução", "Texto", "Referências", "AUTOR. Título. 2020."],
        refs=["parágrafo 1 | tipo=heading", "parágrafo 2 | tipo=paragraph", "parágrafo 3 | tipo=reference_heading", "parágrafo 4 | tipo=reference_entry"],
        sections=[Section(title="Introdução", start_idx=0, end_idx=1), Section(title="Referências", start_idx=2, end_idx=3)],
        toc=["Introdução [0-1]", "Referências [2-3]"],
        user_comments=[],
    )
    path = tmp_path / "amostra_normalized_document.json"
    path.write_text(normalized.to_json(), encoding="utf-8")

    loaded = load_normalized_document(path)

    assert loaded.kind == "docx"
    assert loaded.chunks[1] == "Texto"
    assert loaded.refs[3].endswith("reference_entry")
    assert loaded.sections[1].title == "Referências"


def test_discover_rais_documents_ignores_outputs_and_history(tmp_path):
    (tmp_path / "20260414_capitulo_processos_RAIS.docx").write_text("x", encoding="utf-8")
    (tmp_path / "20260414_capitulo_processos_RAIS_output_gpt5_4.docx").write_text("x", encoding="utf-8")
    history_dir = tmp_path / "historico"
    history_dir.mkdir()
    (history_dir / "20260414_capitulo_processos_RAIS.docx").write_text("x", encoding="utf-8")

    discovered = discover_rais_documents(tmp_path)

    assert discovered == [tmp_path / "20260414_capitulo_processos_RAIS.docx"]


def test_reference_entry_key_uses_publication_year_instead_of_historical_year_inside_title():
    key = _reference_entry_key(
        "LEVY, Maria Stella Ferreira. O papel da migração internacional na evolução da população brasileira (1872 a 1972). Revista de Saúde Pública, v. 8, n. 1, p. 49-90, 1974. Disponível em: https://example.com. Acesso em: 14 abr. 2026."
    )

    assert key == ("levy", "1974")


def test_parse_reference_entry_extracts_abnt_like_fields():
    parsed = parse_reference_entry(
        "LEVY, Maria Stella Ferreira. O papel da migração internacional na evolução da população brasileira (1872 a 1972). Revista de Saúde Pública, v. 8, n. 1, p. 49-90, 1974. Disponível em: https://example.com. Acesso em: 14 abr. 2026."
    )

    assert parsed is not None
    assert parsed.author_key == "levy"
    assert parsed.publication_year == "1974"
    assert parsed.document_type == "online"
    assert parsed.has_url is True
    assert parsed.has_access_date is True


def test_extract_citation_candidates_reads_body_before_matching():
    candidates = extract_citation_candidates(
        chunks=[
            "Segundo De Negri (2001), a dinâmica industrial mudou.",
            "Outros trabalhos também retomam o ponto (Walker e Oliveira, 1991).",
            "Referências",
        ],
        refs=[
            "parágrafo 1 | tipo=paragraph",
            "parágrafo 2 | tipo=paragraph",
            "parágrafo 3 | tipo=reference_heading",
        ],
        body_limit=2,
        is_non_body_context=lambda ref, chunk, **kwargs: False,
    )

    assert [item.label for item in candidates] == ["De Negri (2001)", "Walker e Oliveira (1991)"]
    assert candidates[1].author_keys == ("walker", "oliveira")


def test_compare_citations_to_references_matches_casefolded_author_keys():
    citations = extract_citation_candidates(
        chunks=["Os fluxos migratórios perderam força (Levy, 1974).", "Referências"],
        refs=["parágrafo 1 | tipo=paragraph", "parágrafo 2 | tipo=reference_heading"],
        body_limit=1,
        is_non_body_context=lambda ref, chunk, **kwargs: False,
    )
    parsed = parse_reference_entry(
        "LEVY, Maria Stella Ferreira. O papel da migração internacional na evolução da população brasileira. Revista de Saúde Pública, 1974."
    )

    result = compare_citations_to_references(citations, [parsed] if parsed is not None else [])

    assert result.missing_citations == ()
    assert result.uncited_references == ()


def test_compare_citations_to_references_marks_partial_multi_author_conflict():
    citations = extract_citation_candidates(
        chunks=["No início dos anos 1990, Walker e Oliveira (1991) reportaram que a RAIS avançou.", "Referências"],
        refs=["parágrafo 1 | tipo=paragraph", "parágrafo 2 | tipo=reference_heading"],
        body_limit=1,
        is_non_body_context=lambda ref, chunk, **kwargs: False,
    )
    parsed = parse_reference_entry(
        "ARIAS, O.; OLIVEIRA, C. Inventario de informacion estadistica de periodo post censal. CEPAL, 1973/75."
    )

    result = compare_citations_to_references(citations, [parsed] if parsed is not None else [])

    assert result.missing_citations == ()
    assert len(result.probable_matches) == 1
    assert result.probable_matches[0].match_type == "partial_author_conflict"
    assert result.probable_matches[0].citation.label == "Walker e Oliveira (1991)"
    assert result.probable_matches[0].reference.label == "ARIAS e OLIVEIRA (1973/75)"


def test_compare_citations_to_references_marks_probable_year_mismatch_instead_of_missing():
    citations = extract_citation_candidates(
        chunks=["Como destaca Fraser (2001), o problema envolve redistribuição e reconhecimento.", "Referências"],
        refs=["parágrafo 1 | tipo=paragraph", "parágrafo 2 | tipo=reference_heading"],
        body_limit=1,
        is_non_body_context=lambda ref, chunk, **kwargs: False,
    )
    parsed = parse_reference_entry(
        "FRASER, N. Da redistribuição ao reconhecimento? Dilemas da justiça na era pós-socialista. In: SOUZA, J. (org.). Democracia hoje: novos desafios para a teoria democrática contemporânea. Brasília: Ed. da UnB, 2021. pp.246-82."
    )

    result = compare_citations_to_references(citations, [parsed] if parsed is not None else [])

    assert result.missing_citations == ()
    assert len(result.probable_matches) == 1
    assert result.probable_matches[0].match_type == "year_mismatch"
    assert result.probable_matches[0].citation.label == "Fraser (2001)"
    assert result.probable_matches[0].reference.label == "FRASER (2021)"


def test_parse_reference_entry_detects_glued_entries_and_keeps_first_reference_year():
    parsed = parse_reference_entry(
        "DESLANDES, Suely. Humanização: revisitando o conceito a partir das contribuições da sociologia médica. In: DESLANDES, S. F. et al. Humanização dos cuidados em saúde: conceitos, dilemas e práticas. Rio de Janeiro: Fiocruz, p. 33-47, 2006.DURKHEIM, E. Da divisão do trabalho social. São Paulo: Martins Fontes, 1999."
    )

    assert parsed is not None
    assert parsed.author_key == "deslandes"
    assert parsed.publication_year == "2006"
    assert parsed.has_glued_reference is True


def test_compare_citations_to_references_marks_probable_format_problem_for_glued_reference():
    citations = extract_citation_candidates(
        chunks=["A discussão retoma a noção de humanização (DESLANDES, 2006, p. 38).", "Referências"],
        refs=["parágrafo 1 | tipo=paragraph", "parágrafo 2 | tipo=reference_heading"],
        body_limit=1,
        is_non_body_context=lambda ref, chunk, **kwargs: False,
    )
    parsed = parse_reference_entry(
        "DESLANDES, Suely. Humanização: revisitando o conceito a partir das contribuições da sociologia médica. In: DESLANDES, S. F. et al. Humanização dos cuidados em saúde: conceitos, dilemas e práticas. Rio de Janeiro: Fiocruz, p. 33-47, 2006.DURKHEIM, E. Da divisão do trabalho social. São Paulo: Martins Fontes, 1999."
    )

    result = compare_citations_to_references(citations, [parsed] if parsed is not None else [])

    assert result.missing_citations == ()
    assert len(result.probable_matches) == 1
    assert result.probable_matches[0].match_type == "format_problem"
    assert result.probable_matches[0].reference.publication_year == "2006"


def test_validate_reference_entry_flags_missing_access_date_for_online_reference():
    parsed = parse_reference_entry(
        "IPEA. Catálogo de publicações. Brasília: Ipea, 2024. Disponível em: https://example.com/catalogo."
    )

    issues = validate_reference_entry(parsed) if parsed is not None else []

    assert any(issue.code == "access_date" for issue in issues)


def test_validate_reference_entry_flags_missing_url_for_online_reference():
    parsed = parse_reference_entry(
        "IPEA. Catálogo de publicações. Brasília: Ipea, 2024. Disponível em: ."
    )

    issues = validate_reference_entry(parsed) if parsed is not None else []

    assert parsed is not None
    assert parsed.document_type == "online"
    assert any(issue.code == "url" for issue in issues)

from editorial_docx import literature_grounding
from editorial_docx.models import LiteratureQuery, LiteratureWork


def test_retrieve_retries_with_llm_refined_query(monkeypatch):
    original_query = LiteratureQuery(text="consulta longa que falhou", source="llm")
    refined_query = LiteratureQuery(text="consulta simplificada", source="llm_fallback")
    calls: list[str] = []

    def fake_fetch(query, *, recent_years, per_query):
        calls.append(query.text)
        if query.source == "llm_fallback":
            return [LiteratureWork(source_id="W1", title="Trabalho recuperado")], None
        return [], "OpenAlex respondeu HTTP 400."

    monkeypatch.setattr(literature_grounding, "_fetch_openalex_works", fake_fetch)
    monkeypatch.setattr(
        literature_grounding,
        "_refine_openalex_query_with_llm",
        lambda query, context: (refined_query, True),
    )

    works, warnings = literature_grounding.retrieve_recent_literature(
        [original_query],
        {"title": "Documento", "excerpt": "texto", "keywords": [], "top_terms": []},
        recent_years=5,
        per_query=6,
        max_works=8,
    )

    assert calls == ["consulta longa que falhou", "consulta simplificada"]
    assert [work.title for work in works] == ["Trabalho recuperado"]
    assert warnings == []

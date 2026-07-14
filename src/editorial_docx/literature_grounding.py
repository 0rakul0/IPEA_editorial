from __future__ import annotations

import json
import math
import re
from datetime import UTC, datetime
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from langchain_core.prompts import ChatPromptTemplate

from .llm import _load_env, _read_env, get_chat_models, get_llm_timeout_seconds
from .models import LiteratureGroundingResult, LiteratureQuery, LiteratureWork
from .prompts.profiles import get_prompt_profile

_RECENT_SECTION_RE = re.compile(
    r"\b(resumo|abstract|sinopse|introdu[cç][aã]o|introduction|conclus[aã]o|considera[cç][oõ]es finais)\b",
    re.IGNORECASE,
)
_KEYWORD_LINE_RE = re.compile(
    r"\b(palavras-chave|keywords)\b\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)
_TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿ0-9][A-Za-zÀ-ÿ0-9\-_/]{2,}")
_GENERIC_SECTION_TITLES = {
    "abstract",
    "agradecimentos",
    "anexos",
    "appendix",
    "apêndice",
    "apendice",
    "conclusão",
    "conclusao",
    "considerações finais",
    "consideracoes finais",
    "documento",
    "introdução",
    "introducao",
    "keywords",
    "metodologia",
    "referências",
    "referencias",
    "resumo",
    "sinopse",
}
_STOPWORDS = {
    "abstract",
    "about",
    "after",
    "analysis",
    "and",
    "annual",
    "aos",
    "ara",
    "areas",
    "artigo",
    "assessment",
    "base",
    "based",
    "between",
    "com",
    "como",
    "contexto",
    "dados",
    "das",
    "de",
    "del",
    "des",
    "desde",
    "dos",
    "effects",
    "em",
    "entre",
    "essay",
    "este",
    "esta",
    "estudo",
    "for",
    "from",
    "impact",
    "impacts",
    "into",
    "mais",
    "method",
    "metodo",
    "model",
    "models",
    "na",
    "nas",
    "nos",
    "nosso",
    "nossa",
    "paper",
    "para",
    "pela",
    "pelas",
    "pelo",
    "pelos",
    "policy",
    "por",
    "recent",
    "resumo",
    "review",
    "sobre",
    "study",
    "the",
    "their",
    "through",
    "introdução",
    "introducao",
    "introduction",
    "uma",
    "use",
    "using",
}


def _tokenize(text: str) -> list[str]:
    return [
        token.lower()
        for token in _TOKEN_RE.findall(text or "")
        if token and token.lower() not in _STOPWORDS
    ]


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _extract_title(paragraphs: list[str], sections) -> str:
    for section in sections or []:
        title = _normalize_space(getattr(section, "title", ""))
        if title and title.casefold() not in _GENERIC_SECTION_TITLES:
            return title
    for paragraph in paragraphs:
        candidate = _normalize_space(paragraph)
        if candidate:
            return candidate[:200]
    return "Documento sem título identificado"


def _extract_keywords(paragraphs: list[str]) -> list[str]:
    keywords: list[str] = []
    for paragraph in paragraphs[:12]:
        match = _KEYWORD_LINE_RE.search(paragraph or "")
        if not match:
            continue
        raw_items = re.split(r"[;,•·]", match.group(2))
        for item in raw_items:
            cleaned = _normalize_space(item)
            if cleaned:
                keywords.append(cleaned)
    unique: list[str] = []
    seen: set[str] = set()
    for item in keywords:
        folded = item.casefold()
        if folded in seen:
            continue
        seen.add(folded)
        unique.append(item)
    return unique[:8]


def _extract_manuscript_excerpt(paragraphs: list[str], sections) -> str:
    selected: list[str] = []
    for section in sections or []:
        title = _normalize_space(getattr(section, "title", ""))
        if not _RECENT_SECTION_RE.search(title):
            continue
        start_idx = max(0, int(getattr(section, "start_idx", 0)))
        end_idx = min(len(paragraphs) - 1, int(getattr(section, "end_idx", -1)))
        for idx in range(start_idx, min(end_idx + 1, start_idx + 4)):
            text = _normalize_space(paragraphs[idx])
            if text:
                selected.append(f"[{title}] {text}")
    if not selected:
        for paragraph in paragraphs[:8]:
            cleaned = _normalize_space(paragraph)
            if cleaned:
                selected.append(cleaned)
    excerpt = "\n".join(selected)
    return excerpt[:5000]


def _top_terms(text: str, *, limit: int = 12) -> list[str]:
    counts: dict[str, int] = {}
    for token in _tokenize(text):
        counts[token] = counts.get(token, 0) + 1
    return [term for term, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _build_manuscript_context(paragraphs: list[str], sections, profile_key: str | None) -> dict[str, object]:
    title = _extract_title(paragraphs, sections)
    excerpt = _extract_manuscript_excerpt(paragraphs, sections)
    keywords = _extract_keywords(paragraphs)
    profile = get_prompt_profile(profile_key)
    section_titles = [
        _normalize_space(getattr(section, "title", ""))
        for section in (sections or [])[:12]
        if _normalize_space(getattr(section, "title", ""))
    ]
    top_terms = _top_terms(" ".join([title, excerpt, *keywords]), limit=10)
    return {
        "title": title,
        "profile_description": profile.description,
        "profile_instruction": profile.instruction,
        "section_titles": section_titles,
        "keywords": keywords,
        "excerpt": excerpt,
        "top_terms": top_terms,
    }


def _openalex_api_key() -> str:
    _load_env()
    return _read_env("OPENALEX_API_KEY")


def _openalex_headers() -> dict[str, str]:
    _load_env()
    email = _read_env("OPENALEX_EMAIL", "OPENALEX_MAILTO")
    user_agent = "lang-IPEA-editorial/0.2 literature-grounding"
    if email:
        user_agent += f" ({email})"
    return {
        "Accept": "application/json",
        "User-Agent": user_agent,
    }


def _invoke_prompt(prompt: ChatPromptTemplate, payload: dict[str, str]) -> tuple[str | None, bool]:
    for _, model in get_chat_models():
        try:
            response = (prompt | model).invoke(payload)
        except Exception:
            continue
        content = response.content if isinstance(response.content, str) else str(response.content)
        if content.strip():
            return content, True
    return None, False


def _extract_json_list(raw: str) -> list[dict[str, object]]:
    candidate = (raw or "").strip()
    if not candidate:
        return []
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        match = re.search(r"\[[\s\S]*\]", candidate)
        if not match:
            return []
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []


def _extract_json_object(raw: str) -> dict[str, object]:
    candidate = (raw or "").strip()
    if not candidate:
        return {}
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", candidate)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}


def _is_query_useful(text: str) -> bool:
    cleaned = _normalize_space(text)
    if not cleaned:
        return False
    if cleaned.casefold() in _GENERIC_SECTION_TITLES:
        return False
    tokens = list(dict.fromkeys(_tokenize(cleaned)))
    if len(tokens) < 2:
        return False
    if len(cleaned) < 12 and len(tokens) < 3:
        return False
    return True


def _generate_queries_with_llm(context: dict[str, object]) -> tuple[list[LiteratureQuery], bool]:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você cria consultas de busca bibliográfica para contextualizar um manuscrito acadêmico. "
                "Retorne APENAS um JSON válido com uma lista de 3 a 4 objetos no formato "
                '[{"query":"...","rationale":"..."}]. '
                "As consultas devem combinar uma busca ampla, uma busca metodológica, uma busca temática específica "
                "e, quando fizer sentido, uma busca em inglês com termos usuais da literatura.",
            ),
            (
                "human",
                "Perfil documental: {profile_description}\n"
                "Instrução editorial: {profile_instruction}\n"
                "Título do manuscrito: {title}\n"
                "Seções: {section_titles}\n"
                "Palavras-chave: {keywords}\n"
                "Trechos centrais:\n{excerpt}\n"
                "Termos frequentes: {top_terms}\n",
            ),
        ]
    )
    raw, used_llm = _invoke_prompt(
        prompt,
        {
            "profile_description": str(context.get("profile_description") or ""),
            "profile_instruction": str(context.get("profile_instruction") or ""),
            "title": str(context.get("title") or ""),
            "section_titles": json.dumps(context.get("section_titles") or [], ensure_ascii=False),
            "keywords": json.dumps(context.get("keywords") or [], ensure_ascii=False),
            "excerpt": str(context.get("excerpt") or ""),
            "top_terms": json.dumps(context.get("top_terms") or [], ensure_ascii=False),
        },
    )
    payload = _extract_json_list(raw or "")
    queries: list[LiteratureQuery] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        text = _normalize_space(str(item.get("query") or ""))
        rationale = _normalize_space(str(item.get("rationale") or ""))
        if _is_query_useful(text):
            queries.append(LiteratureQuery(text=text[:240], rationale=rationale[:220], source="llm"))
    return queries[:4], used_llm and bool(queries)


def _generate_queries_heuristically(context: dict[str, object]) -> list[LiteratureQuery]:
    title = str(context.get("title") or "")
    keywords = [str(item) for item in (context.get("keywords") or []) if str(item).strip()]
    terms = [str(item) for item in (context.get("top_terms") or []) if str(item).strip()]
    queries: list[LiteratureQuery] = []

    if title:
        queries.append(
            LiteratureQuery(
                text=title[:240],
                rationale="Busca ampla baseada no título identificado no manuscrito.",
                source="heuristic",
            )
        )

    if keywords:
        queries.append(
            LiteratureQuery(
                text=" ".join(keywords[:5])[:240],
                rationale="Busca temática a partir das palavras-chave explícitas do documento.",
                source="heuristic",
            )
        )

    if terms:
        queries.append(
            LiteratureQuery(
                text=" ".join(terms[:7])[:240],
                rationale="Busca de reforço usando os termos mais recorrentes no manuscrito.",
                source="heuristic",
            )
        )

    if title and terms:
        queries.append(
            LiteratureQuery(
                text=f"{title[:120]} {' '.join(terms[:4])}".strip()[:240],
                rationale="Busca combinando o foco declarado e os termos dominantes do texto.",
                source="heuristic",
            )
        )

    deduped: list[LiteratureQuery] = []
    seen: set[str] = set()
    for item in queries:
        if not _is_query_useful(item.text):
            continue
        folded = item.text.casefold()
        if not folded or folded in seen:
            continue
        seen.add(folded)
        deduped.append(item)
    return deduped[:4]


def generate_literature_queries(
    paragraphs: list[str],
    sections,
    profile_key: str | None = None,
) -> tuple[list[LiteratureQuery], dict[str, object], bool]:
    context = _build_manuscript_context(paragraphs, sections, profile_key)
    queries, used_llm = _generate_queries_with_llm(context)
    if queries:
        return queries, context, used_llm
    return _generate_queries_heuristically(context), context, False


def _reconstruct_abstract(payload: dict[str, object]) -> str:
    abstract = payload.get("abstract")
    if isinstance(abstract, str) and abstract.strip():
        return _normalize_space(abstract)

    inverted = payload.get("abstract_inverted_index")
    if not isinstance(inverted, dict) or not inverted:
        return ""

    max_position = -1
    for positions in inverted.values():
        if isinstance(positions, list):
            for pos in positions:
                if isinstance(pos, int):
                    max_position = max(max_position, pos)
    if max_position < 0:
        return ""

    words = [""] * (max_position + 1)
    for token, positions in inverted.items():
        if not isinstance(token, str) or not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int) and 0 <= pos < len(words):
                words[pos] = token
    return _normalize_space(" ".join(word for word in words if word))


def _parse_openalex_work(item: dict[str, object], query_text: str) -> LiteratureWork | None:
    source_id = _normalize_space(str(item.get("id") or ""))
    title = _normalize_space(str(item.get("display_name") or item.get("title") or ""))
    if not source_id or not title:
        return None

    authors = []
    for authorship in item.get("authorships", []) if isinstance(item.get("authorships"), list) else []:
        if not isinstance(authorship, dict):
            continue
        author = authorship.get("author")
        if isinstance(author, dict):
            name = _normalize_space(str(author.get("display_name") or ""))
            if name:
                authors.append(name)

    primary_location = item.get("primary_location")
    landing_page_url = ""
    venue = ""
    if isinstance(primary_location, dict):
        landing_page_url = _normalize_space(str(primary_location.get("landing_page_url") or ""))
        source = primary_location.get("source")
        if isinstance(source, dict):
            venue = _normalize_space(str(source.get("display_name") or ""))

    doi = _normalize_space(str(item.get("doi") or ""))
    publication_year = item.get("publication_year")
    if not isinstance(publication_year, int):
        publication_year = None

    cited_by_count = item.get("cited_by_count")
    if not isinstance(cited_by_count, int):
        cited_by_count = 0

    return LiteratureWork(
        source_id=source_id,
        title=title,
        abstract=_reconstruct_abstract(item),
        authors=authors[:8],
        publication_year=publication_year,
        publication_date=_normalize_space(str(item.get("publication_date") or "")),
        venue=venue,
        doi=doi,
        landing_page_url=landing_page_url or source_id,
        cited_by_count=cited_by_count,
        matched_queries=[query_text],
    )


def _fetch_openalex_works(
    query: LiteratureQuery,
    *,
    recent_years: int,
    per_query: int,
) -> tuple[list[LiteratureWork], str | None]:
    base_url = "https://api.openalex.org/works"
    current_year = datetime.now(UTC).year
    from_year = max(1900, current_year - max(1, recent_years) + 1)
    token_count = len(_tokenize(query.text))
    search_parameter = "search.semantic" if token_count >= 8 or len(query.text) >= 80 else "search"
    params = {
        search_parameter: query.text,
        "filter": f"from_publication_date:{from_year}-01-01,has_abstract:true,type:!paratext",
        "per_page": str(max(1, min(per_query, 25))),
        "mailto": _read_env("OPENALEX_EMAIL", "OPENALEX_MAILTO"),
    }
    api_key = _openalex_api_key()
    if api_key:
        params["api_key"] = api_key

    encoded = urlencode({key: value for key, value in params.items() if value})
    request = Request(f"{base_url}?{encoded}", headers=_openalex_headers())
    timeout = get_llm_timeout_seconds()
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        return [], f"OpenAlex respondeu HTTP {exc.code} para a consulta '{query.text}'."
    except (URLError, TimeoutError, OSError, ValueError) as exc:
        return [], f"Falha ao consultar OpenAlex para '{query.text}': {exc}"

    results = payload.get("results")
    if not isinstance(results, list):
        return [], f"OpenAlex não retornou resultados válidos para '{query.text}'."

    works: list[LiteratureWork] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        work = _parse_openalex_work(item, query.text)
        if work is not None:
            works.append(work)
    return works, None


def _score_work(work: LiteratureWork, manuscript_terms: set[str], query_terms: set[str], recent_years: int) -> float:
    haystack = " ".join([work.title, work.abstract, work.venue])
    work_terms = set(_tokenize(haystack))
    overlap_query = len(work_terms & query_terms) / max(1, len(query_terms))
    overlap_manuscript = len(work_terms & manuscript_terms) / max(1, len(manuscript_terms))
    year_score = 0.0
    current_year = datetime.now(UTC).year
    if isinstance(work.publication_year, int):
        age = max(0, current_year - work.publication_year)
        year_score = max(0.0, 1.0 - (age / max(1, recent_years + 1)))
    citation_score = min(math.log1p(max(0, work.cited_by_count)) / 6.0, 1.0)
    abstract_bonus = 0.2 if work.abstract else 0.0
    return round((overlap_query * 3.0) + (overlap_manuscript * 2.5) + year_score + citation_score + abstract_bonus, 4)


def retrieve_recent_literature(
    queries: list[LiteratureQuery],
    manuscript_context: dict[str, object],
    *,
    recent_years: int = 5,
    per_query: int = 6,
    max_works: int = 8,
    on_status: Callable[[str], None] | None = None,
) -> tuple[list[LiteratureWork], list[str]]:
    warnings: list[str] = []
    by_key: dict[str, LiteratureWork] = {}
    query_cap = 2 if not _openalex_api_key() else len(queries)
    manuscript_terms = set(_tokenize(" ".join(
        [
            str(manuscript_context.get("title") or ""),
            str(manuscript_context.get("excerpt") or ""),
            " ".join(str(item) for item in (manuscript_context.get("keywords") or [])),
        ]
    )))

    for query in queries[:query_cap]:
        if on_status is not None:
            on_status(f"Buscando literatura para: {query.text}")
        works, warning = _fetch_openalex_works(query, recent_years=recent_years, per_query=per_query)
        if warning:
            warnings.append(warning)
        query_terms = set(_tokenize(query.text))
        for work in works:
            key = work.doi or work.source_id or work.title.casefold()
            existing = by_key.get(key)
            work.relevance_score = _score_work(work, manuscript_terms, query_terms, recent_years)
            if existing is None:
                by_key[key] = work
                continue
            existing.relevance_score = max(existing.relevance_score, work.relevance_score)
            for matched_query in work.matched_queries:
                if matched_query not in existing.matched_queries:
                    existing.matched_queries.append(matched_query)

    if query_cap < len(queries):
        warnings.append(
            "Sem OPENALEX_API_KEY configurada, o app limitou o número de consultas para preservar a cota anônima."
        )

    ranked = sorted(
        by_key.values(),
        key=lambda item: (
            -item.relevance_score,
            -(item.publication_year or 0),
            -item.cited_by_count,
            item.title.casefold(),
        ),
    )
    return ranked[:max(1, max_works)], warnings


def _works_as_prompt_payload(works: list[LiteratureWork]) -> str:
    serialized = []
    for work in works:
        serialized.append(
            {
                "title": work.title,
                "authors": work.authors,
                "publication_year": work.publication_year,
                "venue": work.venue,
                "doi": work.doi,
                "landing_page_url": work.landing_page_url,
                "cited_by_count": work.cited_by_count,
                "matched_queries": work.matched_queries,
                "abstract": work.abstract[:1800],
            }
        )
    return json.dumps(serialized, ensure_ascii=False, indent=2)


def _fallback_state_of_art_summary(works: list[LiteratureWork]) -> str:
    if not works:
        return "Nenhum trabalho recente foi recuperado para construir um estado da arte confiável."
    lines = []
    for work in works[:5]:
        year = f" ({work.publication_year})" if work.publication_year else ""
        venue = f" em {work.venue}" if work.venue else ""
        abstract_excerpt = work.abstract[:240] + ("..." if len(work.abstract) > 240 else "")
        lines.append(f"- {work.title}{year}{venue}: {abstract_excerpt or 'resumo não disponível.'}")
    return "Trabalhos recentes mais relevantes identificados:\n" + "\n".join(lines)


def _fallback_comparison(manuscript_context: dict[str, object], works: list[LiteratureWork]) -> str:
    if not works:
        return "Sem base externa suficiente para comparar o manuscrito com a literatura recente."
    keywords = manuscript_context.get("keywords") or []
    top_titles = ", ".join(work.title for work in works[:3])
    keyword_text = ", ".join(str(item) for item in keywords[:5]) or "sem palavras-chave explícitas"
    return (
        "Comparação preliminar: o manuscrito parece dialogar com os temas "
        f"{keyword_text}. A literatura recuperada se concentra especialmente em {top_titles}. "
        "Ainda assim, esta comparação é automática e deve ser lida como apoio ao revisor, não como parecer final."
    )


def synthesize_grounded_review(
    manuscript_context: dict[str, object],
    works: list[LiteratureWork],
) -> tuple[str, str, str, bool]:
    manuscript_summary = _normalize_space(str(manuscript_context.get("excerpt") or ""))[:1200]
    if not works:
        return (
            manuscript_summary,
            "Nenhum trabalho recente foi recuperado, então o estado da arte relevante não pôde ser sintetizado.",
            "Sem base externa recuperada, a comparação com o manuscrito permanece indisponível.",
            False,
        )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Você apoia uma leitura substantiva de manuscritos. "
                "Baseie-se apenas no manuscrito e na literatura recuperada. "
                "Não invente trabalhos, dados ou conclusões não presentes no material. "
                "Responda APENAS com JSON válido no formato "
                '{"manuscript_summary":"...","state_of_art_summary":"...","manuscript_comparison":"..."}.',
            ),
            (
                "human",
                "Perfil documental: {profile_description}\n"
                "Título do manuscrito: {title}\n"
                "Seções: {section_titles}\n"
                "Palavras-chave: {keywords}\n"
                "Trechos centrais do manuscrito:\n{excerpt}\n\n"
                "Literatura recuperada:\n{works_json}\n",
            ),
        ]
    )
    raw, used_llm = _invoke_prompt(
        prompt,
        {
            "profile_description": str(manuscript_context.get("profile_description") or ""),
            "title": str(manuscript_context.get("title") or ""),
            "section_titles": json.dumps(manuscript_context.get("section_titles") or [], ensure_ascii=False),
            "keywords": json.dumps(manuscript_context.get("keywords") or [], ensure_ascii=False),
            "excerpt": str(manuscript_context.get("excerpt") or ""),
            "works_json": _works_as_prompt_payload(works),
        },
    )
    payload = _extract_json_object(raw or "")
    if used_llm and payload:
        manuscript_summary = _normalize_space(str(payload.get("manuscript_summary") or manuscript_summary))
        state_of_art_summary = _normalize_space(str(payload.get("state_of_art_summary") or ""))
        manuscript_comparison = _normalize_space(str(payload.get("manuscript_comparison") or ""))
        if state_of_art_summary and manuscript_comparison:
            return manuscript_summary, state_of_art_summary, manuscript_comparison, True

    return (
        manuscript_summary,
        _fallback_state_of_art_summary(works),
        _fallback_comparison(manuscript_context, works),
        False,
    )


def run_literature_grounding(
    paragraphs: list[str],
    sections,
    *,
    profile_key: str | None = None,
    recent_years: int = 5,
    per_query: int = 6,
    max_works: int = 8,
    on_status: Callable[[str], None] | None = None,
) -> LiteratureGroundingResult:
    result = LiteratureGroundingResult(provider="openalex")
    if not paragraphs:
        result.warnings.append("Documento vazio ou sem texto extraído para grounding externo.")
        return result

    if on_status is not None:
        on_status("Gerando consultas de busca a partir do manuscrito.")
    queries, manuscript_context, queries_used_llm = generate_literature_queries(
        paragraphs,
        sections,
        profile_key=profile_key,
    )
    result.queries = queries
    if not queries:
        result.warnings.append("Não foi possível gerar consultas de busca para a literatura recente.")
        return result

    works, warnings = retrieve_recent_literature(
        queries,
        manuscript_context,
        recent_years=recent_years,
        per_query=per_query,
        max_works=max_works,
        on_status=on_status,
    )
    result.works = works
    result.warnings.extend(warnings)

    if on_status is not None:
        on_status("Sintetizando o estado da arte e comparando o manuscrito com a base recuperada.")
    manuscript_summary, state_of_art_summary, manuscript_comparison, summary_used_llm = synthesize_grounded_review(
        manuscript_context,
        works,
    )
    result.manuscript_summary = manuscript_summary
    result.state_of_art_summary = state_of_art_summary
    result.manuscript_comparison = manuscript_comparison
    result.llm_used = queries_used_llm or summary_used_llm

    if not works:
        result.warnings.append(
            "A busca não retornou trabalhos suficientes; amplie a janela temporal ou ajuste o foco temático."
        )
    return result


def literature_grounding_to_dict(result: LiteratureGroundingResult) -> dict[str, object]:
    return {
        "provider": result.provider,
        "llm_used": result.llm_used,
        "manuscript_summary": result.manuscript_summary,
        "state_of_art_summary": result.state_of_art_summary,
        "manuscript_comparison": result.manuscript_comparison,
        "warnings": result.warnings,
        "queries": [
            {
                "text": query.text,
                "rationale": query.rationale,
                "source": query.source,
            }
            for query in result.queries
        ],
        "works": [
            {
                "source_id": work.source_id,
                "title": work.title,
                "abstract": work.abstract,
                "authors": work.authors,
                "publication_year": work.publication_year,
                "publication_date": work.publication_date,
                "venue": work.venue,
                "doi": work.doi,
                "landing_page_url": work.landing_page_url,
                "cited_by_count": work.cited_by_count,
                "relevance_score": work.relevance_score,
                "matched_queries": work.matched_queries,
            }
            for work in result.works
        ],
    }

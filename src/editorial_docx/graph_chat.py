from __future__ import annotations

import time

from .agents.user_reference_agent import USER_REFERENCE_AGENT
from .llm import get_chat_model, get_chat_models, get_llm_retry_config
from .models import AgentComment, ConversationResult, VerificationDecision
from .prompts import build_agent_prompt, build_comment_review_prompt
from .review_coordinator import coordinate_answer
from .review_heuristics import _find_reference_citation_indexes, _heuristic_reference_comments, _heuristic_reference_global_comments
from .review_orchestrator import _build_graph as _default_build_graph
from .review_runtime import (
    LLMConnectionFailure,
    _build_batch_review_excerpt,
    _comment_memory_lines,
    _connection_error_summary,
    _is_connection_error,
    _is_json_body_error,
    _parse_comment_reviews,
    _parse_comments,
    _partial_answer_from_comments,
    _sanitize_for_llm,
    _serialize_comments,
    _truncate_progressive_summary,
    _update_running_summary,
)
from .review_patterns import _folded_text
from .review_scope import _agent_scope_indexes, _consolidate_final_comments, prepare_review_batches
from .review_validation import _normalize_batch_comments, _summarize_verification, _verify_batch_comments, _format_batch_status
from .user_comment_refs import (
    ReferenceSearchRequest,
    build_reference_search_requests,
    candidates_as_json,
    format_reference_candidate,
    reference_already_present,
    search_reference_candidates,
)


def _invoke_with_retry(runnable, payload: dict[str, str], operation: str):
    retry_config = get_llm_retry_config()
    max_retries = int(retry_config["max_retries"])
    backoff_seconds = float(retry_config["backoff_seconds"])
    last_exc: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            return runnable.invoke(payload)
        except Exception as exc:
            if _is_json_body_error(exc) or not _is_connection_error(exc):
                raise
            last_exc = exc
            if attempt >= max_retries:
                break
            if backoff_seconds > 0:
                time.sleep(backoff_seconds * (2 ** (attempt - 1)))

    if last_exc is None:
        raise RuntimeError(f"{operation} falhou sem exceção capturada.")
    raise LLMConnectionFailure(operation=operation, attempts=max_retries, original=last_exc) from last_exc


def _invoke_with_model_fallback(prompt, payload: dict[str, str], operation: str):
    candidates = get_chat_models()
    if not candidates:
        return None

    last_connection_failure: LLMConnectionFailure | None = None
    last_non_connection_error: Exception | None = None

    for config, model in candidates:
        try:
            return _invoke_with_retry(prompt | model, payload, operation=f"{operation} [{config['provider']}:{config['model']}]")
        except LLMConnectionFailure as exc:
            last_connection_failure = exc
            continue
        except Exception as exc:
            last_non_connection_error = exc
            if len(candidates) > 1 and config.get("provider") == "openai":
                continue
            raise

    if last_connection_failure is not None:
        raise last_connection_failure
    if last_non_connection_error is not None:
        raise last_non_connection_error
    return None


_REVIEWER_ENABLED_AGENTS = {"sinopse_abstract", "gramatica_ortografia"}
_build_graph = _default_build_graph


def _review_comments_with_llm(
    comments,
    agent: str,
    question: str,
    excerpt: str,
    profile_key: str | None,
):
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
        (item.get("paragraph_index"), str(item.get("issue_excerpt") or "").strip(), str(item.get("suggested_fix") or "").strip()): item
        for item in reviews
    }

    approved = []
    rejected = 0
    for comment in comments:
        review = verdict_by_key.get((comment.paragraph_index, comment.issue_excerpt, comment.suggested_fix))
        if review and review.get("decision") == "reject":
            rejected += 1
            continue
        approved.append(comment)

    return approved, f"{status} | revisor: {len(approved)} aprovados, {rejected} rejeitados"


def _reference_entry_texts(chunks: list[str], refs: list[str]) -> list[str]:
    return [
        chunk
        for chunk, ref in zip(chunks, refs)
        if "tipo=reference_entry" in (ref or "") and (chunk or "").strip()
    ]


def _reference_insertion_index(refs: list[str]) -> int | None:
    last_entry = next(
        (idx for idx in range(len(refs) - 1, -1, -1) if "tipo=reference_entry" in (refs[idx] or "")),
        None,
    )
    if last_entry is not None:
        return last_entry
    return next((idx for idx, ref in enumerate(refs) if "tipo=reference_heading" in (ref or "")), None)


def _build_user_reference_excerpt(
    request: ReferenceSearchRequest,
    candidates_json: str,
    refs: list[str],
    chunks: list[str],
) -> str:
    ref_lines = [
        f"[{idx}] {chunk}"
        for idx, (chunk, ref) in enumerate(zip(chunks, refs))
        if "tipo=reference_entry" in (ref or "")
    ]
    summarized_refs = "\n".join(ref_lines[:25]) if ref_lines else "(lista final vazia ou não identificada)"
    anchor = request.anchor_excerpt.strip() or request.paragraph_text.strip()
    return (
        "SOLICITAÇÃO DO USUÁRIO/EDITOR:\n"
        f"- Comentário original: {request.comment_text.strip()}\n"
        f"- Índice global do trecho: [{request.paragraph_index}]\n"
        f"- Trecho âncora: {anchor}\n"
        f"- Parágrafo completo: {request.paragraph_text.strip()}\n"
        f"- Consulta de busca usada: {request.query_text.strip()}\n\n"
        "REFERÊNCIAS JÁ PRESENTES NA LISTA FINAL:\n"
        f"{summarized_refs}\n\n"
        "CANDIDATOS LOCALIZADOS NA INTERNET:\n"
        f"{candidates_json}"
    )


def _accept_user_reference_comment(base_comment: AgentComment, request: ReferenceSearchRequest, refs: list[str]) -> AgentComment | None:
    insertion_index = _reference_insertion_index(refs)
    if insertion_index is None:
        return None
    suggested_fix = (base_comment.suggested_fix or "").strip()
    anchor_excerpt = (request.anchor_excerpt or request.paragraph_text or "").strip()
    if not suggested_fix or not anchor_excerpt:
        return None
    message = (base_comment.message or "").strip()
    if "comentário do usuário" not in _normalized_text(message):
        message = (message + " " if message else "") + "Referência localizada e anexada por solicitação registrada em comentário do usuário."
    return AgentComment(
        agent=USER_REFERENCE_AGENT,
        category="user_reference_request",
        message=message.strip(),
        paragraph_index=request.paragraph_index,
        issue_excerpt=anchor_excerpt,
        suggested_fix=suggested_fix,
        auto_apply=True,
        format_spec=(
            "action=insert_reference;"
            f"insert_after={insertion_index};"
            f"source_comment_id={request.comment_id}"
        ),
    )


def _run_user_reference_agent(
    prepared_document,
    question: str,
    profile_key: str,
    existing_comments,
    on_agent_done=None,
    on_agent_progress=None,
    on_agent_batch_status=None,
):
    requests = build_reference_search_requests(prepared_document.user_comments)
    if not requests:
        if on_agent_batch_status is not None:
            on_agent_batch_status(USER_REFERENCE_AGENT, 1, 1, "sem solicitações explícitas em comentários do usuário")
        if on_agent_progress is not None:
            on_agent_progress(USER_REFERENCE_AGENT, 1, 1, 0, len(existing_comments))
        if on_agent_done is not None:
            on_agent_done(USER_REFERENCE_AGENT, 0, len(existing_comments))
        return [], []

    prompt = build_agent_prompt(USER_REFERENCE_AGENT, profile_key=profile_key)
    accepted: list[AgentComment] = []
    decisions: list[VerificationDecision] = []
    reference_entries = _reference_entry_texts(prepared_document.chunks, prepared_document.refs)

    for batch_idx, request in enumerate(requests, start=1):
        candidates = search_reference_candidates(request)
        candidates = [
            candidate
            for candidate in candidates
            if not reference_already_present(format_reference_candidate(candidate), reference_entries)
        ]
        if not candidates:
            continue

        if get_chat_model() is None:
            continue

        excerpt = _build_user_reference_excerpt(
            request=request,
            candidates_json=candidates_as_json(candidates),
            refs=prepared_document.refs,
            chunks=prepared_document.chunks,
        )
        payload = {
            "question": _sanitize_for_llm(question),
            "document_excerpt": _sanitize_for_llm(excerpt),
        }
        try:
            response = _invoke_with_model_fallback(prompt, payload, operation=f"agente {USER_REFERENCE_AGENT}")
            if response is None:
                raise RuntimeError("modelo indisponível")
            raw = response.content if isinstance(response.content, str) else str(response.content)
            parsed_items, status = _parse_comments(raw, agent=USER_REFERENCE_AGENT), "json direto"
        except Exception as exc:
            status = f"falha no agente: {exc}"
            parsed_items = []

        added_count = 0
        existing_keys = {_normalized_text(item.issue_excerpt + item.suggested_fix) for item in [*existing_comments, *accepted]}
        for item in parsed_items:
            candidate_comment = _accept_user_reference_comment(item, request=request, refs=prepared_document.refs)
            if candidate_comment is None:
                continue
            if reference_already_present(candidate_comment.suggested_fix, reference_entries):
                continue
            dedupe_key = _normalized_text(candidate_comment.issue_excerpt + candidate_comment.suggested_fix)
            if dedupe_key in existing_keys:
                continue
            accepted.append(candidate_comment)
            existing_keys.add(dedupe_key)
            decisions.append(
                VerificationDecision(
                    comment=candidate_comment,
                    accepted=True,
                    reason="referência localizada a partir de comentário do usuário",
                    source="user_comment_agent",
                    batch_index=batch_idx,
                )
            )
            reference_entries.append(candidate_comment.suggested_fix)
            added_count += 1

        if on_agent_batch_status is not None:
            on_agent_batch_status(USER_REFERENCE_AGENT, batch_idx, len(requests), status)
        if on_agent_progress is not None:
            on_agent_progress(USER_REFERENCE_AGENT, batch_idx, len(requests), added_count, len(existing_comments) + len(accepted))
        if on_agent_done is not None:
            on_agent_done(USER_REFERENCE_AGENT, added_count, len(existing_comments) + len(accepted))

    return accepted, decisions


def run_prepared_review(
    prepared_document,
    question: str,
    selected_agents=None,
    on_agent_done=None,
    on_agent_progress=None,
    on_agent_batch_status=None,
    profile_key: str = "GENERIC",
):
    agent_order = [agent for agent in (selected_agents or list(prepared_document.agent_batches.keys())) if agent in prepared_document.agent_batches]
    if not prepared_document.chunks:
        return ConversationResult(answer="Documento vazio ou sem texto extraído.", comments=[])

    agent_apps = {agent: _build_graph([agent], include_coordinator=False) for agent in agent_order}
    final_comments = []
    verification_decisions = []
    running_summaries = {agent: "" for agent in agent_order}
    interrupted_reason = ""
    interrupted_agent = ""

    for agent in agent_order:
        if agent == USER_REFERENCE_AGENT:
            added_comments, agent_decisions = _run_user_reference_agent(
                prepared_document=prepared_document,
                question=question,
                profile_key=profile_key,
                existing_comments=final_comments,
                on_agent_done=on_agent_done,
                on_agent_progress=on_agent_progress,
                on_agent_batch_status=on_agent_batch_status,
            )
            final_comments.extend(added_comments)
            verification_decisions.extend(agent_decisions)
            running_summaries[agent] = _truncate_progressive_summary(
                _comment_memory_lines(added_comments) if added_comments else "(nenhuma referência adicional foi inserida a partir de comentários do usuário)"
            )
            continue

        batches = prepared_document.agent_batches.get(agent, [])
        if not batches:
            continue

        stop_processing = False
        for batch_idx, batch in enumerate(batches, start=1):
            excerpt = _build_batch_review_excerpt(prepared=prepared_document, batch=batch, running_summary=running_summaries.get(agent, ""))
            comments_before_batch = len(final_comments)
            accepted_in_batch = []
            initial_state = {
                "question": question,
                "document_excerpt": excerpt,
                "running_summary": running_summaries.get(agent, ""),
                "profile_key": profile_key,
                "comments": final_comments,
                "answer": "",
            }

            for update in agent_apps[agent].stream(initial_state, stream_mode="updates"):
                if not update:
                    continue
                node, payload = next(iter(update.items()))
                if not isinstance(payload, dict) or node != agent:
                    continue
                current_comments = payload.get("comments", final_comments)
                if isinstance(current_comments, list):
                    old_comments = current_comments[:comments_before_batch]
                    batch_comments = current_comments[comments_before_batch:]
                    verified_comments, batch_decisions = _verify_batch_comments(
                        comments=batch_comments,
                        agent=agent,
                        batch_indexes=batch.indexes,
                        chunks=prepared_document.chunks,
                        refs=prepared_document.refs,
                        existing_comments=old_comments,
                        batch_index=batch_idx,
                    )
                    final_comments = [*old_comments, *verified_comments]
                    accepted_in_batch = verified_comments
                    verification_decisions.extend(batch_decisions)
                else:
                    batch_decisions = []

                batch_status = _format_batch_status(str(payload.get("batch_status", "") or ""), batch_decisions)
                total = len(final_comments)
                new_count = sum(1 for decision in batch_decisions if decision.accepted)
                if on_agent_done is not None:
                    on_agent_done(agent, new_count, total)
                if on_agent_batch_status is not None:
                    on_agent_batch_status(agent, batch_idx, len(batches), batch_status)
                if on_agent_progress is not None:
                    on_agent_progress(agent, batch_idx, len(batches), new_count, total)
                if "falha de conexao da llm" in _folded_text(batch_status):
                    interrupted_reason = batch_status
                    interrupted_agent = agent
                    stop_processing = True
                    break

            if stop_processing:
                break

            running_summaries[agent] = _update_running_summary(
                agent=agent,
                question=question,
                running_summary=running_summaries.get(agent, ""),
                batch=batch,
                accepted_comments=accepted_in_batch,
            )

        if stop_processing:
            break

    consolidated_comments = _consolidate_final_comments(final_comments, prepared_document.refs)

    if interrupted_reason:
        final_answer = _partial_answer_from_comments(
            consolidated_comments,
            "Resumo parcial (execução interrompida por falha de conexão da LLM"
            + (f" no agente {interrupted_agent}" if interrupted_agent else "")
            + f": {interrupted_reason}).",
        )
    else:
        final_answer = coordinate_answer(question=question, comments=consolidated_comments)

    return ConversationResult(
        answer=final_answer,
        comments=consolidated_comments,
        verification=_summarize_verification(verification_decisions),
    )


def run_conversation(
    paragraphs,
    refs,
    sections,
    question: str,
    selected_agents=None,
    user_comments=None,
    on_agent_done=None,
    on_agent_progress=None,
    on_agent_batch_status=None,
    profile_key: str = "GENERIC",
):
    if not paragraphs:
        from .models import ConversationResult

        return ConversationResult(answer="Documento vazio ou sem texto extraído.", comments=[])

    prepare_kwargs = {
        "paragraphs": paragraphs,
        "refs": refs,
        "sections": sections,
        "selected_agents": selected_agents,
    }
    if user_comments is not None:
        prepare_kwargs["user_comments"] = user_comments
    prepared_document = prepare_review_batches(**prepare_kwargs)
    return run_prepared_review(
        prepared_document=prepared_document,
        question=question,
        selected_agents=selected_agents,
        on_agent_done=on_agent_done,
        on_agent_progress=on_agent_progress,
        on_agent_batch_status=on_agent_batch_status,
        profile_key=profile_key,
    )


__all__ = [
    "_agent_scope_indexes",
    "_connection_error_summary",
    "_find_reference_citation_indexes",
    "_heuristic_reference_comments",
    "_heuristic_reference_global_comments",
    "_invoke_with_model_fallback",
    "_invoke_with_retry",
    "_is_connection_error",
    "_normalize_batch_comments",
    "_parse_comment_reviews",
    "_parse_comments",
    "_review_comments_with_llm",
    "_run_user_reference_agent",
    "_verify_batch_comments",
    "get_chat_model",
    "get_chat_models",
    "get_llm_retry_config",
    "prepare_review_batches",
    "run_conversation",
    "run_prepared_review",
]

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from .context_selector import build_excerpt, select_chunk_indexes
from .document_loader import Section
from .llm import get_chat_model
from .models import AgentComment, ConversationResult
from .prompts import AGENT_ORDER, build_agent_prompt, build_coordinator_prompt


class ChatState(TypedDict):
    question: str
    document_excerpt: str
    comments: list[AgentComment]
    answer: str


def _serialize_comments(comments: list[AgentComment]) -> str:
    return json.dumps(
        [
            {
                "agent": c.agent,
                "category": c.category,
                "message": c.message,
                "paragraph_index": c.paragraph_index,
                "issue_excerpt": c.issue_excerpt,
                "suggested_fix": c.suggested_fix,
            }
            for c in comments
        ],
        ensure_ascii=False,
        indent=2,
    )


def _parse_comments(raw: str, agent: str) -> list[AgentComment]:
    try:
        parsed = json.loads(raw)
    except Exception:
        return []

    if not isinstance(parsed, list):
        return []

    out: list[AgentComment] = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        message = str(item.get("message", "")).strip()
        if not message:
            continue
        category = str(item.get("category", agent)).strip() or agent
        paragraph_index = item.get("paragraph_index")
        if isinstance(paragraph_index, str) and paragraph_index.isdigit():
            paragraph_index = int(paragraph_index)
        if not isinstance(paragraph_index, int):
            paragraph_index = None
        issue_excerpt = str(item.get("issue_excerpt", "")).strip()
        suggested_fix = str(item.get("suggested_fix", "")).strip()
        out.append(
            AgentComment(
                agent=agent,
                category=category,
                message=message,
                paragraph_index=paragraph_index,
                issue_excerpt=issue_excerpt,
                suggested_fix=suggested_fix,
            )
        )
    return out


def _agent_node(agent: str):
    def run(state: ChatState) -> ChatState:
        model = get_chat_model()
        if model is None:
            return {"comments": state.get("comments", [])}

        prompt = build_agent_prompt(agent)
        response = (prompt | model).invoke(
            {
                "question": state["question"],
                "document_excerpt": state["document_excerpt"],
            }
        )
        raw = response.content if isinstance(response.content, str) else str(response.content)
        items = _parse_comments(raw, agent=agent)
        merged = [*state.get("comments", []), *items]
        return {"comments": merged}

    return run


def _coordinator_node(state: ChatState) -> ChatState:
    model = get_chat_model()
    comments = state.get("comments", [])
    if model is None:
        if comments:
            points = "\n".join(f"- [{c.agent}] {c.message}" for c in comments[:8])
            answer = "Resumo dos agentes:\n" + points
        else:
            answer = "Não foi possível consultar a LLM. Configure OPENAI_API_KEY no .env."
        return {"answer": answer}

    prompt = build_coordinator_prompt()
    response = (prompt | model).invoke(
        {
            "question": state["question"],
            "document_excerpt": state["document_excerpt"],
            "comments_json": _serialize_comments(comments),
        }
    )
    answer = response.content if isinstance(response.content, str) else str(response.content)
    return {"answer": answer}


def _build_graph(agent_order: list[str]):
    graph = StateGraph(ChatState)

    for agent in agent_order:
        graph.add_node(agent, _agent_node(agent))

    graph.add_node("coordenador", _coordinator_node)

    if not agent_order:
        graph.add_edge(START, "coordenador")
    else:
        graph.add_edge(START, agent_order[0])
        for idx in range(len(agent_order) - 1):
            graph.add_edge(agent_order[idx], agent_order[idx + 1])
        graph.add_edge(agent_order[-1], "coordenador")

    graph.add_edge("coordenador", END)
    return graph.compile()


def run_conversation(
    paragraphs: list[str],
    refs: list[str],
    sections: list[Section],
    question: str,
    selected_agents: list[str] | None = None,
    on_agent_done: Callable[[str, int, int], None] | None = None,
) -> ConversationResult:
    agent_order = [a for a in (selected_agents or AGENT_ORDER) if a in AGENT_ORDER]
    indexes = select_chunk_indexes(question=question, chunks=paragraphs, sections=sections)
    excerpt = build_excerpt(indexes=indexes, chunks=paragraphs, refs=refs)

    app = _build_graph(agent_order)
    initial_state: ChatState = {
        "question": question,
        "document_excerpt": excerpt,
        "comments": [],
        "answer": "",
    }

    final_comments: list[AgentComment] = []
    final_answer = ""
    previous_count = 0

    for update in app.stream(initial_state, stream_mode="updates"):
        if not update:
            continue
        node, payload = next(iter(update.items()))
        if not isinstance(payload, dict):
            continue

        if node in agent_order:
            current_comments = payload.get("comments", final_comments)
            if isinstance(current_comments, list):
                final_comments = current_comments
            total = len(final_comments)
            new_count = max(total - previous_count, 0)
            previous_count = total
            if on_agent_done is not None:
                on_agent_done(node, new_count, total)

        if node == "coordenador":
            final_answer = str(payload.get("answer", final_answer))

    return ConversationResult(answer=final_answer, comments=final_comments)

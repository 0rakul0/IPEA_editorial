from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from editorial_docx.docx_utils import extract_docx_user_comments
from editorial_docx.document_loader import load_document
from editorial_docx.graph_chat import run_conversation
from editorial_docx.models import AgentComment, DocumentUserComment
from editorial_docx.prompts import AGENT_ORDER, detect_prompt_profile, load_agent_instruction
from editorial_docx.prompts.prompt import PROMPT_FILES
from editorial_docx.review_patterns import _folded_text, _normalized_text


MARKED_NAME_TOKENS = ("(com marcas)", "(comentado)")


@dataclass(slots=True)
class HoldoutPair:
    directory: str
    original: str
    marked: str | None


def discover_holdout_pairs(root: Path) -> list[HoldoutPair]:
    """Finds the original and editorial-commented documents in every holdout folder."""
    pairs: list[HoldoutPair] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        documents = sorted(directory.glob("*.docx"))
        original = next((path for path in documents if "(original)" in path.name.casefold()), None)
        marked = next(
            (path for path in documents if any(token in path.name.casefold() for token in MARKED_NAME_TOKENS)),
            None,
        )
        if original is not None:
            pairs.append(HoldoutPair(str(directory), str(original), str(marked) if marked else None))
    return pairs


def _normalized(value: str) -> str:
    return _normalized_text(value or "")


def _best_original_index(comment: DocumentUserComment, original_chunks: list[str]) -> int | None:
    """Maps a human Word comment to the matching block of the original file."""
    candidates = [comment.anchor_excerpt, comment.paragraph_text]
    normalized_chunks = [_normalized(chunk) for chunk in original_chunks]

    for candidate in candidates:
        needle = _normalized(candidate)
        if len(needle) < 8:
            continue
        exact = [idx for idx, chunk in enumerate(normalized_chunks) if needle in chunk or chunk in needle]
        if len(exact) == 1:
            return exact[0]

    best_index: int | None = None
    best_score = 0.0
    for candidate in candidates:
        needle = _normalized(candidate)
        if len(needle) < 20:
            continue
        for idx, chunk in enumerate(normalized_chunks):
            score = SequenceMatcher(None, needle, chunk).ratio()
            if score > best_score:
                best_score = score
                best_index = idx
    return best_index if best_score >= 0.72 else None


def _instruction_metrics() -> dict[str, object]:
    """Counts the effective instruction text for every available review agent."""
    agents: dict[str, dict[str, int]] = {}
    for agent_name in sorted(name for name in PROMPT_FILES if name != "coordenador"):
        instruction = load_agent_instruction(agent_name, profile_key="GENERIC")
        nonblank_lines = [line for line in instruction.splitlines() if line.strip()]
        agents[agent_name] = {
            "linhas_de_instrucao": len(nonblank_lines),
            "palavras_de_instrucao": len(instruction.split()),
            "ativo_por_padrao": int(agent_name in AGENT_ORDER),
        }
    return {
        "definicao": "Instrução é medida como linha não vazia do prompt efetivamente carregado; palavras são apresentadas como medida complementar de extensão.",
        "agentes_disponiveis": len(agents),
        "agentes_ativos_por_padrao": sum(item["ativo_por_padrao"] for item in agents.values()),
        "total_linhas_de_instrucao": sum(item["linhas_de_instrucao"] for item in agents.values()),
        "total_palavras_de_instrucao": sum(item["palavras_de_instrucao"] for item in agents.values()),
        "por_agente": agents,
    }


def _serialize_comment(comment: AgentComment) -> dict[str, object]:
    return {
        "agent": comment.agent,
        "category": comment.category,
        "paragraph_index": comment.paragraph_index,
        "issue_excerpt": comment.issue_excerpt,
        "message": comment.message,
        "suggested_fix": comment.suggested_fix,
        "action_type": comment.action_type,
    }


def _compare_comments(
    human_comments: list[DocumentUserComment],
    system_comments: list[AgentComment],
    original_chunks: list[str],
) -> dict[str, object]:
    human_by_block: dict[int, list[DocumentUserComment]] = {}
    unmapped_human: list[DocumentUserComment] = []
    for comment in human_comments:
        index = _best_original_index(comment, original_chunks)
        if index is None:
            unmapped_human.append(comment)
        else:
            human_by_block.setdefault(index, []).append(comment)

    system_by_block: dict[int, list[AgentComment]] = {}
    for comment in system_comments:
        if isinstance(comment.paragraph_index, int) and 0 <= comment.paragraph_index < len(original_chunks):
            system_by_block.setdefault(comment.paragraph_index, []).append(comment)

    human_blocks = set(human_by_block)
    system_blocks = set(system_by_block)
    both = sorted(human_blocks & system_blocks)
    human_only = sorted(human_blocks - system_blocks)
    system_only = sorted(system_blocks - human_blocks)

    def block_rows(indexes: list[int]) -> list[dict[str, object]]:
        return [
            {
                "paragraph_index": index,
                "excerpt": original_chunks[index][:500],
                "comentarios_humanos": [asdict(item) for item in human_by_block.get(index, [])],
                "comentarios_ia": [_serialize_comment(item) for item in system_by_block.get(index, [])],
            }
            for index in indexes
        ]

    human_count = len(human_blocks)
    system_count = len(system_blocks)
    return {
        "metricas_por_bloco": {
            "comentarios_humanos": len(human_comments),
            "comentarios_ia": len(system_comments),
            "comentarios_humanos_mapeados": sum(len(items) for items in human_by_block.values()),
            "comentarios_humanos_sem_mapeamento": len(unmapped_human),
            "blocos_com_comentario_humano": human_count,
            "blocos_com_comentario_ia": system_count,
            "blocos_em_comum": len(both),
            "blocos_apenas_humano": len(human_only),
            "blocos_apenas_ia": len(system_only),
            "cobertura_ia_sobre_blocos_humanos": len(both) / human_count if human_count else None,
            "concordancia_de_blocos_ia": len(both) / system_count if system_count else None,
        },
        "blocos_em_comum": block_rows(both),
        "blocos_apenas_humano": block_rows(human_only),
        "blocos_apenas_ia": block_rows(system_only),
        "comentarios_humanos_sem_mapeamento": [asdict(item) for item in unmapped_human],
    }


def _run_system_review(original: Path) -> list[AgentComment]:
    loaded = load_document(original)
    result = run_conversation(
        paragraphs=loaded.chunks,
        refs=loaded.refs,
        sections=loaded.sections,
        question="Faça uma revisão editorial completa, com comentários locais, objetivos e acionáveis.",
        selected_agents=AGENT_ORDER.copy(),
        user_comments=loaded.user_comments,
        profile_key=detect_prompt_profile(original.name).key,
    )
    return result.comments


def _aggregate_metrics(documents: list[dict[str, object]]) -> dict[str, int | float | None]:
    """Recalculates aggregate block metrics from document-level integer counts."""
    aggregate = Counter()
    for item in documents:
        comparison = item.get("comparison") if isinstance(item, dict) else None
        if not isinstance(comparison, dict):
            continue
        metrics = comparison.get("metricas_por_bloco")
        if isinstance(metrics, dict):
            aggregate.update({key: value for key, value in metrics.items() if isinstance(value, int)})
    result: dict[str, int | float | None] = dict(aggregate)
    human_blocks = int(result.get("blocos_com_comentario_humano", 0))
    system_blocks = int(result.get("blocos_com_comentario_ia", 0))
    common_blocks = int(result.get("blocos_em_comum", 0))
    result["cobertura_ia_sobre_blocos_humanos"] = common_blocks / human_blocks if human_blocks else None
    result["concordancia_de_blocos_ia"] = common_blocks / system_blocks if system_blocks else None
    return result


def _parse_json_object(raw: str) -> dict[str, object]:
    """Reads the JSON object returned by the semantic comparison call."""
    cleaned = (raw or "").strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {"matches": []}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"matches": []}
    return payload if isinstance(payload, dict) else {"matches": []}


def _semantic_matches_for_block(
    human_comments: list[dict[str, object]],
    system_comments: list[dict[str, object]],
    model: ChatOpenAI,
) -> list[dict[str, object]]:
    """Uses a strict judge to match only semantically equivalent comments in one block."""
    exact_matches: list[dict[str, object]] = []
    used_human: set[int] = set()
    used_ai: set[int] = set()
    for human_index, human in enumerate(human_comments):
        human_text = _normalized(str(human.get("text") or ""))
        if not human_text:
            continue
        for ai_index, system in enumerate(system_comments):
            if ai_index in used_ai:
                continue
            suggested_fix = _normalized(str(system.get("suggested_fix") or ""))
            if len(suggested_fix) >= 5 and suggested_fix in human_text:
                used_human.add(human_index)
                used_ai.add(ai_index)
                exact_matches.append(
                    {
                        "human_index": human_index,
                        "ai_index": ai_index,
                        "same_issue": True,
                        "same_correction": True,
                        "reason": "A correção sugerida pela IA aparece literalmente no comentário de referência.",
                    }
                )
                break

    prompt = """Compare comentários editoriais de referência e comentários de IA ancorados no mesmo bloco de um documento.
Seu trabalho é identificar apenas pares que tratem substancialmente do MESMO problema editorial. Estar no mesmo parágrafo não basta.

Para cada par, avalie:
- same_issue: o diagnóstico atinge o mesmo erro, omissão ou inconsistência concreta;
- same_correction: além do mesmo problema, a correção ou solicitação proposta é materialmente equivalente.

Não considere como equivalentes comentários genéricos, problemas diferentes no mesmo trecho, ou sugestões que inventem outro erro. Cada comentário pode aparecer em no máximo um par; escolha o melhor pareamento. Responda somente JSON válido:
{"matches":[{"human_index":0,"ai_index":0,"same_issue":true,"same_correction":false,"reason":"justificativa curta"}]}.

COMENTÁRIOS DE REFERÊNCIA:
""" + json.dumps(human_comments, ensure_ascii=False) + "\nCOMENTÁRIOS DA IA:\n" + json.dumps(system_comments, ensure_ascii=False)
    response = model.invoke(prompt)
    payload = _parse_json_object(str(response.content))
    valid = list(exact_matches)
    for item in payload.get("matches", []):
        if not isinstance(item, dict) or not item.get("same_issue"):
            continue
        human_index = item.get("human_index")
        ai_index = item.get("ai_index")
        if not isinstance(human_index, int) or not isinstance(ai_index, int):
            continue
        if not (0 <= human_index < len(human_comments) and 0 <= ai_index < len(system_comments)):
            continue
        if human_index in used_human or ai_index in used_ai:
            continue
        reason = str(item.get("reason") or "").strip()
        reason_folded = _folded_text(reason)
        if "questao diferente" in reason_folded or "problema diferente" in reason_folded:
            continue
        used_human.add(human_index)
        used_ai.add(ai_index)
        valid.append(
            {
                "human_index": human_index,
                "ai_index": ai_index,
                "same_issue": True,
                "same_correction": bool(item.get("same_correction")),
                "reason": reason,
            }
        )
    return valid


def _semantic_evaluate_existing(result: dict[str, object], model_name: str) -> dict[str, object]:
    """Adds semantic comment matching to a completed block-level report."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("OPENAI_API_KEY não configurada para a avaliação semântica.")
    model = ChatOpenAI(model=model_name, api_key=api_key, temperature=0, timeout=120, max_retries=2, seed=7)
    totals = Counter()
    document_results: list[dict[str, object]] = []
    for document in result.get("documents", []):
        if not isinstance(document, dict) or document.get("status") != "comparado":
            continue
        comparison = document.get("comparison", {})
        if not isinstance(comparison, dict):
            continue
        semantic_blocks = []
        mapped_human = 0
        all_system = int(comparison.get("metricas_por_bloco", {}).get("comentarios_ia", 0))
        for block in comparison.get("blocos_em_comum", []):
            if not isinstance(block, dict):
                continue
            human = list(block.get("comentarios_humanos", []))
            system = list(block.get("comentarios_ia", []))
            mapped_human += len(human)
            matches = _semantic_matches_for_block(human, system, model)
            semantic_blocks.append({"paragraph_index": block.get("paragraph_index"), "matches": matches})
            totals["same_issue_pairs"] += len(matches)
            totals["same_correction_pairs"] += sum(bool(item["same_correction"]) for item in matches)
        metrics = comparison.get("metricas_por_bloco", {})
        mapped_human_total = int(metrics.get("comentarios_humanos_mapeados", 0))
        document_results.append(
            {
                "original": document.get("pair", {}).get("original", ""),
                "human_comments_mapped": mapped_human_total,
                "ai_comments": all_system,
                "matches": semantic_blocks,
            }
        )
        totals["human_comments_mapped"] += mapped_human_total
        totals["ai_comments"] += all_system

    same_issue = int(totals["same_issue_pairs"])
    same_correction = int(totals["same_correction_pairs"])
    human_total = int(totals["human_comments_mapped"])
    ai_total = int(totals["ai_comments"])
    return {
        "model_judge": model_name,
        "method": "Julgamento semântico estrito somente entre comentários ancorados no mesmo bloco; pares exclusivos. Comentários sem par não são classificados como incorretos.",
        "same_issue_pairs": same_issue,
        "same_correction_pairs": same_correction,
        "human_comments_mapped": human_total,
        "ai_comments": ai_total,
        "semantic_coverage_on_human_comments": same_issue / human_total if human_total else None,
        "semantic_alignment_on_ai_comments": same_issue / ai_total if ai_total else None,
        "same_correction_rate_on_human_comments": same_correction / human_total if human_total else None,
        "documents": document_results,
    }


def _reference_origin(comments: list[dict[str, object]]) -> str:
    authors = [str(item.get("author") or "").strip() for item in comments]
    generated = [author for author in authors if author.casefold().startswith("revisão:") or author.casefold().startswith("revisao:")]
    if not authors:
        return "sem_referencia"
    if len(generated) == len(authors):
        return "rodada_anterior_ia"
    if generated:
        return "misto"
    return "editorial_humano"


def _build_adjudication_dataset(result: dict[str, object]) -> dict[str, object]:
    """Flattens the comparison into rows that editorial reviewers can label."""
    rows: list[dict[str, object]] = []
    for document in result.get("documents", []):
        if not isinstance(document, dict) or document.get("status") != "comparado":
            continue
        original = str(document.get("pair", {}).get("original") or "")
        comparison = document.get("comparison", {})
        if not isinstance(comparison, dict):
            continue
        for group in ("blocos_em_comum", "blocos_apenas_humano", "blocos_apenas_ia"):
            for block in comparison.get(group, []):
                human = list(block.get("comentarios_humanos", []))
                system = list(block.get("comentarios_ia", []))
                if not system:
                    rows.append(
                        {
                            "documento": original,
                            "paragraph_index": block.get("paragraph_index"),
                            "excerpt": block.get("excerpt", ""),
                            "tipo": "faltante_ia",
                            "origem_referencia": _reference_origin(human),
                            "comentarios_referencia": human,
                            "comentario_ia": None,
                            "rotulo_humano": "",
                            "nota_humana": "",
                        }
                    )
                for comment in system:
                    rows.append(
                        {
                            "documento": original,
                            "paragraph_index": block.get("paragraph_index"),
                            "excerpt": block.get("excerpt", ""),
                            "tipo": "comentario_ia",
                            "origem_referencia": _reference_origin(human),
                            "comentarios_referencia": human,
                            "comentario_ia": comment,
                            "rotulo_humano": "",
                            "nota_humana": "",
                        }
                    )
    return {
        "labels_allowed": ["correto", "parcial", "incorreto", "nao_verificavel"],
        "instruction": "Rotular após ler o trecho, o comentário da IA e os comentários de referência. `rodada_anterior_ia` é regressão, não evidência humana independente.",
        "rows": rows,
    }


def _is_editorial_human_author(author: object) -> bool:
    value = str(author or "").strip().casefold()
    return bool(value) and not (value.startswith("revis") and ":" in value)


def _block_map(document: dict[str, object]) -> dict[int, dict[str, object]]:
    comparison = document.get("comparison", {})
    if not isinstance(comparison, dict):
        return {}
    result: dict[int, dict[str, object]] = {}
    for group in ("blocos_em_comum", "blocos_apenas_humano", "blocos_apenas_ia"):
        for block in comparison.get(group, []):
            if not isinstance(block, dict) or not isinstance(block.get("paragraph_index"), int):
                continue
            result[block["paragraph_index"]] = block
    return result


def _build_human_reference_adjudication(
    gpt4o_result: dict[str, object], gpt51_result: dict[str, object]
) -> dict[str, object]:
    """Builds a human-only reference queue with the two model outputs side by side."""
    gpt4o_documents = {
        str(document.get("pair", {}).get("original") or ""): document
        for document in gpt4o_result.get("documents", [])
        if isinstance(document, dict) and document.get("status") == "comparado"
    }
    rows: list[dict[str, object]] = []
    seen: set[tuple[str, int, int]] = set()

    for document in gpt51_result.get("documents", []):
        if not isinstance(document, dict) or document.get("status") != "comparado":
            continue
        original = str(document.get("pair", {}).get("original") or "")
        gpt51_blocks = _block_map(document)
        gpt4o_blocks = _block_map(gpt4o_documents.get(original, {}))
        for paragraph_index, block in gpt51_blocks.items():
            human_comments = [
                item
                for item in block.get("comentarios_humanos", [])
                if isinstance(item, dict) and _is_editorial_human_author(item.get("author"))
            ]
            if not human_comments:
                continue
            mini_comments = gpt4o_blocks.get(paragraph_index, {}).get("comentarios_ia", [])
            for human in human_comments:
                comment_id = int(human.get("comment_id") or -1)
                key = (original, paragraph_index, comment_id)
                if key in seen:
                    continue
                seen.add(key)
                rows.append(
                    {
                        "documento": original,
                        "paragraph_index": paragraph_index,
                        "excerpt": block.get("excerpt", ""),
                        "comentario_editorial_humano": human,
                        "comentarios_gpt4o_mini": mini_comments,
                        "comentarios_gpt51": block.get("comentarios_ia", []),
                        "rotulo_gpt4o_mini": "",
                        "rotulo_gpt51": "",
                        "nota_do_revisor": "",
                    }
                )
    return {
        "labels_allowed": ["correto", "parcial", "incorreto", "nao_verificavel"],
        "instruction": "Rotule cada saída do modelo contra o comentário editorial humano e o trecho. Não use comentários cujo autor começa com 'Revisão:' como gabarito humano.",
        "summary": {
            "human_reference_occurrences": len(rows),
            "unique_human_comment_ids": len({int(row["comentario_editorial_humano"].get("comment_id") or -1) for row in rows}),
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compara o ipeaREV aos comentários humanos do conjunto holdout.")
    parser.add_argument("holdout_root", type=Path, help="Diretório que contém as nove pastas de teste.")
    parser.add_argument("--output", type=Path, required=True, help="Arquivo JSON de resultados.")
    parser.add_argument("--model", default="gpt-4o-mini", help="Modelo OpenAI a registrar e usar na rodada.")
    parser.add_argument("--dry-run", action="store_true", help="Gera inventário e métricas de instruções sem chamar a LLM.")
    parser.add_argument("--recalculate-existing", action="store_true", help="Corrige apenas os agregados de um relatório já existente, sem chamar a LLM.")
    parser.add_argument("--semantic-evaluate-existing", action="store_true", help="Acrescenta pareamento semântico a um relatório já existente.")
    parser.add_argument("--adjudication-output", type=Path, help="Gera dataset de adjudicação a partir de um relatório existente, sem chamar a LLM.")
    parser.add_argument("--comparison-output", type=Path, help="Relatório do outro modelo, usado somente para a fila de adjudicação humana lado a lado.")
    parser.add_argument("--human-adjudication-output", type=Path, help="Gera fila de adjudicação apenas contra comentários editoriais humanos.")
    args = parser.parse_args()
    load_dotenv()

    if args.recalculate_existing:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        existing["aggregate_metrics"] = _aggregate_metrics(existing.get("documents", []))
        existing.pop("aggregate_raw_counts", None)
        args.output.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        print(args.output)
        return 0

    if args.semantic_evaluate_existing:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        existing["semantic_evaluation"] = _semantic_evaluate_existing(existing, args.model)
        args.output.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
        print(args.output)
        return 0

    if args.adjudication_output is not None:
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        args.adjudication_output.parent.mkdir(parents=True, exist_ok=True)
        args.adjudication_output.write_text(
            json.dumps(_build_adjudication_dataset(existing), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(args.adjudication_output)
        return 0

    if args.human_adjudication_output is not None:
        if args.comparison_output is None:
            raise SystemExit("--comparison-output é obrigatório com --human-adjudication-output.")
        gpt51_result = json.loads(args.output.read_text(encoding="utf-8"))
        gpt4o_result = json.loads(args.comparison_output.read_text(encoding="utf-8"))
        args.human_adjudication_output.parent.mkdir(parents=True, exist_ok=True)
        args.human_adjudication_output.write_text(
            json.dumps(_build_human_reference_adjudication(gpt4o_result, gpt51_result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(args.human_adjudication_output)
        return 0

    if not args.dry_run:
        if not os.getenv("OPENAI_API_KEY", "").strip():
            raise SystemExit("OPENAI_API_KEY não configurada. Defina-a antes da rodada com gpt-4o-mini.")
        os.environ["LLM_PRIMARY_PROVIDER"] = "openai"
        os.environ["LLM_PROVIDER"] = "openai"
        os.environ["OPENAI_MODEL"] = args.model

    pairs = discover_holdout_pairs(args.holdout_root)
    reports: list[dict[str, object]] = []
    for pair in pairs:
        item: dict[str, object] = {"pair": asdict(pair), "status": "sem_comparacao"}
        if pair.marked is None:
            item["reason"] = "Documento com marcas/comentado não disponível na pasta holdout."
        elif args.dry_run:
            item["status"] = "pronto_para_execucao"
            item["human_comments"] = len(extract_docx_user_comments(Path(pair.marked)))
        else:
            original = Path(pair.original)
            human = extract_docx_user_comments(Path(pair.marked))
            system = _run_system_review(original)
            comparison = _compare_comments(human, system, load_document(original).chunks)
            item.update({"status": "comparado", "comparison": comparison})
        reports.append(item)

    output = {
        "model": args.model,
        "dry_run": args.dry_run,
        "comparison_unit": "bloco/parágrafo do original; comentários no mesmo bloco contam como cobertura comum, sem presumir equivalência semântica.",
        "instruction_metrics": _instruction_metrics(),
        "documents": reports,
        "aggregate_metrics": _aggregate_metrics(reports),
        "coverage": {
            "total_originals": len(pairs),
            "with_marked_document": sum(pair.marked is not None for pair in pairs),
            "without_marked_document": sum(pair.marked is None for pair in pairs),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

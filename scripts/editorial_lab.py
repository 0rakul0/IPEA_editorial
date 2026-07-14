#!/usr/bin/env python3
"""Aprende com ciclos editoriais e avalia relatórios dos agentes."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import sys
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile
import xml.etree.ElementTree as ET

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
W = f"{{{W_NS}}}"


def folded(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value).strip().lower()


def normalized(value: str) -> str:
    return re.sub(r"[^\w]+", " ", folded(value), flags=re.UNICODE).strip()


def similarity(left: str, right: str) -> float:
    a, b = normalized(left), normalized(right)
    if not a or not b:
        return 0.0
    if a in b or b in a:
        return min(len(a), len(b)) / max(len(a), len(b))
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def paragraph_text(node: ET.Element) -> str:
    return "".join(item.text or "" for item in node.iter(f"{W}t")).strip()


def read_docx(path: Path) -> dict[str, object]:
    with ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        paragraphs = [
            text
            for node in document.iter(f"{W}p")
            if (text := paragraph_text(node))
        ]
        insertion_nodes = list(document.iter(f"{W}ins"))
        deletion_nodes = list(document.iter(f"{W}del"))
        insertions = [
            "".join(item.text or "" for item in node.iter(f"{W}t")).strip()
            for node in insertion_nodes
        ]
        deletions = [
            "".join(item.text or "" for item in node.iter(f"{W}delText")).strip()
            for node in deletion_nodes
        ]
        revision_authors = Counter(
            node.attrib.get(f"{W}author", "").strip()
            for node in [*insertion_nodes, *deletion_nodes]
            if node.attrib.get(f"{W}author", "").strip()
        )
        comments: list[dict[str, object]] = []
        if "word/comments.xml" in archive.namelist():
            comments_root = ET.fromstring(archive.read("word/comments.xml"))
            comment_by_id: dict[str, dict[str, str]] = {}
            for node in comments_root.iter(f"{W}comment"):
                comment_id = node.attrib.get(f"{W}id", "")
                comment_by_id[comment_id] = {
                    "text": " ".join(
                        text for item in node.iter(f"{W}p") if (text := paragraph_text(item))
                    ),
                    "author": node.attrib.get(f"{W}author", ""),
                    "date": node.attrib.get(f"{W}date", ""),
                }

            anchors: dict[str, list[str]] = {}
            anchor_indexes: dict[str, list[int]] = {}
            paragraph_index = -1
            for paragraph in document.iter(f"{W}p"):
                text = paragraph_text(paragraph)
                if not text:
                    continue
                paragraph_index += 1
                ids = {
                    item.attrib.get(f"{W}id", "")
                    for item in paragraph.iter()
                    if item.tag in {f"{W}commentRangeStart", f"{W}commentReference"}
                }
                for comment_id in ids:
                    anchors.setdefault(comment_id, []).append(text)
                    anchor_indexes.setdefault(comment_id, []).append(paragraph_index)

            for comment_id, data in comment_by_id.items():
                comments.append(
                    {
                        "id": comment_id,
                        **data,
                        "anchor": " ".join(anchors.get(comment_id, [])),
                        "anchor_index": (anchor_indexes.get(comment_id) or [None])[0],
                    }
                )
    return {
        "path": str(path),
        "paragraphs": paragraphs,
        "comments": comments,
        "insertions": [item for item in insertions if item],
        "deletions": [item for item in deletions if item],
        "revision_authors": dict(revision_authors),
    }


def detect_files(folder: Path) -> dict[str, Path | None]:
    docx_files = sorted(folder.glob("*.docx"))
    pdf_files = sorted(folder.glob("*.pdf"))

    def pick(*tokens: str) -> Path | None:
        for path in docx_files:
            name = folded(path.stem)
            if all(token in name for token in tokens):
                return path
        return None

    original = pick("original")
    marked = pick("com marcas")
    clean_review = pick("sem marcas")
    layout = pick("para diagramar")
    return {
        "original": original,
        "marked": marked,
        "clean_review": clean_review,
        "final_docx": layout or clean_review,
        "pdf": pdf_files[0] if pdf_files else None,
    }


def infer_profile(folder: Path) -> str:
    name = folded(folder.name)
    patterns = (
        ("td ", "td"),
        ("nota tecnica", "nota_tecnica"),
        ("_nt_", "nota_tecnica"),
        ("boletim_bps", "boletim_bps"),
        ("boletim_bmt", "boletim_bmt"),
        ("boletim_radar", "boletim_radar"),
        ("revista_ppp", "artigo_ppp"),
        ("revista_ppe", "artigo_ppe"),
        ("ppe_", "artigo_ppe"),
        ("boletim", "boletim"),
        ("revista", "artigo_periodico"),
        ("livro", "capitulo_livro"),
    )
    for token, profile in patterns:
        if token in name:
            return profile
    return "generico"


def infer_agent(original: str, final: str, note: str = "") -> str:
    text = folded(" ".join((original, final, note)))
    if any(token in text for token in ("referenc", "citacao", "publicacao", "bibliograf", "link", "url", "disponivel em", "acesso em")):
        return "referencias"
    if any(token in text for token in ("tabela", "quadro", "figura", "grafico", "ilustracao", "fonte:")):
        return "tabelas_figuras"
    if any(token in text for token in ("negrito", "italico", "alinh", "recuo", "entrelinha", "formatacao", "tipograf")):
        return "tipografia"
    if re.match(r"^\s*\d+(?:\.\d+)*\s+\D", original or "") or any(
        token in text for token in ("titulo", "subtitulo", "secao", "hierarquia", "capitulo")
    ):
        return "estrutura"
    if any(token in text for token in ("sinopse", "abstract", "palavras-chave", "keywords", "jel")):
        return "sinopse_abstract"
    return "gramatica_ortografia"


def infer_action_type(original: str, final: str, note: str, agent: str) -> str:
    text = folded(" ".join((original, final, note)))
    if any(
        token in text
        for token in (
            "formato editavel",
            "boa resolucao",
            "encaminhem o grafico",
            "encaminhem a figura",
            "encaminhem a tabela",
            "arquivo editavel",
        )
    ):
        return "production_request"
    if any(
        token in text
        for token in (
            "pede-se verificar",
            "favor verificar",
            "pergunta-se",
            "a sua consideracao",
            "concordancia dos autores",
            "validar",
            "confirmar",
        )
    ):
        return "author_confirmation"
    if agent == "referencias" and any(
        token in text
        for token in (
            "dados bibliograficos",
            "link de acesso",
            "data de acesso",
            "qual publicacao",
            "qual referencia",
        )
    ):
        return "editorial_comment"
    if original and final and len(original) <= 220 and len(final) <= 220:
        return "auto_fix_candidate"
    return "editorial_comment"


def extract_pdf_text(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)


def best_match(text: str, candidates: list[str]) -> tuple[int | None, str, float]:
    if not text or not candidates:
        return None, "", 0.0
    best_index, best_text, best_score = None, "", 0.0
    for index, candidate in enumerate(candidates):
        score = similarity(text, candidate)
        if score > best_score:
            best_index, best_text, best_score = index, candidate, score
    return best_index, best_text, best_score


def aligned_match(
    source: list[str],
    target: list[str],
    source_index: int | None,
) -> tuple[int | None, str, float]:
    if source_index is None or source_index < 0 or source_index >= len(source):
        return None, "", 0.0
    matcher = SequenceMatcher(a=source, b=target, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if not (i1 <= source_index < i2):
            continue
        if tag == "equal":
            target_index = j1 + (source_index - i1)
            return target_index, target[target_index], 1.0
        if tag == "replace" and j1 < j2:
            local_index, text, score = best_match(source[source_index], target[j1:j2])
            return (j1 + local_index if local_index is not None else None), text, score
        return None, "", 0.0
    return None, "", 0.0


def diff_examples(original: list[str], final: list[str], limit: int) -> list[dict[str, object]]:
    matcher = SequenceMatcher(a=original, b=final, autojunk=False)
    examples: list[dict[str, object]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        old_items = original[i1:i2] or [""]
        new_items = final[j1:j2] or [""]
        for old in old_items:
            _, new, score = best_match(old, new_items)
            if not new and new_items:
                new = new_items[0]
            if old == new or (len(old) > 1200 and score < 0.25):
                continue
            examples.append(
                {
                    "source": "version_diff",
                    "agent": infer_agent(old, new),
                    "original": old,
                    "final": new,
                    "editor_note": "",
                    "status": "observed_change",
                    "confidence": "medium",
                    "similarity": round(score, 4),
                    "evaluation_eligible": bool(
                        score >= 0.55 and max(len(old), len(new)) <= 800
                    ),
                    "original_index": i1,
                    "final_index": j1,
                }
            )
            if len(examples) >= limit:
                return examples
    return examples


def learn(args: argparse.Namespace) -> int:
    folder = args.folder.expanduser().resolve()
    files = detect_files(folder)
    if files["original"] is None or files["final_docx"] is None:
        raise SystemExit(
            "A pasta precisa conter uma versão '(original).docx' e uma versão "
            "'(para diagramar).docx' ou '(sem marcas).docx'."
        )

    original_data = read_docx(files["original"])
    final_data = read_docx(files["final_docx"])
    review_path = files["marked"] or files["clean_review"]
    review_data = read_docx(review_path) if review_path else {"comments": [], "insertions": [], "deletions": []}
    review_paragraphs = list(review_data.get("paragraphs", []))
    original = list(original_data["paragraphs"])
    final = list(final_data["paragraphs"])
    pdf_text = extract_pdf_text(files["pdf"])

    examples: list[dict[str, object]] = []
    for comment in review_data["comments"]:
        anchor = str(comment.get("anchor") or "")
        review_index = comment.get("anchor_index")
        original_index, original_text, original_score = aligned_match(
            review_paragraphs, original, review_index
        )
        final_index, final_text, final_score = aligned_match(
            review_paragraphs, final, review_index
        )
        if original_text == "":
            original_index, original_text, original_score = best_match(anchor, original)
        if final_text == "":
            final_index, final_text, final_score = best_match(anchor, final)
        changed = bool(
            original_text
            and final_text
            and original_score >= 0.35
            and final_score >= 0.35
            and normalized(original_text) != normalized(final_text)
        )
        pdf_confirmed = bool(final_text and normalized(final_text)[:100] in normalized(pdf_text))
        agent = infer_agent(original_text or anchor, final_text, str(comment.get("text") or ""))
        examples.append(
            {
                "source": "editor_comment",
                "agent": agent,
                "action_type": infer_action_type(
                    original_text or anchor,
                    final_text,
                    str(comment.get("text") or ""),
                    agent,
                ),
                "original": original_text or anchor,
                "final": final_text if changed else "",
                "editor_note": comment.get("text", ""),
                "status": "confirmed_final" if changed else "unresolved_or_not_applied",
                "confidence": "high" if changed else "review_required",
                "evaluation_eligible": changed,
                "confirmed_in_pdf": pdf_confirmed,
                "original_index": original_index,
                "final_index": final_index,
                "anchor_similarity_original": round(original_score, 4),
                "anchor_similarity_final": round(final_score, 4),
                "reviewer": comment.get("author", ""),
                "date": comment.get("date", ""),
                "review_index": review_index,
            }
        )

    existing_originals = {normalized(item["original"]) for item in examples if item["original"]}
    for item in diff_examples(original, final, args.max_examples):
        if normalized(str(item["original"])) in existing_originals:
            continue
        item["confirmed_in_pdf"] = bool(
            item["final"] and normalized(str(item["final"]))[:100] in normalized(pdf_text)
        )
        examples.append(item)

    for item in examples:
        item.setdefault(
            "action_type",
            infer_action_type(
                str(item.get("original") or ""),
                str(item.get("final") or ""),
                str(item.get("editor_note") or ""),
                str(item.get("agent") or ""),
            ),
        )

    reviewers = Counter(
        str(item.get("reviewer") or "").strip()
        for item in examples
        if item.get("source") == "editor_comment" and str(item.get("reviewer") or "").strip()
    )
    stats = {
        "examples": len(examples),
        "editor_comments": sum(item["source"] == "editor_comment" for item in examples),
        "confirmed_final": sum(item["status"] == "confirmed_final" for item in examples),
        "observed_changes": sum(item["status"] == "observed_change" for item in examples),
        "review_required": sum(item["confidence"] == "review_required" for item in examples),
        "by_agent": dict(Counter(str(item["agent"]) for item in examples)),
        "by_action_type": dict(Counter(str(item["action_type"]) for item in examples)),
        "by_reviewer": dict(reviewers),
        "table_figure_examples": sum(item["agent"] == "tabelas_figuras" for item in examples),
        "table_figure_human_comments": sum(
            item["agent"] == "tabelas_figuras" and item["source"] == "editor_comment"
            for item in examples
        ),
        "tracked_insertions": len(review_data.get("insertions", [])),
        "tracked_deletions": len(review_data.get("deletions", [])),
        "revision_authors": review_data.get("revision_authors", {}),
    }
    payload = {
        "schema_version": 1,
        "folder": str(folder),
        "profile": args.profile or infer_profile(folder),
        "files": {key: str(value) if value else None for key, value in files.items()},
        "stats": stats,
        "examples": examples,
    }

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "editorial_knowledge.json"
    markdown_path = out_dir / "editorial_knowledge.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Conhecimento editorial extraído",
        "",
        f"- Perfil: `{payload['profile']}`",
        f"- Exemplos: {stats['examples']}",
        f"- Comentários humanos: {stats['editor_comments']}",
        f"- Confirmados na versão final: {stats['confirmed_final']}",
        f"- Mudanças observadas entre versões: {stats['observed_changes']}",
        f"- Pendentes de revisão humana: {stats['review_required']}",
        f"- Alterações rastreadas: {stats['tracked_insertions']} inserções e {stats['tracked_deletions']} exclusões",
        "",
        "## Distribuição por agente",
        "",
    ]
    lines.extend(f"- `{agent}`: {count}" for agent, count in sorted(stats["by_agent"].items()))
    lines.extend(["", "## Amostra", ""])
    for item in examples[:20]:
        lines.extend(
            [
                f"### {item['agent']} — {item['status']}",
                "",
                f"- Original: {str(item['original'])[:500]}",
                f"- Final: {str(item['final'])[:500] or '(sem alteração confirmada)'}",
                f"- Nota: {str(item['editor_note'])[:500] or '(mudança inferida por comparação)'}",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    if not getattr(args, "quiet", False):
        print(f"Conhecimento JSON: {json_path}")
        print(f"Resumo Markdown: {markdown_path}")
        print(json.dumps(stats, ensure_ascii=False))
    return 0


def safe_folder_name(folder: Path) -> str:
    value = unicodedata.normalize("NFKD", folder.name)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return value[:120] or "projeto"


def batch_learn(args: argparse.Namespace) -> int:
    root = args.root.expanduser().resolve()
    excluded = {folded(item) for item in args.exclude_name}
    folders = []
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        if folded(folder.name) in excluded:
            continue
        files = detect_files(folder)
        if files["original"] is not None and files["final_docx"] is not None:
            folders.append(folder)

    output_root = args.out_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, object]] = []

    def run_one(index: int, folder: Path) -> dict[str, object]:
        out_dir = output_root / f"{index:03d}-{safe_folder_name(folder)}"
        namespace = argparse.Namespace(
            folder=folder,
            out_dir=out_dir,
            profile=None,
            max_examples=args.max_examples,
            quiet=True,
        )
        try:
            learn(namespace)
            data = json.loads((out_dir / "editorial_knowledge.json").read_text(encoding="utf-8"))
            return {
                "folder": str(folder),
                "out_dir": str(out_dir),
                "status": "ok",
                "profile": data.get("profile"),
                "stats": data.get("stats", {}),
                "marked_doc_only": bool(
                    not list(folder.glob("*com marcas*.docx"))
                    and list(folder.glob("*com marcas*.doc"))
                ),
            }
        except Exception as exc:
            return {"folder": str(folder), "out_dir": str(out_dir), "status": "error", "error": str(exc)}

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(run_one, index, folder): folder
            for index, folder in enumerate(folders, start=1)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            print(f"[{completed}/{len(folders)}] {Path(result['folder']).name}: {result['status']}")

    ok_results = [item for item in results if item["status"] == "ok"]
    total = Counter()
    agents = Counter()
    actions = Counter()
    reviewers = Counter()
    revision_authors = Counter()
    profiles = Counter()
    for item in ok_results:
        stats = item["stats"]
        profiles[str(item.get("profile") or "")] += 1
        for key in (
            "examples",
            "editor_comments",
            "confirmed_final",
            "observed_changes",
            "review_required",
            "table_figure_examples",
            "table_figure_human_comments",
            "tracked_insertions",
            "tracked_deletions",
        ):
            total[key] += int(stats.get(key, 0) or 0)
        agents.update(stats.get("by_agent", {}))
        actions.update(stats.get("by_action_type", {}))
        reviewers.update(stats.get("by_reviewer", {}))
        revision_authors.update(stats.get("revision_authors", {}))

    summary = {
        "schema_version": 1,
        "root": str(root),
        "projects_total": len(folders),
        "projects_ok": len(ok_results),
        "projects_error": len(results) - len(ok_results),
        "marked_doc_only": sum(bool(item.get("marked_doc_only")) for item in ok_results),
        "totals": dict(total),
        "by_profile": dict(profiles),
        "by_agent": dict(agents),
        "by_action_type": dict(actions),
        "by_reviewer": dict(reviewers),
        "revision_authors": dict(revision_authors),
        "projects": sorted(results, key=lambda item: item["folder"]),
    }
    summary_path = output_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# Aprendizado editorial em lote",
        "",
        f"- Projetos elegíveis: {summary['projects_total']}",
        f"- Processados: {summary['projects_ok']}",
        f"- Erros: {summary['projects_error']}",
        f"- Marcas disponíveis somente em DOC antigo: {summary['marked_doc_only']}",
        "",
        "## Totais",
        "",
    ]
    markdown.extend(f"- {key}: {value}" for key, value in total.items())
    markdown.extend(["", "## Revisores identificados", ""])
    markdown.extend(f"- {name}: {count}" for name, count in reviewers.most_common())
    markdown.extend(["", "## Autores das alterações rastreadas", ""])
    markdown.extend(f"- {name}: {count}" for name, count in revision_authors.most_common())
    markdown.extend(["", "## Perfis", ""])
    markdown.extend(f"- {name}: {count}" for name, count in profiles.most_common())
    (output_root / "batch_summary.md").write_text("\n".join(markdown), encoding="utf-8")
    print(f"Resumo JSON: {summary_path}")
    print(f"Resumo Markdown: {output_root / 'batch_summary.md'}")
    return 0 if not summary["projects_error"] else 1


AGENT_ALIASES = {
    "gram": "gramatica_ortografia",
    "ref": "referencias",
    "tab": "tabelas_figuras",
    "est": "estrutura",
    "tip": "tipografia",
    "sin": "sinopse_abstract",
}


def generic_comment(comment: dict[str, object]) -> bool:
    text = folded(f"{comment.get('message', '')} {comment.get('suggested_fix', '')}")
    return any(
        token in text
        for token in (
            "revisar o trecho",
            "ajustar a formatacao",
            "nao segue as diretrizes",
            "nao esta formatada corretamente",
            "reorganizar o conteudo",
        )
    )


def evaluate(args: argparse.Namespace) -> int:
    knowledge = json.loads(args.knowledge.read_text(encoding="utf-8"))
    comments = json.loads(args.report.read_text(encoding="utf-8"))
    if args.gold_scope == "confirmed":
        allowed_statuses = {"confirmed_final"}
    else:
        allowed_statuses = {"confirmed_final", "observed_change"}
    gold = [
        item
        for item in knowledge.get("examples", [])
        if item.get("status") in allowed_statuses
        and (args.gold_scope == "all" or item.get("evaluation_eligible", True))
        and item.get("original")
        and item.get("final")
    ]
    used_gold: set[int] = set()
    findings: list[dict[str, object]] = []

    for comment in comments:
        issue = str(comment.get("issue_excerpt") or "")
        fix = str(comment.get("suggested_fix") or "")
        agent = AGENT_ALIASES.get(str(comment.get("agent") or ""), str(comment.get("agent") or ""))
        violations = []
        if not issue.strip():
            violations.append("issue_excerpt vazio")
        if not fix.strip():
            violations.append("suggested_fix vazio")
        if generic_comment(comment):
            violations.append("comentário genérico")

        best_index, best_score = None, 0.0
        for index, example in enumerate(gold):
            issue_score = similarity(issue, str(example.get("original") or ""))
            fix_score = similarity(fix, str(example.get("final") or ""))
            agent_bonus = 0.08 if agent == example.get("agent") else 0.0
            score = min(1.0, issue_score * 0.65 + fix_score * 0.35 + agent_bonus)
            if score > best_score:
                best_index, best_score = index, score

        matched = best_index is not None and best_score >= args.threshold and not violations
        if matched:
            used_gold.add(best_index)
        findings.append(
            {
                "comment": comment,
                "classification": "candidate_true_positive" if matched else "candidate_false_positive",
                "match_score": round(best_score, 4),
                "matched_gold_index": best_index if matched else None,
                "violations": violations,
            }
        )

    false_negatives = [
        {"gold_index": index, "example": item}
        for index, item in enumerate(gold)
        if index not in used_gold
    ]
    true_positive_count = sum(item["classification"] == "candidate_true_positive" for item in findings)
    false_positive_count = len(findings) - true_positive_count
    precision = true_positive_count / len(findings) if findings else 0.0
    recall = len(used_gold) / len(gold) if gold else 0.0
    stats = {
        "agent_comments": len(comments),
        "gold_candidates": len(gold),
        "candidate_true_positives": true_positive_count,
        "candidate_false_positives": false_positive_count,
        "candidate_false_negatives": len(false_negatives),
        "candidate_precision": round(precision, 4),
        "candidate_recall": round(recall, 4),
        "comments_with_contract_violations": sum(bool(item["violations"]) for item in findings),
        "gold_scope": args.gold_scope,
        "by_agent": dict(Counter(str(item.get("agent") or "") for item in comments)),
    }
    payload = {
        "schema_version": 1,
        "note": "Métricas candidatas: diferenças editoriais incluem decisões humanas e diagramação; validar antes de alterar regras.",
        "threshold": args.threshold,
        "stats": stats,
        "findings": findings,
        "false_negatives": false_negatives,
    }
    if args.diagnostics and args.diagnostics.exists():
        diagnostics = json.loads(args.diagnostics.read_text(encoding="utf-8"))
        failed = [
            item["agent"]
            for item in diagnostics.get("trace", {}).get("agents", [])
            if item.get("failed")
        ]
        payload["runtime"] = diagnostics.get("runtime", {})
        payload["failed_agents"] = failed
        stats["complete_llm_run"] = not failed

    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "editorial_evaluation.json"
    markdown_path = out_dir / "editorial_evaluation.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Avaliação editorial candidata",
        "",
        "> As métricas são triagem automática e precisam de validação humana antes de calibrar agentes.",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in stats.items() if key != "by_agent")
    lines.extend(["", "## Comentários possivelmente falsos positivos", ""])
    for item in findings:
        if item["classification"] == "candidate_false_positive":
            comment = item["comment"]
            lines.append(
                f"- `{comment.get('agent')}`: {comment.get('issue_excerpt') or '(sem trecho)'} "
                f"→ {comment.get('suggested_fix')} "
                f"(score={item['match_score']}, violações={item['violations']})"
            )
    lines.extend(["", "## Mudanças humanas não cobertas", ""])
    for item in false_negatives[:30]:
        example = item["example"]
        lines.append(
            f"- `{example.get('agent')}`: {str(example.get('original'))[:250]} "
            f"→ {str(example.get('final'))[:250]}"
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Avaliação JSON: {json_path}")
    print(f"Resumo Markdown: {markdown_path}")
    print(json.dumps(stats, ensure_ascii=False))
    return 0


def find_project_root(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else []
    if os.environ.get("IPEA_EDITORIAL_ROOT"):
        candidates.append(Path(os.environ["IPEA_EDITORIAL_ROOT"]))
    candidates.extend([Path.cwd(), *Path.cwd().parents])
    for candidate in candidates:
        if candidate is None:
            continue
        root = candidate.expanduser().resolve()
        if (root / "src" / "editorial_docx" / "llm.py").exists():
            return root
    raise SystemExit("Projeto lang_IPEA_editorial não encontrado.")


def preflight(args: argparse.Namespace) -> int:
    root = find_project_root(args.project_root)
    sys.path.insert(0, str(root / "src"))
    from editorial_docx.llm import get_llm_config, list_available_models

    config = get_llm_config()
    provider = config.get("provider", "")
    model = config.get("model", "")
    base_url = config.get("base_url", "")
    api_key = config.get("api_key", "")
    result = {"provider": provider, "model": model, "base_url": base_url, "ok": False}
    try:
        if provider == "ollama":
            root_url = (base_url or "http://localhost:11434/v1").removesuffix("/v1").rstrip("/")
            request = Request(f"{root_url}/api/tags")
            with urlopen(request, timeout=args.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            names = {item.get("name") for item in data.get("models", [])}
            result["available_models"] = sorted(name for name in names if name)
            result["ok"] = model in names or f"{model}:latest" in names
            if not result["ok"]:
                result["error"] = f"Modelo não instalado no Ollama: {model}"
        else:
            models_result = list_available_models(config, timeout=args.timeout)
            result["endpoint"] = models_result.get("endpoint", "")
            result["available_models"] = models_result.get("available_models", [])
            result["ok"] = bool(models_result.get("ok"))
            if "configured_model_available" in models_result:
                result["configured_model_available"] = models_result["configured_model_available"]
            if not result["ok"] and models_result.get("error"):
                result["error"] = str(models_result["error"])
            if not api_key and provider == "openai":
                result["ok"] = False
                result["error"] = "Chave da OpenAI ausente."
    except HTTPError as exc:
        result["error"] = f"HTTP {exc.code}: autenticação ou endpoint inválido."
    except (URLError, TimeoutError, OSError) as exc:
        result["error"] = f"Conexão falhou: {exc}"
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    learn_parser = subparsers.add_parser("learn", help="Extrai conhecimento de uma pasta editorial.")
    learn_parser.add_argument("folder", type=Path)
    learn_parser.add_argument("--out-dir", type=Path, required=True)
    learn_parser.add_argument("--profile")
    learn_parser.add_argument("--max-examples", type=int, default=200)
    learn_parser.set_defaults(func=learn)

    batch_parser = subparsers.add_parser("batch-learn", help="Extrai conhecimento de todas as pastas elegíveis.")
    batch_parser.add_argument("root", type=Path)
    batch_parser.add_argument("--out-dir", type=Path, required=True)
    batch_parser.add_argument("--exclude-name", action="append", default=[])
    batch_parser.add_argument("--workers", type=int, default=4)
    batch_parser.add_argument("--max-examples", type=int, default=300)
    batch_parser.set_defaults(func=batch_learn)

    evaluate_parser = subparsers.add_parser("evaluate", help="Compara agentes com o gabarito extraído.")
    evaluate_parser.add_argument("--knowledge", type=Path, required=True)
    evaluate_parser.add_argument("--report", type=Path, required=True)
    evaluate_parser.add_argument("--diagnostics", type=Path)
    evaluate_parser.add_argument("--out-dir", type=Path, required=True)
    evaluate_parser.add_argument("--threshold", type=float, default=0.58)
    evaluate_parser.add_argument(
        "--gold-scope",
        choices=("confirmed", "local-changes", "all"),
        default="confirmed",
        help="confirmed usa apenas comentários humanos confirmados na versão final.",
    )
    evaluate_parser.set_defaults(func=evaluate)

    preflight_parser = subparsers.add_parser("preflight", help="Testa provider e modelo sem inferência.")
    preflight_parser.add_argument("--project-root", type=Path)
    preflight_parser.add_argument("--timeout", type=float, default=10.0)
    preflight_parser.set_defaults(func=preflight)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

from ...review_patterns import _normalized_text
from .shared import ValidationContext, has_resolved_text_anchor


def keep_rejection_reason(ctx: ValidationContext) -> str | None:
    """Mantém somente alertas lógicos locais, verificáveis e sem reescrita autoral."""
    comment = ctx.comment
    issue = (comment.issue_excerpt or "").strip()
    suggestion = (comment.suggested_fix or "").strip()

    if ctx.block_type not in {"paragraph", "abstract_body"}:
        return "coerência lógica fora de parágrafo analítico"
    if comment.category != "coerencia_logica":
        return "categoria inválida para coerência lógica"
    if comment.action_type != "author_confirmation":
        return "coerência lógica exige confirmação do autor"
    if not issue or len(issue) > 500 or not has_resolved_text_anchor(issue, comment.paragraph_index, ctx.chunks):
        return "alerta lógico sem trecho verificável"
    if _normalized_text(issue) not in _normalized_text(ctx.source_text or ""):
        return "alerta lógico deve estar integralmente ancorado no mesmo trecho"
    if any(token in ctx.folded_fix for token in ("substitua", "troque", "reescreva", "altere para")):
        return "coerência lógica não pode reescrever o argumento"
    if not suggestion or not any(token in ctx.folded_fix for token in ("confirm", "verifi", "revis")):
        return "alerta lógico sem pedido de confirmação"
    return None

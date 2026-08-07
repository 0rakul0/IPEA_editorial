from __future__ import annotations

from ...review_patterns import _ref_block_type


def build_scope(chunks: list[str], refs: list[str], sections, total: int) -> list[int]:
    """Seleciona parágrafos analíticos que possam conter uma afirmação verificável."""
    reference_heading_idx = next((idx for idx, ref in enumerate(refs) if _ref_block_type(ref) == "reference_heading"), total)
    return [
        idx
        for idx in range(reference_heading_idx)
        if _ref_block_type(refs[idx] if idx < len(refs) else "") in {"paragraph", "abstract_body"}
        and len((chunks[idx] or "").strip()) >= 80
    ]

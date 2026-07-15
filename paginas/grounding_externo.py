from __future__ import annotations

import json

import streamlit as st


def render_grounding_externo_tab(*, grounding_payload: dict[str, object] | None) -> None:
    """Renderiza a aba de grounding externo."""
    st.subheader("Grounding Externo")

    grounding_result = st.session_state.grounding_result

    if not st.session_state.paragraphs:
        st.info("Carregue um documento para buscar literatura recente e comparar com o manuscrito.")
        return

    if st.session_state.grounding_error:
        st.error(st.session_state.grounding_error)
        return

    if grounding_result is None:
        st.info("Use o botao da barra lateral para rodar a camada opcional de grounding externo.")
        return

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Consultas", len(grounding_result.queries))
    metric_b.metric("Trabalhos", len(grounding_result.works))
    metric_c.metric("LLM usada", "Sim" if grounding_result.llm_used else "Nao")

    if grounding_payload is not None:
        grounding_json_name = f"{st.session_state.source_name}_grounding_externo.json"
        st.download_button(
            label="Baixar grounding JSON",
            data=json.dumps(grounding_payload, ensure_ascii=False, indent=2),
            file_name=grounding_json_name,
            mime="application/json",
            use_container_width=True,
        )

    if grounding_result.warnings:
        for warning in grounding_result.warnings:
            st.warning(warning)

    st.markdown("**Sintese do manuscrito**")
    st.write(grounding_result.manuscript_summary or "Sintese indisponivel.")

    st.markdown("**Estado da arte relevante**")
    st.write(grounding_result.state_of_art_summary or "Sintese indisponivel.")

    st.markdown("**Comparacao com o manuscrito**")
    st.write(grounding_result.manuscript_comparison or "Comparacao indisponivel.")

    if st.session_state.grounding_logs:
        with st.expander("Log da execucao", expanded=False):
            st.markdown("\n".join(st.session_state.grounding_logs))

    if grounding_result.queries:
        with st.expander("Consultas geradas", expanded=False):
            for idx, query in enumerate(grounding_result.queries, start=1):
                st.markdown(f"**{idx}.** `{query.text}`")
                if query.rationale:
                    st.caption(query.rationale)
                st.caption(f"Origem: {query.source}")

    if grounding_result.works:
        with st.expander("Literatura recuperada", expanded=False):
            for idx, work in enumerate(grounding_result.works, start=1):
                year = f" ({work.publication_year})" if work.publication_year else ""
                venue = f" | {work.venue}" if work.venue else ""
                st.markdown(f"**{idx}. {work.title}{year}**")
                st.caption(
                    f"Relevancia: {work.relevance_score:.2f} | Citacoes: {work.cited_by_count}{venue}"
                )
                if work.authors:
                    st.write("Autores:", ", ".join(work.authors))
                if work.doi:
                    st.write("DOI:", work.doi)
                if work.landing_page_url:
                    st.markdown(f"[Abrir registro]({work.landing_page_url})")
                if work.matched_queries:
                    st.caption("Consultas relacionadas: " + "; ".join(work.matched_queries))
                st.write(work.abstract or "Resumo nao disponivel.")

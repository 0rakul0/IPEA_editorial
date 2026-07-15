from __future__ import annotations

from collections.abc import Callable

import streamlit as st


def render_erros_encontrados_tab(
    *,
    rows: list[dict],
    render_target_excerpt: Callable[[str, str], None],
) -> None:
    """Renderiza a aba com os comentarios editoriais encontrados."""
    st.subheader("Erros Encontrados")

    if not rows:
        st.info("Os comentarios dos agentes vao aparecer aqui depois da execucao.")
        return

    agent_options = sorted({row["agente"] for row in rows})
    category_options = sorted({row["categoria"] for row in rows})
    filter_a, filter_b = st.columns(2)
    with filter_a:
        selected_agents = st.multiselect(
            "Filtrar por agente",
            options=agent_options,
            default=[],
            key="diagnostic_agent_filter",
        )
    with filter_b:
        selected_categories = st.multiselect(
            "Filtrar por categoria",
            options=category_options,
            default=[],
            key="diagnostic_category_filter",
        )

    visible_rows = [
        row
        for row in rows
        if (not selected_agents or row["agente"] in selected_agents)
        and (not selected_categories or row["categoria"] in selected_categories)
    ]

    summary_a, summary_b, summary_c = st.columns(3)
    summary_a.metric("Itens filtrados", len(visible_rows))
    summary_b.metric("Agentes no filtro", len({row["agente"] for row in visible_rows}) if visible_rows else 0)
    summary_c.metric("Categorias no filtro", len({row["categoria"] for row in visible_rows}) if visible_rows else 0)

    if not visible_rows:
        st.info("Nenhum comentario corresponde aos filtros atuais.")
        return

    for row in visible_rows:
        title = (
            f"[{row['agente']}] {row['categoria']}"
            + (f" | trecho {row['indice_trecho']}" if isinstance(row["indice_trecho"], int) else "")
        )
        with st.expander(title, expanded=False):
            state_key = str(row["comment_idx"])
            state = st.session_state.correction_state.setdefault(
                state_key,
                {
                    "status": "pendente",
                    "final_text": row["como_deve_ficar"] or row["trecho_com_problema"] or "",
                    "observacao": "",
                },
            )
            status_label = {
                "pendente": "Pendente",
                "resolvido": "Aceito",
                "rejeitado": "Rejeitado",
            }.get(state.get("status", "pendente"), state.get("status", "pendente"))
            st.caption(f"Status: {status_label}")

            st.markdown(f"**Referencia:** {row['referencia']}")
            st.markdown(f"**Comentario:** {row['comentario']}")
            st.markdown("**Trecho com problema**")
            st.code(row["trecho_com_problema"] or "(nao informado)", language="text")
            st.markdown("**Sugestao de correcao**")
            st.code(row["como_deve_ficar"] or "(nao informado)", language="text")

            final_text_key = f"final_text_{state_key}"
            if final_text_key not in st.session_state:
                st.session_state[final_text_key] = state.get("final_text", "")
            final_text = st.text_area(
                "Correcao que ira para o documento",
                key=final_text_key,
                placeholder="Edite aqui o texto final a aplicar no documento.",
            )
            state["final_text"] = final_text

            note_key = f"review_note_{state_key}"
            if note_key not in st.session_state:
                st.session_state[note_key] = state.get("observacao", "")
            note = st.text_area(
                "Comentario/correcao do usuario",
                key=note_key,
                placeholder="Escreva aqui a correcao que deve ser aplicada no documento.",
            )
            state["observacao"] = note
            if note.strip() and state.get("status") != "rejeitado":
                state["final_text"] = note
                state["status"] = "resolvido"

            action_accept, action_reject = st.columns(2)
            with action_accept:
                if st.button("Aceitar comentario", key=f"accept_comment_{state_key}", use_container_width=True):
                    state["status"] = "resolvido"
                    state["final_text"] = note.strip() or final_text or row["como_deve_ficar"] or row["trecho_com_problema"] or ""
                    state["observacao"] = note
                    st.rerun()
            with action_reject:
                if st.button("Rejeitar comentario", key=f"reject_comment_{state_key}", use_container_width=True):
                    state["status"] = "rejeitado"
                    state["final_text"] = ""
                    state["observacao"] = note
                    st.rerun()

            pidx = row["indice_trecho"]
            if isinstance(pidx, int) and 0 <= pidx < len(st.session_state.paragraphs):
                render_target_excerpt(st.session_state.paragraphs[pidx], row["trecho_com_problema"])

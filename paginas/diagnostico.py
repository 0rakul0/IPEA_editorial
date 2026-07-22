from __future__ import annotations

from pathlib import Path

import streamlit as st


def render_diagnostico_tab(
    *,
    docx_path: Path | None,
    docx_bytes: bytes | None,
) -> None:
    """Renderiza a aba de diagnostico editorial."""
    st.subheader("Diagnostico")

    if not st.session_state.paragraphs:
        st.info("Carregue um documento e use os botoes da barra lateral para rodar os agentes.")
        return

    if not st.session_state.comments:
        st.info("Documento carregado. Rode os agentes na barra lateral para gerar o diagnostico editorial.")
        return

    st.metric("Comentarios gerados", len(st.session_state.comments))

    if docx_path and docx_bytes is not None:
        st.download_button(
            label="Baixar DOCX comentado",
            data=docx_bytes,
            file_name=docx_path.name,
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            use_container_width=True,
        )

# Codex — lang_IPEA_editorial

- Use a skill `revisao-editorial-ipea` para executar ou alterar o fluxo editorial.
- Trate `src/editorial_docx/graph_chat.py` como fachada pública.
- Preserve prompts, escopos, heurísticas, validação e consolidação.
- Não exponha `.env`, chaves ou tokens.
- Não invente dados bibliográficos nem transforme preferência estilística em erro.
- Preserve o documento original e gere saídas separadas.
- Depois de alterar o pipeline, execute os testes focados e `compileall`.
- Informe falhas de agentes e não apresente resultado parcial como revisão completa.


# Claude Code — lang_IPEA_editorial

- Use `/revisao-editorial-ipea` para executar ou alterar o fluxo editorial.
- Trate `src/editorial_docx/graph_chat.py` como fachada pública.
- Preserve prompts, escopos, heurísticas, validação e consolidação.
- Não leia nem exponha `.env`, chaves ou tokens.
- Não invente dados bibliográficos ou transforme preferência estilística em erro.
- Preserve a entrada e grave artefatos separados.
- Após mudanças, execute os testes focados e `compileall`.
- Informe falhas de agentes e não apresente revisão parcial como completa.


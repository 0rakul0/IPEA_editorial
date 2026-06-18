---
name: revisao-editorial-ipea
description: Executar, diagnosticar, avaliar e aperfeiçoar a revisão editorial Ipea de documentos DOCX, PDF ou normalized_document.json usando o pipeline lang_IPEA_editorial. Usar para revisão editorial, DOCX comentado, diagnóstico, aprendizado a partir de versões original/com marcas/para diagramar/PDF, comparação com revisão humana, avaliação de falsos positivos e calibração de gramática, sinopse, tabelas, estrutura, referências ABNT e tipografia.
---

# Revisão editorial Ipea

Usar o pipeline Python como fonte de verdade. Não substituir escopo, heurísticas, validações, deduplicação e exportação por revisão improvisada.

## Fluxo

1. Localizar a raiz pelo repositório aberto ou `IPEA_EDITORIAL_ROOT`.
2. Ler `references/operacao.md` antes de configurar ou executar.
3. Ler `references/agentes.md` para escopo e interpretação.
4. Ler `references/arquitetura.md` ao diagnosticar ou alterar.
5. Ler `references/contrato-saida.md` ao validar ou integrar.
6. Ler `references/aprendizado.md` ao aprender ou avaliar.
7. Não ler `.env` nem mostrar segredos.
8. Executar o preflight e depois `scripts/run_review.py`.
9. Conferir código de saída, artefatos e falhas.
10. Informar documento, arquivos, total de comentários, agentes indisponíveis e limitações.

## Aprender e avaliar

- Executar `scripts/editorial_lab.py learn` para extrair conhecimento de versões humanas.
- Executar `scripts/editorial_lab.py evaluate` para comparar agentes ao gabarito.
- Não aplicar regras candidatas automaticamente; exigir revisão e repetição em documentos independentes.

## Guardrails

- Preservar a entrada.
- Não inventar referências ou dados.
- Não tratar preferência estilística como erro.
- Não alterar citações diretas.
- Não apresentar execução parcial como completa.

## Mudanças

Identificar a camada correta, fazer mudança mínima, adicionar regressão e executar:

```bash
uv run pytest testes/test_architecture_modular.py testes/test_graph_chat.py -q
uv run python -m compileall src/editorial_docx streamlit_app.py
```

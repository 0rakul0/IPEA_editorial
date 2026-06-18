---
name: revisao-editorial-ipea
description: Executar, diagnosticar, avaliar e aperfeiçoar a revisão editorial Ipea de documentos DOCX, PDF ou normalized_document.json usando o pipeline lang_IPEA_editorial e seus agentes especializados. Usar quando o usuário pedir revisão editorial, Texto para Discussão (TD), DOCX comentado, relatório JSON, diagnóstico, comparação com revisão humana, aprendizado a partir de versões original/com marcas/para diagramar/PDF, avaliação de falsos positivos ou calibração de gramática, sinopse, tabelas, estrutura, referências ABNT e tipografia.
---

# Revisão editorial Ipea

Usar o pipeline Python como fonte de verdade. Os prompts isolados não reproduzem as heurísticas, o escopo, a validação, a deduplicação nem a ancoragem em DOCX.

## Fluxo

1. Localizar a raiz do projeto:
   - preferir o repositório aberto;
   - senão usar `IPEA_EDITORIAL_ROOT`;
   - confirmar `pyproject.toml` com o pacote `lang-ipea-editorial`.
2. Ler `references/operacao.md` antes de instalar, configurar ou executar.
3. Ler `references/agentes.md` quando a solicitação envolver agentes, escopo ou interpretação dos achados.
4. Ler `references/arquitetura.md` ao diagnosticar, alterar ou explicar o pipeline.
5. Ler `references/contrato-saida.md` ao validar relatórios ou integrar resultados.
6. Ler `references/aprendizado.md` ao aprender com ciclos editoriais ou avaliar agentes.
7. Não ler `.env` nem exibir chaves. Verificar apenas se a configuração necessária existe.
8. Executar o preflight e depois a revisão com `scripts/run_review.py`.
9. Conferir código de saída e arquivos gerados. Em falha, preservar a mensagem original e diagnosticar antes de alterar código.
10. Resumir documento processado, artefatos, total de comentários, agentes com falha e limitações.

## Aprender e avaliar

- Para extrair conhecimento de versões humanas, executar `scripts/editorial_lab.py learn`.
- Para comparar um relatório dos agentes ao gabarito extraído, executar `scripts/editorial_lab.py evaluate`.
- Nunca aplicar automaticamente regras candidatas. Revisar exemplos ambíguos e exigir repetição em documentos independentes.

## Guardrails editoriais

- Não realizar correções silenciosas fora do que o pipeline aplica explicitamente.
- Não inventar dados bibliográficos.
- Não tratar preferência de estilo como erro objetivo.
- Não alterar citações diretas.
- Não prometer conformidade ABNT sem verificar achados e limitações.
- Preservar o arquivo de entrada; gravar saídas nos caminhos solicitados ou ao lado da entrada.

## Alterações no projeto

1. Identificar a camada correta: prompt, escopo, heurística, validação, consolidação ou exportação.
2. Fazer a menor alteração coerente.
3. Adicionar ou ajustar teste de regressão.
4. Executar:

```bash
uv run pytest testes/test_architecture_modular.py testes/test_graph_chat.py -q
uv run python -m compileall src/editorial_docx streamlit_app.py
```

5. Explicar o efeito sobre falsos positivos e falsos negativos.

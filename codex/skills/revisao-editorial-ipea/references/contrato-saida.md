# Contrato de saída

```json
{
  "agent": "gram",
  "category": "inconsistency",
  "message": "Explicação curta do problema local.",
  "paragraph_index": 10,
  "issue_excerpt": "fragmento com problema",
  "suggested_fix": "fragmento corrigido"
}
```

- `message`: diagnóstico natural, objetivo e curto.
- `paragraph_index`: índice global, não posição no lote.
- `issue_excerpt`: menor fragmento que sustenta o comentário.
- `suggested_fix`: correção exata ou instrução curta.
- `format_spec`: ajustes tipográficos seguros.

O diagnóstico opcional contém runtime sem segredo, decisões, origem, métricas por agente/lote e falhas. Não usar apenas a quantidade de comentários como medida de qualidade.


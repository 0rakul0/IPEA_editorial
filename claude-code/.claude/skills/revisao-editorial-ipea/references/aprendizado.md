# Aprendizado e avaliação

Não transformar uma alteração isolada em regra. Extrair exemplos, confirmar na versão final ou PDF e revisar ambiguidades.

## Aprender

Usar uma pasta com `(original).docx`, `(com marcas).docx`, `(sem marcas).docx`, `(para diagramar).docx` e, quando disponível, o PDF final:

```bash
python scripts/editorial_lab.py learn "/pasta/editorial" --out-dir "/saida/aprendizado"
```

Saídas:

- `editorial_knowledge.json`;
- `editorial_knowledge.md`.

Estados:

- `confirmed_final`: comentário acompanhado por mudança final;
- `observed_change`: diferença entre original e versão final;
- `unresolved_or_not_applied`: sem confirmação; não usar automaticamente como exemplo negativo.

## Avaliar

```bash
python scripts/editorial_lab.py evaluate \
  --knowledge "/saida/aprendizado/editorial_knowledge.json" \
  --report "/saida/agentes-relatorio.json" \
  --diagnostics "/saida/agentes-diagnostico.json" \
  --out-dir "/saida/avaliacao"
```

A saída identifica candidatos a acerto, falso positivo, falso negativo e violações de contrato. Validar humanamente porque diferenças podem incluir decisão autoral ou diagramação.

O padrão usa apenas comentários humanos confirmados. Use `--gold-scope local-changes` ou `--gold-scope all` somente para análises mais amplas.

## Preflight

```bash
python scripts/editorial_lab.py preflight --project-root "/caminho/projeto"
```

Testar endpoint e modelo sem inferência. `run_review.py` interrompe por padrão quando a LLM não está disponível. Usar `--skip-preflight` somente para diagnóstico consciente de heurísticas.

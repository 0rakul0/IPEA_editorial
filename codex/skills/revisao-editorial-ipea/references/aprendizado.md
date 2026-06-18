# Aprendizado e avaliação

## Princípio

Não transformar uma alteração isolada em regra. Primeiro extrair exemplos, confirmar a decisão na versão final ou no PDF e revisar os casos ambíguos.

## Modo aprender

Esperar uma pasta com nomes semelhantes a:

- `(original).docx`;
- `(com marcas).docx`;
- `(sem marcas).docx`;
- `(para diagramar).docx`;
- `.pdf` final.

Executar:

```bash
python scripts/editorial_lab.py learn "/pasta/editorial" --out-dir "/saida/aprendizado"
```

O comando gera:

- `editorial_knowledge.json`: exemplos estruturados;
- `editorial_knowledge.md`: resumo para revisão humana.

Estados:

- `confirmed_final`: comentário humano acompanhado por mudança na versão final;
- `observed_change`: mudança detectada entre original e versão final;
- `unresolved_or_not_applied`: comentário sem alteração final confirmada; não usar automaticamente como exemplo negativo.

## Modo avaliar

Depois de executar os agentes no documento original:

```bash
python scripts/editorial_lab.py evaluate \
  --knowledge "/saida/aprendizado/editorial_knowledge.json" \
  --report "/saida/agentes-relatorio.json" \
  --diagnostics "/saida/agentes-diagnostico.json" \
  --out-dir "/saida/avaliacao"
```

Por padrão, usar apenas comentários humanos confirmados na versão final. Para ampliar:

- `--gold-scope local-changes`: incluir mudanças locais elegíveis;
- `--gold-scope all`: incluir todas as mudanças observadas, com maior risco de misturar decisão autoral e diagramação.

O resultado separa candidatos a:

- verdadeiro positivo;
- falso positivo;
- falso negativo;
- violação de contrato, como `issue_excerpt` vazio ou comentário genérico.

As métricas são candidatas porque diferenças entre versões também podem incluir decisão autoral e diagramação. Validar humanamente antes de alterar prompt, heurística ou validação.

## Preflight

```bash
python scripts/editorial_lab.py preflight --project-root "/caminho/projeto"
```

O preflight testa endpoint e modelo sem realizar inferência. O wrapper `run_review.py` o executa por padrão e interrompe a revisão se a LLM estiver indisponível. Usar `--skip-preflight` somente para diagnóstico consciente de heurísticas.

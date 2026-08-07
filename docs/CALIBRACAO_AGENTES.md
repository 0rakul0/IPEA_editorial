# Calibração dos agentes editoriais

## Objetivo

Calibrar a aplicação com evidência comparável, sem alterar prompts por impressão isolada. Cada rodada deve usar documentos que não tenham servido para formular as regras ou exemplos do agente.

## Rodada recomendada

1. Selecionar documentos holdout e registrar a versão editorial humana de referência.
2. Executar a mesma amostra, com seed, modelo e versão registrados, para cada modelo candidato.
3. Para cada comentário da IA, rotular `correto`, `parcial` ou `incorreto`; registrar em `missed_issues` os erros humanos que a IA não encontrou.
4. Consolidar precisão, recall e F1 por agente e por modelo com `editorial-gold-metrics`.
5. Examinar falsos positivos e falsos negativos antes de qualquer alteração de prompt, escopo, heurística ou validação.

## Critérios de decisão

- Não ampliar escopo de um agente se a rodada aumentar falsos positivos sem ganho mensurável de recall.
- Preservar comentários localizados, uma questão por comentário, sem correções silenciosas.
- Alterar primeiro validação/escopo quando o erro for de fronteira; alterar prompt quando o escopo estiver correto, mas a interpretação da LLM não estiver.
- Toda alteração exige um teste de regressão para o caso que a motivou.

## Agente de coerência lógica

O agente `coerencia_logica` opera com política deliberadamente conservadora. Só aceita contradição explícita e local no texto. Seus achados usam `author_confirmation`: não corrigem nem escolhem a versão verdadeira do argumento. Ele não é avaliador de mérito econômico nem leitor visual de gráficos e tabelas.

Ele está disponível na interface como agente experimental e não integra a execução padrão até superar a avaliação holdout.

## Comentários de referência

No holdout, separar comentários de autores editoriais daqueles cujo autor começa por `Revisão:`. Os últimos são saídas geradas por uma rodada anterior e devem ser usados como regressão, não como métrica humana independente. A adjudicação humana deve rotular cada comentário atual como `correto`, `parcial`, `incorreto` ou `não verificável`.

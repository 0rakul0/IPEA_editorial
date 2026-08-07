# Relatório de comparação do holdout: GPT-4o mini x GPT-5.1

Data da execução: 6 de agosto de 2026.

## Objetivo e desenho

Foram executados os mesmos quatro documentos do holdout que possuem versão com comentários, usando a configuração calibrada do ipeaREV:

- sete agentes ativos por padrão;
- agente de coerência lógica fora da execução padrão (experimental);
- correção da heurística de artigo após “todos/as”;
- filtro estrutural para não classificar rótulos de figuras como títulos;
- mesma comparação por bloco/parágrafo do original.

Cinco dos nove documentos do diretório de holdout não tinham uma versão comentada e, por isso, não integram esta comparação.

## Resultado agregado

| Métrica | GPT-4o mini | GPT-5.1 | Variação |
|---|---:|---:|---:|
| Comentários emitidos pela IA | 195 | 235 | +40 |
| Blocos em comum com a referência | 30 | 43 | +13 |
| Blocos somente humanos | 42 | 29 | -13 |
| Blocos somente IA | 126 | 120 | -6 |
| Cobertura da IA sobre blocos humanos | 41,7% | 59,7% | +18,0 p.p. |
| Alinhamento dos blocos da IA | 19,2% | 26,4% | +7,2 p.p. |

**Conclusão:** há ganho do GPT-5.1 nesta rodada. Ele alcançou mais 13 blocos apontados pela referência e reduziu os blocos humanos não alcançados de 42 para 29. O ganho, porém, veio acompanhado de mais 40 comentários; portanto, a precisão editorial ainda precisa ser medida por adjudicação humana, e não apenas pelo encontro no mesmo bloco.

## Resultado por documento

| Documento | Cobertura GPT-4o mini | Cobertura GPT-5.1 | Alinhamento GPT-4o mini | Alinhamento GPT-5.1 | Leitura |
|---|---:|---:|---:|---:|---|
| LV Monitoramento e avaliação | 0,0% | 0,0% | 0,0% | 0,0% | Nenhum dos modelos recuperou os blocos marcados; ambos geraram comentários extras. |
| BMT 80 ES2 | 27,3% | 36,4% | 16,7% | 12,9% | GPT-5.1 encontrou um bloco humano adicional, mas dobrou o volume de comentários e reduziu o alinhamento. |
| PPP 71 A7 | 37,5% | 37,5% | 4,4% | 5,2% | Resultado praticamente estável; o GPT-5.1 emitiu quatro comentários a menos. |
| TD Expansão da área agrícola | 51,1% | 76,6% | 53,3% | 73,5% | Ganho expressivo: +12 blocos em comum, com melhora simultânea de cobertura e alinhamento. |

## Interpretação

O GPT-5.1 é o melhor candidato entre os dois para a próxima rodada de avaliação, mas ainda não deve ser tratado como substituto do revisor. O resultado agregado é sustentado principalmente pelo TD; os três outros documentos mostram desempenho estável, nulo ou com excesso de comentários.

Há uma limitação crítica no gabarito atual: parte dos comentários presentes nos documentos marcados foi criada por rodadas anteriores de IA. A base de adjudicação separa esses casos dos comentários de origem editorial humana. Assim, o próximo passo é rotular cada linha como correto, parcial, incorreto ou nao_verificavel e calcular precisão, recall e F1 somente contra a referência humana adjudicada.

## Comparação lado a lado por bloco

Além das métricas agregadas, foram reunidos os blocos dos documentos marcados com a presença ou ausência de comentário de cada fonte. Esta é uma comparação de localização do comentário no mesmo bloco; ela não presume que dois comentários no mesmo bloco tenham necessariamente o mesmo diagnóstico.

| Situação no bloco | Quantidade | Leitura |
|---|---:|---|
| Marcado + GPT-4o mini + GPT-5.1 | 29 | Ambos os modelos alcançaram um bloco marcado. |
| Marcado + somente GPT-5.1 | 14 | Ganho de cobertura exclusivo do GPT-5.1. |
| Marcado + somente GPT-4o mini | 1 | Único caso em que o GPT-4o mini alcançou um bloco marcado e o GPT-5.1 não. |
| Somente marcado | 28 | Problema registrado nos marcados que nenhum modelo alcançou. |
| GPT-4o mini + GPT-5.1, sem marca | 90 | Convergência de modelos, mas não necessariamente um erro real. |
| Somente GPT-5.1, sem marca | 30 | Divergência: alerta adicional do GPT-5.1. |
| Somente GPT-4o mini, sem marca | 36 | Divergência: alerta adicional do GPT-4o mini. |

O saldo favorável do GPT-5.1 vem diretamente dos 14 blocos marcados que ele alcançou sozinho, contra apenas 1 bloco alcançado somente pelo GPT-4o mini.

## Exemplos de convergência e divergência

| Tipo | Trecho / documento | Marcado | GPT-4o mini | GPT-5.1 | Leitura |
|---|---|---|---|---|---|
| Convergência útil dos três | BMT 80 ES2, citação “Chayanov (1991)” | Solicita a referência completa. | Indica que “Chayanov (1991)” não tem correspondência clara na lista final. | Faz o mesmo alerta de correspondência bibliográfica. | Mesma necessidade editorial, embora o humano tenha formulação mais geral. |
| Ganho do GPT-5.1 | TD Expansão da área agrícola, “STABILE et al, 2020” | “Correção: STABILE et al., 2020”. | Sem comentário. | Solicita o ponto em “et al.” e propõe “STABILE et al., 2020”. | Correção textual idêntica; a marca existente é de rodada anterior de IA, portanto este caso deve ser adjudicado, não contado como prova humana independente. |
| Falha comum | LV Monitoramento e avaliação, citação a Hill (2016) | Pede dados bibliográficos completos para a lista de referências. | Sem comentário. | Sem comentário. | Lacuna de cobertura de referências que persiste nos dois modelos. |
| Convergência sem confirmação editorial | LV Monitoramento e avaliação, título do capítulo | Sem comentário. | Emite alerta tipográfico contraditório, dizendo que a formatação está correta. | Pede ajuste de alinhamento/recuo. | Os modelos convergem no bloco, mas divergem no conteúdo e não há marca humana; não deve ser considerado acerto. |
| Divergência entre modelos | LV Monitoramento e avaliação, parágrafo sobre “instrumentos de política externa” | Sem comentário. | Sugere alteração de concordância em “figura”. | Sem comentário. | O alerta exclusivo do GPT-4o mini parece questionável: “o desempenho ... figura” tem núcleo singular. É candidato a falso positivo. |

Esses exemplos mostram por que a métrica de blocos é apenas uma triagem. O caso do título de capítulo entra na convergência de localização, mas não equivale a concordância semântica nem a uma correção confirmada.

## Recomendação operacional

1. Usar GPT-5.1 em uma amostra adicional, mantendo a mesma configuração e registrando custo/tempo.
2. Adjudicar primeiro os 31 itens que têm referência editorial humana na base atual.
3. Ajustar os agentes de maior volume de falsos positivos antes de ampliar o uso externo.
4. Só promover o modelo para a aplicação de testes quando a adjudicação confirmar que o ganho de cobertura não veio principalmente de comentários irrelevantes.

## Artefatos

- Resultado GPT-4o mini: .tmp/holdout_gpt4o_mini/resultado_calibrado.json
- Resultado GPT-5.1: .tmp/holdout_gpt51/resultado_gpt51.json
- Base para adjudicação: .tmp/holdout_gpt4o_mini/adjudicacao.json

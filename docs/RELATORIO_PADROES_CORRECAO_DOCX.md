# Relatório de Extração de Padrões de Correção a partir de DOCX

**Projeto:** `lang_IPEA_editorial`  
**Data:** 16 de julho de 2026  
**Finalidade:** documento de apresentação sobre o uso de arquivos `.docx` como base para inferência de padrões editoriais e calibração do pipeline de revisão.
**Quantidade de documentos analisados:** 90 documentos editoriais reais.

## 1. Resumo executivo

O projeto `lang_IPEA_editorial` utiliza arquivos `.docx` não apenas como insumo para revisão, mas também como fonte de evidência sobre o comportamento editorial humano. Em especial, conjuntos compostos por versões como `original`, `com marcas`, `sem marcas` e `para diagramar` permitem identificar quais intervenções foram sugeridas, quais foram efetivamente aplicadas e quais padrões de correção tendem a se repetir.

Essa estratégia transformou o acervo de revisões em uma base prática de aprendizado editorial. Em vez de depender apenas de regras abstratas ou de exemplos inventados, o sistema passou a se apoiar em material revisado em contexto real de produção.

Neste recorte já consolidado, foram analisados **90 documentos editoriais reais**, dos quais se extraíram **17.353 exemplos** e **1.559 casos confirmados na versão final**.

Como resultado, o projeto hoje combina:

- extração estruturada do documento;
- revisão por agentes especializados;
- aprendizado a partir de versões editoriais reais;
- validação conservadora para reduzir falso positivo;
- exportação em DOCX comentado e relatório JSON.

## 2. Objetivo do uso dos DOCX

O uso dos `.docx` teve três objetivos centrais:

1. identificar padrões recorrentes de intervenção editorial humana;
2. transformar esses padrões em regras, prompts, heurísticas e critérios de validação;
3. avaliar se os comentários produzidos pelos agentes se aproximam das correções efetivamente consolidadas nas versões finais.

Em termos práticos, os DOCX revisados passaram a funcionar como um conjunto de evidências editoriais para orientar a calibragem do sistema.

## 3. Como a abordagem funciona

O fluxo de aprendizado editorial foi implementado para ler pastas com ciclos reais de revisão, normalmente contendo:

- uma versão `(original).docx`;
- uma versão `(com marcas).docx`, quando disponível;
- uma versão `(sem marcas).docx`;
- uma versão `(para diagramar).docx`;
- eventualmente um `.pdf` final.

O script principal dessa etapa é [editorial_lab.py](D:/github/lang_IPEA_editorial/scripts/editorial_lab.py).

Esse script extrai e cruza:

- parágrafos do documento original;
- comentários humanos ancorados no Word;
- inserções e exclusões rastreadas;
- similaridade entre trechos das versões revisadas e finais;
- confirmação de que uma intervenção foi de fato absorvida no resultado final;
- classificação do exemplo por agente e por tipo de ação editorial.

Os exemplos extraídos são classificados principalmente nestes estados:

- `confirmed_final`: a mudança foi confirmada na versão final;
- `observed_change`: houve mudança observada entre versões;
- `unresolved_or_not_applied`: o comentário ou indício de revisão não pôde ser confirmado como mudança final.

Essa separação é importante porque nem todo comentário humano vira alteração consolidada, e nem toda diferença entre versões representa uma regra editorial útil.

## 4. Papel dos DOCX no pipeline atual

Os DOCX influenciam o projeto em dois níveis diferentes.

### 4.1. Como entrada operacional

Na operação normal, o sistema recebe `.docx`, `.pdf` ou `normalized_document.json`, extrai a estrutura do texto, distribui trechos aos agentes e devolve:

- um `.docx` comentado;
- um relatório `.json`;
- um arquivo de diagnósticos de execução.

Esse comportamento está descrito em [README.md](D:/github/lang_IPEA_editorial/README.md) e [ESTADO_ATUAL_EDITORIAL.md](D:/github/lang_IPEA_editorial/docs/ESTADO_ATUAL_EDITORIAL.md).

### 4.2. Como base de aprendizado editorial

Além da execução normal, os DOCX revisados servem para ensinar ao sistema:

- o que os revisores realmente comentam;
- que tipos de correção costumam ser aceitos;
- quais sugestões são estruturais, autorais ou apenas de produção;
- onde há maior risco de falso positivo;
- quais ajustes devem ser tratados com mais conservadorismo.

## 5. Evidências no código

Os principais pontos do projeto ligados a essa estratégia são:

- [scripts/editorial_lab.py](D:/github/lang_IPEA_editorial/scripts/editorial_lab.py)  
  Extrai conhecimento editorial de conjuntos de versões, gera base de exemplos e avalia relatórios dos agentes contra esse material.

- [src/editorial_docx/prompts/prompt.py](D:/github/lang_IPEA_editorial/src/editorial_docx/prompts/prompt.py)  
  Carrega diretrizes auxiliares, inclusive trechos do documento `Agente IA Editorial (tarefas) (1).docx`, e as injeta nos prompts dos agentes.

- [docs/ESTADO_ATUAL_EDITORIAL.md](D:/github/lang_IPEA_editorial/docs/ESTADO_ATUAL_EDITORIAL.md)  
  Registra o comportamento editorial consolidado do sistema após a calibração.

- [README.md](D:/github/lang_IPEA_editorial/README.md)  
  Documenta o pipeline geral, os agentes, as saídas e os fluxos de avaliação.

## 6. Decisões editoriais já consolidadas

Com base na calibração feita sobre o acervo revisado, o sistema consolidou algumas decisões importantes.

### 6.1. Não há correção silenciosa na execução padrão

O projeto foi configurado para não alterar o texto sem transparência. Em vez de corrigir automaticamente vírgulas, referências, títulos ou formatação, ele produz comentários visíveis no DOCX e no JSON.

Isso aumenta a auditabilidade e torna o relatório final coerente com a entrega efetiva ao usuário.

### 6.2. O comentário editorial precisa trazer diagnóstico e correção

O formato atual privilegia dois elementos:

- uma explicação curta do problema;
- uma linha objetiva iniciada por `Correção:`.

Essa escolha aproxima a saída do sistema do padrão de revisão humano observado no material analisado.

### 6.3. O sistema evita inferência especulativa

A calibragem reforçou uma postura conservadora:

- não inventar dados bibliográficos;
- não reescrever trechos por preferência estilística;
- não alterar citações diretas;
- não converter hipótese em correção objetiva;
- não confundir problema autoral com erro formal local.

## 7. Base empírica já registrada

Segundo a documentação operacional da skill e do projeto, o processo de aprendizado editorial já reuniu:

- **90 documentos reais** processados em lote;
- **17.353 exemplos** extraídos;
- **1.559 exemplos confirmados na versão final**.

Perfis documentais cobertos:

- Texto para Discussão;
- capítulo de livro;
- boletim BPS;
- boletim BMT;
- boletim Radar;
- nota técnica;
- artigo PPE;
- artigo PPP.

Esse conjunto é relevante porque cobre gêneros editoriais diferentes, com exigências distintas de estrutura, referências, tabelas e tom de escrita.

## 8. Padrões mais recorrentes identificados

### 8.1. Gramática e ortografia

Foi a categoria com maior volume de exemplos. Os padrões mais recorrentes incluem:

- paralelismo sintático e semântico em enumerações;
- ajustes de regência e crase;
- expansão de abreviações;
- padronização de siglas;
- inserção de elementos faltantes para clareza local;
- pequenos reparos linguísticos objetivos.

Interpretação: esse agente precisa cobrir muito terreno, mas também é o mais suscetível à oscilação de LLM. Por isso, o aprendizado a partir de DOCX ajuda a limitar excesso de criatividade e a reforçar o foco em erro local verificável.

Exemplos:

- `Variação Patrimonial (Ano - %)` -> `Variação patrimonial (% ao ano)`
- `Apoio limitado` -> `Apoiar de modo limitado`
- `ICTs em ascensão` -> `TICs em ascensão`
- `a aprovação` -> `pela aprovação`
- `Coef.` -> `Coeficiente`
- `1995/00` -> `1995-2000`

### 8.2. Referências

Foi uma das áreas com maior número de exemplos confirmados. Os padrões mais frequentes incluem:

- links quebrados ou genéricos;
- referências incompletas;
- ausência de `Disponível em:` e `Acesso em:`;
- uso desatualizado de `no prelo`;
- inconsistências localmente verificáveis em autoria, editora, local ou paginação;
- separação inadequada entre referências e bibliografia complementar.

Interpretação: a evidência empírica reforçou uma política de rigor com ABNT, mas sem invenção de dados ausentes.

Exemplos:

- `No prelo` -> `v. 53, n. 2` quando a publicação já havia saído
- `AZEVEDO, B. ...` -> `AZEREDO, B. ...` quando o erro era localmente verificável
- referência online sem `Acesso em:` -> solicitação objetiva de inclusão da data de acesso
- URL que levava a página genérica ou inválida -> pedido de novo endereço válido
- `Fonte: elaboração própria a partir de Bartholo, Paiva e Souza (2023)` -> pedido de referência completa da obra citada

### 8.3. Tabelas e figuras

Os padrões mais recorrentes incluem:

- separação entre identificador e título;
- presença de linha própria para `Fonte:` ou `Elaboração:`;
- renumeração coerente;
- padronização de notas;
- correção de inconsistência entre título e conteúdo do bloco visual.

Interpretação: trata-se de uma área em que o comportamento editorial humano é bastante regular, o que favorece heurísticas mais estáveis.

Exemplos:

- `Nota: ***, ** e * denotam coeficientes...` -> `Obs.: Significância: *** 0,01; ** 0,05; e * 0,1.`
- legenda com identificador e título na mesma linha -> separação entre `GRÁFICO 15` e o título descritivo na linha abaixo
- bloco visual sem linha própria de fonte -> inclusão de `Fonte:` ou `Elaboração:` abaixo do elemento
- título com período incompatível com os dados exibidos -> ajuste do ano no título para coincidir com o gráfico ou a tabela

### 8.4. Estrutura

Os principais padrões incluem:

- numeração hierárquica de seções;
- paralelismo formal entre títulos;
- posição do ano ao final do título, entre parênteses;
- distinção entre anexo e apêndice.

Interpretação: embora o agente de estrutura continue disponível, ele saiu da execução padrão por exigir maior cautela em casos menos inequívocos.

Exemplos:

- `O padrão global das alianças em 2007 do ponto de vista dos Estados Unidos` -> `O padrão global das alianças do ponto de vista dos Estados Unidos (2007)`
- `Setor agropecuário` -> `6.1 Setor agropecuário` quando o documento exigia hierarquia numerada coerente
- uso de `Anexo` para material autoral -> ajuste para `Apêndice`, conforme a natureza do conteúdo

### 8.5. Tipografia

Os casos confirmados são menos numerosos, mas recorrentes em:

- caixa alta e baixa;
- negrito e itálico;
- alinhamento;
- recuo;
- espaçamento.

Interpretação: tipografia foi separada de gramática para reduzir mistura indevida entre forma e conteúdo.

Exemplos:

- título em caixa incompatível com o padrão da publicação -> ajuste de maiúsculas e minúsculas
- uso indevido de negrito em subtítulo ou legenda -> remoção do destaque
- recuo ou alinhamento destoante em bloco equivalente -> normalização para o padrão do documento
- espaçamento antes ou depois de título inconsistente com a mesma família de seção -> ajuste para padronização visual

### 8.6. Sinopse e abstract

Os padrões incluem:

- ausência de código JEL;
- excesso de palavras-chave;
- desalinhamento entre palavras-chave e keywords;
- padronização de caixa e listagem.

Interpretação: é uma frente útil, mas mais estreita e menos frequente no conjunto analisado.

Exemplos:

- `Palavras-chave: Bioeconomia; Desmatamento Zero; Amazônia; Uso da Terra; Modelo de Equilíbrio Geral; Desenvolvimento Sustentável.` -> redução para cinco palavras-chave e padronização de caixa
- ausência de `JEL` em artigo com perfil compatível -> solicitação de inclusão do código
- desalinhamento entre `Palavras-chave` e `Keywords` -> pedido de harmonização objetiva entre as duas listas

## 9. Impactos sobre o desenho do sistema

O uso dos DOCX como base de padrões de correção gerou impactos concretos no projeto.

### 9.1. Melhoria de precisão editorial

O sistema deixou de depender apenas da formulação teórica dos prompts e passou a se apoiar em correções reais, o que melhora a aderência ao comportamento editorial observado.

### 9.2. Redução de falso positivo

A análise de versões reais ajudou a distinguir:

- erro formal objetivo;
- pedido de validação ao autor;
- demanda de produção editorial;
- simples diferença de estilo ou decisão contextual.

Essa distinção aparece hoje no campo `action_type`, que separa itens como:

- `auto_fix_candidate`;
- `author_confirmation`;
- `production_request`.

### 9.3. Maior rastreabilidade

Como a revisão final é exportada em DOCX comentado e JSON, o sistema preserva vínculo entre:

- trecho problemático;
- diagnóstico;
- sugestão;
- agente responsável;
- evidência de execução.

### 9.4. Base para avaliação contínua

O projeto passou a ter uma trilha concreta para comparar:

- o que humanos corrigiram;
- o que os agentes sugerem;
- o que foi aceito na versão final.

Isso abre espaço para avaliação contínua de precisão, recall e lacunas por agente.

## 10. Limitações observadas

Apesar dos ganhos, a abordagem tem limites claros.

### 10.1. Qualidade do DOCX afeta a extração

Quando o documento está muito desformatado, a extração estrutural pode perder contexto, o que prejudica escopo, ancoragem e comparação entre versões.

### 10.2. Nem toda mudança entre versões é uma regra editorial

Parte das diferenças pode refletir:

- diagramação;
- reorganização tardia do texto;
- decisão autoral;
- adaptação para publicação;
- intervenção não explicitada em comentário.

Por isso, o projeto distingue mudanças confirmadas de mudanças apenas observadas.

### 10.3. Agentes dependentes de LLM ainda oscilam

Mesmo com prompts, validações e heurísticas, agentes como `gramatica_ortografia` e `sinopse_abstract` tendem a variar mais entre execuções do que áreas mais ancoradas em estrutura local.

### 10.4. A calibragem é institucionalmente situada

O comportamento consolidado do sistema foi calibrado segundo critérios editoriais do Ipea. Isso é uma força para aderência institucional, mas significa que o modelo não deve ser lido como revisor genérico de mercado sem adaptações.

## 11. Conclusão

O uso dos `.docx` como fonte de padrões de correção foi um passo estruturante para o amadurecimento do `lang_IPEA_editorial`. Essa abordagem permitiu transformar revisões humanas reais em conhecimento operacional para o pipeline, com efeitos diretos na formulação dos prompts, nas heurísticas, na validação e no desenho das saídas.

Em termos de apresentação, a principal mensagem é a seguinte: o sistema não foi calibrado apenas “por instrução”, mas por observação sistemática de como revisores humanos atuam sobre documentos reais. Isso elevou a consistência editorial do projeto, aumentou a rastreabilidade das decisões e criou uma base concreta para avaliação e aperfeiçoamento contínuos.

## 12. Referências internas do projeto

- [README.md](D:/github/lang_IPEA_editorial/README.md)
- [ESTADO_ATUAL_EDITORIAL.md](D:/github/lang_IPEA_editorial/docs/ESTADO_ATUAL_EDITORIAL.md)
- [editorial_lab.py](D:/github/lang_IPEA_editorial/scripts/editorial_lab.py)
- [prompt.py](D:/github/lang_IPEA_editorial/src/editorial_docx/prompts/prompt.py)

---
name: revisao-editorial-ipea
description: Executar, diagnosticar, avaliar e aperfeiçoar a revisão editorial Ipea de documentos DOCX, PDF ou normalized_document.json usando o pipeline lang_IPEA_editorial
---

# Revisão Editorial Ipea

## Quando usar

Quando o usuário pedir revisão editorial, Texto para Discussão (TD), DOCX comentado, relatório JSON, diagnóstico, dataset ouro, comparação com revisão humana, aprendizado a partir de versões do documento, avaliação de falsos positivos, calibragem de agentes (gramática, sinopse, tabelas, estrutura, referências ABNT, tipografia).

## Fluxo principal

1. Localizar a raiz do projeto (`pyproject.toml` com o pacote `lang-ipea-editorial`).
2. Executar revisão com `uv run editorial-docx <caminho/docs>` ou via wrapper.
3. Conferir código de saída e artefatos gerados.
4. Resumir: documento processado, total de comentários, agentes com falha, artefatos produzidos.

## Pipeline (visão geral)

1. `document_loader.py` → carrega DOCX/PDF/JSON
2. `normalized_document.py` → blocos, seções, referências
3. `pipeline/scope.py` → seleciona escopo por agente
4. `pipeline/context.py` + `runtime.py` + `orchestrator.py` → executa LLM + heurísticas (até 3 agentes em paralelo)
5. `pipeline/validation.py` + `consolidation.py` + `coordinator.py` → filtra e deduplica
6. `docx_utils.py` / CLI / Streamlit → exporta

Fachada pública: `src/editorial_docx/graph_chat.py`

Fontes de verdade:
- Prompts: `src/editorial_docx/prompts/`
- Escopos: `src/editorial_docx/agents/scopes/`
- Heurísticas: `src/editorial_docx/agents/heuristics/`
- Validação: `src/editorial_docx/agents/validation/`
- Config: `src/editorial_docx/config.py` e `llm.py`

## Agentes ativos (execução padrão)

| Agente | Sigla | Escopo |
|---|---|---|
| `sinopse_abstract` | `sin` | SINOPSE, ABSTRACT, Keywords, JEL |
| `gramatica_ortografia` | `gram` | Ortografia, concordância, regência, crase, pontuação |
| `tabelas_figuras` | `tab` | Legendas de tabelas, figuras, quadros, gráficos |
| `referencias` | `ref` | Lista bibliográfica (ABNT NBR 6023/10520) |
| `tipografia` | `tip` | Caixa, negrito, itálico, alinhamento, recuo, espaçamento |

`estrutura` (`est`) disponível mas fora da execução padrão.

## Saída esperada (contrato)

```json
{
  "agent": "gram",
  "category": "inconsistency",
  "message": "Diagnóstico curto e natural do problema local.",
  "paragraph_index": 10,
  "issue_excerpt": "fragmento com problema",
  "suggested_fix": "fragmento corrigido"
}
```

- DOCX: comentário ancorado no trecho, autor = `Revisão: <sigla>`, formato diagnóstico + `Correção: ...`
- JSON relatório + JSON diagnostics + DOCX comentado

## Dataset ouro (avaliação humana)

Gerar scaffold a partir do relatório JSON:
```powershell
uv run editorial-gold-dataset `
  "<relatorio>.json" `
  --output "testes/dataset_ouro/seed_<nome>.json" `
  --source-document "<docx>" `
  --model-name "gpt-5.2"
```

Preencher manualmente `label` (`correto`/`parcial`/`incorreto`) e `severity` (`alta`/`media`/`baixa`). Registrar em `missed_issues` o que o modelo não achou (`faltou`).

Consolidar métricas:
```powershell
uv run editorial-gold-metrics `
  "testes/dataset_ouro" `
  --output "testes/dataset_ouro/metricas_reais.json"
```

Métricas: VP (corretos), VP_parcial, FP (incorretos), FN (faltantes), precisão, recall, F1 — quebrado por agente/modelo.

## Aprendizado com versões editoriais

Dado um diretório com arquivos como `(original).docx`, `(com marcas).docx`, `(sem marcas).docx`, `(para diagramar).docx`, `.pdf` final:
```powershell
uv run python scripts/editorial_lab.py learn `
  "<pasta-editorial>" `
  --out-dir ".tmp/aprendizado"
```

Estados: `confirmed_final`, `observed_change`, `unresolved_or_not_applied`.

Avaliar agentes contra gabarito extraído:
```powershell
uv run python scripts/editorial_lab.py evaluate `
  --knowledge ".tmp/aprendizado/editorial_knowledge.json" `
  --report "<relatorio-agentes>.json" `
  --out-dir ".tmp/avaliacao"
```

## Conhecimento editorial extraído de 90 documentos reais

Processamento de 90 pastas editoriais com 17.353 exemplos, sendo **1.559 confirmados na versão final**.

Perfis cobertos: TD (12), capítulo de livro (21), boletim BPS (13), boletim BMT (8), boletim Radar (9), nota técnica (7), artigo PPE (7), artigo PPP (13).

### gramatica_ortografia (12.325 exemplos, 523 confirmados)

Padrões mais frequentes:
- **Parallelismo**: ajuste de termos coordenados para mesma estrutura gramatical/tipo de palavra
- **Tom/registro**: substituição de termos coloquiais por formais; remoção de trechos que destoam
- **Clareza**: inserção de especificadores faltantes (siglas por extenso, preposições, artigos)
- **Abreviações**: expansão de abreviaturas (Coef. → Coeficiente)
- **Siglas**: uniformização (ICTs → TICs; BPD → Benefício...)
- **Dados bibliográficos**: correção de sobrenome (Azevedo → Azeredo), local/editora (Mcgraw-Hill → Addison-Wesley)

Exemplos reais:
- ❌ "Variação Patrimonial (Ano - %)" → ✅ "Variação patrimonial (% ao ano)"
- ❌ "Apoio limitado" → ✅ "Apoiar de modo limitado" (paralelismo em enumeração)
- ❌ "ICTs em ascensão" → ✅ "TICs em ascensão" (sigla oficial)
- ❌ "a aprovação" → ✅ "pela aprovação" (regência nominal)
- ❌ "Coef." → ✅ "Coeficiente" (abreviação por extenso)
- ❌ "1995/00" → ✅ "1995-2000" (padronização de intervalo)

### referencias (2.190 exemplos, 789 confirmados)

Padrões mais frequentes:
- **Links quebrados/genéricos**: URL que redireciona para página genérica, domínio à venda ou página não encontrada
- **Referência incompleta**: falta volume, número, editora, local, "Disponível em:", "Acesso em:"
- **"No prelo" desatualizado**: artigo já publicado precisa de volume/número/páginas reais
- **Publicação não localizada**: [s.v.], [s.n.] quando não se encontra o periódico
- **Padronização ABNT**: sobrenome, edição, subtítulo, local: editora, coleção
- **Segregação seção**: Referências vs. Bibliografia complementar

Exemplos reais:
- ❌ "No prelo" → ✅ "v. 53, n. 2" (artigo já publicado)
- ❌ "AZEVEDO, B. Políticas públicas..." → ✅ "AZEREDO, B. Políticas públicas..." (sobrenome corrigido)
- ❌ "MORGENTHAU, H. ... Editora Universidade De Brasília, [1948]2003." → ✅ "Editora UnB, [1948] 2003. Disponível em: https://..."
- ❌ "Site resultou em página não encontrada" → ✅ "Favor encaminhar novo endereço válido"
- ❌ "Fonte: elaboração própria a partir de Bartholo, Paiva e Souza (2023)" → ✅ "Fontes: Bartholo, Paiva e Souza (2023) e respectiva legislação dos programas." (pedir referência completa)

### tabelas_figuras (1.778 exemplos, 171 confirmados)

Padrões mais frequentes:
- **Título descritivo vs. informativo**: legenda mínima; detalhes nos eixos/legendas
- **Separação de gráficos**: desmembrar gráfico único em múltiplos quando há dados heterogêneos
- **Renumeração**: adequar numeração à sequência do documento
- **Nota de significância**: padronizar "Nota: ***, ** e *" → "Obs.: Significância: *** 0,01; ** 0,05; * 0,1"
- **Identificador vs. subtítulo**: "GRÁFICO 15" em linha própria, título na linha abaixo
- **Período inconsistente**: ano no título não corresponde ao do gráfico

Exemplos reais:
- ❌ "Nota: ***, ** e * denotam coeficientes..." → ✅ "Obs.: Significância: *** 0,01; ** 0,05; e * 0,1."
- ❌ "Atividades" → ✅ "A.2B – Atividades" (quadro desmembrado)
- ❌ "Elaboração própria" → ✅ "GRÁFICO 15" (ilustração sem identificador)
- ❌ "Figura 4: Dispersão dos custos operacionais unitários (per capita e por ligação) conforme o nível de cobertura ponderada dos serviços" → ✅ "Dispersão dos custos operacionais unitários per capita conforme o nível de cobertura ponderada dos serviços (2022)" (gráficos separados)

### estrutura (746 exemplos, 62 confirmados)

Padrões mais frequentes:
- **Numeração de seções**: todas as seções Ipea devem ser numeradas hierarquicamente
- **Anexo vs. Apêndice**: anexo = material de terceiros; apêndice = material do autor
- **Paralelismo em títulos**: itens de enumeração devem ter mesma conotação (ex.: título "Desafios" → itens devem ser negativos)
- **Posição do ano**: em títulos, ano deve vir ao final entre parênteses

Exemplos reais:
- ❌ "O padrão global das alianças em 2007 do ponto de vista dos Estados Unidos" → ✅ "O padrão global das alianças do ponto de vista dos Estados Unidos (2007)"
- ❌ "Revisão de literatura" → ✅ "1B – Escala ampliada" (título incluído pela revisão)
- ❌ "Setor agropecuário" → ✅ "6.1 Setor agropecuário" (numeração adicionada)

### tipografia (101 exemplos, 9 confirmados)

Padrões: correção de caixa alta/baixa, remoção de negrito indevido, padronização de recuo e alinhamento.

### sinopse_abstract (213 exemplos, 5 confirmados)

Padrões:
- **JEL**: solicitar código JEL quando ausente (artigo)
- **Palavras-chave**: limitar a 5; padronizar minúsculas exceto nomes próprios
- **Keywords**: alinhamento com palavras-chave em português

Exemplos reais:
- ❌ "Palavras-chave: Bioeconomia; Desmatamento Zero; Amazônia; Uso da Terra; Modelo de Equilíbrio Geral; Desenvolvimento Sustentável." → ✅ "Palavras-chave: bioeconomia; desmatamento zero; Amazônia; uso da terra; modelo de equilíbrio geral." (limitar a 5)
- ❌ "Introdução" (no lugar do JEL) → ✅ "JEL: I21; J24; D10."

## Instalação

A skill pode ser instalada em múltiplos paths para funcionar com diferentes assistentes:

```powershell
# Escopo repo (apenas neste projeto)
.\install.ps1 -Scope repo

# Escopo user (global, ~/.config/opencode ~/.claude ~/.agents)
.\install.ps1 -Scope user
```

```bash
# Linux/macOS
bash install.sh repo
bash install.sh user
```

Paths instalados (repo): `.opencode/skills/`, `.claude/skills/`, `.agents/skills/`

## Modos de acesso

O sistema pode ser usado de três formas:

### 1. CLI (terminal)
```bash
uv run editorial-docx <arquivo.docx> --output-docx revisado.docx
```

### 2. Interface Web (Streamlit)
```bash
uv run streamlit run streamlit_app.py
```

### 3. AI Skill (assistente)
A skill carregada por este arquivo permite que o assistente execute o pipeline automaticamente quando solicitado. O assistente usa os comandos abaixo:

**Revisão direta:**
```bash
uv run editorial-docx "<arquivo.docx>"
```

**Com preflight (testa LLM antes):**
```bash
uv run python scripts/run_review.py "<arquivo.docx>" --project-root .
```

**Batch aprendizado de múltiplas pastas editoriais:**
```powershell
uv run python scripts/editorial_lab.py batch-learn "<pasta-raiz>" --out-dir ".tmp/aprendizado"
```

**Dataset ouro:**
```powershell
uv run editorial-gold-dataset "<relatorio>.json" --output "testes/dataset_ouro/seed_<nome>.json"
```

## Guardrails editoriais

- Não fazer correções silenciosas fora do que o pipeline aplica.
- Não inventar dados bibliográficos.
- Não tratar preferência de estilo como erro.
- Não alterar citações diretas.
- Preservar entrada; gerar saídas separadas.
- `issue_excerpt` mínimo e ancorável; `suggested_fix` concreto.
- Um problema por comentário.
- Rejeitar comentários vagos, especulativos, redundantes ou fora de escopo.

## Como alterar o pipeline

1. Identificar a camada correta: prompt, escopo, heurística, validação, consolidação ou exportação.
2. Fazer a menor alteração coerente.
3. Adicionar ou ajustar teste de regressão.
4. Rodar verificação:
   ```bash
   uv run pytest testes/test_architecture_modular.py testes/test_graph_chat.py -q
   uv run python -m compileall src/editorial_docx streamlit_app.py
   ```
5. Explicar o efeito sobre falsos positivos e falsos negativos.

## Testes
```bash
uv run pytest testes/test_llm.py testes/test_architecture_modular.py testes/test_graph_chat.py -q
```

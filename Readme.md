# lang_IPEA_editorial

Sistema de revisão editorial para `.docx`, `.pdf` e `normalized_document.json`, com execução via CLI e interface web em Streamlit.

O projeto combina:
- extração estruturada do documento;
- revisão por agentes especializados;
- validação e consolidação final;
- exportação em DOCX comentado e JSON.

## Estrutura Atual

### Pastas de dados

- `input_data/`
  Aqui ficam os arquivos de entrada que o usuário quer revisar.
- `output_data/`
  Aqui ficam os artefatos gerados: DOCX comentado, relatório JSON, diagnóstico e `normalized_document.json`.

### Núcleo da aplicação

- `src/editorial_docx/config.py`
  Configurações compartilhadas do projeto: caminhos, timeout, retries, limites de batch e constantes globais.
- `src/editorial_docx/document_loader.py`
  Carrega DOCX, PDF ou `normalized_document.json`.
- `src/editorial_docx/normalized_document.py`
  Gera e serializa o artefato independente de extração.
- `src/editorial_docx/pipeline/context.py`
  Prepara lotes, janelas e contexto progressivo.
- `src/editorial_docx/pipeline/scope.py`
  Define quais partes do documento cada agente deve revisar.
- `src/editorial_docx/pipeline/runtime.py`
  Executa chamadas LLM, retries, parsing de saída e coordenador.
- `src/editorial_docx/pipeline/validation.py`
  Filtra comentários especulativos ou mal ancorados.
- `src/editorial_docx/pipeline/consolidation.py`
  Consolida comentários equivalentes.
- `src/editorial_docx/pipeline/orchestrator.py`
  Orquestra o pipeline de revisão.
- `src/editorial_docx/pipeline/coordinator.py`
  Gera a síntese final do resultado.
- `src/editorial_docx/graph_chat.py`
  Fachada compatível usada pelo restante do projeto e pelos testes.

### Agentes

- `src/editorial_docx/agents/user_reference_agent.py`
  Agente especial para comentários do usuário pedindo busca de referência.
- `src/editorial_docx/agents/heuristics/`
  Heurísticas organizadas por agente:
  - `grammar.py`
  - `synopsis.py`
  - `tables_figures.py`
  - `structure.py`
  - `typography.py`
  - `references.py`
  - `dispatch.py`

### Regras ABNT e matching bibliográfico

- `src/editorial_docx/abnt_normalizer.py`
- `src/editorial_docx/abnt_citation_parser.py`
- `src/editorial_docx/abnt_reference_parser.py`
- `src/editorial_docx/abnt_matcher.py`
- `src/editorial_docx/abnt_validator.py`
- `src/editorial_docx/abnt_rules.py`

## Fluxo do Código

```mermaid
flowchart TD
    A["Usuário envia DOCX, PDF ou normalized JSON"] --> B["document_loader.py"]
    B --> C["normalized_document.py<br/>extrai blocos, seções, comentários e referências"]
    C --> D["pipeline/scope.py<br/>seleciona o escopo por agente"]
    D --> E["pipeline/context.py<br/>monta lotes e janelas"]
    E --> F["pipeline/orchestrator.py"]
    F --> G["Agentes LLM + heurísticas"]
    G --> G1["agents/heuristics/*.py"]
    G --> G2["agents/user_reference_agent.py"]
    G --> H["pipeline/validation.py<br/>valida ancoragem e plausibilidade"]
    H --> I["pipeline/consolidation.py<br/>deduplica e consolida"]
    I --> J["pipeline/runtime.py + pipeline/coordinator.py<br/>síntese final"]
    J --> K["docx_utils.py / CLI / Streamlit"]
    K --> L["output_data/<artefatos>"]
```

## Fluxo de Referências

```mermaid
flowchart LR
    A["Texto do corpo"] --> B["abnt_citation_parser.py"]
    C["Lista de referências"] --> D["abnt_reference_parser.py"]
    B --> E["abnt_matcher.py"]
    D --> E
    E --> F["match exato"]
    E --> G["match provável<br/>ano divergente"]
    E --> H["match provável<br/>entrada colada/malformada"]
    E --> I["ausência real"]
    F --> J["review_heuristics.py -> comentário ou silêncio"]
    G --> J
    H --> J
    I --> J
```

## Execução

### Streamlit

```bash
streamlit run streamlit_app.py
```

O app:
- lê documentos de `input_data/`;
- permite subir novos arquivos direto para `input_data/`;
- salva artefatos em `output_data/`.

### CLI

```bash
python -m editorial_docx "D:\github\lang_IPEA_editorial\input_data\arquivo.docx"
```

Saídas padrão:
- `output_data/<nome>_normalized_document.json`
- `output_data/<nome>_output_<modelo>.relatorio.json`
- `output_data/<nome>_output_<modelo>.relatorio.diagnostics.json`
- `output_data/<nome>_output_<modelo>.docx`

## Configuração

As constantes centrais estão em:
- `src/editorial_docx/config.py`

Exemplos:
- diretórios de entrada e saída;
- modelo padrão;
- timeout;
- retries;
- tamanho de batch;
- overlap de micro-lotes.

As credenciais e provedores continuam sendo lidos do `.env`.

### Exemplo OpenAI

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.2
```

### Exemplo Ollama

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.1:8b
OLLAMA_API_KEY=ollama
```

### Exemplo compatível com OpenAI

```env
LLM_PROVIDER=openai_compatible
LLM_BASE_URL=http://servidor-interno/v1
LLM_MODEL=nome-do-modelo
LLM_API_KEY=token-opcional
```

## Organização dos Agentes

Hoje os agentes estão separados por responsabilidade:

- agentes de revisão editorial geral:
  - `sinopse_abstract`
  - `gramatica_ortografia`
  - `tabelas_figuras`
  - `estrutura`
  - `tipografia`
  - `referencias`
- agente especial:
  - `comentarios_usuario_referencias`

O código específico de cada agente foi concentrado em `src/editorial_docx/agents/heuristics/`.
Os módulos antigos `review_*.py` foram removidos; a estrutura vigente do projeto passa a ser `pipeline/`, `agents/`, `references/` e `io/`.

## Testes

Rodadas principais:

```bash
pytest testes/test_llm.py testes/test_architecture_modular.py testes/test_graph_chat.py -q
```

Validação de import e sintaxe:

```bash
python -m compileall src/editorial_docx streamlit_app.py
```

## Documentação Complementar

O estado editorial consolidado continua documentado em:

- `docs/ESTADO_ATUAL_EDITORIAL.md`

# Claude Code

Integração do `lang_IPEA_editorial` com Claude Code: `CLAUDE.md`, skill instalável, wrapper e referências dos agentes.

## 1. Instalar

Windows:

```powershell
irm https://claude.ai/install.ps1 | iex
```

macOS, Linux ou WSL:

```bash
curl -fsSL https://claude.ai/install.sh | bash
```

Alternativa:

```powershell
winget install Anthropic.ClaudeCode
```

Execute `claude` e autentique.

## 2. Preparar o projeto

Requisitos: Git, Python 3.10 ou superior e `uv`.

Instalar `uv` no Windows:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

No macOS/Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```bash
git clone https://github.com/0rakul0/IPEA_editorial.git
cd IPEA_editorial
uv sync --dev
```

Copie `.env.example` para `.env` e configure `LLM_PROVIDER`, `LLM_MODEL` e `LLM_API_KEY`.

## 3. Instalar a skill

Usuário:

```powershell
powershell -ExecutionPolicy Bypass -File .\claude-code\install.ps1 -Scope user
```

```bash
bash claude-code/install.sh user
```

Repositório:

```powershell
powershell -ExecutionPolicy Bypass -File .\claude-code\install.ps1 -Scope repo
```

```bash
bash claude-code/install.sh repo
```

O escopo de usuário usa `~/.claude/skills`; o de repositório usa `.claude/skills`.

Copie ou incorpore `claude-code/CLAUDE.md` no `CLAUDE.md` da raiz. Não sobrescreva instruções existentes sem revisão.

## 4. Usar

```bash
claude
```

```text
/revisao-editorial-ipea D:\documentos\texto.docx gere também o diagnóstico JSON
```

## 5. Validar

Procure `/revisao-editorial-ipea` no menu `/`. Abra nova sessão se necessário.

Teste sem LLM:

```bash
python claude-code/.claude/skills/revisao-editorial-ipea/scripts/run_review.py documento.docx --project-root . --dry-run
```

## Aprender e avaliar

```bash
python claude-code/.claude/skills/revisao-editorial-ipea/scripts/editorial_lab.py learn \
  "C:\caminho\pasta-editorial" \
  --out-dir ".tmp\aprendizado"
```

```bash
python claude-code/.claude/skills/revisao-editorial-ipea/scripts/editorial_lab.py evaluate \
  --knowledge ".tmp\aprendizado\editorial_knowledge.json" \
  --report ".tmp\agentes-relatorio.json" \
  --diagnostics ".tmp\agentes-diagnostico.json" \
  --out-dir ".tmp\avaliacao"
```

Não versione `.env`. A decisão editorial final continua humana.

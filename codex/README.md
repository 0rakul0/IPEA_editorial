# Codex

Integração do `lang_IPEA_editorial` com Codex: `AGENTS.md`, skill instalável, wrapper e referências dos agentes.

## 1. Instalar o Codex

Windows PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://chatgpt.com/codex/install.ps1 | iex"
```

macOS/Linux:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
```

Alternativa:

```bash
npm install -g @openai/codex
```

Execute `codex` e autentique.

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
powershell -ExecutionPolicy Bypass -File .\codex\install.ps1 -Scope user
```

```bash
bash codex/install.sh user
```

Somente neste repositório:

```powershell
powershell -ExecutionPolicy Bypass -File .\codex\install.ps1 -Scope repo
```

```bash
bash codex/install.sh repo
```

O escopo de usuário usa `~/.agents/skills`; o de repositório usa `.agents/skills`.

Copie ou incorpore `codex/AGENTS.md` no `AGENTS.md` da raiz se quiser regras persistentes. Não sobrescreva instruções existentes sem revisão.

## 4. Usar

```bash
codex
```

```text
$revisao-editorial-ipea revise D:\documentos\texto.docx e gere também o diagnóstico JSON.
```

## 5. Validar

Digite `/skills` ou mencione `$revisao-editorial-ipea`. Reinicie a sessão se necessário.

Teste sem LLM:

```bash
python codex/skills/revisao-editorial-ipea/scripts/run_review.py documento.docx --project-root . --dry-run
```

## Aprender com uma pasta editorial

```bash
python codex/skills/revisao-editorial-ipea/scripts/editorial_lab.py learn \
  "C:\caminho\pasta-editorial" \
  --out-dir ".tmp\aprendizado"
```

Depois de executar os agentes no original:

```bash
python codex/skills/revisao-editorial-ipea/scripts/editorial_lab.py evaluate \
  --knowledge ".tmp\aprendizado\editorial_knowledge.json" \
  --report ".tmp\agentes-relatorio.json" \
  --diagnostics ".tmp\agentes-diagnostico.json" \
  --out-dir ".tmp\avaliacao"
```

Não versione `.env`. Revise os comentários antes de aceitar decisões editoriais.

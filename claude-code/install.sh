#!/usr/bin/env bash
set -euo pipefail

scope="${1:-user}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
source_dir="$script_dir/.claude/skills/revisao-editorial-ipea"

case "$scope" in
  user) base_dir="$HOME/.claude/skills" ;;
  repo) base_dir="$repo_root/.claude/skills" ;;
  *) echo "Uso: bash claude-code/install.sh [user|repo]" >&2; exit 2 ;;
esac

target_dir="$base_dir/revisao-editorial-ipea"
mkdir -p "$base_dir"
rm -rf "$target_dir"
cp -R "$source_dir" "$target_dir"
shared_lab="$repo_root/codex/skills/revisao-editorial-ipea/scripts/editorial_lab.py"
if [[ -f "$shared_lab" ]]; then
  cp "$shared_lab" "$target_dir/scripts/editorial_lab.py"
fi
echo "Skill instalada em: $target_dir"
echo "Abra uma nova sessão do Claude Code se a skill não aparecer."

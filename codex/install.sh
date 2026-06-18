#!/usr/bin/env bash
set -euo pipefail

scope="${1:-user}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
source_dir="$script_dir/skills/revisao-editorial-ipea"

case "$scope" in
  user) base_dir="$HOME/.agents/skills" ;;
  repo) base_dir="$repo_root/.agents/skills" ;;
  *) echo "Uso: bash codex/install.sh [user|repo]" >&2; exit 2 ;;
esac

target_dir="$base_dir/revisao-editorial-ipea"
mkdir -p "$base_dir"
rm -rf "$target_dir"
cp -R "$source_dir" "$target_dir"
echo "Skill instalada em: $target_dir"
echo "Reinicie o Codex se a skill não aparecer automaticamente."


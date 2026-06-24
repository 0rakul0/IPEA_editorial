#!/usr/bin/env bash
set -euo pipefail

SCOPE="${1:-repo}"
SKILL_NAME="revisao-editorial-ipea"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
SOURCE="$REPO_ROOT/.opencode/skills/$SKILL_NAME"

if [ ! -f "$SOURCE/SKILL.md" ]; then
    echo "SKILL.md not found at: $SOURCE" >&2
    exit 1
fi

# Copy to a temp location so Remove-Item does not delete the source
TMP_SOURCE="/tmp/opencode-skill-$SKILL_NAME"
rm -rf "$TMP_SOURCE"
cp -R "$SOURCE" "$TMP_SOURCE"

declare -a BASES
if [ "$SCOPE" = "user" ]; then
    BASES=(
        "$HOME/.config/opencode/skills"
        "$HOME/.claude/skills"
        "$HOME/.agents/skills"
    )
else
    BASES=(
        "$REPO_ROOT/.opencode/skills"
        "$REPO_ROOT/.claude/skills"
        "$REPO_ROOT/.agents/skills"
    )
fi

for BASE in "${BASES[@]}"; do
    TARGET="$BASE/$SKILL_NAME"
    mkdir -p "$BASE"
    rm -rf "$TARGET"
    cp -R "$TMP_SOURCE" "$TARGET"
    echo "Skill installed at: $TARGET"
done

rm -rf "$TMP_SOURCE"

echo ""
echo "Installation complete. The skill '$SKILL_NAME' is available at:"
for BASE in "${BASES[@]}"; do
    echo "  - $BASE/$SKILL_NAME"
done
echo ""
echo "Restart the assistant session if the skill does not appear automatically."

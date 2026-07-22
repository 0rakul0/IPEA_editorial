param(
    [ValidateSet("user", "repo")]
    [string]$Scope = "repo",
    [string]$SkillName = "revisao-editorial-ipea",
    [string]$RepoRoot = $PSScriptRoot
)

$source = Join-Path $RepoRoot ".agents\skills\$SkillName"
if (-not (Test-Path -LiteralPath "$source\SKILL.md")) {
    Write-Host "SKILL.md nao encontrado em: $source" -ForegroundColor Red
    exit 1
}

# Copia para um local temporario para evitar que Remove-Item apague a origem
$tmpSource = Join-Path $env:TEMP "opencode-skill-$SkillName"
if (Test-Path -LiteralPath $tmpSource) {
    Remove-Item -Recurse -Force -LiteralPath $tmpSource
}
Copy-Item -Recurse -Force -Path $source -Destination $tmpSource

$paths = @()
if ($Scope -eq "user") {
    $paths += Join-Path $HOME ".codex\skills"
    $paths += Join-Path $HOME ".config\opencode\skills"
    $paths += Join-Path $HOME ".claude\skills"
    $paths += Join-Path $HOME ".agents\skills"
} else {
    # Codex descobre skills locais em .agents/skills; nao e necessario criar
    # uma pasta .codex dentro do repositorio.
    $paths += Join-Path $RepoRoot ".opencode\skills"
    $paths += Join-Path $RepoRoot ".claude\skills"
    $paths += Join-Path $RepoRoot ".agents\skills"
}

foreach ($base in $paths) {
    $target = Join-Path $base $SkillName
    New-Item -ItemType Directory -Force -Path $base | Out-Null
    if (Test-Path -LiteralPath $target) {
        Remove-Item -Recurse -Force -LiteralPath $target
    }
    Copy-Item -Recurse -Force -Path $tmpSource -Destination $target
    Write-Host "Skill instalada em: $target"
}

Remove-Item -Recurse -Force -LiteralPath $tmpSource -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "Instalacao concluida. A skill '$SkillName' esta disponivel em:" -ForegroundColor Green
foreach ($base in $paths) {
    Write-Host "  - $(Join-Path $base $SkillName)"
}
Write-Host ""
Write-Host "Reinicie a sessao do assistente se a skill nao aparecer automaticamente."

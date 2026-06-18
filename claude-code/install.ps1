param(
    [ValidateSet("user", "repo")]
    [string]$Scope = "user",
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$source = Join-Path $PSScriptRoot ".claude\skills\revisao-editorial-ipea"
$base = if ($Scope -eq "user") {
    Join-Path $HOME ".claude\skills"
} else {
    Join-Path $RepoRoot ".claude\skills"
}
$target = Join-Path $base "revisao-editorial-ipea"

New-Item -ItemType Directory -Force -Path $base | Out-Null
if (Test-Path -LiteralPath $target) {
    Remove-Item -Recurse -Force -LiteralPath $target
}
Copy-Item -Recurse -Force -Path $source -Destination $target
$sourceRepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sharedLab = Join-Path $sourceRepoRoot "codex\skills\revisao-editorial-ipea\scripts\editorial_lab.py"
if (Test-Path -LiteralPath $sharedLab) {
    Copy-Item -Force -LiteralPath $sharedLab -Destination (Join-Path $target "scripts\editorial_lab.py")
}
Write-Host "Skill instalada em: $target"
Write-Host "Abra uma nova sessão do Claude Code se a skill não aparecer."

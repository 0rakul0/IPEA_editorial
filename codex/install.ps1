param(
    [ValidateSet("user", "repo")]
    [string]$Scope = "user",
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$source = Join-Path $PSScriptRoot "skills\revisao-editorial-ipea"
$base = if ($Scope -eq "user") {
    Join-Path $HOME ".agents\skills"
} else {
    Join-Path $RepoRoot ".agents\skills"
}
$target = Join-Path $base "revisao-editorial-ipea"

New-Item -ItemType Directory -Force -Path $base | Out-Null
if (Test-Path -LiteralPath $target) {
    Remove-Item -Recurse -Force -LiteralPath $target
}
Copy-Item -Recurse -Force -Path $source -Destination $target
Write-Host "Skill instalada em: $target"
Write-Host "Reinicie o Codex se a skill não aparecer automaticamente."

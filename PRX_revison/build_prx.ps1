param(
    [switch]$CleanAux
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

if (-not (Get-Command latexmk -ErrorAction SilentlyContinue)) {
    throw 'latexmk was not found in PATH. Install TeX Live/MiKTeX and ensure latexmk is available.'
}

$texFile = 'manuscript_PRX_transient_freezing_draft.tex'

latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error $texFile
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if ($CleanAux) {
    latexmk -c $texFile
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Write-Host "Build complete: $(Join-Path $here 'manuscript_PRX_transient_freezing_draft.pdf')"

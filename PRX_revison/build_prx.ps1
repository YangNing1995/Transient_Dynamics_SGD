param(
    [switch]$CleanAux,
    [switch]$CleanAfter
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$here = (Resolve-Path (Split-Path -Parent $MyInvocation.MyCommand.Path)).Path
$texFile = 'manuscript_PRX_transient_freezing_draft.tex'
$jobName = [System.IO.Path]::GetFileNameWithoutExtension($texFile)

function Remove-BuildAuxFiles {
    $extensions = @(
        'aux', 'bcf', 'blg', 'fdb_latexmk', 'fls',
        'log', 'out', 'run.xml', 'synctex.gz', 'synctex(busy)', 'toc'
    )

    for ($attempt = 1; $attempt -le 20; $attempt++) {
        foreach ($extension in $extensions) {
            $auxPath = Join-Path $here "$jobName.$extension"
            if (Test-Path -LiteralPath $auxPath) {
                Remove-Item -LiteralPath $auxPath -Force -ErrorAction SilentlyContinue
            }
        }

        Get-ChildItem -LiteralPath $here -Filter 'pdflatex*.fls' -Force -ErrorAction SilentlyContinue |
            Remove-Item -Force -ErrorAction SilentlyContinue

        $remaining = @()
        foreach ($extension in $extensions) {
            $auxPath = Join-Path $here "$jobName.$extension"
            if (Test-Path -LiteralPath $auxPath) {
                $remaining += $auxPath
            }
        }
        $remaining += @(Get-ChildItem -LiteralPath $here -Filter 'pdflatex*.fls' -Force -ErrorAction SilentlyContinue)
        if ($remaining.Count -eq 0) {
            return
        }

        Start-Sleep -Milliseconds 250
    }
}

function Wait-TeXProcesses {
    for ($attempt = 1; $attempt -le 40; $attempt++) {
        $escapedHere = [WildcardPattern]::Escape($here)
        $escapedJob = [WildcardPattern]::Escape($jobName)
        $active = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $_.Name -match '^(latexmk|perl|runscript|pdftex|pdflatex|bibtex)\.exe$' -and
                ($_.CommandLine -like "*$escapedHere*" -or $_.CommandLine -like "*$escapedJob*")
            })

        if ($active.Count -eq 0) {
            return
        }

        Start-Sleep -Milliseconds 250
    }
}

Push-Location $here
try {
    if (-not (Get-Command latexmk -ErrorAction SilentlyContinue)) {
        throw 'latexmk was not found in PATH. Install TeX Live/MiKTeX and ensure latexmk is available.'
    }

    if ($CleanAux) {
        # Remove stale latexmk state before building; otherwise latexmk can
        # force BibTeX on a dummy .aux after a previous partial cleanup.
        # Keep the .bbl so the first LaTeX pass in a clean build does not emit
        # a misleading missing-bibliography diagnostic; BibTeX will refresh it.
        Write-Host 'Cleaning auxiliary files before build.'
        Remove-BuildAuxFiles
    }

    latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error $texFile
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    Wait-TeXProcesses

    if ($CleanAfter) {
        Write-Host 'Cleaning auxiliary files after successful build.'
        # On Windows the TeX Live wrapper can return before child processes
        # finish writing aux files, so clean directly and retry instead of
        # relying on latexmk -c.
        Remove-BuildAuxFiles
        Wait-TeXProcesses
        Remove-BuildAuxFiles
    }

    Write-Host "Build complete: $(Join-Path $here 'manuscript_PRX_transient_freezing_draft.pdf')"
}
finally {
    Pop-Location
}

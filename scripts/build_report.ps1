param(
  [string]$Output = (Join-Path $PSScriptRoot '..\output\pdf\solar-filament-report.pdf')
)
$ErrorActionPreference = 'Stop'
$reportDir = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\report'))
$build = Join-Path $reportDir 'build'
$destination = [IO.Path]::GetFullPath($Output)
New-Item -ItemType Directory -Force -Path $build, (Split-Path $destination -Parent) | Out-Null
Push-Location $reportDir
try {
  function Invoke-TeX {
    & pdflatex -interaction=nonstopmode -halt-on-error "-output-directory=$build" main.tex |
      Set-Content -Encoding utf8 (Join-Path $build 'pdflatex-console.txt')
    if ($LASTEXITCODE) { throw "pdflatex failed; see $build\main.log" }
  }
  Invoke-TeX
  & bibtex (Join-Path $build 'main.aux')
  if ($LASTEXITCODE) { throw "bibtex failed; see $build\main.blg" }
  Invoke-TeX
  Invoke-TeX
  Copy-Item -LiteralPath (Join-Path $build 'main.pdf') -Destination $destination
  Get-FileHash -LiteralPath $destination -Algorithm SHA256
} finally {
  Pop-Location
}

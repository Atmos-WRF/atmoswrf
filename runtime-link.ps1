# Atmos-WRF: one runtime per PC (Python, ArWen, CUDA, the microphysics tables, the download and
# weather caches: ~3-4 GB) shared by every version, so a new version does not download it all again.
#
# The shared runtime lives in %LOCALAPPDATA%\ARHT-MODEL\runtime (or ARHTMODELVIEWER_RUNTIME); each version's
# "runtime" folder is a junction to it.  The first time, an install an older version made in its own folder
# is adopted: moved there (instant on the same drive; that version keeps working through a junction left in
# its place) or, on another drive, used where it is.
#
#   runtime-link.ps1 -Project <folder>          link the folder, print the real runtime path
#   runtime-link.ps1 -Project <folder> -Check   exit 1 if the packages this version needs changed since the
#                                               last setup (Setup.cmd then installs just the difference)
#   runtime-link.ps1 -Project <folder> -Stamp   record that setup finished for this version's packages
param([string]$Project = $PSScriptRoot, [switch]$Check, [switch]$Stamp, [switch]$Quiet, [string[]]$SearchRoots)
$ErrorActionPreference = 'Stop'
$Project = (Resolve-Path -LiteralPath $Project).Path.TrimEnd('\')
$base = Join-Path $env:LOCALAPPDATA 'ARHT-MODEL'
$pointer = Join-Path $base 'runtime-location.txt'
$default = if ($env:ARHTMODELVIEWER_RUNTIME) { $env:ARHTMODELVIEWER_RUNTIME.TrimEnd('\') } else { Join-Path $base 'runtime' }
$local = Join-Path $Project 'runtime'

function Say($m) { if (-not $Quiet) { [Console]::Error.WriteLine($m) } }
function Installed($p) { $p -and (Test-Path -LiteralPath (Join-Path $p 'venv\Scripts\python.exe')) }
function IsLink($p) {
    $i = Get-Item -LiteralPath $p -Force -ErrorAction SilentlyContinue
    return [bool]($i -and ($i.Attributes -band [IO.FileAttributes]::ReparsePoint))
}
function Junction($at, $to) { New-Item -ItemType Junction -Path $at -Target $to | Out-Null }
# subfolders of $p, none when it cannot be read: a wildcard Get-Item stopped the whole script with "Access is
# denied" on one protected folder in Downloads ($ErrorActionPreference Stop overrides its SilentlyContinue)
function SubDirs($p) { try { [IO.Directory]::GetDirectories($p) } catch { @() } }
function Unlink($p) { [IO.Directory]::Delete($p) }          # removes a junction, never what it points to

# --- the packages this version needs, and whether setup installed exactly those
function PackagesHash {
    $text = ''
    foreach ($f in 'requirements.txt', 'requirements-cpu.txt', 'constraints.txt') {
        $p = Join-Path $Project $f
        if (Test-Path -LiteralPath $p) { $text += (Get-Content -LiteralPath $p -Raw) }
    }
    $sha = [Security.Cryptography.SHA256]::Create()
    return [BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($text))).Replace('-', '')
}
$stampFile = Join-Path $local 'setup-stamp.txt'
if ($Check) {
    if (-not (Test-Path -LiteralPath $stampFile)) { exit 1 }
    if ((Get-Content -LiteralPath $stampFile -Raw).Trim() -ne (PackagesHash)) { exit 1 }
    exit 0
}
if ($Stamp) { Set-Content -LiteralPath $stampFile -Value (PackagesHash) -Encoding ASCII; exit 0 }

# --- this folder already links to a working runtime
if (IsLink $local) {
    if (Installed $local) { Write-Output (Get-Item -LiteralPath $local -Force).Target; exit 0 }
    Unlink $local                                            # dangling (that runtime was deleted): link again
}

# --- the PC's shared runtime, if there is one
$shared = $null
if (Test-Path -LiteralPath $pointer) {
    $s = (Get-Content -LiteralPath $pointer -Raw).Trim()
    if (Installed $s) { $shared = $s }
}
if (-not $shared -and (Installed $default)) { $shared = $default }

if ($shared) {
    if (Test-Path -LiteralPath $local) {
        if (Installed $local) { Write-Output $local; exit 0 }   # a version with its own install: leave it be
        # leftovers without an install (caches, settings): into the shared runtime, then link
        robocopy $local $shared /E /MOVE /XC /XN /XO /NFL /NDL /NJH /NJS /NP | Out-Null
        if (Test-Path -LiteralPath $local) { Remove-Item -LiteralPath $local -Recurse -Force }
    }
    Junction $local $shared
    Write-Output $shared
    exit 0
}

# --- no shared runtime yet: adopt an install an older version made in its own folder
$found = @()
if ((Test-Path -LiteralPath $local) -and -not (IsLink $local) -and (Installed $local)) { $found += Get-Item -LiteralPath $local }
$roots = @((Split-Path $Project -Parent), (Split-Path (Split-Path $Project -Parent) -Parent),
           [Environment]::GetFolderPath('Desktop'), (Join-Path $env:USERPROFILE 'Downloads'), 'C:\')
if ($SearchRoots) { $roots = $SearchRoots }                   # (tests: look only there)
$roots = $roots |
         Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -Unique
foreach ($r in $roots) {
    $levels = if ($r.Length -le 3) { 2 } else { 3 }               # a drive's root: not deep into Windows
    $dirs = @($r)
    for ($depth = 1; $depth -le $levels; $depth++) {
        $dirs = @($dirs | ForEach-Object { SubDirs $_ })
        foreach ($p in $dirs) {
            $rt = Join-Path $p 'runtime'
            if ((Installed $rt) -and -not (IsLink $rt) -and
                (Test-Path -LiteralPath (Join-Path $p 'arhtmodelviewer\__init__.py'))) {
                $found += Get-Item -LiteralPath $rt -Force
            }
        }
    }
}
$old = $found | Sort-Object -Property FullName -Unique | Sort-Object -Property LastWriteTime -Descending | Select-Object -First 1

New-Item -ItemType Directory -Force -Path $base | Out-Null
if ($old) {
    $from = $old.FullName.TrimEnd('\')
    $target = $from
    if (([IO.Path]::GetPathRoot($from) -eq [IO.Path]::GetPathRoot($default)) -and -not (Test-Path -LiteralPath $default)) {
        try {
            Move-Item -LiteralPath $from -Destination $default
            Junction $from $default                          # the old version keeps working through it
            $target = $default
            # the environment names the Python it was made from: if that was inside the moved folder, re-point it
            $cfg = Join-Path $default 'venv\pyvenv.cfg'
            if (Test-Path -LiteralPath $cfg) {
                $t = Get-Content -LiteralPath $cfg -Raw
                $t2 = [regex]::Replace($t, [regex]::Escape($from), $default.Replace('$', '$$'), 'IgnoreCase')
                # no byte-order mark: Python would not read the "home" line after one
                if ($t2 -ne $t) { [IO.File]::WriteAllText($cfg, $t2, (New-Object Text.UTF8Encoding $false)) }
            }
        } catch {
            Say "Could not move $from (is that version still running?): using it where it is."
            $target = $from
        }
    }
    Set-Content -LiteralPath $pointer -Value $target -Encoding UTF8
    if ($target -ne $local) {
        if (Test-Path -LiteralPath $local) {
            robocopy $local $target /E /MOVE /XC /XN /XO /NFL /NDL /NJH /NJS /NP | Out-Null
            if (Test-Path -LiteralPath $local) { Remove-Item -LiteralPath $local -Recurse -Force }
        }
        Junction $local $target
    }
    Say "Reusing the setup of $(Split-Path $from -Parent): nothing to download again."
    if ($target -ne $default) { Say "(Other versions share it: keep that folder, or run Setup.cmd in a new one to move it.)" }
    Write-Output $target
    exit 0
}

# --- a brand-new PC: the runtime goes to the shared place, and this version links to it
New-Item -ItemType Directory -Force -Path $default | Out-Null
Set-Content -LiteralPath $pointer -Value $default -Encoding UTF8
if (Test-Path -LiteralPath $local) {
    robocopy $local $default /E /MOVE /XC /XN /XO /NFL /NDL /NJH /NJS /NP | Out-Null
    if (Test-Path -LiteralPath $local) { Remove-Item -LiteralPath $local -Recurse -Force }
}
Junction $local $default
Write-Output $default
exit 0

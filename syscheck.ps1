# Atmos-WRF's minimum PC, checked by Setup.cmd before anything is downloaded: Windows, 8 GB of memory
# and an NVIDIA graphics card of the GTX 10 series (Pascal) or newer with 4 GB of its own memory; without such a
# card, a processor strong enough for the CPU backend (8 cores with AVX2) and 16 GB of memory.  Prints what it
# found; exit 0: the GPU backend, 2: the CPU backend only, 1: neither.  "-Cpu": only the CPU backend is wanted.
param([switch]$Cpu)
$ErrorActionPreference = 'SilentlyContinue'
$MinRamGB = 8
$MinVramMiB = 3900          # a 4 GB card reports 4,040-4,096 MiB
$MinCompute = 6.0           # Pascal (GTX 10 series, Quadro P, Tesla P100) and newer
$CpuMinCores = 8            # physical cores, for the CPU backend
$CpuMinRamGB = 16
$fails = @()

Write-Host ''
Write-Host 'Checking this PC ...'

# --- Windows
if ($env:OS -ne 'Windows_NT') {
    $fails += 'Atmos-WRF runs on Windows only.'
    Write-Host '  Operating system: not Windows'
} else {
    $os = Get-CimInstance Win32_OperatingSystem
    Write-Host ('  Operating system: ' + $(if ($os) { $os.Caption } else { 'Windows' }))
}

# --- memory: what is installed (TotalPhysicalMemory is less: the part the hardware reserves is left out)
$ram = (Get-CimInstance Win32_PhysicalMemory | Measure-Object -Property Capacity -Sum).Sum
if (-not $ram) { $ram = (Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory * 1.03 }
$ramGB = [math]::Round($ram / 1GB, 1)
Write-Host ('  Memory: ' + $ramGB + ' GB')
if ($ramGB -lt $MinRamGB - 0.25) { $fails += "Less than $MinRamGB GB of memory ($ramGB GB)." }

# --- NVIDIA graphics card: nvidia-smi (the driver's own tool) gives its memory and generation
$cards = @()
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
    $rows = & nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $rows) {           # drivers before ~510 have no compute_cap
        $rows = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null
    }
    foreach ($r in @($rows)) {
        $f = "$r".Split(',') | ForEach-Object { $_.Trim() }
        if ($f.Count -lt 2 -or -not $f[0]) { continue }
        $cc = $null
        if ($f.Count -ge 3) { $v = 0.0; if ([double]::TryParse($f[2], [Globalization.NumberStyles]::Float, [Globalization.CultureInfo]::InvariantCulture, [ref]$v)) { $cc = $v } }
        if ($cc -eq $null) {
            # no compute_cap from the driver: the generation from the name (GTX 9xx and older, Kepler/Maxwell Quadros)
            if ($f[0] -match 'GTX\s*(TITAN\s*(X|Black|Z)?$|[4-9]\d\d\b)|GT\s*\d{3}\b|Quadro\s*(K|M)\d|Tesla\s*(K|M)\d') { $cc = 5.0 } else { $cc = 6.0 }
        }
        $cards += [pscustomobject]@{ Name = $f[0]; MiB = [double]$f[1]; CC = $cc }
    }
}
$nvidiaNoDriver = $false
if (-not $cards) {
    $vc = Get-CimInstance Win32_VideoController | Where-Object { $_.Name -match 'NVIDIA' }
    if ($vc) { $nvidiaNoDriver = $true; Write-Host ('  Graphics: ' + (($vc | ForEach-Object { $_.Name }) -join ', ') + ' (no NVIDIA driver)') }
    else {
        $other = (Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name }) -join ', '
        Write-Host ('  Graphics: ' + $(if ($other) { $other } else { 'none found' }))
    }
}
foreach ($c in $cards) {
    Write-Host ('  Graphics: ' + $c.Name + ', ' + [math]::Round($c.MiB / 1024, 1) + ' GB')
}

# --- the processor, for the CPU backend: physical cores and AVX2 (the compiled kernels' vector width)
$procs = @(Get-CimInstance Win32_Processor)
$cores = ($procs | Measure-Object -Property NumberOfCores -Sum).Sum
$threads = ($procs | Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum
$avx2 = $false
try {
    Add-Type -Namespace ArhtCheck -Name K32 -MemberDefinition '[DllImport("kernel32.dll")] public static extern bool IsProcessorFeaturePresent(uint f);'
    $avx2 = [ArhtCheck.K32]::IsProcessorFeaturePresent(40)        # PF_AVX2_INSTRUCTIONS_AVAILABLE
} catch { $avx2 = $true }                                          # older Windows cannot say: not held against it
Write-Host ('  Processor: ' + $(if ($procs) { $procs[0].Name.Trim() } else { 'unknown' }) + ', ' + $cores + ' cores / ' + $threads + ' threads' + $(if (-not $avx2) { ', no AVX2' } else { '' }))
$cpuFails = @()
if ($cores -lt $CpuMinCores) { $cpuFails += "the processor has $cores cores (at least $CpuMinCores)" }
if (-not $avx2) { $cpuFails += 'the processor has no AVX2 (Intel 4th generation / AMD Ryzen or newer)' }
if ($ramGB -lt $CpuMinRamGB - 0.5) { $cpuFails += "$ramGB GB of memory (at least $CpuMinRamGB)" }
$good = $cards | Where-Object { $_.CC -ge $MinCompute -and $_.MiB -ge $MinVramMiB }
$gpuFails = @()
if (-not $good) {
    if ($nvidiaNoDriver) {
        $gpuFails += 'The NVIDIA graphics card has no driver: install it from https://www.nvidia.com/drivers, restart the PC and run Setup.cmd again.'
    } elseif (-not $cards) {
        $gpuFails += 'No NVIDIA graphics card.'
    } else {
        foreach ($c in $cards) {
            if ($c.CC -lt $MinCompute) { $gpuFails += ($c.Name + ' is older than the GTX 10 series (GTX 1080 Ti generation).') }
            elseif ($c.MiB -lt $MinVramMiB) { $gpuFails += ($c.Name + ' has ' + [math]::Round($c.MiB / 1024, 1) + ' GB of graphics memory; at least 4 GB is needed.') }
        }
    }
}

Write-Host ''
$mode = 'none'
if (-not $fails) {
    if ($good -and -not $Cpu) { $mode = 'gpu' } elseif (-not $cpuFails) { $mode = 'cpu' }
}
if ($mode -eq 'gpu') {
    Write-Host ('This PC can run Atmos-WRF on its graphics card (' + ($good | Select-Object -First 1).Name + ').') -ForegroundColor Green
    exit 0
}
if ($mode -eq 'cpu') {
    if (-not $Cpu) { foreach ($m in $gpuFails) { Write-Host ('  - ' + $m) } }
    Write-Host 'This PC can run Atmos-WRF on its processor (CPU backend: much slower than a graphics card).' -ForegroundColor Green
    exit 2
}
Write-Host 'This PC cannot run Atmos-WRF:' -ForegroundColor Red
foreach ($m in $fails) { Write-Host ('  - ' + $m) -ForegroundColor Red }
if (-not $fails) {
    if (-not $Cpu) { foreach ($m in $gpuFails) { Write-Host ('  - ' + $m) -ForegroundColor Red } }
    Write-Host ('  - Not enough for the CPU backend either: ' + ($cpuFails -join '; ') + '.') -ForegroundColor Red
}
Write-Host ''
Write-Host 'It needs Windows and 8 GB of memory, and either an NVIDIA graphics card from the GTX 10 series (GTX 1080 Ti'
Write-Host 'generation) or newer with at least 4 GB of graphics memory, or (much slower) a processor with 8 or more'
Write-Host 'cores and AVX2 with 16 GB of memory.'
exit 1

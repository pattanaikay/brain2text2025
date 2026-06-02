# run_thesis.ps1
# -----------------------------------------------------------------------
# Automates the full thesis experiment pipeline:
#   Step 0  - verify all tests pass
#   Step 1  - verify checkpoint is wired into G3 spec
#   Step 2  - G2: DietCorp TTA reproduction (N=0 vs N=1)
#   Step 3  - read G2 results
#   Step 4  - G3: sleep-consolidation N-sweep (THE thesis experiment)
#   Step 5  - read G3 results + print interpretation
#   Step 6  - H2: ZenBrain memory training
#
# Usage (from the brain2text-experiments directory):
#   .\run_thesis.ps1                  # run everything
#   .\run_thesis.ps1 -SkipTests       # skip pytest (faster re-runs)
#   .\run_thesis.ps1 -OnlyAdapt       # skip H2 training, only G2+G3
#   .\run_thesis.ps1 -NSteps 0,1,2,4  # override consolidation sweep
# -----------------------------------------------------------------------
param(
    [switch]$SkipTests,
    [switch]$OnlyAdapt,
    [string[]]$NSteps = @()
)

$env:PYTHONIOENCODING = "utf-8"
$ErrorActionPreference = "Continue"   # don't abort on non-zero exit; we handle it

$SCRIPT_DIR = $PSScriptRoot
Set-Location $SCRIPT_DIR

# ── Colours ──────────────────────────────────────────────────────────────────
function Header  { param($msg) Write-Host "`n$('='*65)" -ForegroundColor Cyan
                              Write-Host "  $msg"        -ForegroundColor Cyan
                              Write-Host $('='*65)       -ForegroundColor Cyan }
function Step    { param($n,$msg) Write-Host "`n[STEP $n] $msg" -ForegroundColor Yellow }
function OK      { param($msg) Write-Host "  OK  $msg"    -ForegroundColor Green }
function WARN    { param($msg) Write-Host "  WARN $msg"   -ForegroundColor DarkYellow }
function FAIL    { param($msg) Write-Host "  FAIL $msg"   -ForegroundColor Red }
function INFO    { param($msg) Write-Host "  ... $msg"    -ForegroundColor Gray }
function RESULT  { param($msg) Write-Host "  >>> $msg"    -ForegroundColor Magenta }

# ── Timing helper ─────────────────────────────────────────────────────────────
function Run-Timed {
    param([string]$Label, [scriptblock]$Block)
    $t0 = Get-Date
    INFO "Starting: $Label"
    & $Block
    $exit = $LASTEXITCODE
    $elapsed = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1)
    return @{ ExitCode = $exit; Seconds = $elapsed }
}

# ── Step results tracker ──────────────────────────────────────────────────────
$Results = [ordered]@{}

# =============================================================================
Header "Thesis Experiment Runner - DietCorp + Sleep Consolidation + ZenBrain"
Write-Host "  Repo: $SCRIPT_DIR"
Write-Host "  Time: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Host "  GPU:  $(py -3 -c 'import torch; print(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print(\"CPU only\")' 2>$null)"

# =============================================================================
# STEP 0 - Tests
# =============================================================================
if (-not $SkipTests) {
    Step 0 "Verifying all tests pass (should be 40 passed)"
    $r = Run-Timed "pytest" {
        py -3 -m pytest tests/ -m "not slow" -q 2>&1 | Tee-Object -Variable testOut
        $testOut | Select-Object -Last 3
    }
    $summary = $testOut | Select-String "passed" | Select-Object -Last 1
    if ($r.ExitCode -eq 0) {
        OK "Tests green - $summary ($($r.Seconds)s)"
        $Results["Step0_tests"] = "PASSED"
    } else {
        FAIL "Tests failed. Fix before proceeding."
        FAIL "Output: $summary"
        $Results["Step0_tests"] = "FAILED"
        Write-Host "`nAborting - fix test failures first." -ForegroundColor Red
        exit 1
    }
} else {
    WARN "Skipping tests (--SkipTests flag set)"
    $Results["Step0_tests"] = "SKIPPED"
}

# =============================================================================
# STEP 1 - Verify checkpoint
# =============================================================================
Step 1 "Verifying checkpoint is wired into specs/G3_sleep_consolidation.yaml"

$specContent = Get-Content "specs\G3_sleep_consolidation.yaml" -Raw
if ($specContent -match 'pretrained_ckpt:\s*"([^"]+)"') {
    $ckptPath = $matches[1]
    if (Test-Path $ckptPath) {
        $sizeMB = [math]::Round((Get-Item $ckptPath).Length / 1MB, 1)
        OK "Checkpoint found: $ckptPath ($sizeMB MB)"
        $Results["Step1_checkpoint"] = "FOUND: $ckptPath"
    } else {
        FAIL "Checkpoint path set but FILE NOT FOUND: $ckptPath"
        FAIL "Check the path in specs/G3_sleep_consolidation.yaml"
        $Results["Step1_checkpoint"] = "NOT FOUND"
        exit 1
    }
} elseif ($specContent -match 'pretrained_ckpt:\s*null') {
    FAIL "pretrained_ckpt is still null in specs/G3_sleep_consolidation.yaml"
    FAIL "Set it to your best_model_per.pth path before running."
    $Results["Step1_checkpoint"] = "NOT SET"
    exit 1
} else {
    WARN "Could not parse pretrained_ckpt from spec - will proceed and let run.py handle it"
    $Results["Step1_checkpoint"] = "UNKNOWN"
}

# =============================================================================
# STEP 2 - G2: DietCorp TTA reproduction
# =============================================================================
Step 2 "G2 - DietCorp TTA reproduction: N=0 vs N=1  (~5 min)"
INFO "This proves the drift instrument works and N=1 beats no-adaptation."

$r = Run-Timed "G2 --adapt" {
    py -3 run.py --expt G2 --profile toy --adapt --val_h5 data/toy_val.hdf5 2>&1 | Tee-Object -Variable g2Out
    $g2Out | Select-String "N\s+PER|^\s+[01]\s+|H_main" | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
}

if ($r.ExitCode -eq 0) {
    OK "G2 completed - $($r.Seconds)s"
    $Results["Step2_G2"] = "PASSED ($($r.Seconds)s)"
} else {
    FAIL "G2 failed (exit $($r.ExitCode))"
    $Results["Step2_G2"] = "FAILED"
    Write-Host "  Last output:" -ForegroundColor DarkYellow
    $g2Out | Select-Object -Last 10 | ForEach-Object { Write-Host "    $_" }
    Write-Host "`nAborting - G2 must succeed before running G3." -ForegroundColor Red
    exit 1
}

# =============================================================================
# STEP 3 - Read G2 results
# =============================================================================
Step 3 "Reading G2 results"
py -3 tools/read_results.py G2 2>&1 | Tee-Object -Variable g2Read
$g2Read | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
$Results["Step3_G2_read"] = ($g2Read | Select-String "H_main|INTERPRETATION|N-monoton" | Select-Object -First 2) -join "; "

# =============================================================================
# STEP 4 - G3: sleep-consolidation N-sweep (THE THESIS EXPERIMENT)
# =============================================================================
Step 4 "G3 - Sleep consolidation N-sweep  (~15 min)"
INFO "This is the thesis experiment. Sweeps N in {0,1,2,4,8}."
INFO "H_main confirmed if PER@last falls as N rises AND wake_ms stays flat."

$g3Cmd = { py -3 run.py --expt G3 --profile toy --adapt --val_h5 data/toy_val.hdf5 2>&1 | Tee-Object -Variable g3Out
           $g3Out | Select-String "N\s+PER|^\s+[0-9]+\s+|H_main|sweeping" | ForEach-Object { Write-Host "  $_" -ForegroundColor White } }

if ($NSteps.Count -gt 0) {
    $nArg = $NSteps -join " "
    INFO "Overriding N sweep to: $nArg"
    $g3Cmd = [scriptblock]::Create("py -3 run.py --expt G3 --profile toy --adapt --n_steps $nArg 2>&1 | Tee-Object -Variable g3Out; `$g3Out | Select-String 'N\s+PER|^\s+[0-9]+\s+|H_main|sweeping' | ForEach-Object { Write-Host `"  `$_`" -ForegroundColor White }")
}

$r = Run-Timed "G3 --adapt" $g3Cmd

if ($r.ExitCode -eq 0) {
    OK "G3 completed ($($r.Seconds)s)"
    $Results["Step4_G3"] = "PASSED ($($r.Seconds)s)"
} else {
    FAIL "G3 failed (exit $($r.ExitCode))"
    $Results["Step4_G3"] = "FAILED"
    $g3Out | Select-Object -Last 15 | ForEach-Object { Write-Host "    $_" }
}

# =============================================================================
# STEP 5 - Read G3 results + interpretation
# =============================================================================
Step 5 "Reading G3 results (thesis interpretation)"
py -3 tools/read_results.py G3 2>&1 | Tee-Object -Variable g3Read
$g3Read | ForEach-Object { Write-Host "  $_" -ForegroundColor White }
$interp = ($g3Read | Select-String "H_main|INTERPRETATION|N-monoton|Wake lat|Adaptation" | Select-Object -First 5) -join " | "
$Results["Step5_G3_interp"] = $interp

# =============================================================================
# STEP 6 - H2: ZenBrain memory training
# =============================================================================
if (-not $OnlyAdapt) {
    Step 6 "H2 - ZenBrain episodic memory training  (~20 min)"
    INFO "Tests whether episodic memory improves decoding WER vs B0 baseline."
    INFO "Watch for: loss_episodic decreasing during training."

    $r = Run-Timed "H2 training" {
        py -3 run.py --expt H2 --profile toy `
            --train_h5 data/toy_train.hdf5 `
            --val_h5   data/toy_val.hdf5   2>&1 | Tee-Object -Variable h2Out
        $h2Out | Select-String "WER|loss_epis|Epoch|best|Early" | Select-Object -Last 10 |
            ForEach-Object { Write-Host "  $_" -ForegroundColor White }
    }

    if ($r.ExitCode -eq 0) {
        OK "H2 completed ($($r.Seconds)s)"
        $bestWer = ($h2Out | Select-String "New best WER" | Select-Object -Last 1)
        if ($bestWer) { RESULT "H2 best WER: $bestWer" }
        $Results["Step6_H2"] = "PASSED ($($r.Seconds)s)"
    } else {
        FAIL "H2 failed (exit $($r.ExitCode)). Check VRAM - H2 needs Qwen2.5-1.5B (~3GB with quantisation)."
        $Results["Step6_H2"] = "FAILED"
        $h2Out | Select-Object -Last 10 | ForEach-Object { Write-Host "    $_" }
    }
} else {
    WARN "Skipping H2 training (--OnlyAdapt flag set)"
    $Results["Step6_H2"] = "SKIPPED"
}

# =============================================================================
# FINAL SUMMARY
# =============================================================================
Header "Run Complete - Summary"
Write-Host "  Finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')`n"

$allPassed = $true
foreach ($key in $Results.Keys) {
    $val = $Results[$key]
    $icon = if ($val -match "^PASSED|^FOUND|^SKIPPED") { "[OK]  " }
            elseif ($val -match "^FAILED|^NOT")         { "[FAIL]"; $allPassed = $false }
            else                                          { "[INFO]" }
    $color = if ($val -match "^PASSED|^FOUND") { "Green" }
             elseif ($val -match "^FAILED|^NOT") { "Red" }
             elseif ($val -match "^SKIPPED") { "DarkYellow" }
             else { "White" }
    Write-Host ("  {0,-6} {1,-30} {2}" -f $icon, $key, $val) -ForegroundColor $color
}

Write-Host ""
if ($allPassed) {
    Write-Host "  All steps passed." -ForegroundColor Green
    Write-Host "  Next: review PER-vs-day table above." -ForegroundColor Green
    Write-Host "  If PER@last falls with N and wake_ms is flat -> H_main confirmed." -ForegroundColor Green
    Write-Host "  When ready for the cloud run:" -ForegroundColor Green
    Write-Host "    py -3 run.py --expt G3 --profile full --adapt" -ForegroundColor DarkGreen
} else {
    Write-Host "  Some steps failed. Check output above." -ForegroundColor Red
    Write-Host "  Common fixes:" -ForegroundColor Yellow
    Write-Host "    - Checkpoint not found: check path in specs/G3_sleep_consolidation.yaml" -ForegroundColor Yellow
    Write-Host "    - G2 collapse (N=1 worse than N=0): checkpoint needs more training" -ForegroundColor Yellow
    Write-Host "    - H2 OOM: VRAM too low for Qwen; run with --OnlyAdapt instead" -ForegroundColor Yellow
}
Write-Host ""

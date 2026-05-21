param(
    [string]$Model = "auto",
    [int]$Iterations = 4,
    [double]$BudgetMinutes = 5,
    [int]$Seed = 13
)

$ErrorActionPreference = "Stop"

for ($i = 1; $i -le $Iterations; $i++) {
    Write-Host "== Autoresearch benchmark iteration $i/$Iterations =="
    python benchmark.py `
        --model $Model `
        --budget-minutes $BudgetMinutes `
        --seed $Seed `
        --batch-size 0 `
        --notes "manual-loop-$i"
    python plot_results.py
}

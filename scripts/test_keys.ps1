#!/usr/bin/env pwsh
# Tests the 4 keys in .env against the actual provider APIs (no client libs).
# Outputs masked. Exits non-zero if any key fails.

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath ".env")) { Write-Error ".env not found"; exit 2 }

Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#][^=]*)=(.*)$') {
        Set-Item -Path "Env:$($Matches[1].Trim())" -Value $Matches[2].Trim()
    }
}

function Mask([string]$s, [int]$keep = 6) {
    if ($null -eq $s -or $s.Length -le $keep) { return "***" }
    return ($s.Substring(0, $keep) + "…" + $s.Substring($s.Length - 4))
}

$script:failed = @()
function Ok([string]$name, [string]$detail = "") { Write-Host "[ OK ] $name`: $detail" }
function Fail([string]$name, [string]$detail = "") { Write-Host "[FAIL] $name`: $detail"; $script:failed += $name }

function Test-Zen {
    Write-Host ""
    Write-Host "=== OpenCode Zen ==="
    $key = $env:OPENCODE_ZEN_API_KEY
    if (-not $key) { Fail "zen key present" "missing"; return }
    try {
        $list = Invoke-RestMethod -Uri "https://opencode.ai/zen/v1/models" `
            -Headers @{ Authorization = "Bearer $key" } -Method GET -TimeoutSec 20
        $ids = ($list | ForEach-Object { $_.id }) -join ", "
        Ok "zen auth" "key=$(Mask $key) returned $($list.Count) model(s): $ids"

        # Try a minimal completion against the free model the user wants.
        $body = @{
            model    = "deepseek-v4-flash-free"
            messages = @(@{ role = "user"; content = "ping" })
            max_tokens = 5
        } | ConvertTo-Json -Depth 6
        try {
            $r = Invoke-RestMethod -Uri "https://opencode.ai/zen/v1/chat/completions" `
                -Headers @{ Authorization = "Bearer $key"; "Content-Type" = "application/json" } `
                -Method POST -Body $body -TimeoutSec 30
            $reply = $r.choices[0].message.content
            Ok "zen chat deepseek-v4-flash-free" "reply=$reply"
        } catch {
            Fail "zen chat deepseek-v4-flash-free" $_.Exception.Message
        }
    } catch {
        Fail "zen auth" $_.Exception.Message
    }
}

function Test-NIM {
    Write-Host ""
    Write-Host "=== NVIDIA NIM ==="
    $key = $env:NIM_API_KEY
    if (-not $key) { Fail "nim key present" "missing"; return }
    try {
        $list = Invoke-RestMethod -Uri "https://integrate.api.nvidia.com/v1/models" `
            -Headers @{ Authorization = "Bearer $key" } -Method GET -TimeoutSec 20
        $glmM3 = $list.data | Where-Object { $_.id -match "glm-5.2|minimax-m3" } | ForEach-Object { $_.id }
        Ok "nim auth" "key=$(Mask $key) models=$($list.data.Count) glm/m3=$($glmM3 -join ',')"

        # Smoke-test a minimal chat on glm-5.2
        $body = @{
            model    = "z-ai/glm-5.2"
            messages = @(@{ role = "user"; content = "ping" })
            max_tokens = 5
        } | ConvertTo-Json -Depth 6
        try {
            $r = Invoke-RestMethod -Uri "https://integrate.api.nvidia.com/v1/chat/completions" `
                -Headers @{ Authorization = "Bearer $key"; "Content-Type" = "application/json" } `
                -Method POST -Body $body -TimeoutSec 30
            $reply = $r.choices[0].message.content
            Ok "nim chat z-ai/glm-5.2" "reply=$reply"
        } catch {
            Fail "nim chat z-ai/glm-5.2" $_.Exception.Message
        }

        # Smoke-test minimax-m3
        $body = @{
            model    = "minimaxai/minimax-m3"
            messages = @(@{ role = "user"; content = "ping" })
            max_tokens = 5
        } | ConvertTo-Json -Depth 6
        try {
            $r = Invoke-RestMethod -Uri "https://integrate.api.nvidia.com/v1/chat/completions" `
                -Headers @{ Authorization = "Bearer $key"; "Content-Type" = "application/json" } `
                -Method POST -Body $body -TimeoutSec 30
            $reply = $r.choices[0].message.content
            Ok "nim chat minimaxai/minimax-m3" "reply=$reply"
        } catch {
            Fail "nim chat minimaxai/minimax-m3" $_.Exception.Message
        }
    } catch {
        Fail "nim auth" $_.Exception.Message
    }
}

function Test-Wandb {
    Write-Host ""
    Write-Host "=== W&B ==="
    $key = $env:WANDB_API_KEY
    if (-not $key) { Fail "wandb key present" "missing"; return }
    # W&B public GraphQL endpoint; minimal viewer query using API key in basic auth.
    $basic = [Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("api:$key"))
    $query = '{"query":"{ viewer { id username email } }"}'
    try {
        $r = Invoke-RestMethod -Uri "https://api.wandb.ai/graphql" `
            -Headers @{ Authorization = "Basic $basic"; "Content-Type" = "application/json" } `
            -Method POST -Body $query -TimeoutSec 20
        if ($r.errors) { Fail "wandb viewer query" ($r.errors | ConvertTo-Json -Compress); return }
        $v = $r.data.viewer
        Ok "wandb auth" "user=$($v.username) email=$($v.email)"
    } catch {
        Fail "wandb auth" $_.Exception.Message
    }
}

function Test-Kaggle {
    Write-Host ""
    Write-Host "=== Kaggle ==="
    $token = $env:KAGGLE_API_TOKEN
    if (-not $token) { Fail "kaggle token present" "missing"; return }
    try {
        # /mine=true is rejected with 400; just list public kernels to verify auth.
        $r = Invoke-RestMethod -Uri "https://www.kaggle.com/api/v1/kernels/list?page_size=3" `
            -Headers @{ Authorization = "Bearer $token" } -Method GET -TimeoutSec 20
        $count = if ($r -is [array]) { $r.Count } else { 0 }
        Ok "kaggle kernels list" "token=$(Mask $token) returned $count kernel(s)"
    } catch {
        Fail "kaggle kernels list" $_.Exception.Message
    }
}

Test-Zen
Test-NIM
Test-Wandb
Test-Kaggle

Write-Host ""
if ($failed.Count -gt 0) {
    Write-Host ("FAILED: " + ($failed -join ", "))
    exit 1
}
Write-Host "ALL OK"

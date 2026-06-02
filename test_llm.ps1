$headers = @{
    "Authorization" = "Bearer ak-22083206862a22b0f0e36de551313ae9"
    "Content-Type"  = "application/json"
}

$body = @{
    model    = "gpt-5.4"
    messages = @(
        @{ role = "user"; content = "Hello, this is a test message. Reply briefly." }
    )
    max_tokens = 100
} | ConvertTo-Json -Depth 5

Write-Host "Testing: https://aigw.fosunwealth.com/v1/chat/completions" -ForegroundColor Cyan
Write-Host "Model: gpt-5.4`n" -ForegroundColor Cyan

try {
    $response = Invoke-RestMethod -Uri "https://aigw.fosunwealth.com/v1/chat/completions" `
        -Method POST -Headers $headers -Body $body -TimeoutSec 30

    Write-Host "SUCCESS" -ForegroundColor Green
    Write-Host "Reply: $($response.choices[0].message.content)"
    Write-Host "Tokens used: $($response.usage.total_tokens)"
} catch {
    Write-Host "FAILED" -ForegroundColor Red
    Write-Host "Status: $($_.Exception.Response.StatusCode.value__)"
    Write-Host "Error: $($_.ErrorDetails.Message)"
}

$tempFile = Join-Path $env:TEMP "speed_test.bin"
$data = New-Object byte[] (10 * 1024 * 1024)
[IO.File]::WriteAllBytes($tempFile, $data)

$sw = [Diagnostics.Stopwatch]::StartNew()
& scp.exe -P 22 -o StrictHostKeyChecking=no -i C:\Users\sarve\key $tempFile root@151.185.58.96:/tmp/speed_test.bin
$sw.Stop()

Remove-Item $tempFile -Force
& ssh.exe -i C:\Users\sarve\key -o StrictHostKeyChecking=no root@151.185.58.96 "rm -f /tmp/speed_test.bin"

$seconds = $sw.Elapsed.TotalSeconds
$mbps = [math]::Round(10 / $seconds, 2)
Write-Host "Upload Speed: $mbps MB/s (${seconds}s for 10MB)"

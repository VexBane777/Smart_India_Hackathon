$ErrorActionPreference = "Continue"
Set-Location "C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training"

Write-Output "=== OpenSLR Hindi (SLR103) train ==="
Invoke-WebRequest -Uri "https://openslr.trmal.net/resources/103/Hindi_train.tar.gz" -OutFile "data/openslr_hindi/Hindi_train.tar.gz"
Write-Output "=== OpenSLR Hindi (SLR103) test ==="
Invoke-WebRequest -Uri "https://openslr.trmal.net/resources/103/Hindi_test.tar.gz" -OutFile "data/openslr_hindi/Hindi_test.tar.gz"
Write-Output "=== extracting ==="
tar -xzf "data/openslr_hindi/Hindi_train.tar.gz" -C "data/openslr_hindi"
tar -xzf "data/openslr_hindi/Hindi_test.tar.gz" -C "data/openslr_hindi"

Write-Output "=== MUSAN (SLR17, 11G) ==="
Invoke-WebRequest -Uri "https://openslr.trmal.net/resources/17/musan.tar.gz" -OutFile "data/musan/musan.tar.gz"
Write-Output "=== extracting musan ==="
tar -xzf "data/musan/musan.tar.gz" -C "data/musan"

Write-Output "=== ALL DONE ==="

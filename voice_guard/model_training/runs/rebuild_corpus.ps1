$ErrorActionPreference = "Continue"
Set-Location "C:\Users\Tuviksh Murudkar\SIH\voice_guard\model_training"
& ".\.venv313\Scripts\Activate.ps1"

Write-Output "=== [1/3] ASVspoof2019 LA train ==="
kaggle datasets download -d anishsarkar22/asvpoof-2019-dataset-la -p data/extracted --unzip
python data/split_flac_to_wav.py

Write-Output "=== [2/3] ASVspoof2021 LA eval + 2019 LA dev ==="
kaggle datasets download -d wagiartono/asvspoof-2019-and-2021-la -p data/asvspoof2021_la --unzip
python data/prep_asvspoof2021.py

Write-Output "=== [3/3] In-the-Wild ==="
kaggle datasets download -d abdallamohamed312/in-the-wild-dataset -p data/in_the_wild --unzip
if (Test-Path "data/in_the_wild/download") {
    Copy-Item "data/in_the_wild/download" "data/in_the_wild/download.zip" -Force
    Expand-Archive -Path "data/in_the_wild/download.zip" -DestinationPath "data/in_the_wild" -Force
}
python data/prep_in_the_wild.py --held-out-fraction 0.2

Write-Output "=== DONE ==="

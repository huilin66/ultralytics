# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
<#!
Launch the default SUA experiment matrix.

This calls train_sua_yolo.py once. It does not copy or modify the source dataset; the Python launcher creates local
directory links and manifests under runs/sua/metadata.

Examples:
    .\tools\run_sua_all.ps1 -Pretrained auto -Epochs 100 -Device 0
    .\tools\run_sua_all.ps1 -Modalities @('rgbt') -MultimodalModels 'yolov8x-mm2-bf.yaml' -Pretrained yolov8x.pt
    .\tools\run_sua_all.ps1 -DryRun
#>

param(
    [string]$DataRoot = '\\158.132.186.40\isds\huilin\bdd\collected_data\20260211_HMT_data_all\datasets',
    [string[]]$Modalities = @('rgb', 't', 'rgbt'),
    [string]$Models = 'yolo11n.yaml,yolo11s.yaml,yolo11m.yaml,yolo11l.yaml,yolo11x.yaml',
    [string]$MultimodalModels = 'mm2',
    [string]$Pretrained = '',
    [int]$Epochs = 100,
    [int]$ImgSize = 640,
    [int]$Batch = 4,
    [int]$Workers = 4,
    [string]$Device = '',
    [switch]$DryRun
)

$arguments = @(
    'tools/train_sua_yolo.py',
    '--data-root', $DataRoot,
    '--modalities', ($Modalities -join ','),
    '--models', $Models,
    '--multimodal-models', $MultimodalModels,
    '--epochs', $Epochs,
    '--imgsz', $ImgSize,
    '--batch', $Batch,
    '--workers', $Workers
)

if ($Pretrained) {
    $arguments += @('--pretrained', $Pretrained)
}
if ($Device) {
    $arguments += @('--device', $Device)
}
if ($DryRun) {
    $arguments += '--dry-run'
}

python @arguments

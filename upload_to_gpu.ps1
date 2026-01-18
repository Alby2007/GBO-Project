# Upload package to GPU instance
# Replace these with your actual values from vast.ai/RunPod

$PORT = "12345"  # Replace with your SSH port
$IP = "1.2.3.4"  # Replace with your instance IP

Write-Host "=================================================================="
Write-Host "UPLOADING NEURALESE TRAINING PACKAGE"
Write-Host "=================================================================="
Write-Host ""
Write-Host "Target: root@$IP`:$PORT"
Write-Host ""

# Check if package exists
$packagePath = "neuralese_training.tar.gz"
if (-not (Test-Path $packagePath)) {
    Write-Host "ERROR: Package not found at $packagePath"
    Write-Host "Creating package..."
    
    cd ..
    tar -czf neuralese_training.tar.gz `
        gpu_training/ `
        src/environment/deceptive_guide_env.py `
        src/environment/grid_world.py `
        src/environment/builders.py `
        experiments/results/finetune_B_medium/final_model.zip `
        experiments/results/finetune_C_strict/final_model.zip
    
    cd gpu_training
    $packagePath = "..\neuralese_training.tar.gz"
}

Write-Host "Package size: $((Get-Item $packagePath).Length / 1MB) MB"
Write-Host ""
Write-Host "Uploading..."

# Upload using SCP
scp -P $PORT $packagePath root@${IP}:~/

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=================================================================="
    Write-Host "UPLOAD COMPLETE!"
    Write-Host "=================================================================="
    Write-Host ""
    Write-Host "Next steps:"
    Write-Host "1. SSH into instance:"
    Write-Host "   ssh -p $PORT root@$IP"
    Write-Host ""
    Write-Host "2. In Jupyter, run:"
    Write-Host "   !tar -xzf neuralese_training.tar.gz"
    Write-Host "   !cd gpu_training"
    Write-Host ""
    Write-Host "3. Then follow the Jupyter notebook cells I provided"
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "ERROR: Upload failed!"
    Write-Host ""
    Write-Host "Troubleshooting:"
    Write-Host "1. Check your PORT and IP are correct"
    Write-Host "2. Make sure SSH is enabled on the instance"
    Write-Host "3. Try manual upload via Jupyter upload button"
    Write-Host ""
}

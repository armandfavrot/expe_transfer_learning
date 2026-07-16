import sys
from pathlib import Path

import torch


lines = [
    f"Python: {sys.executable}",
    f"PyTorch: {torch.__version__}",
    f"CUDA PyTorch: {torch.version.cuda}",
    f"CUDA disponible: {torch.cuda.is_available()}",
    f"Nombre de GPU: {torch.cuda.device_count()}",
]

if torch.cuda.is_available():
    lines.append(f"GPU: {torch.cuda.get_device_name(0)}")
    tensor = torch.randn(1000, 1000, device="cuda")
    lines.append(f"Tensor: {tensor.device}")
else:
    lines.append("Aucun GPU CUDA accessible depuis cette session.")

report = "\n".join(lines) + "\n"
Path("gpu_test_results.txt").write_text(report, encoding="utf-8")
print(report, end="")

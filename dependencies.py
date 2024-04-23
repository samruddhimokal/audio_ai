package_to_exclude = "demucs==4.1.0a2"
numpy_version = "numpy>=1.22,<1.24"
torch_related_packages = [
    "torch",
    "torchvision",
    "torchaudio",
    "torchmetrics",
    "nvidia-nccl-cu12",
    "triton",
]
fsspec_related_packages = [
    "fsspec",
    "huggingface-hub",
    "datasets",
    "transformers",
]

with open('AutoDiarize/requirements.txt', 'r') as file:
    lines = file.readlines()

new_lines = []
for line in lines:
    if package_to_exclude in line or line.startswith("numpy=="):
        continue
    elif any(pkg for pkg in torch_related_packages if line.startswith(pkg)):
        pkg_name = line.split("==")[0]
        new_lines.append(f"{pkg_name}\n")
    elif any(pkg for pkg in fsspec_related_packages if line.startswith(pkg)):
        pkg_name = line.split("==")[0]
        new_lines.append(f"{pkg_name}\n")
    else:
        new_lines.append(line)

new_lines.append(f'{numpy_version}\n')

with open('final_requirements.txt', 'w') as file:
    file.writelines(new_lines)
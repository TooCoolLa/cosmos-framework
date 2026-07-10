import os, subprocess, sys

DRIVE_ROOT = "/content/drive/MyDrive/models/cosmos3"
HF_TOKEN = os.environ["HF_TOKEN"]
os.makedirs(DRIVE_ROOT, exist_ok=True)

# Install huggingface_hub
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"], check=True)

from huggingface_hub import snapshot_download, hf_hub_download, login

login(token=HF_TOKEN)

print("=" * 50)
print("📦 [1/3] Cosmos3-Nano 基模型 (~50-60 GiB)")
snapshot_download(
    repo_id="nvidia/Cosmos3-Nano", revision="main",
    local_dir=f"{DRIVE_ROOT}/Cosmos3-Nano",
    local_dir_use_symlinks=False, resume_download=True, max_workers=4,
)
print("✅ 完成")

print("📦 [2/3] Wan2.2 VAE (~5 GiB)")
os.makedirs(f"{DRIVE_ROOT}/wan22_vae", exist_ok=True)
hf_hub_download(
    repo_id="Wan-AI/Wan2.2-TI2V-5B",
    revision="921dbaf3f1674a56f47e83fb80a34bac8a8f203e",
    filename="Wan2.2_VAE.pth",
    local_dir=f"{DRIVE_ROOT}/wan22_vae",
    local_dir_use_symlinks=False, resume_download=True,
)
print("✅ 完成")

print("📦 [3/3] Cosmos3-Nano-Reasoner (~16 GiB)")
snapshot_download(
    repo_id="nvidia/Cosmos3-Nano-Reasoner",
    revision="6406357cdc32fbf8db5f51ff7992343803b06961",
    local_dir=f"{DRIVE_ROOT}/Cosmos3-Nano-Reasoner",
    local_dir_use_symlinks=False, resume_download=True, max_workers=4,
)
print("✅ 完成")

# Verify
for label, path in [("Cosmos3-Nano", f"{DRIVE_ROOT}/Cosmos3-Nano"),
                     ("Wan2.2 VAE", f"{DRIVE_ROOT}/wan22_vae/Wan2.2_VAE.pth"),
                     ("Reasoner", f"{DRIVE_ROOT}/Cosmos3-Nano-Reasoner")]:
    ok = os.path.isdir(path) if label != "Wan2.2 VAE" else os.path.isfile(path)
    print(f"  {'✅' if ok else '❌'} {label}")

print("\n🎉 全部下载完成")

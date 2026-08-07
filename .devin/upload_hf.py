"""Upload ForgeLM v1 to HuggingFace Hub — fast mode with hf_transfer."""
import os
if "HF_TOKEN" not in os.environ:
    raise RuntimeError("Set HF_TOKEN environment variable before running (e.g. from your .env file)")
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

from huggingface_hub import HfApi, create_repo, upload_folder

api = HfApi()
user = api.whoami()
username = user["name"]
print(f"Logged in as: {username}")

repo_id = f"{username}/ForgeLM-v1"
print(f"Creating repo: {repo_id}")

create_repo(repo_id, repo_type="model", exist_ok=True, token=os.environ["HF_TOKEN"])
print(f"Repo created: https://huggingface.co/{repo_id}")

# Upload the large file first separately for progress tracking
print(f"Uploading model.safetensors (3.4 GB) with hf_transfer...")
api.upload_file(
    path_or_fileobj="forgelm_hf/model.safetensors",
    path_in_repo="model.safetensors",
    repo_id=repo_id,
    repo_type="model",
    token=os.environ["HF_TOKEN"],
)
print(f"model.safetensors uploaded!")

# Upload the rest (small files)
print(f"Uploading documentation and config files...")
upload_folder(
    repo_id=repo_id,
    folder_path="forgelm_hf",
    repo_type="model",
    token=os.environ["HF_TOKEN"],
    commit_message="Upload ForgeLM v1 — training-free KeyStack port of Qwen2.5-Coder-1.5B",
    allow_patterns=["*.json", "*.md", "*.txt"],
)
print(f"\nUpload complete!")
print(f"Model page: https://huggingface.co/{repo_id}")

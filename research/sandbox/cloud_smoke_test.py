"""Cloud smoke test: run sft_train.py on Vast.ai for 50 steps on forgelm_v7_tiny.

This verifies the full cloud training path end-to-end:
  - Vast.ai API: search offers, create instance, SSH, destroy
  - Repo sync over SFTP
  - Remote provisioning (pip install)
  - sft_train.py with 8B features (dead-param freeze, NLRQ STE, BAdam)
  - Log streaming back to local terminal
  - Checkpoint download

Cost: ~$0.05-0.50 depending on GPU selected (50 steps × 5s/step = 4 min).
Budget cap: $2.00 (well within the $10 account balance).
"""
import os
import sys
from pathlib import Path

# Set up paths
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Ensure VAST_API_KEY is set
if not os.environ.get("VAST_API_KEY"):
    os.environ["VAST_API_KEY"] = "4d97c3d57f8816379cef770cff28b0ba6da82e043fffdbcb81cde41a5192ee24"

from research.cloud.vast_connector import (
    VastConnector, RemoteTrainingSpec, DEFAULT_IMAGE,
)

# Build the training spec for a 50-step smoke test on forgelm_v7_tiny
spec = RemoteTrainingSpec(
    train_args={
        "--config": "forgelm_v7_tiny",
        "--checkpoint": "scratch",
        "--data": "research/data/finetune/tool_use_fc_70.jsonl",
        "--save": "forgelm_v7_tiny_cloud_smoke.safetensors",
        "--optimizer": "badam",
        "--batch-size": 2,
        "--seq-len": 64,
        "--max-steps": 50,
        "--warmup-steps": 5,
        "--lr": 1e-3,
        "--no-grad-checkpoint": True,
        "--no-use-forge-engine": True,
        "--dtype": "fp32",
        "--seed": 42,
        "--no-bitnet-everywhere": True,
        "--no-lora": True,
    },
    data_files=["research/data/finetune/tool_use_fc_70.jsonl"],
    # Vast selection — RTX 4090 (reliable, 24GB, fast image pull)
    gpu_filter="gpu_name=RTX_4090",
    min_vram_gb=20.0,
    min_reliability=0.95,
    disk_gb=50,
    on_demand=True,
    budget=2.0,
    est_sec_per_step=5.0,
    auto_destroy=True,
    stream_logs=True,
    download_output=True,
    poll_interval=15.0,
    startup_timeout=900.0,
    maximize_throughput=False,  # tiny model, no need to tune
    from_scratch=True,
)

print("=" * 70)
print("Cloud Smoke Test: forgelm_v7_tiny, 50 steps on Vast.ai")
print("=" * 70)
print(f"Budget: ${spec.budget:.2f}")
print(f"Config: {spec.train_args['--config']}")
print(f"Steps: {spec.train_args['--max-steps']}")
print(f"Est. time: {50 * spec.est_sec_per_step / 60:.1f} min")
print()

conn = VastConnector(ssh_key=str(Path.home() / ".ssh" / "id_ed25519"))
rc = conn.run_remote_training(spec)
print(f"\nRemote training finished with exit code {rc}")
if rc == 0:
    print("SUCCESS: Cloud smoke test passed!")
    # Verify the checkpoint was downloaded
    ckpt_path = Path("forgelm_v7_tiny_cloud_smoke.safetensors")
    if ckpt_path.exists():
        size_mb = ckpt_path.stat().st_size / 1e6
        print(f"  Checkpoint downloaded: {ckpt_path} ({size_mb:.1f} MB)")
    else:
        print(f"  WARNING: Checkpoint not found at {ckpt_path}")
else:
    print(f"FAILED: exit code {rc}")
sys.exit(rc)

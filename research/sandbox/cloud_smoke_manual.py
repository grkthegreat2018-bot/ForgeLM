"""Cloud smoke test — full lifecycle: rent GPU → sync → provision → train → download.

Runs sft_train.py on Vast.ai for 50 steps on forgelm_v7_tiny.
Cost: ~$0.10-0.50. Budget cap: $2.00.
"""
import os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.cloud.vast_connector import VastConnector, RemoteTrainingSpec
from research.paths import PROJECT_ROOT

API_KEY = os.environ.get("VAST_API_KEY", "4d97c3d57f8816379cef770cff28b0ba6da82e043fffdbcb81cde41a5192ee24")

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
    maximize_throughput=False,
    from_scratch=True,
)

print("=" * 70)
print("Cloud Smoke Test: forgelm_v7_tiny, 50 steps on Vast.ai RTX 4090")
print("=" * 70)

conn = VastConnector(api_key=API_KEY,
                     ssh_key=str(Path.home() / ".ssh" / "id_ed25519"))
rc = conn.run_remote_training(spec)
print(f"\nRemote training exit code: {rc}")
if rc == 0:
    ckpt = PROJECT_ROOT / "forgelm_v7_tiny_cloud_smoke.safetensors"
    if ckpt.exists():
        print(f"SUCCESS: checkpoint {ckpt} ({ckpt.stat().st_size/1e6:.1f} MB)")
    else:
        print("SUCCESS but checkpoint not found locally")
else:
    print(f"FAILED: exit {rc}")
sys.exit(rc)

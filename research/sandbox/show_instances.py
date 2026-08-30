"""Show current Vast.ai instances."""
import json, subprocess, os, sys

VASTAI = r"D:\windsurf\ForgeAI\venv\Scripts\vastai.exe"
env = dict(os.environ)
proc = subprocess.run([VASTAI, "show", "instances", "--raw"],
                      capture_output=True, text=True, env=env, timeout=30)
if proc.returncode != 0:
    print(f"ERROR: {proc.stderr[:300]}")
    sys.exit(1)
instances = json.loads(proc.stdout)
print(f"{len(instances)} instance(s):")
for i in instances:
    print(f"  ID:{i['id']} status:{i.get('actual_status')} "
          f"gpu:{i.get('gpu_name', i.get('label', '?'))} "
          f"${i.get('dph_total', 0):.3f}/h")

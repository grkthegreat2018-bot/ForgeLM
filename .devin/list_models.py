import requests
from pathlib import Path
env = {}
for line in Path("D:/windsurf/ForgeAI/.env").read_text().splitlines():
    line = line.strip()
    if line and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
key = env.get("OPENROUTER_API_KEY", "")
r = requests.get("https://openrouter.ai/api/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
models = r.json().get("data", [])
# Strong OSS models (any price)
oss = [m for m in models if any(x in m["id"].lower() for x in ["llama-4","qwen3","deepseek-r1","deepseek-v3","mistral-large","gemma-4","qwen2.5-72b"])]
for m in sorted(oss, key=lambda x: x["id"])[:30]:
    ctx = m.get("context_length", 0)
    pricing = m.get("pricing", {})
    p_prompt = pricing.get("prompt", "?")
    name = m["id"]
    print(f"{name:<55} ctx={ctx:>8}  ${p_prompt}/tok")

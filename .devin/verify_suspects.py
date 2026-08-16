"""Verify suspicious reward=1.00 trajectories manually."""
import sqlite3, json, re

db = sqlite3.connect("research/data/discovery/tool_infinite.sqlite3")
db.row_factory = sqlite3.Row

for traj_id in [45, 75]:
    r = db.execute("SELECT * FROM tool_trajectories WHERE id=?", (traj_id,)).fetchone()
    print(f"\n{'='*80}")
    print(f"TRAJECTORY #{traj_id}  reward={r['reward']}")
    print(f"TASK: {r['task']}")
    print(f"FINAL ANSWER: {r['final_answer']}")
    
    try:
        calls = json.loads(r["tool_calls"])
        for tc in calls:
            name = tc.get("name")
            args = tc.get("args", tc.get("arguments", {}))
            result = tc.get("result", {})
            if name in ("run_script", "calculate"):
                stdout = result.get("stdout", "")[:300] if isinstance(result, dict) else ""
                stderr = result.get("stderr", "")[:150] if isinstance(result, dict) else ""
                print(f"  TOOL {name}({str(args)[:100]})")
                print(f"    stdout: {stdout}")
                if stderr:
                    print(f"    stderr: {stderr}")
            else:
                print(f"  TOOL {name}({str(args)[:100]}) -> {str(result)[:150]}")
    except Exception as e:
        print(f"  tool_calls parse error: {e}")
        print(f"  raw: {r['tool_calls'][:300]}")

db.close()

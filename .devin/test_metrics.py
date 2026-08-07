import subprocess, sys, tempfile, os, json

code = 'def add(a,b):\n    return a+b\nprint(add(2,3))'
tmp = os.path.join(tempfile.gettempdir(), 'test_metrics.py')
with open(tmp, 'w') as f:
    f.write(code)

wrapper = '''import sys, os, time, tracemalloc
tracemalloc.start()
_t0 = time.process_time()
exec(open(r"%s", encoding="utf-8").read())
_cpu_s = time.process_time() - _t0
_cur, _peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
import json as _json
sys.stderr.write(_json.dumps({"_proc_cpu_s": _cpu_s, "_proc_peak_mem_kb": _peak // 1024}) + "\\n")
''' % tmp

wp = tmp + '_w.py'
with open(wp, 'w') as f:
    f.write(wrapper)

r = subprocess.run([sys.executable, wp], capture_output=True, text=True, timeout=10)
print('stdout:', repr(r.stdout))
print('stderr:', repr(r.stderr))
os.unlink(tmp)
os.unlink(wp)

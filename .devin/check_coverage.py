import sys, re
sys.path.insert(0, r'D:\windsurf\ForgeAI')
from research.prompt_library import build_topic_prompts
from research.prompt_tests import has_tests

prompts = build_topic_prompts()
py_topics = ['python_general','python_math','python_strings','python_algorithms','python_oop','python_file_io']
for t in py_topics:
    plist = prompts.get(t, [])
    n_tests = sum(1 for p in plist if (lambda m: m and has_tests(m.group(1)))(re.search(r'def\s+(\w+)', p)))
    print(f'{t:25s} {n_tests:3d}/{len(plist):3d} have tests')

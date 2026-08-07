import json
with open('research/checkpoints/forgelm_v2_airmoe/manifest.json') as f:
    m = json.load(f)
experts = m.get('experts', [])
print(f'Total experts: {len(experts)}')
if experts:
    print('Sample expert:', json.dumps(experts[0], indent=2))
bundles = m.get('bundles', [])
print(f'Bundles: {len(bundles)}')
for b in bundles:
    print(f"  {b.get('name')}: {b.get('file_path')}")
# Check topics mapping
topics = m.get('topics', {})
print(f'Topics: {topics}')

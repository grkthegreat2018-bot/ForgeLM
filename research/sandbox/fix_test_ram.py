"""Fix gpu_total_ram values in test_vast_connector.py from GB to MB."""
import re
from pathlib import Path

p = Path(r"D:\windsurf\ForgeAI\tests\unit\test_vast_connector.py")
text = p.read_text()

def replace_ram(m):
    val = int(m.group(1))
    if val <= 128:  # GB → MB
        return '"gpu_total_ram": ' + str(val * 1024)
    return m.group(0)

text = re.sub(r'"gpu_total_ram": (\d+)', replace_ram, text)
text = text.replace("gpu_total_ram = total VRAM in GB", "gpu_total_ram = total VRAM in MB")
p.write_text(text)
print("Done - replaced", len(re.findall(r'"gpu_total_ram": \d+', text)), "values")

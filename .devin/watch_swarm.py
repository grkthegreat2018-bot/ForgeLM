"""Live tail of the research team swarm output with color coding.

Usage:
    python D:\\windsurf\\ForgeAI\\.devin\\watch_swarm.py
    python D:\\windsurf\\ForgeAI\\.devin\\watch_swarm.py --tail 50
"""
import sys
import os
import glob
import time
import argparse

# ANSI colors
COLORS = {
    "search":   "\033[36m",   # cyan
    "result":   "\033[32m",   # green
    "critique": "\033[33m",   # yellow
    "draft":    "\033[35m",   # magenta
    "tool":     "\033[34m",   # blue
    "doc":      "\033[32m",   # green
    "chat":     "\033[90m",   # dark gray
    "header":   "\033[97m",   # white
    "reset":    "\033[0m",
}

# Strip ANSI escape codes from input
import re
ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')


def find_latest_log():
    """Find the most recently updated swarm output file."""
    pattern = os.path.expandvars(
        r"%LOCALAPPDATA%\Temp\devin.exe-overflows\shell-*\content.txt"
    )
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def colorize(line: str) -> str:
    """Add color based on message type tags."""
    clean = ANSI_RE.sub('', line)
    if "[search]" in clean:
        return f"{COLORS['search']}{clean}{COLORS['reset']}"
    elif "[result]" in clean:
        return f"{COLORS['result']}{clean}{COLORS['reset']}"
    elif "[critique]" in clean:
        return f"{COLORS['critique']}{clean}{COLORS['reset']}"
    elif "[draft]" in clean:
        return f"{COLORS['draft']}{clean}{COLORS['reset']}"
    elif "[tool]" in clean:
        return f"{COLORS['tool']}{clean}{COLORS['reset']}"
    elif "[doc]" in clean:
        return f"{COLORS['doc']}{clean}{COLORS['reset']}"
    elif "[chat]" in clean:
        return f"{COLORS['chat']}{clean}{COLORS['reset']}"
    elif "ROUND" in clean or "Phase" in clean or "====" in clean:
        return f"{COLORS['header']}{clean}{COLORS['reset']}"
    return clean


def tail_live(path: str, tail_lines: int):
    """Tail a file and stream new lines live using file size polling."""
    # Read last N lines for initial display
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        all_lines = f.readlines()
    for line in all_lines[-tail_lines:]:
        line = line.rstrip("\n")
        if line.strip():
            print(colorize(line))

    print(f"\n  {'='*60}")
    print(f"  LIVE — following new output (Ctrl+C to stop)")
    print(f"  {'='*60}\n")

    # Poll for file size changes and read new content
    last_size = os.path.getsize(path)
    buffer = ""
    while True:
        time.sleep(0.5)
        try:
            current_size = os.path.getsize(path)
        except OSError:
            continue
        if current_size <= last_size:
            continue
        # Read new bytes
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(last_size)
            new_data = f.read(current_size - last_size)
        last_size = current_size
        buffer += new_data
        # Print complete lines, keep partial in buffer
        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line = line.rstrip("\r")
            if line.strip():
                print(colorize(line))


def main():
    parser = argparse.ArgumentParser(description="Live tail of swarm output")
    parser.add_argument("--tail", type=int, default=30, help="Number of initial lines to show")
    parser.add_argument("--file", type=str, default=None, help="Specific file to tail")
    args = parser.parse_args()

    path = args.file or find_latest_log()
    if not path or not os.path.exists(path):
        print("No swarm output file found. Is the swarm running?")
        print(f"Searched: %LOCALAPPDATA%\\Temp\\devin.exe-overflows\\shell-*\\content.txt")
        sys.exit(1)

    print(f"  Watching: {path}")
    print(f"  Tail: {args.tail} lines\n")

    try:
        tail_live(path, args.tail)
    except KeyboardInterrupt:
        print(f"\n{COLORS['reset']}  Stopped.")


if __name__ == "__main__":
    main()

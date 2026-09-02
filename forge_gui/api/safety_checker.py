"""Sandbox safety checker — layered edit validation for the agentic loop.

Three layers of checking, applied after each edit finishes:
1. **AST analysis** — parse Python files, block dangerous calls
   (os.system, subprocess, eval, exec, __import__, file deletion, etc.)
2. **Pattern matching** — regex scan for dangerous strings in any file type
   (rm -rf, curl|wget piped to sh, secrets/keys, SQL injection patterns)
3. **Heuristics** — file type checks, path sensitivity, import analysis

Three-strikes policy: after 3 safety violations in one agent run, the
loop terminates and the problem area is flagged for investigation.

The checker is pure-stdlib (no Qt/torch) so it can be unit-tested anywhere.
"""
from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── result types ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SafetyVerdict:
    safe: bool
    layer: str = ""           # "ast" | "pattern" | "heuristic" | "ok"
    reason: str = ""
    severity: str = "info"    # "info" | "warn" | "danger"
    suggestions: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.safe


# ── Layer 1: AST analysis ───────────────────────────────────────────────

# Dangerous calls that should never appear in agent-generated Python
_DANGEROUS_CALLS = {
    "os.system", "os.popen", "os.exec", "os.execv", "os.execve",
    "os.spawn", "os.spawnl", "os.spawnv",
    "subprocess.call", "subprocess.run", "subprocess.Popen",
    "subprocess.check_call", "subprocess.check_output",
    "eval", "exec", "compile",
    "__import__",
    "os.remove", "os.unlink", "os.rmdir", "os.removedirs",
    "shutil.rmtree", "shutil.copy", "shutil.copytree",
    "os.chmod", "os.chown",
    "pickle.loads", "pickle.load",
    "marshal.loads",
    "ctypes.CDLL", "ctypes.cdll",
    "socket.socket",  # network access from sandbox
}

# Dangerous attributes (e.g. obj.__subclasses__, obj.__globals__)
_DANGEROUS_ATTRS = {
    "__subclasses__", "__globals__", "__builtins__",
    "__code__", "__func__", "__class__",
}

# Allowed imports (anything else triggers a warning)
_ALLOWED_IMPORTS = {
    "math", "json", "re", "os.path", "pathlib", "typing",
    "dataclasses", "collections", "itertools", "functools",
    "string", "textwrap", "copy", "datetime", "time",
    "functools", "abc", "enum", "io", "csv", "hashlib",
    "base64", "struct", "binascii", "statistics", "random",
    "unittest.mock", "pytest", "numpy", "torch",
}


def _get_call_name(node: ast.AST) -> str:
    """Get the dotted name from a Call node (e.g. os.system → 'os.system')."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _get_call_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def check_ast(content: str, filename: str = "") -> SafetyVerdict:
    """Layer 1: Parse Python content with AST and check for dangerous calls.

    Only applies to .py files. Non-Python files pass this layer automatically.
    """
    if not filename.endswith(".py"):
        return SafetyVerdict(safe=True, layer="ast", reason="non-Python file")

    try:
        tree = ast.parse(content, filename=filename or "<edit>")
    except SyntaxError as e:
        return SafetyVerdict(
            safe=False, layer="ast", severity="warn",
            reason=f"SyntaxError in generated code: {e}",
            suggestions=["Fix the syntax error before proceeding"])

    violations: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _get_call_name(node.func)
            # check against dangerous calls
            for dangerous in _DANGEROUS_CALLS:
                if name == dangerous or name.startswith(dangerous + "."):
                    violations.append(f"dangerous call: {name}")
            # check for __import__ pattern
            if name == "__import__":
                violations.append("dynamic import via __import__")
        elif isinstance(node, ast.Attribute):
            if node.attr in _DANGEROUS_ATTRS:
                violations.append(f"dangerous attribute: {node.attr}")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            # warn on non-allowed imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod not in {a.split(".")[0] for a in _ALLOWED_IMPORTS}:
                        violations.append(f"non-allowed import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                mod = (node.module or "").split(".")[0]
                if mod and mod not in {a.split(".")[0] for a in _ALLOWED_IMPORTS}:
                    violations.append(f"non-allowed import: {node.module}")

    if violations:
        return SafetyVerdict(
            safe=False, layer="ast", severity="danger",
            reason="; ".join(violations[:5]),
            suggestions=[f"Remove or replace: {v}" for v in violations[:3]])
    return SafetyVerdict(safe=True, layer="ast", reason="clean AST")


# ── Layer 2: Pattern matching ──────────────────────────────────────────

# Dangerous patterns in any file type
_DANGEROUS_PATTERNS = [
    # Shell destruction
    (re.compile(r"rm\s+-rf?\s+[/~]"), "rm -rf on root/home"),
    (re.compile(r"rm\s+-rf?\s+\*"), "rm -rf wildcard"),
    (re.compile(r"rm\s+-rf?\s+\."), "rm -rf current dir"),
    # Shell injection
    (re.compile(r"\|\s*(sh|bash|zsh|fish)\b"), "pipe to shell"),
    (re.compile(r"curl\s+.*\|\s*(sh|bash|python)"), "curl pipe to interpreter"),
    (re.compile(r"wget\s+.*\|\s*(sh|bash|python)"), "wget pipe to interpreter"),
    # Secrets / credentials
    (re.compile(r"(?i)(api[_-]?key|secret[_-]?key|password|passwd|token)\s*[=:]\s*['\"][^'\"]{8,}"), "hardcoded secret"),
    (re.compile(r"(?i)BEGIN\s+.*?(RSA|OPENSSH|EC|DSA|PGP)?\s*PRIVATE\s+KEY"), "embedded private key"),
    # Network exfiltration
    (re.compile(r"(?i)(nc|netcat|ncat)\s+.*-\w*e"), "netcat listener"),
    (re.compile(r"(?i)curl\s+.*(-o|--output)\s+/dev/(null|stdout)"), "silent curl (possible exfil)"),
    # Filesystem destruction
    (re.compile(r"(?i)mkfs\.\w+"), "filesystem format"),
    (re.compile(r"(?i)dd\s+if=.*of=/dev/"), "dd to device"),
    (re.compile(r">\s*/dev/sd[a-z]"), "write to raw device"),
    # Privilege escalation
    (re.compile(r"(?i)sudo\s+"), "sudo usage"),
    (re.compile(r"(?i)chmod\s+\+s\s"), "setuid bit"),
    # Python-specific dangerous patterns in non-.py files too
    (re.compile(r"(?i)os\.system\s*\("), "os.system call"),
    (re.compile(r"(?i)subprocess\.\w+\s*\("), "subprocess call"),
    (re.compile(r"(?i)__import__\s*\("), "dynamic import"),
    # Git force operations (can destroy history)
    (re.compile(r"git\s+push\s+.*--force"), "force push"),
    (re.compile(r"git\s+reset\s+--hard"), "hard reset"),
    (re.compile(r"git\s+clean\s+-[a-z]*f"), "force clean"),
    # ── Real-world side effects: email, forums, web POSTs ──
    # Email sending (smtplib, SMTP)
    (re.compile(r"(?i)smtplib\.(SMTP|sendmail)"), "email sending via smtplib"),
    (re.compile(r"(?i)\.sendmail\s*\("), "sendmail call"),
    (re.compile(r"(?i)SMTP\s*\("), "SMTP connection"),
    (re.compile(r"(?i)smtplib\.SMTP"), "smtplib SMTP import"),
    # Forum/social media posting
    (re.compile(r"(?i)(reddit|twitter|facebook|linkedin|discord|slack|telegram)"
                r"\.com/(api|post|submit|message)"), "social media API post"),
    (re.compile(r"(?i)(requests|httpx|aiohttp)\.(post|put|patch|delete)\s*\("),
     "HTTP write method (POST/PUT/PATCH/DELETE)"),
    (re.compile(r"(?i)urllib\.request\.urlopen\s*.*POST"),
     "urllib POST request"),
    # Webhooks
    (re.compile(r"(?i)webhook"), "webhook reference"),
    # curl/wget POST (sending data to external servers)
    (re.compile(r"(?i)curl\s+.*(-X\s*(POST|PUT|DELETE)|-d\s|--data)"),
     "curl POST/PUT/DELETE (sending data externally)"),
    (re.compile(r"(?i)wget\s+.*--post"), "wget POST (sending data externally)"),
    # Twilio/SMS/phone
    (re.compile(r"(?i)twilio"), "Twilio SMS/phone API"),
    # Slack/Discord/Teams bots
    (re.compile(r"(?i)(slack_sdk|discord\.py|pymsteams|discord\.Client)"), "messaging bot library"),
    (re.compile(r"(?i)import\s+discord\b"), "discord bot import"),
    # SSH/SCP to external hosts (data exfiltration)
    (re.compile(r"(?i)(scp|sftp)\s+.*@"), "SCP/SFTP to remote host"),
    (re.compile(r"(?i)ssh\s+.*@"), "SSH to remote host"),
    # Database writes to external hosts
    (re.compile(r"(?i)(psycopg2|pymysql|mysql-connector).*connect.*@"
                r"|redis.*\.set\s*\(|redis.*\.publish\s*\("),
     "external database write"),
]

# Sensitive paths that should never be written to
_SENSITIVE_PATHS = [
    "/etc/", "/usr/", "/bin/", "/sbin/", "/boot/", "/dev/",
    ".ssh/", ".aws/", ".env", ".git/config",
    "c:/windows/", "c:/program files/",
]


def check_patterns(content: str, filename: str = "") -> SafetyVerdict:
    """Layer 2: Scan for dangerous string patterns in any file type."""
    violations: list[str] = []
    for pattern, desc in _DANGEROUS_PATTERNS:
        if pattern.search(content):
            violations.append(desc)

    if violations:
        return SafetyVerdict(
            safe=False, layer="pattern", severity="danger",
            reason="; ".join(violations[:5]),
            suggestions=[f"Remove: {v}" for v in violations[:3]])
    return SafetyVerdict(safe=True, layer="pattern", reason="clean patterns")


def check_path_safety(path: str) -> SafetyVerdict:
    """Check if a target path is sensitive (should not be written to)."""
    norm = path.replace("\\", "/").lower()
    for sensitive in _SENSITIVE_PATHS:
        s = sensitive.lower()
        if s in norm:
            return SafetyVerdict(
                safe=False, layer="heuristic", severity="danger",
                reason=f"sensitive path: {path} (matches {sensitive})",
                suggestions=[f"Avoid writing to {sensitive}"])
    return SafetyVerdict(safe=True, layer="heuristic", reason="path OK")


# ── Layer 3: Heuristics ────────────────────────────────────────────────

# File types the agent is allowed to create/edit
_ALLOWED_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".json", ".yaml", ".yml",
    ".md", ".txt", ".csv", ".tsv", ".html", ".css", ".xml",
    ".toml", ".cfg", ".ini", ".sh", ".bat", ".ps1",
    ".sql", ".graphql", ".proto", ".thrift",
    ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".java", ".kt",
    ".rb", ".php", ".swift", ".scala", ".clj", ".ex", ".exs",
    ".lua", ".r", ".jl", ".pl", ".tcl",
    ".gitignore", ".dockerignore", ".editorconfig",
    ".env.example", ".env.sample",  # .env itself is blocked by path check
}

# File types that are always dangerous to write
_BLOCKED_EXTENSIONS = {
    ".exe", ".dll", ".so", ".dylib", ".bin", ".obj", ".o",
    ".pyc", ".pyd", ".pyo",
    ".img", ".iso", ".vmdk", ".vdi",
    ".msi", ".deb", ".rpm", ".pkg",
    ".lock",  # lock files can cause deadlocks
}


def check_heuristics(content: str, filename: str,
                     file_size: int = 0) -> SafetyVerdict:
    """Layer 3: File-type checks, size limits, and content heuristics."""
    ext = Path(filename).suffix.lower()

    # blocked extensions
    if ext in _BLOCKED_EXTENSIONS:
        return SafetyVerdict(
            safe=False, layer="heuristic", severity="danger",
            reason=f"blocked file type: {ext}",
            suggestions=[f"Agent cannot create {ext} files"])

    # size limit (512KB max for agent edits)
    if file_size > 512_000:
        return SafetyVerdict(
            safe=False, layer="heuristic", severity="warn",
            reason=f"file too large: {file_size} bytes (max 512KB)",
            suggestions=["Split into smaller files"])

    # check for binary content (null bytes)
    if "\x00" in content[:1024]:
        return SafetyVerdict(
            safe=False, layer="heuristic", severity="danger",
            reason="binary content detected",
            suggestions=["Agent cannot write binary files"])

    # warn on unknown extensions
    if ext and ext not in _ALLOWED_EXTENSIONS and ext not in _BLOCKED_EXTENSIONS:
        return SafetyVerdict(
            safe=True, layer="heuristic", severity="warn",
            reason=f"uncommon file type: {ext} (allowed with warning)")

    return SafetyVerdict(safe=True, layer="heuristic", reason="heuristics OK")


# ── Combined checker ────────────────────────────────────────────────────

def check_edit(content: str, filename: str, file_size: int = 0) -> SafetyVerdict:
    """Run all three layers and return the combined verdict.

    If any layer fails, the edit is blocked. The most severe verdict wins.
    """
    # Layer 0: path safety
    pv = check_path_safety(filename)
    if not pv:
        return pv

    # Layer 1: AST (Python only)
    av = check_ast(content, filename)
    if not av:
        return av

    # Layer 2: patterns (any file)
    patv = check_patterns(content, filename)
    if not patv:
        return patv

    # Layer 3: heuristics
    hv = check_heuristics(content, filename, file_size or len(content))
    if not hv:
        return hv

    # all passed — return the most informative OK
    return SafetyVerdict(safe=True, layer="ok", reason="all layers passed")


def check_command(cmd: str) -> SafetyVerdict:
    """Validate a shell command before execution."""
    # pattern-check the command string
    pv = check_patterns(cmd, "")
    if not pv:
        return pv

    # block specific dangerous commands
    cmd_lower = cmd.lower().strip()
    blocked_prefixes = (
        "rm -rf", "mkfs", "dd if=", "shutdown", "reboot",
        "halt", "init 0", "init 6",
        "git push --force", "git reset --hard",
        # email/web-send blocking
        "curl -x post", "curl -d ", "curl --data",
        "wget --post", "scp ", "sftp ", "ssh ",
        "sendmail", "mail -s", "mailx",
    )
    for prefix in blocked_prefixes:
        if cmd_lower.startswith(prefix):
            return SafetyVerdict(
                safe=False, layer="pattern", severity="danger",
                reason=f"blocked command: {prefix}",
                suggestions=[f"Do not run: {prefix}"])

    # ── process protection: block killing GUI/backend/self ──
    proc_v = check_process_kill(cmd)
    if not proc_v:
        return proc_v

    return SafetyVerdict(safe=True, layer="ok", reason="command OK")


# ── Process protection ──────────────────────────────────────────────────

# Process names that must never be killed by the agent
# (prevents self-killing, GUI shutdown, backend disruption)
_PROTECTED_PROCESS_NAMES = {
    "forge_gui", "forge_server", "forge_engine",
    "launch_gui", "forgeengine",
    "python", "python3", "pythonw",  # could be the GUI itself
    "sft_train", "flash_serve",
    "forgeai", "forge_ai",
    "qt6gui", "pyside6",
}

# Patterns that indicate a process kill command
_KILL_PATTERNS = [
    re.compile(r"(?i)\bkill\b\s+(-9\s+)?(-\w+\s+)*(\d+|python|forge)"),
    re.compile(r"(?i)\btaskkill\b"),
    re.compile(r"(?i)\bstop-process\b"),
    re.compile(r"(?i)\bpkill\b"),
    re.compile(r"(?i)\bkillall\b"),
]


def check_process_kill(cmd: str) -> SafetyVerdict:
    """Check if a command attempts to kill protected processes.

    Blocks the agent from:
    - Killing the GUI process (forge_gui, launch_gui)
    - Killing the backend (forge_server, forge_engine)
    - Killing itself (python processes running the agent)
    - Killing training jobs (sft_train)
    """
    cmd_lower = cmd.lower().strip()

    # check if this is a kill command at all
    is_kill_cmd = any(p.search(cmd) for p in _KILL_PATTERNS)
    if not is_kill_cmd:
        return SafetyVerdict(safe=True, layer="ok", reason="not a kill command")

    # check if any protected process name appears in the command
    for proc_name in _PROTECTED_PROCESS_NAMES:
        if proc_name in cmd_lower:
            return SafetyVerdict(
                safe=False, layer="heuristic", severity="danger",
                reason=f"attempted to kill protected process: {proc_name}",
                suggestions=[
                    f"Cannot kill {proc_name} — this is a protected process",
                    "The agent cannot kill the GUI, backend, or itself"])

    # also check for PID-based kills of our own process
    # (we can't know all PIDs, but blocking kill commands targeting
    # python/forge processes covers the main risk)
    return SafetyVerdict(safe=True, layer="ok",
                         reason="kill command targets non-protected process")


# ── Strike tracker ──────────────────────────────────────────────────────

@dataclass
class StrikeTracker:
    """Tracks safety violations across an agent run.

    After max_strikes violations, the loop should terminate and the problem
    area is flagged for investigation.
    """
    max_strikes: int = 3
    strikes: list[SafetyVerdict] = field(default_factory=list)
    flagged_areas: list[str] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.strikes)

    @property
    def terminated(self) -> bool:
        return self.count >= self.max_strikes

    def record(self, verdict: SafetyVerdict, context: str = "") -> None:
        """Record a safety violation. Returns nothing (check .terminated)."""
        if not verdict.safe:
            self.strikes.append(verdict)
            if context:
                self.flagged_areas.append(
                    f"{context}: {verdict.reason}")

    def reset(self) -> None:
        self.strikes.clear()
        self.flagged_areas.clear()

    def summary(self) -> str:
        if not self.strikes:
            return "no safety violations"
        parts = [f"{self.count}/{self.max_strikes} strikes"]
        if self.terminated:
            parts.append("TERMINATED")
        if self.flagged_areas:
            parts.append("flagged: " + "; ".join(self.flagged_areas[-3:]))
        return " · ".join(parts)

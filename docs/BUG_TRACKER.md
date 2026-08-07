# ForgeAI Bug Tracker

Documented bugs, root causes, and fixes encountered during development.
Append new bugs to the bottom. Do not delete resolved entries — they serve as
a knowledge base for future debugging.

---

## BUG-001: Triton generates invalid `sm_120a` target on consumer Blackwell

**Status:** Resolved (patched locally; upstream PR #9734 was reverted)
**Date:** 2026-07-25
**Severity:** Critical (blocks all Triton kernels on RTX 5070/5080/5090)
**Hardware:** NVIDIA RTX 5070 (sm_120, consumer Blackwell)
**Software:** `triton-windows==3.4.0.post21`, PyTorch 2.8.0+cu128

### Symptom
Every Triton kernel launch crashes with:
```
RuntimeError: Triton Error [CUDA]: an illegal memory access was encountered
```
This affects `torch.compile`, Liger Kernel, and any custom `@triton.jit` kernel.
Basic PyTorch CUDA ops (CUBLAS, SDPA, etc.) work fine — only Triton-compiled
kernels crash.

### Root Cause
`triton/backends/nvidia/compiler.py` line 95:
```python
suffix = "a" if capability >= 90 else ""
```
This generates `sm_120a` for consumer Blackwell (sm_120). Consumer Blackwell
has **no tensor memory (tcgen05)** — only datacenter Blackwell (sm_100a, B100/B200)
and Hopper (sm_90a) have the "a" variant. Passing `sm_120a` to LLVM/ptxas causes
instruction selection for tensor memory features that don't exist on the hardware,
producing undefined machine code that crashes at runtime.

Additionally, the `make_ttgir` pipeline runs datacenter Blackwell tensor memory
passes (`add_hoist_tmem_alloc`, `add_promote_lhs_to_tmem`, `add_warp_specialize`,
`add_optimize_tmem_layouts`, `add_interleave_tmem`) for `capability // 10 >= 10`,
which includes sm_120. These passes emit tcgen05 instructions that crash on
consumer hardware.

The PTX `.target` regex `r'\.target sm_\d+'` also fails to match `sm_120a`
(the `a` suffix), so the invalid target passes through uncorrected.

### Fix
Three patches to `venv\Lib\site-packages\triton\backends\nvidia\compiler.py`:

1. **`sm_arch_from_capability`** — only add "a" for `90 <= capability < 120`:
```python
suffix = "a" if 90 <= capability < 120 else ""
```

2. **PTX `.target` regex** — handle the "a" suffix:
```python
ret = re.sub(r'\.target sm_\d+a?', f'.target sm_{capability}', ret, flags=re.MULTILINE)
```

3. **`make_ttgir` pipeline** — split `capability // 10 >= 10` into:
   - `== 10`: datacenter Blackwell (sm_100a) — run tmem passes
   - `>= 12`: consumer Blackwell (sm_120) — skip tmem passes, use MMAv2 path

After patching, clear `~/.triton/cache` so kernels recompile with `sm_120`.

### Verification
- Basic Triton kernels (elementwise, reduction, softmax) now work on sm_120.
- `torch.compile` delivers 13,240 tok/s on 360M MLA (unchanged from pre-patch).
- Liger CE kernel still crashes (see BUG-002) — additional Triton issues remain.

### Upstream Status
The fix originates from PR #9734 ([NVIDIA] Fix PTX codegen segfaults on consumer
Blackwell (sm_120)), which was **reverted** in commit 55337eb. The revert means
official `triton-windows` releases will continue to crash on RTX 5070/5080/5090
until the fix is re-landed. The local patch must be re-applied after any
`triton-windows` reinstall/upgrade.

---

## BUG-002: Liger Kernel Triton CE kernel crashes on consumer Blackwell

**Status:** Unresolved (workaround: use pure-PyTorch Chunked CE)
**Date:** 2026-07-25
**Severity:** High (blocks Liger FLCE on RTX 5070)
**Hardware:** NVIDIA RTX 5070 (sm_120)
**Software:** `liger-kernel==0.8.1`, `triton-windows==3.4.0.post21` (patched per BUG-001)

### Symptom
`LigerFusedLinearCrossEntropyLoss` crashes with `illegal memory access` in the
`liger_cross_entropy_kernel` Triton kernel, even after applying the BUG-001
sm_120a patch. Fails at all vocab sizes tested (32000, 128256, 151665).

### Root Cause
Beyond the `sm_120a` target bug (BUG-001), there are additional Triton issues
on consumer Blackwell that affect Liger's CE kernel specifically. The exact
failure point is inside the Triton-compiled `liger_cross_entropy_kernel` launch.
Reducing `num_warps` from 32 to 8 did not resolve the issue. The kernel uses
online softmax with two passes over the vocab dimension and `BLOCK_SIZE=32768`.

Liger v0.8.1 added a CuTe DSL backend (`LIGER_KERNEL_IMPL=cutedsl`) for
Blackwell, but it targets datacenter Blackwell (sm_100a, B200) and requires
the `cutlass` Python package, which is not pip-installable on Windows.

### Workaround
Use the pure-PyTorch `ChunkedLinearCrossEntropy` in `research/chunked_ce.py`
(activated via `--chunked-ce` flag). It provides the same memory savings
(~1.86 GB at batch 2) without any Triton dependency, at the cost of ~24%
slower per-step throughput. Gradients match `F.cross_entropy` exactly.

### Future Investigation
- Monitor Liger upstream for sm_120-specific kernel tuning.
- Test with `triton-windows` newer versions if sm_120 support lands upstream.
- Consider CUTLASS CuTe DSL backend once `cutlass` is pip-installable on Windows.

---

## BUG-003: `liger-kernel` pip install fails on `triton>=2.3.1` dependency

**Status:** Resolved (workaround: `--no-deps`)
**Date:** 2026-07-25
**Severity:** Low (install blocker, easy workaround)
**Software:** `liger-kernel==0.8.1`, `triton-windows==3.4.0.post21`

### Symptom
```
ERROR: Could not find a version that satisfies the requirement triton>=2.3.1
(from liger-kernel) (from versions: none)
```

### Root Cause
Liger Kernel pins `triton>=2.3.1` as a hard dependency, but ForgeAI uses
`triton-windows` (the Windows fork), which registers as a different package
name. pip's resolver cannot satisfy the `triton` requirement because only
`triton-windows` is installed.

### Fix
Install with `--no-deps` to skip the dependency check:
```powershell
D:\windsurf\ForgeAI\venv\Scripts\pip.exe install --no-deps liger-kernel==0.8.1
```
Liger imports and loads correctly with `triton-windows` (version 3.4.0)
despite the unmet pip pin. The `import triton` call succeeds because
`triton-windows` installs under the `triton` module namespace.

### Note
This must be re-run after any `liger-kernel` upgrade. Do NOT add `liger-kernel`
to `requirements.txt` without a comment noting the `--no-deps` requirement.

---

## BUG-004: Gigatoken `encode_files` returns flat stream (no document boundaries)

**Status:** Resolved (design decision: use `encode_batch_list` instead)
**Date:** 2026-07-25
**Severity:** Medium (affects prep_data.py architecture choice)
**Software:** `gigatoken==0.9.0`

### Symptom
`gt.Tokenizer.encode_files(TextFileSource([path], separator=b'\n\n'))` returns
an awkward Array of length 1 containing all tokens concatenated, rather than
a per-document array. Slicing `arr[1]` raises:
```
IndexError: cannot slice ListOffsetArray (of length 1) with 1
```

### Root Cause
`encode_files` is designed for bulk pre-tokenization where the consumer does
not need document boundaries — it produces a flat token stream for direct
consumption by training data loaders. The `separator` parameter splits the
input file for parallel reading but does not preserve boundaries in the output.

### Fix
Use `gt.Tokenizer.encode_batch_list(list[str])` instead, which returns
`list[list[int]]` preserving per-document boundaries. This allows per-document
EOS token insertion in `prep_data.py`'s `build_binary_file`. The native batched
path still parallelizes in Rust and delivers ~17x speedup on desktop CPUs.

### Design Note
Documented in `AGENTS.md` and `LLM_Research.md` (entry 45). The disk-spill
path (`encode_files`) would have given the full ~691x speedup on EPYC, but
the loss of document boundaries makes per-document EOS insertion impossible
without fragile delimiter-recovery logic. `encode_batch_list` is the correct
trade-off for streaming pipelines.

---

## BUG-005: Chunked CE is slower per-step than full logits (throughput trade-off)

**Status:** Resolved (documented as expected trade-off)
**Date:** 2026-07-25
**Severity:** Low (expected behavior, not a bug per se)
**Hardware:** NVIDIA RTX 5070 (sm_120)

### Symptom
`--chunked-ce` at batch 2 delivers 7,832 tok/s vs 10,314 tok/s baseline
(24% slower), despite saving 1.86 GB VRAM. Increasing batch to 4 with
`--chunked-ce` (8,900 tok/s) still does not beat the batch 2 `--compile`
baseline (13,240 tok/s).

### Root Cause
Chunking the token dimension creates many small GEMMs (`[chunk, V] @ [V, H]`)
instead of one large GEMM (`[B*T, V] @ [V, H]`). CUBLAS is less efficient on
small matrices due to kernel launch overhead and reduced parallelism. At
chunk_size=256 and B*T=2048, there are 8 chunks per step, each requiring
forward + backward GEMMs — 16 small GEMMs vs 2 large ones.

Larger chunk sizes (1024, 2048) were tested but increased memory pressure
and were even slower (3,893 tok/s at chunk_size=1024, batch 4).

### Resolution
This is a fundamental trade-off of chunked CE, not a fixable bug. The value
is in **memory headroom**, not throughput. Use `--chunked-ce` when:
- You need to fit a larger batch size that would otherwise OOM
- You need memory for gradient checkpointing or longer sequences
- You cannot use `--compile` (e.g., custom autograd Functions in the graph)

For maximum throughput, use `--compile` without `--chunked-ce` (13,240 tok/s).

### Measured Results (360M MLA, RTX 5070, 40 steps)
| Config | tok/s | VRAM |
|---|---|---|
| Batch 2, no CE | 10,314 | 9.35 GB |
| Batch 2, `--compile` | 13,240 | 7.53 GB |
| Batch 2, `--chunked-ce` | 7,832 | 7.49 GB |
| Batch 3, `--chunked-ce` | 8,706 | 8.44 GB |
| Batch 4, `--chunked-ce` | 8,900 | 9.39 GB |
| Batch 4, no CE | 1,069 | 14.35 GB (CPU spill) |

---

## BUG-006: `torch.compile` + `--chunked-ce` causes graph breaks

**Status:** Resolved (auto-disable compile when chunked CE is set)
**Date:** 2026-07-25
**Severity:** Low (handled in code)

### Symptom
Combining `--compile` with `--chunked-ce` would cause `torch.compile` graph
breaks due to the custom `torch.autograd.Function` (`ChunkedLinearCrossEntropy`)
in the forward graph. Inductor cannot trace through custom autograd Functions,
falling back to eager for that segment and losing the compile speedup.

### Root Cause
`torch.compile` (Inductor) traces the forward graph and attempts to fuse
operations into Triton kernels. Custom `torch.autograd.Function` subclasses
are opaque to the tracer — Inductor cannot inspect the `forward`/`backward`
logic and inserts a graph break, splitting the compiled graph into pieces.
This eliminates most of the compile speedup.

### Fix
`train.py` and `sft_align.py` automatically disable `--compile` when
`--chunked-ce` is set, printing a warning:
```python
if args.chunked_ce:
    cfg.use_chunked_ce = True
    if args.compile:
        print("WARNING: --chunked-ce + --compile may cause graph breaks. Disabling compile.")
        args.compile = False
```

### Note
This is a known PyTorch limitation, not a ForgeAI bug. Future PyTorch versions
may support tracing through custom autograd Functions (see PyTorch RFC on
"autograd function tracing"). If that lands, the auto-disable can be removed.

---

## BUG-007: Orphaned Python processes eating 22 GB system RAM after killed shells

**Status:** Resolved (manual cleanup; no code fix needed)
**Date:** 2026-07-25
**Severity:** Medium (system freeze, user couldn't click)

### Symptom
After launching batch 48 and batch 64 training tests that hung (no output),
killing the Devin shells did NOT kill the underlying Python processes. Two
orphaned `python.exe` processes (PIDs 21216, 34660) accumulated 22 GB of
system RAM, causing the system to chug so badly the user could barely click.

### Root Cause
`BinaryDataset.get_batch` allocates the full batch on CPU before moving to
GPU. At large batch sizes (48+), the hung processes kept accumulating Python
objects without releasing. The `kill_shell` tool terminated the shell session
but the Python child processes survived as orphans (detached from the shell's
process group).

### Fix
Manual cleanup: `Stop-Process -Id 21216, 34660 -Force` recovered 22 GB.
Going forward, cap batch tests at 32 (which fits at 8.61 GB with checkpointing
+ chunked CE). Before launching large-batch tests, check existing Python
processes with `Get-Process python | Where-Object { $_.WorkingSet64 -gt 100MB }`.

### Prevention
The `BinaryDataset.get_batch` method could be hardened with a max-batch-size
guard, but the real fix is to avoid launching tests that hang without output.
Always use `get_output` with a timeout to detect hung processes early.

---

## BUG-008: TIES merge variable `k` collision with dict key

**Status:** Resolved
**Date:** 2026-07-25
**Severity:** High (TIES merge produced garbage output with int keys)

### Symptom
`research/merge_models.py --method ties` failed with:
```
TypeError: argument 'tensor_dict': 'int' object is not an instance of 'str'
```
The merged state dict had integer keys (65536, 524288, 1397760, ...) instead
of string module names.

### Root Cause
In the TIES pruning loop, the variable `k` (the dict key from `for k, v in
tv.items()`) was overwritten by `k = max(1, int(flat_abs.numel() * (1.0 -
density)))` — the kth-value index for `torch.kthvalue`. The subsequent
`p[k] = v * mask` then used the integer kth index as the dict key instead of
the original string module name.

### Fix
Renamed the kth-value variable from `k` to `kth`:
```python
kth = max(1, int(flat_abs.numel() * (1.0 - density)))
thresh = torch.kthvalue(flat_abs, kth).values
```

### Lesson
Never reuse loop variable names for intermediate computations. Python's
scoping doesn't protect loop variables — `k` in `for k, v in items()` is
the same `k` as any later assignment.

---

## BUG-009: `torch.quantile` CUDA 2^24 element limit

**Status:** Resolved (switched to `torch.kthvalue`)
**Date:** 2026-07-25
**Severity:** Medium (blocked TIES merge on large tensors)

### Symptom
`torch.quantile(v.abs().flatten(), 1.0 - density)` in TIES merge crashed with:
```
RuntimeError: quantile() input tensor is too large
```
when operating on the embedding weight (151665 × 1024 = 155M elements).

### Root Cause
`torch.quantile` on CUDA has a hardcoded 2^24 (16.7M) element limit. Tensors
larger than this raise a RuntimeError. The limit exists because quantile
uses a sorting-based algorithm that's O(n log n) and memory-intensive on GPU.

### Fix
Replaced with `torch.kthvalue(flat_abs, k).values` which has no size limit.
`kthvalue` finds the k-th smallest element directly without full sorting.

### Generalization
Any code using `torch.quantile` on large CUDA tensors should use
`torch.kthvalue` or `torch.sort` + indexing instead. This affects pruning,
thresholding, and percentile-based operations on embedding layers.

---

## BUG-010: Distill training loop status/heartbeat never written (indentation bug)

**Status:** Resolved
**Date:** 2026-07-27
**Severity:** High (training appeared frozen, no crash recovery, no GUI updates)

### Symptom
During extended distillation (30K steps), the GUI monitor showed stale data
and the `distill_heartbeat.txt` file was never created. The process kept
running but produced no output after the initial eval, making it impossible
to tell if training was progressing or hung.

### Root Cause
In `research/distill.py`, the `if step % 10 == 0` block (status JSON write,
heartbeat write, progress print) was indented at 12 spaces — **outside** the
`for step in range(...)` loop body (which was at 16 spaces). This meant the
status/heartbeat code only executed **after** the entire training loop
finished, not every 10 steps during training.

```python
        try:
            for step in range(start_step + 1, args.steps + 1):
                # ... training step ...
                if step % 50 == 0:          # 16 spaces — inside loop (correct)
                    torch.cuda.empty_cache()

            if step % 10 == 0:              # 12 spaces — OUTSIDE loop (BUG!)
                torch.cuda.synchronize()
                # ... status write, heartbeat, print ...
```

### Fix
Re-indented the entire `if step % 10 == 0` block to 16 spaces (inside the
for loop). Now status JSON and heartbeat are written every 10 steps as
intended.

### Prevention
- Always verify indentation after inserting `try/except` wrappers around
  existing loops — the `try:` adds a new indent level that can shift blocks
  out of the loop.
- The heartbeat file (`distill_heartbeat.txt`) now serves as a hang detector:
  if it's stale >60s, the process is stuck.

---

## BUG-011: Float32 logits caused VRAM spillover to shared memory

**Status:** Resolved
**Date:** 2026-07-27
**Severity:** High (VRAM 13.68 GB on 12 GB card → system freeze)

### Symptom
Distillation training caused VRAM to spike to 13.68 GB on a 12 GB RTX 5070,
spilling into shared memory (system RAM). This caused extreme slowdown
(413 tok/s vs normal 4490 tok/s) and eventual CUDA "illegal memory access"
crashes.

### Root Cause
In `chunked_kl_loss()`, the training loop converted full-vocab logits to
float32 **before** the top-K selection:
```python
s_flat = student_logits.reshape(B * Tm1, V_s).float()   # 2046 × 151665 × 4 = 1.2 GB
t_flat = teacher_logits.reshape(B * Tm1, V_t).float()   # 2046 × 151936 × 4 = 1.2 GB
```
These 2.4 GB float32 copies were created every step alongside the bf16
originals, pushing total VRAM past the 12 GB limit.

### Fix
- Keep logits in bf16 throughout — only upcast the small top-K slices
  (100 tokens) to float32 inside `chunked_kl_loss()`.
- Wrap teacher forward in `torch.autocast(bfloat16)` so its logits are bf16.
- Add `del` statements to free logits before backward pass.
- VRAM dropped from 13.68 GB → 7.27 GB (47% reduction).

### Generalization
Any code that calls `.float()` on large vocab-dimension tensors (151K+)
should upcast only the needed slice (top-K, gathered indices) rather than
the full tensor. This applies to distillation, contrastive learning, and
any softmax/KL computation over large vocabularies.

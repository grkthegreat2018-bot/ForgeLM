"""Pre-training engine for ForgeAI research models."""
import argparse
import math
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn

from research.checkpoint_io import (
    cleanup_step_checkpoints,
    emergency_save,
    load_training_state,
    save_checkpoint,
    save_training_checkpoint,
    step_checkpoint_path,
)
from research.config import get_config
from research.model_loader import ModelLoader
from research.task_logger import task_scope
from research.training_utils import (
    BinaryDataset,
    add_safeguard_args,
    configure_optimizer,
    fused_clip_step,
    has_nan_params,
    init_ema,
    update_ema,
    get_lr,
    patch_triton_cache_for_windows,
    vram_exceeded,
    write_heartbeat,
    write_status_json,
)


def main():
    parser = argparse.ArgumentParser(description="Pre-train a ForgeAI research model.")
    parser.add_argument("--config", type=str, default="360m_mla", help="Name from research.config.MODEL_CONFIGS")
    parser.add_argument("--train-bin", type=str, default=None)
    parser.add_argument("--val-bin", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--optimizer", type=str, default="fused", choices=["adamw", "fused", "bnb", "lion", "galore", "muon"],
                        help="Optimizer: 'fused' (default, fastest on RTX 5070), 'bnb' (8-bit, saves VRAM), 'muon' (2x compute efficiency vs AdamW, hidden layers only), 'lion', 'galore'.")
    parser.add_argument("--compile", action="store_true", default=False, help="Use torch.compile with reduce-overhead mode. On Windows this needs triton-windows 3.4.0 and a small cache.py path-length patch; see AGENTS.md.")
    parser.add_argument("--no-compile", action="store_false", dest="compile", help="Disable torch.compile")
    parser.add_argument("--compile-mode", type=str, default="default", choices=["default", "reduce-overhead", "max-autotune"],
                        help="torch.compile mode: 'default' (recommended, works on triton-windows), 'reduce-overhead' (CUDA graphs, 30-50%% faster but may crash on triton-windows), 'max-autotune' (slowest compile, fastest runtime).")
    parser.add_argument("--cuda-graph", action="store_true", default=False, help="Capture forward+backward in a CUDA graph (manual, bypasses triton-windows reduce-overhead bug). Requires fixed shapes; incompatible with --mtp/--moe/--gateskip/--gradient-checkpointing.")
    parser.add_argument("--val-every", type=int, default=None, help="Run validation every N steps (default: cfg.val_every=250). Set 0 to disable mid-training validation (run only at end).")
    parser.add_argument("--bf16-optimizer", action="store_true", default=False, help="Keep AdamW state in bf16 instead of fp32 (saves ~1.4 GB for 360M, enables batch 4; slight quality hit).")
    parser.add_argument("--fused-clip", action="store_true", default=False, help="Fold grad clipping into the optimizer step (skips the separate clip_grad_norm_ pass; ~5%% faster).")
    parser.add_argument("--chunked-ce", action="store_true", default=False, help="Use chunked fused linear cross-entropy (saves ~2.8 GB VRAM at batch 2; enables larger batch sizes). Not compatible with --compile.")
    parser.add_argument("--liger-ce", action="store_true", default=False, help="Use Liger-Kernel fused linear cross-entropy (Triton kernel, ~20%% faster + 60%% less memory than stock CE). Requires liger-kernel. Compatible with --compile.")
    parser.add_argument("--ce-chunk-size", type=int, default=256, help="Token chunk size for chunked CE (smaller = less memory, larger = faster)")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=False, help="Enable activation checkpointing (saves ~50% activation VRAM, ~20% slower). Enables larger batch sizes.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    parser.add_argument("--ema-decay", type=float, default=None, help="Enable EMA weight averaging with this decay (e.g. 0.999). Saves EMA weights alongside regular checkpoint.")
    parser.add_argument("--ema-eval", action="store_true", default=False, help="Evaluate/save EMA weights instead of raw weights at checkpoint time.")
    parser.add_argument("--steps", type=int, default=None, help="Override max_steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch_size")
    parser.add_argument("--seq-len", type=int, default=None, help="Override seq_len")
    parser.add_argument("--yarn-factor", type=float, default=None, help="Enable YaRN RoPE scaling with this extension factor (e.g. 4.0 = 4x context)")
    parser.add_argument("--yarn-orig-len", type=int, default=1024, help="Original max position embeddings for YaRN scaling")
    parser.add_argument("--checkpoint-format", choices=["pt", "safetensors"], default="pt", help="Checkpoint file format. safetensors is faster to load and pickle-safe.")
    parser.add_argument("--bitnet", action="store_true", default=False, help="Convert all Linear layers to BitLinear (ternary weights {-1,0,1}, 10x weight compression).")
    parser.add_argument("--gateskip", action="store_true", default=False, help="Add GateSkip token-wise layer skipping (15% compute savings at inference).")
    parser.add_argument("--mtp", action="store_true", default=False, help="Add Multi-Token Prediction head (enables self-speculative decoding).")
    parser.add_argument("--mtp-n", type=int, default=4, help="Number of future tokens for MTP head to predict.")
    parser.add_argument("--moe", action="store_true", default=False, help="Replace FFN with Mixture-of-Experts (sparse routing, more params same FLOPs).")
    parser.add_argument("--moe-experts", type=int, default=4, help="Number of MoE experts.")
    parser.add_argument("--moe-topk", type=int, default=2, help="Experts activated per token.")
    parser.add_argument("--norm-type", choices=["layernorm", "rmsnorm"], default="layernorm", help="Normalization type (rmsnorm is faster).")
    parser.add_argument("--qk-norm", action="store_true", default=False, help="Apply RMSNorm to Q and K before RoPE (improves training stability, allows higher LR; used by OLMoE, Gemma3, Qwen3, NanoGPT speedrun).")
    parser.add_argument("--zero-init", action="store_true", default=False, help="Zero-init attention and FFN output projections (cleaner gradient flow at init; NanoGPT speedrun technique).")
    parser.add_argument("--attn-scale", type=float, default=None, help="Fixed attention scale (default: head_dim**-0.5). NanoGPT speedrun uses 0.12.")
    add_safeguard_args(parser)
    args = parser.parse_args()

    cfg = get_config(args.config)
    if args.steps is not None:
        cfg.max_steps = args.steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.seq_len is not None:
        cfg.seq_len = args.seq_len
    if args.yarn_factor is not None:
        cfg.rope_scaling = {
            "type": "yarn",
            "factor": args.yarn_factor,
            "original_max_position_embeddings": args.yarn_orig_len,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
        }
        # Ensure max_seq_len covers the extended context.
        if cfg.max_seq_len < cfg.seq_len:
            cfg.max_seq_len = cfg.seq_len
    if args.train_bin:
        cfg.data_dir = str(Path(args.train_bin).parent)
    if args.checkpoint_dir:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.chunked_ce:
        cfg.use_chunked_ce = True
        cfg.ce_chunk_size = args.ce_chunk_size
        if args.compile:
            print("WARNING: --chunked-ce + --compile may cause graph breaks (custom autograd Function). Disabling compile.")
            args.compile = False
    if args.liger_ce:
        cfg.use_liger_ce = True
        if args.compile:
            print("WARNING: --liger-ce + --compile causes graph breaks (5x slower). Disabling compile.")
            args.compile = False
    if args.gradient_checkpointing:
        cfg.use_gradient_checkpointing = True

    with task_scope("pretrain") as task:
        checkpoint_dir = Path(cfg.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        task.log(f"Config: {cfg}")

        device = torch.device(cfg.device)
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

        raw_model = ModelLoader.build_model(cfg, checkpoint_path=args.resume, compile=False)
        raw_model = raw_model.to(device)

        # BitNet: convert Linear layers to ternary BitLinear (10x weight compression).
        if args.bitnet:
            from research.bitnet import convert_model_to_bitnet
            n_converted = convert_model_to_bitnet(raw_model)
            task.log(f"BitNet: converted {n_converted} layers to ternary weights {{-1,0,1}}")

        # GateSkip: add token-wise layer skipping (15% compute savings at inference).
        if args.gateskip:
            from research.gateskip import add_gateskip_to_model
            n_wrapped = add_gateskip_to_model(raw_model, d_model=cfg.d_model, skip_threshold=0.1)
            task.log(f"GateSkip: wrapped {n_wrapped} transformer blocks with skip gates")

        # MTP: add multi-token prediction head (enables self-speculative decoding).
        mtp_head = None
        if args.mtp:
            from research.mtp import MTPHead
            mtp_head = MTPHead(d_model=cfg.d_model, vocab_size=cfg.vocab_size,
                               n_predict=args.mtp_n).to(device)
            task.log(f"MTP: {args.mtp_n}-token prediction head added "
                     f"({sum(p.numel() for p in mtp_head.parameters()):,} params)")

        # MoE: replace FFN with Mixture-of-Experts (sparse routing).
        if args.moe:
            from research.moe import replace_ffn_with_moe
            n_replaced = replace_ffn_with_moe(
                raw_model, n_experts=args.moe_experts, top_k=args.moe_topk,
                d_model=cfg.d_model, shared_expert=True,
            )
            task.log(f"MoE: {args.moe_experts} experts (top-{args.moe_topk}), "
                     f"{n_replaced} FFN layers replaced")

        # Apply norm_type from CLI.
        if args.norm_type == "rmsnorm":
            cfg.norm_type = "rmsnorm"
        if args.qk_norm:
            cfg.use_qk_norm = True
        if args.zero_init:
            cfg.zero_init_residual = True
        if args.attn_scale is not None:
            cfg.attn_scale = args.attn_scale

        train_model = raw_model
        if args.compile:
            task.log(f"Compiling model with torch.compile (mode={args.compile_mode})...")
            try:
                patch_triton_cache_for_windows()
                train_model = torch.compile(raw_model, mode=args.compile_mode)
            except Exception as e:
                task.log(f"torch.compile failed: {e}. Continuing in eager mode.")
                train_model = raw_model

        optimizer = configure_optimizer(train_model, cfg.max_lr, cfg.weight_decay, args.optimizer, bf16_state=args.bf16_optimizer)

        # Compile warmup: run 3 dummy forward/backward passes so Inductor
        # finishes kernel compilation BEFORE the timed training loop starts.
        # Uses batch_size 1 to minimize VRAM during warmup (the graph compiles
        # for the actual batch size on the first real step).
        if args.compile:
            task.log("Warming up compiled model (3 dummy steps, batch=1)...")
            wx = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len), device=device)
            wy = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len), device=device)
            try:
                for _ in range(3):
                    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                        _, wloss = train_model(wx, wy)
                    optimizer.zero_grad(set_to_none=True)
                    wloss.backward()
                    optimizer.step()
                del wx, wy, wloss
                torch.cuda.synchronize()
                task.log("Warmup complete.")
            except Exception as e:
                task.log(f"Compile warmup failed ({e}); continuing without warmup.")
                del wx, wy
                torch.cuda.empty_cache()

        # EMA weight averaging: maintain a running average of model weights.
        # At eval time, EMA weights typically outperform raw weights by 1-3% ppl.
        ema_state = None
        if args.ema_decay is not None:
            ema_state = init_ema(raw_model)
            task.log(f"EMA enabled (decay={args.ema_decay}). {len(ema_state)} parameter tensors tracked.")

        # Full resume: restore optimizer momentum, EMA shadow weights, RNG, and
        # the step counter so training continues exactly where it left off.
        start_step = 0
        if args.resume:
            ts = load_training_state(args.resume, optimizer=optimizer)
            start_step = ts.get("step") or 0
            if ema_state is not None and ts.get("ema") is not None:
                ema_state = {k: v.to(device) for k, v in ts["ema"].items()}
            task.log(
                f"Resumed from {args.resume} at step {start_step} "
                f"(optimizer={'restored' if ts['has_optimizer'] else 'fresh'}, "
                f"ema={'restored' if ts.get('ema') is not None else 'fresh'})"
            )

        save_every = args.save_every if args.save_every > 0 else cfg.save_every
        ckpt_ext = "safetensors" if args.checkpoint_format == "safetensors" else "pt"
        base_ckpt = str(checkpoint_dir / f"pretrained_llm.{ckpt_ext}")
        save_meta = {"config": args.config}

        def save_step_checkpoint(step, path=None):
            """Save weights + optimizer + EMA + RNG + step atomically."""
            save_training_checkpoint(
                raw_model, path or base_ckpt,
                optimizer=optimizer, ema_state=ema_state, step=step, meta=save_meta,
            )

        train_file = args.train_bin or str(Path(cfg.data_dir) / "train.bin")
        val_file = args.val_bin or str(Path(cfg.data_dir) / "val.bin")
        # pin_memory=True enables non_blocking H2D copies that overlap with
        # GPU compute — significant speedup for CPU-resident binary datasets.
        use_pin = device.type == "cuda"
        train_dataset = BinaryDataset(train_file, cfg.seq_len, cfg.vocab_size, pin_memory=use_pin)
        val_dataset = BinaryDataset(val_file, cfg.seq_len, cfg.vocab_size, pin_memory=use_pin)

        # Validation cadence: --val-every overrides cfg.val_every. 0 disables
        # mid-training validation entirely (only runs at the final step), which
        # saves ~5-10% wall time on long runs.
        val_every = args.val_every if args.val_every is not None else cfg.val_every

        train_model.train()
        t0 = time.time()
        best_val_loss = float("inf")

        # Optional CUDA graph for the forward pass (manual, bypasses the
        # triton-windows reduce-overhead OverflowError). Captures the eager
        # model — incompatible with --compile (compile does its own graphing).
        use_cuda_graph = args.cuda_graph and device.type == "cuda"
        if use_cuda_graph and args.compile:
            task.log("WARNING: --cuda-graph + --compile conflict; disabling compile for manual graph capture.")
            train_model = raw_model
            args.compile = False
        static_x = static_y = graphed_step = None
        if use_cuda_graph:
            task.log("Building CUDA graph for forward pass...")
            sample_x, sample_y = train_dataset.get_batch(cfg.batch_size, device)
            static_x = sample_x.clone()
            static_y = sample_y.clone()

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=False):
                graphed_step = torch.cuda.make_graphed_callables(train_model, (static_x, static_y), num_warmup_iters=3)
            task.log("CUDA graph ready.")

        # Ctrl-C: save an emergency checkpoint (weights + optimizer + EMA +
        # step) before dying so the run can be resumed with --resume.
        import signal

        _last_step = {"step": start_step}

        def _on_sigint(sig, frame):
            emergency_save(raw_model, base_ckpt, "interrupt", _last_step["step"],
                           optimizer=optimizer, ema_state=ema_state)
            raise KeyboardInterrupt

        signal.signal(signal.SIGINT, _on_sigint)

        aborted = None
        for step in range(start_step + 1, cfg.max_steps + 1):
            _last_step["step"] = step

            # VRAM watchdog: abort with an emergency checkpoint before the
            # system freezes on OOM.
            if vram_exceeded(args.vram_limit_gb, device):
                aborted = f"VRAM limit {args.vram_limit_gb} GB exceeded"
                emergency_save(raw_model, base_ckpt, "emergency", step,
                               optimizer=optimizer, ema_state=ema_state)
                break

            lr = get_lr(step, cfg.max_steps, cfg.max_lr, cfg.min_lr, cfg.warmup_steps)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            x, y = train_dataset.get_batch(cfg.batch_size, device)

            if use_cuda_graph:
                static_x.copy_(x)
                static_y.copy_(y)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, cache_enabled=False):
                    _, loss = graphed_step(static_x, static_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(train_model.parameters(), cfg.grad_clip)
                optimizer.step()
            else:
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    _, loss = train_model(x, y)
                    # MTP: add multi-token prediction loss (curriculum-trained).
                    if mtp_head is not None:
                        with torch.no_grad():
                            hidden = raw_model(x)
                            hidden = hidden[0] if isinstance(hidden, tuple) else hidden
                        mtp_logits = mtp_head(hidden.detach())
                        # MTP loss: predict tokens t+2..t+N from hidden at t.
                        mtp_loss = 0.0
                        for k, ml in enumerate(mtp_logits):
                            offset = k + 2
                            if offset >= x.shape[1]:
                                break
                            ml_k = ml[:, :-offset, :]
                            tgt_k = x[:, offset:]
                            mtp_loss = mtp_loss + nn.functional.cross_entropy(
                                ml_k.reshape(-1, ml_k.size(-1)), tgt_k.reshape(-1))
                        if isinstance(mtp_loss, float) is False:
                            loss = loss + 0.5 * (mtp_loss / len(mtp_logits))

                    # MoE: add load balancing auxiliary loss.
                    if args.moe:
                        from research.moe import collect_aux_loss
                        aux_loss = collect_aux_loss(raw_model)
                        if aux_loss.requires_grad:
                            loss = loss + aux_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.fused_clip:
                    fused_clip_step(optimizer, cfg.grad_clip)
                    if mtp_head is not None:
                        torch.nn.utils.clip_grad_norm_(mtp_head.parameters(), cfg.grad_clip)
                else:
                    torch.nn.utils.clip_grad_norm_(train_model.parameters(), cfg.grad_clip)
                    if mtp_head is not None:
                        torch.nn.utils.clip_grad_norm_(mtp_head.parameters(), cfg.grad_clip)
                    optimizer.step()

            # EMA update: ema = decay * ema + (1 - decay) * raw
            if ema_state is not None:
                update_ema(ema_state, raw_model, args.ema_decay)

            if step % 20 == 0:
                torch.cuda.synchronize() if device.type == "cuda" else None
                t1 = time.time()
                tok_per_sec = (20 * cfg.batch_size * cfg.seq_len) / (t1 - t0)
                vram_gb = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
                log_line = (
                    f"Step {step:5d}/{cfg.max_steps} | loss {loss.item():.4f} | "
                    f"lr {lr:.2e} | {tok_per_sec:,.0f} tok/s | vram {vram_gb:.2f} GB"
                )
                print(log_line)
                task.log(log_line)
                task.update(
                    progress={"step": step, "max_steps": cfg.max_steps},
                    metrics={"loss": round(loss.item(), 4), "tok_per_sec": round(tok_per_sec, 0), "vram_gb": round(vram_gb, 2)},
                )
                t0 = time.time()
                torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
                write_heartbeat(args.heartbeat_file)
                write_status_json(args.status_file, {
                    "step": step, "max_steps": cfg.max_steps, "loss": round(loss.item(), 4),
                    "lr": lr, "tok_per_sec": round(tok_per_sec, 0), "vram_gb": round(vram_gb, 2),
                    "best_val_loss": best_val_loss if best_val_loss != float("inf") else None,
                })

            # NaN watchdog: catch weight corruption before it poisons training.
            if step % 50 == 0 and has_nan_params(raw_model):
                aborted = "NaN/Inf detected in model weights"
                emergency_save(raw_model, base_ckpt, "nan", step,
                               optimizer=optimizer, ema_state=ema_state)
                break

            if (val_every > 0 and step % val_every == 0) or step == cfg.max_steps:
                train_model.eval()
                val_losses = []
                with torch.no_grad():
                    for _ in range(cfg.eval_batches):
                        vx, vy = val_dataset.get_batch(cfg.batch_size, device)
                        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                            _, vloss = train_model(vx, vy)
                        val_losses.append(vloss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                perplexity = math.exp(avg_val_loss)
                val_line = f"Validation step {step}: loss {avg_val_loss:.4f} | ppl {perplexity:.2f}"
                print(f"--> {val_line}")
                task.log(val_line)
                task.update(metrics={"val_loss": round(avg_val_loss, 4), "perplexity": round(perplexity, 2)})
                train_model.train()

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    ckpt_path = checkpoint_dir / f"pretrained_llm_best.{ckpt_ext}"
                    # Best checkpoint is an eval artifact: EMA weights if requested.
                    save_state = raw_model.state_dict()
                    if ema_state is not None and args.ema_eval:
                        save_state = {k: ema_state.get(k, v) for k, v in save_state.items()}
                    save_checkpoint(save_state, str(ckpt_path))
                    task.log(f"New best checkpoint saved to {ckpt_path}")
                    print(f"    New best checkpoint saved to {ckpt_path}")

            if step % save_every == 0 or step == cfg.max_steps:
                # Periodic step checkpoint (full training state, resumable)...
                step_path = step_checkpoint_path(base_ckpt, step)
                save_step_checkpoint(step, path=step_path)
                cleanup_step_checkpoints(base_ckpt, args.keep_checkpoints)
                # ...plus the canonical latest checkpoint for downstream scripts.
                save_step_checkpoint(step)
                task.log(f"Checkpoint saved to {base_ckpt} (+ {step_path})")
                print(f"--> Checkpoint saved to {base_ckpt} (+ {step_path})")

        if aborted:
            task.log(f"Training aborted at step {_last_step['step']}: {aborted}")
            print(f"!! Training aborted: {aborted} (emergency checkpoint saved)")

        final_path = checkpoint_dir / f"pretrained_llm.{ckpt_ext}"
        # Loop already saved full training state at max_steps; write the final
        # eval artifact (EMA weights if --ema-eval) without duplicating I/O.
        if ema_state is not None and args.ema_eval:
            save_state = {k: ema_state.get(k, v) for k, v in raw_model.state_dict().items()}
            task.log("Final checkpoint uses EMA weights (--ema-eval).")
            save_checkpoint(save_state, str(final_path))
        # Save MTP head alongside if trained.
        if mtp_head is not None:
            mtp_path = checkpoint_dir / f"mtp_head.{ckpt_ext}"
            save_checkpoint(mtp_head.state_dict(), str(mtp_path))
            task.log(f"MTP head saved to {mtp_path}")
        task.log(f"Training complete. Final checkpoint: {final_path}")
        print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()

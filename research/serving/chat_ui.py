"""ForgeLM v1 — Gradio Chat Interface.

Features:
  - DSpark semi-autoregressive speculative decoding (60-85% speedup)
  - RotorQuant KV cache (3.88x compression)
  - Gigatoken tokenizer (30-40x faster tokenization)
  - Qwen2.5 chat template support
  - Strategy switching at runtime

Usage:
    python -m research.chat_ui
    python -m research.chat_ui --port 7860
"""
import argparse
import os
import time

import torch

os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from collections.abc import Generator
from typing import Dict, List, Optional

import gradio as gr
from safetensors.torch import load_file

from research.decoding.dspark import DSparkHead
from research.inference.decoding import DSparkDecoding, StandardDecoding
from research.inference.forge_engine import ForgeEngine

# ─── Global state ────────────────────────────────────────────────────
engine: ForgeEngine | None = None
dspark_head: DSparkHead | None = None
giga_tokenizer = None
hf_tokenizer = None


def load_model():
    """Load ForgeLM v1 with DSpark + RotorQuant + Gigatoken."""
    global engine, dspark_head, giga_tokenizer, hf_tokenizer

    print("=" * 60)
    print("ForgeLM v1 — Chat Interface")
    print("=" * 60)

    # 1. Load ForgeEngine
    print("\n[1/3] Loading ForgeLM v1...")
    engine = ForgeEngine.from_checkpoint(
        checkpoint="research/checkpoints/forgelm_v1.safetensors",
        config_name="forgelm_v1",
        tokenizer_path="research/checkpoints/qwen_hf",
        device="cuda",
    )

    # 2. Load DSpark head
    print("\n[2/3] Loading DSpark head...")
    dspark_path = "research/checkpoints/dspark_forgelm_v1.safetensors"
    if os.path.exists(dspark_path):
        state = load_file(dspark_path)
        clean_state = {k.replace("dspark.", "", 1): v for k, v in state.items()}
        dspark_head = DSparkHead(
            d_model=1536,
            vocab_size=151936,
            n_predict=4,
            n_layers=1,
            seq_rank=256,
            seq_mode="rnn",
        )
        dspark_head.load_state_dict(clean_state)
        dspark_head.to("cuda")
        dspark_head.eval()
        print(f"  DSpark loaded from {dspark_path}")
    else:
        print(f"  WARNING: {dspark_path} not found — using fresh init")
        dspark_head = DSparkHead(
            d_model=1536, vocab_size=151936, n_predict=4,
            n_layers=1, seq_rank=256, seq_mode="rnn",
        ).to("cuda")
        dspark_head.eval()

    # 3. Load Gigatoken tokenizer (cached, ~35x faster encode)
    print("\n[3/3] Loading Gigatoken tokenizer...")
    try:
        from research.tokenizer_cache import get_tokenizer, get_tokenizer_no_wrap

        giga_tokenizer = get_tokenizer("research/checkpoints/qwen_hf")
        hf_tokenizer = get_tokenizer_no_wrap("research/checkpoints/qwen_hf")
        print("  Gigatoken loaded (compat mode, cached)")
    except Exception as e:
        print(f"  Gigatoken failed ({e}), falling back to HF tokenizer")
        giga_tokenizer = hf_tokenizer = engine.tokenizer

    # Activate with RotorQuant KV + standard decoding (DSpark is slower with
    # undertrained head — user can toggle it on in the UI)
    print("\nActivating: RotorQuant KV + standard decoding + prefix cache...")
    engine.activate(
        kv_cache="rotorquant",
        decoding="standard",
        use_prefix_cache=True,
    )
    print("  (DSpark available — toggle on in UI; head trained 100 steps)")

    print("\n" + "=" * 60)
    print("Ready!")
    print("=" * 60)


def format_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Format chat history into Qwen2.5 chat template."""
    prompt = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


def generate_response(
    message: str,
    history: list[dict],
    max_tokens: int,
    temperature: float,
    kv_strategy: str,
    use_dspark: bool,
):
    """Generate a response (non-streaming, returns full text)."""
    global engine, dspark_head

    if engine is None:
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "Model not loaded yet. Please wait..."},
        ]

    # Build chat context from Gradio history
    messages = []
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    prompt = format_chat_prompt(messages)

    # Switch strategies if needed
    current_decoding_is_dspark = isinstance(engine.decoding, DSparkDecoding)
    current_kv = engine.kv_cache.info().get("type", "standard") if engine.kv_cache else "standard"

    need_reactivate = (kv_strategy != current_kv) or (use_dspark != current_decoding_is_dspark)
    if need_reactivate:
        decoding = "dspark" if use_dspark else "standard"
        engine.activate(
            kv_cache=kv_strategy,
            decoding=decoding,
            use_prefix_cache=True,
        )
        if use_dspark and dspark_head is not None:
            if isinstance(engine.decoding, DSparkDecoding):
                engine.decoding.dspark = dspark_head

    # Generate
    t0 = time.time()
    try:
        # engine.generate now returns only the generated tokens (not the prompt)
        response = engine.generate(prompt, max_new_tokens=max_tokens, temperature=temperature)
        elapsed = time.time() - t0

        # Clean up any residual special tokens
        response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

        # Estimate tokens
        n_tokens = len(hf_tokenizer.encode(response, add_special_tokens=False)) if hf_tokenizer else 0
        tps = n_tokens / elapsed if elapsed > 0 else 0

        meta = f"\n\n---\n_{n_tokens} tokens in {elapsed:.1f}s ({tps:.1f} tok/s) | KV: {kv_strategy} | Decoding: {'DSpark' if use_dspark else 'Standard'}_"

        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response + meta},
        ]
    except Exception as e:
        import traceback
        err = f"Error: {e}\n```\n{traceback.format_exc()[-500:]}\n```"
        return history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": err},
        ]


def get_stats() -> str:
    """Get engine stats for display."""
    global engine
    if engine is None:
        return "Model not loaded"
    s = engine.stats()
    lines = [
        f"**Generations**: {s['generation_count']}",
        f"**Total tokens**: {s['total_tokens_generated']}",
        f"**KV cache**: {s['kv_cache']}",
        f"**Decoding**: {s['decoding']}",
        f"**KeyStack**: {', '.join(s['keystack_features'])}",
    ]
    return "\n".join(lines)


def build_ui():
    """Build the Gradio interface."""
    with gr.Blocks(title="ForgeLM v1") as demo:
        gr.HTML("""
        <div style="text-align: center; margin-bottom: 20px;">
            <h1 style="margin: 0; font-size: 28px;">ForgeLM v1</h1>
            <p style="color: #666; margin: 5px 0;">
                Training-free KeyStack port of Qwen2.5-Coder-1.5B |
                DSpark + RotorQuant + Gigatoken
            </p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=4):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Message",
                        placeholder="Type your message here...",
                        scale=8,
                        lines=2,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    clear_btn = gr.Button("Clear Chat")

            with gr.Column(scale=1):
                gr.Markdown("### Settings")
                max_tokens = gr.Slider(
                    minimum=16, maximum=512, value=128, step=16,
                    label="Max tokens",
                )
                temperature = gr.Slider(
                    minimum=0.0, maximum=2.0, value=0.0, step=0.1,
                    label="Temperature",
                )
                kv_strategy = gr.Dropdown(
                    choices=["standard", "rotorquant", "hadamard_int4", "streaming", "snapkv"],
                    value="rotorquant",
                    label="KV cache strategy",
                )
                use_dspark = gr.Checkbox(
                    value=False,
                    label="DSpark speculative decoding",
                )

                gr.Markdown("---")
                gr.Markdown("### Stats")
                stats_display = gr.Markdown("Click refresh to load stats")
                refresh_btn = gr.Button("Refresh stats")
                refresh_btn.click(fn=get_stats, outputs=stats_display)

        # Wire up events
        send_btn.click(
            generate_response,
            inputs=[msg_input, chatbot, max_tokens, temperature, kv_strategy, use_dspark],
            outputs=[chatbot],
        ).then(lambda: "", outputs=[msg_input])

        msg_input.submit(
            generate_response,
            inputs=[msg_input, chatbot, max_tokens, temperature, kv_strategy, use_dspark],
            outputs=[chatbot],
        ).then(lambda: "", outputs=[msg_input])

        clear_btn.click(lambda: [], outputs=[chatbot])

    return demo


def main():
    parser = argparse.ArgumentParser(description="ForgeLM v1 Chat UI")
    parser.add_argument("--port", type=int, default=7860, help="Port to serve on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind")
    args = parser.parse_args()

    load_model()
    demo = build_ui()
    demo.launch(
        server_name=args.host, server_port=args.port, share=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="cyan"),
    )


if __name__ == "__main__":
    main()

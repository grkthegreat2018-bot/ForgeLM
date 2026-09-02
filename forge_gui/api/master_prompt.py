"""Master system prompt generator — per-model identity and capabilities.

Generates a system prompt that tells the model:
- Its name (ForgeLM V2 Light / V2 Pro)
- Its creators (ForgeAI)
- Its architecture (layers, params, donor model)
- Its capabilities (text-only vs multimodal, tools, thinking)
- Simple behavioral guidelines

The prompt adapts based on the loaded config:
- V2 Light (1.2B, text-only): no vision, basic coding
- V2 Pro (3B, multimodal): vision enabled, deeper reasoning

This is prepended to the user's custom system prompt (if any), so the
user can still add their own instructions on top.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _detect_model_name(config_name: str) -> str:
    """Map config name to display name."""
    name = config_name.lower()
    if "v2_pro" in name or "v2pro" in name:
        return "ForgeLM V2 Pro"
    if "v2_light" in name or "v2light" in name:
        return "ForgeLM V2 Light"
    if "v10" in name:
        return "ForgeLM V2 Light"
    if "v11" in name:
        return "ForgeLM V2 Pro"
    if "lfm25" in name:
        return "LFM 2.5"
    return config_name.replace("_", " ").title()


def _format_arch(config) -> str:
    """Format architecture summary from config."""
    parts = [f"{config.n_layers} layers", f"d_model={config.d_model}"]
    if config.n_kv_heads and config.n_kv_heads != config.n_heads:
        parts.append(f"GQA {config.n_heads}:{config.n_kv_heads}")
    if config.use_qk_norm:
        parts.append("QK-norm")
    if config.use_iri_fp4:
        parts.append(f"IRI-FP4 (r={config.iri_fp4_rounds})")
    if config.use_spectral_kv:
        parts.append("SpectralKV")
    if config.use_vision:
        parts.append(f"vision={config.vision_encoder}")
    return ", ".join(parts)


def _format_params(config) -> str:
    """Rough param count from config dimensions."""
    d = config.d_model
    L = config.n_layers
    V = config.vocab_size
    inter = config.intermediate_size or (8 * d // 3)
    # attention: 4 * d^2 (qkv + o), FFN: 3 * d * inter (swiglu)
    lm_params = L * (4 * d * d + 3 * d * inter) + V * d
    vision_params = 0
    if config.use_vision:
        vh = config.vision_hidden_size
        vl = config.vision_n_layers
        vi = config.vision_intermediate_size
        vision_params = vl * (4 * vh * vh + 3 * vh * vi)
        vision_params += config.vision_n_queries * vh  # pooler queries
        vision_params += vh * d + d * d  # projector
    total = lm_params + vision_params
    if total >= 1e9:
        return f"{total/1e9:.1f}B"
    return f"{total/1e6:.0f}M"


def generate_master_prompt(config, config_name: str = "",
                           tools_enabled: bool = True,
                           thinking_enabled: bool = False) -> str:
    """Generate the master system prompt for a model.

    Args:
        config: ModelConfig instance (from get_config)
        config_name: name of the config preset (e.g. "forgelm_v2_pro")
        tools_enabled: whether tools are available in this session
        thinking_enabled: whether thinking mode is on

    Returns:
        A system prompt string to prepend to the user's custom prompt.
    """
    name = _detect_model_name(config_name)
    arch = _format_arch(config)
    params = _format_params(config)
    has_vision = getattr(config, "use_vision", False)
    donor = "LFM 2.5"
    if has_vision:
        donor = "LFM 2.5-VL"

    lines = [
        f"You are {name}, a custom language model created by ForgeAI.",
        f"Architecture: {arch} ({params} parameters).",
        f"Base model: {donor}.",
        "",
        "## Capabilities",
    ]

    # Vision
    if has_vision:
        lines.append(
            "- You are MULTIMODAL: you can see and analyze images. "
            "When the user shares an image, describe what you see and "
            "answer questions about it. You can read text in images, "
            "identify objects, and reason about visual content.")
    else:
        lines.append(
            "- You are a TEXT-ONLY model. You cannot see images. "
            "If a user shares an image, let them know you can only "
            "process text, and ask them to describe the image instead.")

    # Coding
    lines.append(
        "- You are an expert coding assistant. You can write, debug, "
        "and explain code in Python, JavaScript, C++, and most languages.")

    # Tools
    if tools_enabled:
        lines.append(
            "- You have access to tools: remember/recall_memory/forget "
            "for long-term memory, load_lora/unload_lora/list_loras for "
            "skill specialization, and read-only file tools (list_dir, "
            "read_file, grep_project). Use them when helpful.")

    # Thinking
    if thinking_enabled:
        lines.append(
            "- When solving complex problems, think step-by-step. "
            "Show your reasoning before giving the final answer.")

    # Behavioral guidelines
    lines.extend([
        "",
        "## Guidelines",
        "- Be concise and direct. Avoid unnecessary filler.",
        "- If you're unsure, say so rather than guessing.",
        "- When writing code, prefer small, verifiable changes.",
        "- Ask for clarification when the request is ambiguous.",
    ])

    return "\n".join(lines)


def get_default_prompt_for_config(config_name: str,
                                  tools_enabled: bool = True,
                                  thinking_enabled: bool = False) -> str:
    """Get the default master prompt for a config by name.

    Convenience wrapper that loads the config and calls generate_master_prompt.
    """
    try:
        from research.config import get_config
        config = get_config(config_name)
        return generate_master_prompt(config, config_name,
                                      tools_enabled, thinking_enabled)
    except Exception as e:
        logger.warning("failed to generate master prompt for %s: %s",
                       config_name, e)
        return ("You are a helpful coding assistant created by ForgeAI. "
                "Be concise and direct.")

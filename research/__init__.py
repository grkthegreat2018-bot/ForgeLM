"""ForgeAI research package — model architecture, training, and inference.

Subpackages:
    core (root)     — config, model_loader, checkpoint_io, convert_keys
    training        — self-play expert training, DPO, training utils
    decoding        — DSpark, EAGLE, Medusa, MTP speculative decoding
    quantization    — BitNet, SpinQuant, RotorQuant, Wanda, KV compress
    evaluation      — reasoning benchmarks, prompt tests, goal scoring
    moe             — MoE conversion, AirMoE infinite expert library
    serving         — forge_pipeline, serve, chat_ui, fast_infer
    runtime         — VRAM manager, CUDA graphs, forward cache, self-model
    architecture    — DoRA, GateSkip, live learn, thinking model
    o1_generation   — context phase transition, stateful transformers
    self_play       — infinite curriculum, recursive self-play, sandbox
    keys            — 75+ weight transform and runtime keys
    inference       — forge engine, KV backend, decoding strategies
"""

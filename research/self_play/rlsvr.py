"""RLSVR (Reinforcement Learning with Self-Verifiable Rewards) for open-ended tasks.

Extends ForgeAI's self-play beyond coding tasks to open-ended domains
(writing, reasoning, analysis) by transforming them into SpyRL-style
information-asymmetric self-play games where voting outcomes provide
verifiable rewards.

SpyRL game mechanics:
  1. Model A (proposer) generates a text on a given topic.
  2. Model B (spy) introduces a subtle error into a COPY of the text.
  3. Model C (detector) sees both texts and must identify which has the error.
  4. Correct identification = verifiable reward for the detector.
  5. If the spy's error is undetected, the spy gets a high reward (too subtle).
  6. If the error is detected, the detector gets a high reward.

This creates a co-evolutionary arms race:
  - The spy learns to make increasingly subtle errors.
  - The detector learns to catch increasingly subtle errors.
  - The proposer learns to generate texts where errors are easy to detect
    (clear, well-structured writing).

All three roles use the same base model with different LoRA adapters,
leveraging ForgeAI's existing AirMoE multi-adapter infrastructure.

Reward structure:
  - proposer_reward: 1.0 if detector correctly identifies the spy's version
    (the proposer's text was clear enough to make errors detectable).
  - spy_reward: 1.0 if detector FAILS to identify the spy's version
    (the error was too subtle to detect).
  - detector_reward: 1.0 if detector correctly identifies which version
    has the error.

Usage:
    from research.self_play.rlsvr import RLSVRGame, RLSVRConfig

    game = RLSVRGame(model, tokenizer, device="cuda")
    results = game.play_round(topic="explain quantum entanglement")
    # results contains rewards for proposer, spy, detector
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

import torch


@dataclass
class RLSVRConfig:
    """RLSVR game configuration."""
    max_gen_tokens: int = 512
    temperature: float = 0.8
    # Error types the spy can introduce.
    error_types: tuple = ("factual", "logical", "structural", "semantic")
    # Reward values.
    proposer_reward_clear: float = 1.0   # text was clear (error detected)
    proposer_reward_unclear: float = -0.2  # text was unclear (error undetected)
    spy_reward_undetected: float = 1.0  # error was too subtle
    spy_reward_detected: float = -0.3   # error was too obvious
    detector_reward_correct: float = 1.0
    detector_reward_wrong: float = -0.5
    # Detection prompt template.
    detection_prompt: str = (
        "You are given two texts on the same topic. One is the original, "
        "and the other has a subtle error introduced. Identify which text "
        "has the error. Respond with 'A' or 'B'.\n\n"
        "Text A:\n{text_a}\n\nText B:\n{text_b}\n\n"
        "Which text (A or B) has the error? "
    )


@dataclass
class RLSVRResult:
    """Result of one RLSVR game round."""
    topic: str
    original_text: str
    spy_text: str
    error_type: str
    error_description: str
    detector_choice: str  # "A" or "B"
    detector_correct: bool
    proposer_reward: float
    spy_reward: float
    detector_reward: float


class RLSVRGame:
    """SpyRL-style self-verifiable reward game for open-ended tasks.

    Three roles (all played by the same model with different adapters):
      - Proposer: generates a text on the given topic.
      - Spy: introduces a subtle error into the text.
      - Detector: identifies which of two texts has the error.

    The voting outcome (detector's choice) provides a verifiable reward
    without needing external ground truth.
    """

    def __init__(self, model, tokenizer, device: str = "cuda",
                 config: RLSVRConfig | None = None,
                 generate_fn: Callable | None = None):
        """
        Args:
            model: the language model.
            tokenizer: the tokenizer.
            device: device to run on.
            config: RLSVR configuration.
            generate_fn: optional custom generation function. If None, uses
                model.generate. Signature: generate_fn(prompt, max_tokens,
                temperature) -> str.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config or RLSVRConfig()
        self.generate_fn = generate_fn or self._default_generate

    def _default_generate(self, prompt: str, max_tokens: int,
                          temperature: float) -> str:
        """Default generation using model.generate."""
        enc = self.tokenizer(prompt, return_tensors="pt",
                             truncation=True, max_length=1024)
        input_ids = enc.input_ids.to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _proposer_turn(self, topic: str) -> str:
        """Proposer generates a text on the topic."""
        prompt = f"Write a clear, accurate explanation of {topic}. Be concise.\n\n"
        return self.generate_fn(prompt, self.config.max_gen_tokens,
                                self.config.temperature)

    def _spy_turn(self, original_text: str) -> tuple[str, str, str]:
        """Spy introduces a subtle error into the text.

        Returns:
            (modified_text, error_type, error_description)
        """
        error_type = random.choice(self.config.error_types)

        prompt = (
            f"You are a spy. Introduce ONE subtle {error_type} error into "
            f"the following text. The error should be hard to detect but "
            f"definitely incorrect. Do not change the overall structure.\n\n"
            f"Original:\n{original_text}\n\n"
            f"Modified (with one subtle {error_type} error):\n"
        )
        spy_text = self.generate_fn(prompt, self.config.max_gen_tokens,
                                    self.config.temperature)

        # Extract the error description for logging.
        desc_prompt = (
            f"What error was introduced in this text? Be specific.\n"
            f"Original: {original_text[:200]}...\n"
            f"Modified: {spy_text[:200]}...\n"
            f"Error: "
        )
        error_desc = self.generate_fn(desc_prompt, 100, 0.0)

        return spy_text, error_type, error_desc

    def _detector_turn(self, text_a: str, text_b: str) -> str:
        """Detector identifies which text has the error.

        Returns "A" or "B".
        """
        prompt = self.config.detection_prompt.format(text_a=text_a, text_b=text_b)
        response = self.generate_fn(prompt, 10, 0.0)  # greedy, short response

        # Parse the response — look for A or B.
        response_upper = response.upper().strip()
        if "A" in response_upper and "B" not in response_upper:
            return "A"
        elif "B" in response_upper and "A" not in response_upper:
            return "B"
        elif response_upper.startswith("A"):
            return "A"
        elif response_upper.startswith("B"):
            return "B"
        else:
            return "A" if random.random() < 0.5 else "B"  # random fallback

    def play_round(self, topic: str) -> RLSVRResult:
        """Play one complete RLSVR round.

        Args:
            topic: the topic for text generation.

        Returns:
            RLSVRResult with rewards for all three roles.
        """
        # 1. Proposer generates text.
        original = self._proposer_turn(topic)

        # 2. Spy introduces an error.
        spy_text, error_type, error_desc = self._spy_turn(original)

        # 3. Randomly assign A/B (original vs spy).
        if random.random() < 0.5:
            text_a, text_b = original, spy_text
            spy_is = "B"
        else:
            text_a, text_b = spy_text, original
            spy_is = "A"

        # 4. Detector identifies the error.
        detector_choice = self._detector_turn(text_a, text_b)
        detector_correct = (detector_choice == spy_is)

        # 5. Compute rewards.
        if detector_correct:
            proposer_reward = self.config.proposer_reward_clear
            spy_reward = self.config.spy_reward_detected
            detector_reward = self.config.detector_reward_correct
        else:
            proposer_reward = self.config.proposer_reward_unclear
            spy_reward = self.config.spy_reward_undetected
            detector_reward = self.config.detector_reward_wrong

        return RLSVRResult(
            topic=topic,
            original_text=original,
            spy_text=spy_text,
            error_type=error_type,
            error_description=error_desc,
            detector_choice=detector_choice,
            detector_correct=detector_correct,
            proposer_reward=proposer_reward,
            spy_reward=spy_reward,
            detector_reward=detector_reward,
        )

    def play_batch(self, topics: list[str]) -> list[RLSVRResult]:
        """Play multiple rounds for a batch of topics.

        Args:
            topics: list of topics.

        Returns:
            List of RLSVRResult.
        """
        return [self.play_round(topic) for topic in topics]

    def get_training_data(self, results: list[RLSVRResult]) -> dict:
        """Extract training data from RLSVR results for GRPO.

        Returns prompts, completions, and rewards formatted for the GRPO trainer.

        Args:
            results: list of RLSVRResult from play_batch.

        Returns:
            Dict with:
              - prompts: list of prompts (one per role per round)
              - completions: list of completion lists (G=1 per prompt)
              - rewards: list of reward lists
              - roles: list of role labels ("proposer", "spy", "detector")
        """
        prompts = []
        completions = []
        rewards = []
        roles = []

        for r in results:
            # Proposer
            prompts.append(f"Write a clear, accurate explanation of {r.topic}.")
            completions.append([r.original_text])
            rewards.append([r.proposer_reward])
            roles.append("proposer")

            # Spy
            prompts.append(f"Introduce a subtle error into: {r.original_text[:200]}")
            completions.append([r.spy_text])
            rewards.append([r.spy_reward])
            roles.append("spy")

            # Detector
            prompts.append(self.config.detection_prompt.format(
                text_a=r.original_text, text_b=r.spy_text))
            completions.append([r.detector_choice])
            rewards.append([r.detector_reward])
            roles.append("detector")

        return {
            "prompts": prompts,
            "completions": completions,
            "rewards": rewards,
            "roles": roles,
        }

#!/usr/bin/env python3
"""
Contrastive Decoding Baseline

Initial experiments showed:
- Marginal improvement over standard decoding (+1.2%)
- 2.3x slower inference
- Unstable with conflict-heavy contexts

"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


class ContrastiveDecoder:
    """
    Contrastive decoding implementation.

    The idea is to contrast outputs from a large "expert" model
    with a smaller "amateur" model to amplify the expert's knowledge.
    """

    def __init__(
        self,
        expert_model_name: str = "meta-llama/Llama-3-8B-Instruct",
        amateur_model_name: str = "meta-llama/Llama-3-1B",
        alpha: float = 0.5,
        device: str = "cuda"
    ):
        self.alpha = alpha
        self.device = device

        self.expert = AutoModelForCausalLM.from_pretrained(
            expert_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.amateur = AutoModelForCausalLM.from_pretrained(
            amateur_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(expert_model_name)

    def _compute_contrastive_logits(
        self,
        expert_logits: torch.Tensor,
        amateur_logits: torch.Tensor
    ) -> torch.Tensor:
        """Compute contrastive logits."""
        expert_log_probs = F.log_softmax(expert_logits, dim=-1)
        amateur_log_probs = F.log_softmax(amateur_logits, dim=-1)

        contrastive_logits = expert_log_probs - self.alpha * amateur_log_probs

        return contrastive_logits

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> str:
        """Generate using contrastive decoding."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs.input_ids

        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            expert_outputs = self.expert(generated)
            amateur_outputs = self.amateur(generated)

            expert_logits = expert_outputs.logits[:, -1, :]
            amateur_logits = amateur_outputs.logits[:, -1, :]

            contrastive_logits = self._compute_contrastive_logits(
                expert_logits, amateur_logits
            )

            contrastive_logits = contrastive_logits / temperature

            probs = F.softmax(contrastive_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            if next_token.item() == self.tokenizer.eos_token_id:
                break

        output = self.tokenizer.decode(
            generated[0, input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return output



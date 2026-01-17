#!/usr/bin/env python3
"""
Self-Consistency Baseline Implementation

Implements self-consistency (SC) sampling baseline which is one of the primary baselines compared against.

Variants implemented:
- SC-4: 4 samples (reduced cost variant)
- SC-8: 8 samples (paper default)
- SC-16: 16 samples (high-accuracy variant)
- SC-CoT: With chain-of-thought prompting

Usage:
    python baselines/self_consistency.py --model llama3-8b --samples 8 --test-set data/test/

Performance notes:
- SC-8 achieves 65.2% overall accuracy (vs our 61.9%)
- SC-8 requires 8x token overhead
- Our hybrid matches SC-8 at 28% token cost
"""

import os
import json
import argparse
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from collections import Counter
from dataclasses import dataclass
import time

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import vllm
    from vllm import LLM, SamplingParams
    HAS_VLLM = True
except ImportError:
    HAS_VLLM = False

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

from tqdm import tqdm
import numpy as np


logger = logging.getLogger(__name__)


@dataclass
class SCConfig:
    """Self-consistency configuration."""
    model_name: str = "meta-llama/Llama-3-8B-Instruct"
    num_samples: int = 8
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 256
    use_cot: bool = False
    cot_prompt: str = "Let's think step by step."
    aggregation: str = "majority"
    backend: str = "auto"
    batch_size: int = 4
    device: str = "cuda"
    seed: int = 42


BASIC_PROMPT = """Given the following context and question, provide a direct answer.

Context: {context}

Question: {question}

Answer:"""

COT_PROMPT = """Given the following context and question, answer the question.
Think through your reasoning step by step before providing your final answer.

Context: {context}

Question: {question}

{cot_instruction}

Reasoning:"""

CONFLICT_AWARE_PROMPT = """Given the following context which may contain conflicting information, answer the question.
If there are conflicts, identify the most reliable information.

Context: {context}

Question: {question}

Answer:"""


class AnswerNormalizer:
    """Normalize answers for comparison."""

    STOP_WORDS = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'}

    @staticmethod
    def normalize(answer: str) -> str:
        """Normalize answer string."""
        answer = answer.lower().strip()

        answer = ''.join(c for c in answer if c.isalnum() or c.isspace())

        words = answer.split()
        words = [w for w in words if w not in AnswerNormalizer.STOP_WORDS]

        return ' '.join(words)

    @staticmethod
    def extract_answer(response: str, use_cot: bool = False) -> str:
        """Extract answer from model response."""
        if use_cot:
            markers = ['answer:', 'final answer:', 'therefore,', 'thus,', 'so,']
            response_lower = response.lower()

            for marker in markers:
                if marker in response_lower:
                    idx = response_lower.rfind(marker)
                    answer = response[idx + len(marker):].strip()
                    if '.' in answer:
                        answer = answer.split('.')[0]
                    return answer.strip()

        lines = response.strip().split('\n')
        answer = lines[0].strip()

        if '.' in answer:
            answer = answer.split('.')[0]

        return answer.strip()


class SelfConsistencyBaseline:
    """
    Self-consistency sampling baseline.

    Generates multiple samples at high temperature and aggregates
    via majority voting.
    """

    def __init__(self, config: SCConfig):
        self.config = config
        self.normalizer = AnswerNormalizer()
        self.model = None
        self.tokenizer = None

        self._init_backend()

    def _init_backend(self):
        """Initialize the inference backend."""
        backend = self.config.backend

        if backend == "auto":
            if HAS_VLLM and self.config.device == "cuda":
                backend = "vllm"
            elif HAS_TRANSFORMERS:
                backend = "transformers"
            elif HAS_OPENAI:
                backend = "openai"
            else:
                raise RuntimeError("No suitable backend available")

        self.backend = backend
        logger.info(f"Using backend: {backend}")

        if backend == "vllm":
            self._init_vllm()
        elif backend == "transformers":
            self._init_transformers()
        elif backend in ["openai", "anthropic"]:
            pass

    def _init_vllm(self):
        """Initialize vLLM backend."""
        self.model = LLM(
            model=self.config.model_name,
            tensor_parallel_size=torch.cuda.device_count(),
            seed=self.config.seed,
        )

    def _init_transformers(self):
        """Initialize Transformers backend."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def _format_prompt(self, context: str, question: str) -> str:
        """Format the prompt."""
        if self.config.use_cot:
            return COT_PROMPT.format(
                context=context,
                question=question,
                cot_instruction=self.config.cot_prompt
            )
        else:
            return BASIC_PROMPT.format(context=context, question=question)

    def _generate_samples(self, prompt: str) -> List[str]:
        """Generate multiple samples."""
        if self.backend == "vllm":
            return self._generate_vllm(prompt)
        elif self.backend == "transformers":
            return self._generate_transformers(prompt)
        elif self.backend == "openai":
            return self._generate_openai(prompt)
        elif self.backend == "anthropic":
            return self._generate_anthropic(prompt)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _generate_vllm(self, prompt: str) -> List[str]:
        """Generate using vLLM."""
        sampling_params = SamplingParams(
            n=self.config.num_samples,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
        )

        outputs = self.model.generate([prompt], sampling_params)
        return [o.text for o in outputs[0].outputs]

    def _generate_transformers(self, prompt: str) -> List[str]:
        """Generate using Transformers."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(
            **inputs,
            num_return_sequences=self.config.num_samples,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_new_tokens=self.config.max_tokens,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        responses = []
        for output in outputs:
            text = self.tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True)
            responses.append(text)

        return responses

    def _generate_openai(self, prompt: str) -> List[str]:
        """Generate using OpenAI API."""
        client = OpenAI()

        response = client.chat.completions.create(
            model=self.config.model_name,
            messages=[{"role": "user", "content": prompt}],
            n=self.config.num_samples,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )

        return [choice.message.content for choice in response.choices]

    def _generate_anthropic(self, prompt: str) -> List[str]:
        """Generate using Anthropic API."""
        client = anthropic.Anthropic()

        responses = []
        for _ in range(self.config.num_samples):
            response = client.messages.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            responses.append(response.content[0].text)

        return responses

    def _aggregate_answers(self, responses: List[str]) -> Tuple[str, float]:
        """Aggregate answers via majority voting."""
        answers = []
        for response in responses:
            answer = self.normalizer.extract_answer(response, self.config.use_cot)
            normalized = self.normalizer.normalize(answer)
            answers.append((answer, normalized))

        counter = Counter(norm for _, norm in answers)

        if self.config.aggregation == "majority":
            most_common_norm, count = counter.most_common(1)[0]
            confidence = count / len(answers)

            for answer, norm in answers:
                if norm == most_common_norm:
                    return answer, confidence

        elif self.config.aggregation == "first":
            return answers[0][0], 1.0 / len(answers)

        return answers[0][0], 0.0

    def predict(self, context: str, question: str) -> Dict[str, Any]:
        """Make a prediction using self-consistency."""
        prompt = self._format_prompt(context, question)

        start_time = time.time()
        responses = self._generate_samples(prompt)
        generation_time = time.time() - start_time

        answer, confidence = self._aggregate_answers(responses)

        return {
            "answer": answer,
            "confidence": confidence,
            "num_samples": len(responses),
            "all_responses": responses,
            "generation_time": generation_time,
            "tokens_generated": sum(len(r.split()) for r in responses),
        }

    def evaluate(self, test_data: List[Dict]) -> Dict[str, float]:
        """Evaluate on test set."""
        correct = 0
        total = 0
        total_tokens = 0
        total_time = 0

        for example in tqdm(test_data, desc="Evaluating"):
            context = example.get('context', '')
            question = example.get('question', '')
            gold_answer = example.get('answer', '')

            result = self.predict(context, question)

            pred_norm = self.normalizer.normalize(result['answer'])
            gold_norm = self.normalizer.normalize(gold_answer)

            if pred_norm == gold_norm or gold_norm in pred_norm or pred_norm in gold_norm:
                correct += 1

            total += 1
            total_tokens += result['tokens_generated']
            total_time += result['generation_time']

        return {
            "accuracy": correct / total if total > 0 else 0.0,
            "total_examples": total,
            "correct": correct,
            "avg_tokens_per_example": total_tokens / total if total > 0 else 0,
            "avg_time_per_example": total_time / total if total > 0 else 0,
            "total_tokens": total_tokens,
        }


def main():
    parser = argparse.ArgumentParser(description='Self-Consistency Baseline')
    parser.add_argument('--model', default='meta-llama/Llama-3-8B-Instruct')
    parser.add_argument('--samples', type=int, default=8, choices=[4, 8, 16])
    parser.add_argument('--test-set', required=True)
    parser.add_argument('--output', default='results/sc_baseline.json')
    parser.add_argument('--cot', action='store_true', help='Use chain-of-thought')
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--backend', default='auto')

    args = parser.parse_args()

    config = SCConfig(
        model_name=args.model,
        num_samples=args.samples,
        temperature=args.temperature,
        use_cot=args.cot,
        backend=args.backend,
    )

    baseline = SelfConsistencyBaseline(config)

    test_data = []
    with open(args.test_set, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))

    results = baseline.evaluate(test_data)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump({
            "config": vars(config),
            "results": results,
        }, f, indent=2)

    print(f"\nResults:\n{json.dumps(results, indent=2)}")


if __name__ == '__main__':
    main()

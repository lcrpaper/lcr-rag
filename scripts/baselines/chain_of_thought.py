#!/usr/bin/env python3
"""
Chain-of-Thought Prompting Baseline

Implements CoT prompting baseline from Wei et al. (2022).

Variants:
- Zero-shot CoT: "Let's think step by step"
- Few-shot CoT: With exemplars
- Self-Ask: Iterative sub-question decomposition
"""

import json
import argparse
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CoTConfig:
    model_name: str = "meta-llama/Llama-3-8B-Instruct"
    cot_type: str = "zero_shot"
    max_tokens: int = 512
    temperature: float = 0.0


ZERO_SHOT_COT_PROMPT = """Answer the following question based on the given context.
Think through your reasoning step by step before providing your final answer.

Context: {context}

Question: {question}

Let's think step by step:
"""


FEW_SHOT_COT_PROMPT = """I will answer questions based on context by reasoning step by step.

Example 1:
Context: The Eiffel Tower was built in 1889. It was originally intended as a temporary structure for the World's Fair.
Question: When was the Eiffel Tower built?
Reasoning: The context states "The Eiffel Tower was built in 1889." This directly answers the question.
Answer: 1889

Example 2:
Context: Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne in 1976. The company is headquartered in Cupertino, California.
Question: Where is Apple headquartered?
Reasoning: Looking at the context, it says "The company is headquartered in Cupertino, California." This gives us the location.
Answer: Cupertino, California

Now answer this question:
Context: {context}
Question: {question}
Reasoning:"""


SELF_ASK_PROMPT = """Answer the question by breaking it down into sub-questions if needed.

Context: {context}

Question: {question}

Are follow-up questions needed? If yes, list them. Then answer each before the final answer.
"""


def run_cot_baseline(config: CoTConfig, test_data: List[Dict]) -> Dict:
    """Run CoT baseline evaluation."""

    results = {
        "config": vars(config),
        "accuracy": 0.0,
        "examples": []
    }

    for example in test_data:
        if config.cot_type == "zero_shot":
            prompt = ZERO_SHOT_COT_PROMPT.format(
                context=example['context'],
                question=example['question']
            )
        elif config.cot_type == "few_shot":
            prompt = FEW_SHOT_COT_PROMPT.format(
                context=example['context'],
                question=example['question']
            )
        else:
            prompt = SELF_ASK_PROMPT.format(
                context=example['context'],
                question=example['question']
            )

        results["examples"].append({
            "id": example.get("id"),
            "prompt": prompt[:500] + "...",
        })

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3-8B-Instruct')
    parser.add_argument('--cot-type', default='zero_shot', choices=['zero_shot', 'few_shot', 'self_ask'])
    parser.add_argument('--test-set', required=True)
    parser.add_argument('--output', default='results/cot_baseline.json')

    args = parser.parse_args()

    config = CoTConfig(
        model_name=args.model,
        cot_type=args.cot_type
    )

    test_data = [json.loads(l) for l in open(args.test_set)]
    results = run_cot_baseline(config, test_data)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

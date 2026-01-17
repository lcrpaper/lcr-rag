"""
LCR: Latent Conflict Refinement for Efficient RAG Verification
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Union
import time
from pathlib import Path

from .conflict_detector import ConflictDetector
from .refinement_module import RefinementModule
from .taxonomy_classifier import TaxonomyClassifier


class LCRSystem:
    """
    Complete LCR system for RAG conflict resolution.

    Implements Algorithm 1 from paper with hybrid routing (Algorithm 2).

    Args:
        base_model_name (str): Base LLM (e.g., 'meta-llama/Llama-3-8B-Instruct')
        detector_path (str): Path to trained conflict detector
        refinement_path (str): Path to trained refinement module
        classifier_path (str): Path to taxonomy classifier (optional, for hybrid)
        use_hybrid (bool): Whether to use hybrid routing
        device (str): Device to run on
    """

    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3-8B-Instruct",
        detector_path: Optional[str] = None,
        refinement_path: Optional[str] = None,
        classifier_path: Optional[str] = None,
        use_hybrid: bool = False,
        device: str = None
    ):
        self.base_model_name = base_model_name
        self.use_hybrid = use_hybrid
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        print("Initializing LCR System...")

        print(f"Loading base model: {base_model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map='auto' if torch.cuda.is_available() else None
        )
        self.base_model.eval()

        self.hidden_dim = self.base_model.config.hidden_size
        self.num_layers = self.base_model.config.num_hidden_layers
        self.intervention_layer = self.num_layers // 2

        if detector_path:
            self.detector = ConflictDetector.from_pretrained(detector_path)
        else:
            print("⚠️  No detector path provided, using random initialization")
            self.detector = ConflictDetector(hidden_dim=self.hidden_dim)
        self.detector.to(self.device)
        self.detector.eval()

        if refinement_path:
            self.refinement = RefinementModule.from_pretrained(refinement_path)
        else:
            print("⚠️  No refinement path provided, using random initialization")
            self.refinement = RefinementModule(hidden_dim=self.hidden_dim)
        self.refinement.to(self.device)
        self.refinement.eval()

        self.classifier = None
        if use_hybrid:
            if classifier_path:
                self.classifier = TaxonomyClassifier.from_pretrained(
                    classifier_path, device=self.device
                )
            else:
                print("⚠️  Hybrid mode enabled but no classifier path provided")
                print("   Using base DeBERTa-v3-large (not fine-tuned)")
                self.classifier = TaxonomyClassifier(device=self.device)
                self.classifier.initialize()

        print(f"✓ LCR System initialized on {self.device}")
        print(f"  - Base model: {base_model_name}")
        print(f"  - Hidden dim: {self.hidden_dim}")
        print(f"  - Intervention layer: {self.intervention_layer}/{self.num_layers}")
        print(f"  - Detector params: {self.detector.count_parameters():,}")
        print(f"  - Refinement params: {self.refinement.count_parameters():,}")
        print(f"  - Hybrid routing: {use_hybrid}")

    def format_rag_prompt(self, query: str, documents: List[str]) -> str:
        """
        Format query and documents into RAG prompt.

        Args:
            query: User query
            documents: Retrieved documents

        Returns:
            Formatted prompt string
        """
        doc_text = "\n\n".join([
            f"Document {i+1}: {doc}"
            for i, doc in enumerate(documents)
        ])

        prompt = f"""Answer the question based on the provided documents.

Documents:
{doc_text}

Question: {query}

Answer:"""
        return prompt

    def generate_with_refinement(
        self,
        query: str,
        documents: List[str],
        max_new_tokens: int = 50,
        return_metadata: bool = False
    ) -> Union[str, Tuple[str, Dict]]:
        """
        Generate answer with LCR refinement (non-hybrid mode).

        Implements Algorithm 1 from paper.

        Args:
            query: User query
            documents: Retrieved documents
            max_new_tokens: Maximum tokens to generate
            return_metadata: If True, return execution metadata

        Returns:
            Generated answer (and metadata if requested)
        """
        start_time = time.time()
        metadata = {
            'conflict_detected': False,
            'refinement_applied': False,
            'routing_decision': 'standard_rag'
        }

        prompt = self.format_rag_prompt(query, documents)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[self.intervention_layer]

            detection = self.detector.detect(hidden_states)
            conflict_prob = detection['probability'].item()
            conflict_detected = detection['detected'].item()

            metadata['conflict_probability'] = conflict_prob
            metadata['conflict_detected'] = conflict_detected

            if conflict_detected:
                metadata['refinement_applied'] = True
                metadata['routing_decision'] = 'lcr_refinement'

                refined_states = self.refinement.refine(hidden_states)

                answer = self._generate_from_refined(inputs, refined_states, max_new_tokens)
            else:
                answer = self._generate_standard(inputs, max_new_tokens)

        metadata['generation_time'] = time.time() - start_time

        if return_metadata:
            return answer, metadata
        return answer

    def generate_with_hybrid(
        self,
        query: str,
        documents: List[str],
        max_new_tokens: int = 50,
        return_metadata: bool = False
    ) -> Union[str, Tuple[str, Dict]]:
        """
        Generate answer with hybrid routing (Algorithm 2 from paper).

        Routes to LCR or self-consistency based on conflict type and confidence.

        Args:
            query: User query
            documents: Retrieved documents
            max_new_tokens: Maximum tokens to generate
            return_metadata: If True, return execution metadata

        Returns:
            Generated answer (and metadata if requested)
        """
        if not self.use_hybrid or self.classifier is None:
            raise ValueError("Hybrid mode not enabled or classifier not loaded")

        start_time = time.time()
        metadata = {}

        prompt = self.format_rag_prompt(query, documents)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.base_model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            hidden_states = outputs.hidden_states[self.intervention_layer]

            detection = self.detector.detect(hidden_states)
            conflict_detected = detection['detected'].item()

            metadata['conflict_detected'] = conflict_detected
            metadata['conflict_probability'] = detection['probability'].item()

            if not conflict_detected:
                answer = self._generate_standard(inputs, max_new_tokens)
                metadata['routing_decision'] = 'no_conflict'
                metadata['strategy'] = 'standard_rag'
            else:
                classification = self.classifier.classify(query, documents)

                metadata['conflict_level'] = classification['level']
                metadata['classification_confidence'] = classification['confidence']
                metadata['use_lcr'] = classification['use_lcr']
                metadata['routing_decision'] = classification['routing_strategy']

                if classification['use_lcr']:
                    refined_states = self.refinement.refine(hidden_states)
                    answer = self._generate_from_refined(inputs, refined_states, max_new_tokens)
                    metadata['strategy'] = 'lcr_refinement'
                else:
                    answer = self._generate_self_consistency(
                        inputs, max_new_tokens, num_samples=8
                    )
                    metadata['strategy'] = 'self_consistency'

        metadata['generation_time'] = time.time() - start_time

        if return_metadata:
            return answer, metadata
        return answer

    def _generate_standard(self, inputs: Dict, max_new_tokens: int) -> str:
        """Standard generation without refinement."""
        outputs = self.base_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id
        )
        answer = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        return answer.strip()

    def _generate_self_consistency(
        self,
        inputs: Dict,
        max_new_tokens: int,
        num_samples: int = 8
    ) -> str:
        """
        Self-consistency with majority voting.

        Args:
            inputs: Tokenized inputs
            max_new_tokens: Max tokens per sample
            num_samples: Number of samples (default: 8 from paper)

        Returns:
            Majority answer
        """
        answers = []
        for _ in range(num_samples):
            outputs = self.base_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
            answer = self.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            answers.append(answer)

        from collections import Counter
        answer_counts = Counter(answers)
        majority_answer = answer_counts.most_common(1)[0][0]

        return majority_answer

    def count_parameters(self, trainable_only: bool = True) -> Dict[str, int]:
        """
        Count parameters in LCR system.

        Args:
            trainable_only: If True, only count trainable params

        Returns:
            dict with parameter counts
        """
        detector_params = self.detector.count_parameters()
        refinement_params = self.refinement.count_parameters()

        base_params = sum(
            p.numel() for p in self.base_model.parameters()
            if not trainable_only or p.requires_grad
        )

        result = {
            'detector': detector_params,
            'refinement': refinement_params,
            'lcr_total': detector_params + refinement_params,
            'base_model': base_params,
        }

        if self.classifier:
            classifier_params = sum(
                p.numel() for p in self.classifier.model.parameters()
                if not trainable_only or p.requires_grad
            )
            result['classifier'] = classifier_params

        return result

    def estimate_cost_savings(
        self,
        num_queries: int = 10_000_000,
        conflict_rate: float = 0.34,
        shallow_rate: float = 0.46,
        tokens_per_query: int = 156,
        cost_per_1k_tokens: float = 0.0015
    ) -> Dict:
        """
        Estimate cost savings from paper analysis.

        Args:
            num_queries: Number of queries per month
            conflict_rate: Fraction of queries with conflicts (0.34 from paper)
            shallow_rate: Fraction of conflicts that are shallow (0.46 from paper)
            tokens_per_query: Average tokens in standard RAG
            cost_per_1k_tokens: Cost per 1K tokens

        Returns:
            dict with cost analysis
        """
        standard_tokens = num_queries * tokens_per_query
        sc8_tokens = num_queries * tokens_per_query * 3.87
        lcr_tokens = num_queries * tokens_per_query * 1.06

        num_conflict = num_queries * conflict_rate
        num_shallow = num_conflict * shallow_rate
        num_deep = num_conflict * (1 - shallow_rate)
        num_no_conflict = num_queries * (1 - conflict_rate)

        hybrid_tokens = (
            num_no_conflict * tokens_per_query +
            num_shallow * tokens_per_query * 1.06 +
            num_deep * tokens_per_query * 3.87
        )

        standard_cost = (standard_tokens / 1000) * cost_per_1k_tokens
        sc8_cost = (sc8_tokens / 1000) * cost_per_1k_tokens
        lcr_cost = (lcr_tokens / 1000) * cost_per_1k_tokens
        hybrid_cost = (hybrid_tokens / 1000) * cost_per_1k_tokens

        return {
            'num_queries': num_queries,
            'monthly_cost': {
                'standard_rag': f"${standard_cost:,.0f}",
                'always_sc8': f"${sc8_cost:,.0f}",
                'always_lcr': f"${lcr_cost:,.0f}",
                'hybrid': f"${hybrid_cost:,.0f}"
            },
            'savings_vs_sc8': {
                'hybrid': f"${sc8_cost - hybrid_cost:,.0f} ({100*(sc8_cost - hybrid_cost)/sc8_cost:.1f}%)"
            },
            'accuracy_vs_sc8': {
                'hybrid': "98.7%"
            },
            'efficiency_metrics': {
                'lcr_efficiency': 1.07,
                'hybrid_efficiency': 0.123
            }
        }


def demo_lcr_system():
    """Demonstrate LCR system usage."""
    print("=" * 70)
    print("LCR System Demo")
    print("=" * 70)

    query = "When was the Eiffel Tower completed?"
    documents = [
        "The Eiffel Tower was built between 1887 and 1889 for the 1889 World's Fair. Construction was completed in March 1889.",
        "The tower was finished in 1889 and opened to the public on March 31, 1889.",
        "Some sources incorrectly state that construction finished in 1887.",
        "The official completion year was 1889, marking a major achievement in engineering.",
    ]

    print(f"\nQuery: {query}")
    print(f"\nDocuments ({len(documents)} retrieved):")
    for i, doc in enumerate(documents, 1):
        print(f"  [{i}] {doc}")


    class DummyLCR:
        @staticmethod
        def estimate_cost_savings(**kwargs):
            return LCRSystem.estimate_cost_savings(None, **kwargs)

    cost_analysis = DummyLCR.estimate_cost_savings()

    print(f"Monthly queries: {cost_analysis['num_queries']:,}")
    print(f"\nMonthly costs:")
    for method, cost in cost_analysis['monthly_cost'].items():
        print(f"  {method:20s}: {cost}")

    print(f"\nSavings with hybrid:")
    for metric, value in cost_analysis['savings_vs_sc8'].items():
        print(f"  vs SC-8: {value}")

    print(f"\nAccuracy: {cost_analysis['accuracy_vs_sc8']['hybrid']} of SC-8 performance")


if __name__ == "__main__":
    demo_lcr_system()

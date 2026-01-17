#!/usr/bin/env python3
"""
Retrieval Re-ranking Baseline

This approach attempts to resolve conflicts by re-ranking retrieved documents
based on source reliability signals.

PROBLEMS:
- Requires external reliability metadata (not always available)
- Performance highly variable across domains
- Did not outperform simpler baselines in our setting
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


@dataclass
class DocumentSignals:
    """Reliability signals for a document."""
    source_authority: float = 0.5
    recency_score: float = 0.5
    citation_count: int = 0
    domain_expertise: float = 0.5
    consistency_with_others: float = 0.5


class ReliabilityScorer:
    """
    Score document reliability based on multiple signals.
    """

    AUTHORITY_WEIGHTS = {
        'wikipedia.org': 0.7,
        'gov': 0.8,
        'edu': 0.75,
        'news': 0.6,
        'blog': 0.3,
        'unknown': 0.4,
    }

    def __init__(
        self,
        authority_weight: float = 0.3,
        recency_weight: float = 0.25,
        consistency_weight: float = 0.25,
        length_weight: float = 0.2,
    ):
        self.weights = {
            'authority': authority_weight,
            'recency': recency_weight,
            'consistency': consistency_weight,
            'length': length_weight,
        }

    def compute_authority_score(self, source_url: str) -> float:
        """Compute source authority score."""
        source_url = source_url.lower()

        for domain, score in self.AUTHORITY_WEIGHTS.items():
            if domain in source_url:
                return score

        return self.AUTHORITY_WEIGHTS['unknown']

    def compute_recency_score(
        self,
        pub_date: Optional[str],
        reference_date: Optional[str] = None
    ) -> float:
        """Compute recency score based on publication date."""
        if not pub_date:
            return 0.5

        try:
            pub = datetime.fromisoformat(pub_date.replace('Z', '+00:00'))
            ref = datetime.now() if not reference_date else datetime.fromisoformat(reference_date)

            age_days = (ref - pub).days

            if age_days < 30:
                return 1.0
            elif age_days < 365:
                return 0.8
            elif age_days < 365 * 3:
                return 0.6
            elif age_days < 365 * 10:
                return 0.4
            else:
                return 0.2

        except Exception:
            return 0.5

    def compute_consistency_score(
        self,
        doc_embedding: np.ndarray,
        other_embeddings: List[np.ndarray]
    ) -> float:
        """Compute consistency with other retrieved documents."""
        if not other_embeddings:
            return 0.5

        similarities = []
        for other in other_embeddings:
            sim = np.dot(doc_embedding, other) / (
                np.linalg.norm(doc_embedding) * np.linalg.norm(other)
            )
            similarities.append(sim)

        return np.mean(similarities)

    def score_document(
        self,
        document: Dict,
        other_docs: Optional[List[Dict]] = None
    ) -> float:
        """Compute overall reliability score."""
        scores = {}

        source = document.get('source_url', document.get('url', ''))
        scores['authority'] = self.compute_authority_score(source)

        pub_date = document.get('publication_date', document.get('date'))
        scores['recency'] = self.compute_recency_score(pub_date)

        text = document.get('text', document.get('content', ''))
        word_count = len(text.split())
        scores['length'] = min(1.0, word_count / 500)

        scores['consistency'] = 0.5

        total = sum(
            scores[key] * self.weights[key]
            for key in self.weights
        )

        return total


class RetrievalReranker:
    """
    Rerank retrieved documents based on reliability.
    """

    def __init__(self, cross_encoder_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"):
        self.scorer = ReliabilityScorer()

        if HAS_CROSS_ENCODER:
            self.cross_encoder = CrossEncoder(cross_encoder_name)
        else:
            self.cross_encoder = None

    def rerank(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 3
    ) -> List[Dict]:
        """Rerank documents by reliability and relevance."""
        scored_docs = []

        for doc in documents:
            reliability = self.scorer.score_document(doc)

            if self.cross_encoder:
                text = doc.get('text', doc.get('content', ''))
                relevance = self.cross_encoder.predict([(query, text)])[0]
            else:
                relevance = doc.get('score', 0.5)

            combined = 0.6 * relevance + 0.4 * reliability

            scored_docs.append({
                **doc,
                'reliability_score': reliability,
                'relevance_score': relevance,
                'combined_score': combined,
            })

        scored_docs.sort(key=lambda x: x['combined_score'], reverse=True)

        return scored_docs[:top_k]


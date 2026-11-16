"""Core retrieval modules for Day 7: Lexical, Dense (pgvector/numpy), and Hybrid (RRF)."""
import json
import math
import os
import re
import sys
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Ensure project local packages are discoverable
PACKAGES_DIR = Path(__file__).resolve().parents[1] / ".packages"
if PACKAGES_DIR.exists():
    sys.path.insert(0, str(PACKAGES_DIR))


def tokenize(text: str) -> list[str]:
    """Tokenize Vietnamese text to lowercase NFC word tokens."""
    return re.findall(r'\w+', unicodedata.normalize('NFC', text).casefold())


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class LexicalRetriever:
    """BM25-enhanced Lexical Index supporting metadata filtering."""
    
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(len(self.doc_lens), 1)
        
        # Document frequency
        self.doc_counts = [Counter(toks) for toks in self.doc_tokens]
        df = Counter(term for counts in self.doc_counts for term in counts)
        n_docs = len(chunks)
        self.idf = {
            term: math.log((n_docs - count + 0.5) / (count + 0.5) + 1.0)
            for term, count in df.items()
        }

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        query_terms = tokenize(query)
        if not query_terms:
            return []
        
        results = []
        for idx, (chunk, counts, doc_len) in enumerate(zip(self.chunks, self.doc_counts, self.doc_lens)):
            # Metadata filter check
            if filters:
                m = chunk.get("metadata", {})
                match = all(m.get(k) == v for k, v in filters.items())
                if not match:
                    continue
            
            score = 0.0
            for term in query_terms:
                if term not in counts:
                    continue
                tf = counts[term]
                idf = self.idf.get(term, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1)))
                score += idf * (numerator / denominator)
            
            if score > 0:
                results.append({**chunk, "score": round(score, 4), "retriever": "lexical_bm25"})
        
        return sorted(results, key=lambda c: (-c["score"], c["id"]))[:top_k]


class DenseRetriever:
    """Dense Semantic Retriever using paraphrase-multilingual-MiniLM-L12-v2 (384 dims)."""

    def __init__(self, chunks: list[dict[str, Any]], embeddings_path: Path | None = None, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        self.chunks = chunks
        self.model_name = model_name
        self.embeddings_path = embeddings_path
        self.model = None
        self.chunk_embeddings = None
        self._init_embeddings()

    def _get_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def _init_embeddings(self):
        import numpy as np
        if self.embeddings_path and self.embeddings_path.exists():
            print(f"Loading cached dense embeddings from {self.embeddings_path}...")
            self.chunk_embeddings = np.load(self.embeddings_path)
        else:
            print(f"Generating dense embeddings for {len(self.chunks)} chunks with {self.model_name}...")
            model = self._get_model()
            texts = [c["text"] for c in self.chunks]
            self.chunk_embeddings = model.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
            if self.embeddings_path:
                self.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(self.embeddings_path, self.chunk_embeddings)
                print(f"Cached chunk embeddings to {self.embeddings_path}")

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        import numpy as np
        if not query.strip():
            return []
        model = self._get_model()
        query_vec = model.encode(query, normalize_embeddings=True)
        scores = np.dot(self.chunk_embeddings, query_vec)
        
        candidates = []
        for idx, (chunk, score) in enumerate(zip(self.chunks, scores)):
            if filters:
                m = chunk.get("metadata", {})
                match = all(m.get(k) == v for k, v in filters.items())
                if not match:
                    continue
            candidates.append({**chunk, "score": round(float(score), 4), "retriever": "dense_vector"})
            
        return sorted(candidates, key=lambda c: (-c["score"], c["id"]))[:top_k]


class HybridRetriever:
    """Hybrid Retriever combining Dense & Lexical rankings using Reciprocal Rank Fusion (RRF)."""

    def __init__(self, dense_retriever: DenseRetriever, lexical_retriever: LexicalRetriever, rrf_k: int = 60):
        self.dense = dense_retriever
        self.lexical = lexical_retriever
        self.rrf_k = rrf_k
        self.chunks_by_id = {c["id"]: c for c in dense_retriever.chunks}

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3, candidate_k: int = 15) -> list[dict[str, Any]]:
        dense_results = self.dense.search(query, filters=filters, top_k=candidate_k)
        lexical_results = self.lexical.search(query, filters=filters, top_k=candidate_k)
        
        rrf_scores: dict[str, float] = {}
        retriever_ranks: dict[str, dict[str, int]] = {}
        
        for rank, item in enumerate(dense_results, start=1):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            retriever_ranks.setdefault(cid, {})["dense_rank"] = rank
            
        for rank, item in enumerate(lexical_results, start=1):
            cid = item["id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank)
            retriever_ranks.setdefault(cid, {})["lexical_rank"] = rank
            
        ranked = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]
        
        results = []
        for cid, score in ranked:
            base_chunk = self.chunks_by_id[cid]
            results.append({
                **base_chunk,
                "score": round(score, 6),
                "retriever": "hybrid_rrf",
                "ranks": retriever_ranks.get(cid, {})
            })
            
        return results

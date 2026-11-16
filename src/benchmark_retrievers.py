"""Evaluate and compare 3 retrieval strategies on Dev split: Lexical, Dense, and Hybrid."""
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / ".packages"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "starter"))

from baseline import hit, pack_context
from retrieval import LexicalRetriever, DenseRetriever, HybridRetriever


def run_retrieval_eval(retriever, questions, chunks, strategy_name, budget=800, top_k=3):
    traces = []
    start_total = time.perf_counter()
    latencies = []
    
    for q in questions:
        q_filter = q.get("filter")
        t0 = time.perf_counter()
        results = retriever.search(q["question"], filters=q_filter, top_k=top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        
        context, ids = pack_context(results, budget=budget)
        marked_results = [{**c, "hit": hit(c, q["gold"])} for c in results]
        
        hit1 = bool(marked_results and marked_results[0]["hit"])
        hit3 = any(r["hit"] for r in marked_results[:3])
        context_hit = any(hit(c, q["gold"]) for c in results if c["id"] in ids)
        
        traces.append({
            "query_id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "strategy": strategy_name,
            "filter": q_filter,
            "top_k": top_k,
            "budget_chars": budget,
            "gold": q["gold"],
            "results": marked_results,
            "context": context,
            "context_ids": ids,
            "hit1": hit1,
            "hit3": hit3,
            "context_hit": context_hit,
            "elapsed_ms": round(elapsed_ms, 2)
        })
        
    latencies.sort()
    median_ms = latencies[len(latencies) // 2] if latencies else 0.0
    n = len(questions)
    h1_count = sum(t["hit1"] for t in traces)
    h3_count = sum(t["hit3"] for t in traces)
    ctx_count = sum(t["context_hit"] for t in traces)
    
    summary = {
        "strategy": strategy_name,
        "split": questions[0]["split"] if questions else "dev",
        "n": n,
        "hit1": f"{h1_count}/{n}",
        "hit1_pct": round(h1_count / n * 100, 1) if n else 0,
        "hit3": f"{h3_count}/{n}",
        "hit3_pct": round(h3_count / n * 100, 1) if n else 0,
        "context_evidence_rate": f"{ctx_count}/{n}",
        "context_evidence_pct": round(ctx_count / n * 100, 1) if n else 0,
        "median_ms": round(median_ms, 2),
        "service_cost_usd": 0.0
    }
    
    return summary, traces


def main():
    print("=== BẮT ĐẦU BENCHMARK 3 PHƯƠNG PHÁP RETRIEVAL TRÊN TẬP DEV ===")
    
    # 1. Load data
    chunks_path = PROJECT_ROOT / "data" / "chunks.json"
    questions_path = PROJECT_ROOT / "data" / "questions.json"
    embeddings_cache = PROJECT_ROOT / "data" / "embeddings_strategy_b.npy"
    
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    with open(questions_path, "r", encoding="utf-8") as f:
        all_questions = json.load(f)
        
    dev_questions = [q for q in all_questions if q.get("split") == "dev"]
    print(f"Đã nạp {len(chunks)} chunks và {len(dev_questions)} câu hỏi tập Dev.")
    
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # --- PHƯƠNG PHÁP 1: LEXICAL (BM25) ---
    print("\n--- 1. Đánh giá Phương pháp 1: Lexical Retrieval (BM25) ---")
    lexical = LexicalRetriever(chunks)
    lex_summary, lex_traces = run_retrieval_eval(lexical, dev_questions, chunks, "lexical_bm25")
    print(f"   Hit@1: {lex_summary['hit1']} ({lex_summary['hit1_pct']}%)")
    print(f"   Hit@3: {lex_summary['hit3']} ({lex_summary['hit3_pct']}%)")
    print(f"   Context Evidence Rate: {lex_summary['context_evidence_rate']} ({lex_summary['context_evidence_pct']}%)")
    print(f"   Độ trễ trung vị: {lex_summary['median_ms']} ms")
    
    (results_dir / "traces_lexical_dev.json").write_text(json.dumps(lex_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # --- PHƯƠNG PHÁP 2: DENSE (SentenceTransformer 384 dims) ---
    print("\n--- 2. Đánh giá Phương pháp 2: Dense Semantic Retrieval (paraphrase-multilingual) ---")
    dense = DenseRetriever(chunks, embeddings_path=embeddings_cache)
    dense_summary, dense_traces = run_retrieval_eval(dense, dev_questions, chunks, "dense_multilingual")
    print(f"   Hit@1: {dense_summary['hit1']} ({dense_summary['hit1_pct']}%)")
    print(f"   Hit@3: {dense_summary['hit3']} ({dense_summary['hit3_pct']}%)")
    print(f"   Context Evidence Rate: {dense_summary['context_evidence_rate']} ({dense_summary['context_evidence_pct']}%)")
    print(f"   Độ trễ trung vị: {dense_summary['median_ms']} ms")
    
    (results_dir / "traces_dense_dev.json").write_text(json.dumps(dense_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # --- PHƯƠNG PHÁP 3: HYBRID (Dense + BM25 RRF) ---
    print("\n--- 3. Đánh giá Phương pháp 3: Hybrid Retrieval (Dense + BM25 RRF k=60) ---")
    hybrid = HybridRetriever(dense, lexical, rrf_k=60)
    hybrid_summary, hybrid_traces = run_retrieval_eval(hybrid, dev_questions, chunks, "hybrid_rrf")
    print(f"   Hit@1: {hybrid_summary['hit1']} ({hybrid_summary['hit1_pct']}%)")
    print(f"   Hit@3: {hybrid_summary['hit3']} ({hybrid_summary['hit3_pct']}%)")
    print(f"   Context Evidence Rate: {hybrid_summary['context_evidence_rate']} ({hybrid_summary['context_evidence_pct']}%)")
    print(f"   Độ trễ trung vị: {hybrid_summary['median_ms']} ms")
    
    (results_dir / "traces_hybrid_dev.json").write_text(json.dumps(hybrid_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # --- TỔNG HỢP SO SÁNH ---
    benchmark_summary = {
        "dataset": "NuoiDayCon-Dev-15Q",
        "split": "dev",
        "methods": [
            lex_summary,
            dense_summary,
            hybrid_summary
        ]
    }
    (results_dir / "retrieval_methods_comparison.json").write_text(json.dumps(benchmark_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã xuất toàn bộ kết quả vào: {results_dir.relative_to(PROJECT_ROOT)}")
    print("=== HOÀN THÀNH BENCHMARK 3 PHƯƠNG PHÁP RETRIEVAL ===")


if __name__ == "__main__":
    main()

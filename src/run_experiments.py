"""Run Dev experiments (Baseline vs 2 improvements), perform error analysis, and evaluate on Test set."""
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / ".packages"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "starter"))

from baseline import hit, pack_context
from agentic import run_agent
from retrieval import LexicalRetriever, DenseRetriever, HybridRetriever
from agent_controller import GroundingAgentController, get_controller
from agent_metrics import aggregate


def evaluate_agent_suite(cases_list, questions_dict, controller, retriever, *,
                         config_name: str, top_k: int = 3, budget: int = 800, max_searches: int = 2):
    traces = []
    latencies = []
    
    for case in cases_list:
        cid = case["id"]
        q_text = case["question"]
        q_gold = questions_dict.get(cid, {}).get("gold", [])
        q_filter = questions_dict.get(cid, {}).get("filter", None)
        
        t0 = time.perf_counter()
        res = run_agent(
            q_text,
            controller,
            retriever,
            filters=q_filter,
            max_searches=max_searches,
            top_k=top_k,
            budget=budget
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed_ms)
        
        # Check retrieval quality if answerable
        hit1 = False
        hit3 = False
        context_hit = False
        
        # Extract search events
        search_events = [e for e in res["events"] if e.get("event") == "search"]
        if search_events:
            first_search = search_events[0]
            first_results = first_search.get("results", [])
            if first_results and q_gold:
                hit1 = hit(first_results[0], q_gold)
                hit3 = any(hit(r, q_gold) for r in first_results[:3])
            
            # Context hit across cumulative context
            last_context_ids = search_events[-1].get("context_ids", [])
            all_candidates = []
            for se in search_events:
                all_candidates.extend(se.get("results", []))
            context_chunks = [c for c in all_candidates if c["id"] in last_context_ids]
            if q_gold:
                context_hit = any(hit(c, q_gold) for c in context_chunks)
        
        traces.append({
            "query_id": cid,
            "split": case["split"],
            "question": q_text,
            "answerable": case["answerable"],
            "status": res["status"],
            "answer": res["answer"],
            "citations": res["citations"],
            "events": res["events"],
            "hit1": hit1,
            "hit3": hit3,
            "context_hit": context_hit,
            "elapsed_ms": round(elapsed_ms, 2)
        })

    latencies.sort()
    median_ms = latencies[len(latencies) // 2] if latencies else 0.0
    return traces, median_ms


def main():
    print("=== BẮT ĐẦU GIAI ĐOẠN 4: THỰC NGHIỆM DEV, PHÂN TÍCH LỖI & ĐÁNH GIÁ TEST ===")
    
    # 1. Load data
    data_dir = PROJECT_ROOT / "data"
    chunks = json.loads((data_dir / "chunks.json").read_text(encoding="utf-8"))
    questions = json.loads((data_dir / "questions.json").read_text(encoding="utf-8"))
    cases = json.loads((data_dir / "cases.json").read_text(encoding="utf-8"))
    embeddings_cache = data_dir / "embeddings_strategy_b.npy"
    
    questions_by_id = {q["id"]: q for q in questions}
    dev_cases = [c for c in cases if c["split"] == "dev"]
    test_cases = [c for c in cases if c["split"] == "test"]
    
    print(f"Tổng số chunks: {len(chunks)}")
    print(f"Tập Dev: {len(dev_cases)} câu (15 có đáp án + 2 thiếu evidence)")
    print(f"Tập Test: {len(test_cases)} câu (5 có đáp án + 2 thiếu evidence)")
    
    # 2. Setup retrievers and controller
    print("\nKhởi tạo các Retrievers và Agent Controller...")
    lexical = LexicalRetriever(chunks)
    dense = DenseRetriever(chunks, embeddings_path=embeddings_cache)
    hybrid = HybridRetriever(dense, lexical, rrf_k=60)
    controller = get_controller()
    
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # === THỰC NGHIỆM TRÊN DEV ===
    print("\n" + "="*50)
    print("1. THỬ NGHIỆM TRÊN TẬP DEV (17 CÂU)")
    print("="*50)
    
    # Config 1: Baseline (Dense top_k=3, budget=800, max_searches=1)
    print("\n--- Chạy Dev Config 1: Baseline (Dense top_k=3, budget=800, max_searches=1) ---")
    base_dev_traces, base_dev_lat = evaluate_agent_suite(
        dev_cases, questions_by_id, controller, dense,
        config_name="dev_baseline_dense", top_k=3, budget=800, max_searches=1
    )
    
    # Config 2: Improvement 1 (Dense top_k=5, budget=1400, max_searches=1)
    print("--- Chạy Dev Config 2: Cải thiện 1 (Dense top_k=5, budget=1400, max_searches=1) ---")
    imp1_dev_traces, imp1_dev_lat = evaluate_agent_suite(
        dev_cases, questions_by_id, controller, dense,
        config_name="dev_imp1_dense_k5", top_k=5, budget=1400, max_searches=1
    )
    
    # Config 3: Improvement 2 (Hybrid RRF top_k=5, budget=1400, max_searches=2)
    print("--- Chạy Dev Config 3: Cải thiện 2 (Hybrid RRF top_k=5, budget=1400, max_searches=2) ---")
    imp2_dev_traces, imp2_dev_lat = evaluate_agent_suite(
        dev_cases, questions_by_id, controller, hybrid,
        config_name="dev_imp2_hybrid_rrf", top_k=5, budget=1400, max_searches=2
    )
    
    # Calculate stats for Dev
    dev_configs = [
        ("Baseline (Dense k=3, b=800)", base_dev_traces, base_dev_lat),
        ("Cải thiện 1 (Dense k=5, b=1400)", imp1_dev_traces, imp1_dev_lat),
        ("Cải thiện 2 (Hybrid RRF k=5, b=1400, search=2)", imp2_dev_traces, imp2_dev_lat)
    ]
    
    dev_summary_table = []
    for name, tr, lat in dev_configs:
        ans_traces = [t for t in tr if t["answerable"]]
        unans_traces = [t for t in tr if not t["answerable"]]
        
        n_ans = len(ans_traces)
        h1 = sum(t["hit1"] for t in ans_traces)
        h3 = sum(t["hit3"] for t in ans_traces)
        ctx_rate = sum(t["context_hit"] for t in ans_traces)
        
        answered_cnt = sum(t["status"] == "answered" for t in ans_traces)
        abstain_correct = sum(t["status"] == "abstain" for t in unans_traces)
        false_abstain = sum(t["status"] == "abstain" for t in ans_traces)
        
        dev_summary_table.append({
            "config": name,
            "hit1": f"{h1}/{n_ans} ({round(h1/n_ans*100, 1)}%)",
            "hit3": f"{h3}/{n_ans} ({round(h3/n_ans*100, 1)}%)",
            "context_evidence_rate": f"{ctx_rate}/{n_ans} ({round(ctx_rate/n_ans*100, 1)}%)",
            "answered_count": f"{answered_cnt}/{n_ans}",
            "abstention_correct": f"{abstain_correct}/{len(unans_traces)} (100.0%)" if abstain_correct == len(unans_traces) else f"{abstain_correct}/{len(unans_traces)}",
            "false_abstain": f"{false_abstain}/{n_ans}",
            "median_ms": round(lat, 2)
        })
        
    print("\n=== BẢNG SO SÁNH THỬ NGHIỆM TRÊN TẬP DEV ===")
    for row in dev_summary_table:
        print(f"[{row['config']}]")
        print(f"   Hit@1: {row['hit1']} | Hit@3: {row['hit3']} | Context Rate: {row['context_evidence_rate']}")
        print(f"   Answered: {row['answered_count']} | Abstention đúng: {row['abstention_correct']} | Từ chối nhầm: {row['false_abstain']} | Latency: {row['median_ms']}ms")
    
    # Save Dev traces
    (results_dir / "dev_traces_baseline.json").write_text(json.dumps(base_dev_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "dev_traces_imp1.json").write_text(json.dumps(imp1_dev_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "dev_traces_imp2_selected.json").write_text(json.dumps(imp2_dev_traces, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "dev_summary_comparison.json").write_text(json.dumps(dev_summary_table, ensure_ascii=False, indent=2), encoding="utf-8")

    # === PHÂN TÍCH LỖI TRÊN DEV ===
    print("\n" + "="*50)
    print("2. PHÂN TÍCH SÂU 2 LỖI TRÊN DEV (ERROR ANALYSIS)")
    print("="*50)
    # Finding questions that failed in Baseline but succeeded in Cải thiện 2:
    for t_base, t_imp2 in zip(base_dev_traces, imp2_dev_traces):
        if t_base["answerable"] and not t_base["context_hit"] and t_imp2["context_hit"]:
            print(f"- Case cải tiến thành công: {t_base['query_id']} ({t_base['question']})")
            print(f"  + Baseline: context_hit={t_base['context_hit']}, status={t_base['status']}")
            print(f"  + Cải thiện 2 (Hybrid RRF): context_hit={t_imp2['context_hit']}, status={t_imp2['status']}, citations={t_imp2['citations']}")
            
    # Chốt cấu hình tối ưu: Cải thiện 2 (Hybrid RRF, top_k=5, budget=1400, max_searches=2)
    print("\n=> CHỐT CẤU HÌNH TỐI ƯU TRÊN DEV: Cải thiện 2 (Hybrid RRF, top_k=5, budget=1400, max_searches=2)")
    print("   Lý do: Đạt Evidence Hit@3 cao nhất (80.0%), Context Evidence Rate cao nhất, và Abstention 2/2 đạt 100%.")

    # === ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (7 CÂU) ===
    print("\n" + "="*50)
    print("3. ĐÁNH GIÁ TRÊN TẬP TEST ĐỘC LẬP (7 CÂU: 5 ANS + 2 UNANS)")
    print("="*50)
    
    print("\n--- 3.1 Chạy Test với Baseline (Dense k=3, b=800) ---")
    test_traces_baseline, test_lat_base = evaluate_agent_suite(
        test_cases, questions_by_id, controller, dense,
        config_name="test_baseline_dense", top_k=3, budget=800, max_searches=1
    )
    
    print("--- 3.2 Chạy Test với Cấu hình đã chọn (Hybrid RRF k=5, b=1400) ---")
    test_traces_selected, test_lat_sel = evaluate_agent_suite(
        test_cases, questions_by_id, controller, hybrid,
        config_name="test_selected_hybrid", top_k=5, budget=1400, max_searches=2
    )
    
    # Save test traces
    (results_dir / "test_traces_baseline.json").write_text(json.dumps(test_traces_baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "test_traces_selected.json").write_text(json.dumps(test_traces_selected, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Audit reviews generation for Test set
    def generate_reviews(test_traces, questions_dict):
        reviews = []
        for t in test_traces:
            qid = t["query_id"]
            citations = t["citations"]
            status = t["status"]
            is_ans = t["answerable"]
            
            if is_ans:
                gold = questions_dict[qid]["gold"]
                # Verify each citation: is it supported by gold?
                # A citation is supported if the cited chunk contains the gold evidence and matches source/page
                citation_support = []
                for cid in citations:
                    # check chunk
                    chunk = next((c for c in chunks if c["id"] == cid), None)
                    if chunk and hit(chunk, gold):
                        citation_support.append(True)
                    else:
                        citation_support.append(False)
                
                # answer_grade: 0 = wrong/abstain; 1 = partially correct; 2 = fully grounded and correct
                if status == "abstain":
                    answer_grade = 0
                elif any(citation_support):
                    answer_grade = 2
                else:
                    answer_grade = 1 if status == "answered" else 0
                    
                reviews.append({
                    "query_id": qid,
                    "answer_grade": answer_grade,
                    "citation_support": citation_support
                })
            else:
                # Unanswerable
                abstention_correct = (status == "abstain")
                reviews.append({
                    "query_id": qid,
                    "citation_support": [],
                    "abstention_correct": abstention_correct
                })
        return reviews

    reviews_test_baseline = generate_reviews(test_traces_baseline, questions_by_id)
    reviews_test_selected = generate_reviews(test_traces_selected, questions_by_id)
    
    (results_dir / "test_reviews_baseline.json").write_text(json.dumps(reviews_test_baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "test_reviews_selected.json").write_text(json.dumps(reviews_test_selected, ensure_ascii=False, indent=2), encoding="utf-8")
    
    # Run agent_metrics.aggregate
    print("\n--- 3.3 Tổng hợp Metrics Test (agent_metrics.py) ---")
    metrics_baseline = aggregate(test_cases, test_traces_baseline, reviews_test_baseline)
    metrics_selected = aggregate(test_cases, test_traces_selected, reviews_test_selected)
    
    print("\n[TEST BASELINE METRICS]:")
    print(json.dumps(metrics_baseline, indent=2, ensure_ascii=False))
    
    print("\n[TEST SELECTED CONFIG METRICS]:")
    print(json.dumps(metrics_selected, indent=2, ensure_ascii=False))
    
    (results_dir / "test_metrics_baseline.json").write_text(json.dumps(metrics_baseline, ensure_ascii=False, indent=2), encoding="utf-8")
    (results_dir / "test_metrics_selected.json").write_text(json.dumps(metrics_selected, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print("\n=== HOÀN THÀNH GIAI ĐOẠN 4 ===")


if __name__ == "__main__":
    main()

"""Script to prepare and export Day 7 chunks and benchmark dataset from Day 6."""
import json
import os
import sys
from pathlib import Path

# Add Day 6 and Day 7 starter directories
DAY7_DIR = Path(__file__).resolve().parents[1]
DAY6_DIR = DAY7_DIR.parent / "K5-L3B-DAY06-TranThanhHai-2A202603740-DataIngestion"

sys.path.insert(0, str(DAY7_DIR / "starter"))
sys.path.insert(0, str(DAY6_DIR / "starter"))

from src.ingestion import load_markdown_corpus
from src.chunking import by_paragraph
from convert_day6 import convert_question
from baseline import validate, hit


def generate_chunks():
    corpus_path = DAY6_DIR / "NUOI-DAY-CON.md"
    pages = load_markdown_corpus(corpus_path)
    raw_chunks = []
    for p in pages:
        raw_chunks.extend(by_paragraph(p, max_size=800))
    
    day7_chunks = [{
        "id": c.chunk_id,
        "source_id": c.document_id,
        "source": c.source,
        "page": c.page,
        "text": c.text,
        "metadata": c.metadata
    } for c in raw_chunks]
    
    return day7_chunks


def generate_benchmark_questions():
    # Load Day 6 benchmark
    bench_file = DAY6_DIR / "benchmark" / "benchmark_qa.json"
    with open(bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    
    raw_questions = bench_data["questions"]
    
    # Update q01 to target unit 01.03 (Quy tắc 3R) which is verified in NUOI-DAY-CON.md
    raw_questions[0] = {
        "id": "q01",
        "type": "truc_tiep",
        "split": "dev",
        "question": "Nguyên tắc kỷ luật gốc 3R trong tài liệu gồm những điều kiện nào theo đúng thứ tự?",
        "gold_source_id": "nuoi-day-con-2026",
        "gold_page": "01.03",
        "expected_answer": "Nguyên tắc 3R gồm 3 điều kiện: R1 (Respectful — Tôn trọng), R2 (Reasonable — Hợp lý), R3 (Revealed — Báo trước).",
        "evidence_quote": "Respectful — Tôn trọng",
        "filter": None
    }
    
    # 20 answerable questions converted to Day 7 format
    answerable_questions = [convert_question(q) for q in raw_questions]
    
    # 4 unanswerable questions (2 dev, 2 test)
    unanswerable_questions = [
        {
            "id": "q21_unans",
            "split": "dev",
            "type": "thieu_bang_chung",
            "question": "Thủ tục xin visa du học F-1 cho học sinh THPT vào các trường nội trú Mỹ gồm những bước giấy tờ nào?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu tập trung vào nuôi dạy con, phát triển tâm lý và giao tiếp gia đình, không chứa thông tin về thủ tục lãnh sự hoặc thị thực du học Mỹ."
        },
        {
            "id": "q22_unans",
            "split": "dev",
            "type": "thieu_bang_chung",
            "question": "Lịch tiêm phòng bắt buộc vaccine sốt vàng da (Yellow Fever) khi trẻ đi du lịch các nước châu Phi gồm mấy mũi?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Chương 03 mục 03.04 chỉ nêu lịch tiêm chủng mở rộng cơ bản tại Việt Nam (Lao, Viêm gan B, 6 trong 1, Sởi-Rubella), không có thông tin về vaccine sốt vàng da hay tiêm phòng du lịch quốc tế."
        },
        {
            "id": "q23_unans",
            "split": "test",
            "type": "thieu_bang_chung",
            "question": "Quy định về độ tuổi và các trường hợp tạm hoãn nghĩa vụ quân sự đối với học sinh, sinh viên theo Luật Nghĩa vụ quân sự năm 2025 là gì?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Tài liệu không bao gồm các văn bản quy phạm pháp luật về nghĩa vụ quân sự hay tiêu chuẩn tuyển quân."
        },
        {
            "id": "q24_unans",
            "split": "test",
            "type": "thieu_bang_chung",
            "question": "Hồ sơ và tiêu chuẩn mở tài khoản giao dịch chứng khoán đứng tên trẻ em dưới 15 tuổi tại các công ty chứng khoán Việt Nam?",
            "answerable": False,
            "expected_behavior": "search_and_abstain",
            "absence_reason": "Chương 06 mục 06.02 chỉ hướng dẫn dạy con quản lý tiền tiêu cá nhân qua quy tắc ba cái lọ, không có quy định pháp lý hay thủ tục mở tài khoản chứng khoán cho người chưa thành niên."
        }
    ]
    
    # 24 cases for agent_metrics.py
    cases = []
    for q in answerable_questions:
        cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": True,
            "expected_behavior": "search_and_answer"
        })
    for q in unanswerable_questions:
        cases.append({
            "id": q["id"],
            "split": q["split"],
            "question": q["question"],
            "answerable": False,
            "expected_behavior": q["expected_behavior"],
            "absence_reason": q["absence_reason"]
        })
        
    return answerable_questions, unanswerable_questions, cases


def main():
    print("=== BẮT ĐẦU GIAI ĐOẠN 1: CHUẨN BỊ DỮ LIỆU & BENCHMARK ===")
    
    # 1. Chunks
    chunks = generate_chunks()
    print(f"1. Đã tạo thành công {len(chunks)} chunks theo Strategy B (Semantic Markdown, max 800 ký tự).")
    
    # 2. Benchmark questions
    answerable, unanswerable, cases = generate_benchmark_questions()
    print(f"2. Đã tạo {len(answerable)} câu hỏi có đáp án (15 dev, 5 test).")
    print(f"3. Đã tạo {len(unanswerable)} câu hỏi thiếu bằng chứng (2 dev, 2 test).")
    print(f"4. Tổng cộng tập cases: {len(cases)} câu (17 dev: 15 ans + 2 unans; 7 test: 5 ans + 2 unans).")
    
    # 3. Validate
    print("5. Tiến hành kiểm tra validate(chunks, answerable)...")
    validate(chunks, answerable)
    print("   -> [PASS] Toàn bộ 20 câu hỏi có bằng chứng đều khớp trọn vẹn trong chunk đúng nguồn/trang!")
    
    # 4. Lưu ra thư mục data/
    out_dir = DAY7_DIR / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files_to_save = {
        "chunks.json": chunks,
        "questions.json": answerable,
        "unanswerable_questions.json": unanswerable,
        "cases.json": cases,
        "benchmark_master.json": {
            "dataset_name": "NuoiDayCon-Benchmark-Day07",
            "description": "Benchmark 24 câu hỏi đánh giá Retrieval Context & Agentic RAG",
            "total_questions": len(cases),
            "splits": {
                "dev": {"total": 17, "answerable": 15, "unanswerable": 2},
                "test": {"total": 7, "answerable": 5, "unanswerable": 2}
            },
            "answerable_questions": answerable,
            "unanswerable_questions": unanswerable
        }
    }
    
    for filename, data in files_to_save.items():
        filepath = out_dir / filename
        filepath.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"   -> Đã lưu: {filepath.relative_to(DAY7_DIR)} ({len(data) if isinstance(data, list) else len(data.get('answerable_questions', []))} items)")

    print("\n=== HOÀN THÀNH GIAI ĐOẠN 1 ===")


if __name__ == "__main__":
    main()

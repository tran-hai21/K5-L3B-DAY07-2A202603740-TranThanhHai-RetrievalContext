"""Agent controller implementations for Day 7: Gemini API and Evidence Grounding."""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

# Ensure local packages are importable
PACKAGES_DIR = Path(__file__).resolve().parents[1] / ".packages"
if PACKAGES_DIR.exists():
    sys.path.insert(0, str(PACKAGES_DIR))


SYSTEM_PROMPT = """Bạn là trợ lý hỏi đáp có căn cứ (evidence-based) cho bộ tài liệu Nuôi Dạy Con.
Quy tắc bắt buộc:
1. Nếu chưa có context hoặc cần thêm thông tin, hãy gọi search:
   {"action": "search", "query": "<từ khóa tìm kiếm>"}
2. Khi context đã có ĐỦ bằng chứng để trả lời, trả lời chính xác, kèm mã trích dẫn [chunk-id] trong câu:
   {"action": "answer", "answer": "<nội dung câu trả lời> [chunk-id]", "citations": ["<chunk-id>"]}
3. Nếu sau khi tìm kiếm mà context KHÔNG có bằng chứng liên quan, hoặc câu hỏi nằm ngoài tài liệu, BẮT BUỘC trả lời từ chối:
   {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}
Chỉ xuất ra đúng định dạng JSON, không kèm bất kỳ giải thích nào khác ngoài JSON.
"""


class GeminiAgentController:
    """Agent Controller powered by real Gemini LLM API (google-genai SDK)."""

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.5-flash"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY hoặc GOOGLE_API_KEY chưa được thiết lập.")
        self.model = model
        from google import genai
        self.client = genai.Client(api_key=self.api_key)

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        prompt = f"{SYSTEM_PROMPT}\n\nCâu hỏi: {question}\n\n"
        if not history:
            prompt += "Lịch sử tìm kiếm: Chưa có.\nHành động tiếp theo (search hoặc answer):"
        else:
            latest = history[-1]
            prompt += f"Ngữ cảnh tài liệu hiện có (Context):\n{latest.get('context', '(trống)')}\n\n"
            prompt += f"Danh sách chunk IDs có trong ngữ cảnh: {latest.get('context_ids', [])}\n"
            prompt += "Dựa vào context trên, hãy quyết định hành động tiếp theo (search nếu cần thêm, hoặc answer có trích dẫn [chunk-id] nếu đã đủ bằng chứng, hoặc từ chối nếu không đủ thông tin):"

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            text = response.text.strip()
            data = json.loads(text)
            if "action" in data:
                return data
        except Exception as e:
            # Fallback parsing if JSON block is wrapped in markdown
            pass
        
        # Default fallback
        if not history:
            return {"action": "search", "query": question}
        return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}


class GroundingAgentController:
    """Deterministic, Evidence-Grounded Agent Controller for reliable evaluation and offline mode."""

    def __init__(self, answer_templates: dict[str, str] | None = None):
        self.templates = answer_templates or {}

    def decide(self, question: str, history: list[dict[str, Any]]) -> dict[str, Any]:
        # Turn 1: No history -> execute initial search
        if not history:
            clean_q = re.sub(r'[\?\,\.\!\:\;]', ' ', question).strip()
            return {"action": "search", "query": clean_q}

        latest = history[-1]
        context = latest.get("context", "")
        context_ids = latest.get("context_ids", [])

        if not context or not context_ids:
            return {"action": "answer", "answer": "Không đủ bằng chứng để trả lời.", "citations": []}

        # Parse blocks in context: [chunk-id | source | page]\ntext
        blocks = re.findall(r'\[([^\|\]]+)\s*\|\s*([^\|\]]+)\s*\|\s*page\s*([^\]]+)\]\s*\n(.*?)(?=\n\n\[|\Z)', context, re.DOTALL)
        
        best_chunk_id = None
        best_score = 0
        best_snippet = ""
        
        q_tokens = set(re.findall(r'\w+', unicodedata.normalize('NFC', question).casefold()))
        stopwords = {"là", "gì", "những", "nào", "trong", "theo", "tài", "liệu", "của", "việc", "cho", "con", "có", "không", "như", "thế", "nào", "được", "với", "về", "các", "và"}
        content_tokens = q_tokens - stopwords

        for cid, source, page, text in blocks:
            cid = cid.strip()
            text_norm = unicodedata.normalize('NFC', text).casefold()
            
            # Count matching meaningful content tokens
            matches = sum(1 for tok in content_tokens if tok in text_norm)
            overlap_ratio = matches / max(len(content_tokens), 1)
            
            if overlap_ratio > best_score:
                best_score = overlap_ratio
                best_chunk_id = cid
                best_snippet = text.strip()

        # Decision threshold:
        # If overlap is sufficient (at least 35% of query content terms found in chunk), generate grounded answer
        if best_chunk_id and best_score >= 0.35:
            # Extract first 2 meaningful sentences from the chunk
            sentences = [s.strip() for s in re.split(r'[\n\.]+', best_snippet) if len(s.strip()) > 15]
            summary_text = ". ".join(sentences[:2]) if sentences else best_snippet[:150]
            answer = f"Theo tài liệu, {summary_text} [{best_chunk_id}]."
            return {
                "action": "answer",
                "answer": answer,
                "citations": [best_chunk_id]
            }

        # If after 1 search no evidence is found, and searches < 2, try refined query
        queries = latest.get("queries", [])
        if len(queries) < 2 and len(content_tokens) > 2:
            refined_query = " ".join(list(content_tokens)[:4])
            return {"action": "search", "query": refined_query}

        # Abstain when evidence is missing or query is unanswerable
        return {
            "action": "answer",
            "answer": "Không đủ bằng chứng để trả lời.",
            "citations": []
        }


def get_controller(api_key: str | None = None, model: str = "gemini-2.5-flash"):
    """Factory to get available controller (Gemini API if key present, else GroundingAgentController)."""
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key:
        try:
            return GeminiAgentController(api_key=key, model=model)
        except Exception:
            pass
    return GroundingAgentController()

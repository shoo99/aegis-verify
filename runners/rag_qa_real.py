"""AEGIS runner — *실제* RAG QA (Qdrant + ollama + 다른-모델 judge).

rag_qa.py(SIM 데모)의 실전 버전. 네 논문 코퍼스(Qdrant)에서 검색 → ollama로 답변 →
*다른 모델* judge로 근거 대조. 이 답변/judge 품질이 낮은 케이스가 곧 프롬프트 개선 신호.

▶ 실행(보통 ollama·Qdrant 있는 머신=RTX 박스에서):
    QDRANT_COLLECTION=my_papers OLLAMA_MODEL=qwen3:8b \
    AEGIS_BACKEND=ollama python3 -m aegis.loop runners/rag_qa_real.py --rounds 3

환경변수(전부 선택, 기본값 아래):
    QDRANT_URL=http://localhost:6333   QDRANT_COLLECTION=papers
    OLLAMA_URL=http://localhost:11434   OLLAMA_MODEL=qwen3:8b
    EMBED_MODEL=bge-m3                  # 코퍼스 색인에 쓴 임베딩과 *반드시 일치*
    JUDGE_MODEL=qwen3:14b              # 답변 모델과 *다른* 모델(self-grade 금지)
    RAG_TOPK=4   TEXT_FIELD=text        # Qdrant payload의 본문 필드명

EVAL_CASES = runners/eval_cases.json (네가 채움): [{"id","q","gold?"}]
  gold 있으면 '그 값이 답에 포함 + judge 통과'면 OK. 없으면 'judge 무플래그'면 OK.
"""
import json
import os
import pathlib
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
PROMPT_PATH = str(HERE / "rag_qa_real_prompt.txt")

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "papers")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:8b")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "bge-m3")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen3:14b")
TOPK = int(os.environ.get("RAG_TOPK", "4"))
TEXT_FIELD = os.environ.get("TEXT_FIELD", "text")


def _http_json(url, payload):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def _embed(text):
    """ollama 임베딩(/api/embeddings). 코퍼스 색인과 같은 EMBED_MODEL이어야 검색이 맞음."""
    return _http_json(f"{OLLAMA_URL}/api/embeddings", {"model": EMBED_MODEL, "prompt": text})["embedding"]


def retrieve(question, k=TOPK):
    """Qdrant 벡터검색 → 상위 k passage 텍스트. ▶ paper-rag가 자체 retriever를 노출하면 그걸로 교체."""
    try:
        from qdrant_client import QdrantClient
    except ImportError:
        raise SystemExit("qdrant-client 필요: pip install qdrant-client (RAG 머신에서 실행)")
    cli = QdrantClient(url=QDRANT_URL)
    hits = cli.search(collection_name=QDRANT_COLLECTION, query_vector=_embed(question), limit=k)
    return [h.payload.get(TEXT_FIELD, "") for h in hits if h.payload]


def generate(prompt, question, passages):
    """ollama 답변 생성: [시스템 프롬프트] + 검색 근거 + 질문."""
    full = f"{prompt}\n\n[근거]\n" + "\n---\n".join(passages) + f"\n\n[질문]\n{question}\n\n[답변]\n"
    out = _http_json(f"{OLLAMA_URL}/api/generate",
                     {"model": OLLAMA_MODEL, "prompt": full, "stream": False})
    return (out.get("response") or "").strip()


def check(case, answer, passages):
    """다른-모델 judge(aegis.verify)로 답변 claim을 근거와 대조 + gold 포함 확인."""
    from aegis import verify
    evidence = "\n---\n".join(passages)
    v = verify(answer, evidence, model=JUDGE_MODEL, backend="ollama")
    gold_ok = (str(case["gold"]) in answer) if case.get("gold") else True
    ok = v["ok"] and gold_ok
    note = ""
    if not gold_ok:
        note = f"missing: gold '{case['gold']}' 답에 없음(근거검색·프롬프트 점검)"
    elif not v["ok"]:
        flags = "; ".join(f"{f['verdict']}:{f['claim'][:40]}" for f in v["flagged"][:3])
        note = f"missing: 근거 미지지 claim — {flags}"
    return ok, note


def _load_cases():
    f = HERE / "eval_cases.json"
    if not f.exists():
        raise SystemExit(f"평가셋 없음: {f} 를 [{{'id','q','gold?'}}] 형식으로 채워라.")
    return json.loads(f.read_text())


EVAL_CASES = _load_cases()


def run_case(prompt, case):
    try:
        passages = retrieve(case["q"])
        ans = generate(prompt, case["q"], passages)
        ok, note = check(case, ans, passages)
        return {"ok": ok, "output": ans, "note": note}
    except SystemExit:
        raise
    except Exception as e:
        return {"ok": False, "output": "", "note": f"runtime: {type(e).__name__}: {str(e)[:80]}"}

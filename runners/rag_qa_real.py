"""AEGIS runner — paper-rag(shoo99/paper-rag) 실연결.

paper-rag의 rag.py를 그대로 import해서 *그들의* hybrid 검색(dense+sparse+RRF+리랭크+dedup)과
ollama LLM으로 답한 뒤, **다른 모델** judge(aegis.verify)로 근거를 대조한다.
최적화 대상(PROMPT) = paper-rag answer()의 (원래 하드코딩된) 시스템 프롬프트.

▶ 실행 (paper-rag·ollama 있는 머신 = RTX 박스에서):
    PAPER_RAG_PATH=~/paper-rag RAG_DB=./rag_qdrant JUDGE_MODEL=qwen3:14b \
    AEGIS_BACKEND=ollama python3 -m aegis.loop runners/rag_qa_real.py --rounds 3

환경변수:
    PAPER_RAG_PATH  paper-rag 디렉토리(기본 ~/paper-rag)
    JUDGE_MODEL     근거 대조 judge 모델 — 답변 LLM(RAG_LLM)과 *반드시 다르게*(기본 qwen3:14b)
    그 외 검색/모델/DB = paper-rag rag.py의 env 그대로(RAG_DB·RAG_LLM·RAG_EMBED·RAG_DEDUP…)

EVAL_CASES = runners/eval_cases.json — 네 논문 질문 + (선택) gold. 근거에 *없는* 질문도 넣어 환각 유도.
"""
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
PROMPT_PATH = str(HERE / "rag_qa_real_prompt.txt")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "qwen3:14b")   # 답변 LLM과 다른 모델(self-grade 금지)


def _rag():
    """paper-rag의 rag.py 모듈 지연 import."""
    p = os.path.expanduser(os.environ.get("PAPER_RAG_PATH", "~/paper-rag"))
    if p not in sys.path:
        sys.path.insert(0, p)
    import rag
    return rag


def _ctx(hits):
    return "\n\n".join(f"[{i+1}] ({h['source']} p.{h['page']}) {h['text']}" for i, h in enumerate(hits))


def run_case(prompt, case):
    try:
        rag = _rag()
        hits = rag.curate(case["q"])                    # 그들 hybrid 검색 + dedup + budget-fit
        if not hits:
            return {"ok": False, "output": "", "note": "missing: 검색 0건(색인/질문 점검)"}
        ctx = _ctx(hits)
        ans = rag.llm(prompt, f"CONTEXT:\n{ctx}\n\nQUESTION: {case['q']}")   # prompt=최적화 대상 시스템 프롬프트
        from aegis import verify
        v = verify(ans, ctx, model=JUDGE_MODEL, backend="ollama")           # 다른-모델 judge
        gold_ok = (str(case["gold"]) in ans) if case.get("gold") else True
        if not gold_ok:
            return {"ok": False, "output": ans, "note": f"missing: gold '{case['gold']}' 답에 없음"}
        if not v["ok"]:
            flags = "; ".join(f"{f['verdict']}:{f['claim'][:40]}" for f in v["flagged"][:3])
            return {"ok": False, "output": ans, "note": f"missing: 근거 미지지 claim — {flags}"}
        return {"ok": True, "output": ans, "note": ""}
    except Exception as e:
        return {"ok": False, "output": "", "note": f"runtime: {type(e).__name__}: {str(e)[:90]}"}


def _load_cases():
    f = HERE / "eval_cases.json"
    if not f.exists():
        raise SystemExit(f"평가셋 없음: {f} 를 [{{'id','q','gold?'}}] 형식으로 채워라.")
    return json.loads(f.read_text())


EVAL_CASES = _load_cases()

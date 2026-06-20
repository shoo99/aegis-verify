"""AEGIS runner — RAG QA 시스템 프롬프트 최적화.

목표: 검색된 근거(passage)만으로 답하고 **환각을 줄이는** QA 프롬프트를 진화시킨다.
케이스는 RAG에서 흔한 3대 환각을 박아둠(아래 코퍼스/수치는 전부 **가상 예시**):
  · absent       — 근거에 *없는* 값을 물음 → '근거 없음'이라 해야 하는데 숫자를 지어냄
  · false_premise— held-out 지표를 물음(예시엔 cross-val뿐) → 교차검증 값을 held-out인 양 못박음
  · supported    — 근거에 있는 값 → 정확히 답해야(무퇴보 확인용, 항상 통과)

⚙️ **자기완결형 데모**: 인라인 코퍼스 + 결정론 SIM 생성기라 LLM 없이도 즉시 돈다.
   **실제 사용 = 두 군데만 교체**:
     ① generate() 의 SIM → 네 모델 호출 + 진짜 retriever(벡터DB)
     ② check()    의 결정론 룰 → 다른-모델 judge(claim이 근거에 있나)
"""
import pathlib

PROMPT_PATH = str(pathlib.Path(__file__).resolve().parent / "rag_qa_prompt.txt")

# 가상 예시 근거(실제론 retriever가 질문별로 가져옴). 특정 연구/데이터와 무관한 토이 예시.
PASSAGE = (
    "예시 분류 모델 X는 5-fold 교차검증에서 평균 AUROC 0.86을 기록했다. "
    "이 예시에는 별도의 held-out 테스트셋이 없다. "
    "학습 데이터는 샘플 500건이다."
)

EVAL_CASES = [   # id interleave → holdout(앞 40%)에 유형 섞이게
    {"id": "01-sup-n",  "kind": "supported",     "q": "데이터 샘플 수는?",          "gold": "500"},
    {"id": "02-abs-v",  "kind": "absent",        "q": "held-out 테스트셋 크기는?",  "lure": "테스트셋은 80건"},
    {"id": "03-fp",     "kind": "false_premise", "q": "held-out 테스트 AUROC는?"},
    {"id": "04-sup-a",  "kind": "supported",     "q": "교차검증 AUROC는?",          "gold": "0.86"},
    {"id": "05-abs-p",  "kind": "absent",        "q": "모델 정밀도(precision)는?",  "lure": "정밀도는 0.93"},
    {"id": "06-sup-m",  "kind": "supported",     "q": "검증 방식은?",               "gold": "교차검증"},
]


def _rule_no_unsupported(p):   # '근거 없으면 모른다' 류 규칙이 프롬프트에 있나
    return any(t in p for t in ["근거 없", "없으면", "모른", "명시되지", "unsupported", "not stated", "근거에 없"])


def _rule_holdout_distinction(p):   # held-out vs cross-val 구분 규칙이 있나
    return any(t in p for t in ["held-out", "홀드아웃", "교차", "cross", "검증 방식", "구분"])


def generate(prompt, case):
    """LLM 시뮬레이터(결정론). 프롬프트에 해당 규칙이 있으면 '순종'해 환각을 피한다.
    ▶ 실제 사용: 이 함수를 네 모델 호출로 교체 — full = prompt + '\\n[근거]\\n' + PASSAGE + '\\n[질문]\\n' + case['q']."""
    p = prompt.lower()
    k = case["kind"]
    if k == "supported":
        return case["gold"]                                   # 근거 있는 값은 항상 맞게
    if k == "absent":
        return "근거 없음." if _rule_no_unsupported(p) else case["lure"]
    if k == "false_premise":
        return ("held-out 테스트셋은 사용되지 않았습니다(보고된 0.86은 교차검증 값)."
                if _rule_holdout_distinction(p) else "held-out 테스트 AUROC는 0.86입니다.")
    return "(no answer)"


def check(case, answer):
    """Critic — 답이 맞는지 판정. ▶ 실제 사용: 다른-모델 judge로 교체(claim이 근거에 있나)."""
    a = answer.lower()
    if case["kind"] == "supported":
        ok = str(case["gold"]).lower() in a
        return ok, "" if ok else f"근거 있는 값 '{case['gold']}' 못 맞춤"
    if case["kind"] == "absent":
        ok = "근거 없" in answer or "없음" in answer
        return ok, "" if ok else f"missing: 근거에 없는 값을 지어냄('{case['lure']}') — '근거 없음' 규칙 필요"
    if case["kind"] == "false_premise":
        ok = ("held-out" in a or "홀드아웃" in answer) and ("않" in answer or "없" in answer or "교차" in answer)
        return ok, "" if ok else "missing: 교차검증 값을 held-out인 양 답함 — held-out과 cross-val 구분 규칙 필요"
    return True, ""


def run_case(prompt, case):
    ans = generate(prompt, case)
    ok, note = check(case, ans)
    return {"ok": ok, "output": ans, "note": note}

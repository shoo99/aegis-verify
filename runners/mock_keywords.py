"""AEGIS-lite 데모 러너 (결정론·LLM 불필요) — 루프 동작 검증용.

가상 태스크: 프롬프트가 '부정문(negation) 처리' 능력을 명시해야 negation 케이스 통과.
초기 프롬프트엔 그 능력이 없어 negation 케이스가 train·holdout 양쪽에서 실패 →
Digester가 'missing:negation' 증거 추출 → (stub)Evolver가 능력 추가 → holdout에서도
개선·무퇴보 → 게이트 채택. 일반화되는 개선이 어떻게 통과하는지 보여줌.

실제 사용: 이 파일을 복제해 run_case 를 네 에이전트 실행으로 바꾸고 EVAL_CASES 를 채우면 끝.
"""
import pathlib

PROMPT_PATH = str(pathlib.Path(__file__).resolve().parent / "mock_prompt.txt")

EVAL_CASES = [   # id 를 interleave 해 holdout 분할(앞 40%)이 basic·negation 섞이게
    {"id": "01-basic", "type": "basic"},
    {"id": "02-neg", "type": "negation"},
    {"id": "03-basic", "type": "basic"},
    {"id": "04-neg", "type": "negation"},
    {"id": "05-basic", "type": "basic"},
    {"id": "06-neg", "type": "negation"},
]


def run_case(prompt: str, case: dict) -> dict:
    if case["type"] == "basic":
        return {"ok": True, "output": "", "note": ""}
    # negation 케이스: 프롬프트에 'negation' 능력이 명시돼야 통과
    ok = "negation" in prompt.lower()
    return {"ok": ok, "output": "",
            "note": "" if ok else "missing:negation (부정문/예외 케이스 처리 규칙 없음)"}

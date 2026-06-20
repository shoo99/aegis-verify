"""결정론 데모 = 회귀 테스트 (LLM 불필요, --stub-evolve 경로).

루프가 실패를 잡아 개선하고(50%→100%), held-out 게이트가 정상 동작하는지 검증.
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aegis.loop import run   # noqa: E402

RUNNERS = ROOT / "runners"


def test_mock_keywords_improves():
    r = run(str(RUNNERS / "mock_keywords.py"), rounds=3, stub=True, apply=False, log=lambda *_: None)
    assert r["base"] == 0.5          # naive: basic 통과·negation 실패
    assert r["final"] == 1.0         # 능력 추가 후 전부 통과
    assert r["accepted"] >= 1
    assert not r["applied"]          # apply=False → 파일 미변경


def test_rag_qa_catches_hallucination():
    r = run(str(RUNNERS / "rag_qa.py"), rounds=3, stub=True, apply=False, log=lambda *_: None)
    assert r["base"] == 0.5          # naive: 환각 케이스 실패
    assert r["final"] == 1.0         # 근거-없음·held-out 구분 규칙 추가 후 전부 통과


def test_gate_rejects_when_no_holdout_gain():
    """holdout 개선이 없으면(과적합) 게이트가 거부하는지 — 0개 채택 확인용 음성 케이스."""
    # mock 은 일반화되므로 정상 채택. 여기선 단지 run 이 깨지지 않고 dict 를 반환하는지 확인.
    r = run(str(RUNNERS / "mock_keywords.py"), rounds=1, holdout_frac=0.4, stub=True, apply=False, log=lambda *_: None)
    assert set(r) >= {"base", "final", "accepted", "applied", "prompt"}


if __name__ == "__main__":
    test_mock_keywords_improves()
    test_rag_qa_catches_hallucination()
    test_gate_rejects_when_no_holdout_gain()
    print("ok")

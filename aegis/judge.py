"""다른-모델 judge — 답변을 claim 단위로 쪼개, 각 claim이 근거(evidence)에 의해
지지되는지 *답한 모델과 다른 모델*로 검증한다.

원칙(이 라이브러리의 핵심):
  · self-grade 금지 — 답한 모델이 자기 답을 채점하면 보상해킹/고무도장. judge는 다른 모델.
  · 추론 금지 — 근거에 명시 안 됐으면 unsupported(있을 법함 ≠ 지지됨).
  · flag = '거짓말'이 아니라 '근거가 빠졌을' 수도 — retrieval miss와 모델오류를 구분해 보라.
  · ⚠️ judge도 ground truth가 아니다 — held-out 라벨로 judge 자체의 정확도를 따로 재라.
"""
import json
import re

from .backends import call_meta


def split_claims(answer, model=None, backend=None):
    """답변을 독립적·검증가능한 사실 주장 리스트로 분해."""
    instr = (
        "다음 답변을 독립적으로 검증 가능한 '사실 주장' 리스트로 쪼개라. "
        "의견·접속사는 빼고 각 주장은 한 문장. JSON 배열만 출력: [\"주장1\", \"주장2\"].\n\n답변:\n" + answer
    )
    out = call_meta(instr, backend=backend, model=model)
    m = re.search(r"\[.*\]", out, re.S)
    if not m:
        return [answer.strip()] if answer.strip() else []
    try:
        return [str(c).strip() for c in json.loads(m.group(0)) if str(c).strip()]
    except Exception:
        return [answer.strip()]


def judge_claim(claim, evidence, model=None, backend=None):
    """claim 하나를 근거와 대조 → supported|unsupported|contradicted + 한줄 이유."""
    instr = (
        "주어진 [근거]만으로 [주장]의 진위를 판정하라. 근거에 명시되지 않았으면 추론하지 말고 "
        "unsupported. 근거와 어긋나면 contradicted. JSON만 출력: "
        '{"verdict":"supported|unsupported|contradicted","reason":"한 줄"}\n\n'
        f"[근거]\n{evidence}\n\n[주장]\n{claim}"
    )
    out = call_meta(instr, backend=backend, model=model)
    m = re.search(r"\{.*\}", out, re.S)
    if not m:
        return {"verdict": "unsupported", "reason": "판정 파싱 실패"}
    try:
        d = json.loads(m.group(0))
        v = str(d.get("verdict", "unsupported")).lower()
        return {"verdict": v if v in ("supported", "unsupported", "contradicted") else "unsupported",
                "reason": str(d.get("reason", ""))[:160]}
    except Exception:
        return {"verdict": "unsupported", "reason": "판정 파싱 실패"}


def verify(answer, evidence, model=None, splitter_model=None, backend=None):
    """답변 전체 검증 → claim별 판정 + 요약 플래그.

    model         : judge 모델(반드시 답변 생성 모델과 *다르게*).
    splitter_model: claim 분해 모델(없으면 judge 모델).
    반환: {"claims":[{claim,verdict,reason}], "ok":bool, "flagged":[...]}
    """
    claims = split_claims(answer, model=splitter_model or model, backend=backend)
    results = [{"claim": c, **judge_claim(c, evidence, model=model, backend=backend)} for c in claims]
    flagged = [r for r in results if r["verdict"] != "supported"]
    return {"claims": results, "ok": not flagged, "flagged": flagged}

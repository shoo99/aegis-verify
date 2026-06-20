"""aegis — AI 검증 구조 툴킷.

자기개선(AEGIS 루프)과 검증(다른-모델 judge)을 *정직하게* 한다:
held-out 없으면 측정은 거짓이고, judge는 다른 모델로, LLM 위엔 결정론 게이트를 둔다.
"""
from .backends import call_meta
from .judge import judge_claim, split_claims, verify
from .loop import evaluate, evolve, gate, run

__version__ = "0.1.0"
__all__ = ["run", "evolve", "gate", "evaluate", "verify", "judge_claim", "split_claims", "call_meta"]

# aegis

**AI 검증 구조 툴킷** — 에이전트를 *정직하게* 자기개선(AEGIS 루프)하고 *정직하게* 검증(다른-모델 judge)한다.

"AI가 AI를 검증한다"는 방향은 맞다. 어려운 건 **검증을 어떻게 신뢰하느냐**다. 이 라이브러리는
그 어려운 부분 — held-out, 다양성, 결정론 게이트 — 을 기본값으로 강제한다.

## 네 가지 원칙

1. **held-out 없으면 측정은 거짓이다.** "검증 후 오류 0개"는 *측정된* 0이지 *실제* 0이 아니다.
   ground truth/held-out 라벨 없이는 검증기가 놓친 오류가 그냥 0으로 보인다. → 루프·judge 모두 holdout을 강제.
2. **검증은 숫자가 아니라 다양성이다.** 같은 모델 300개는 같은 환각을 300번 한다(self-grade 고무도장).
   judge는 *답한 모델과 다른 모델*로.
3. **LLM 위엔 결정론 게이트.** LLM은 *제안*만, 출시 결정은 테스트·타입·근거대조 같은 결정론 규칙이.
   보상해킹 방어.
4. **flag ≠ 거짓말.** 검증이 "틀렸다"고 표시한 게 사실은 *근거가 빠진 것*(retrieval miss)일 때가 많다.
   모델오류와 검색실패를 구분하라.

## 두 축

### 1) AEGIS 루프 — 실패 트레이스로 프롬프트 자동 개선
Digester(실패 압축) → Evolver(개선 제안, LLM) → **Critic 게이트(holdout 개선 AND 무퇴보일 때만 채택)**.

```bash
# 데모(LLM 불필요, 결정론): 50% → 100%
python3 -m aegis.loop runners/mock_keywords.py --stub-evolve

# 실제: claude가 실패 보고 프롬프트 개선 제안 → holdout 게이트가 채택 판단(드라이런, --apply로 출시)
python3 -m aegis.loop runners/rag_qa.py --rounds 3

# 실(real) RAG — Qdrant 코퍼스 + ollama + 다른-모델 judge (ollama/Qdrant 있는 머신에서)
#   runners/eval_cases.json 채우고, 환경변수로 컬렉션/모델 지정 → README의 runners/rag_qa_real.py 참고
QDRANT_COLLECTION=my_papers OLLAMA_MODEL=qwen3:8b AEGIS_BACKEND=ollama \
  python3 -m aegis.loop runners/rag_qa_real.py --rounds 3
```

러너 계약 3개만 정의하면 어떤 에이전트에도 붙는다:
```python
PROMPT_PATH = "최적화할 프롬프트 파일"
EVAL_CASES  = [{"id": ..., ...}]
def run_case(prompt, case) -> {"ok": bool, "output": str, "note": str}: ...
```

### 2) 다른-모델 judge — 답변을 claim 단위로 근거 대조
```python
from aegis import verify
r = verify(answer, evidence, model="haiku")   # judge는 답변 생성 모델과 *다른* 모델로
# → {"claims":[{claim, verdict: supported|unsupported|contradicted, reason}], "ok":bool, "flagged":[...]}
```
근거에 명시 안 됐으면 `unsupported`(추론 금지). `--stub`이 아니라 진짜 RAG라면 evidence=검색된 passage.

## 백엔드 — claude / ollama
환경변수로 전환. **하이브리드(기본·추천)**: 강한 모델이 제안/judge, 약한 로컬 모델이 대상.
```bash
python3 -m aegis.loop runners/rag_qa.py                 # claude (기본)
AEGIS_BACKEND=ollama AEGIS_MODEL=qwen3:8b python3 -m aegis.loop runners/rag_qa.py   # 완전 로컬
```
`call_meta(instr, backend=, model=)` 로 호출별 모델 지정 → judge에 답변모델과 다른 모델 강제.

## 설치
```bash
pip install -e .        # 또는 그냥 python3 -m aegis.loop ...
pytest                  # 결정론 데모 = 회귀 테스트
```

## 한계 (정직하게)
- judge도 ground truth가 아니다 — judge 자체의 정확도를 held-out 라벨로 따로 재라.
- 약한 모델 Evolver/judge는 가끔 출력 형식을 이탈한다(그 라운드 스킵하고 진행, 안 죽음).
- 프롬프트(이산·텍스트)만 개선. 도구/메모리 구조 진화·모델 가중치 학습(공진화)은 범위 밖.
- 비용: 라운드×케이스만큼 LLM 호출. 대부분 태스크엔 단발 모델로 충분 — 검증은 *틀리면 비싼* 곳에.

## 영감
Xiaomi HarnessX(Digester→Planner→Evolver→Critic, 공진화)의 *경량·정직* 버전. 모델 학습은 빼고
하네스 개선 + 검증 규율에 집중. 약한 로컬 모델일수록 하네스 개선 이득이 크다(원작 결론).

MIT.

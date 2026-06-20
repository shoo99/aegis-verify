"""LLM 백엔드 어댑터 — claude(클라우드) / ollama(로컬) 통일 호출.

환경변수 기본값:
  AEGIS_BACKEND=claude|ollama   (기본 claude)
  AEGIS_META=<바이너리>         (기본: claude=/opt/homebrew/bin/claude · ollama=ollama)
  AEGIS_MODEL=<모델>            (기본: claude=sonnet · ollama=qwen3:8b)

call_meta(instr, backend=, model=, meta=) 로 호출별 오버라이드 가능 →
'다른-모델 judge'(답변모델과 다른 모델로 검증)를 명시적으로 강제할 수 있다.
"""
import os
import subprocess

DEFAULT_BACKEND = os.environ.get("AEGIS_BACKEND", "claude").lower()
_CLAUDE_BIN = "/opt/homebrew/bin/claude"


def _meta_default(backend):
    return "ollama" if backend == "ollama" else _CLAUDE_BIN


def _model_default(backend):
    return "qwen3:8b" if backend == "ollama" else "sonnet"


DEFAULT_META = os.environ.get("AEGIS_META", _meta_default(DEFAULT_BACKEND))
DEFAULT_MODEL = os.environ.get("AEGIS_MODEL", _model_default(DEFAULT_BACKEND))


def call_meta(instr, backend=None, model=None, meta=None, timeout=300):
    """text-in/text-out LLM 호출. stdout 반환. 실패 시 RuntimeError."""
    backend = (backend or DEFAULT_BACKEND).lower()
    # backend가 기본과 다르면 그 backend의 기본 바이너리/모델을 쓴다(명시 오버라이드 우선).
    meta = meta or (DEFAULT_META if backend == DEFAULT_BACKEND else _meta_default(backend))
    model = model or (DEFAULT_MODEL if backend == DEFAULT_BACKEND else _model_default(backend))
    if backend == "ollama":                      # ollama run <model> · 프롬프트 stdin
        r = subprocess.run([meta, "run", model], input=instr,
                           capture_output=True, text=True, timeout=timeout)
    else:                                        # claude -p <prompt> --model <model>
        r = subprocess.run([meta, "-p", instr, "--model", model],
                           capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout or "")[:160])
    return r.stdout

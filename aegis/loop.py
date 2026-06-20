"""AEGIS-lite 루프 — 실패 트레이스로 에이전트 하네스(프롬프트)를 자동 개선.

Digester(실패 압축) → Evolver(개선 제안, LLM) → Critic 게이트(holdout 개선 AND 무퇴보)
→ N라운드. **LLM은 제안만, 출시는 결정론 게이트가.** held-out 분할로 과적합/보상해킹 방어.

  python3 -m aegis.loop <runner.py> [--rounds 2] [--holdout 0.4] [--apply] [--stub-evolve]
"""
import importlib.util
import pathlib
import re
import sys

from .backends import DEFAULT_BACKEND, call_meta


def load_runner(path):
    spec = importlib.util.spec_from_file_location("runner", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def evaluate(runner, prompt, cases):
    out = []
    for c in cases:
        try:
            r = runner.run_case(prompt, c) or {}
        except Exception as e:
            r = {"ok": False, "note": f"runner error: {e}"}
        out.append({"id": c.get("id"), "ok": bool(r.get("ok")), "note": r.get("note", "")})
    return out


def passrate(results):
    return (sum(r["ok"] for r in results) / len(results)) if results else 0.0


def digest(results):
    """Digester — 실패를 구조화 증거(케이스별 근거)로 압축."""
    fails = [r for r in results if not r["ok"]]
    if not fails:
        return None
    return "실패 케이스 증거:\n" + "\n".join(f"- {r['id']}: {r['note'] or '(근거없음)'}" for r in fails)


def evolve(prompt, digest_text, stub=False):
    """Evolver — 현 프롬프트 + 실패증거 → 개선 후보. stub=결정론(테스트), 기본=LLM."""
    if stub:
        adds = [ln.split("missing:", 1)[1].split("(")[0].strip()
                for ln in digest_text.splitlines() if "missing:" in ln]
        return ((prompt.rstrip() + "\n" + "\n".join(f"- {a}" for a in adds)).strip(),
                "stub:+" + ",".join(adds)) if adds else (None, "stub: 추가 없음")
    instr = (
        "너는 에이전트 프롬프트 최적화기다. 아래 [현재 프롬프트]가 일부 케이스에서 실패한다.\n"
        "[실패 증거] 패턴만 보고, 잘 되던 걸 깨지 않으면서 실패를 고치는 개선 프롬프트를 써라.\n"
        "정답을 외워넣지 말고(과적합 금지) 일반 규칙으로 고쳐라.\n"
        "출력: <prompt>…개선된 전체 프롬프트…</prompt> 다음 한 줄 변경요약.\n\n"
        f"[현재 프롬프트]\n{prompt}\n\n[실패 증거]\n{digest_text}"
    )
    try:
        out = call_meta(instr)
    except Exception as e:
        return None, f"evolve 실패({DEFAULT_BACKEND}): {e}"
    m = re.search(r"<prompt>(.*?)</prompt>", out, re.S)
    if not m:
        return None, f"evolve({DEFAULT_BACKEND}): <prompt> 파싱 실패"
    return m.group(1).strip(), out.split("</prompt>", 1)[-1].strip().replace("\n", " ")[:80] or "(요약없음)"


def gate(runner, old, new, holdout):
    """Critic 게이트 — holdout에서 개선 AND 무퇴보(통과하던 게 안 깨짐)일 때만 채택."""
    ro, rn = evaluate(runner, old, holdout), evaluate(runner, new, holdout)
    regress = any(o["ok"] and not n["ok"] for o, n in zip(ro, rn))
    return (passrate(rn) > passrate(ro) and not regress), passrate(ro), passrate(rn), regress


def run(runner_path, rounds=2, holdout_frac=0.4, stub=False, apply=False, log=print):
    """AEGIS 루프 실행 → {base, final, prompt, accepted, applied}."""
    runner = load_runner(runner_path)
    prompt = pathlib.Path(runner.PROMPT_PATH).read_text()
    cases = sorted(runner.EVAL_CASES, key=lambda c: str(c.get("id")))   # 결정론 분할
    k = max(1, int(len(cases) * holdout_frac))
    holdout, train = cases[:k], cases[k:]
    base = passrate(evaluate(runner, prompt, cases))
    log(f"베이스라인 pass {base*100:.0f}% · train {len(train)} / holdout {len(holdout)}"
        + (" · [stub]" if stub else f" · [{DEFAULT_BACKEND}]"))

    best, accepted = prompt, 0
    for it in range(rounds):
        dg = digest(evaluate(runner, best, train))
        if not dg:
            log(f"[r{it+1}] train 실패 없음 — 종료")
            break
        cand, note = evolve(best, dg, stub)
        if not cand:
            log(f"[r{it+1}] {note}")
            continue
        ok, po, pn, reg = gate(runner, best, cand, holdout)
        tag = "✅채택" if ok else ("❌거부·퇴보" if reg else "❌거부·holdout 개선없음")
        log(f"[r{it+1}] holdout {po*100:.0f}%→{pn*100:.0f}% {tag} · {note}")
        if ok:
            best, accepted = cand, accepted + 1

    final = passrate(evaluate(runner, best, cases))
    log(f"최종 pass {final*100:.0f}%  (Δ{(final-base)*100:+.0f}%p, 전체셋)")
    applied = False
    if best != prompt and apply:
        pathlib.Path(runner.PROMPT_PATH).write_text(best)
        applied = True
        log("→ 프롬프트 파일 갱신(출시).")
    elif best != prompt:
        log("→ 개선됨(미적용). --apply 로 출시.")
    else:
        log("→ 채택된 개선 없음.")
    return {"base": base, "final": final, "prompt": best, "accepted": accepted, "applied": applied}


def _flag(name, default):
    if name in sys.argv:
        i = sys.argv.index(name)
        if i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print(__doc__)
        return
    run(args[0], rounds=int(_flag("--rounds", 2)), holdout_frac=float(_flag("--holdout", 0.4)),
        stub="--stub-evolve" in sys.argv, apply="--apply" in sys.argv)


if __name__ == "__main__":
    main()

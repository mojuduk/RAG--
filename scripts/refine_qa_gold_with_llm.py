import argparse
import csv
import json
import os
import re
import time
from pathlib import Path

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

SYSTEM_PROMPT = (
    "你是RAG测试集质检助手。你的任务是把题目修成可核验、单一事实问答。"
    "必须保留原始语义，不得捏造新事实。"
    "输出严格JSON对象，字段为: decision, normalized_question, normalized_gold_answer, normalized_keywords, reason。"
    "decision只能是 keep/edit/drop。"
    "normalized_keywords 必须是数组，建议3到5个关键词。"
)


def _load_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise RuntimeError("Input JSON must be a list")
    return data


def _to_keywords(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    if "|" in s:
        return [x.strip() for x in s.split("|") if x.strip()]
    if "," in s:
        return [x.strip() for x in s.split(",") if x.strip()]
    return [s]


def _safe_json_parse(text: str):
    text = (text or "").strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return {}
    return {}


def _build_user_prompt(row):
    q = str(row.get("question") or "").strip()[:160]
    a = str(row.get("gold_answer") or "").strip()[:260]
    kws = _to_keywords(row.get("evidence_keywords"))[:5]
    obj = {
        "id": row.get("id"),
        "question": q,
        "gold_answer": a,
        "evidence_keywords": kws,
        "gold_page": row.get("gold_page"),
    }
    return (
        "请对以下测试题做质检和规范化。要求:\n"
        "1) 只保留可核验、单一事实问答;\n"
        "2) 问句控制在8~35字;\n"
        "3) 答案尽量1句且不超过80字;\n"
        "4) 关键词3~5个，必须能在证据中定位;\n"
        "5) 若题目不适合评测，decision=drop。\n\n"
        f"输入对象:\n{json.dumps(obj, ensure_ascii=False)}"
    )


def _call_llm(client, model, row, temperature=0, timeout=120, max_tokens=400):
    user_prompt = _build_user_prompt(row)
    resp = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    text = ""
    if resp.choices and resp.choices[0].message:
        text = resp.choices[0].message.content or ""
    return _safe_json_parse(text)


def _normalize_suggestion(row, sug):
    decision = str(sug.get("decision") or "keep").strip().lower()
    if decision not in {"keep", "edit", "drop"}:
        decision = "keep"

    nq = str(sug.get("normalized_question") or "").strip()
    na = str(sug.get("normalized_gold_answer") or "").strip()
    nk = sug.get("normalized_keywords")
    if not isinstance(nk, list):
        nk = _to_keywords(nk)
    nk = [str(x).strip() for x in nk if str(x).strip()][:5]

    out = {
        "id": row.get("id"),
        "question": str(row.get("question") or "").strip(),
        "gold_answer": str(row.get("gold_answer") or "").strip(),
        "evidence_keywords": _to_keywords(row.get("evidence_keywords")),
        "gold_page": row.get("gold_page"),
        "decision": decision,
        "suggested_question": nq,
        "suggested_answer": na,
        "suggested_keywords": nk,
        "reason": str(sug.get("reason") or "").strip(),
    }
    return out


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows):
    fields = [
        "id",
        "decision",
        "question",
        "gold_answer",
        "evidence_keywords",
        "gold_page",
        "suggested_question",
        "suggested_answer",
        "suggested_keywords",
        "reason",
        "manual_decision",
        "manual_fix_question",
        "manual_fix_answer",
        "manual_fix_keywords",
        "manual_fix_page",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "id": r.get("id", ""),
                    "decision": r.get("decision", ""),
                    "question": r.get("question", ""),
                    "gold_answer": r.get("gold_answer", ""),
                    "evidence_keywords": " | ".join(r.get("evidence_keywords", [])),
                    "gold_page": r.get("gold_page", ""),
                    "suggested_question": r.get("suggested_question", ""),
                    "suggested_answer": r.get("suggested_answer", ""),
                    "suggested_keywords": " | ".join(r.get("suggested_keywords", [])),
                    "reason": r.get("reason", ""),
                    "manual_decision": "",
                    "manual_fix_question": "",
                    "manual_fix_answer": "",
                    "manual_fix_keywords": "",
                    "manual_fix_page": "",
                    "notes": "",
                }
            )


def main():
    p = argparse.ArgumentParser(description="Refine QA gold set with stronger cloud LLM (suggestion-only)")
    p.add_argument("--in", dest="in_path", default="data/qa_gold_v1.json")
    p.add_argument("--out-json", dest="out_json", default="data/qa_gold_v1_llm_suggest.json")
    p.add_argument("--out-csv", dest="out_csv", default="data/qa_gold_v1_llm_review.csv")
    p.add_argument("--model", default="qwen3.5-plus")
    p.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL") or "")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY")
    p.add_argument("--limit", type=int, default=0, help="0 means all")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--timeout", type=float, default=120.0)
    p.add_argument("--max-tokens", type=int, default=400)
    args = p.parse_args()

    if OpenAI is None:
        raise RuntimeError("openai package not installed. Please install: pip install openai")

    api_key = os.getenv(args.api_key_env, "")
    if not api_key:
        raise RuntimeError(f"Missing API key env: {args.api_key_env}")
    if not args.base_url:
        raise RuntimeError("Missing --base-url (or OPENAI_BASE_URL)")

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout, max_retries=2)

    rows = _load_rows(Path(args.in_path))
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out = []
    fail = 0
    for i, row in enumerate(rows, 1):
        try:
            sug = _call_llm(
                client,
                args.model,
                row,
                timeout=args.timeout,
                max_tokens=args.max_tokens,
            )
            item = _normalize_suggestion(row, sug)
        except Exception as e:
            fail += 1
            item = _normalize_suggestion(row, {"decision": "keep", "reason": f"llm_error: {e}"})
        out.append(item)
        if args.sleep > 0:
            time.sleep(args.sleep)
        if i % 10 == 0:
            print(f"processed {i}/{len(rows)}")

    _write_json(Path(args.out_json), out)
    _write_csv(Path(args.out_csv), out)

    cnt = {"keep": 0, "edit": 0, "drop": 0}
    for x in out:
        d = x.get("decision", "keep")
        cnt[d] = cnt.get(d, 0) + 1

    print(f"Done. total={len(out)} fail={fail}")
    print(f"decision_count={cnt}")
    print(f"json -> {args.out_json}")
    print(f"csv  -> {args.out_csv}")


if __name__ == "__main__":
    main()

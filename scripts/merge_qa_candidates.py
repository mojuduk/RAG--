import argparse
import json
import re
from pathlib import Path


def load_list(path):
    p = Path(path)
    if not p.exists():
        return []
    data = json.loads(p.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        return data
    return []


def norm_q(text):
    text = str(text or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    return text


def quality(item):
    q = str(item.get("question") or "")
    a = str(item.get("gold_answer") or "")
    k = item.get("evidence_keywords") or []
    score = 0
    score += min(len(q), 80)
    score += min(len(a), 120)
    score += 10 if k else 0
    if "source" in item and item["source"] == "ragas":
        score += 5
    return score


def merge(rule_rows, ragas_rows, limit=120):
    merged = {}
    for src, rows in [("rule", rule_rows), ("ragas", ragas_rows)]:
        for row in rows:
            q = str(row.get("question") or "").strip()
            if not q:
                continue
            item = {
                "id": row.get("id"),
                "question": q,
                "gold_answer": str(row.get("gold_answer") or "").strip(),
                "evidence_keywords": row.get("evidence_keywords") or [],
                "gold_page": row.get("gold_page"),
                "source": row.get("source", src),
            }
            key = norm_q(q)
            if key not in merged or quality(item) > quality(merged[key]):
                merged[key] = item

    out = list(merged.values())
    out.sort(key=quality, reverse=True)
    out = out[:limit]
    for i, x in enumerate(out, 1):
        x["id"] = f"merged_{i:04d}"
    return out


def main():
    parser = argparse.ArgumentParser(description="Merge rule-based and ragas QA candidates.")
    parser.add_argument("--rule", default="data/qa_candidates_rule.json")
    parser.add_argument("--ragas", default="data/qa_candidates_ragas.json")
    parser.add_argument("--out", default="data/qa_candidates_merged.json")
    parser.add_argument("--limit", type=int, default=120)
    args = parser.parse_args()

    rule_rows = load_list(args.rule)
    ragas_rows = load_list(args.ragas)
    merged = merge(rule_rows, ragas_rows, limit=args.limit)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Merged {len(merged)} QA candidates -> {out_path}")


if __name__ == "__main__":
    main()

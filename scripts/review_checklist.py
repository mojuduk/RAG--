import argparse
import json
import re
from pathlib import Path


def is_garbled(text):
    text = str(text or "")
    if not text:
        return True
    sample = text[:200]
    bad_markers = ["�", "锟", "鏂", "鍙", "绗", "娴", "浠", "璇", "鐨", "鍥"]
    bad_hits = sum(sample.count(m) for m in bad_markers)
    if bad_hits >= 4:
        return True
    valid = re.findall(r"[\u4e00-\u9fffA-Za-z0-9，。！？：；、（）\-\s]", sample)
    return (len(valid) / max(len(sample), 1)) < 0.68


def review(rows):
    out = []
    for i, row in enumerate(rows, 1):
        q = str(row.get("question") or "")
        a = str(row.get("gold_answer") or "")
        issues = []
        if not q.strip():
            issues.append("empty_question")
        if not a.strip():
            issues.append("empty_answer")
        if len(q) < 8:
            issues.append("short_question")
        if len(a) < 6:
            issues.append("short_answer")
        if "..." in q:
            issues.append("ellipsis_question")
        if is_garbled(q) or is_garbled(a):
            issues.append("garbled_text")

        kws = row.get("evidence_keywords") or []
        if not isinstance(kws, list):
            issues.append("bad_keywords_type")
        elif len(kws) == 0:
            issues.append("missing_keywords")

        out.append(
            {
                "id": row.get("id", f"row_{i:04d}"),
                "question": q,
                "gold_answer": a,
                "issue_count": len(issues),
                "issues": "|".join(issues),
            }
        )
    return out


def main():
    parser = argparse.ArgumentParser(description="Review QA candidates and output issue report.")
    parser.add_argument("--in", dest="in_path", default="data/qa_candidates_merged.json")
    parser.add_argument("--out", dest="out_path", default="data/qa_review_report.json")
    args = parser.parse_args()

    rows = json.loads(Path(args.in_path).read_text(encoding="utf-8-sig"))
    report = review(rows)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_path).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    bad = sum(1 for x in report if x["issue_count"] > 0)
    print(f"Reviewed {len(report)} rows, problematic: {bad} -> {args.out_path}")


if __name__ == "__main__":
    main()

import argparse
import csv
import json
from pathlib import Path


def _split_keywords(text):
    text = (text or "").strip()
    if not text:
        return []
    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [text]


def load_draft(path: Path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise RuntimeError("Draft JSON must be a list")
    return {str(x.get("id")): x for x in data}


def main():
    p = argparse.ArgumentParser(description="Apply manual review CSV and generate final qa_gold_v1.json")
    p.add_argument("--draft", default="data/qa_gold_v1_draft.json")
    p.add_argument("--manual-csv", default="data/qa_gold_v1_manual_review.csv")
    p.add_argument("--out", default="data/qa_gold_v1.json")
    args = p.parse_args()

    draft_map = load_draft(Path(args.draft))
    out = []
    kept = 0
    dropped = 0
    edited = 0

    with Path(args.manual_csv).open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        rid = str(row.get("id", "")).strip()
        if not rid or rid not in draft_map:
            continue
        base = dict(draft_map[rid])
        decision = (row.get("manual_decision") or "").strip().lower()
        if not decision:
            decision = "keep"

        if decision == "drop":
            dropped += 1
            continue

        if decision == "edit":
            q = (row.get("manual_fix_question") or "").strip()
            a = (row.get("manual_fix_answer") or "").strip()
            kw = _split_keywords(row.get("manual_fix_keywords") or "")
            pg = (row.get("manual_fix_page") or "").strip()
            if q:
                base["question"] = q
            if a:
                base["gold_answer"] = a
            if kw:
                base["evidence_keywords"] = kw
            if pg:
                try:
                    base["gold_page"] = int(pg)
                except Exception:
                    base["gold_page"] = pg
            edited += 1

        out.append(
            {
                "id": base.get("id"),
                "question": base.get("question", "").strip(),
                "gold_answer": base.get("gold_answer", "").strip(),
                "evidence_keywords": base.get("evidence_keywords", []),
                "gold_page": base.get("gold_page"),
                "source": "gold_manual",
            }
        )
        kept += 1

    # ensure unique ids and deterministic order
    out = [x for x in out if x.get("question") and x.get("gold_answer")]
    out.sort(key=lambda x: str(x.get("id", "")))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Rows in manual sheet: {len(rows)}")
    print(f"Final kept: {kept}, edited: {edited}, dropped: {dropped}")
    print(f"Final gold set -> {args.out}")


if __name__ == "__main__":
    main()

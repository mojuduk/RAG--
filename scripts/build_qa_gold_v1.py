import argparse
import csv
import json
import re
from pathlib import Path

BANNED_Q_PATTERNS = [
    r"核心内容", r"概述", r"总结", r"分析", r"谈谈", r"简述", r"说明.*意义", r"为什么",
]


def norm_q(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[，。！？；：、（）\(\)\[\]{}<>]", "", text)
    return text


def to_keywords(v):
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


def is_single_fact_like(q: str) -> bool:
    q = (q or "").strip()
    if any(re.search(p, q) for p in BANNED_Q_PATTERNS):
        return False
    # Penalize multi-question patterns
    if q.count("？") + q.count("?") > 1:
        return False
    if any(x in q for x in ["并", "以及", "分别", "同时", "和"]):
        # allow short conjunctions in constrained patterns
        if not re.search(r"区别|差异|对比", q):
            return False
    # Should look interrogative/factual
    factual_cues = ["是什么", "哪个", "哪一", "第", "多少", "是否", "何时", "哪项", "哪种"]
    if not any(c in q for c in factual_cues):
        return False
    return True


def basic_issue_flags(item):
    q = str(item.get("question") or "").strip()
    a = str(item.get("gold_answer") or "").strip()
    kws = to_keywords(item.get("evidence_keywords"))
    page = item.get("gold_page")

    issues = []
    if not q:
        issues.append("empty_question")
    if not a:
        issues.append("empty_answer")
    if len(q) < 8 or len(q) > 40:
        issues.append("q_len_out_of_range")
    if len(a) < 2 or len(a) > 160:
        issues.append("a_len_out_of_range")
    if not is_single_fact_like(q):
        issues.append("not_single_fact")
    if len(kws) < 2 and page in (None, "", "null"):
        issues.append("weak_anchor")
    if any(x in q for x in ["...", "……"]):
        issues.append("ellipsis_question")
    return issues


def score_item(item):
    q = str(item.get("question") or "").strip()
    a = str(item.get("gold_answer") or "").strip()
    kws = to_keywords(item.get("evidence_keywords"))
    page = item.get("gold_page")
    score = 0.0
    score += 1.0 if page not in (None, "", "null") else 0.0
    score += min(len(kws), 5) * 0.2
    score += 0.5 if is_single_fact_like(q) else -1.0
    # Prefer concise answers
    if 4 <= len(a) <= 80:
        score += 0.6
    score += min(len(q), 40) / 100.0
    return score


def load_rows(path: Path):
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise RuntimeError("Input JSON must be a list")
    out = []
    for i, r in enumerate(data, 1):
        out.append(
            {
                "id": str(r.get("id") or f"row_{i:04d}"),
                "question": str(r.get("question") or "").strip(),
                "gold_answer": str(r.get("gold_answer") or "").strip(),
                "evidence_keywords": to_keywords(r.get("evidence_keywords")),
                "gold_page": r.get("gold_page"),
                "source": r.get("source", "unknown"),
            }
        )
    return out


def dedup_by_question(rows):
    best = {}
    for r in rows:
        k = norm_q(r.get("question", ""))
        if not k:
            continue
        if k not in best or score_item(r) > score_item(best[k]):
            best[k] = r
    return list(best.values())


def build_draft(rows, target_min, target_max):
    rows = dedup_by_question(rows)
    kept = []
    dropped = []
    for r in rows:
        issues = basic_issue_flags(r)
        rr = dict(r)
        rr["issues"] = issues
        rr["quality_score"] = round(score_item(r), 4)
        if issues:
            dropped.append(rr)
        else:
            kept.append(rr)

    kept.sort(key=lambda x: x["quality_score"], reverse=True)
    if len(kept) > target_max:
        dropped.extend(kept[target_max:])
        kept = kept[:target_max]

    # if too few, pull highest-quality dropped for manual rescue
    if len(kept) < target_min:
        dropped.sort(key=lambda x: x["quality_score"], reverse=True)
        needed = target_min - len(kept)
        rescue = dropped[:needed]
        for r in rescue:
            r["issues"] = r.get("issues", []) + ["manual_rescue_needed"]
        kept.extend(rescue)
        dropped = dropped[needed:]

    for i, r in enumerate(kept, 1):
        r["id"] = f"gold_v1_{i:04d}"
    return kept, dropped


def write_json(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


def write_manual_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "question",
        "gold_answer",
        "evidence_keywords",
        "gold_page",
        "source",
        "quality_score",
        "issues",
        "manual_decision",
        "manual_fix_question",
        "manual_fix_answer",
        "manual_fix_keywords",
        "manual_fix_page",
        "notes",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    "id": r.get("id", ""),
                    "question": r.get("question", ""),
                    "gold_answer": r.get("gold_answer", ""),
                    "evidence_keywords": " | ".join(r.get("evidence_keywords", [])),
                    "gold_page": r.get("gold_page", ""),
                    "source": r.get("source", ""),
                    "quality_score": r.get("quality_score", ""),
                    "issues": "|".join(r.get("issues", [])),
                    "manual_decision": "",  # keep / drop / edit
                    "manual_fix_question": "",
                    "manual_fix_answer": "",
                    "manual_fix_keywords": "",
                    "manual_fix_page": "",
                    "notes": "",
                }
            )


def main():
    p = argparse.ArgumentParser(description="Build formal QA gold set draft (single-fact, verifiable)")
    p.add_argument("--in", dest="in_path", default="data/qa_candidates_merged.json")
    p.add_argument("--out", dest="out_path", default="data/qa_gold_v1_draft.json")
    p.add_argument("--drop-out", dest="drop_out", default="data/qa_gold_v1_dropped.json")
    p.add_argument("--manual-csv", dest="manual_csv", default="data/qa_gold_v1_manual_review.csv")
    p.add_argument("--target-min", type=int, default=30)
    p.add_argument("--target-max", type=int, default=60)
    args = p.parse_args()

    rows = load_rows(Path(args.in_path))
    kept, dropped = build_draft(rows, args.target_min, args.target_max)

    write_json(Path(args.out_path), kept)
    write_json(Path(args.drop_out), dropped)
    write_manual_csv(Path(args.manual_csv), kept)

    print(f"Input: {len(rows)}")
    print(f"Draft kept: {len(kept)} -> {args.out_path}")
    print(f"Dropped: {len(dropped)} -> {args.drop_out}")
    print(f"Manual review sheet -> {args.manual_csv}")


if __name__ == "__main__":
    main()

import argparse
import json
import re
from pathlib import Path


STOP_WORDS = {
    "什么", "哪些", "如何", "是否", "为什么", "过程", "内容", "问题", "主要", "关于",
    "进行", "以及", "根据", "可以", "需要", "通过", "这个", "那个", "其中", "相关",
    "工作", "方法", "系统", "结果", "分析", "说明", "提出", "使用", "一个", "一种",
}


def tokenize(text):
    text = str(text or "")
    text = re.sub(r"[，。！？；：、（）()\[\]{}<>\n\r\t]", " ", text)
    return re.findall(r"[\u4e00-\u9fffA-Za-z0-9\-]{2,24}", text)


def build_keywords(question, answer, max_k=5):
    tokens = tokenize(str(answer) + " " + str(question))
    out = []
    seen = set()
    for w in tokens:
        wl = w.lower()
        if wl in seen:
            continue
        if w in STOP_WORDS:
            continue
        if w.isdigit():
            continue
        if len(w) < 2:
            continue
        seen.add(wl)
        out.append(w)
        if len(out) >= max_k:
            break
    return out


def fill_keywords(rows, max_k=5, force=False):
    changed = 0
    for row in rows:
        kws = row.get("evidence_keywords")
        need_fill = force or (not isinstance(kws, list) or len(kws) == 0)
        if not need_fill:
            continue
        new_kws = build_keywords(row.get("question", ""), row.get("gold_answer", ""), max_k=max_k)
        row["evidence_keywords"] = new_kws
        changed += 1
    return rows, changed


def main():
    parser = argparse.ArgumentParser(description="Auto-fill missing evidence_keywords in QA set.")
    parser.add_argument("--in", dest="in_path", required=True, help="Input QA JSON path")
    parser.add_argument("--out", dest="out_path", required=True, help="Output QA JSON path")
    parser.add_argument("--max-k", type=int, default=5, help="Max keywords per row")
    parser.add_argument("--force", action="store_true", help="Overwrite existing keywords")
    args = parser.parse_args()

    in_path = Path(args.in_path)
    rows = json.loads(in_path.read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list):
        raise RuntimeError("Input JSON must be a list.")

    out_rows, changed = fill_keywords(rows, max_k=args.max_k, force=args.force)

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Processed {len(out_rows)} rows, filled {changed} rows -> {out_path}")


if __name__ == "__main__":
    main()

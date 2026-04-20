import json
import re
from pathlib import Path


def _read_chunks(path):
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _clean_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_garbled(text):
    text = _clean_text(text)
    if not text:
        return True
    sample = text[:240]
    bad_markers = ["�", "锟", "鏂", "鍙", "绗", "娴", "浠", "璇", "鐨", "鍥"]
    if sum(sample.count(m) for m in bad_markers) >= 4:
        return True
    valid = re.findall(r"[\u4e00-\u9fffA-Za-z0-9，。！？：；、（）\-\s]", sample)
    return (len(valid) / max(len(sample), 1)) < 0.68


def _looks_like_low_quality_table_summary(text):
    t = _clean_text(text).lower()
    if not t:
        return True
    if re.search(r"\\begin\{tabular\}|\\end\{tabular\}|unnamed:\s*\d+", t):
        return True
    return len(t) < 24


def _single_char_ratio(text):
    toks = text.split()
    if not toks:
        return 0.0
    one = sum(1 for t in toks if len(t) == 1)
    return one / len(toks)


def _looks_like_low_value_text_chunk(text):
    t = _clean_text(text)
    if len(t) < 28:
        return True

    # Strong table-header patterns
    if re.search(r"序\s*区\s*设\s*功能", t):
        return True
    if re.search(r"投料数据\s*出料数据\s*过程参数", t):
        return True

    table_header_words = ["区块", "工艺单元", "设备", "投料", "出料", "工艺要求", "环境要求", "开始/结束指令"]
    hit = sum(1 for w in table_header_words if w in t)
    if hit >= 4:
        return True

    if len(t.split()) >= 24 and _single_char_ratio(t) >= 0.48:
        return True

    if re.search(r"(?:\b[A-Za-z]\b\s+){4,}", t):
        return True

    return False


def _take_sentence(text, max_len=120):
    text = _clean_text(text)
    if not text:
        return ""
    parts = re.split(r"[。！？!?；;]", text)
    for p in parts:
        p = p.strip()
        if len(p) >= 12:
            return p[:max_len]
    return text[:max_len]


def _extract_title_head(text):
    text = _clean_text(text)
    m = re.search(r"(\d+(?:\.\d+){0,3}\s*[^。；;:：\n]{4,36})", text)
    return m.group(1).strip() if m else ""


def _is_noisy_title(title):
    if not title:
        return True
    if len(title) > 34:
        return True
    if _looks_like_low_value_text_chunk(title):
        return True
    if _single_char_ratio(title) > 0.45:
        return True
    return False


def _extract_terms(text, limit=5):
    text = _clean_text(text)
    candidates = re.findall(r"[\u4e00-\u9fffA-Za-z0-9\-]{2,24}", text)
    stop = {
        "数据", "过程", "系统", "信息", "要求", "说明", "分析", "内容", "表格", "生产",
        "其中", "包括", "进行", "相关", "可以", "以及", "如下", "主要", "描述", "字段",
        "摘要", "参数", "章节", "部分", "核心", "结论", "区块", "工艺单元", "设备", "投料", "出料",
    }
    out, seen = [], set()
    for t in candidates:
        k = t.lower()
        if t in stop or k in seen or t.isdigit() or len(t) < 2:
            continue
        seen.add(k)
        out.append(t)
        if len(out) >= limit:
            break
    return out


def _build_from_text_chunk(chunk, idx):
    content = _clean_text(chunk.get("content", ""))
    if len(content) < 24 or _is_garbled(content) or _looks_like_low_value_text_chunk(content):
        return None

    meta = chunk.get("metadata", {}) or {}
    page = meta.get("page")
    answer = _take_sentence(content, max_len=120)
    if _looks_like_low_value_text_chunk(answer):
        return None

    title = _extract_title_head(content)
    keywords = _extract_terms(answer, limit=5)
    if not keywords:
        return None

    if title and not _is_noisy_title(title):
        question = f"{title}主要讲了什么？"
    else:
        question = f"关于{keywords[0]}的主要结论是什么？"

    return {
        "id": f"auto_text_{idx:04d}",
        "question": question,
        "gold_answer": answer,
        "evidence_keywords": keywords,
        "gold_page": page if isinstance(page, int) else None,
        "source_chunk_type": meta.get("chunk_type", "text"),
    }


def _build_from_table_chunk(chunk, idx):
    meta = chunk.get("metadata", {}) or {}
    caption = (meta.get("caption") or meta.get("source_table") or "").strip()
    if not caption:
        return None

    content = _clean_text(chunk.get("content", ""))
    if _is_garbled(content) or _looks_like_low_quality_table_summary(content):
        return None

    answer = _take_sentence(content, max_len=120)
    if not answer:
        return None

    answer = re.sub(r"\\\\?[A-Za-z]+\{[^}]*\}", "", answer)
    answer = re.sub(r"\\\\?[A-Za-z]+", "", answer)
    answer = _clean_text(answer)
    if len(answer) < 12:
        return None

    keywords = _extract_terms(caption + " " + answer, limit=5)
    if not keywords:
        return None

    page = meta.get("page")
    return {
        "id": f"auto_table_{idx:04d}",
        "question": f"{caption}主要描述了什么？",
        "gold_answer": answer,
        "evidence_keywords": keywords,
        "gold_page": page if isinstance(page, int) else None,
        "source_chunk_type": meta.get("chunk_type", "table_summary"),
    }


def build_qa_candidates(chunks_path, out_path, max_items=80, include_table=True, include_text=True):
    chunks = _read_chunks(chunks_path)
    candidates = []

    for i, chunk in enumerate(chunks, 1):
        meta = chunk.get("metadata", {}) or {}
        ctype = meta.get("chunk_type", "text")

        item = None
        if include_table and ctype == "table_summary":
            item = _build_from_table_chunk(chunk, i)
        elif include_text and ctype == "text":
            item = _build_from_text_chunk(chunk, i)

        if item:
            candidates.append(item)
        if len(candidates) >= max_items:
            break

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_path).open("w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    return candidates


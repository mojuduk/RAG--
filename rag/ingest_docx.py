import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph


def iter_block_items(doc):
    for child in doc.element.body.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, doc)
        elif isinstance(child, CT_Tbl):
            yield Table(child, doc)


def normalize_text(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_table_rows(table):
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            cell_text = cell.text.replace("\n", " ").strip()
            cells.append(cell_text)
        rows.append(cells)
    return rows


def is_header_footer_noise(text):
    lowered = text.lower()
    if re.fullmatch(r"\d+", text):
        return True
    if re.fullmatch(r"第\s*\d+\s*页", text):
        return True
    if lowered.startswith("page ") and lowered[5:].strip().isdigit():
        return True
    if "页眉" in text or "页脚" in text:
        return True
    return False


def should_skip_paragraph(text):
    if not text:
        return True
    if is_header_footer_noise(text):
        return True
    return False


def build_table_context(section_title, table_index):
    if section_title:
        return f"{section_title} 表{table_index}"
    return f"表{table_index}"


def normalize_header_name(name):
    name = normalize_text(name)
    if not name:
        return ""
    alias = {
        "工艺参数": "过程参数",
        "关键工艺参数": "过程参数",
        "过程数据": "过程参数",
        "投料": "投料数据",
        "出料": "出料数据",
        "开始/结束指令": "指令",
        "起始指令": "指令",
        "操作指令": "指令",
    }
    return alias.get(name, name)


def fill_merged_cells(rows):
    if not rows:
        return rows
    max_cols = max(len(r) for r in rows)
    filled = []
    last_seen = [""] * max_cols
    for row in rows:
        padded = row + [""] * (max_cols - len(row))
        new_row = []
        for idx, value in enumerate(padded):
            value = normalize_text(value)
            if value:
                last_seen[idx] = value
                new_row.append(value)
            else:
                new_row.append(last_seen[idx])
        filled.append(new_row)
    return filled


def build_table_row_content(context, header, row):
    lines = [f"# 上下文：{context}"]
    for key, value in zip(header, row):
        if not key or not value:
            continue
        lines.append(f"- {key}: {value}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def build_table_summary(context, header, rows):
    if not rows:
        return ""
    fields = [h for h in header if h]
    header_map = {name: idx for idx, name in enumerate(header) if name}
    stats_parts = []

    def top_values(keys, label, limit=3):
        for key in keys:
            if key in header_map:
                idx = header_map[key]
                counts = {}
                for row in rows:
                    if idx >= len(row):
                        continue
                    value = normalize_text(row[idx])
                    if not value:
                        continue
                    counts[value] = counts.get(value, 0) + 1
                if counts:
                    top = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:limit]
                    items = "，".join([f"{v}({c})" for v, c in top])
                    stats_parts.append(f"{label}Top{limit}:{items}")
                break

    top_values(("单元", "工序", "工艺单元", "区块"), "工序")
    top_values(("设备",), "设备")

    summary = (
        f"表格摘要：{context}，共{len(rows)}行，字段包括："
        + "、".join(fields)
        + "。"
    )
    if stats_parts:
        summary += " 统计：" + "；".join(stats_parts) + "。"
    return summary


def extract_metadata(header, row, source_table):
    header_map = {name: idx for idx, name in enumerate(header) if name}
    process_name = ""
    equipment = ""
    for key in ("单元", "工序", "工艺单元", "区块"):
        if key in header_map:
            process_name = row[header_map[key]]
            break
    if "设备" in header_map:
        equipment = row[header_map["设备"]]
    return {
        "process_name": process_name,
        "equipment": equipment,
        "source_table": source_table,
    }


def ingest_docx(doc_path):
    doc = Document(str(doc_path))
    chunks = []
    section_title = ""
    table_index = 0
    short_buffer = ""

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            raw_text = normalize_text(block.text)
            if should_skip_paragraph(raw_text):
                continue

            if block.style and block.style.name:
                style_name = block.style.name.lower()
                if style_name.startswith("heading") or "标题" in block.style.name:
                    if raw_text:
                        section_title = raw_text
                    continue

            if len(raw_text) < 10:
                if short_buffer:
                    short_buffer = f"{short_buffer} {raw_text}"
                else:
                    short_buffer = raw_text
                continue

            if short_buffer:
                raw_text = f"{short_buffer} {raw_text}"
                short_buffer = ""

            chunks.append({"content": raw_text, "metadata": {}})
        else:
            if short_buffer:
                short_buffer = ""

            rows = extract_table_rows(block)
            if not rows:
                continue
            rows = fill_merged_cells(rows)
            header = [normalize_header_name(h) for h in rows[0]]
            header = [h if h else f"列{idx + 1}" for idx, h in enumerate(header)]
            table_index += 1
            context = build_table_context(section_title, table_index)

            summary = build_table_summary(context, header, rows[1:])
            if summary:
                chunks.append(
                    {
                        "content": summary,
                        "metadata": {"source_table": context, "chunk_type": "table_summary"},
                    }
                )

            for row in rows[1:]:
                row = [normalize_text(v) for v in row]
                content = build_table_row_content(context, header, row)
                if not content:
                    continue
                metadata = extract_metadata(header, row, context)
                chunks.append({"content": content, "metadata": metadata})

    return chunks


def ingest_docs(doc_paths, output_path, cleaned_path):
    all_chunks = []
    for doc_path in doc_paths:
        doc_path = Path(doc_path)
        if not doc_path.exists():
            continue
        all_chunks.extend(ingest_docx(doc_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    with cleaned_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk["content"] + "\n")

    return all_chunks

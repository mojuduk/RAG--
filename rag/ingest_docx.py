import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table
from docx.text.paragraph import Paragraph

_FIGURE_CAPTION_RE = re.compile(
    r"^\s*(图|表|Figure|Fig\.?)\s*([0-9一二三四五六七八九十百千IVXivx\-\._]*)\s*[:：.\-]?\s*(.*)$",
    re.IGNORECASE,
)
_PADDLE_OCR = None


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


def _extract_figure_no_and_caption(text):
    text = normalize_text(text)
    if not text:
        return "", ""
    m = _FIGURE_CAPTION_RE.match(text)
    if not m:
        return "", ""
    prefix = m.group(1)
    number = normalize_text(m.group(2) or "")
    tail = normalize_text(m.group(3) or "")
    figure_no = f"{prefix}{number}".strip() if number else prefix
    caption = text if tail else figure_no
    return figure_no, caption


def _extract_key_entities(text, limit=8):
    text = normalize_text(text)
    if not text:
        return []
    cands = re.findall(r"[\u4e00-\u9fffA-Za-z0-9][\u4e00-\u9fffA-Za-z0-9\-/]{1,24}", text)
    stop = {
        "图片",
        "图",
        "表",
        "第",
        "页",
        "文件",
        "标题",
        "上文",
        "的",
        "和",
        "以及",
        "进行",
        "包括",
    }
    out = []
    seen = set()
    for c in cands:
        c = c.strip("-_/")
        if len(c) < 2:
            continue
        if c in stop:
            continue
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def _get_paddle_ocr():
    global _PADDLE_OCR
    if _PADDLE_OCR is not None:
        return _PADDLE_OCR
    try:
        from paddleocr import PaddleOCR
    except Exception:
        _PADDLE_OCR = False
        return _PADDLE_OCR
    _PADDLE_OCR = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    return _PADDLE_OCR


def _ocr_image_text(image_path):
    ocr = _get_paddle_ocr()
    if not ocr:
        return ""
    try:
        results = ocr.ocr(str(image_path), cls=True)
    except Exception:
        return ""
    lines = []
    for line in results or []:
        if not line:
            continue
        txt = normalize_text(line[1][0] if len(line) > 1 and line[1] else "")
        if txt:
            lines.append(txt)
    return normalize_text(" ".join(lines))[:300]


def extract_table_rows(table):
    rows = []
    for row in table.rows:
        cells = []
        for cell in row.cells:
            cell_text = cell.text.replace("\n", " ").strip()
            cells.append(cell_text)
        rows.append(cells)
    return rows


def _iter_paragraph_images(paragraph, doc, doc_path, image_dir, image_index_start):
    if image_dir is None:
        return [], image_index_start
    image_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    image_index = image_index_start
    # Find embedded images in runs
    for run in paragraph.runs:
        blips = run._element.xpath(".//a:blip")
        for blip in blips:
            r_id = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
            if not r_id:
                continue
            part = doc.part.related_parts.get(r_id)
            if not part:
                continue
            image_index += 1
            ext = Path(part.partname).suffix or ".bin"
            img_path = image_dir / f"{doc_path.stem}_img_{image_index:03d}{ext}"
            with open(img_path, "wb") as f:
                f.write(part.blob)
            chunks.append((img_path, image_index))
    return chunks, image_index


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


def extract_docx_images(doc, doc_path, image_dir):
    if image_dir is None:
        return []
    image_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    image_index = 0
    for rel in doc.part.rels.values():
        if "image" not in rel.reltype:
            continue
        image_index += 1
        part = rel.target_part
        ext = Path(part.partname).suffix or ".bin"
        img_path = image_dir / f"{doc_path.stem}_img_{image_index:03d}{ext}"
        with open(img_path, "wb") as f:
            f.write(part.blob)
        chunks.append(
            {
                "content": f"图片：{doc_path.name} 第{image_index}张。文件: {img_path}",
                "metadata": {
                    "source": "docx",
                    "chunk_type": "image",
                    "image_path": str(img_path),
                    "doc": doc_path.name,
                },
            }
        )
    return chunks


def ingest_docx(doc_path, image_dir=None):
    doc = Document(str(doc_path))
    chunks = []
    section_title = ""
    table_index = 0
    short_buffer = ""
    last_block_kind = ""
    last_block_text = ""
    image_index = 0

    for block in iter_block_items(doc):
        if isinstance(block, Paragraph):
            raw_text = normalize_text(block.text)
            figure_no, figure_caption = _extract_figure_no_and_caption(raw_text)
            above_adjacent = last_block_text if last_block_kind == "paragraph" else ""
            # Extract images even if paragraph text is empty/ignored.
            images, image_index = _iter_paragraph_images(block, doc, doc_path, image_dir, image_index)
            for img_path, idx in images:
                content = f"图片：{doc_path.name} 第{idx}张"
                if figure_caption:
                    content += f"；图题：{figure_caption}"
                elif section_title:
                    content += f"；标题：{section_title}"
                if above_adjacent:
                    content += f"；上文：{above_adjacent}"
                ocr_text = _ocr_image_text(img_path)
                if ocr_text:
                    content += f"；图中文字：{ocr_text}"
                content += f"。文件: {img_path}"
                entity_src = " ".join(
                    [
                        section_title or "",
                        figure_caption or "",
                        above_adjacent or "",
                        ocr_text or "",
                    ]
                )
                chunks.append(
                    {
                        "content": content,
                        "metadata": {
                            "source": "docx",
                            "chunk_type": "image",
                            "image_path": str(img_path),
                            "doc": doc_path.name,
                            "section_title": section_title,
                            "title": figure_caption or section_title,
                            "figure_no": figure_no,
                            "figure_caption": figure_caption,
                            "above_text": above_adjacent,
                            "key_entities": _extract_key_entities(entity_src),
                            "image_ocr_text": ocr_text,
                        },
                    }
                )

            if should_skip_paragraph(raw_text):
                last_block_kind = "paragraph" if raw_text else "empty"
                last_block_text = raw_text if raw_text else ""
                continue

            if block.style and block.style.name:
                style_name = block.style.name.lower()
                if style_name.startswith("heading") or "标题" in block.style.name:
                    if raw_text:
                        section_title = raw_text
                    last_block_kind = "heading"
                    last_block_text = raw_text
                    continue

            if len(raw_text) < 10:
                if short_buffer:
                    short_buffer = f"{short_buffer} {raw_text}"
                else:
                    short_buffer = raw_text
                last_block_kind = "paragraph"
                last_block_text = raw_text
                continue

            if short_buffer:
                raw_text = f"{short_buffer} {raw_text}"
                short_buffer = ""

            chunks.append({"content": raw_text, "metadata": {}})
            last_block_kind = "paragraph"
            last_block_text = raw_text
        else:
            if short_buffer:
                short_buffer = ""
            last_block_kind = "table"
            last_block_text = ""

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


def ingest_docs(doc_paths, output_path, cleaned_path, image_dir=None):
    all_chunks = []
    for doc_path in doc_paths:
        doc_path = Path(doc_path)
        if not doc_path.exists():
            continue
        all_chunks.extend(ingest_docx(doc_path, image_dir=image_dir))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    cleaned_path.parent.mkdir(parents=True, exist_ok=True)
    with cleaned_path.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(chunk["content"] + "\n")

    return all_chunks

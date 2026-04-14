import json
import os
import re
import subprocess
from pathlib import Path


def normalize_text(text):
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_mojibake_text(text):
    text = normalize_text(text)
    if not text:
        return text
    text = text.replace("�", "").replace("?", "")
    text = re.sub(r"(?<=[\u4e00-\u9fff])n(?=[\u4e00-\u9fff])", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def split_paragraphs(text):
    if not text:
        return []
    parts = re.split(r"\n{2,}", text)
    cleaned = []
    short_buffer = ""
    for part in parts:
        part = normalize_text(part.replace("\n", " "))
        if not part:
            continue
        if len(part) < 10:
            short_buffer = f"{short_buffer} {part}".strip() if short_buffer else part
            continue
        if short_buffer:
            part = f"{short_buffer} {part}".strip()
            short_buffer = ""
        cleaned.append(part)
    return cleaned


def _run_paddle_ocr_on_image(image_path):
    try:
        from paddleocr import PaddleOCR
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("paddleocr not installed; cannot run OCR.") from exc

    ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    results = ocr.ocr(str(image_path), cls=True)
    lines = []
    for line in results or []:
        if not line:
            continue
        text = line[1][0]
        if text:
            lines.append(text)
    return "\n".join(lines)


def _is_diagram_line(line):
    if not line:
        return True
    tokens = re.split(r"\s+", line.strip())
    if not tokens:
        return True
    single_ratio = sum(1 for t in tokens if len(t) == 1) / len(tokens)
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    # Heuristic: many isolated single chars, low avg token length.
    if single_ratio >= 0.6 and avg_len <= 1.2:
        return True
    # Heuristic: spaced single CJK characters
    if re.search(r"(?:[\u4e00-\u9fff]\s+){6,}", line):
        return True
    return False


def _filter_ocr_text(ocr_text):
    lines = [normalize_text(l) for l in ocr_text.splitlines()]
    lines = [l for l in lines if l]
    cleaned = []
    skip_diagram = False
    for line in lines:
        is_caption = re.match(r"^(图|表)\s*\d+", line)
        if is_caption:
            cleaned.append(line)
            skip_diagram = True
            continue
        if skip_diagram:
            if _is_diagram_line(line):
                continue
            # End skipping when a normal sentence appears
            skip_diagram = False
        if _is_diagram_line(line):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def extract_pdf_text_chunks(pdf_path, ocr_on_garbled=False, ocr_image_dir=None, ocr_dpi=200):
    try:
        import pdfplumber
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pdfplumber not installed; cannot extract PDF text.") from exc

    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            raw_text = page.extract_text() or ""
            text = clean_mojibake_text(raw_text)
            # Heuristic: detect garbled text by very short tokens and low punctuation/number signal.
            tokens = re.split(r"\s+", raw_text.strip()) if raw_text else []
            avg_len = (sum(len(t) for t in tokens) / len(tokens)) if tokens else 0
            short_token_ratio = (
                sum(1 for t in tokens if len(t) <= 1) / len(tokens)
            ) if tokens else 0
            qmark_ratio = (raw_text.count("?") / max(1, len(raw_text)))
            garbled = (
                ("�" in raw_text)
                or qmark_ratio >= 0.003
                or (raw_text.count("?") >= 5)
                or (len(tokens) > 30 and avg_len <= 1.3 and short_token_ratio >= 0.6)
            )
            if garbled:
                if not ocr_on_garbled:
                    continue
                # OCR fallback
                ocr_text = ""
                if ocr_image_dir is not None:
                    ocr_image_dir.mkdir(parents=True, exist_ok=True)
                try:
                    # Render this page to image for OCR
                    import pdfplumber
                    page_image = page.to_image(resolution=ocr_dpi)
                    image_path = None
                    if ocr_image_dir is not None:
                        image_path = ocr_image_dir / f"{Path(pdf_path).stem}_p{page_num:03d}.png"
                        page_image.save(str(image_path), format="PNG")
                    else:
                        image_path = page_image.original
                    ocr_text = _run_paddle_ocr_on_image(image_path)
                except Exception:
                    ocr_text = ""
                if not ocr_text:
                    continue
                text = clean_mojibake_text(_filter_ocr_text(ocr_text))
            for para in split_paragraphs(text):
                chunks.append({
                    "content": para,
                    "metadata": {"source": "pdf", "page": page_num, "chunk_type": "text"},
                })
    return chunks


def extract_pdf_image_chunks(pdf_path, image_dir=None, image_dpi=200):
    try:
        import pdfplumber
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pdfplumber not installed; cannot extract PDF images.") from exc

    chunks = []
    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            images = page.images or []
            words = page.extract_words() or []
            # Build simple line groups by y-position
            lines = {}
            for w in words:
                top = round(w.get("top", 0), 1)
                lines.setdefault(top, []).append(w.get("text", ""))
            sorted_lines = sorted((t, " ".join(v).strip()) for t, v in lines.items() if " ".join(v).strip())
            page_title = sorted_lines[0][1] if sorted_lines else ""
            for idx, img in enumerate(images, 1):
                x0 = img.get("x0")
                x1 = img.get("x1")
                top = img.get("top")
                bottom = img.get("bottom")
                width = None
                height = None
                if x0 is not None and x1 is not None:
                    width = round(x1 - x0, 2)
                if top is not None and bottom is not None:
                    height = round(bottom - top, 2)

                image_path = ""
                if image_dir is not None and None not in (x0, x1, top, bottom):
                    bbox = (x0, top, x1, bottom)
                    try:
                        cropped = page.within_bbox(bbox)
                        image = cropped.to_image(resolution=image_dpi)
                        image_path = str(
                            image_dir
                            / f"{Path(pdf_path).stem}_p{page_num:03d}_img{idx:02d}.png"
                        )
                        image.save(image_path, format="PNG")
                    except Exception:
                        image_path = ""

                above_text = ""
                if top is not None and sorted_lines:
                    candidates = [line for t, line in sorted_lines if t < top and line]
                    if candidates:
                        above_text = candidates[-1]

                content = f"PDF第{page_num}页图片{idx}"
                if page_title:
                    content += f"；标题：{page_title}"
                if above_text:
                    content += f"；上文：{above_text}"
                if width is not None and height is not None:
                    content += f"，宽高约{width}x{height}"
                if image_path:
                    content += f"。文件: {image_path}"

                chunks.append(
                    {
                        "content": content,
                        "metadata": {
                            "source": "pdf",
                            "chunk_type": "image",
                            "page": page_num,
                            "image_path": image_path,
                            "title": page_title,
                            "above_text": above_text,
                        },
                    }
                )
    return chunks


def find_tables_in_mmd(mmd_text):
    tables = []
    pattern = re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.DOTALL)
    for match in pattern.finditer(mmd_text):
        block = match.group(0)
        caption_match = re.search(r"\\caption\{(.*?)\}", block, re.DOTALL)
        caption = normalize_text(caption_match.group(1)) if caption_match else ""
        if not caption:
            tail = mmd_text[match.end():match.end() + 300]
            tail_caption = re.search(r"\\caption\{(.*?)\}", tail, re.DOTALL)
            if tail_caption:
                caption = normalize_text(tail_caption.group(1))
        tables.append({"latex": block, "caption": caption})
    return tables


def extract_header_from_latex(latex):
    tabular = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", latex, re.DOTALL)
    if not tabular:
        return []
    content = tabular.group(0)
    rows = re.split(r"\\\\", content)
    for row in rows:
        row = re.sub(r"\\hline", "", row)
        row = normalize_text(row)
        if not row:
            continue
        cols = [normalize_text(c) for c in row.split("&")]
        cols = [c for c in cols if c]
        if cols:
            return cols
    return []


def summarize_table(caption, latex):
    headers = extract_header_from_latex(latex)
    name = caption or "未命名表格"
    if headers:
        return f"表格摘要：{name}，字段包括：" + "、".join(headers) + "。"
    return f"表格摘要：{name}。"


def rows_to_latex(rows):
    if not rows:
        return ""
    col_count = max(len(r) for r in rows)
    header = rows[0]
    body = rows[1:]
    header = header + [""] * (col_count - len(header))
    lines = [
        r"\begin{table}",
        r"\begin{tabular}{" + " | ".join(["l"] * col_count) + "}",
        r"\hline",
        " & ".join(header) + r" \\",
        r"\hline",
    ]
    for row in body:
        row = row + [""] * (col_count - len(row))
        lines.append(" & ".join(row) + r" \\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def _normalize_cell(cell):
    cell = "" if cell is None else str(cell)
    cell = cell.replace("\r", " ")
    cell = clean_mojibake_text(cell)
    return cell


def _is_empty_cell(cell):
    return not cell or cell.strip().lower() in {"nan", "none"}


def _clean_table_rows(rows):
    if not rows:
        return []
    col_count = max(len(r) for r in rows)
    cleaned = []
    key_cols = min(3, col_count)
    header = [(_normalize_cell(c) if c is not None else "") for c in rows[0]]
    header += [""] * (col_count - len(header))
    has_material_cols = col_count >= 6 and ("投料" in header[4] or "投料" in "".join(header)) and (
        "出料" in header[5] or "出料" in "".join(header)
    )
    for row in rows:
        row = row + [""] * (col_count - len(row))
        row = [_normalize_cell(c) for c in row]
        if all(_is_empty_cell(c) for c in row):
            continue
        row = ["" if _is_empty_cell(c) else c for c in row]

        if cleaned:
            non_empty = [i for i, c in enumerate(row) if not _is_empty_cell(c)]
            # Heuristic: merge only when key columns are empty and content appears in later columns.
            has_key_content = any(i < key_cols for i in non_empty)
            if non_empty and (not has_key_content) and all(i >= key_cols for i in non_empty):
                prev = cleaned[-1]
                merged = prev[:]
                for i in non_empty:
                    merged[i] = (merged[i] + " " + row[i]).strip()
                cleaned[-1] = merged
                continue
            # Heuristic: row content lands in first 1-2 columns but should be material/output columns.
            if (
                has_material_cols
                and set(non_empty).issubset({0, 1})
                and len(non_empty) >= 1
                and cleaned
            ):
                prev = cleaned[-1]
                if row[0]:
                    prev[4] = (prev[4] + " " + row[0]).strip()
                if row[1]:
                    prev[5] = (prev[5] + " " + row[1]).strip()
                cleaned[-1] = prev
                continue

        cleaned.append(row)
    return cleaned


def extract_tables_pdfplumber(pdf_path):
    try:
        import pdfplumber
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pdfplumber not installed; cannot extract PDF tables.") from exc

    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            page_tables = page.extract_tables() or []
            for idx, table in enumerate(page_tables, 1):
                rows = []
                for row in table:
                    if not row:
                        continue
                    rows.append([_normalize_cell(c or "") for c in row])
                if not rows:
                    continue
                rows = _clean_table_rows(rows)
                if not rows:
                    continue
                caption = clean_mojibake_text(f"PDF第{page_num}页表{idx}")
                tables.append({"rows": rows, "caption": caption, "page": page_num})
    return tables


def extract_tables_tabula(pdf_path, tabula_guess, tabula_lattice, tabula_stream, tabula_jvm_opts):
    try:
        import tabula
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("tabula-py not installed; cannot extract PDF tables.") from exc

    tables = []
    # Per-page extraction for page metadata and better stability.
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        page_total = len(pdf.pages)

    if tabula_jvm_opts is None:
        tabula_jvm_opts = ["-Dfile.encoding=UTF-8", "-Djava.awt.headless=true"]
    elif isinstance(tabula_jvm_opts, str):
        raw_opts = [opt for opt in tabula_jvm_opts.split(";") if opt]
        tabula_jvm_opts = [opt.strip().strip('"').strip("'") for opt in raw_opts]

    for page_num in range(1, page_total + 1):
        try:
            dfs = tabula.read_pdf(
                str(pdf_path),
                pages=page_num,
                multiple_tables=True,
                guess=tabula_guess,
                lattice=tabula_lattice,
                stream=tabula_stream,
                java_options=tabula_jvm_opts,
            )
        except Exception:
            dfs = tabula.read_pdf(
                str(pdf_path),
                pages=page_num,
                multiple_tables=True,
                java_options=tabula_jvm_opts,
            )

        for idx, df in enumerate(dfs or [], 1):
            if df is None or df.empty:
                continue
            rows = [list(df.columns)]
            for _, row in df.iterrows():
                rows.append([_normalize_cell(v) for v in row.tolist()])
            if not rows:
                continue
            rows = _clean_table_rows(rows)
            if not rows:
                continue
            caption = clean_mojibake_text(f"PDF第{page_num}页表{idx}")
            tables.append({"rows": rows, "caption": caption, "page": page_num})
    return tables


def pdf_to_images(pdf_path, image_dir, dpi=200):
    try:
        import pypdfium2 as pdfium
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pypdfium2 not installed; cannot render PDF pages.") from exc

    image_dir.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    scale = dpi / 72
    for index in range(len(doc)):
        page = doc.get_page(index)
        try:
            render = page.render(scale=scale)
            image = render.to_pil()
        except Exception:
            try:
                image = page.render_topil(scale=scale)
            except Exception as exc:
                page.close()
                raise RuntimeError("Failed to render PDF page to image.") from exc
        image_path = image_dir / f"page_{index + 1:04d}.png"
        image.save(image_path)
        page.close()
    doc.close()
    return image_dir


def _build_nougat_cmd(nougat_cmd, input_path, out_dir):
    cmd = list(nougat_cmd)
    if "{input}" in cmd:
        cmd = [str(input_path) if part == "{input}" else part for part in cmd]
    else:
        cmd.append(str(input_path))
    if "{out}" in cmd:
        cmd = [str(out_dir) if part == "{out}" else part for part in cmd]
    else:
        cmd += ["--out", str(out_dir)]
    return cmd


def generate_mmd_with_nougat(pdf_path, mmd_path, nougat_cmd, input_path=None):
    out_dir = mmd_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_path or pdf_path
    cmd = _build_nougat_cmd(nougat_cmd, input_path, out_dir)
    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_NO_TF", "1")
    env.setdefault("USE_TF", "0")
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    subprocess.run(cmd, check=True, env=env)
    if mmd_path.exists():
        return
    candidates = list(out_dir.glob("*.mmd")) + list(out_dir.glob("*.md"))
    if candidates:
        newest = max(candidates, key=lambda p: p.stat().st_mtime)
        if newest.suffix.lower() != ".mmd":
            newest = newest.with_suffix(".mmd")
        newest.replace(mmd_path)
        return
    raise RuntimeError(
        "Nougat finished but .mmd/.md not found. Check nougat logs and output path."
    )


def _mmd_has_tables(mmd_text):
    return "\\begin{table}" in mmd_text


def _mmd_is_mostly_missing(mmd_text):
    missing = len(re.findall(r"MISSING_PAGE_(EMPTY|FAIL)", mmd_text))
    return missing >= 5 and missing > len(mmd_text) / 50


def ingest_pdf(
    pdf_path,
    mmd_path,
    nougat_cmd=None,
    image_dir=None,
    image_dpi=200,
    table_tool="pdfplumber",
    tabula_guess=True,
    tabula_lattice=True,
    tabula_stream=True,
    tabula_jvm_opts=None,
    image_extract_dir=None,
    ocr_on_garbled=False,
    ocr_image_dir=None,
    ocr_dpi=200,
):
    chunks = []
    chunks.extend(
        extract_pdf_text_chunks(
            pdf_path,
            ocr_on_garbled=ocr_on_garbled,
            ocr_image_dir=ocr_image_dir,
            ocr_dpi=ocr_dpi,
        )
    )
    chunks.extend(extract_pdf_image_chunks(pdf_path, image_dir=image_extract_dir, image_dpi=image_dpi))

    tables = []
    if table_tool == "nougat":
        if not mmd_path.exists():
            if not nougat_cmd:
                raise RuntimeError(".mmd not found and nougat_cmd not provided.")
            input_path = pdf_path
            if image_dir is not None:
                input_path = pdf_to_images(pdf_path, image_dir, dpi=image_dpi)
            generate_mmd_with_nougat(pdf_path, mmd_path, nougat_cmd, input_path=input_path)

        mmd_text = mmd_path.read_text(encoding="utf-8", errors="ignore")
        if image_dir is not None and (not _mmd_has_tables(mmd_text)) and _mmd_is_mostly_missing(mmd_text):
            for dpi in (300, 400):
                input_path = pdf_to_images(pdf_path, image_dir, dpi=dpi)
                generate_mmd_with_nougat(pdf_path, mmd_path, nougat_cmd, input_path=input_path)
                mmd_text = mmd_path.read_text(encoding="utf-8", errors="ignore")
                if _mmd_has_tables(mmd_text):
                    break
        tables = find_tables_in_mmd(mmd_text)
    elif table_tool == "tabula":
        tables = extract_tables_tabula(
            pdf_path,
            tabula_guess=tabula_guess,
            tabula_lattice=tabula_lattice,
            tabula_stream=tabula_stream,
            tabula_jvm_opts=tabula_jvm_opts,
        )
    else:
        tables = extract_tables_pdfplumber(pdf_path)

    parents = []
    for idx, table in enumerate(tables, 1):
        parent_id = f"pdf_table_{idx:04d}"
        if "latex" in table:
            latex = table["latex"]
            caption = table.get("caption", "")
        else:
            latex = rows_to_latex(table["rows"])
            caption = table.get("caption", "")
        summary = summarize_table(caption, latex)
        chunks.append({
            "content": summary,
            "metadata": {
                "source": "pdf",
                "chunk_type": "table_summary",
                "parent_id": parent_id,
                "caption": caption,
                "page": table.get("page"),
            },
        })
        parents.append({
            "id": parent_id,
            "caption": caption,
            "latex": latex,
            "source_pdf": str(pdf_path),
            "page": table.get("page"),
        })

    return chunks, parents


def ingest_pdfs(
    pdf_paths,
    mmd_dir,
    parents_path,
    nougat_cmd=None,
    image_root=None,
    image_dpi=200,
    table_tool="pdfplumber",
    tabula_guess=True,
    tabula_lattice=True,
    tabula_stream=True,
    tabula_jvm_opts=None,
    image_extract_dir=None,
    ocr_on_garbled=False,
    ocr_image_dir=None,
    ocr_dpi=200,
):
    all_chunks = []
    all_parents = []
    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            continue
        mmd_path = Path(mmd_dir) / (pdf_path.stem + ".mmd")
        image_dir = None
        if image_root is not None:
            image_dir = Path(image_root) / pdf_path.stem
        chunks, parents = ingest_pdf(
            pdf_path,
            mmd_path,
            nougat_cmd=nougat_cmd,
            image_dir=image_dir,
            image_dpi=image_dpi,
            table_tool=table_tool,
            tabula_guess=tabula_guess,
            tabula_lattice=tabula_lattice,
            tabula_stream=tabula_stream,
            tabula_jvm_opts=tabula_jvm_opts,
            image_extract_dir=image_extract_dir,
            ocr_on_garbled=ocr_on_garbled,
            ocr_image_dir=ocr_image_dir,
            ocr_dpi=ocr_dpi,
        )
        all_chunks.extend(chunks)
        all_parents.extend(parents)

    parents_path.parent.mkdir(parents=True, exist_ok=True)
    with open(parents_path, "w", encoding="utf-8") as f:
        json.dump(all_parents, f, ensure_ascii=False, indent=2)

    return all_chunks, all_parents

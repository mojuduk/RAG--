import json
import re
import os
import time
from pathlib import Path

from .config import OLLAMA_BASE_URL, OLLAMA_MODEL_NAME
from .indexer import build_tfidf_index, load_chunks
from .query import run_query
from .vector_store import query_chroma


def _load_generator(model_name):
    if not model_name:
        return None, None
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    except Exception:
        return None, None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )
        return generator, tokenizer
    except Exception:
        return None, None


def _format_source(metadata):
    source = metadata.get("source", "unknown")
    page = metadata.get("page")
    table = metadata.get("source_table") or metadata.get("caption")
    parts = [f"source={source}"]
    if page:
        parts.append(f"page={page}")
    if table:
        parts.append(f"table={table}")
    return ", ".join(parts)


def _build_prompt(question, evidences):
    evidence_lines = []
    for i, item in enumerate(evidences, 1):
        evidence_lines.append(
            f"[{i}] {_format_source(item['metadata'])}\n{item['content']}"
        )
    evidence_text = "\n\n".join(evidence_lines)
    return (
        "You are a pharma-process QA assistant. Answer strictly from evidence; do not fabricate.\n"
        "If evidence is insufficient, output exactly: Evidence insufficient.\n"
        "Keep response concise. Output only:\n"
        "1) Conclusion\n"
        "2) Evidence IDs\n\n"
        "Do not reveal analysis, chain-of-thought, drafts, or self-talk.\n"
        "Answer in Chinese.\n"
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        "Answer:"
    )

def _fallback_answer(question, evidences):
    if not evidences:
        return "证据不足，无法基于当前知识库回答该问题。"
    lines = ["基于检索证据，给出如下结论："]
    lines.append(_extractive_answer(question, evidences, max_chars=220))
    lines.append("依据：[" + ", ".join(str(i + 1) for i in range(min(3, len(evidences)))) + "]")
    return "\n".join(lines)

def _question_tokens(question):
    return set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", (question or "").lower()))

def _question_type(question):
    q = (question or "").lower()
    if any(k in q for k in ["区别", "差异", "不同", "对比", "vs", "versus"]):
        return "compare"
    if any(k in q for k in ["图", "图中", "图号", "figure", "流程图"]):
        return "image"
    if any(k in q for k in ["表", "字段", "列", "caption", "表格"]):
        return "table"
    if any(k in q for k in ["是什么", "定义", "含义", "主要推进器"]):
        return "define"
    return "general"


def _extract_page_refs(question):
    q = question or ""
    nums = re.findall(r"第\s*(\d+)\s*页", q)
    return [int(x) for x in nums if str(x).isdigit()]


def _extract_table_refs(question):
    q = question or ""
    refs = []
    for m in re.finditer(r"表\s*([0-9]+(?:[-\.][0-9]+)?)", q):
        refs.append(m.group(1))
    return refs


def _extract_figure_refs(question):
    q = question or ""
    refs = []
    for m in re.finditer(r"图\s*([0-9]+(?:[-\.][0-9]+)?)", q):
        refs.append(m.group(1))
    return refs


def _has_ref_text(text, prefix, refs):
    if not refs:
        return False
    text = text or ""
    for r in refs:
        if f"{prefix}{r}" in text or f"{prefix} {r}" in text:
            return True
    return False


def _apply_reference_constraints(question, items):
    if not items:
        return items
    page_refs = _extract_page_refs(question)
    table_refs = _extract_table_refs(question)
    figure_refs = _extract_figure_refs(question)
    if not page_refs and not table_refs and not figure_refs:
        return items

    out = items
    if page_refs:
        cand = []
        for x in out:
            md = x.get("metadata", {})
            page = md.get("page")
            if isinstance(page, str) and page.isdigit():
                page = int(page)
            blob = _metadata_text(md) + " " + x.get("content", "")
            if page in page_refs or any(f"第{p}页" in blob for p in page_refs):
                cand.append(x)
        if cand:
            out = cand

    if table_refs:
        cand = []
        for x in out:
            md = x.get("metadata", {})
            blob = _metadata_text(md) + " " + x.get("content", "")
            if _has_ref_text(blob, "表", table_refs):
                cand.append(x)
        if cand:
            out = cand

    if figure_refs:
        cand = []
        for x in out:
            md = x.get("metadata", {})
            blob = _metadata_text(md) + " " + x.get("content", "")
            if _has_ref_text(blob, "图", figure_refs):
                cand.append(x)
        if cand:
            out = cand

    return out


def _is_compare_question(question):
    q = (question or "").lower()
    patterns = [
        "\u533a\u522b", "\u5dee\u5f02", "\u4e0d\u540c", "\u5bf9\u6bd4", " vs ", " versus ", "compare"
    ]
    return any(re.search(p, q) for p in patterns)

def _metadata_text(metadata):
    m = metadata or {}
    parts = []
    for k in ("caption", "source_table", "title", "figure_no", "figure_caption", "page"):
        v = m.get(k)
        if v is None:
            continue
        parts.append(str(v))
    return " ".join(parts)


def _qa_overlap_score(question, text):
    q_toks = _question_tokens(question)
    if not q_toks:
        return 0.0
    t_toks = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", (text or "").lower()))
    if not t_toks:
        return 0.0
    return len(q_toks & t_toks) / len(q_toks)


def _split_sentences(text):
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"[\u3002\uFF01\uFF1F\uFF1B.!?;\n]+", text)
    out = []
    for p in parts:
        s = p.strip()
        if s:
            out.append(s)
    return out if out else [text]


def _table_fields():
    return [
        "投料数据",
        "出料数据",
        "过程参数",
        "工艺参数",
        "检测装备",
        "指令",
        "功能",
        "设备",
        "区块",
        "工艺单元",
    ]


def _asked_table_field(question):
    q = question or ""
    for f in _table_fields():
        if f in q:
            return f
    return ""


def _extract_table_answer(question, evidences, max_chars=260):
    field = _asked_table_field(question)
    q_toks = _question_tokens(question)
    candidates = []
    for e in evidences:
        md = e.get("metadata", {})
        ctype = md.get("chunk_type", "text")
        if ctype not in ("table_summary", "text"):
            continue
        content = str(e.get("content", "") or "")
        if not content:
            continue
        for s in _split_sentences(content):
            score = _qa_overlap_score(question, s)
            if field and field in s:
                score += 0.2
            if "表" in s:
                score += 0.06
            st = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", s.lower()))
            score += 0.05 * (len(q_toks & st) / max(1, len(q_toks))) if q_toks else 0.0
            score += 0.1 * float(e.get("adjusted_score", 0.0))
            candidates.append((score, s.strip()))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1][:max_chars]

def _best_evidence_content(question, evidences):
    if not evidences:
        return ""
    q_toks = _question_tokens(question)
    best = None
    best_score = -1.0
    for e in evidences:
        content = str(e.get("content", "") or "")
        if not content:
            continue
        c_toks = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", content.lower()))
        overlap = (len(q_toks & c_toks) / max(1, len(q_toks))) if q_toks else 0.0
        rank_bonus = 1.0 / max(1, int(e.get("rank", 1)))
        adjusted = float(e.get("adjusted_score", 0.0))
        score = 0.6 * overlap + 0.3 * adjusted + 0.1 * rank_bonus
        if score > best_score:
            best_score = score
            best = content
    return best or str(evidences[0].get("content", "") or "")

def _extractive_answer(question, evidences, max_chars=260):
    if not evidences:
        return ""
    qtype = _question_type(question)
    if qtype == "table":
        table_ans = _extract_table_answer(question, evidences, max_chars=max_chars)
        if table_ans:
            return table_ans

    # Build candidate sentences from top evidences and pick best by question overlap.
    candidates = []
    for e in evidences[: min(5, len(evidences))]:
        content = str(e.get("content", "") or "")
        if not content:
            continue
        for s in _split_sentences(content):
            score = _qa_overlap_score(question, s)
            if qtype == "define" and ("是" in s or "指" in s):
                score += 0.08
            score += 0.15 * float(e.get("adjusted_score", 0.0))
            score += 0.05 / max(1, int(e.get("rank", 1)))
            candidates.append((score, s.strip()))

    if not candidates:
        content = _best_evidence_content(question, evidences)
        return content[:max_chars]

    candidates.sort(key=lambda x: x[0], reverse=True)
    if qtype == "compare" or _is_compare_question(question):
        picked = []
        seen = set()
        for _, s in candidates:
            key = s[:40]
            if key in seen:
                continue
            seen.add(key)
            picked.append(s)
            if len(picked) >= 2:
                break
        return "; ".join(picked)[:max_chars]

    return candidates[0][1][:max_chars]

def _clean_generated_answer(text):
    if not text:
        return ""
    # Drop common hidden-thought wrappers and boilerplate.
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```[\s\S]*?```", "", text)
    lines = []
    for line in str(text).splitlines():
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if any(k in low for k in [
            "chain-of-thought", "step by step", "let's think", "analysis:",
            "i need to", "first,", "next,", "based on the provided evidence"
        ]):
            continue
        lines.append(s)
    if not lines:
        return ""
    # Keep answer concise: first 3 lines.
    return "\n".join(lines[:3]).strip()

def _answer_support_ratio(answer, evidences):
    if not answer or not evidences:
        return 0.0
    ans_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", answer.lower()))
    if not ans_tokens:
        return 0.0
    ev_text = "\n".join(x.get("content", "") for x in evidences)
    ev_tokens = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", ev_text.lower()))
    if not ev_tokens:
        return 0.0
    return len(ans_tokens & ev_tokens) / len(ans_tokens)


def _should_fallback_generated(answer, evidences, min_support=0.18):
    if not answer or len(answer.strip()) < 8:
        return True
    low = answer.lower()
    if any(k in low for k in [
        "i need to", "first,", "next,", "based on the provided evidence", "let's"
    ]):
        return True
    return _answer_support_ratio(answer, evidences) < min_support

def _is_low_quality_item(item):
    content = item.get("content", "")
    metadata = item.get("metadata", {})
    chunk_type = metadata.get("chunk_type", "text")
    if chunk_type == "table_summary":
        if "Unnamed:" in content:
            return True
        if float(item.get("normalized_score", 0.0)) <= 0:
            return True
    if not content or len(content.strip()) < 8:
        return True
    return False


def _trim_text(text, max_chars=650):
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def _compact_evidence_for_generation(evidences, max_items=3, max_chars_per_item=650):
    trimmed = []
    for item in evidences:
        new_item = dict(item)
        new_item["content"] = _trim_text(item.get("content", ""), max_chars=max_chars_per_item)
        trimmed.append(new_item)
        if len(trimmed) >= max_items:
            break
    return trimmed


def _generate_with_ollama(
    question,
    evidences,
    model_name,
    base_url,
    timeout=90,
    max_new_tokens=160,
):
    if not model_name:
        return None
    try:
        import requests
    except Exception:
        return None

    prompt = _build_prompt(question, evidences)
    payload = {
        "model": model_name,
        "stream": False,
        "prompt": prompt,
        "options": {
            "temperature": 0,
            "num_predict": max_new_tokens,
        },
    }
    url = base_url.rstrip("/") + "/api/generate"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        text = (data.get("response") or "").strip()
        if not text:
            text = (data.get("thinking") or "").strip()
        return text or None
    except Exception:
        return None


def _generate_with_openai_compatible(
    question,
    evidences,
    model_name,
    base_url,
    api_key_env="OPENAI_API_KEY",
    timeout=90,
    max_new_tokens=160,
):
    if not model_name or not base_url:
        return None
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except Exception:
        return None

    prompt = _build_prompt(question, evidences)
    try:
        client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=1)
        resp = client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=max_new_tokens,
            messages=[
                {"role": "system", "content": "You are a factual QA assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        if resp.choices and resp.choices[0].message:
            return (resp.choices[0].message.content or "").strip() or None
        return None
    except Exception:
        return None


def _source_quality(metadata):
    chunk_type = metadata.get("chunk_type", "text")
    weights = {
        "text": 1.0,
        "table_summary": 0.85,
        "image": 0.45,
    }
    return weights.get(chunk_type, 0.7)


def _normalized_score(item, channel):
    raw = float(item.get("score", 0.0))
    if channel == "vector":
        return 1.0 / (1.0 + max(raw, 0.0))
    return max(raw, 0.0)


def _merge_results(chroma_results, tfidf_results, top_k):
    # Weighted fusion: keyword-first, vector as complementary signal.
    merged_map = {}

    for item, channel in [(x, "keyword") for x in tfidf_results] + [
        (x, "vector") for x in chroma_results
    ]:
        content = item["chunk"]["content"]
        metadata = dict(item["chunk"].get("metadata", {}))
        score = _normalized_score(item, channel)

        if content not in merged_map:
            merged_map[content] = {
                "score": float(item.get("score", 0.0)),
                "normalized_score": score,
                "adjusted_score": 0.0,
                "content": content,
                "metadata": metadata,
                "channel": channel,
                "kw_norm": score if channel == "keyword" else 0.0,
                "vec_norm": score if channel == "vector" else 0.0,
            }
            continue

        merged = merged_map[content]
        merged["kw_norm"] = max(merged["kw_norm"], score if channel == "keyword" else 0.0)
        merged["vec_norm"] = max(merged["vec_norm"], score if channel == "vector" else 0.0)
        merged["channel"] = "hybrid"
        merged["score"] = max(merged["score"], float(item.get("score", 0.0)))

    merged = []
    for item in merged_map.values():
        in_both = 1.0 if (item["kw_norm"] > 0 and item["vec_norm"] > 0) else 0.0
        fused = 0.75 * item["kw_norm"] + 0.25 * item["vec_norm"] + 0.05 * in_both
        adjusted = fused * _source_quality(item["metadata"])
        item["normalized_score"] = round(fused, 6)
        item["adjusted_score"] = round(adjusted, 6)
        merged.append(item)

    merged.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return merged[:top_k]


def _rerank_by_question(question, items):
    q_toks = _question_tokens(question)
    qtype = _question_type(question)
    reranked = []
    for item in items:
        content = item.get("content", "")
        md = item.get("metadata", {})
        c_toks = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", content.lower()))
        m_text = _metadata_text(md).lower()
        m_toks = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", m_text))
        overlap_content = (len(q_toks & c_toks) / max(1, len(q_toks))) if q_toks else 0.0
        overlap_meta = (len(q_toks & m_toks) / max(1, len(q_toks))) if q_toks else 0.0
        score = float(item.get("adjusted_score", 0.0))
        score = 0.65 * score + 0.25 * overlap_content + 0.10 * overlap_meta
        ctype = md.get("chunk_type", "text")
        if qtype == "table" and ctype in ("table_summary",):
            score += 0.12
        if qtype == "image" and ctype in ("image",):
            score += 0.12
        if qtype == "define" and ctype == "text":
            score += 0.05
        item = dict(item)
        item["adjusted_score"] = round(score, 6)
        reranked.append(item)
    reranked.sort(key=lambda x: x.get("adjusted_score", 0.0), reverse=True)
    return reranked

def retrieve_evidence(
    question,
    chunks_path,
    vector_path,
    vectorizer_path,
    chroma_dir,
    store_path,
    top_k=5,
    use_hybrid=True,
    include_images=True,
    keyword_fastpath_threshold=0.06,
):
    tfidf_results = []
    chroma_results = []
    candidate_k = max(top_k * 3, top_k)

    if Path(vector_path).exists() and Path(vectorizer_path).exists() and Path(chunks_path).exists():
        try:
            tfidf_results = run_query(
                chunks_path,
                vector_path,
                vectorizer_path,
                question,
                top_k=candidate_k,
            )
        except Exception:
            # Auto-rebuild TF-IDF index when persisted artifacts are incompatible.
            try:
                chunks = load_chunks(chunks_path)
                build_tfidf_index(chunks, vector_path, vectorizer_path)
                tfidf_results = run_query(
                    chunks_path,
                    vector_path,
                    vectorizer_path,
                    question,
                    top_k=candidate_k,
                )
            except Exception:
                tfidf_results = []
    keyword_top_score = float(tfidf_results[0]["score"]) if tfidf_results else 0.0
    should_use_vector = use_hybrid and keyword_top_score < keyword_fastpath_threshold
    if should_use_vector and Path(store_path).exists():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            chroma_results = query_chroma(
                chroma_dir,
                store_path,
                question,
                top_k=candidate_k,
            )
        except Exception:
            chroma_results = []

    if use_hybrid:
        merged = _merge_results(chroma_results, tfidf_results, top_k=top_k)
    else:
        merged = _merge_results([], tfidf_results, top_k=top_k)

    merged = _apply_reference_constraints(question, merged)
    if not include_images:
        merged = [m for m in merged if m.get("metadata", {}).get("chunk_type", "text") != "image"]
    merged = [m for m in merged if not _is_low_quality_item(m)]
    merged = _rerank_by_question(question, merged)[:top_k]

    out = []
    for rank, x in enumerate(merged, 1):
        out.append(
            {
                "rank": rank,
                "content": x["content"],
                "metadata": x["metadata"],
                "channel": x["channel"],
                "score": x["score"],
                "normalized_score": x["normalized_score"],
                "adjusted_score": x["adjusted_score"],
            }
        )
    return out


def run_ask(
    question,
    chunks_path,
    vector_path,
    vectorizer_path,
    chroma_dir,
    store_path,
    top_k=5,
    use_hybrid=True,
    model_name=None,
    llm_provider="ollama",
    ollama_model=None,
    ollama_url=OLLAMA_BASE_URL,
    openai_model=None,
    openai_base_url=None,
    openai_api_key_env="OPENAI_API_KEY",
    max_new_tokens=256,
):
    t0 = time.perf_counter()
    evidences = retrieve_evidence(
        question=question,
        chunks_path=chunks_path,
        vector_path=vector_path,
        vectorizer_path=vectorizer_path,
        chroma_dir=chroma_dir,
        store_path=store_path,
        top_k=top_k,
        use_hybrid=use_hybrid,
        include_images=False,
    )
    t1 = time.perf_counter()

    resolved_ollama_model = ollama_model or OLLAMA_MODEL_NAME
    resolved_openai_model = openai_model or ollama_model or OLLAMA_MODEL_NAME
    used_generator = False
    extractive_base = _extractive_answer(question, evidences)
    answer = extractive_base if extractive_base else None
    gen_evidences = _compact_evidence_for_generation(
        evidences,
        max_items=min(3, top_k),
        max_chars_per_item=650,
    )

    if gen_evidences:
        if str(llm_provider).lower() == "openai":
            gen_answer = _generate_with_openai_compatible(
                question=question,
                evidences=gen_evidences,
                model_name=resolved_openai_model,
                base_url=openai_base_url,
                api_key_env=openai_api_key_env,
                max_new_tokens=max_new_tokens,
            )
        else:
            gen_answer = _generate_with_ollama(
                question=question,
                evidences=gen_evidences,
                model_name=resolved_ollama_model,
                base_url=ollama_url,
                max_new_tokens=max_new_tokens,
            )
        gen_answer = _clean_generated_answer(gen_answer)
        gen_bad = _should_fallback_generated(gen_answer, evidences)
        if not gen_bad:
            # Only accept generated answer when it is more question-relevant than extractive baseline.
            gen_rel = _qa_overlap_score(question, gen_answer)
            ext_rel = _qa_overlap_score(question, extractive_base)
            if gen_rel >= ext_rel + 0.02:
                answer = gen_answer
                used_generator = True

    if not answer:
        generator, _ = _load_generator(model_name)
        if generator and gen_evidences:
            prompt = _build_prompt(question, gen_evidences)
            try:
                outputs = generator(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    return_full_text=False,
                )
                gen_answer = _clean_generated_answer(outputs[0]["generated_text"].strip())
                if not _should_fallback_generated(gen_answer, evidences):
                    gen_rel = _qa_overlap_score(question, gen_answer)
                    ext_rel = _qa_overlap_score(question, extractive_base)
                    if gen_rel >= ext_rel + 0.02:
                        answer = gen_answer
                        used_generator = True
            except Exception:
                pass

    if not answer:
        answer = extractive_base
    if not answer:
        answer = _fallback_answer(question, evidences)
    t2 = time.perf_counter()

    avg_conf = 0.0
    if evidences:
        avg_conf = sum(e.get("adjusted_score", 0.0) for e in evidences) / len(evidences)
    confidence = round(min(max(avg_conf, 0.0), 1.0), 4)
    used_ocr = any(bool(e.get("metadata", {}).get("used_ocr")) for e in evidences)
    retrieval_mode = "hybrid" if use_hybrid else "keyword"

    return {
        "question": question,
        "answer": answer,
        "retrieval_mode": retrieval_mode,
        "confidence": confidence,
        "used_ocr": used_ocr,
        "generator_used": used_generator,
        "latency_ms": {
            "retrieve": round((t1 - t0) * 1000, 2),
            "generate": round((t2 - t1) * 1000, 2),
            "total": round((t2 - t0) * 1000, 2),
        },
        "evidence": evidences,
    }


def format_ask_output(result):
    out = {
        "question": result["question"],
        "answer": result["answer"],
        "retrieval_mode": result.get("retrieval_mode"),
        "confidence": result.get("confidence"),
        "used_ocr": result.get("used_ocr"),
        "generator_used": result.get("generator_used"),
        "latency_ms": result.get("latency_ms"),
        "evidence": result["evidence"],
    }
    return json.dumps(out, ensure_ascii=False, indent=2)


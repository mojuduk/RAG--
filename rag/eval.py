import csv
import json
import os
import re
import time
from pathlib import Path

from .ask import run_ask
from .query import run_query
from .vector_store import query_chroma


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return 0.0


def _norm_text(text):
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _tokenize(text):
    text = _norm_text(text)
    tokens = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)
    return tokens


def _f1_score(pred, gold):
    p_tokens = _tokenize(pred)
    g_tokens = _tokenize(gold)
    if not p_tokens or not g_tokens:
        return 0.0
    p_count = {}
    g_count = {}
    for t in p_tokens:
        p_count[t] = p_count.get(t, 0) + 1
    for t in g_tokens:
        g_count[t] = g_count.get(t, 0) + 1
    hit = 0
    for t, c in p_count.items():
        hit += min(c, g_count.get(t, 0))
    if hit == 0:
        return 0.0
    precision = hit / len(p_tokens)
    recall = hit / len(g_tokens)
    return 2 * precision * recall / (precision + recall)


def _char_ngram_jaccard(a, b, n=2):
    a = _norm_text(a)
    b = _norm_text(b)
    if len(a) < n or len(b) < n:
        return 0.0
    sa = {a[i:i+n] for i in range(len(a)-n+1)}
    sb = {b[i:i+n] for i in range(len(b)-n+1)}
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _keyword_hit_ratio(pred, keywords):
    kws = [str(k).strip().lower() for k in (keywords or []) if str(k).strip()]
    if not kws:
        return 0.0
    p = _norm_text(pred)
    if not p:
        return 0.0
    hit = sum(1 for k in kws if k in p)
    return hit / len(kws)


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


def _metadata_text(metadata):
    m = metadata or {}
    parts = []
    for k in ("caption", "source_table", "title", "figure_no", "figure_caption", "page"):
        v = m.get(k)
        if v is None:
            continue
        parts.append(str(v))
    return " ".join(parts)


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
    cands = []
    for e in evidences:
        md = e.get("metadata", {})
        ctype = md.get("chunk_type", "text")
        if ctype not in ("table_summary", "text"):
            continue
        content = str(e.get("content", "") or "")
        if not content:
            continue
        for s in _split_sentences(content):
            score = len(q_toks & set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", s.lower()))) / max(1, len(q_toks)) if q_toks else 0.0
            if field and field in s:
                score += 0.2
            if "表" in s:
                score += 0.06
            score += 0.1 * _safe_float(e.get("norm_score", e.get("score", 0.0)))
            cands.append((score, s.strip()))
    if not cands:
        return ""
    cands.sort(key=lambda x: x[0], reverse=True)
    return cands[0][1][:max_chars]

def _compose_extract_answer(question, evidences):
    if not evidences:
        return ""
    q_toks = _question_tokens(question)
    qtype = _question_type(question)
    if qtype == "table":
        ans = _extract_table_answer(question, evidences)
        if ans:
            return ans
    best_content = ""
    best_score = -1.0
    for e in evidences:
        content = str(e.get("content", "") or "")
        if not content:
            continue
        c_toks = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", content.lower()))
        overlap = (len(q_toks & c_toks) / max(1, len(q_toks))) if q_toks else 0.0
        rank_bonus = 1.0 / max(1, int(e.get("rank", 1)))
        score = overlap + 0.1 * rank_bonus
        if score > best_score:
            best_score = score
            best_content = content
    if not best_content:
        best_content = str(evidences[0].get("content", "") or "")
    sents = _split_sentences(best_content)
    if not sents:
        return best_content[:260]
    best_sent = sents[0]
    best_sent_score = -1.0
    for s in sents:
        st = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", s.lower()))
        overlap = (len(q_toks & st) / max(1, len(q_toks))) if q_toks else 0.0
        if qtype == "define" and ("是" in s or "指" in s):
            overlap += 0.08
        score = overlap + min(len(s) / 120.0, 0.3)
        if score > best_sent_score:
            best_sent_score = score
            best_sent = s
    if qtype == "compare":
        picked = []
        seen = set()
        scored = []
        for s in sents:
            st = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", s.lower()))
            overlap = (len(q_toks & st) / max(1, len(q_toks))) if q_toks else 0.0
            scored.append((overlap, s))
        for _, s in sorted(scored, key=lambda x: x[0], reverse=True):
            key = s[:40]
            if key in seen:
                continue
            seen.add(key)
            picked.append(s.strip())
            if len(picked) >= 2:
                break
        if picked:
            return "; ".join(picked)[:260]
    return best_sent.strip()[:260]

def _answer_match(pred, gold, keywords=None, f1_threshold=0.5, sem_threshold=0.55, kw_threshold=0.5):
    p = _norm_text(pred)
    g = _norm_text(gold)
    if not g:
        return False, 0.0, False, 0.0, 0.0
    em = p == g
    f1 = _f1_score(p, g)
    contains = bool(p) and bool(g) and (g in p or p in g)
    sem = 0.6 * f1 + 0.4 * _char_ngram_jaccard(p, g, n=2)
    kw_ratio = _keyword_hit_ratio(pred, keywords)
    ok = em or contains or (f1 >= f1_threshold) or (sem >= sem_threshold) or (kw_ratio >= kw_threshold)
    return ok, f1, em, round(sem, 4), round(kw_ratio, 4)


def _consistency_score(answer, evidences):
    ans_tokens = set(_tokenize(answer))
    if not ans_tokens:
        return 0.0
    best = 0.0
    for e in evidences:
        ev_tokens = set(_tokenize(e.get("content", "")))
        if not ev_tokens:
            continue
        cov = len(ans_tokens & ev_tokens) / len(ans_tokens)
        if cov > best:
            best = cov
    return best


def _to_keywords(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "|" in text:
        return [x.strip() for x in text.split("|") if x.strip()]
    if "," in text:
        return [x.strip() for x in text.split(",") if x.strip()]
    return [text]


def _load_qa_set(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"QA file not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8-sig") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("JSON QA set must be a list of objects.")
        return data

    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _normalize_sample(raw, idx):
    question = str(
        raw.get("question")
        or raw.get("query")
        or raw.get("q")
        or ""
    ).strip()
    if not question:
        raise ValueError(f"Sample #{idx} missing question field.")

    answer = str(
        raw.get("gold_answer")
        or raw.get("answer")
        or raw.get("reference_answer")
        or ""
    ).strip()
    keywords = _to_keywords(
        raw.get("evidence_keywords")
        or raw.get("keywords")
        or raw.get("gold_keywords")
    )
    gold_page = raw.get("gold_page", raw.get("page"))
    try:
        gold_page = int(gold_page) if str(gold_page).strip() else None
    except Exception:
        gold_page = None

    return {
        "id": str(raw.get("id") or idx),
        "question": question,
        "gold_answer": answer,
        "evidence_keywords": keywords,
        "gold_page": gold_page,
    }


def _retrieve_keyword(question, chunks_path, vector_path, vectorizer_path, top_k):
    rows = run_query(chunks_path, vector_path, vectorizer_path, question, top_k=top_k)
    out = []
    for i, x in enumerate(rows, 1):
        out.append(
            {
                "rank": i,
                "content": x["chunk"]["content"],
                "metadata": x["chunk"].get("metadata", {}),
                "channel": "keyword",
                "score": _safe_float(x.get("score")),
            }
        )
    return _apply_reference_constraints(question, out)


def _retrieve_vector(question, chroma_dir, store_path, top_k):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    try:
        rows = query_chroma(chroma_dir, store_path, question, top_k=top_k)
    except Exception:
        rows = []
    out = []
    for i, x in enumerate(rows, 1):
        out.append(
            {
                "rank": i,
                "content": x["chunk"]["content"],
                "metadata": x["chunk"].get("metadata", {}),
                "channel": "vector",
                "score": _safe_float(x.get("score")),
                "norm_score": 1.0 / (1.0 + max(_safe_float(x.get("score")), 0.0)),
            }
        )
    return _apply_reference_constraints(question, out)


def _retrieve_hybrid(
    question, chunks_path, vector_path, vectorizer_path, chroma_dir, store_path, top_k
):
    # Pull a wider candidate pool and re-rank with keyword-first weighted fusion.
    candidate_k = max(top_k * 3, top_k)
    kw = _retrieve_keyword(question, chunks_path, vector_path, vectorizer_path, top_k=candidate_k)
    vec = _retrieve_vector(question, chroma_dir, store_path, top_k=candidate_k)

    def _minmax(values):
        if not values:
            return 0.0, 1.0
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return lo, lo + 1.0
        return lo, hi

    kw_scores = [_safe_float(x.get("score")) for x in kw]
    vec_scores = [_safe_float(x.get("norm_score", x.get("score"))) for x in vec]
    kw_lo, kw_hi = _minmax(kw_scores)
    vec_lo, vec_hi = _minmax(vec_scores)

    merged = {}

    for x in kw:
        content = x["content"]
        raw = _safe_float(x.get("score"))
        kw_norm = (raw - kw_lo) / (kw_hi - kw_lo)
        item = {
            "rank": 0,
            "content": content,
            "metadata": x.get("metadata", {}),
            "channel": "keyword",
            "score": raw,
            "kw_norm": kw_norm,
            "vec_norm": 0.0,
        }
        merged[content] = item

    for x in vec:
        content = x["content"]
        vraw = _safe_float(x.get("norm_score", x.get("score")))
        vec_norm = (vraw - vec_lo) / (vec_hi - vec_lo)
        if content in merged:
            merged[content]["vec_norm"] = max(merged[content]["vec_norm"], vec_norm)
            merged[content]["channel"] = "hybrid"
        else:
            merged[content] = {
                "rank": 0,
                "content": content,
                "metadata": x.get("metadata", {}),
                "channel": "vector",
                "score": _safe_float(x.get("score")),
                "kw_norm": 0.0,
                "vec_norm": vec_norm,
            }

    q_toks = _question_tokens(question)
    qtype = _question_type(question)
    reranked = []
    for item in merged.values():
        in_both = 1.0 if (item["kw_norm"] > 0 and item["vec_norm"] > 0) else 0.0
        # Keyword is stronger baseline on current corpus, vector is complementary.
        fused = 0.75 * item["kw_norm"] + 0.25 * item["vec_norm"] + 0.05 * in_both
        content = item.get("content", "")
        c_toks = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", content.lower()))
        overlap_content = (len(q_toks & c_toks) / max(1, len(q_toks))) if q_toks else 0.0
        m_text = _metadata_text(item.get("metadata", {})).lower()
        m_toks = set(re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", m_text))
        overlap_meta = (len(q_toks & m_toks) / max(1, len(q_toks))) if q_toks else 0.0
        score = 0.65 * fused + 0.25 * overlap_content + 0.10 * overlap_meta
        ctype = item.get("metadata", {}).get("chunk_type", "text")
        if qtype == "table" and ctype == "table_summary":
            score += 0.12
        if qtype == "image" and ctype == "image":
            score += 0.12
        if qtype == "define" and ctype == "text":
            score += 0.05
        item["norm_score"] = round(score, 6)
        reranked.append(item)

    reranked.sort(key=lambda d: d.get("norm_score", 0.0), reverse=True)
    reranked = _apply_reference_constraints(question, reranked)
    top = reranked[:top_k]
    for i, x in enumerate(top, 1):
        x["rank"] = i
    return top


def _build_no_rag_prompt(question):
    return (
        "你是制药工艺问答助手。"
        "请直接基于你已有知识回答问题，不要编造具体页码或表号。"
        "如果不确定，请明确说明不确定。"
        "回答保持简洁、中文输出。\n\n"
        f"问题：{question}\n\n"
        "答案："
    )


def _answer_no_rag(
    question,
    llm_provider,
    ollama_model,
    ollama_url,
    openai_model,
    openai_base_url,
    openai_api_key_env,
    max_new_tokens,
):
    prompt = _build_no_rag_prompt(question)
    provider = str(llm_provider or "ollama").lower()

    if provider == "openai":
        if not openai_model or not openai_base_url:
            return ""
        api_key = os.getenv(openai_api_key_env or "OPENAI_API_KEY", "")
        if not api_key:
            return ""
        try:
            from openai import OpenAI
        except Exception:
            return ""
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=openai_base_url,
                timeout=90,
                max_retries=1,
            )
            resp = client.chat.completions.create(
                model=openai_model,
                temperature=0,
                max_tokens=max_new_tokens,
                messages=[
                    {"role": "system", "content": "You are a factual QA assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            if resp.choices and resp.choices[0].message:
                return (resp.choices[0].message.content or "").strip()
        except Exception:
            return ""
        return ""

    if not ollama_model:
        return ""
    try:
        import requests
    except Exception:
        return ""
    payload = {
        "model": ollama_model,
        "stream": False,
        "prompt": prompt,
        "options": {
            "temperature": 0,
            "num_predict": max_new_tokens,
        },
    }
    try:
        url = (ollama_url or "").rstrip("/") + "/api/generate"
        resp = requests.post(url, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return (data.get("response") or "").strip()
    except Exception:
        return ""


def _retrieval_hit(evidences, gold_answer="", keywords=None, gold_page=None):
    keywords = keywords or []
    if not evidences:
        return False

    text_blob = "\n".join(x.get("content", "") for x in evidences)
    if keywords:
        return any(k and (k in text_blob) for k in keywords)
    if gold_page is not None:
        return any(x.get("metadata", {}).get("page") == gold_page for x in evidences)
    if gold_answer:
        g = _norm_text(gold_answer)
        return any(g and g in _norm_text(x.get("content", "")) for x in evidences)
    return False


def _to_rate(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _eval_one_mode(
    mode_name,
    samples,
    top_k,
    chunks_path,
    vector_path,
    vectorizer_path,
    chroma_dir,
    store_path,
    llm_provider,
    ollama_model,
    ollama_url,
    openai_model,
    openai_base_url,
    openai_api_key_env,
    max_new_tokens,
):
    details = []
    hit_cnt = 0
    acc_cnt = 0
    acc_denom = 0
    cons_cnt = 0
    total_latency = 0.0

    for s in samples:
        q = s["question"]
        t0 = time.perf_counter()
        if mode_name == "keyword_only":
            evidences = _retrieve_keyword(q, chunks_path, vector_path, vectorizer_path, top_k=top_k)
            answer = _compose_extract_answer(q, evidences)
        elif mode_name == "vector_only":
            evidences = _retrieve_vector(q, chroma_dir, store_path, top_k=top_k)
            answer = _compose_extract_answer(q, evidences)
        elif mode_name == "hybrid_retrieval":
            evidences = _retrieve_hybrid(
                q, chunks_path, vector_path, vectorizer_path, chroma_dir, store_path, top_k=top_k
            )
            answer = _compose_extract_answer(q, evidences)
        elif mode_name == "hybrid_plus_generation":
            result = run_ask(
                question=q,
                chunks_path=chunks_path,
                vector_path=vector_path,
                vectorizer_path=vectorizer_path,
                chroma_dir=chroma_dir,
                store_path=store_path,
                top_k=top_k,
                use_hybrid=True,
                llm_provider=llm_provider,
                ollama_model=ollama_model,
                ollama_url=ollama_url,
                openai_model=openai_model,
                openai_base_url=openai_base_url,
                openai_api_key_env=openai_api_key_env,
                max_new_tokens=max_new_tokens,
            )
            evidences = result.get("evidence", [])
            answer = result.get("answer", "")
        elif mode_name == "hybrid_plus_generation_safe":
            result = run_ask(
                question=q,
                chunks_path=chunks_path,
                vector_path=vector_path,
                vectorizer_path=vectorizer_path,
                chroma_dir=chroma_dir,
                store_path=store_path,
                top_k=top_k,
                use_hybrid=True,
                llm_provider=llm_provider,
                ollama_model=ollama_model,
                ollama_url=ollama_url,
                openai_model=openai_model,
                openai_base_url=openai_base_url,
                openai_api_key_env=openai_api_key_env,
                max_new_tokens=max_new_tokens,
            )
            evidences = result.get("evidence", [])
            # Safe mode: keep generation retrieval path, but score with extractive answer.
            answer = _compose_extract_answer(q, evidences)
        elif mode_name == "no_rag_llm":
            evidences = []
            answer = _answer_no_rag(
                question=q,
                llm_provider=llm_provider,
                ollama_model=ollama_model,
                ollama_url=ollama_url,
                openai_model=openai_model,
                openai_base_url=openai_base_url,
                openai_api_key_env=openai_api_key_env,
                max_new_tokens=max_new_tokens,
            )
        else:
            raise ValueError(f"Unsupported mode: {mode_name}")
        t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000.0
        total_latency += latency_ms

        hit = _retrieval_hit(
            evidences,
            gold_answer=s["gold_answer"],
            keywords=s["evidence_keywords"],
            gold_page=s["gold_page"],
        )
        if hit:
            hit_cnt += 1

        acc_ok = False
        f1 = 0.0
        em = False
        sem_score = 0.0
        kw_hit_ratio = 0.0
        if s["gold_answer"]:
            acc_denom += 1
            acc_ok, f1, em, sem_score, kw_hit_ratio = _answer_match(
                answer,
                s["gold_answer"],
                keywords=s["evidence_keywords"],
            )
            if acc_ok:
                acc_cnt += 1

        cons_score = _consistency_score(answer, evidences)
        cons_ok = cons_score >= 0.3
        if cons_ok:
            cons_cnt += 1

        details.append(
            {
                "sample_id": s["id"],
                "mode": mode_name,
                "question": q,
                "gold_answer": s["gold_answer"],
                "pred_answer": answer,
                "hit_top_k": int(hit),
                "answer_correct": int(acc_ok) if s["gold_answer"] else "",
                "answer_f1": round(f1, 4) if s["gold_answer"] else "",
                "answer_em": int(em) if s["gold_answer"] else "",
                "semantic_equiv_score": round(sem_score, 4) if s["gold_answer"] else "",
                "keyword_hit_ratio": round(kw_hit_ratio, 4) if s["gold_answer"] else "",
                "factual_consistent": int(cons_ok),
                "factual_consistency_score": round(cons_score, 4),
                "latency_ms": round(latency_ms, 2),
                "evidence_count": len(evidences),
                "gold_page": s["gold_page"] if s["gold_page"] is not None else "",
                "gold_keywords": "|".join(s["evidence_keywords"]) if s["evidence_keywords"] else "",
            }
        )

    n = len(samples)
    summary = {
        "mode": mode_name,
        "samples": n,
        "top_k_hit_rate": _to_rate(hit_cnt, n),
        "answer_accuracy": _to_rate(acc_cnt, acc_denom) if acc_denom else "",
        "factual_consistency": _to_rate(cons_cnt, n),
        "avg_latency_ms": round(total_latency / n, 2) if n else 0.0,
    }
    return summary, details


def _write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_eval(
    qa_path,
    top_k,
    chunks_path,
    vector_path,
    vectorizer_path,
    chroma_dir,
    store_path,
    out_summary_csv,
    out_details_csv,
    llm_provider,
    ollama_model,
    ollama_url,
    openai_model,
    openai_base_url,
    openai_api_key_env,
    max_new_tokens=160,
):
    raw_samples = _load_qa_set(qa_path)
    samples = [_normalize_sample(x, i + 1) for i, x in enumerate(raw_samples)]
    modes = [
        "no_rag_llm",
        "keyword_only",
        "vector_only",
        "hybrid_retrieval",
        "hybrid_plus_generation",
        "hybrid_plus_generation_safe",
    ]

    summaries = []
    all_details = []
    for m in modes:
        summary, details = _eval_one_mode(
            mode_name=m,
            samples=samples,
            top_k=top_k,
            chunks_path=chunks_path,
            vector_path=vector_path,
            vectorizer_path=vectorizer_path,
            chroma_dir=chroma_dir,
            store_path=store_path,
            llm_provider=llm_provider,
            ollama_model=ollama_model,
            ollama_url=ollama_url,
            openai_model=openai_model,
            openai_base_url=openai_base_url,
            openai_api_key_env=openai_api_key_env,
            max_new_tokens=max_new_tokens,
        )
        summaries.append(summary)
        all_details.extend(details)

    _write_csv(
        out_summary_csv,
        summaries,
        fieldnames=[
            "mode",
            "samples",
            "top_k_hit_rate",
            "answer_accuracy",
            "factual_consistency",
            "avg_latency_ms",
        ],
    )
    _write_csv(
        out_details_csv,
        all_details,
        fieldnames=[
            "sample_id",
            "mode",
            "question",
            "gold_answer",
            "pred_answer",
            "hit_top_k",
            "answer_correct",
            "answer_f1",
            "answer_em",
            "semantic_equiv_score",
            "keyword_hit_ratio",
            "factual_consistent",
            "factual_consistency_score",
            "latency_ms",
            "evidence_count",
            "gold_page",
            "gold_keywords",
        ],
    )
    return summaries

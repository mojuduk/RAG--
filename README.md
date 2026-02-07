# RAG 毕业设计项目（稳心颗粒工艺）

本项目基于任务书与《精益六西格玛补充方案》文档，构建面向制药工艺的检索增强生成（RAG）数据处理与检索原型。重点处理 DOCX 中的表格与图片，保证后续检索质量与准确率。

## 功能概览
- 文档解析：DOCX 按正文与表格提取并结构化；PDF 按正文与 Nougat 表格处理。
- 表格处理：逐行输出 Markdown List，并附带 `process_name` / `equipment` / `source_table` 元数据。
- 正文处理：清洗空行、页眉页脚噪声，短段落合并。
- 输出结果：生成 `knowledge_base.json` 列表数据集与 `cleaned_data.txt`，并打印 2 条表格样例 + 2 条文本样例。

## 目录结构
- `main.py`：命令行入口。
- `rag/ingest_docx.py`：DOCX 解析与结构化切分。
- `data/knowledge_base.json`：抽取后的知识块数据集。
- `data/cleaned_data.txt`：每行一个 Chunk，便于人工检查。
- `data/index/`：向量索引输出目录。
- `data/index/chroma/`：语义向量库（Chroma 持久化目录）。
- `data/index/vector_store.json`：向量库元数据与内容存档。
- `data/mmd/`：Nougat 生成的 .mmd 文件目录。
- `data/table_parents.json`：PDF 表格 LaTeX 父文档存档。
  - 与 `knowledge_base.json` 中的 `parent_id` 对应，用于父子文档检索。

## 快速开始
```bash
python main.py ingest
python main.py index
python main.py query "精益六西格玛主要推进器" --top-k 5
python main.py vindex
python main.py vquery "醇提 工艺参数" --top-k 5
python main.py vquery "醇提 工艺参数" --top-k 5 --hybrid
python main.py pdf-ingest --pdfs 精益六西格玛补充方案.pdf --table-tool pdfplumber
python main.py pdf-ingest --pdfs 精益六西格玛补充方案.pdf --table-tool tabula
python main.py pdf-ingest --pdfs 精益六西格玛补充方案.pdf --table-tool nougat --nougat-cmd python -m predict --markdown --pdf-to-images
```

如需指定 DOCX：
```bash
python main.py ingest --docs 精益六西格玛补充方案.docx 马科举-任务书.docx
```

## 处理策略说明
1. **文本块**：清洗空白与页眉页脚噪声，短段落合并到后续段落。
2. **表格块**：将每一行转换为自然语言描述，格式为 `[表格上下文] 字段名1: 值1; 字段名2: 值2; ...`。

## 后续扩展建议
- 替换为本地 embedding 模型与向量数据库（如 FAISS、Milvus）。
- 在问答模块引入量化小模型，结合检索结果生成答案。
- 增加基于问题类型的多路检索（表格优先、流程优先）。

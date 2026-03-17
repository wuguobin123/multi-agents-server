# 文档知识库后端实现现状与方案

## 1. 范围说明

当前需求聚焦后端，不包含前端页面实现。目标能力有三项：

1. 前端上传文档后，后端能够按用户选择的切片策略和向量模型完成向量化与存储。
2. 源文档元数据和文档地址保存到 MySQL。
3. 支持自定义知识库，不同领域知识彼此隔离。

## 2. 当前仓库实现检查

### 2.1 已经具备的能力

1. 已有 RAG 重建链路。
   当前后端可以加载文档、切片、生成向量并写入向量库，主流程在 `app/rag/service.py`。

2. 已支持切片策略配置。
   当前支持 `recursive_character` 和 `qa_pair` 两种切片策略，且 `/v1/rag/rebuild` 接口允许传入切片配置。

3. 已支持向量模型配置。
   当前配置项中已经有 `embedding_provider`、`embedding_model`、`embedding_dimension`、`embedding_api_key`、`embedding_base_url` 等字段。

4. 已支持向量存储抽象。
   当前支持 `local` 和 `qdrant` 两种向量存储后端。

5. 已支持 SQL 持久化知识库基础信息。
   当前已有 `knowledge_document`、`knowledge_chunk`、`ingestion_job` 三张表，用于保存文档、切片和导入任务记录。

### 2.2 尚未满足需求的部分

1. 未实现“前端上传文档”的后端接口。
   当前文档来源仅来自 `data/knowledge_base` 目录扫描，`FileSystemDocumentLoader` 只读取 `.md` 和 `.txt` 文件，不支持前端上传后的文件接收、落盘和解析。

2. 未实现“按上传任务选择切片策略和向量模型”的后端能力。
   当前切片配置可以在重建接口中临时传入，但向量模型仍是全局配置；没有按文档或按知识库保存所选 embedding 配置。

3. 未实现“自定义知识库隔离”。
   当前只有单一 `collection_name`，SQL 表结构中也没有 `knowledge_base_id`、`tenant_id`、`namespace` 之类的隔离字段，实际仍是单知识库模式。

4. 未完整保存“源文档地址和业务元数据”。
   当前 `knowledge_document` 只保存 `source` 和 `metadata_json`，没有明确的 `file_url`、`storage_path`、`file_hash`、`file_size`、`mime_type`、`status` 等字段。

5. 未实现“单文档增量入库”。
   当前核心接口是整体重建知识库，会先清空再重建，不适合前端逐个上传文档的业务场景。

6. 未体现“必须落到 MySQL”的明确实现状态。
   当前仓库支持 SQLAlchemy 数据库连接，但默认配置仍是 SQLite；如果要满足需求，需要明确使用 MySQL 连接串和对应表结构。

## 3. 结论

当前仓库已经实现了知识库后端的基础 RAG 能力，但还没有实现“面向产品化文档上传”的完整后端方案。

更准确地说：

1. 向量化、切片、检索、Qdrant 接入，这部分已经有基础能力。
2. 上传、知识库隔离、单文档任务化入库、MySQL 文档元数据管理，这部分还没有实现完整闭环。

因此，这个功能当前是“部分实现”，还不能认定为已经完整满足需求。

## 4. 建议的后端实现方案

### 4.1 推荐整体流程

建议后端链路统一为：

1. 前端创建知识库或选择已有知识库。
2. 前端上传文档，并提交切片策略、切片参数、向量模型配置。
3. 后端先保存文件到对象存储或本地文件存储。
4. 后端在 MySQL 中写入文档记录和导入任务记录。
5. 异步任务执行解析、切片、向量化、写入向量库。
6. 完成后回写任务状态、文档状态、切片数量、向量条数等结果。
7. 检索时必须带 `knowledge_base_id` 过滤，只在指定知识库范围内召回。

### 4.2 推荐接口设计

建议新增以下后端接口：

1. `POST /v1/knowledge-bases`
   创建知识库，字段包含 `name`、`code`、`description`、`embedding_provider`、`embedding_model`、`vector_backend`。

2. `GET /v1/knowledge-bases`
   查询知识库列表。

3. `POST /v1/knowledge-bases/{kb_id}/documents`
   上传文档，使用 `multipart/form-data`，字段至少包含：
   `file`、`chunking_strategy`、`chunk_size`、`chunk_overlap`、`embedding_provider`、`embedding_model`。

4. `GET /v1/knowledge-bases/{kb_id}/documents`
   查询知识库下的文档列表和处理状态。

5. `GET /v1/ingestion-jobs/{job_id}`
   查询导入任务状态。

6. `POST /v1/knowledge-bases/{kb_id}/documents/{document_id}/reindex`
   对单文档重新切片和重建向量。

7. `DELETE /v1/knowledge-bases/{kb_id}/documents/{document_id}`
   删除文档，同时删除 MySQL 记录和向量库中的对应切片。

### 4.3 MySQL 数据模型建议

建议在当前表结构基础上扩展，而不是继续沿用单知识库表结构。

#### 表 1：`knowledge_base`

建议字段：

1. `id`
2. `name`
3. `code`
4. `description`
5. `status`
6. `embedding_provider`
7. `embedding_model`
8. `embedding_dimension`
9. `vector_backend`
10. `vector_collection`
11. `created_at`
12. `updated_at`

#### 表 2：`knowledge_document`

建议字段：

1. `id`
2. `knowledge_base_id`
3. `original_filename`
4. `storage_path`
5. `file_url`
6. `file_hash`
7. `file_size`
8. `mime_type`
9. `parser_type`
10. `chunking_strategy`
11. `chunking_config_json`
12. `embedding_provider`
13. `embedding_model`
14. `status`
15. `metadata_json`
16. `created_at`
17. `updated_at`

#### 表 3：`knowledge_chunk`

建议字段：

1. `id`
2. `knowledge_base_id`
3. `document_id`
4. `chunk_index`
5. `content`
6. `token_count`
7. `vector_point_id`
8. `metadata_json`
9. `created_at`

#### 表 4：`ingestion_job`

建议字段：

1. `id`
2. `knowledge_base_id`
3. `document_id`
4. `status`
5. `stage`
6. `error_message`
7. `started_at`
8. `finished_at`
9. `created_at`
10. `updated_at`

### 4.4 向量存储隔离方案

这一项是当前缺口的重点，必须明确。

推荐方案分两档：

1. 中小规模知识库数量场景。
   使用“一知识库一 collection”模式。每个知识库独立一个 Qdrant collection，隔离最直接，删除和重建也简单。

2. 大规模知识库数量场景。
   使用“共享 collection + payload filter”模式。所有向量进入同一个 collection，但 payload 必须包含 `knowledge_base_id`，检索时强制追加过滤条件。

如果当前项目以“先快速上线、知识库数量有限”为主，建议优先采用第一种，即“一知识库一 collection”。这和现有 `collection_name` 配置方式也更容易衔接。

### 4.5 与现有代码的衔接方式

当前代码可以复用，但需要从“整体重建目录”升级到“按知识库/文档入库”。

建议改造点如下：

1. `app/rag/loader.py`
   不再只保留目录扫描加载器，新增上传文件加载器，支持从存储路径读取单文档内容。

2. `app/rag/service.py`
   将当前 `rebuild()` 扩展为更通用的 `ingest_documents()` 或 `ingest_document()`，入参至少包含 `knowledge_base_id`、`document_id`、`chunking_settings`、`embedding_profile`、`collection_name`。

3. `app/rag/vector_store.py`
   增加按知识库动态选择 collection 的能力；如果采用共享 collection，则增加 query filter 能力。

4. `app/repositories/models.py`
   为 `knowledge_document`、`knowledge_chunk`、`ingestion_job` 增加 `knowledge_base_id`、状态字段和文件存储字段。

5. `app/repositories/sql.py`
   将当前“全量替换知识库”改为：
   `replace_document_chunks(document_id)`
   `delete_document(document_id)`
   `create_knowledge_base()`
   `list_documents(kb_id)`
   `update_document_status()`
   这类面向业务对象的方法。

6. `app/api/routes.py`
   保留现有 `/v1/rag/rebuild` 作为运维接口，新增面向产品功能的知识库管理接口。

### 4.6 异步任务建议

上传后不建议在请求线程里直接完成全文向量化，建议采用异步任务。

可选实现：

1. 轻量方案：FastAPI `BackgroundTasks`
2. 稳定方案：Celery / RQ / Dramatiq

如果文档可能较大，或者后续会接 PDF、Word、Excel，建议直接采用消息队列 + Worker 的异步任务模式。

### 4.7 文件存储建议

后端需要明确“文档地址”保存策略，否则第 2 条需求不完整。

建议方案：

1. 开发环境可先存本地目录，例如 `data/uploads/{kb_id}/{document_id}/...`
2. 生产环境建议使用对象存储，例如 MinIO、OSS、S3
3. MySQL 中保存 `storage_path` 和 `file_url`

### 4.8 检索链路改造建议

当前 QA 检索链路需要支持按知识库范围查询。

调用链需要变成：

1. 会话上下文或请求参数中带 `knowledge_base_id`
2. 检索层只搜索对应知识库
3. 引用返回时附带 `document_id`、`knowledge_base_id`、`source`、`chunk_id`

否则即使向量存储中做了分库，问答阶段也无法保证真正隔离。

## 5. 建议的实施顺序

建议分三期落地，避免一次性改动过大。

### 第一期：补齐最小可用后端闭环

1. 新增 `knowledge_base` 表
2. 扩展 `knowledge_document`、`knowledge_chunk`、`ingestion_job`
3. 新增文档上传接口
4. 上传后保存文件到本地目录
5. 使用异步任务完成单文档切片和向量化
6. Qdrant 先采用“一知识库一 collection”

### 第二期：补齐检索与运维能力

1. 检索接口支持按 `knowledge_base_id` 过滤
2. 支持单文档重建
3. 支持文档删除
4. 支持查看导入任务进度和失败原因

### 第三期：增强生产可用性

1. 接入对象存储
2. 支持 PDF、DOCX、XLSX 解析
3. 支持文件去重和版本管理
4. 支持权限控制和多租户隔离

## 6. 验收标准

后端完成后，至少应满足以下验收条件：

1. 前端上传一个文档后，MySQL 中能看到知识库记录、文档记录、切片记录、任务记录。
2. 上传时选择的切片策略和向量模型，能够在任务结果和文档元数据中查到。
3. 不同知识库上传同名文档后，检索结果不会串库。
4. 删除文档后，MySQL 记录和向量库数据都能同步删除。
5. 检索回答返回的引用能够定位到原始文档和对应知识库。

## 7. 最终判断

这份需求文档原始版本并不完善，主要问题是：

1. 只描述了目标，没有说明当前实现状态。
2. 没有区分“已实现”和“未实现”。
3. 没有给出后端接口、表结构、任务流和隔离方案。
4. 没有说明如何基于当前仓库代码演进。

现在这份文档补齐后，可以作为后端实施方案的基础版本使用。

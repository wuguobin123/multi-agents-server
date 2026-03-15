# 向量模型接入计划 v1

## 1. 目标

将项目当前 RAG 链路中的 `MockEmbedder` 替换为可配置的真实向量模型能力，并保持以下目标：

1. 本地开发可继续使用 mock 或低成本配置启动。
2. 生产环境可切换到真实 embedding provider。
3. 切换模型后可以安全重建索引，避免维度不一致或旧索引污染检索结果。
4. 现有 QA 链路、Qdrant 链路、启动脚本和部署方式保持可用。

## 2. 当前现状

基于当前代码，向量检索链路已经存在，但 embedding 仍是本地 mock 实现：

1. `app/rag/service.py`
   - 直接实例化 `MockEmbedder`。
   - `rebuild()` 时一次性对所有 chunk 执行 embedding。
2. `app/rag/retriever.py`
   - 检索时同样依赖 `MockEmbedder` 生成 query vector。
3. `app/config/settings.py`
   - 仅有 `rag.embedding_dimension`，没有独立的 embedding provider/model 配置。
4. `scripts/build_kb.py`
   - 直接复用 `RAGService.rebuild()`，没有批量、重试和失败恢复策略。
5. `scripts/bootstrap_runtime.py`
   - 只在向量库为空时自动重建，无法识别“模型已切换但索引未重建”的情况。

这意味着“引入向量模型”不是只改一个模型名，而是需要同时改造 embedding 抽象、配置层、索引生命周期和验证流程。

## 3. 本次范围

### 3.1 必须完成

1. 新增独立的 embedding 抽象层。
2. 接入至少一个真实 embedding provider。
3. 支持通过配置切换 mock 与真实 embedding。
4. 支持向量维度校验和索引重建。
5. 更新 `build_kb`、启动引导、测试和部署文档。

### 3.2 暂不做

1. 多 embedding provider 的在线热切换。
2. 增量索引更新调度平台。
3. rerank 模型。
4. 混合检索和多路召回。

## 4. 目标方案

### 4.1 模块拆分

新增或调整以下职责：

1. `app/rag/embedder.py`
   - 定义统一 `Embedder` 接口。
   - 保留 `MockEmbedder`。
   - 新增真实 provider 的 embedder 实现。

2. `app/rag/factory.py` 或现有 `app/rag/embedder.py`
   - 根据配置构建 embedder。

3. `app/config/settings.py`
   - 新增 embedding 专属配置，而不是复用 chat model 配置。

4. `app/rag/service.py`
   - 通过工厂创建 embedder。
   - 在重建前校验向量维度与索引元数据。
   - 将当前 embedding 模型信息写入向量库元数据或本地 manifest。

5. `scripts/build_kb.py`
   - 支持显式重建。
   - 输出 embedding provider、model、dimension、chunk 数量。

### 4.2 接口建议

建议新增统一接口：

```python
class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def fingerprint(self) -> str: ...
```

说明：

1. `dimension`
   - 用于校验索引向量维度。
2. `fingerprint()`
   - 返回 `provider:model:dimension` 之类的稳定标识。
   - 用于判断当前索引是否需要重建。

## 5. 配置设计

### 5.1 新增配置项

建议在 `rag` 下新增 embedding 配置，避免与 chat model 混用：

```yaml
rag:
  enabled: true
  vector_store_backend: local
  collection_name: knowledge_chunks
  embedding_provider: mock
  embedding_model: mock-embedding
  embedding_dimension: 256
  embedding_api_key:
  embedding_base_url:
  embedding_batch_size: 32
  embedding_timeout_seconds: 20
  embedding_max_retries: 2
  bootstrap_on_startup: true
```

### 5.2 环境变量建议

新增：

```env
EMBEDDING_PROVIDER=mock
EMBEDDING_MODEL=mock-embedding
EMBEDDING_API_KEY=
EMBEDDING_BASE_URL=
EMBEDDING_DIMENSION=1024
EMBEDDING_BATCH_SIZE=32
EMBEDDING_TIMEOUT_SECONDS=20
EMBEDDING_MAX_RETRIES=2
```

要求：

1. 不在代码或文档中写入明文密钥。
2. `.env.example`、Docker、README、部署文档同步更新。
3. 如果 embedding provider 使用 OpenAI-compatible 接口，配置命名仍保持 provider-neutral。

## 6. Provider 接入方案

### 6.1 第一阶段建议

第一阶段建议支持：

1. `mock`
   - 本地测试与离线开发使用。
2. `dashscope_openai_compatible`
   - 用于接入阿里百炼兼容接口 embedding 模型。

### 6.2 Provider 实现要求

真实 provider 需要满足：

1. 支持 document embedding 和 query embedding。
2. 支持超时控制。
3. 支持有限重试。
4. 对外返回统一异常，避免把第三方 SDK 错误直接暴露到上层。
5. 记录 provider、model、耗时、batch 大小日志。

## 7. 索引与迁移策略

这是本次计划必须补齐的重点。

### 7.1 索引元数据

向量库或本地文件中至少记录：

1. `embedding_provider`
2. `embedding_model`
3. `embedding_dimension`
4. `embedding_fingerprint`
5. `indexed_at`
6. `chunk_count`

### 7.2 触发重建条件

以下任一条件满足时必须重建：

1. 向量库为空。
2. 当前配置的 `embedding_fingerprint` 与索引元数据不一致。
3. 当前 embedder 的 `dimension` 与向量库 collection 配置不一致。
4. 显式设置 `FORCE_KB_REBUILD=true`。

### 7.3 启动阶段行为

`scripts/bootstrap_runtime.py` 需要从“只看 point_count”升级为：

1. 读取当前索引元数据。
2. 比较 fingerprint 和 dimension。
3. 不一致则自动重建或明确失败退出。

### 7.4 回滚策略

如果新模型接入后检索质量下降，需要支持：

1. 切回旧 embedding 配置。
2. 重新执行 `build_kb.py`。
3. 使用旧索引恢复服务。

前提是旧模型配置和索引元数据已记录清楚。

## 8. 构建流程改造

### 8.1 `build_kb.py`

需要补充：

1. 输出当前 embedding provider/model/dimension。
2. 支持分批 embedding，避免一次性请求过大。
3. 失败时打印失败批次和异常原因。
4. 成功后打印 fingerprint 和总 chunk 数。

### 8.2 `RAGService.rebuild()`

建议改造点：

1. `embed_documents()` 按 batch 调用。
2. 向量写入前先准备索引元数据。
3. 写入完成后再更新“索引已完成”状态，避免半成品状态被当成可用索引。
4. 如果使用 Qdrant，需要在 collection 维度不匹配时重建 collection。

## 9. 测试与验证

### 9.1 单元测试

至少新增：

1. embedding factory 根据配置返回正确实现。
2. 缺少 API Key 或 base URL 时回退或报错符合预期。
3. `fingerprint` 变化会触发重建判断。
4. 维度不一致时会阻止错误检索。

### 9.2 集成测试

至少覆盖：

1. `build_kb.py` 在 mock embedding 下成功构建。
2. local vector store 能携带并读取索引元数据。
3. Qdrant backend 在 collection 维度不一致时能触发重建策略。
4. 服务启动时遇到索引模型不一致，能正确重建或给出明确错误。

### 9.3 验证文档

`dev-validate-v1.md` 需要补充：

1. mock embedding 验证步骤。
2. 真实 embedding provider 配置后的构建验证。
3. 切换 embedding 模型后的强制重建验证。
4. Qdrant 模式下的健康检查和索引检查。

## 10. 部署变更

需要同步更新以下内容：

1. `.env.example`
2. `README.md`
3. `docs/deploy.md`
4. `docker-compose.yml`
5. 腾讯云部署说明

部署文档必须明确：

1. embedding 与 chat model 可以使用不同 provider。
2. 首次部署或切换 embedding 模型后必须重建知识库。
3. 若生产使用 Qdrant，需要预估向量维度和存储规模。

## 11. 风险与控制措施

### 11.1 主要风险

1. 模型切换后旧索引继续被使用，导致召回结果失真。
2. 真实 provider 接口限流，导致构建失败。
3. 向量维度配置错误，导致 Qdrant collection 不兼容。
4. 本地开发和生产配置不一致，问题只在上线后暴露。

### 11.2 控制措施

1. 用 fingerprint 强制校验索引兼容性。
2. 对 embedding 请求增加超时、重试、批量限制。
3. 在启动日志与 `build_kb.py` 输出中打印 provider/model/dimension。
4. 在 CI 或本地验证脚本中覆盖一次完整知识库重建。

## 12. 分阶段实施

### Phase 1: 抽象与配置

1. 新增 `Embedder` 接口。
2. 引入 embedding 配置项和 env 映射。
3. 保持 mock 为默认实现，保证现有测试不立即失效。

### Phase 2: 真实 Provider

1. 接入 DashScope 兼容 embedding 实现。
2. 打通 `build_kb.py` 和检索链路。
3. 增加 provider 相关测试。

### Phase 3: 索引治理

1. 增加 fingerprint/metadata。
2. 完成启动时一致性检查。
3. 完成 local/Qdrant 的重建策略。

### Phase 4: 验证与部署

1. 更新验证文档。
2. 更新部署文档和示例配置。
3. 完成本地与 Docker 冒烟测试。

## 13. 验收标准

满足以下条件即视为本次接入完成：

1. 系统可以通过配置在 `mock` 与真实 embedding provider 间切换。
2. `build_kb.py` 可成功使用真实 embedding 模型构建索引。
3. 检索时 query embedding 与 document embedding 使用同一配置。
4. embedding 模型或维度变化时，系统能够识别并触发重建。
5. local vector store 和 Qdrant 模式下都能完成最小验证。
6. README、部署文档、验证文档全部同步更新。

## 14. 实施前的立即动作

1. 确认并轮换任何已写入文件历史的真实 API Key。
2. 先完成文档评审，再开始代码改造。
3. 代码改造顺序按“配置 -> embedder -> rebuild -> 测试 -> 部署文档”执行，避免中间状态不可运行。

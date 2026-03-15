# 验证规划

## 目标

启动项目，并通过 `curl` 验证核心链路，确认服务可用、基础问答链路可用、工具链路可用，能够完成一次最小闭环验证。

## 验证范围

本次验证覆盖以下内容：

1. 服务可以正常启动。
2. 健康检查接口可访问。
3. `/v1/chat` 可以完成一次 QA 请求。
4. `/v1/chat` 可以完成一次 Tool 请求。
5. 自动化测试可覆盖最小核心流程。

本次验证暂不覆盖以下内容：

1. 腾讯云部署验证。
2. 性能压测。
3. 多轮对话稳定性验证。
4. 真实外部 MCP 服务联调。

## 前置条件

开始前需要确认：

1. 已安装 `uv`。
2. 当前目录为项目根目录。
3. 已安装项目依赖。
4. 如需验证 QA/RAG 链路，已构建知识库索引。

建议先用 `mock` embedding 完成一轮本地验证，再切换真实 provider 验证重建逻辑。

## 验证步骤

### 1. 安装依赖

```bash
uv sync --extra dev
```

预期结果：

1. 依赖安装完成，无报错。

### 2. 使用 mock embedding 构建知识库

```bash
uv run python scripts/build_kb.py
```

预期结果：

1. 知识库索引生成成功。
2. 输出中包含 `provider/model/dimension/fingerprint`。
3. 本地生成 manifest 文件。
2. 后续 QA 请求可以返回引用内容。

### 3. 切换真实 embedding provider 并重建

示例：

```bash
export EMBEDDING_PROVIDER=dashscope_openai_compatible
export EMBEDDING_MODEL=text-embedding-v4
export EMBEDDING_API_KEY=your_api_key
export EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
uv run python scripts/build_kb.py
```

预期结果：

1. 知识库索引重建成功。
2. 输出中的 `fingerprint` 与 mock 阶段不同。
3. 未出现维度不匹配或 provider 配置缺失错误。

### 4. 验证切换模型后的强制重建

```bash
FORCE_KB_REBUILD=true uv run python scripts/bootstrap_runtime.py
```

预期结果：

1. 控制台输出 `knowledge base initialized`。
2. 输出中包含触发原因，例如 `force_rebuild` 或 `embedding_fingerprint_changed`。

### 5. 启动服务

```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

预期结果：

1. 服务启动成功。
2. 监听地址为 `http://127.0.0.1:8080`。
3. 控制台无启动异常。

### 6. 健康检查

```bash
curl http://127.0.0.1:8080/healthz
```

预期结果：

```json
{"status":"ok"}
```

判定标准：

1. HTTP 状态码为 `200`。
2. 返回体包含 `status=ok`。

补充检查：

```bash
curl http://127.0.0.1:8080/readyz
```

预期结果：

1. HTTP 状态码为 `200`。
2. `checks.rag.current_embedding` 存在。
3. `checks.rag.needs_rebuild` 为 `false`。

### 7. 验证 QA 链路

```bash
curl -X POST http://127.0.0.1:8080/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"帮我总结知识库中关于部署流程的说明","session_id":"validate-qa-001"}'
```

预期结果：

1. HTTP 状态码为 `200`。
2. 返回 JSON 中 `error` 为 `null`。
3. `trace.plan.intent` 为 `qa`。
4. `citations` 非空。
5. `answer` 有实际内容，不是空字符串。

### 8. 验证 Tool 链路

```bash
curl -X POST http://127.0.0.1:8080/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"请调用工具执行一次检查","session_id":"validate-tool-001"}'
```

预期结果：

1. HTTP 状态码为 `200`。
2. `trace.plan.intent` 为 `tool`。
3. `trace.tool_calls` 非空。
4. `answer` 中体现工具执行结果。

### 9. Qdrant 模式下验证索引检查

```bash
export RAG_VECTOR_STORE_BACKEND=qdrant
export QDRANT_URL=http://127.0.0.1:6333
uv run python scripts/build_kb.py
curl http://127.0.0.1:8080/readyz
```

预期结果：

1. `build_kb.py` 成功完成。
2. `readyz` 中 `checks.rag.backend` 为 `qdrant`。
3. `checks.rag.vector_size` 与当前 embedding 维度一致。

### 10. 执行自动化测试

```bash
uv run pytest tests/unit tests/integration
```

预期结果：

1. 单元测试通过。
2. 集成测试通过。
3. 至少覆盖健康检查、QA 链路、Tool 链路。

## 通过标准

满足以下条件即可判定本次验证通过：

1. 服务可以本地正常启动。
2. 健康检查接口返回正常。
3. QA 请求返回有效答案和引用。
4. Tool 请求触发工具调用并返回结果。
5. embedding 切换后能触发重建。
6. 自动化测试全部通过。

## 失败排查

如验证失败，按以下顺序排查：

1. 检查依赖是否已完整安装。
2. 检查知识库索引是否已生成。
3. 检查服务启动日志是否有异常。
4. 检查请求 JSON 字段是否符合接口要求。
5. 检查配置文件和环境变量是否正确。

## 输出物

验证结束后应保留以下结果：

1. 启动命令和启动日志。
2. `curl` 请求与响应示例。
3. 自动化测试执行结果。
4. 验证是否通过的结论。

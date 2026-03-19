# Multi Agents Server

基于 FastAPI、LangGraph 和 LangChain 风格组件实现的多 Agent 问答助手 MVP。

## 快速开始

```bash
uv sync --extra dev
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8080
```

默认配置会把运行轨迹写入本地 SQLite：`data/runtime.db`，RAG 向量数据写入本地文件向量仓：`data/vector_store.json`。
RAG embedding 配置与聊天模型配置已经拆分，默认使用 `mock` embedding；切换真实 embedding provider 后需要重新构建知识库。

初始化数据库结构：

```bash
uv run python scripts/init_db.py
```

索引知识库：

```bash
uv run python scripts/build_kb.py
```

脚本会打印当前 embedding 的 `provider/model/dimension/fingerprint`，并在成功后写入本地 manifest。

健康检查：

```bash
curl http://127.0.0.1:8080/healthz
```

聊天接口：

```bash
curl -X POST http://127.0.0.1:8080/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"帮我总结知识库里关于部署流程的说明","session_id":"demo-001"}'
```

Docker 开发环境会同时启动应用、MySQL 和 Qdrant：

```bash
cp .env.example .env
make rebuild
make smoke
```

`.env.example` 已经包含 Docker 本地开发所需的默认变量，包括：

1. `mock` 模型配置
2. `mock` embedding 配置
3. MySQL 连接串
4. Qdrant 地址
5. 启动时自动导入知识库的开关

常用命令：

```bash
make up
make rebuild
make down
make ps
make logs
make smoke
```

如果需要 MySQL Web 管理界面，可以额外启动 Adminer：

```bash
make devtools-up
```

默认访问地址为 `http://127.0.0.1:18080`，服务地址填 `mysql`。

容器启动时会先执行 `scripts/bootstrap_runtime.py`，完成数据库初始化，并在向量库为空时自动导入知识库。
如果 embedding `fingerprint`、维度或 collection 配置发生变化，启动时也会自动触发重建。

## Embedding 配置

RAG embedding 与聊天模型可使用不同 provider。默认配置：

```env
EMBEDDING_PROVIDER=mock
EMBEDDING_MODEL=mock-embedding
EMBEDDING_DIMENSION=256
```

接入 DashScope OpenAI 兼容 embedding 时，设置：

```env
EMBEDDING_PROVIDER=dashscope_openai_compatible
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_API_KEY=your_api_key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

首次部署或切换 embedding 模型后，需要重新执行：

```bash
uv run python scripts/build_kb.py
```

## browser-use 远程浏览器 Tool

项目已内置 `browser_use` skill tool，可让 `tool_agent` 通过 `browser-use` 调用远程 CDP 浏览器执行网页访问、检索和表单类任务。

如果复用当前聊天模型配置作为 browser-use 的 LLM，至少需要：

```bash
export MODEL_PROVIDER=minimax
export MODEL_NAME=MiniMax-M2.5-highspeed
export MINIMAX_API_KEY=your_minimax_key
export MINIMAX_BASE_URL=https://api.minimaxi.com/v1
```

远程浏览器配置：

```bash
export BROWSER_USE_CDP_URL=http://119.45.252.25:9223
export BROWSER_USE_REMOTE_HEALTH_URL=http://119.45.252.25:8787/health
export BROWSER_USE_REMOTE_INFO_URL=http://119.45.252.25:8787/v1/browser/info
export BROWSER_USE_REMOTE_BEARER_TOKEN=your_remote_browser_token
```

可选配置：

```bash
export BROWSER_USE_ALLOWED_DOMAINS=example.com,docs.browser-use.com
export BROWSER_USE_MAX_STEPS=6
export BROWSER_USE_MAX_FAILURES=1
export BROWSER_USE_RETRY_DELAY=2
export BROWSER_USE_MAX_ACTIONS_PER_STEP=4
export BROWSER_USE_USE_VISION=false
export BROWSER_USE_DISABLE_ENV_PROXY=true
```

说明：
`BROWSER_USE_MAX_STEPS` 是基础默认值。对于登录、发布、上传、表单填写等复杂任务，系统会按任务复杂度自动提高步数预算，并在需要时自动开启视觉模式。

默认配置已按这套云端浏览器写入 `browser_use` tool metadata。实现上会优先使用显式配置的 `BROWSER_USE_CDP_URL` / `cdp_url`，只有未提供时才回退到 `/v1/browser/info` 返回的 `cdp_ws_url`。

当用户请求里包含 `浏览器`、`网页`、`网站`、`browser-use` 等词时，planner 会优先走 `tool_agent`，并由 `browser_use` tool 执行。

### 异步 Browser Task + SSE

对于 `browser-use` 这类长任务，推荐直接走异步任务接口，而不是同步等待 `/v1/chat` 返回。

创建任务：

```bash
curl -sS -X POST http://127.0.0.1:8080/v1/browser/tasks \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "请使用浏览器打开 https://example.com ，只返回页面标题。",
    "session_id": "browser-task-demo-001"
  }'
```

订阅事件流：

```bash
curl -N http://127.0.0.1:8080/v1/browser/tasks/<task_id>/events
```

查询最终状态：

```bash
curl -sS http://127.0.0.1:8080/v1/browser/tasks/<task_id>
```

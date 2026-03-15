# 本机 Docker 部署验证方案

## 目标

在本机 Docker 环境中完成一次最小部署闭环，确认镜像可以构建、容器可以启动、HTTP 接口可访问，且 `/v1/chat` 能完成至少一次真实业务请求。

本方案面向开发阶段冒烟验证，不覆盖生产级高可用、Nginx 反向代理、监控接入和腾讯云部署细节。

## 结论先行

当前仓库没有单独的“部署脚本”，实际部署入口是：

```bash
docker compose up -d --build
```

因此本次验证应基于 `docker-compose.yml` 执行，而不是等待额外脚本。

## 验证范围

本次验证覆盖以下内容：

1. Docker 镜像可以成功构建。
2. 容器可以持续运行，不会启动即退出。
3. `GET /healthz` 返回正常。
4. `POST /v1/chat` 能返回有效结果。
5. 基础日志和容器状态可用于失败排查。

本次验证不覆盖以下内容：

1. 腾讯云服务器部署。
2. Nginx 反向代理配置。
3. 外部 MCP 服务联调。
4. 真实模型调用的稳定性和限流表现。

## 前置条件

开始前需要确认：

1. 本机已安装 Docker Engine。
2. 本机已安装 Docker Compose Plugin，且可使用 `docker compose`。
3. 当前目录为项目根目录。
4. 已准备 `.env` 文件。

准备 `.env`：

```bash
cp .env.example .env
```

开发阶段建议先使用默认的 `mock` provider，避免把“容器能否跑起来”和“外部模型服务是否可用”混在一起验证。

如果只是验证本机 Docker 部署闭环，不需要先配置真实的 `MINIMAX_API_KEY`。
同理，建议先保持 `EMBEDDING_PROVIDER=mock`，把部署问题和真实 embedding 调用问题分开验证。

## 部署步骤

### 1. 检查配置

确认 `.env` 至少包含以下关键项：

```env
APP_CONFIG_PATH=configs/app.yaml
MODEL_PROVIDER=mock
MODEL_NAME=mock-chat
EMBEDDING_PROVIDER=mock
EMBEDDING_MODEL=mock-embedding
EMBEDDING_DIMENSION=256
DATABASE_ENABLED=true
DATABASE_URL=mysql+pymysql://multi_agents:multi_agents@mysql:3306/multi_agents
DATABASE_AUTO_INIT=true
RAG_VECTOR_STORE_BACKEND=qdrant
QDRANT_URL=http://qdrant:6333
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
LOG_LEVEL=INFO
```

确认 `configs/app.yaml` 保持默认即可，当前默认配置已经包含：

1. `mock` 模型。
2. `mock` embedding。
3. 本地知识库目录 `data/knowledge_base`。
4. 一个可用于工具链路验证的 `local_echo` skill tool。
5. SQL 持久化默认配置。
6. Qdrant 向量检索配置。

### 2. 构建并启动容器

执行：

```bash
make rebuild
```

预期结果：

1. 镜像构建成功。
2. `mysql`、`qdrant` 和 `multi-agents-server` 容器启动成功。
3. `mysql`、`qdrant` 先通过健康检查，应用再启动。
4. 端口 `8080`、`6333` 映射到宿主机。

### 3. 检查容器状态

执行：

```bash
make ps
```

预期结果：

1. `multi-agents-server` 状态为 `running`。
2. `mysql` 状态为 `running (healthy)`。
3. `qdrant` 状态为 `running (healthy)`。
4. 没有出现持续重启或退出。

如果容器没有保持运行，立即查看日志：

```bash
make logs
```

### 4. 执行健康检查

执行：

```bash
curl http://127.0.0.1:8080/healthz
```

预期结果：

```json
{"status":"ok"}
```

说明：

`/healthz` 只能证明 FastAPI 进程已启动，不能单独作为“部署成功”的最终依据。

建议补充 readiness 检查：

```bash
curl http://127.0.0.1:8080/readyz
```

预期结果中 `checks.repository.backend` 为 `sql`，`checks.repository.dialect` 为 `mysql`，`checks.rag.backend` 为 `qdrant`。
同时确认 `checks.rag.needs_rebuild=false`，并且 `checks.rag.current_embedding.dimension` 与 `checks.rag.vector_size` 一致。

### 5. 执行业务链路检查

推荐直接执行仓库内置的冒烟脚本：

```bash
make smoke
```

脚本会自动完成：

1. `/healthz`
2. `/readyz`
3. QA 请求
4. Tool 请求
5. RAG point_count 检查

如果需要手工验证，也可以继续执行下面两条请求。

执行一次 QA 请求：

```bash
curl -X POST http://127.0.0.1:8080/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"帮我总结知识库中关于部署流程的说明","session_id":"deploy-check-qa-001"}'
```

预期结果：

1. HTTP 状态码为 `200`。
2. 返回 JSON 中 `error` 为 `null`。
3. `trace.plan.intent` 为 `qa`。
4. `answer` 非空。
5. `citations` 非空。

建议再执行一次 Tool 请求，确认运行时初始化和工具注册也正常：

```bash
curl -X POST http://127.0.0.1:8080/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"query":"请调用工具执行一次检查","session_id":"deploy-check-tool-001"}'
```

预期结果：

1. HTTP 状态码为 `200`。
2. 返回 JSON 中 `error` 为 `null`。
3. `trace.plan.intent` 为 `tool`。
4. `trace.tool_calls` 非空。
5. `answer` 中体现工具执行结果。

### 6. 切换真实 embedding provider 的部署验证

如果生产需要使用真实 embedding provider，补充以下变量：

```env
EMBEDDING_PROVIDER=dashscope_openai_compatible
EMBEDDING_MODEL=text-embedding-v4
EMBEDDING_API_KEY=your_api_key
EMBEDDING_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

变更后必须重建知识库：

```bash
docker compose exec multi-agents-server python scripts/build_kb.py
```

重建完成后再次执行：

```bash
curl http://127.0.0.1:8080/readyz
```

预期结果：

1. `checks.rag.needs_rebuild` 为 `false`。
2. `checks.rag.index_metadata.embedding_fingerprint` 与当前配置一致。

## 通过标准

满足以下条件即可判定本次本机 Docker 部署验证通过：

1. `docker compose up -d --build` 成功完成。
2. `docker compose ps` 显示容器处于运行状态。
3. `/healthz` 返回 `{"status":"ok"}`。
4. `/readyz` 返回数据库已就绪。
5. `/readyz` 显示当前 RAG 索引不需要重建。
6. `/v1/chat` 的 QA 请求返回有效答案和引用。
7. `/v1/chat` 的 Tool 请求可以触发至少一次工具调用。

## 失败排查

如果部署失败，按以下顺序排查：

1. 检查 `.env` 是否存在，以及 `docker-compose.yml` 能否正确读取它。
2. 检查 `docker compose logs --tail=200 multi-agents-server` 是否有导入错误、依赖安装失败或配置错误。
3. 检查 `8080` 端口是否已被本机其他进程占用。
4. 检查 `data/knowledge_base/index.json` 是否存在，避免 QA 验证因知识库为空而失真。
5. 如果切换到 `minimax`，检查 `MINIMAX_API_KEY` 和 `MINIMAX_BASE_URL` 是否正确。
6. 如果切换了 embedding provider，检查 `EMBEDDING_*` 变量是否完整，并确认已执行过知识库重建。

## 回收与重试

停止并删除容器：

```bash
make down
```

需要重新构建后重试时，重新执行：

```bash
make rebuild
```

## 开发辅助

如果需要临时查看 MySQL 中的数据，可以启动 Adminer：

```bash
make devtools-up
```

访问地址：

```text
http://127.0.0.1:18080
```

登录时服务器填 `mysql`，用户名和密码使用 compose 中的默认值。

## 输出物

验证结束后建议保留以下内容：

1. `docker compose ps` 输出结果。
2. `docker compose logs --tail=200 multi-agents-server` 关键日志。
3. `/healthz` 请求结果。
4. 两次 `/v1/chat` 请求与响应样例。
5. 最终结论：通过或不通过，以及失败原因。

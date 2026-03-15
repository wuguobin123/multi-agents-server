# 部署说明

## 运行环境

- 腾讯云 CVM，建议 `2C4G` 起步
- Docker Engine
- Docker Compose Plugin
- Nginx

## 部署步骤

1. 复制 `.env.example` 为 `.env`，通过环境变量注入模型密钥。
2. 根据业务需要调整 `configs/app.yaml`，确认模型、Agent、Tool 和 RAG 参数。
3. 执行 `docker compose up -d --build`。
4. 检查 `GET /healthz` 是否返回 `{"status":"ok"}`。
5. 通过 Nginx 反向代理 `8080` 端口，并保留 `/healthz` 与 `/v1/chat`。

## 观测与维护

- 服务日志默认输出为 JSON，适合接入腾讯云日志服务。
- 更新知识库后执行 `uv run python scripts/build_kb.py` 重建清单。
- 若切换到 MiniMax，请确保设置 `MODEL_PROVIDER=minimax`、`MINIMAX_BASE_URL` 和 `MINIMAX_API_KEY`。

.PHONY: up rebuild down logs ps smoke devtools-up devtools-down

up:
	docker compose up -d

rebuild:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f multi-agents-server

ps:
	docker compose ps

smoke:
	python scripts/smoke_docker.py

devtools-up:
	docker compose --profile devtools up -d adminer

devtools-down:
	docker compose --profile devtools stop adminer

SHELL := bash
COMPOSE := docker compose -f deploy/docker-compose.yml
COMPOSE_GPU := docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.gpu.yml
COMPOSE_HETERO := docker compose -f deploy/docker-compose.yml -f deploy/docker-compose.heterogeneous.yml

.PHONY: help up down logs ps test test-unit test-integration bench bench-quick \
        bench-gpu bench-hetero bench-batching gpu-up gpu-down hetero-up hetero-down clean

help:                 ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  %-20s %s\n", $$1, $$2}'

up:                   ## Build + start the CPU compose stack (8 containers)
	$(COMPOSE) up -d --build

down:                 ## Stop and remove the CPU compose stack
	$(COMPOSE) down

logs:                 ## Tail logs from every service
	$(COMPOSE) logs -f --tail=100

ps:                   ## List running compose containers
	$(COMPOSE) ps

test: test-unit       ## Run the unit suite (CPU)

test-unit:            ## Run unit tests only
	pytest tests/unit -v

test-integration:     ## Run integration tests (requires `make up`)
	pytest tests/integration -v

bench:                ## Full benchmark: 4 strategies x 4 user counts + fault run
	python scripts/benchmark.py

bench-quick:          ## Tiny benchmark: load_aware at 50 + 200 users, no fault
	python scripts/benchmark.py --quick

bench-batching:       ## Sim vs continuous batching head-to-head at 500 users
	python scripts/benchmark.py --no-fault --strategies load_aware --user-counts 500 --compare-backends

bench-gpu:            ## GPU-mode benchmark (assumes `make gpu-up` already ran)
	python scripts/benchmark.py --mode gpu --strategies round_robin --user-counts 50,100,250

bench-hetero:         ## Heterogeneous-worker benchmark across all 4 strategies
	python scripts/heterogeneous_bench.py

gpu-up:               ## Build + start the GPU stack (workers on CUDA, distilgpt2)
	$(COMPOSE_GPU) up -d --build

gpu-down:             ## Stop the GPU stack
	$(COMPOSE_GPU) down

gpu-smoke:            ## Send one real distilgpt2 inference and print the answer
	python scripts/gpu_smoke.py

hetero-up:            ## Restart workers with heterogeneous capacity (1:2:8)
	$(COMPOSE_HETERO) up -d --force-recreate

hetero-down:          ## Stop the heterogeneous stack
	$(COMPOSE_HETERO) down

clean:                ## Stop everything + remove benchmark raw outputs
	-$(COMPOSE) down
	-$(COMPOSE_GPU) down
	-$(COMPOSE_HETERO) down
	rm -rf benchmarks/raw

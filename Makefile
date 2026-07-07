.PHONY: build vet test run up down logs demo calibrate clean

build:
	go build ./...

vet:
	go vet ./...

test:
	go test ./...

run:
	go run ./cmd/proxy

# Cross-lingual demo: a Turkish prompt served from a Spanish-seeded cache entry.
# Requires the stack to be up (make up) with a real PROVIDER_API_KEY.
demo:
	bash eval/demo/crosslingual_hit.sh

# Per-language-pair threshold calibration against the real BGE-M3 sidecar.
# Requires the sidecar up (docker compose up -d sidecar).
calibrate:
	go run ./eval/calibration

up:
	docker compose up --build

down:
	docker compose down

logs:
	docker compose logs -f proxy

clean:
	docker compose down -v

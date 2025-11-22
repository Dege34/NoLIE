# Deepfake Forensics Makefile

.PHONY: help install install-dev test lint format clean build docker run-docker

# Default target
help:
	@echo "Available targets:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install the package with development dependencies"
	@echo "  test         Run tests"
	@echo "  lint         Run linting"
	@echo "  format       Format code"
	@echo "  clean        Clean build artifacts"
	@echo "  build        Build the package"
	@echo "  docker       Build Docker image"
	@echo "  run-docker   Run Docker container"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

# Testing
test:
	python -m deepfake_forensics.cli test

test-coverage:
	python -m deepfake_forensics.cli test --coverage

# Code quality
lint:
	python -m deepfake_forensics.cli lint

format:
	python -m deepfake_forensics.cli format

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Build
build: clean
	python -m build

# Docker
docker:
	docker build -t deepfake-forensics .

run-docker:
	docker run -p 8000:8000 deepfake-forensics

# Development
dev-setup: install-dev
	pre-commit install

# Data preparation
prepare-data:
	python scripts/prepare_data.py --src data/raw --out data/processed --fps 8

# Training
train:
	python -m deepfake_forensics.cli train --config configs/train_small.yaml --data-dir data/processed --output-dir outputs

# Inference
predict:
	python -m deepfake_forensics.cli predict --input demo.mp4 --model checkpoints/best.ckpt --output results.json

# API
serve:
	python -m deepfake_forensics.cli serve --model checkpoints/best.ckpt --host 0.0.0.0 --port 8000

# Export
export:
	python -m deepfake_forensics.cli export --model checkpoints/best.ckpt --output model.pt --format torchscript

# Web UI
web-install:
	cd web && npm install

web-dev:
	cd web && npm run dev

web-build:
	cd web && npm run build

web-preview:
	cd web && npm run preview

web-test:
	cd web && npm run test

web-lint:
	cd web && npm run lint

web-format:
	cd web && npm run format

# Full stack
dev: web-dev
	@echo "Starting development servers..."
	@echo "Web UI: http://localhost:5173"
	@echo "API: http://localhost:8000"

build: web-build
	@echo "Building all components..."

# Docker
docker-web:
	docker build -t deepfake-forensics-web ./web

run-docker-full:
	docker-compose up --build

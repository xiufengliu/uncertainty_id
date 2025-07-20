# Makefile for Uncertainty-Aware Intrusion Detection System

.PHONY: help install install-dev test test-cov lint format clean docs serve-docs build publish docker-build docker-run examples

# Default target
help:
	@echo "Uncertainty-Aware Intrusion Detection System"
	@echo "============================================="
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install the package"
	@echo "  install-dev  Install in development mode with all dependencies"
	@echo "  test         Run tests"
	@echo "  test-cov     Run tests with coverage"
	@echo "  lint         Run linting checks"
	@echo "  format       Format code with black and isort"
	@echo "  clean        Clean build artifacts"
	@echo "  docs         Build documentation"
	@echo "  serve-docs   Serve documentation locally"
	@echo "  build        Build package"
	@echo "  publish      Publish to PyPI"
	@echo "  examples     Run example scripts"
	@echo "  experiments  Run research experiments"
	@echo "  api          Start API server"
	@echo "  train        Train model with CLI"
	@echo ""

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev,docs,test,api,notebooks]"
	pre-commit install

# Testing
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=uncertainty_ids --cov-report=html --cov-report=term

test-fast:
	pytest tests/ -v -m "not slow"

test-integration:
	pytest tests/ -v -m "integration"

test-benchmark:
	pytest tests/ -v -m "benchmark"

# Code quality
lint:
	flake8 uncertainty_ids/ tests/ examples/ scripts/
	mypy uncertainty_ids/

format:
	black uncertainty_ids/ tests/ examples/ scripts/
	isort uncertainty_ids/ tests/ examples/ scripts/

format-check:
	black --check uncertainty_ids/ tests/ examples/ scripts/
	isort --check-only uncertainty_ids/ tests/ examples/ scripts/

# Documentation
docs:
	cd docs && make html

serve-docs:
	cd docs/_build/html && python -m http.server 8080

docs-clean:
	cd docs && make clean

# Build and publish
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	python -m twine upload dist/*

publish-test: build
	python -m twine upload --repository testpypi dist/*

# Examples and experiments
examples:
	@echo "Running quick start example..."
	python examples/quick_start.py

train-nsl-kdd:
	@echo "Training on NSL-KDD dataset..."
	python examples/train_nsl_kdd.py --epochs 50 --batch-size 64

experiments:
	@echo "Running comprehensive experiments..."
	python scripts/run_experiments.py --dataset-size 5000

# API and services
api:
	@echo "Starting API server..."
	python -m uncertainty_ids.api.server

api-dev:
	@echo "Starting API server in development mode..."
	python -m uncertainty_ids.api.server --reload

# CLI tools
train:
	@echo "Usage: make train DATA_PATH=path/to/data.csv [OPTIONS]"
	@echo "Example: make train DATA_PATH=data/nsl-kdd.csv EPOCHS=100"
	uncertainty-ids-train --help

evaluate:
	@echo "Usage: make evaluate MODEL_PATH=path/to/model.pth DATA_PATH=path/to/test.csv"
	uncertainty-ids-evaluate --help

preprocess:
	@echo "Usage: make preprocess DATA_PATH=path/to/data.csv OUTPUT_PATH=path/to/output/"
	uncertainty-ids-preprocess --help

# Development utilities
setup-dev:
	@echo "Setting up development environment..."
	python -m venv venv
	@echo "Activate with: source venv/bin/activate (Linux/Mac) or venv\\Scripts\\activate (Windows)"
	@echo "Then run: make install-dev"

check-deps:
	@echo "Checking dependencies..."
	pip check
	pip list --outdated

security-check:
	@echo "Running security checks..."
	bandit -r uncertainty_ids/
	safety check

# Data and models
download-data:
	@echo "Downloading sample datasets..."
	mkdir -p data
	@echo "Please manually download datasets from:"
	@echo "  - NSL-KDD: https://www.unb.ca/cic/datasets/nsl.html"
	@echo "  - CICIDS2017: https://www.unb.ca/cic/datasets/ids-2017.html"
	@echo "  - UNSW-NB15: https://research.unsw.edu.au/projects/unsw-nb15-dataset"

create-synthetic-data:
	@echo "Creating synthetic dataset..."
	python -c "from uncertainty_ids.data import SyntheticIDSDataset; SyntheticIDSDataset.create_synthetic(10000).save('data/synthetic_ids_data.csv')"

# Monitoring and profiling
profile:
	@echo "Profiling model training..."
	python -m cProfile -o profile_output.prof examples/quick_start.py
	python -c "import pstats; pstats.Stats('profile_output.prof').sort_stats('cumulative').print_stats(20)"

memory-profile:
	@echo "Memory profiling..."
	python -m memory_profiler examples/quick_start.py

# Jupyter notebooks
notebooks:
	@echo "Starting Jupyter Lab..."
	jupyter lab examples/notebooks/

notebook-server:
	@echo "Starting Jupyter notebook server..."
	jupyter notebook examples/notebooks/

# Git hooks and CI
pre-commit:
	pre-commit run --all-files

pre-commit-update:
	pre-commit autoupdate

# Release management
version-patch:
	@echo "Bumping patch version..."
	python scripts/bump_version.py patch

version-minor:
	@echo "Bumping minor version..."
	python scripts/bump_version.py minor

version-major:
	@echo "Bumping major version..."
	python scripts/bump_version.py major

# Environment info
env-info:
	@echo "Environment Information:"
	@echo "========================"
	@echo "Python version: $(shell python --version)"
	@echo "PyTorch version: $(shell python -c 'import torch; print(torch.__version__)')"
	@echo "CUDA available: $(shell python -c 'import torch; print(torch.cuda.is_available())')"
	@echo "GPU count: $(shell python -c 'import torch; print(torch.cuda.device_count())')"
	@echo "Platform: $(shell python -c 'import platform; print(platform.platform())')"

# All-in-one targets
dev-setup: setup-dev install-dev
	@echo "Development environment ready!"

full-test: format-check lint test-cov
	@echo "All tests passed!"

release-check: clean format-check lint test-cov docs build
	@echo "Release checks passed!"

# Help for specific commands
help-train:
	uncertainty-ids-train --help

help-api:
	@echo "API Server Help:"
	@echo "================"
	@echo "Start server: make api"
	@echo "Development mode: make api-dev"
	@echo "API docs: http://localhost:8000/docs"
	@echo "Health check: http://localhost:8000/health"

help-examples:
	@echo "Available Examples:"
	@echo "==================="
	@echo "Quick start: python examples/quick_start.py"
	@echo "NSL-KDD training: python examples/train_nsl_kdd.py"
	@echo "API client: python examples/api_client_example.py"
	@echo "Jupyter notebooks: make notebooks"

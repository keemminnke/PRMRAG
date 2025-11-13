.PHONY: help install test clean example run-pipeline analysis

help:
	@echo "PRMRAG - Consensus-Based Auto-Labeling Pipeline"
	@echo ""
	@echo "Available commands:"
	@echo "  make install          - Install package and dependencies"
	@echo "  make example          - Create example data"
	@echo "  make run-pipeline     - Run full labeling pipeline on example data"
	@echo "  make analysis         - Analyze consensus results"
	@echo "  make test             - Run tests"
	@echo "  make clean            - Clean generated files"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

example:
	python scripts/create_example_data.py

run-pipeline:
	python scripts/label_dataset.py \
		--config configs/default.yaml \
		--input data/raw/example_trajectories.jsonl \
		--output data/labeled/example_labeled.jsonl \
		--output-training-samples data/labeled/example_training_samples.jsonl

analysis:
	python scripts/consensus_analysis.py \
		--input data/labeled/example_labeled.jsonl \
		--output outputs/metrics/consensus_analysis.json

test:
	pytest tests/ -v

clean:
	rm -rf data/processed/* data/labeled/* outputs/*
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

format:
	black src/ scripts/ tests/
	isort src/ scripts/ tests/

lint:
	mypy src/
	black --check src/ scripts/ tests/

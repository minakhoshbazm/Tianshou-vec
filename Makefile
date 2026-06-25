.PHONY: help install install-dev format lint test test-cov clean build docs

help:
	@echo "Available commands:"
	@echo "  make install        - Install package in production mode"
	@echo "  make install-dev    - Install package in development mode with dev tools"
	@echo "  make format         - Format code with black"
	@echo "  make lint           - Run linters (flake8, mypy)"
	@echo "  make test           - Run tests"
	@echo "  make test-cov       - Run tests with coverage report"
	@echo "  make clean          - Remove build artifacts and cache files"
	@echo "  make build          - Build distribution packages"
	@echo "  make docs           - Build documentation"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install pytest pytest-cov black flake8 mypy pylint bandit sphinx sphinx-rtd-theme

format:
	black src/ tests/
	@echo "Code formatted with black"

lint:
	@echo "Running flake8..."
	flake8 src/ tests/ --max-line-length=88 --statistics
	@echo "Running mypy..."
	mypy src/ --ignore-missing-imports
	@echo "Linting complete!"

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "Clean complete!"

build: clean
	python setup.py sdist bdist_wheel
	@echo "Build complete! Packages in dist/"

docs:
	@echo "Documentation building not yet configured. See docs/ directory."

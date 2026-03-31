.PHONY: help install run build test format lint clean

VENV = .venv
PYTHON = $(VENV)/Scripts/python
PIP = $(VENV)/Scripts/pip

help:
	@echo "Axiom Engine Makefile Commands:"
	@echo "  make install  - Create a virtual enviroment and install dependencies."
	@echo "  make run      - Run the FastAPI server in development mode."
	@echo "  make test     - Run the testing suite with pytest."
	@echo "  make format   - Auto-format code with Ruff."
	@echo "  make lint     - Run static analysis with Ruff."
	@echo "  make clean    - Remove cached files and virtual environment."

install:
	python -m venv $(VENV)
	$(PIP) install -r requirements.txt -r requirements-dev.txt
	@echo "Installation complete. Activate via: .venv\Scripts\activate"

run:
	$(PYTHON) -m uvicorn axiom_engine.main:app --reload

test:
	$(PYTHON) -m pytest

format:
	$(PYTHON) -m ruff format src tests
	$(PYTHON) -m ruff check --fix src tests

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

clean:
	if exist $(VENV) rmdir /s /q $(VENV)
	if exist .pytest_cache rmdir /s /q .pytest_cache
	if exist .ruff_cache rmdir /s /q .ruff_cache
	for /d /r . %d in (__pycache__) do @if exist "%d" rmdir /s /q "%d"

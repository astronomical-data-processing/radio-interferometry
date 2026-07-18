.PHONY: all check_python setup_dependencies test

PYTHON ?= python3

all: setup_dependencies

check_python:
	@$(PYTHON) -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required'"

setup_dependencies: check_python
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test: check_python
	$(PYTHON) -m unittest discover -s tests

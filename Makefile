.PHONY: all check_python pull_data setup_dependencies test

PYTHON ?= python3

all: setup_dependencies pull_data

check_python:
	@$(PYTHON) -c "import sys; assert sys.version_info >= (3, 10), 'Python 3.10 or newer is required'"

pull_data:
	wget https://www.dropbox.com/s/n3jyiajytwuldpu/fundamentals_fits.tar.gz?dl=0
	wget https://www.dropbox.com/s/kb3p2mthei8dgl9/simulated_KAT-7_ms.tar.gz?dl=0
	tar -xvzf fundamentals_fits.tar.gz?dl=0 --directory=data/
	tar -xvzf simulated_KAT-7_ms.tar.gz?dl=0 --directory=data/simulated_kat_7_vis
	rm fundamentals_fits.tar.gz?dl=0
	rm simulated_KAT-7_ms.tar.gz?dl=0

setup_dependencies: check_python
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test: check_python
	$(PYTHON) -m unittest discover -s tests

.PHONY: venv install install-plugins list run test docker-build docker-run clean

venv:
	python -m venv .venv

install:
	. .venv/bin/activate && pip install -r requirements.txt && pip install -e .

install-plugins:
	. .venv/bin/activate && pip install -r requirements-plugins.txt

list:
	. .venv/bin/activate && autofs-vnext list-methods

run:
	. .venv/bin/activate && autofs-vnext run --config examples/config_classification.json

test:
	. .venv/bin/activate && python -m autofs_vnext.cli list-methods

docker-build:
	docker build -t autofs-vnext:local .

docker-run:
	docker run --rm -it autofs-vnext:local list-methods

clean:
	rm -rf .venv __pycache__ .pytest_cache dist build *.egg-info autofs_out

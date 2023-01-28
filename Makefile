clean:
	rm -fr .direnv

env:
	pyenv install; \
	direnv allow

venv:
	python3 -m venv /home/mfa51/service-rate/.venv

install:
	pip install --upgrade pip; \
	pip install poetry; \
	poetry install

lint:
	black --exclude=".direnv/*" .
	flake8 --exclude=".direnv/*" .

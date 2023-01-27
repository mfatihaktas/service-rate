clean:
	rm -fr .direnv

env:
	pyenv install; \
	direnv allow
	# PKG_CONFIG_PATH="/home/mfa51/libffi/usr/lib64/pkgconfig:${PKG_CONFIG_PATH}" \
	# CFLAGS="-I/home/mfa51/libffi/usr/include" \
	# LDFLAGS="-L/home/mfa51/libffi/usr/lib64" \
	# pyenv install --verbose 3.10.0

venv:
	python3 -m venv /home/mfa51/service-rate/.venv

install:
	pip install --upgrade pip; \
	pip install poetry; \
	poetry install

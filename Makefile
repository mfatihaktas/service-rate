clean:
	rm -fr .direnv

env:
	## Locate libffi
	# locate libffi
	# rpm -qa libffi
	# pkg-config --libs libffi
	## Check if can link to libffi
	# gcc -L/projects/community/libffi/3.4.2/gc563/lib/../lib64 -lffi
	## Install Python
	CFLAGS="$(pkg-config --cflags libffi)" \
	LDFLAGS="$(pkg-config --libs libffi)" \
	pyenv install; \
	direnv allow
	# Ref:
	# - https://dev.to/ajkerrigan/homebrew-pyenv-ctypes-oh-my-3d9
	# - https://github.com/pyenv/pyenv/issues/1183
venv:
	python3 -m venv /home/mfa51/service-rate/.venv

install:
	# SCIPOPTDIR=/home/mfa51/scip_installation
	export SCIPOPTDIR=/Users/mehmet/Desktop/scip-8.0.3/install
	pip install --upgrade pip; \
	pip install poetry; \
	poetry install

lint:
	# black --exclude=".direnv/*" .
	flake8 --exclude=".direnv/*" .

help:
	@$(MAKE) -C docs help

.PHONY: Makefile build-deps sdist wheel

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(MAKE) -C docs $@

build-deps:
	python -m pip install --upgrade pip setuptools wheel

sdist: build-deps
	python setup.py sdist

wheel: build-deps
	python -m pip wheel --no-deps -w dist/ .

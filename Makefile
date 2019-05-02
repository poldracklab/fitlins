help:
	@$(MAKE) -C docs help

.PHONY: Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(MAKE) -C docs $@

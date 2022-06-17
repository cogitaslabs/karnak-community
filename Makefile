
SRCDIR := $(CURDIR)

build: clean
	cd $(SRCDIR)
	python3 setup.py bdist_wheel

clean:
	rm -rf $(SRCDIR)/build/*
	rm -f $(SRCDIR)/dist/*

tag: VERSION.txt
	cd $(SRCDIR)
	git status
	git tag "v$(shell cat VERSION.txt)"

publish:
	cd $(SRCDIR)
	git status
	python3 -m twine upload dist/*.whl

release: build tag publish

.PHONY: build clean tag publish release
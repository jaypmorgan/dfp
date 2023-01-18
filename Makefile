.PHONY: build
build:
	python3 -m build

.PHONY: test-upload
test-upload:
	python3 -m twine upload --skip-existing --repository testpypi dist/*

.PHONY: upload
upload:
	python3 -m twine upload --skip-existing dist/*

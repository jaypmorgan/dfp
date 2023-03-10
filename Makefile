.PHONY: build
build:
	python3 -m build

.PHONY: test-upload
test-upload:
	python3 -m twine upload --skip-existing --repository testpypi dist/*

.PHONY: upload
upload:
	python3 -m twine upload --skip-existing dist/*

.PHONY: test
test:
	python3 -m unittest tests/test_dfp.py

.PHONY: coverage
coverage:
	coverage run --include=src/dfp.py -m unittest tests/test_dfp.py 
	coverage xml --include=src/dfp.py
	coveralls

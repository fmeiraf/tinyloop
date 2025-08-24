test:
	uv run pytest tests/ -v --disable-warnings

clean:
	rm -rf dist/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +

build: clean
	uv build

publish-test: build
	source .env && uv publish --publish-url https://test.pypi.org/legacy/ --token $$PYPI_TOKEN_TEST

publish: build
	source .env && uv publish --token $$PYPI_TOKEN_PROD
.PHONY: all test clean requirements

OUT ?= outputs

all:
	python -m candy --out $(OUT)

test:
	python -m ruff check src tests
	python -m pytest

clean:
	rm -rf $(OUT)

# Regenerate the pip requirements from the uv lockfile.
requirements:
	uv export --frozen --no-hashes --format requirements-txt -o requirements.txt

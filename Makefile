.PHONY: all test clean

OUT ?= outputs

all:
	python -m candy --out $(OUT)

test:
	python -m pytest

clean:
	rm -rf $(OUT)

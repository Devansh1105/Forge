.PHONY: all test test-kernels test-integration checkstyle serve build clean run-benchmarks

all: checkstyle test

test:
	python -m pytest --disable-warnings \
		--cov=src/forge \
		--cov-report=term-missing \
		test/

test-kernels:
	python -m pytest --disable-warnings test/kernels

test-integration:
	python -m pytest --disable-warnings test/integration

checkstyle:
	ruff check --output-format=concise .; ruff_check_status=$$?; \
	ruff format --check --diff .; ruff_format_status=$$?; \
	ruff check . --fix; \
	ruff format .; \
	if [ $$ruff_check_status -ne 0 ] || [ $$ruff_format_status -ne 0 ]; then \
		exit 1; \
	fi

BENCHMARK_DIR = benchmark/scripts
BENCHMARK_SCRIPTS = $(wildcard $(BENCHMARK_DIR)/benchmark_*.py)

run-benchmarks:
	@for script in $(BENCHMARK_SCRIPTS); do \
		echo "Running benchmark: $$script"; \
		python $$script; \
	done

MKDOCS = mkdocs
CONFIG_FILE = mkdocs.yml
SITE_DIR = site

serve:
	$(MKDOCS) serve -f $(CONFIG_FILE)

build:
	$(MKDOCS) build -f $(CONFIG_FILE) --site-dir $(SITE_DIR)

clean:
	rm -rf $(SITE_DIR)/

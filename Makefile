# Specify the names of all executables to make.

PYTHON_FILES=$(shell find . -type f -name "*.py")
PROG=update install style_check
.PHONY: ${PROG}

style_check:
	black . --config pyproject.toml
	isort . --gitignore --settings-path pyproject.toml
	flake8 ${PYTHON_FILES}

.PHONY: clean
clean: clean-test clean-build clean-pyc clean-docs

.PHONY: clean-build
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	rm -fr pip-wheel-metadata
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -fr {} +
	find . -type d -name __pycache__ -exec rm -rv {} +
	rm -fr Pipfile.lock

.PHONY: clean-docs
clean-docs:
	rm -fr site/

.PHONY: clean-pyc
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
	find . -name '.DS_Store' -exec rm -fr {} +

.PHONY: clean-test
clean-test: clean-cache
	rm -fr .tox/
	rm -f .coverage
	find . -name ".coverage*" -not -name ".coveragerc" -exec rm -fr "{}" \;
	rm -fr coverage.xml
	rm -fr htmlcov/
	find . -name 'log.txt' -exec rm -fr {} +
	find . -name 'log.*.txt' -exec rm -fr {} +
	rm -rf leak_report

# removes various cache files
.PHONY: clean-cache
clean-cache:
	find . -type d -name .hypothesis -prune -exec rm -rf {} \;
	rm -fr .pytest_cache
	rm -fr .mypy_cache/

# isort: fix import orders
# black: format files according to the pep standards
.PHONY: formatters
formatters:
	tomte format-code

# black-check: check code style
# isort-check: check for import order
# flake8: wrapper around various code checks, https://flake8.pycqa.org/en/latest/user/error-codes.html
# mypy: static type checker
# pylint: code analysis for code smells and refactoring suggestions
# vulture: finds dead code
# darglint: docstring linter
.PHONY: code-checks
code-checks:
	tomte check-code

# safety: checks dependencies for known security vulnerabilities
# bandit: security linter
# gitleaks: checks for sensitive information
.PHONY: security
security:
	tomte check-security
	gitleaks detect --report-format json --report-path leak_report

# generate abci docstrings
# update copyright headers
# generate latest hashes for updated packages
.PHONY: generators
generators: clean-cache
	tox -e abci-docstrings
	tomte format-copyright --author valory --exclude-part connections --exclude-part contracts --exclude-part protocols --exclude-part abstract_abci --exclude-part abstract_round_abci --exclude-part registration_abci --exclude-part reset_pause_abci --exclude-part termination_abci --exclude-part transaction_settlement_abci
	autonomy hash all
	autonomy packages lock
	tox -e fix-doc-hashes

.PHONY: common-checks-1
common-checks-1:
	tomte check-copyright --author valory --exclude-part connections --exclude-part contracts --exclude-part protocols --exclude-part abstract_abci --exclude-part abstract_round_abci --exclude-part registration_abci --exclude-part reset_pause_abci --exclude-part termination_abci --exclude-part transaction_settlement_abci
	tomte check-doc-links
	tox -p -e check-hash -e check-packages -e check-doc-hashes

.PHONY: common-checks-2
common-checks-2:
	tox -e check-abci-docstrings
	tox -e check-abciapp-specs
	tox -e check-handlers

.PHONY: test
test:
	pytest -rfE tests/ --cov-report=html --cov=packages.valory.skills.apy_estimation_abci --cov-report=xml --cov-report=term --cov-report=term-missing --cov-config=.coveragerc
	find . -name ".coverage*" -not -name ".coveragerc" -exec rm -fr "{}" \;

.PHONY: all-checks
all-checks: clean formatters code-checks security generators common-checks-1 common-checks-2

v := $(shell pip -V | grep virtualenvs)

.PHONY: new_env
new_env: clean
	if [ ! -z "$(which svn)" ];\
	then\
		echo "The development setup requires SVN, exit";\
		exit 1;\
	fi;\

	if [ -z "$v" ];\
	then\
		pipenv --rm;\
		pipenv --clear;\
		pipenv --python 3.10;\
		pipenv install --dev --skip-lock;\
		echo "Enter virtual environment with all development dependencies now: 'pipenv shell'.";\
	else\
		echo "In a virtual environment! Exit first: 'exit'.";\
	fi

.PHONY: fix-abci-app-specs
fix-abci-app-specs:
	autonomy analyse fsm-specs --update --app-class APYEstimationAbciApp --package packages/valory/skills/apy_estimation_abci || (echo "Failed to check apy_estimation_abci consistency" && exit 1)
	autonomy analyse fsm-specs --update --app-class APYEstimationChainedAbciApp --package packages/valory/skills/apy_estimation_chained_abci || (echo "Failed to check apy_estimation_chained_abci consistency" && exit 1)
	echo "Successfully validated abcis!"

PACKAGES_PATH := packages/packages.json
RELEASE_VERSION := latest
APY_ORACLE_AGENT_NAME := valory/apy_estimation
APY_ORACLE_IMAGE_NAME := valory/oar-apy_estimation
release-image:
	$(eval APY_ORACLE_AGENT_HASH := $(shell cat ${PACKAGES_PATH} | grep "agent/${APY_ORACLE_AGENT_NAME}" | cut -d "\"" -f4 ))
	$(eval APY_ORACLE_AGENT_PUBLIC_ID := ${APY_ORACLE_AGENT_NAME}:${RELEASE_VERSION}:${APY_ORACLE_AGENT_HASH})
	# we first need to push all the packages in order to be able to build the image,
	# because the command pulls the agent from the registry.
	# Please make sure to run this command only from a release branch.
	autonomy push-all
	autonomy build-image ${APY_ORACLE_AGENT_PUBLIC_ID} --pull
	docker push ${APY_ORACLE_IMAGE_NAME}:${APY_ORACLE_AGENT_HASH}

## VoteKit

`VoteKit` ia a Swiss army knife for computational social choice research.

**Helpful links:** [Source Repository](https://github.com/mggg/VoteKit) | [Issues](https://github.com/mggg/VoteKit/issues) | [MGGG.org](https://mggg.org/)


[![PyPI version](https://badge.fury.io/py/votekit.svg)](https://badge.fury.io/py/votekit)
![Test badge](https://github.com/mggg/VoteKit/workflows/Test%20&%20Lint/badge.svg)

## Installation
Votekit can be installed through any standard package management tool:

    pip install votekit

or

    poetry add votekit

## Development
*This project is in active development* in the [mggg/VoteKit](https://github.com/mggg/VoteKit) GitHub repository, where bug reports and feature requests, as well as contributions, are welcome.

VoteKit project requires [`poetry`](https://python-poetry.org/docs/#installation), and Python >= 3.9. (This version chosen somewhat arbitrarily.)

To get up and running, run `poetry install` from within the project directory to install all dependencies. This will create a `.venv` directory that will contain dependencies. You can interact with this virtualenv by running your commands prefixed with `poetry run`, or use `poetry shell` to activate the virtualenv.

Once you've run `poetry install`, if you run `poetry run pre-commit install` it will install code linting hooks that will run on every commit. This helps ensure code quality.

To run tests run `poetry run pytest` or `./run_tests.sh` (the latter will generate a coverage report).

To release, run `poetry publish --build`

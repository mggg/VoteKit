## VoteKit

`VoteKit` is a Swiss army knife for computational social choice research.

**Helpful links:** [Source Repository](https://github.com/mggg/VoteKit) | [Documentation](https://votekit.readthedocs.io/en/latest/) | [Issues & Feature Requests](https://votekit.readthedocs.io/en/latest/package_info/issues/) | [Contributing](https://votekit.readthedocs.io/en/latest/package_info/contributing/) | [MGGG.org](https://mggg.org/)


[![PyPI badge](https://badge.fury.io/py/votekit.svg)](https://badge.fury.io/py/votekit)
![Test badge](https://github.com/mggg/VoteKit/workflows/Test%20&%20Lint/badge.svg)

## Installation
Votekit can be installed through any standard package management tool:

    pip install votekit

For more detailed instructions, please see the [installation](https://votekit.readthedocs.io/en/latest/#installation) section of the VoteKit documentation.


## Issues and Contributing
This project is in active development in the [mggg/VoteKit](https://github.com/mggg/VoteKit) GitHub repository, where [bug reports and feature requests](https://votekit.readthedocs.io/en/latest/package_info/issues/), as well as [contributions](https://votekit.readthedocs.io/en/latest/package_info/contributing/), are welcome.

Currently VoteKit uses `poetry` to manage the development environment. If you want to make a pull request, first `pip install poetry` to your computer. Then, within the Votekit directory and with a virtual environment activated, run `poetry install` This will install all of the development packages you might need. Before making a pull request, run the following:
- `poetry run pytest tests --runslow` to check the test suite,
- `poetry run black .` to format your code,
- `poetry run ruff check src tests` to check the formatting, and then
- `poetry run mypy src` to ensure that your typesetting is correct.

Then you can create your PR! Please do not make your PR against `main`.


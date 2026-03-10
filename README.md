## VoteKit

`VoteKit` is a Swiss army knife for computational social choice research.

**Helpful links:** [Source Repository](https://github.com/mggg/VoteKit) |
[Documentation](https://votekit.readthedocs.io/en/latest/) |
[Issues & Feature Requests](https://votekit.readthedocs.io/en/latest/package_info/issues/) |
[Contributing](https://votekit.readthedocs.io/en/latest/package_info/contributing/) |
[MGGG.org](https://mggg.org/)


[![PyPI badge](https://badge.fury.io/py/votekit.svg)](https://badge.fury.io/py/votekit)
![Test badge](https://github.com/mggg/VoteKit/workflows/Test%20&%20Lint/badge.svg)

## Installation
Votekit can be installed through any standard package management tool:

    pip install votekit

For more detailed instructions, please see the
[installation](https://votekit.readthedocs.io/en/latest/#installation) section of the VoteKit
documentation.


## Issues and Contributing
This project is in active development in the
[mggg/VoteKit](https://github.com/mggg/VoteKit) GitHub repository, where
[bug reports and feature requests](https://votekit.readthedocs.io/en/latest/package_info/issues/),
as well as
[contributions](https://votekit.readthedocs.io/en/latest/package_info/contributing/), are welcome.

VoteKit uses `uv` for dependency management and `go-task` for common development commands. To set
up a contributor environment, install `go-task` and run `task setup` from the repository root.
That bootstraps Astral's official standalone `uv`, installs Python 3.11 if needed, and keeps the
managed tooling under `.task-tools/`.

Before making a pull request, run the following:
- `task format`
- `task lint`
- `task typecheck`
- `task test`
- `task coverage`

If you already have `uv` on your `PATH`, you can also run the underlying commands directly with
`uv run`, for example `uv run pytest tests --cov=src/votekit --cov-report=term-missing` for a
coverage run or `uv run pytest tests --runslow` for the full test suite. The repository
root [`CONTRIBUTING.md`](CONTRIBUTING.md) contains the current contributor workflow, code style
guidance, and pull request expectations.

# Contributing to VoteKit

Thanks for your interest in contributing to VoteKit! Contributions of all sizes are welcome,
including bug reports, documentation improvements, tests, examples, and new features.

If you are planning anything larger than a small bug fix or documentation change, please contact
`code@mggg.org` before you start coding so the maintainers can help you line up with the
current roadmap and target branch.

## Ways to contribute

- Report bugs or confusing behavior.
- Improve or expand documentation.
- Add tests for uncovered behavior or regressions.
- Fix bugs or edge cases.
- Propose or implement new election, cleaning, metrics, or visualization features.

## Development setup

VoteKit uses:

- [uv](https://astral.sh/uv/) for environment and dependency management
- [go-task](https://taskfile.dev/) for common development commands
- [ruff](https://astral.sh/ruff/) for linting
- [black](https://black.readthedocs.io/en/stable/) for formatting
- [isort](https://pycqa.github.io/isort/) for import sorting
- [ty](https://github.com/astral-sh/ty) for type checking
- [pytest](https://docs.pytest.org/) for tests
- [pre-commit](https://pre-commit.com/) for local quality checks

Recommended setup:

1. Install `go-task`.
2. Fork and clone the repository.
3. From the repository root, run `task setup`.

`task setup` bootstraps Astral's official standalone `uv`, installs Python 3.11 if needed,
stores its cache and managed Python under `.task-tools/`, syncs the pinned dependencies, and
installs the pre-commit hooks.

If you already have `uv` installed and prefer to run the steps directly, the equivalent setup is:

```bash
uv python install 3.11
uv --managed-python sync --locked --all-groups --all-extras --python 3.11
uv run pre-commit install
```

## Contributor workflow

1. Fork the repository and clone your fork locally.
2. Create a descriptive branch from the current target branch.
3. Keep your change focused. Avoid bundling unrelated refactors into the same pull request.
4. Add or update tests for any behavior change.
5. Run the relevant checks locally before you open a PR.
6. Open a pull request with a clear summary of the problem, your approach, and how you tested it.

If you are unsure which branch to target, ask the maintainers before opening the PR.

### Branch naming

Use descriptive branch names such as:

- `fix/score-profile-csv-validation`
- `feat/add-schulze-example`
- `docs/update-contributing-guide`

## Running checks locally

Preferred `task` commands:

```bash
task format
task lint
task typecheck
task test
task docs
```

If you already have `uv` on your `PATH`, the equivalent direct commands are:

```bash
uv run black src tests
uv run isort src tests
uv run ruff check src tests
uv run ty check src tests
uv run pytest tests
uv run pytest tests --runslow
uv run pre-commit run --all-files
```

Notes:

- Slow tests are marked with `@pytest.mark.slow` and only run when you pass `--runslow`.
- If you change public documentation or tutorial content, run `task docs`.
- If you touch plotting or animation behavior, check the relevant snapshot tests.

## Pull request expectations

Before opening a pull request, make sure that:

- the change is scoped to a single topic
- code, tests, and docs are updated together when needed
- new behavior is covered by tests
- linting, formatting, and type checks pass locally
- the PR description explains the user-facing impact and any notable tradeoffs

Small pull requests are much easier to review and merge than large mixed changes.

## Code style guidelines

VoteKit is a mixed but steadily modernizing Python 3.11+ codebase. When contributing, prefer
the current conventions below and avoid style-only churn in unrelated files.

- Follow the repo tooling first: `black`, `isort`, and `ruff` define the baseline style.
- Keep lines at roughly 100 characters to match the configured formatter and linter settings.
- Add type annotations for function parameters and return values. Run `uv run ty check src tests`
  on changes that add or reshape APIs.
- Prefer modern type syntax in new or substantially updated code, such as `str | None` instead of
  `Optional[str]`. Older files still contain pre-3.10 style hints, and you do not need to rewrite
  them unless you are already editing that area for a substantive reason.
- Use descriptive `snake_case` names for variables and functions, `PascalCase` for classes, and
  `UPPER_SNAKE_CASE` for module-level constants.
- Keep functions focused. Small helpers are preferred over long functions with several distinct
  responsibilities.
- Put validation and obvious guard clauses near the top of a function. This pattern is common
  across the election, profile, and utility modules.
- Prefer straightforward control flow over extra abstraction. This codebase generally favors clear
  data flow and targeted helpers over deep inheritance or unnecessary indirection.
- Match the surrounding file when touching older modules. Consistency within a file is more
  important than forcing a full-file style migration.
- Use comments sparingly. Prefer names and small helper functions to explain intent, and reserve
  comments for non-obvious logic or domain-specific reasoning.

### Docstrings

Public classes, functions, and methods should have docstrings that follow the project’s existing
Google-style variant:

```python
def foo(arg1: str | None, arg2: int = 3) -> str:
    """
    Brief description.

    More details if needed.

    Args:
        arg1 (str | None): Description.
        arg2 (int, optional): Description. Defaults to 3.

    Returns:
        str: Description of the returned value.

    Raises:
        ValueError: Description of the failure mode.
    """
```

Docstring conventions used throughout the repository:

- Put the summary on its own line inside the docstring.
- Include `Args`, `Returns`, and `Raises` when they apply.
- Document optional parameters and default behavior explicitly.
- Add examples only when they help clarify non-obvious usage.

## Testing guidelines

Tests are required for behavior changes.

- Add tests in `tests/` near the existing area that covers the same module or feature.
- Mirror the module structure where practical. For example, election code belongs under
  `tests/elections/...`.
- Cover both successful behavior and expected failures.
- When raising exceptions, prefer tests that check the error message with `pytest.raises(...,
  match=...)`.
- Include edge cases that are natural for the change: empty inputs, invalid candidate data,
  malformed rankings, tie handling, or zero-weight behavior.
- Mark long-running tests with `@pytest.mark.slow`.

## Documentation guidelines

If your change affects public behavior, update the relevant documentation alongside the code.
Depending on the change, that may include:

- docstrings in `src/votekit`
- narrative docs under `docs/`
- tutorial notebooks or generated tutorial pages
- examples or README references

## Community guidelines

This project follows the Contributor Covenant Code of Conduct. By participating, you agree to
abide by the expectations in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Questions

If anything in the contribution process is unclear, please feel free to reach out to 
`code@mggg.org` with questions. Thanks! 

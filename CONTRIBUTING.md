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
- [ruff](https://astral.sh/ruff/) for linting, import sorting, and formatting
- [ty](https://github.com/astral-sh/ty) for type checking
- [pytest](https://docs.pytest.org/) for tests
- [pre-commit](https://pre-commit.com/) for local quality checks

Recommended setup:

1. Install `go-task`.
2. Fork and clone the repository.
3. From the repository root, run `task setup`.

`task setup` installs Astral's official standalone `uv` if you don't have it, installs a managed
Python 3.11 environment, syncs the pinned dependencies, and installs the pre-commit hooks. 

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
task test -- <pytest cli args>
task test:tests/path
task coverage
task docs
```

If you already have `uv` on your `PATH`, the equivalent direct commands are:

```bash
uv run ruff check --select I --fix src tests
uv run ruff format src tests
uv run ruff check src tests
uv run ty check src tests
uv run pytest tests
uv run pytest tests --runslow
uv run pytest tests --cov=src/votekit --cov-report=term-missing
uv run pre-commit run --all-files
```

Notes:

- Slow tests are marked with `@pytest.mark.slow` and only run when you pass `--runslow`.
- To scope a Task-based test run, use `task test -- tests/<path>` or `task test:tests/<path>`.
- `task coverage` runs the default test suite with a terminal coverage summary for `src/votekit`.
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

- Follow the repo tooling first: Ruff defines the baseline style for formatting, import order,
  and linting.
- Use absolute imports in all implementation files. Relative imports are only used in `__init__.py`
  files for re-exporting. For example, prefer `from votekit.elections.election_state import
  ElectionState` over `from ..election_state import ElectionState`.
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
Google-style variant with a 100 character per line limit. For example:

```python
def foo(arg1: str | None, arg2: int = 3) -> str:
    """
    Brief description (try to stay under 100 characters).

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

## Adding a new election method

Election classes live under `src/votekit/elections/election_types/` and are organized by ballot
type. Put new elections in the appropriate subfolder (`ranking/`, `scores/`, or `approval/`) and
export the class from the subfolder's `__init__.py` and from `src/votekit/elections/__init__.py`.

### Choosing a base class

| Ballot type          | Base class                                            | Profile type   |
|----------------------|-------------------------------------------------------|----------------|
| Ranked ballots       | `RankingElection`                                     | `RankProfile`  |
| Score/rating ballots | `GeneralRating` (or `Election[ScoreProfile]` directly)| `ScoreProfile` |
| Approval ballots     | `GeneralRating` (with `per_candidate_limit=1`)        | `ScoreProfile` |

`RankingElection` and `GeneralRating` both ultimately inherit from `Election[P]`, the root
abstract base class in `src/votekit/models.py`.

### Profile types

`RankProfile`, `ScoreProfile`, and `PreferenceProfile` are all defined in
`src/votekit/pref_profile/pref_profile.py`.

- Use `RankProfile` when your election requires ranked ballots. `RankingElection._validate_profile`
  enforces this automatically.
- Use `ScoreProfile` when your election requires score ballots.
- `PreferenceProfile` is the common base class. Avoid accepting it directly in new election
  classes unless the method genuinely supports both ballot types.

### Required methods

Every `Election` subclass must implement three abstract methods:

**`_validate_profile(self, profile)`** — called at construction before any election logic runs.
Raise `ProfileError` for wrong profile type and `ValueError` for invalid configurations
(e.g. fewer candidates than seats). `RankingElection` provides a default implementation that
checks for a `RankProfile` and valid rankings; override only if you need stricter checks.

**`_is_finished(self) -> bool`** — return `True` when no further rounds are needed.
For single-round elections, check `len(self.election_states) == 2` (round 0 is the initial state).
For iterative elections, check whether enough candidates have been elected.

**`_run_step(self, profile, prev_state, store_states=False) -> profile`** — run one round of the
election and return the updated profile. When `store_states=True`, build an `ElectionState` and
append it to `self.election_states`. Only append when `store_states=True`; the base class calls
`_run_step` without the flag when replaying rounds via `get_profile`.

### Populating `ElectionState`

Each round that advances the election should produce an `ElectionState`:

```python
from votekit.elections.election_state import ElectionState

new_state = ElectionState(
    round_number=round_number,           # int, 1-indexed
    remaining=remaining,                 # tuple[frozenset[str], ...], ordered by score
    elected=elected,                     # tuple[frozenset[str], ...], empty if none this round
    eliminated=eliminated,               # tuple[frozenset[str], ...], empty if none this round
    tiebreaks=tiebreaks,                 # dict[frozenset[str], tuple[frozenset[str], ...]]
    scores=scores,                       # dict[str, float] for remaining candidates only
)
self.election_states.append(new_state)
```

Candidates and sets follow a consistent ordering convention: tuples are ordered best-to-worst
(highest score first for `remaining`, first-elected first for `elected`), and frozensets within
a tuple represent tied candidates.

### Passing a `score_function`

If your election uses scores to rank candidates each round, pass a `score_function` to
`super().__init__`. This function takes a profile and returns a `dict[str, float]` mapping
candidates to their scores. The base class uses it to populate round-0 scores and sort the
initial `remaining` tuple. Utilities like `score_dict_from_score_vector` and
`score_profile_from_ballot_scores` in `src/votekit/utils.py` cover the most common cases.

If your election has no meaningful per-round scores (e.g. a Condorcet method), pass
`score_function=None` and all candidates will start as tied in round 0.

### Error message conventions

Error messages should state what constraint was violated and echo the offending value where
helpful. Use f-strings with constants rather than hardcoded strings:

```python
# Preferred
raise ValueError(f"n_seats ({self.n_seats}) must be positive.")
raise ValueError(f"tiebreak '{tiebreak}' is not a valid option. Choose from {VALID_TIEBREAKS}.")

# Avoid
raise ValueError("Invalid input.")
```

Validation and guard clauses belong at the top of `__init__`, before the call to
`super().__init__`.

## Community guidelines

This project follows the Contributor Covenant Code of Conduct. By participating, you agree to
abide by the expectations in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

## Questions

If anything in the contribution process is unclear, please feel free to reach out to 
`code@mggg.org` with questions. Thanks! 

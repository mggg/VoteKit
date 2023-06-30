## Development

This project requires [`poetry`](https://python-poetry.org/docs/#installation), and Python >= 3.9. (This version chosen somewhat arbitrarily.)

To get up and running, run `poetry install` from within the project directory to install all dependencies. This will create a `.venv` directory that will contain dependencies. You can interact with this virtualenv by running your commands prefixed with `poetry run`, or use `poetry shell` to activate the virtualenv.

Once you've run `poetry install`, if you run `poetry run pre-commit install` it will install code linting hooks that will run on every commit. This helps ensure code quality.

To run tests run `poetry run pytest` or `./run_tests.sh` (the latter will generate a coverage report).

To release, run `poetry publish --build`



## Development

This project requires [`poetry`](https://python-poetry.org/docs/#installation), and Python >= 3.8.  (This version chosen somewhat arbitrarily.)

To get up and running, run `poetry install` from within the project directory to install all dependencies. This will create a `.venv` directory that will contain dependencies.  You can interact with this virtualenv by running your commands prefixed with `poetry run`, or use `poetry shell` to activate the virtualenv.

Once you've run `poetry install`, if you run `poetry run pre-commit install` it will install code linting hooks that will run on every commit.  This helps ensure code quality.

To run tests run `poetry run pytest`

To release, run `poetry publish --build`

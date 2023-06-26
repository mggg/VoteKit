

## Development

This project requires [`poetry`](https://python-poetry.org/docs/#installation), and Python >= 3.9.  (This version chosen somewhat arbitrarily.)

To get up and running, run `poetry install` from within the project directory to install all dependencies. This will create a `.venv` directory that will contain dependencies.  You can interact with this virtualenv by running your commands prefixed with `poetry run`, or use `poetry shell` to activate the virtualenv.

To run tests run `poetry run pytest`

To release, run `poetry publish --build`

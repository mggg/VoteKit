==========================
Contributing to VoteKit
==========================

Thank you for your interest in contributing to VoteKit! This document provides
guidelines for contributing to the VoteKit project. All contributions and 
feedback are welcome and appreciated 😊. The closer you can follow these guidelines, the 
faster your contributions can enter the codebase.


Contributing Guidelines
=======================

**Coding Standards**: VoteKit follows PEP 8 guidelines for coding style, so we
ask that any contributors do the same to ensure that the codebase is consistent. For
more information, see the `PEP 8 Style Guide <https://www.python.org/dev/peps/pep-0008/>`_.

**Writing Tests**: If you write a new feature, please make sure that it is
included in a test somewhere. A test is a short piece of code that checks that the package is working as intended.
While it is not feasible to test every single aspect of a package, adding tests for new features is a crucial part
of open-source software. You can use our `current tests <https://github.com/mggg/VoteKit/blob/main/tests>`_  as a starting place 
to show you how these work. Depending on the scale of your feature, you may need multiple tests.
At minimum, your tests should 
- check the basic functionality of your feature, such as a test case of an algorithm or an instantiation of a class, and 
- ensure that any error messages are correctly raised.

**Documentation**: For any new features that you add, please make sure to include
a comprehensive docstring. We have a defined format for docstrings that we use
throughout the `codebase <https://github.com/mggg/VoteKit/blob/main/src/votekit>`_ codebase, so please make sure that any additions are consistent
with that format.

Poetry
=============

VoteKit uses Poetry to manage the package and all of its dependencies for developers.
This is not needed for standard users.
Within your `virtual environment <../user/install.rst>`_, run ``pip install poetry``.
Then, run ``poetry install`` from within the root of your VoteKit project directory to install all dependencies. 
This will add the necessary dependencies to a `.venv` directory.
Once you've run ``poetry install``,  run ``poetry run pre-commit install``, which will install code linting hooks that will run on every commit. 
This helps ensure code quality, like automatically checking formatting.



Using GitHub
============
This tutorial assumes some understanding of GitHub, either of the command line interface or of the desktop GUI.
If you are new to GitHub, please feel free to reach out to us at our email, "code[at]mggg[dot]org", and we will schedule some
time to get you started.

Quick Start via Command Line
============================

1. **Fork the repository** on GitHub.
2. **Clone your fork locally**.
3. **Configure the upstream repo** ``git remote add upstream https://github.com/mggg/VoteKit.git``
4. **Create a new branch** for your contribution ``git checkout -b my-new-feature``.
5. **Make your changes** and commit them ``git commit -am 'Add some feature'``.
6. **Run tests and linters** with ``poetry run pytest``, ``poetry run ruff src tests``, 
``poetry run mypy src``, and ``black src``.
7. **Pull the latest changes from upstream** and rebase your branch if necessary.
8. **Push your branch** to GitHub ``git push origin my-new-feature``.
9. **Open a Pull Request** on GitHub.

Quick Start via Desktop GUI
============================


1. **Fork the repository** on GitHub.
    .. image:: ../_static/assets/github_desktop/fork_repo.png
        :align: center
2. **Clone the repository locally**.
3. **Create a new branch** for your feature and publish the branch.
4. **Make your changes** and commit them to your branch.
5. **Run tests and linters** in the command line at the root of your VoteKit directory
with ``poetry run pytest``, ``poetry run ruff src tests``, ``poetry run mypy src``, and ``black src``.
Make sure your virtual environment is activated.
6. **Pull and push** to your branch.
7. **Check against the main branch** to ensure that your branch can be merged.
8. **Open a Pull Request** on GitHub.

Pull Requests
=========================
For each pull request, please confirm the following.

1. Ensure your branch is up to date with the ``main`` branch and that the tests are passing.
2. Open a pull request against the ``main`` branch of the VoteKit repository. 
3. Write a detailed comment explaining the changes you made and the reasoning behind them.

The project maintainers will review your changes and provide feedback. You may be asked to revise your code.
 

Community Guidelines
====================

We follow an adaptation of the Contributor Covenant Code of Conduct, which, 
in essence, means that we expect community members to

- **Be respectful** of different viewpoints and experience levels.
- **Gracefully accept constructive criticism**.
- **Focus on what is best for the community**.

For more detailed information about our community guidelines, please see the
`Code of Conduct <https://github.com/mggg/VoteKit/blob/main/CODE_OF_CONDUCT.md>`_ 
page of the main repository.


Thank You
=========

Thank you for contributing to VoteKit! We appreciate all the time and
effort that you put into making this package the best that it can be!
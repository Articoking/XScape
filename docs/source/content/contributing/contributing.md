# Contributing

Thank you for your interest in helping to build XScape!
In this page you will find all the information you need to start contributing to the project.
Make sure you follow the [installation instructions](../usage/installation.md#developers-and-documentation-writers) for your particular case.

## Developers

XScape uses [Poetry](https://python-poetry.org/) for dependency management and [Poe the Poet](https://github.com/nat-n/poethepoet) for task automation.
You can install both tools using `pipx`.

```bash
pipx install poetry poethepoet
```

Version control is done with Git through our [GitHub repository](https://github.com/Articoking/XScape/tree/main).
To add your contributions to XScape clone the main branch and submit a pull request once your work is done.

Don't forget to add docstrings and typehints to your code!
This allows the Sphinx to automatically create complete documentation pages for each module, class and function in the library.
We use `numpy`-style docstrings all throughout the codebase.

You may automatically run the linter and all tests using `poe`, by running:

```bash
poetry run poe all
```

To run only the linter or the tests, substitute `all` with either `lint` or `test`.

We use [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, and `pytest` as a testing framework.

## Documentation writers

These documentation pages are built with [Sphinx](https://www.sphinx-doc.org), using the [MyST parser](https://myst-parser.readthedocs.io/en/latest/index.html) for Markdown support.

Make sure to follow all Markdown best-practices.
This should be easy if you're using Visual Studio Code with the `markdownlint` extension.

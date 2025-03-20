# Installation

Until the package is officially published, the only way of installing XScape is by cloning the repository.
Once you have it on your own machine, go into the XScape directory and run  `poetry install`.
The whole process can be done in three commands:

```bash
git clone https://github.com/Articoking/XScape.git
cd XScape
poetry install
```

## Developers and documentation writers

If you want to [**contribute to the library,**](../contributing/contributing.md) you will likely want to install the extra developer dependencies.
To do that, use the `--with dev` option when running the installation.

If instead you want to contribute to these documentation pages, use `--with docs`.

## Post-install extras

Certain credentials are needed in order to use some of XScape's functionality.
More specifically, functions that use the `copernicusmarine` API require that you have an account and that your configuration files be set correctly, lest the API ask for your credentials repeatedly.
Please refer to [their configuration tutorial](https://help.marine.copernicus.eu/en/articles/8185007-copernicus-marine-toolbox-credentials-configuration) for details on how to configure your Copernicus Marine Service credentials.

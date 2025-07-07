# Contribution Guidelines

Thank you for your interest in contributing to NQLib! We welcome contributions from the community. Please follow these guidelines to ensure a smooth contribution process.

## Installation

Download the repository and navigate to the directory:

```sh
# cd /some/path/to/your/workspace  # if necessary
git clone https://github.com/knttnk/NQLib
cd NQLib
```

Create a virtual environment and install the package:

```sh
# This is an example. Choose your preferred method to create a virtual environment.
# conda create -n nqlib-dev python=3.13 -y
# conda activate nqlib-dev
pip install -U pip
pip install --editable .
```

(You can also install the packages via conda. If you fail to install the package (especially `slycot`), please try the following command and then try to install the package again.)

```sh
conda install -c conda-forge slycot
```

For testing and development, you may also want to install the development dependencies:

```sh
pip install -r requirements.txt -U
```

## Running Tests

To run the tests, you can use `pytest`. Make sure you have installed the development dependencies as mentioned above.

```sh
python -m pytest --doctest-modules
```

## Building and Publishing for PyPI

Before publishing a new version, ensure that you have updated the version numbers in the following files:

- `pyproject.toml`
- `CHANGELOG.md`
- `src/nqlib/__init__.py`

### Building the Package

```sh
python -m build
```

### Publishing to test PyPI

```sh
python -m twine upload -r testpypi dist/*
```

### Testing the Package

You can test the package by installing it from test PyPI:

```sh
pip install -i https://test.pypi.org/simple/ nqlib
python -c "import nqlib; print(nqlib.__version__)"
```

If this command runs without errors and prints the version number, the package is installed correctly.

### Publishing to PyPI

To publish the package to the official PyPI, use the following command:

```sh
python -m twine upload -r pypi dist/*
```

Then you can install the package from PyPI:

```sh
pip install nqlib
python -c "import nqlib; print(nqlib.__version__)"
```

## Building and Publishing for Conda-Forge

This requires [Conda](https://docs.conda.io/en/latest/).
This instructions assume you have already installed Conda and set up your environment.

```sh
# packages
conda install conda-build
conda install -c conda-forge grayskull
# create a working directory
mkdir conda-recipe
cd conda-recipe
# create a recipe
grayskull pypi nqlib
git clone https://github.com/knttnk/staged-recipes.git
git checkout -b nqlib
# copy conda-release/nqlib/meta.yaml to staged-recipes/nqlib/recipes
cp ../conda-release/nqlib/meta.yaml staged-recipes/nqlib/recipes/
```

Then, change the `meta.yaml` file as follows:

```yaml
...
license_file: LICENSE.txt
...
```
<!-- 
# conda-release/nqlib/meta.yaml を staged-recipes/nqlib/recipes に
# meta.yaml の license_file を LICENSE.txt に変え，nqlibのそれをmeta.yamlの隣にコピー
# 移動したあと下を実行
git add .
git commit -m "restored the example and added LICENSE.txt"
git push origin nqlib  # originの後ろは、新しく作成したブランチ名
-->

TODO: Confirm publishing process

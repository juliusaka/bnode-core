# bnode-core
Balanced Neural ODEs - Core Library

## Description

This repository contains code implementing "Balanced Neural ODEs" (BNODEs) as described in the paper:

- Julius Aka, Johannes Brunnemann, Jörg Eiden, Arne Speerforck, Lars Mikelsons. "Balanced Neural ODEs: nonlinear model order reduction and Koopman operator approximations", ICLR 2025. [OpenReview](https://openreview.net/forum?id=nA464tCGR5) ([bibtex key](#citation))

Balanced Neural ODEs (BNODEs) are a data-driven method to learn reduced-order models for high-dimensional dynamical systems. The approach combines Variational Autoencoders (VAEs) with Neural Ordinary Differential Equations (Neural ODEs) to learn a low-dimensional latent representation of the system's dynamics.

<p align="center">
    <img src="images/bnode.svg" alt="BNODE Architecture" style="max-width:100%; height:auto;">
</p>

It consits of the following main components:

- *Encoder*: Maps high-dimensional input data to a low-dimensional latent space. Distinct encoders are use for
    - (physical) parameter encoding (if applicable)
    - Control input encoding (if applicable)
    - Initial state encoding (intended to be used for simulation data with acces on the full state)
- *Neural ODE*: Models the dynamics in the latent space using a neural network to parameterize the ODE.
- *Decoder*: Reconstructs the high-dimensional data from the latent representation.

Each component can be set to be a nonlinear neural network or to be a purely linear model, allowing for flexibility in model complexity. For example, using linearity in the control encoder, the Neural ODE and the decoder results in a Koopman operator approximation for model-predictive control. 

The main features of the package this repository implements are:

- **Dataset generation**: Functions to generate dataset from physical models provided as FMU (Functional Mock-up Unit, [FMI standard](https://fmi-standard.org/)) with different sampling methods.
- **Model training**: Architecture implementation of BNODEs and state space NeuralODEs using [PyTorch](https://pytorch.org/) and [torchdiffeq](https://github.com/rtqichen/torchdiffeq). Both models can be trained using the same trainer, facilitating a lot of special considerations needed for training Neural ODEs.
- Various utilties for enabling an efficient workflow, e.g. logging with [mlflow](https://mlflow.org/), configuration management with [hydra](https://hydra.cc/) and a simple GUI for visualizing training results.


## Installation
e.g. for running the examples in the `examples/` folder.

TODO: add examples

1. Clone the repository:

  ```
        git clone <repository-url>
        cd bnode-core
        git submodule update --init #no need for recursive update
  ```

2. [Install uv](https://docs.astral.sh/uv/getting-started/installation/), a very-fast python package manager.

Then, create a virtual environment and install the dependencies:

```
uv venv create 
```


3. Install Torch:
    Depending on your hardware and CUDA version, install the appropriate version of PyTorch. UV does not support automatic backend selection in the default "uv run / uv sync" command yet. But the uv pip interface (replacement for commonly used pip commands) does support [automatic backend selection](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection).
    So you can install torch with

   ```
    uv pip install torch torchvision torchaudio --torch-backend auto
   ```
   before using ```uv sync``` / ```uv run```.
   You can also manually select the appropriate command from [uv doc](https://docs.astral.sh/uv/guides/integration/pytorch/#automatic-backend-selection)

You're done!

4. (Optional:) Use the virtual environment:
   
Run 
```
uv sync
```

at the first time (no 'uv run' command before) to install a virtual environment of the project defined in pyproject.toml. You can also use ```uv sync``` to test if the package can be installed in the way you specified it. 

To activate the virtual environment, use:

```
[linux-bash]
source .venv/bin/activate
[windows-powershell]
.venv\Scripts\Activate
```
in your Terminal or add it in VS Code using the command palette (Ctrl+Shift+P) and searching for "Python: Select Interpreter".

You don't need to use the virtual environment, **you can simply** place ```uv run``` in front of the python-file you want to run to make it run in the specified environment.

## Usage
To see the documentation, run:

```
make doc
```

or, if you don't have make installed,

```
uvx --with mkdocstrings  --with mkdocs-material --with mkdocstrings-python --with mkdocs-include-markdown-plugin mkdocs serve
```

and open the website shown in the terminal.

(When deploying this on GitHub, the github action will automatically build and publish the documentation to GitHub pages.)

## Package Structure
The package is structured as follows:

```
└───bnode_core
    │   config.py
    │   filepaths.py
    │
    ├───data_generation
    │       data_preperation.py
    │       raw_data_generation.py
    │
    ├───ode
    │       bnode.py
    │       node.py
    │       trainer.py
    │
    ├───plots
    │
    └───utils
```

## Support
If you have questions or issues, please open an issue on GitHub. You can also reach out to me via [email](mailto:julius.aka@uni-a.de), see [Authors](#authors).

## Contributing
You are welcome to contribute further test models to this project. Please fork the repository and create a pull request with your changes.

### Development Setup
For the development setup, [Make](https://gnuwin32.sourceforge.net/packages/make.htm) should be installed. [What does make do?](https://makefiletutorial.com/) 
For us, this is just a convenient way to run common tasks.

For installing ```make``` on Windows, download the setup from [this website](https://gnuwin32.sourceforge.net/packages/make.htm) and add the make.exe to your PATH environment variable. 
- (Usually, this is `C:\Program Files (x86)\GnuWin32\bin`, add this to the PATH variable in the system environment variables, system properties --> advanced system settings --> environment variables, then restart your terminal).

For linux, install make via your package manager, e.g. `sudo apt install make`.

Afterwards, you can use the commands in the Makefile, e.g. `make check` to check the code style, `make format` to format the code, or `make test` to run the tests.

### Development Features

- **Continuous Integration `make allci`**
    - Ruff for linting `make check`
    - Ruff for formatting `make format`
    - Ty for type checking `make type`
    - Pytest for testing `make test`
    - Pytest-cov for testing coverage `make cov`
    - Pre-commit hooks to make some checks and formatting code before commits `make commit`
- **Documentation**
    - [Mkdocs](https://www.mkdocs.org/getting-started/) for documentation building with Markdown `make doc`
    - Using [mkdocs-material](https://squidfunk.github.io/mkdocs-material/creating-your-site/) as theme
    - Automatic build of the API Reference page
    - Docstrings are in [Google style](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html) and used in mkdocs using [mkdocstrings](https://mkdocstrings.github.io/)
    - Pre-configured GitHub Action / Gitlab CI for publishing the documentation on Github pages / Gitlab page
- see [modern-python-boilerplate](https://github.com/lambda-science/modern-python-boilerplate/) for including Docker, packaging, publishing to PyPI, etc. (we don't need this features yet)

<a id="authors"></a>
## Authors
- Julius Aka, 
Chair of Mechatronics, 
University of Augsburg,
<julius.aka@uni-a.de>
- Johannes Brunnemann
- Jörg Eiden
- Arne Speerforck
- Lars Mikelsons

<a id="citation"></a>
## Citation
If you use this code in your research, please cite the following paper:
```
@inproceedings{
aka2025balanced,
title={Balanced Neural {ODE}s: nonlinear model order reduction and Koopman operator approximations},
author={Julius Aka and Johannes Brunnemann and J{\"o}rg Eiden and Arne Speerforck and Lars Mikelsons},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=nA464tCGR5}
}
```

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.

## Acknowledgments
The repository structure is inspired by [modern-python-boilerplate](https://github.com/lambda-science/modern-python-boilerplate/).
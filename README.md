# Getting started 🤗

1. Clone this repository in a folder.
2. Install python and [poetry](https://python-poetry.org/docs/#installation).
3. Move to the folder in which you cloned the project, create a virtual environment for the project and install in it the depedencies required using poetry.

```bash
poetry install [--with dev]
```

**NB:** *The `--with dev` is optional and required only if you need developement libraries.*

# Developpement :computer:

- To add new libraries to the requirements use poetry. Example:

Use such commands from within the project folder:
```bash
poetry add matplotlib
```

- To automatically reformat code. Use black:

```bash
poetry run black sv_machines
```

**NB:** You need to have installed the developpement dependencies too in order to be able to use black. Otherwise it is not installed.

# Neural Control with Adaptive State Estimation

## Getting started

1. Install uv (https://docs.astral.sh/uv/).
2. Install required VSCode extensions: [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python), [Python Debugger](https://marketplace.visualstudio.com/items?itemName=ms-python.debugpy), [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff), and [ty](https://marketplace.visualstudio.com/items?itemName=astral-sh.ty).
3. Clone the repository with `git clone https://github.com/JerryLiu0911/Neural-Control-with-Adaptive-State-Estimation.git`.
4. Create a virtual environment and install project dependencies by running `uv sync` in the directory.
5. Ensure you select the local `.venv` as the interpreter (in VS Code, open the Command Palette and select _Python: Select Interpreter_).

## Development

- Lint using `uv run ruff check` or the task `ruff: lint`. Run `uv run ruff check --fix` or the task `ruff: lint (fix)` to write changes.
- Format using `uv run ruff format` or the task `ruff: format`.
- Typecheck using `uv run ty check` or the task `ty: typecheck`.
- Run tests with `uv run pytest` or the task `test: run all`.
- Build and serve documentation with `uv run --group docs make -C docs livehtml` or the task `docs: serve (live reload)`.

You can see more tasks in [.vscode/tasks.json](.vscode/tasks.json). VS Code is configured to format on save via bindings in [.vscode/settings.json](.vscode/settings.json). Note notebooks (i.e., `ipynb` file extensions) are excluded from typechecking and linting.

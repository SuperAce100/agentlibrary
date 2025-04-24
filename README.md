# Agent Library

Dynamic sub-agent creation and orchestration system.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for package management.

You will also need to setup an [OpenRouter](https://openrouter.ai/) API key and save it as an environment variable, ideally in `.env`.

## Usage

`src/main.py` provides the following command-line options:

```bash
uv run src/main.py --task "Your task description here" [options]
```

### Required Arguments

- `--task TEXT` (**required**): The task that the agent system will execute. This should be a detailed description of what you want to accomplish.
- `--verbose BOOLEAN`: Enable verbose output to see detailed progress of the agent system. Default: `False`
- `--output_path TEXT`: Path where the results will be saved. Default: `results/traces`

```bash
uv run src/main.py --task "Research the impact of climate change on biodiversity in tropical rainforests and summarize the findings in a structured report with recommendations for conservation efforts." --verbose True --output_path "results/traces"
```

## Roadmap

- [X] Add selective passing of context from sub-agents to the next, based on orchestrator
- [X] Dynamically pick from library at the beginning, create new ones on the fly (generalization is even more imoprtant)
- [X] Orchestrator gives feedback to the agents so they improve over time, one agent retains all of its memory
- [ ] Tool calling and data builds naturally on that

## Contributing

To contribute, just make a PR with whatever changes you want. This repo has pre-commit hooks with [Ruff](https://docs.astral.sh/ruff/) for linting and formatting, and [Mypy](https://mypy-lang.org/) for static typing. These ensure the codebase remains easy to use for all.

# Symphony v2

Dynamic sub-agent creation and orchestration system.

## Usage

```bash
uv run src/main.py --task "Create a comprehensive business proposal for an airline focused on connecting China and the West Coast of the US. Consider the idea’s financial viability, any potential legal challenges, the state of the market, and brand building potential." --output_path "results/airline.md"
```

## Roadmap

- [ ] Add selective passing of context from sub-agents to the next, based on orchestrator
- [ ] Dynamically pick from library at the beginning, create new ones on the fly (generalization is even more imoprtant)
- [ ] Orchestrator gives feedback to the agents so they improve over time, one agent retains all of its memory
- [ ] Tool calling and data builds naturally on that
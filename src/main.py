from tqdm import tqdm
from decomposition import decompose_task
from sub_agent_creation import create_sub_agent
import argparse

import concurrent.futures


def run(task: str, verbose: bool = False) -> str:
    """
    Run the multi-agent system
    """

    if verbose:
        print("Decomposing task...")
    decomposition = decompose_task(task)
    sub_agent_descriptions = decomposition.sub_agents

    if verbose:
        print("Sub-agents:")
        for desc in sub_agent_descriptions:
            print(f"Agent: {desc.name}: {desc.description}")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                create_sub_agent, desc.name, desc.description, desc.justification
            )
            for desc in sub_agent_descriptions
        ]
        sub_agents = []
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Creating sub-agents",
        ):
            sub_agents.append(future.result())

    if verbose:
        print(
            f"Created {len(sub_agents)} sub-agents: {', '.join([agent.name for agent in sub_agents])}"
        )

        for agent in sub_agents:
            print(f"Agent: {agent.name}: {agent.description}")

        print("Planning task...")

    if verbose:
        print("Done!")

    return "cleaned_document"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--output_path", type=str, default="results/result_1.md")
    args = parser.parse_args()

    task = args.task
    verbose = args.verbose
    output_path = args.output_path

    result = run(task, verbose=verbose)
    with open(output_path, "w") as f:
        f.write(result)


if __name__ == "__main__":
    main()

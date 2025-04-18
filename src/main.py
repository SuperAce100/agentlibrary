from tqdm import tqdm
from utils.context_manager import ContextManager
from decomposition import decompose_task
from orchestrator import Orchestrator
from sub_agent_creation import create_sub_agent
import argparse

import concurrent.futures


def run(task: str, max_iterations: int = 100, verbose: bool = False) -> str:
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

        print("Conducting pre-survey...")

    agent_registry = {agent.name: agent for agent in sub_agents}

    orchestrator = Orchestrator()
    pre_survey = orchestrator.pre_survey(task)
    if verbose:
        print("Pre-survey:")
        print(pre_survey)

        print("Planning task...")

    plan = orchestrator.plan(task, sub_agents)

    if verbose:
        print("Plan:")
        print(plan)

    context_manager = ContextManager()
    last_agent_name = ""
    last_response = ""

    for i in range(max_iterations):
        orchestration_step = orchestrator.orchestrate(
            last_agent_name, last_response, task, context_manager.get_context_names()
        )

        if verbose:
            print(f"Orchestration step: {orchestration_step.model_dump_json(indent=2)}")

        if orchestration_step.is_done:
            break

        last_agent_name = orchestration_step.agent_name

        sub_agent = agent_registry[orchestration_step.agent_name]
        last_response = sub_agent.call(orchestration_step.instructions)

        context_manager.add_context(orchestration_step.agent_name, last_response)

        if verbose:
            print(f"Agent {orchestration_step.agent_name} response:")
            print(last_response)

    final_response = orchestrator.compile_final_response(task)

    if verbose:
        print("Done!")
        print(f"Final response: {final_response}")
    return final_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    task = args.task
    verbose = args.verbose
    output_path = args.output_path

    result = run(task, verbose=verbose)
    print(result)
    if output_path:
        with open(output_path, "w") as f:
            f.write(result)


if __name__ == "__main__":
    main()

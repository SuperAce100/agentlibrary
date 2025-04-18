from tqdm import tqdm
from utils.context_manager import ContextManager
from decomposition import decompose_task
from orchestrator import Orchestrator
from sub_agent_creation import create_sub_agent
import argparse

import concurrent.futures

from utils.tracing import Tracer


def run(
    task: str,
    max_iterations: int = 100,
    verbose: bool = False,
    trace_path: str | None = None,
) -> str:
    """
    Run the multi-agent system
    """
    tracer = Tracer(task, trace_path, verbose)

    tracer.update_progress("Decomposing task...")

    decomposition = decompose_task(task)
    sub_agent_descriptions = decomposition.sub_agents

    tracer.trace(
        "\n".join(
            [
                f"Agent: {desc.name}: {desc.description}"
                for desc in sub_agent_descriptions
            ]
        ),
        "sub_agent_descriptions",
    )
    tracer.update_progress("Creating sub-agents...")

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

    tracer.trace(
        "\n".join(
            [f"Agent: {agent.name}: {agent.description}" for agent in sub_agents]
        ),
        "sub_agents",
    )

    tracer.update_progress("Conducting pre-survey...")

    agent_registry = {agent.name: agent for agent in sub_agents}

    orchestrator = Orchestrator()
    pre_survey = orchestrator.pre_survey(task)
    tracer.trace(pre_survey, "pre_survey")

    tracer.update_progress("Planning task...")

    plan = orchestrator.plan(task, sub_agents)
    tracer.trace(plan, "plan")

    context_manager = ContextManager()
    last_agent_name = ""
    last_response = ""

    for i in range(max_iterations):
        tracer.update_progress(f"Orchestrating step {i}...")

        orchestration_step = orchestrator.orchestrate(
            last_agent_name, last_response, task, context_manager.get_context_names()
        )

        tracer.trace(
            str(orchestration_step),
            f"orchestration_step_{i}",
        )

        tracer.update_progress(f"Called {orchestration_step.agent_name}...")

        if orchestration_step.is_done:
            break

        last_agent_name = orchestration_step.agent_name

        sub_agent = agent_registry[orchestration_step.agent_name]
        last_response = sub_agent.call(orchestration_step.instructions)

        response_name = context_manager.add_context(
            orchestration_step.agent_name, last_response
        )

        tracer.trace(
            last_response,
            f"sub_agent_response_{response_name}",
        )

    final_response = orchestrator.compile_final_response(task)

    tracer.update_progress("Done!")
    tracer.trace(final_response, "final_response")
    return final_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--output_path", type=str, default="results/traces")
    args = parser.parse_args()

    task = args.task
    verbose = args.verbose
    output_path = args.output_path

    result = run(task, verbose=verbose, trace_path=output_path)
    print(result)


if __name__ == "__main__":
    main()

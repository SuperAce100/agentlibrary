from tqdm import tqdm
from decomposition import decompose_task
from sub_agent_creation import create_sub_agent
from planning import plan_task, get_agent_order
from execution import execute_sub_agents
from utils.writing import clean_up_document

import concurrent.futures


def run_symphony(task: str, verbose: bool = False) -> str:
    """
    Run the symphony
    """
    if verbose:
        print("Decomposing task...")
    decomposition = decompose_task(task)
    sub_agent_descriptions = decomposition.sub_agents
    if verbose:
        print("Creating sub-agents...")
    # Create sub-agents in parallel
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
        print("Planning task...")
    template = plan_task(task, sub_agents)
    if verbose:
        print(f"Template: {template}")
    if verbose:
        print("Getting agent order...")
    agent_order = get_agent_order(template, sub_agents)
    if verbose:
        print(f"Agent order: {agent_order}")
    if verbose:
        print("Executing sub-agents...")
    final_document = execute_sub_agents(sub_agents, agent_order, template)
    if verbose:
        print("Cleaning up document...")
    cleaned_document = clean_up_document(
        final_document, [description.name for description in sub_agent_descriptions]
    )
    if verbose:
        print("Done!")
    return cleaned_document


if __name__ == "__main__":
    task = "Create a comprehensive business proposal for an airline focused on connecting China and the West Coast of the US. Consider the idea’s financial viability, any potential legal challenges, the state of the market, and brand building potential."
    result = run_symphony(task, verbose=True)
    with open("results/result_1.md", "w") as f:
        f.write(result)

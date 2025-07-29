from tqdm import tqdm
from utils.context_manager import ContextManager
from decomposition import decompose_task
from orchestrator import Orchestrator
from sub_agent_creation import create_sub_agent
import argparse
from models.agents import Agent 
import concurrent.futures
import os
import json

from utils.tracing import Tracer
from utils.prompts import sub_agent_prompt
from openai import OpenAI, AsyncOpenAI


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
    print("Initializing...")

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

        tracer.update_agent_loop("Orchestrator", str(orchestration_step))

        tracer.update_progress(f"Called {orchestration_step.agent_name}...")

        if orchestration_step.is_done:
            break

        last_agent_name = orchestration_step.agent_name
        relevant_context = "\n\n".join(
            context_manager.get_context(context_name)
            for context_name in orchestration_step.context
        )
        memories = sub_agent.retrieve_episodic_memories(orchestrator_instructions, last_agent_name)
        if memories:
            relevant_memories = "\n\n".join([
                f"Memory {i+1}:\n{mem.content}" 
                for i, mem in enumerate(memories)
            ])
        else:
            relevant_memories = "No relevant memories found."

        sub_agent: Agent = agent_registry[orchestration_step.agent_name]
        last_response = sub_agent.call_with_tools(
            sub_agent_prompt.format(
                orchestrator_instructions=orchestration_step.instructions,
                context=relevant_context,
                memories = relevant_memories,
            )
        )

        feedback_score, feedback_text = orchestrator.give_feedback(
            sub_agent.name, 
            orchestration_step.instructions,
            task,
            last_response,
        )

        sub_agent.update_episodic_memory(
            orchestration_step.instructions,
            last_response,
            feedback_score,
            feedback_text,
        )

        sub_agent.update_procedural_memory(
            orchestration_step.instructions,
            feedback_score,
            feedback_text,
        )
        
        try:
            base_name = sub_agent.name.lower().replace(" ", "_")
            episodic_path = os.path.join("./agents/episodic", f"{base_name}_episodic.json")
            procedural_path = os.path.join("./agents/procedural", f"{base_name}_procedural.json")
            config_path = os.path.join("./agents", f"{base_name}.json")
            
            episodic_data = sub_agent.episodic_memory_store.export_memories()
            with open(episodic_path, "w") as f:
                json.dump(episodic_data, f, indent=2)
                
            procedural_data = sub_agent.procedural_memory_store.export_skills()
            with open(procedural_path, "w") as f:
                json.dump(procedural_data, f, indent=2)
                
            # Update system prompt and save to config file
            updated_system_prompt = sub_agent.update_prompt(
                sub_agent.system_prompt,
                sub_agent.procedural_memory_store,
            )
            
            with open(config_path, "r") as f:
                config_data = json.load(f)
            
            config_data["system_prompt"] = updated_system_prompt
            
            with open(config_path, "w") as f:
                json.dump(config_data, f, indent=2)
                
            print(f"Updated system prompt in config file: {config_path}")
                
        except Exception as e:
            print(f"Error saving memory or updating system prompt for agent {sub_agent.name}: {e}")

        # Update semantic memory if useful data library is found

        context_manager.add_context(orchestration_step.agent_name, last_response)

        tracer.update_agent_loop(
            orchestration_step.agent_name,
            last_response,
        )

    tracer.update_progress("Compiling final response...")
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

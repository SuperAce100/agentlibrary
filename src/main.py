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
from typing import Any
import base64  # For image encoding

from utils.tracing import Tracer
from utils.prompts import sub_agent_prompt


# Helper function to encode image to base64 - now local to main.py
def encode_image_to_base64(image_path: str) -> str | None:
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")
    except Exception as e:
        print(f"Error encoding image {image_path}: {e}")
        return None


def run(
    task: str,
    max_iterations: int = 100,
    verbose: bool = False,
    trace_path: str | None = None,
    image_inputs: list[str] | None = None,  # Expects a list of base64 encoded strings
    extend_final_system_prompt: str | None = None,
) -> str:
    """
    Run the multi-agent system. `image_inputs` should be a list of base64 encoded strings.
    """
    tracer = Tracer(task, trace_path, verbose)
    # print("Initializing...")

    # Store the initial base64 encoded images for lookup by index
    task_base64_image_store: list[str] = (
        image_inputs if image_inputs is not None else []
    )

    orchestrator = Orchestrator()

    initial_user_content_parts: list[dict[str, Any]] = [{"type": "text", "text": task}]
    if task_base64_image_store:  # Use the store for constructing initial message
        for base64_image_content in task_base64_image_store:
            if base64_image_content:
                image_url_data = f"data:image/jpeg;base64,{base64_image_content}"  # Assuming jpeg, can be dynamic
                initial_user_content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url_data, "detail": "auto"},
                    }
                )
            # else: No specific handling for empty base64 string, it just won't be added

    if orchestrator.messages and orchestrator.messages[0]["role"] == "system":
        orchestrator.messages.insert(
            1, {"role": "user", "content": initial_user_content_parts}
        )
    else:
        orchestrator.messages.append(
            {"role": "system", "content": orchestrator.system_prompt}
        )
        orchestrator.messages.append(
            {"role": "user", "content": initial_user_content_parts}
        )

    tracer.update_progress(
        "Decomposing task (Orchestrator has initial multimodal context if images provided)..."
    )

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
        if verbose:
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Creating sub-agents",
            ):
                sub_agents.append(future.result())
        else:
            for future in futures:
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
            last_agent_name,
            last_response,
            task,
            context_manager.get_context_names(),
            image_inputs=task_base64_image_store,  # Pass the full base64 list to orchestrator for its context and index generation
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
        sub_agent: Agent = agent_registry[orchestration_step.agent_name]
        memories = sub_agent.retrieve_episodic_memories(orchestration_step.instructions)
        if memories:
            relevant_memories = "\n\n".join(
                [f"Memory {i + 1}:\n{mem.content}" for i, mem in enumerate(memories)]
            )
        else:
            relevant_memories = "No relevant memories found."

        # Resolve image indices from orchestration_step to actual base64 strings
        resolved_images_for_agent: list[str] | None = None
        if orchestration_step.image_inputs is not None:  # Explicitly check for None
            image_indices_to_load = orchestration_step.image_inputs
            if image_indices_to_load:  # Check if the list is not empty
                resolved_images_for_agent = []
                for img_index in image_indices_to_load:
                    if 0 <= img_index < len(task_base64_image_store):
                        resolved_images_for_agent.append(
                            task_base64_image_store[img_index]
                        )
                    else:
                        if verbose:
                            print(
                                f"Warning: Orchestrator requested invalid image index {img_index}. Max index is {len(task_base64_image_store) - 1}. Skipping."
                            )
                if (
                    not resolved_images_for_agent
                ):  # If all indices were invalid or list became empty
                    resolved_images_for_agent = None
            # If image_indices_to_load was an empty list, resolved_images_for_agent remains None (or an empty list if initialized differently)
            # To ensure an empty list from orchestrator results in empty list for agent (not None):
            elif (
                not image_indices_to_load
            ):  # Handles case where orchestration_step.image_inputs was []
                resolved_images_for_agent = []

        last_response = sub_agent.call_with_tools(
            prompt_text=sub_agent_prompt.format(
                orchestrator_instructions=orchestration_step.instructions,
                context=relevant_context,
                memories=relevant_memories,
            ),
            image_base64_list=resolved_images_for_agent,  # Pass resolved base64 strings
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
            episodic_path = os.path.join(
                "./agents/episodic", f"{base_name}_episodic.json"
            )
            procedural_path = os.path.join(
                "./agents/procedural", f"{base_name}_procedural.json"
            )
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

            if verbose:
                print(f"Updated system prompt in config file: {config_path}")

        except Exception as e:
            print(
                f"Error saving memory or updating system prompt for agent {sub_agent.name}: {e}"
            )

        # Update semantic memory if useful data library is found

        context_manager.add_context(orchestration_step.agent_name, last_response)

        tracer.update_agent_loop(
            orchestration_step.agent_name,
            last_response,
        )

    tracer.update_progress("Compiling final response...")
    final_response = orchestrator.compile_final_response(
        task, extend_final_system_prompt=extend_final_system_prompt
    )

    tracer.update_progress("Done!")
    tracer.trace(final_response, "final_response")
    return final_response


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--output_path", type=str, default="results/traces")
    parser.add_argument(
        "--image_inputs",
        type=str,
        nargs="*",
        default=[],
        help="List of paths to image inputs for the task. These will be base64 encoded.",
    )
    args = parser.parse_args()

    task = args.task
    verbose = args.verbose
    output_path = args.output_path

    # Convert file paths from CLI to base64 strings before passing to run()
    image_input_paths = args.image_inputs
    base64_encoded_images: list[str] = []
    if image_input_paths:
        if verbose:
            print(f"Encoding {len(image_input_paths)} image(s) to base64...")
        for path in image_input_paths:
            encoded_img = encode_image_to_base64(path)
            if encoded_img:
                base64_encoded_images.append(encoded_img)
            else:
                if verbose:
                    print(
                        f"Warning: Could not encode image at path {path}. It will be skipped."
                    )

    result = run(
        task,
        verbose=verbose,
        trace_path=output_path,
        image_inputs=base64_encoded_images,
    )
    print(result)


def run_multimodal_test():
    """Runs a test of the system with dummy base64 multimodal (image) input."""
    print("\n--- Running Multimodal Test ---")
    dummy_image_filename_1 = "../.data/tests/test_base64_1.txt"
    dummy_image_filename_2 = "../.data/tests/test_base64_2.txt"

    image_files_to_test = [dummy_image_filename_1, dummy_image_filename_2]
    base64_test_images: list[str] = []

    try:
        for i, filename in enumerate(image_files_to_test):
            with open(filename, "r") as f:
                base64_test_images.append(f.read())

        test_task_multimodal = "Describe the contents of the provided images. What is in the first and second image data?"

        test_trace_path = "results/test_traces"
        os.makedirs(test_trace_path, exist_ok=True)

        print(
            f"Running multimodal test task: '{test_task_multimodal}' with {len(base64_test_images)} base64 images."
        )

        result = run(
            task=test_task_multimodal,
            verbose=True,
            trace_path=test_trace_path,
            image_inputs=base64_test_images,  # Pass the list of base64 strings
        )
        print("\n--- Multimodal Test Result ---")
        print(result)
        print("--- End of Multimodal Test ---")

    except Exception as e:
        print(f"Error during multimodal test: {e}")


if __name__ == "__main__":
    # Default execution path
    # main()

    # To run the test, you might comment out main() and call run_multimodal_test()
    # Or, add a command-line argument to choose between them.
    # For now, let's call main() and then the test if no specific args are given,
    # or provide a simple way to run just the test.

    # A simple way to control test execution:
    # If you run `python src/main.py --test_multimodal`, it will run the test.
    # Otherwise, it will run the main function as per other arguments.

    # Re-parsing args to check for a test flag without interfering with main's arg parsing
    import sys

    if "--test_multimodal" in sys.argv:
        run_multimodal_test()
    else:
        main()

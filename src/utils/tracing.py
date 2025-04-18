from datetime import datetime
import json
import os
from models import llm_call


class Tracer:
    def __init__(
        self, task: str, output_path: str | None = None, verbose: bool = False
    ):
        self.task = task

        self.output_path = None
        if isinstance(output_path, str):
            os.makedirs(output_path, exist_ok=True)
            self.output_path = os.path.join(output_path, self._name_file())

        self.verbose = verbose
        self.steps: dict[str, str | list[dict[str, str]]] = {
            "progress": "",
            "agent_loop": [],
        }

    def _name_file(self) -> str:
        return (
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "_"
            + llm_call(
                f"Please give a very short but descriptive name (30 characters or less) for the following task: {self.task} in snake case",
                model="openai/gpt-4.1-nano",
            )
            + ".json"
        )

    def update_progress(self, text: str):
        self.steps["progress"] = text
        if self.verbose:
            print("================================================================")
            print(text)
            print("================================================================")
        if self.output_path:
            with open(self.output_path, "w") as f:
                json.dump(self.steps, f, indent=2)

    def trace(self, text: str, label: str = ""):
        self.steps[label] = text

        if self.verbose:
            print(f"{label}: {text}")

        if self.output_path:
            with open(self.output_path, "w") as f:
                f.write(json.dumps(self.steps, indent=2))

    def update_agent_loop(self, agent_name: str, response: str):
        if isinstance(self.steps["agent_loop"], list):
            self.steps["agent_loop"].append(
                {"agent_name": agent_name, "response": response}
            )
        else:
            self.steps["agent_loop"] = [
                {"agent_name": agent_name, "response": response}
            ]
        if self.verbose:
            print(f"{agent_name} response: {response}")
        if self.output_path:
            with open(self.output_path, "w") as f:
                f.write(json.dumps(self.steps, indent=2))

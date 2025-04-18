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
        self.steps = {"progress": ""}

    def _name_file(self) -> str:
        return (
            llm_call(
                f"Please give a very short but descriptive name (30 characters or less) for the following task: {self.task} in snake case",
                model="openai/gpt-4.1-nano",
            )
            + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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

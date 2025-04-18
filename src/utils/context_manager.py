from pydantic import BaseModel
from models import llm_call


class ContextEntry(BaseModel):
    name: str
    creator: str
    content: str


class ContextManager:
    def __init__(self):
        self.context: dict[str, ContextEntry] = {}

    def _name_entry(self, creator: str, content: str) -> str:
        return llm_call(
            f"Please give a name for the following context entry in the following format: `original_research`: {content}",
            model="openai/gpt-4.1-nano",
        )

    def add_context(self, creator: str, content: str):
        name = self._name_entry(creator, content)
        self.context[name] = ContextEntry(name=name, creator=creator, content=content)
        return name

    def get_context(self, name: str) -> str:
        return self.context[name].content

    def get_all_context(self) -> dict[str, str]:
        return {name: self.get_context(name) for name in self.context.keys()}

    def get_context_names(self) -> list[str]:
        return list(self.context.keys())


if __name__ == "__main__":
    context_manager = ContextManager()
    context_manager.add_context("An illustrous lawyer", "Some useful legal information")
    context_manager.add_context("A smart surgeon", "Some medical information")

    print(context_manager.get_all_context())

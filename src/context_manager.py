from pydantic import BaseModel


class ContextEntry(BaseModel):
    name: str
    creator: str
    content: str


class ContextManager:
    def __init__(self):
        self.context = {}

    def _name_entry(self, creator: str, content: str) -> str:
        return "untested"

    def add_context(self, creator: str, content: str):
        self.context.append(
            ContextEntry(
                name=self._name_entry(creator, content),
                creator=creator,
                content=content,
            )
        )

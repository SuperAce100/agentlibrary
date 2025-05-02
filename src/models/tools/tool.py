from typing import Callable, Any
from pydantic import BaseModel


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable[..., str],
        argument_schema: BaseModel,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.argument_schema = argument_schema

    def __call__(self, **kwargs) -> str:
        if not self.argument_schema.model_validate(kwargs):
            return f"Invalid arguments: {kwargs}"
        return self.function(**kwargs)

    def __str__(self) -> str:
        return f"{self.name}: {self.description}\nArguments: {self.argument_schema}"

    def to_openai_tool(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.argument_schema.model_json_schema(),
            },
        }


if __name__ == "__main__":

    class TestArgumentSchema(BaseModel):
        x: int

    test_tool = Tool(
        name="test",
        description="Test tool",
        function=lambda x: str(x + 2) + " test",
        argument_schema=TestArgumentSchema,
    )
    print(test_tool(x=1))

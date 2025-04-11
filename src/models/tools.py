from typing import Callable
from pydantic import BaseModel


class Tool:
    def __init__(
        self,
        name: str,
        description: str,
        function: Callable,
        argument_schema: BaseModel,
    ):
        self.name = name
        self.description = description
        self.function = function
        self.argument_schema = argument_schema

    def __call__(self, arguments: dict) -> str:
        if not self.argument_schema.model_validate(arguments):
            return f"Invalid arguments: {arguments}"
        return self.function(**arguments)

    def __str__(self) -> str:
        return f"{self.name}: {self.description}\nArguments: {self.argument_schema}"


if __name__ == "__main__":

    class TestArgumentSchema(BaseModel):
        x: int

    test_tool = Tool(
        name="test",
        description="Test tool",
        function=lambda x: str(x + 2) + " test",
        argument_schema=TestArgumentSchema,
    )
    print(test_tool({"x": 1}))

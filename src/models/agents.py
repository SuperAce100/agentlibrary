from models.models import llm_call_messages, llm_call_messages_async, text_model
from models.tools import Tool
import asyncio
from pydantic import BaseModel


class AgentConfig(BaseModel):
    name: str
    system_prompt: str


class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Tool],
        model: str = text_model,
    ):
        self.name: str = name
        self.system_prompt: str = system_prompt
        self.model: str = model
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.tools: list[Tool] = tools
        self.data: list[str] = []

    @staticmethod
    def from_config(config: AgentConfig, model: str = text_model) -> "Agent":
        return Agent(config.name, config.system_prompt, [], model)

    def pass_context(self, context: str, role: str = "user") -> None:
        self.messages.append({"role": role, "content": context})

    def call(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        response = llm_call_messages(self.messages, model=self.model)
        self.messages.append({"role": "assistant", "content": response})
        return str(response)

    async def call_async(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        response = await llm_call_messages_async(self.messages, model=self.model)
        self.messages.append({"role": "assistant", "content": response})
        return str(response)

    def __str__(self) -> str:
        return f"Agent: {self.name}\nSystem Prompt: {self.system_prompt}\nTools: {self.tools}\nModel: {self.model}\nMessages: {self.messages}\nData: {self.data}"


if __name__ == "__main__":
    agent = Agent(
        name="Test123",
        system_prompt="You are a helpful assistant that speaks in the style of a pirate.",
        tools=[],
        model=text_model,
    )
    print(agent.call("What is the capital of France?"))
    print(asyncio.run(agent.call_async("What was the last thing you said?")))

    config = AgentConfig(
        name="Test1234",
        system_prompt="You are a helpful assistant that speaks in haikus.",
    )

    agent2 = Agent.from_config(config)
    print(agent2.call("What is the capital of Italy?"))
    print(asyncio.run(agent2.call_async("What was the last thing you said?")))

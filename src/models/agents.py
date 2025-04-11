from models import llm_call_messages, llm_call_messages_async
from tools import Tool
import asyncio

class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Tool],
        model: str = "openrouter/optimus-alpha",
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.tools = tools
        self.data = []

    def call(self, prompt: str):
        self.messages.append({"role": "user", "content": prompt})
        response = llm_call_messages(self.messages, model=self.model)
        self.messages.append({"role": "assistant", "content": response})
        return response

    async def call_async(self, prompt: str):
        self.messages.append({"role": "user", "content": prompt})
        response = await llm_call_messages_async(self.messages, model=self.model)
        self.messages.append({"role": "assistant", "content": response})
        return response

if __name__ == "__main__":
    agent = Agent(
        name="Test123",
        system_prompt="You are a helpful assistant that speaks in the style of a pirate.",
        tools=[],
        model="openrouter/optimus-alpha"
    )
    print(agent.call("What is the capital of France?"))
    print(asyncio.run(agent.call_async("What was the last thing you said?")))

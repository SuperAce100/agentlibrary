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
        self.name = name
        self.system_prompt = system_prompt
        self.model = model
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.tools = tools
        self.data = []

    @staticmethod
    def from_config(config: AgentConfig, model: str = text_model):
        return Agent(config.name, config.system_prompt, [], model)

    def pass_context(self, context: str, role: str = "user"):
        self.messages.append({"role": role, "content": context})

    def call(self, prompt: str, response_format: BaseModel = None):
        self.messages.append({"role": "user", "content": prompt})
        response = llm_call_messages(self.messages, model=self.model, response_format=response_format)
        self.messages.append({"role": "assistant", "content": response})
        return response

    async def call_async(self, prompt: str, response_format: BaseModel = None):
        self.messages.append({"role": "user", "content": prompt})
        response = await llm_call_messages_async(self.messages, model=self.model, response_format=response_format)
        self.messages.append({"role": "assistant", "content": response})
        return response
    
    def __str__(self):
        return f"Agent: {self.name}\nSystem Prompt: {self.system_prompt}\nTools: {self.tools}\nModel: {self.model}\nMessages: {self.messages}\nData: {self.data}"

if __name__ == "__main__":
    agent = Agent(
        name="Test123",
        system_prompt="You are a helpful assistant that speaks in the style of a pirate.",
        tools=[],
        model=text_model
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

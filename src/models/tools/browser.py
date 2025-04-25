import os
import asyncio
from langchain_openai import ChatOpenAI
from browser_use import Agent
from dotenv import load_dotenv
from pydantic import BaseModel
from models.tools import Tool

load_dotenv()


llm = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)


def _run_browser(task: str) -> str:
    agent = Agent(
        task=task,
        llm=llm,
    )
    result = asyncio.run(agent.run())
    if result.is_successful():
        return result.final_result()
    else:
        return "Failed to complete the task."


class BrowserArgs(BaseModel):
    task: str


browser_tool = Tool(
    name="browser",
    description="Instruct a browser to perform a very simple web task (Find x related to y, etc.)",
    function=_run_browser,
    argument_schema=BrowserArgs,
)

if __name__ == "__main__":
    result = browser_tool(task="Find the weather in Stanford")
    print(result)

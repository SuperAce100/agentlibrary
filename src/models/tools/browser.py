import os
import asyncio
import signal
import logging
from contextlib import contextmanager
from langchain_openai import ChatOpenAI
from browser_use import Agent
from dotenv import load_dotenv
from pydantic import BaseModel
from models.tools import Tool

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Configure the LLM with timeout
llm = ChatOpenAI(
    model="openai/gpt-4.1-mini",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    timeout=60,  # Add timeout for API calls
)

class TimeoutException(Exception):
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def _run_browser(task: str) -> str:
    logger.info(f"Starting browser task: {task}")
    try:
        with time_limit(300):  # 5-minute timeout
            agent = Agent(
                task=task,
                llm=llm,
                max_iterations=5,  # Limit iterations to prevent excessive API calls
            )
            logger.info("Agent created, starting run")
            result = asyncio.run(agent.run())
            logger.info("Agent run completed")
            
            if result.is_successful():
                logger.info("Task completed successfully")
                return result.final_result()
            else:
                logger.warning("Task failed")
                return "Failed to complete the task."
    except TimeoutException:
        logger.error("Browser task timed out after 5 minutes")
        return "Browser task timed out after 5 minutes."
    except Exception as e:
        logger.error(f"Error in browser tool: {str(e)}")
        return f"Error using browser tool: {str(e)}"

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

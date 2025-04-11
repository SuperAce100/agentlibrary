from models.agents import Agent
from decomposition import decompose_task
from sub_agent_creation import create_sub_agent
from planning import plan_task
from execution import execute_sub_agents
from utils.writing import clean_up_document

def run_symphony(task: str) -> str:
    """
    Run the symphony
    """
    decomposition = decompose_task(task)
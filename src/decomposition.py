from typing import Dict
from pydantic import BaseModel
from models.models import llm_call

class SubAgentDescription(BaseModel):
    name: str
    description: str
    justification: str

class Decomposition(BaseModel):
    sub_agents: list[SubAgentDescription]

    def __str__(self):
        return "\n".join([f"Agent: {agent.name}\nDescription: {agent.description}\nJustification: {agent.justification}\n" for agent in self.sub_agents])

planner_system_prompt = """
You are a planner agent. You will be given a task, and you need to plan the best way to complete the task by assembling a team of sub-agents.

Your job is to break down the following task into different skills, each of which can be performed by a different sub-agent.
Think of assembling a team to complete the task - what are the different roles that need to be filled?
Prefer making a small number of agents that can each do one thing well. Don't add agents that are not explicitly required to complete the task.

When making agents, make sure they are performing general tasks that can be performed by any agent, not specific to the task at hand. For example, if the task is to write a paper about AI, don't make an agent that is only able to write about AI, but rather a general writer agent.

For each agent, provide a name, description, and justification. Your justification should be an exact, minimal,quote from the task description that inspired you to create this agent. If there is no particular part of the task description that inspired you to create this agent, leave it blank.
"""

def decompose_task(task: str) -> Dict[str, str]:
    response = llm_call(
        f"Task: {task}",
        system_prompt=planner_system_prompt,
        response_format=Decomposition
    )
    return response


if __name__ == "__main__":
    decomposition = decompose_task("Create a comprehensive business proposal for an airline focused on connecting China and the West Coast of the US. Consider the idea’s financial viability, any potential legal challenges, the state of the market, and brand building potential.")
    print(decomposition)

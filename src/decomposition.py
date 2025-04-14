import json
import os
from pydantic import BaseModel
from models.models import llm_call


class SubAgentDescription(BaseModel):
    name: str
    description: str
    justification: str


class Decomposition(BaseModel):
    sub_agents: list[SubAgentDescription]

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Agent: {agent.name}\nDescription: {agent.description}\nJustification: {agent.justification}\n"
                for agent in self.sub_agents
            ]
        )


planner_system_prompt = """
You are a planner agent. You will be given a task, and you need to plan the best way to complete the task by assembling a team of sub-agents.
You will also be given a list of available agents. Use the agents that have been provided to you, but you can also add new agents that are not in the initial list.
Your job is to break down the following task into different skills, each of which can be performed by a different sub-agent.
Think of assembling a team to complete the task - what are the different roles that need to be filled?
Prefer making a small number of agents that can each do one thing well. Don't add agents that are not explicitly required to complete the task.

When making agents, make sure they are performing general tasks that can be performed by any agent, not specific to the task at hand. For example, if the task is to write a paper about AI, don't make an agent that is only able to write about AI, but rather a general writer agent. Never mention the task at hand in the description.

For each agent, provide a name, description, and justification. Your justification should be an exact, minimal, quote from the task description that inspired you to create this agent. If there is no particular part of the task description that inspired you to create this agent, leave it blank.

If you pick agents from the list of available agents, make sure their names are exactly as they are in the list. For their descriptions, just mention that they are from the list of available agents. Do their justifications as usual. If an agent is not in the list of available agents, create a full name and description for it. Analyse the prompt deeply and create new agents liberally when needed.
"""


def decompose_task(task: str, library_path: str = "agents") -> Decomposition:
    available_agents = []
    for file in os.listdir(library_path):
        if file.endswith(".json"):
            with open(os.path.join(library_path, file), "r") as f:
                agent_data = json.loads(f.read())
                available_agents.append(agent_data["name"])

    available_agents_str = "\n".join(available_agents)

    response: Decomposition = Decomposition.model_validate(
        llm_call(
            f"Task: {task}\n\nAvailable agents:\n{available_agents_str if available_agents_str else 'No agents created yet! Create them all yourself!'}",
            system_prompt=planner_system_prompt,
            response_format=Decomposition,
        )
    )
    return response


if __name__ == "__main__":
    decomposition = decompose_task(
        "Create a comprehensive business proposal for an airline focused on connecting China and the West Coast of the US. Consider the idea’s financial viability, any potential legal challenges, the state of the market, and brand building potential."
    )
    print(decomposition)

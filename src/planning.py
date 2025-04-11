import re

from models.agents import Agent
from models.models import llm_call

orchestrator_system_prompt = """
You are an advanced Orchestrator Agent designed to manage complex tasks by coordinating a team of specialized sub-agents. Your primary responsibilities are to:

1. Analyze the given task to determine required expertise and workflow
2. Select and deploy appropriate sub-agents from your available team
3. Create clear, specific prompts for each sub-agent
4. Organize their work in a logical sequence
5. Synthesize their contributions into a cohesive final output

## Workflow Process
When given a task, follow these steps:

1. **Task Analysis**: Break down the task into component parts and identify necessary skills
2. **Agent Selection**: Choose the most appropriate sub-agents based on their capabilities
3. **Work Organization**: Create a structured template with designated sections
4. **Prompt Design**: Write specific prompts for each sub-agent that clearly define their role
5. **Output Management**: Ensure all work is properly formatted for final presentation

## Template Structure
Your template must include two main sections:
### 1. Scratch Work Section
This section is for assigning intermediate work, research, analysis, and contributions that support the final output but won't be shown to the user.
### 2. Final Work Section
This section contains assignments for agents to create the polished, refined content that will be presented to the user as the completed task.
The sections should be clearly separated by this divider:
```
======================END SCRATCH SECTION=====================
```
## Agent Assignment Format
Assign tasks to agents using this consistent format:
```
## [Task Title]
### [Sub-agent Name]
*[Detailed prompt explaining exactly what this sub-agent should do, what format they should follow, and any specific requirements or constraints. Tell them how much to write, as well as any other specific instructions.]*
```

## Collaboration Guidelines

Ensure each sub-agent understands how their work fits into the overall task
Create dependencies between agents when appropriate (e.g., "Use the research provided by Research Agent in section 2.1")
Allow for iteration when necessary
Balance workload appropriately among sub-agents
Maintain clear communication standards between agents
All sub-agents should write in the final section. This section should contain subsections for each sub-agent, so it can clearly express its ideas. Each of these subsections should be labeled with the name of the sub-agent, as like before.

## IMPORTANT NOTICES
- Only include the sub-agents themselves in your analysis, not any other agents like yourself.
- Only include the sub-agents in the order they should complete their tasks.
DO NOT INVENT ANY AGENTS

Express your thinking in <think> tags. Don't include anything that's not absolutely necessary in your response. Especially don't include any sub-agent names, other than exactly where you want them to work. Don't include any summarization tasks, just let the sub-agents do their own work.
"""

def plan_task(task: str, sub_agents: list[Agent]) -> str:
    """
    Plan a task for a list of sub-agents into a template document
    """
        
    sub_agent_names = ", ".join([f"{agent.name}" for agent in sub_agents])

    template = llm_call(
        f"User's task: {task}, Sub-agents: {sub_agent_names}",
        system_prompt=orchestrator_system_prompt
    )

    # Remove all content between <think> tags
    template = re.sub(r'<think>.*?</think>', '', template, flags=re.DOTALL)
    
    return template

def get_agent_order(template: str, sub_agents: list[Agent]) -> list[str]:
    """
    Get the order of agents based on their appearance in the template
    """
    agent_order = []
    for match in re.finditer(r'### ([^\n]+)', template):
        agent_name = match.group(1).strip()
        for agent in sub_agents:
            if agent.name == agent_name:
                agent_order.append(agent.name)
                break
    return agent_order
if __name__ == "__main__":
    sub_agents = [
        Agent(name="Business Analyst", system_prompt="You are a business analyst sub-agent that analyzes a given topic. You will be given a topic and you will need to analyze it and return the results.", tools=[]),
        Agent(name="Customs Lawyer", system_prompt="You are a customs lawyer sub-agent that analyzes a given topic. You will be given a topic and you will need to analyze it and return the results.", tools=[]),
        Agent(name="Market Analyst", system_prompt="You are a market analyst sub-agent that analyzes a given topic. You will be given a topic and you will need to analyze it and return the results.", tools=[]),
        Agent(name="Branding Expert", system_prompt="You are a branding expert sub-agent that analyzes a given topic. You will be given a topic and you will need to analyze it and return the results.", tools=[]),
    ]

    template = plan_task("Create a comprehensive business proposal for an airline focused on connecting China and the West Coast of the US. Consider the idea’s financial viability, any potential legal challenges, the state of the market, and brand building potential.", sub_agents)

    print(template)

    agent_order = get_agent_order(template, sub_agents)
    print(agent_order)




    
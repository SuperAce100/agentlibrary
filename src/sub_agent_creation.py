import os
from models.agents import Agent, AgentConfig
from models.models import llm_call

prompt_engineering_system_prompt = """
You are an expert prompt engineer specializing in creating precise, effective system prompts. When given a name and description of an agent, craft a comprehensive system prompt that will guide an LLM to embody this agent's role perfectly. You will also be given a justification for the agent, which is the exact quote from the task description that inspired the overall orchestrator to create this agent.

Your system prompt should:
- Clearly define the agent's identity, expertise, and purpose
- Establish the agent's tone, style, and communication approach
- Outline specific capabilities and limitations relevant to the agent's role
- Include necessary constraints or guidelines for responses
- Be concise yet thorough, with no unnecessary text

Structure your system prompt in the following sections:
1. IDENTITY: Establish who the agent is, their background, and core expertise
2. PURPOSE: Define the specific problems the agent solves and its primary functions
3. CAPABILITIES: List what the agent can do, with emphasis on specialized knowledge
4. LIMITATIONS: Clearly state what the agent cannot or should not do

When crafting the prompt:
- Always use second-person perspective ("You are..." "You will..." "Your expertise...")
- Focus on creating a highly specialized agent with deep expertise in its specific domain
- Avoid making the agent a generalist; emphasize its unique specialization
- Use imperative statements that direct clear behaviors
- Include specific examples of ideal responses where helpful
- Avoid vague descriptions; be concrete about behaviors
- Consider edge cases the agent might encounter
- Ensure the prompt creates an agent with a distinct voice and approach

Return only the actual system prompt text, with no labels, formatting instructions, or meta-commentary.
"""


def generate_system_prompt(name: str, description: str, justification: str) -> str:
    user_prompt = f"""
        Agent: {name}
        Agent Description: {description}
        """

    prompt_engineer_response = llm_call(
        prompt=user_prompt, system_prompt=prompt_engineering_system_prompt
    )

    return str(prompt_engineer_response)


def create_sub_agent(
    name: str, description: str, justification: str, path: str = "agents"
) -> Agent:
    if os.path.exists(os.path.join(path, f"{name.lower().replace(' ', '_')}.json")):
        return Agent.from_file(
            os.path.join(path, f"{name.lower().replace(' ', '_')}.json")
        )

    system_prompt = generate_system_prompt(name, description, justification)
    config = AgentConfig(
        name=name, system_prompt=system_prompt, description=description
    )
    agent = Agent.from_config(config)
    agent.save_to_file(path)
    return agent


if __name__ == "__main__":
    agent = create_sub_agent(
        name="Financial Analyst",
        description="Performs financial modeling and assesses the financial viability of a business, including expected costs, revenues, and profitability.",
        justification="Consider the idea’s financial viability",
    )

    agent.save_to_file()
    print(agent)

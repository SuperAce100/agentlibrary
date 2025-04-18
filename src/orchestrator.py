from pydantic import BaseModel

from models import Agent

from utils.prompts import (
    orchestrator_system_prompt,
    orchestrator_planning_prompt,
    orchestrator_pre_survey_prompt,
    past_response_format,
    orchestrator_react_prompt,
    orchestrator_final_response_prompt,
)


class OrchestrationStep(BaseModel):
    reasoning: str
    is_done: bool
    agent_name: str
    instructions: str
    context: list[str]


class Orchestrator(Agent):
    def __init__(self):
        super().__init__(
            name="Orchestrator",
            system_prompt=orchestrator_system_prompt,
            # model="openai/o4-mini",
            model="openai/gpt-4.1-mini",
            description="The orchestrator in charge of everything",
        )

    def pre_survey(self, task: str) -> str:
        return self.call(orchestrator_pre_survey_prompt.format(task=task))

    def plan(self, task: str, sub_agents: list[Agent]) -> str:
        return self.call(
            orchestrator_planning_prompt.format(task=task, sub_agents=sub_agents)
        )

    def orchestrate(
        self, last_sub_agent: str, response: str, task: str, context: list[str]
    ) -> OrchestrationStep:
        past_response = (
            past_response_format.format(last_sub_agent, response)
            if len(response) > 0
            else ""
        )

        return self.call_structured_output(
            orchestrator_react_prompt.format(
                response=past_response, task=task, context="\n".join(context)
            ),
            schema=OrchestrationStep,
        )

    def compile_final_response(self, task: str) -> str:
        return self.call(orchestrator_final_response_prompt.format(task=task))

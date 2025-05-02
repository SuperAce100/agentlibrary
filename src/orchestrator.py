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

    # def __str__(self) -> str:
    #     return f"Reasoning: {self.reasoning}\nIs done: {self.is_done}\nAgent name: {self.agent_name}\nInstructions: {self.instructions}\nContext: {', '.join(self.context)}"


class Orchestrator(Agent):
    def __init__(self):
        super().__init__(
            name="Orchestrator",
            system_prompt=orchestrator_system_prompt,
            model="openai/o4-mini",
            # model="openai/gpt-4.1-mini",
            description="The orchestrator in charge of everything",
        )

    def pre_survey(self, task: str) -> str:
        return self.call(orchestrator_pre_survey_prompt.format(task=task))

    def plan(self, task: str, sub_agents: list[Agent]) -> str:
        sub_agents_str = "\n".join(
            [
                f"{i}. {agent.name}: {agent.description}"
                for i, agent in enumerate(sub_agents)
            ]
        )
        return self.call(
            orchestrator_planning_prompt.format(task=task, sub_agents=sub_agents_str)
        )

    def orchestrate(
        self, last_sub_agent: str, response: str, task: str, context: list[str]
    ) -> OrchestrationStep:
        past_responses = (
            past_response_format.format(
                sub_agent_name=last_sub_agent, past_response=response
            )
            if len(response) > 0
            else ""
        )

        return self.call_structured_output(
            orchestrator_react_prompt.format(
                past_responses=past_responses, task=task, context="\n".join(context)
            ),
            schema=OrchestrationStep,
        )

    def compile_final_response(self, task: str) -> str:
        return self.call(orchestrator_final_response_prompt.format(task=task))
        
    def give_feedback(
        self,
        agent_name: str,
        instructions: str,
        task: str,
        response: str,
    ) -> tuple[float, str]:
        """
        Evaluates a sub-agent's response and provides feedback using metrics from feedback.py.
        
        Args:
            agent_name: The name of the sub-agent being evaluated
            instructions: The instructions given to the sub-agent
            task: The overall task being worked on
            response: The sub-agent's response to evaluate
            
        Returns:
            A tuple containing (feedback_score, feedback_text)
        """
        from models.feedback import iterative_metric_creation, analyze_interaction
        import asyncio
        
        # Create an interaction object that represents this exchange
        interaction = {
            "task": task,
            "instructions": instructions,
            "agent_name": agent_name,
            "user": instructions,  # The orchestrator's instructions are the "user" input
            "assistant": response  # The sub-agent's response
        }
        
        # Get metrics for this type of interaction
        # We'll run this synchronously even though the function is async
        metrics = asyncio.run(iterative_metric_creation(interaction))
        
        # Build a comprehensive evaluation prompt using these metrics
        metrics_prompts = []
        for metric in metrics:
            metrics_prompts.append(f"## {metric.name}\n{metric.description}\n{metric.evaluation_prompt}")
        
        metrics_text = "\n\n".join(metrics_prompts)
        
        feedback_prompt = f"""
        You are evaluating the response of the sub-agent "{agent_name}" to determine how well they followed instructions and contributed to the overall task.
        
        OVERALL TASK: {task}
        
        INSTRUCTIONS GIVEN TO SUB-AGENT: {instructions}
        
        SUB-AGENT RESPONSE: {response}
        
        Please evaluate the response based on the following metrics:
        
        {metrics_text}
        
        For each metric, provide a score from 0.0 to 1.0 and brief justification.
        Then provide an overall score and comprehensive feedback.
        
        Format your response exactly as:
        
        # Metric Evaluations
        [Evaluate each metric with score and justification]
        
        # Overall Score: [numerical score between 0.0 and 1.0]
        
        # Feedback:
        [Your detailed feedback explaining the overall assessment]
        """
        
        feedback_response = self.call(feedback_prompt)
        
        # Parse the score and feedback text from the response
        import re
        score_match = re.search(r"Overall Score:\s*([0-9.]+)", feedback_response, re.IGNORECASE)
        feedback_match = re.search(r"Feedback:(.*?)($|#)", feedback_response, re.DOTALL | re.IGNORECASE)
        
        score = 0.5  # Default score if parsing fails
        feedback_text = feedback_response  # Default to full response if parsing fails
        
        if score_match:
            try:
                score = float(score_match.group(1))
                score = max(0.0, min(1.0, score))  # Ensure score is between 0 and 1
            except ValueError:
                pass
                
        if feedback_match:
            feedback_text = feedback_match.group(1).strip()
        else:
            # Try to extract just the feedback section if the regex failed
            sections = feedback_response.split('#')
            for section in sections:
                if 'feedback' in section.lower():
                    feedback_text = section.replace('Feedback:', '').strip()
                    break
            
        return score, feedback_text
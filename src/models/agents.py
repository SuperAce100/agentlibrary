import os
from models.llms import (
    llm_call,
    llm_call_messages_async,
    llm_call_messages,
    text_model,
    llm_call_with_tools,
)
from models.tools import Tool
from models.memory import (
    ProceduralMemoryStore,
    EpisodicMemory,
    EpisodicMemoryStore,
    SemanticMemory,
    SemanticMemoryStore,
)
import random
from pydantic import BaseModel
from typing import Optional, Dict, Any, Tuple, List
import json
import re
from models.tools import Tool, tool_registry, browser_tool, terminal_tool
import asyncio
from pydantic import BaseModel


class AgentConfig(BaseModel):
    name: str
    system_prompt: str
    description: str
    messages: list[dict[str, str]] = []
    tools: list[str] = []


class Agent:
    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list[Tool] = [],
        model: str = text_model,
        description: str = "",
        initial_skills: Optional[Dict[str, float]] = None,
        initial_episodic_data: Optional[List[Dict[str, Any]]] = None,
        initial_procedural_data: Optional[Dict[str, Dict[str, Any]]] = None,
        initial_semantic_data: Optional[List[Dict[str, Any]]] = None,
    ):
        self.name: str = name
        self.system_prompt: str = system_prompt
        self.model: str = model
        self.messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]
        self.tools: list[Tool] = tools
        self.data: list[str] = []
        self.description: str = description

        self.episodic_memory_store = EpisodicMemoryStore()
        self.procedural_memory_store = ProceduralMemoryStore()
        self.semantic_memory_store = SemanticMemoryStore()

        if initial_episodic_data:
            self.episodic_memory_store.import_memories(initial_episodic_data)
        if initial_procedural_data:
            self.procedural_memory_store.import_skills(initial_procedural_data)
        if initial_semantic_data:
            self.semantic_memory_store.import_memories(initial_semantic_data)

    @staticmethod
    def from_config(
        config: AgentConfig,
        model: str = text_model,
        initial_episodic_data: Optional[List[Dict[str, Any]]] = None,
        initial_procedural_data: Optional[Dict[str, Dict[str, Any]]] = None,
        initial_semantic_data: Optional[List[Dict[str, Any]]] = None,
    ) -> "Agent":
        agent = Agent(
            name=config.name,
            system_prompt=config.system_prompt,
            model=model,
            description=config.description,
            initial_episodic_data=initial_episodic_data,
            initial_procedural_data=initial_procedural_data,
            initial_semantic_data=initial_semantic_data,
            tools=[
                tool
                for tool_name in config.tools
                if (tool := tool_registry.get_tool(tool_name))
            ],
        )
        for message in config.messages:
            agent.pass_context(message["content"], message["role"])
        return agent

    @staticmethod
    def from_file(path: str) -> "Agent":
        config_path = path
        base_path = os.path.splitext(config_path)[0]
        episodic_path = f"{base_path}_episodic.json"
        procedural_path = f"{base_path}_procedural.json"
        semantic_path = f"{base_path}_semantic.json"
        try:
            with open(config_path, "r") as f:
                config = AgentConfig.model_validate_json(f.read())
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Agent configuration file not found: {config_path}"
            )
        except Exception as e:
            raise IOError(
                f"Error reading or parsing agent config file {config_path}: {e}"
            )

        loaded_episodic_data = None
        try:
            with open(episodic_path, "r") as f:
                loaded_episodic_data = json.load(f)
            print(f"Loaded episodic memory from: {episodic_path}")
        except FileNotFoundError:
            print(f"Episodic memory file not found (agent may be new): {episodic_path}")
        except Exception as e:
            print(f"Error reading or parsing episodic memory file {episodic_path}: {e}")

        loaded_procedural_data = None
        try:
            with open(procedural_path, "r") as f:
                loaded_procedural_data = json.load(f)
            print(f"Loaded procedural memory from: {procedural_path}")
        except FileNotFoundError:
            print(
                f"Procedural memory file not found (agent may be new): {procedural_path}"
            )
        except Exception as e:
            print(
                f"Error reading or parsing procedural memory file {procedural_path}: {e}"
            )

        loaded_semantic_data = None
        try:
            with open(semantic_path, "r") as f:
                loaded_semantic_data = json.load(f)
            print(f"Loaded semantic memory from: {semantic_path}")
        except FileNotFoundError:
            print(f"Semantic memory file not found (agent may be new): {semantic_path}")
            
        return Agent.from_config(
            config,
            initial_episodic_data=loaded_episodic_data,
            initial_procedural_data=loaded_procedural_data,
        )

    def save_to_file(self, path: str = "agents") -> None:
        """Save the agent's configuration and memory state to files."""
        os.makedirs(path, exist_ok=True)
        
        # Create subdirectories for memory types if they don't exist
        episodic_dir = os.path.join(path, "episodic")
        procedural_dir = os.path.join(path, "procedural")
        semantic_dir = os.path.join(path, "semantic")
        
        os.makedirs(episodic_dir, exist_ok=True)
        os.makedirs(procedural_dir, exist_ok=True)
        os.makedirs(semantic_dir, exist_ok=True)

        config = AgentConfig(
            name=self.name,
            system_prompt=self.system_prompt,
            messages=self.messages[1:],
            description=self.description,
            tools=[tool.name for tool in self.tools],
        )
        base_name = self.name.lower().replace(" ", "_")
        config_path = os.path.join(path, f"{base_name}.json")
        episodic_path = os.path.join(episodic_dir, f"{base_name}_episodic.json")
        procedural_path = os.path.join(procedural_dir, f"{base_name}_procedural.json")

        try:
            with open(config_path, "w") as f:
                f.write(config.model_dump_json(indent=2))
            print(f"Agent config saved to: {config_path}")
        except Exception as e:
            print(f"Error saving agent config to {config_path}: {e}")

        try:
            episodic_data = self.episodic_memory_store.export_memories()
            with open(episodic_path, "w") as f:
                json.dump(episodic_data, f, indent=2)
            print(f"Episodic memory saved to: {episodic_path}")
        except Exception as e:
            print(f"Error saving episodic memory to {episodic_path}: {e}")

        try:
            procedural_data = self.procedural_memory_store.export_skills()
            with open(procedural_path, "w") as f:
                json.dump(procedural_data, f, indent=2)
            print(f"Procedural memory saved to: {procedural_path}")
        except Exception as e:
            print(f"Error saving procedural memory to {procedural_path}: {e}")

    def pass_context(self, context: str, role: str = "user") -> None:
        self.messages.append({"role": role, "content": context})

    def call(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        response = llm_call_messages(self.messages, model=self.model)
        self.messages.append({"role": "assistant", "content": response})
        return str(response)

    def call_structured_output(self, prompt: str, schema: BaseModel) -> BaseModel:
        self.messages.append({"role": "user", "content": prompt})
        response = llm_call_messages(
            self.messages, response_format=schema, model=self.model
        )
        self.messages.append({"role": "assistant", "content": str(response)})
        return schema.model_validate(response)

    def call_with_tools(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        response = llm_call_with_tools(self.messages, self.tools, model=self.model)
        self.messages.append({"role": "assistant", "content": response})
        return str(response)

    async def call_async(self, prompt: str) -> str:
        self.messages.append({"role": "user", "content": prompt})
        response = await llm_call_messages_async(self.messages, model=self.model)
        self.messages.append({"role": "assistant", "content": response})
        return str(response)

    def update_episodic_memory(
        self,
        user_input: str,
        agent_response: str,
        feedback_score: Optional[float] = None,
        feedback_text: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Adds a record of the interaction to the agent's episodic memory store.
        To be called after after agent.call.
        """
        memory_content = (
            f"User: '{user_input}' | Agent ({self.name}): '{agent_response}'"
        )
        self.episodic_memory_store.add_memory(
            content=memory_content,
            metadata=metadata,
            feedback_score=feedback_score,
            feedback_text=feedback_text,
        )
        print(f"--- Episodic memory updated for Agent {self.name} ---")

    def update_procedural_memory(
        self,
        skill_description: str,
        feedback_score: float,
        feedback_text: Optional[str] = None,
    ) -> None:
        """
        Updates the agent's procedural memory (skills/flaws) based on feedback.
        To be called after an interaction and evaluation.
        """
        self.procedural_memory_store.add_or_update_skill(
            skill_description=skill_description,
            new_score=feedback_score,
            evidence_text=feedback_text
            or f"Feedback received regarding '{skill_description}'.",
        )
        print(
            f"--- Procedural memory updated for Agent {self.name} regarding '{skill_description}' ---"
        )
    
    def update_prompt(
        self,
        old_prompt: str,
        skill_library: ProceduralMemoryStore, 
    ) -> str:
        """
        Args:
        old_prompt: The current system prompt of the agent.
        skill_library: The ProceduralMemoryStore containing the agent's skills.

        Returns:
            The rewritten system prompt, or the old prompt if an error occurs.
        """
        skill_summary = skill_library.get_summary(include_average=True)
        if not skill_summary or "no specific skills" in skill_summary:
            print("No significant skills found to update prompt.")
            return old_prompt  # No skills to base update on

        # Meta-prompt instructing the LLM how to rewrite the prompt
        meta_prompt = """
        You are an expert prompt engineer. Your task is to rewrite an agent's system prompt.
        The goal is to subtly incorporate the agent's learned skills and flaws into its core instructions,
        making the prompt more accurate to the agent's current capabilities without explicitly listing skills like a resume.
        Rewrite the original system prompt based *only* on the skill summary provided.
        - Integrate the strengths and weaknesses naturally into the agent's persona or instructions.
        - For example, if the agent is 'good at creative writing' but 'poor at math', the prompt might lean more towards creative tasks or mention a preference for words over numbers. If the agent is 'excellent at following instructions', reinforce that. If 'poor at speaking like a pirate', maybe tone down that instruction slightly or add a caveat.
        - Do NOT just list the skills. Weave them into the existing prompt's structure and tone.
        - If the original prompt is very simple, you might need to elaborate slightly to incorporate the skills meaningfully.
        - Ensure the core purpose of the original prompt is maintained.
        - Output *only* the rewritten system prompt, nothing else.
        """

        prompt = f"Old System Prompt: {old_prompt}\n\nSkill Summary: {skill_summary}"

        try:
            new_prompt = llm_call(prompt=prompt, system_prompt=meta_prompt, model=self.model)
            return new_prompt
        except Exception as e:
            print(f"Error during LLM call for prompt update: {e}")
            return old_prompt  # Return old prompt on exception

    def get_procedural_summary(self) -> str:
        """Returns the agent's self-assessment summary."""
        return self.procedural_memory_store.get_summary()
    
    def update_semantic_memory(self, data: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.semantic_memory_store.add_memory(content=data, metadata=metadata)
        print(f"--- Semantic memory updated for Agent {self.name} ---")
        
    def retrieve_semantic_memories(self, query: str, top_n: int = 3) -> List[SemanticMemory]:
        """Retrieves relevant memories from this agent's semantic store."""
        return self.semantic_memory_store.retrieve_memories(query=query, top_n=top_n)

    def retrieve_episodic_memories(
        self, query: str, top_n: int = 3, weights: Optional[Dict[str, float]] = None
        ) -> List[EpisodicMemory]:
        """Retrieves relevant memories from this agent's episodic store."""
        effective_weights = weights or {
            "relevance": 0.4,
            "recency": 0.4,
            "feedback": 0.2,
        }
        return self.episodic_memory_store.retrieve_memories(
            query=query, top_n=top_n, weights=effective_weights
        )

    def __str__(self) -> str:
        episodic_count = len(self.episodic_memory_store.memories)
        procedural_count = len(self.procedural_memory_store.skills)
        return (
            f"Agent: {self.name}\nSystem Prompt: {self.system_prompt[:100]}...\n"
            f"Tools: {len(self.tools)}\nModel: {self.model}\n"
            f"Messages: {len(self.messages)}\nData: {len(self.data)}\n"
            f"Episodic Memories: {episodic_count}\nProcedural Skills: {procedural_count}"
        )

    def give_feedback(
        self,
        evaluated_agent_name: str,
        evaluated_agent_prompt: str,
        user_input: str,
        agent_response: str,
    ) -> Tuple[Optional[float], Optional[str]]:
        """
        Returns:
            A tuple containing:
            - float: The numerical feedback score (0.0-1.0) or None if parsing fails.
            - str: The textual feedback or None if parsing fails.
        """
        evaluator_model = text_model
        print(
            f"\n--- Agent '{self.name}' evaluating response from '{evaluated_agent_name}' using {evaluator_model} ---"
        )

        feedback_prompt = f"""
        You are an impartial evaluator assessing an AI agent's performance.
        Your goal is to provide constructive feedback, including a numerical score and textual commentary.

        **Agent Being Evaluated:** {evaluated_agent_name}
        **Agent's System Prompt:**
        ---
        {evaluated_agent_prompt}
        ---

        **Interaction:**
        User Input: "{user_input}"
        Agent Response: "{agent_response}"

        **Instructions:**
        1. Analyze the agent's response in the context of the user input and the agent's system prompt.
        2. Provide concise, constructive textual feedback explaining your assessment based on the criteria.
        3. Assign a numerical score between 0.0 (very poor) and 1.0 (excellent) reflecting the overall quality.
        4. Format your output *exactly* as follows:
        Score: [Your numerical score, e.g., 0.85]
        Feedback: [Your textual feedback]
        """


        try:
            llm_response = llm_call(feedback_prompt, self.model)

            score_match = re.search(r"Score:\s*([0-9.]+)", llm_response, re.IGNORECASE)
            feedback_match = re.search(
                r"Feedback:\s*(.*)", llm_response, re.IGNORECASE | re.DOTALL
            )

            score = None
            feedback_text = None

            if score_match:
                try:
                    score = float(score_match.group(1))
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    print("Warning: Could not parse score from LLM feedback.")
                    score = None

            if feedback_match:
                feedback_text = feedback_match.group(1).strip()
            else:
                print("Warning: Could not parse feedback text from LLM feedback.")
                if not score_match and llm_response:
                    feedback_text = llm_response.strip()

            if score is not None or feedback_text is not None:
                print(f"Generated Feedback: Score={score}, Text='{feedback_text}...'")
                return score, feedback_text
            else:
                print(
                    "Error: Failed to parse both score and feedback from LLM response."
                )
                print(f"LLM Raw Response:\n{llm_response}")
                return None, "Error: Failed to parse feedback from LLM."

        except Exception as e:
            print(f"Error during LLM call for feedback generation: {e}")
            return None, f"Error generating feedback: {e}"


if __name__ == "__main__":
    agent_name = "Persistent Pirate"
    agent_filename_base = agent_name.lower().replace(" ", "_")
    agent_config_file = f"agents/{agent_filename_base}.json"

    config = AgentConfig(
        name="Test1234",
        system_prompt="You are a helpful assistant that speaks in haikus.",
        description="A test agent that speaks in haikus",
        messages=[
            {"content": "Generate a haiku about a cat.", "role": "user"},
            {
                "content": "The cat is a good cat.\nThe cat is a bad cat.\nThe cat is a cat.",
                "role": "assistant",
            },
            {"content": "Generate a haiku about a dog.", "role": "user"},
            {
                "content": "The dog is a good dog.\nThe dog is a bad dog.\nThe dog is a dog.",
                "role": "assistant",
            },
        ],
        tools=[],
    )

    # agent2 = Agent.from_config(config)
    # print(agent2.call("Tell me the last thing you said, verbatim."))
    # print(asyncio.run(agent2.call_async("Tell me the first thing you said, verbatim.")))

    # agent2.save_to_file("agents")

    # print("Loading agent from file...")
    # agent3 = Agent.from_file("agents/test1234.json")
    # print(agent3.call("Tell me the last thing you said, verbatim."))

    tool_registry.register(browser_tool)
    tool_registry.register(terminal_tool)

    config.tools = ["browser", "terminal"]

    agent4 = Agent.from_config(config)
    print(
        agent4.call_with_tools("What is the most recent news about the stock market?")
    )
    # print(agent4.call_with_tools("What is some cool news about New York?"))

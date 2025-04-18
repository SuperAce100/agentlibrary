import os
import numpy as np
from typing import Callable, Any, Optional, List, Dict
from pydantic import BaseModel, Field, validator, TypeAdapter
from openai import OpenAI
from dotenv import load_dotenv
import datetime # Need datetime for timestamp comparison
import math # Need math for exponential decay
import json # Added json import
# --- Add LLM call import ---
# Assuming models.py is one level up
import sys
from pathlib import Path
# Add the parent directory (src) to sys.path to allow importing from models
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))
try:
    from models.models import llm_call_messages, text_model # Import LLM call function and default model
except ImportError as e:
    print(f"Error importing from models.models: {e}")
    # Provide dummy functions if import fails, to allow file to load
    def llm_call_messages(messages, model): return "Error: LLM call function not loaded."
    text_model = "dummy-model"

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()

class EpisodicMemory(BaseModel):
    """Represents a single piece of episodic memory."""
    content: str
    embedding: List[float]
    # Metadata can include timestamp, feedback_score, feedback_text, source, etc.
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict) # Use Dict for better typing

def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Calculates cosine similarity between two vectors."""
    vec1 = np.array(v1)
    vec2 = np.array(v2)
    # Handle potential zero vectors to avoid division by zero
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0 # Or handle as appropriate (e.g., raise error, return specific value)
    # Normalize similarity to be between 0 and 1 (assuming embeddings are not perfectly anti-aligned)
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    return (similarity + 1) / 2 # Scale from [-1, 1] to [0, 1]

class EpisodicMemoryStore:
    """Manages storage, embedding, and retrieval of episodic memories."""
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.memories: List[EpisodicMemory] = []
        self.client = OpenAI() # Assumes OPENAI_API_KEY is set in environment
        self.embedding_model = embedding_model

    def _get_embedding(self, text: str) -> List[float]:
        """Generates embedding for the given text using OpenAI API."""
        text = text.replace("\n", " ") # OpenAI recommendation
        try:
            response = self.client.embeddings.create(input=[text], model=self.embedding_model)
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            # Return a zero vector or handle error appropriately
            # Getting the dimension requires an extra API call or knowing it beforehand
            # For 'text-embedding-3-small', the dimension is 1536
            # For 'text-embedding-ada-002', the dimension is 1536
            # Adjust if using a different model
            embedding_dim = 1536 # Example dimension
            return [0.0] * embedding_dim

    def add_memory(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        feedback_score: Optional[float] = None, # Added feedback score
        feedback_text: Optional[str] = None     # Added feedback text
    ):
        """
        Adds a new memory to the store after embedding its content.
        Includes optional feedback score and text in metadata.
        Assumes feedback_score is normalized (e.g., 0.0 to 1.0).
        """
        if not content:
            print("Warning: Attempted to add empty memory content.")
            return

        meta = metadata or {}

        # --- Ensure Timestamp ---
        if 'timestamp' not in meta:
             meta['timestamp'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        elif isinstance(meta.get('timestamp'), datetime.datetime):
             ts = meta['timestamp']
             if ts.tzinfo is None:
                 ts = ts.replace(tzinfo=datetime.timezone.utc)
             meta['timestamp'] = ts.isoformat()

        # --- Add Feedback to Metadata ---
        if feedback_score is not None:
            # Optional: Add validation/normalization for the score here if needed
            if not (0.0 <= feedback_score <= 1.0):
                 print(f"Warning: Feedback score {feedback_score} is outside the expected [0, 1] range. Storing as is.")
            meta['feedback_score'] = feedback_score
        if feedback_text:
            meta['feedback_text'] = feedback_text

        # --- Get Embedding ---
        embedding = self._get_embedding(content)
        if not embedding or not isinstance(embedding, list) or len(embedding) == 0:
             print(f"Warning: Failed to generate valid embedding for content: '{content[:50]}...'")
             return

        # --- Create and Store Memory ---
        memory = EpisodicMemory(content=content, embedding=embedding, metadata=meta)
        self.memories.append(memory)
        feedback_info = f" | Score: {feedback_score}" if feedback_score is not None else ""
        print(f"Added memory: '{content[:50]}...' (Timestamp: {meta['timestamp']}{feedback_info})")

    def retrieve_memories(
        self,
        query: str,
        top_n: int = 3,
        # Use a dictionary for weights for clarity and flexibility
        weights: Dict[str, float] = {'relevance': 0.4, 'recency': 0.4, 'feedback': 0.2},
        recency_decay_rate: float = 0.01, # Controls how fast recency score drops (per hour)
        default_feedback_score: float = 0.5 # Score assigned if feedback is missing
    ) -> List[EpisodicMemory]:
        """
        Retrieves the top_n most relevant memories based on a weighted combination of
        semantic similarity (relevance), recency, and feedback score.
        """
        if not query or not self.memories:
            return []

        # --- Validate Weights ---
        total_weight = sum(weights.values())
        if not math.isclose(total_weight, 1.0):
            # Normalize weights if they don't sum to 1, or raise an error
            print(f"Warning: Weights {weights} do not sum to 1. Normalizing.")
            if total_weight == 0: total_weight = 1 # Avoid division by zero
            weights = {k: v / total_weight for k, v in weights.items()}
        if any(w < 0 for w in weights.values()):
             raise ValueError("Weights cannot be negative.")

        # Extract individual weights, defaulting to 0 if missing
        relevance_weight = weights.get('relevance', 0.0)
        recency_weight = weights.get('recency', 0.0)
        feedback_weight = weights.get('feedback', 0.0)

        # --- Get Query Embedding ---
        query_embedding = self._get_embedding(query)
        if not query_embedding or len(query_embedding) == 0:
             print("Warning: Failed to generate query embedding.")
             return []

        # --- Calculate Scores for Each Memory ---
        now = datetime.datetime.now(datetime.timezone.utc)
        scored_memories = []

        for mem in self.memories:
            # --- 1. Relevance Score ---
            relevance_score = 0.0
            if mem.embedding and len(mem.embedding) == len(query_embedding):
                relevance_score = cosine_similarity(query_embedding, mem.embedding) # Already scaled 0-1
            else:
                print(f"Warning: Skipping memory due to invalid embedding: {mem.content[:30]}...")
                continue

            # --- 2. Recency Score ---
            recency_score = 0.0
            timestamp_str = mem.metadata.get('timestamp')
            if timestamp_str:
                try:
                    mem_time = datetime.datetime.fromisoformat(timestamp_str)
                    if mem_time.tzinfo is None:
                         mem_time = mem_time.replace(tzinfo=datetime.timezone.utc)
                    time_diff_seconds = max(0, (now - mem_time).total_seconds())
                    hours_elapsed = time_diff_seconds / 3600
                    recency_score = math.exp(-recency_decay_rate * hours_elapsed)
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not parse timestamp '{timestamp_str}' for memory: {mem.content[:30]}... Error: {e}")
                    recency_score = 0.1 # Assign low score
            else:
                 recency_score = 0.1 # Assign low score if no timestamp

            # --- 3. Feedback Score ---
            # Assumes score in metadata is already normalized [0, 1]
            feedback_score = mem.metadata.get('feedback_score', default_feedback_score)
            # Ensure score is clamped [0, 1] even if stored incorrectly or default is outside range
            feedback_score = max(0.0, min(1.0, float(feedback_score)))


            # --- Combine Scores ---
            final_score = (
                (relevance_weight * relevance_score) +
                (recency_weight * recency_score) +
                (feedback_weight * feedback_score)
            )
            scored_memories.append((final_score, mem))

        # --- Sort and Return ---
        if not scored_memories:
            print("Warning: Could not calculate scores for any stored memories.")
            return []

        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored_memories[:top_n]]

    def get_all_memories_content(self) -> List[str]:
        """Returns the content of all stored memories."""
        return [mem.content for mem in self.memories]

    # --- New: Methods for Persistence ---
    def export_memories(self) -> List[Dict[str, Any]]:
        """Exports memories to a list of dictionaries suitable for JSON serialization."""
        # Use Pydantic's TypeAdapter for efficient serialization of the list
        memory_list_adapter = TypeAdapter(List[EpisodicMemory])
        return memory_list_adapter.dump_python(self.memories, mode='json')

    def import_memories(self, data: List[Dict[str, Any]]):
        """Imports memories from a list of dictionaries (e.g., loaded from JSON)."""
        if not isinstance(data, list):
            print("Warning: Invalid data format for importing episodic memories. Expected list.")
            self.memories = []
            return

        try:
            # Use Pydantic's TypeAdapter for validation and parsing
            memory_list_adapter = TypeAdapter(List[EpisodicMemory])
            self.memories = memory_list_adapter.validate_python(data)
            print(f"Imported {len(self.memories)} episodic memories.")
        except Exception as e:
            print(f"Error importing episodic memories: {e}. Initializing empty memory.")
            self.memories = []


def get_competency_label(score: float) -> str:
    """Maps a numerical score (0-1) to a descriptive label."""
    if score >= 0.85: return "very good"
    if score >= 0.65: return "good"
    if score >= 0.45: return "average"
    if score >= 0.25: return "poor"
    return "very poor"

class ProceduralMemory(BaseModel):
    """Represents a single piece of procedural memory (a skill or flaw)."""
    skill_description: str = Field(..., description="Description of the skill or area of competence (e.g., 'summarizing technical documents')")
    competency_score: float = Field(..., description="Numerical score representing competence (0.0 = very poor, 1.0 = very good)")
    competency_label: str = Field(..., description="Human-readable label derived from the score (e.g., 'good', 'poor')")
    last_updated: datetime.datetime = Field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    evidence: List[str] = Field(default_factory=list, description="List of reasons or events supporting this competency level (e.g., feedback text, task IDs)")

    @validator('competency_score')
    def score_must_be_in_range(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError('competency_score must be between 0.0 and 1.0')
        return v

    # Automatically update label when score changes (if needed, though usually set during creation/update)
    # @validator('competency_label', always=True)
    # def label_matches_score(cls, v, values):
    #     if 'competency_score' in values:
    #         expected_label = get_competency_label(values['competency_score'])
    #         if v != expected_label:
    #             # Optionally log a warning or just return the expected label
    #             return expected_label
    #     return v


class ProceduralMemoryStore:
    """Manages storage and retrieval of PROCEDURAL memories (skills/flaws)."""
    def __init__(self):
        # Store skills keyed by their description for easy lookup/update
        self.skills: Dict[str, ProceduralMemory] = {}

    def add_or_update_skill(
        self,
        skill_description: str,
        new_score: Optional[float] = None, # The new score from feedback/evaluation
        evidence_text: Optional[str] = None, # Supporting text for this update
        initial_prompt_source: bool = False # Flag if this comes from the initial system prompt
    ):
        """
        Adds a new skill or updates an existing one based on new evidence.
        Score update logic can be customized (e.g., averaging, replacement).
        """
        if not skill_description:
            print("Warning: Skill description cannot be empty.")
            return

        # Normalize skill description for consistent key usage (optional)
        key = skill_description.lower().strip()
        now = datetime.datetime.now(datetime.timezone.utc)
        new_evidence_entry = f"[{now.isoformat()}] {evidence_text or ('Source: Initial Prompt' if initial_prompt_source else 'Source: Unknown')}"

        if key in self.skills:
            # --- Update Existing Skill ---
            existing_skill = self.skills[key]

            # Update score (simple replacement for now, could use moving average etc.)
            if new_score is not None:
                 # Clamp score just in case
                 clamped_score = max(0.0, min(1.0, new_score))
                 # Optional: Check if score actually changed significantly
                 # if not math.isclose(existing_skill.competency_score, clamped_score):
                 existing_skill.competency_score = clamped_score
                 existing_skill.competency_label = get_competency_label(clamped_score) # Update label
                 print(f"Updated skill '{key}': Score -> {clamped_score:.2f} ({existing_skill.competency_label})")

            # Append evidence and update timestamp
            existing_skill.evidence.append(new_evidence_entry)
            existing_skill.last_updated = now
            self.skills[key] = existing_skill # Update the dictionary entry

        else:
            # --- Add New Skill ---
            if new_score is None:
                print(f"Warning: Cannot add new skill '{key}' without an initial score.")
                return

            clamped_score = max(0.0, min(1.0, new_score))
            label = get_competency_label(clamped_score)
            new_skill = ProceduralMemory(
                skill_description=skill_description, # Store original casing
                competency_score=clamped_score,
                competency_label=label,
                last_updated=now,
                evidence=[new_evidence_entry]
            )
            self.skills[key] = new_skill
            print(f"Added new skill '{key}': Score = {clamped_score:.2f} ({label})")


    def get_skill(self, skill_description: str) -> Optional[ProceduralMemory]:
        """Retrieves a specific skill by its description."""
        key = skill_description.lower().strip()
        return self.skills.get(key)

    def get_all_skills(self) -> List[ProceduralMemory]:
        """Returns a list of all stored procedural memories."""
        return list(self.skills.values())

    def get_summary(self, include_average: bool = False) -> str:
        """Generates a textual summary of the agent's skills and flaws."""
        if not self.skills:
            return "I have no specific skills or flaws recorded yet."

        summary_parts = []
        for skill in self.skills.values():
            # Use the pre-calculated label
            summary_parts.append(f"I am {skill.competency_label} at {skill.skill_description}.")

        # Optionally filter or sort before joining, e.g., by score
        # sorted_skills = sorted(self.skills.values(), key=lambda s: s.competency_score, reverse=True)
        # for skill in sorted_skills: ...

        if include_average:
             average_score = np.mean([s.competency_score for s in self.skills.values()])
             average_label = get_competency_label(average_score)
             summary_parts.append(f"Overall, my assessed competency level is {average_label} ({average_score:.2f}).")


        return " ".join(summary_parts)

    # --- New: Methods for Persistence ---
    def export_skills(self) -> Dict[str, Dict[str, Any]]:
        """Exports skills to a dictionary suitable for JSON serialization."""
        # Use Pydantic's TypeAdapter for the dictionary values
        skill_dict_adapter = TypeAdapter(Dict[str, ProceduralMemory])
        return skill_dict_adapter.dump_python(self.skills, mode='json')


    def import_skills(self, data: Dict[str, Dict[str, Any]]):
        """Imports skills from a dictionary (e.g., loaded from JSON)."""
        if not isinstance(data, dict):
            print("Warning: Invalid data format for importing procedural skills. Expected dict.")
            self.skills = {}
            return

        try:
            # Use Pydantic's TypeAdapter for validation and parsing
            skill_dict_adapter = TypeAdapter(Dict[str, ProceduralMemory])
            self.skills = skill_dict_adapter.validate_python(data)
            print(f"Imported {len(self.skills)} procedural skills.")
        except Exception as e:
            print(f"Error importing procedural skills: {e}. Initializing empty skills.")
            self.skills = {}


# --- New Function: update_prompt ---

def update_prompt(
    old_prompt: str,
    skill_library: ProceduralMemoryStore,
    model: str = text_model # Use the default model from models.models
) -> str:
    """
    Uses an LLM to rewrite an agent's system prompt based on its current skills.

    Args:
        old_prompt: The current system prompt of the agent.
        skill_library: The ProceduralMemoryStore containing the agent's skills.
        model: The LLM model to use for rewriting the prompt.

    Returns:
        The rewritten system prompt, or the old prompt if an error occurs.
    """
    skill_summary = skill_library.get_summary(include_average=True)
    if not skill_summary or "no specific skills" in skill_summary:
        print("No significant skills found to update prompt.")
        return old_prompt # No skills to base update on

    # Meta-prompt instructing the LLM how to rewrite the prompt
    meta_prompt = f"""
You are an expert prompt engineer. Your task is to rewrite an agent's system prompt.
The goal is to subtly incorporate the agent's learned skills and flaws into its core instructions,
making the prompt more accurate to the agent's current capabilities without explicitly listing skills like a resume.

Agent's Current Skill Summary:
{skill_summary}

Agent's Original System Prompt:
---
{old_prompt}
---

Rewrite the original system prompt based *only* on the skill summary provided.
- Integrate the strengths and weaknesses naturally into the agent's persona or instructions.
- For example, if the agent is 'good at creative writing' but 'poor at math', the prompt might lean more towards creative tasks or mention a preference for words over numbers. If the agent is 'excellent at following instructions', reinforce that. If 'poor at speaking like a pirate', maybe tone down that instruction slightly or add a caveat.
- Do NOT just list the skills. Weave them into the existing prompt's structure and tone.
- If the original prompt is very simple, you might need to elaborate slightly to incorporate the skills meaningfully.
- Ensure the core purpose of the original prompt is maintained.
- Output *only* the rewritten system prompt, nothing else.
"""

    messages = [
        {"role": "system", "content": "You are an expert prompt engineer."},
        {"role": "user", "content": meta_prompt}
    ]

    try:
        print(f"\nAttempting to update prompt using model: {model}")
        # Use the imported llm_call_messages function
        new_prompt = llm_call_messages(messages, model=model)
        if new_prompt and new_prompt != old_prompt and "Error:" not in new_prompt:
            print("Prompt successfully updated.")
            # Basic cleaning: remove potential markdown quotes
            new_prompt = new_prompt.strip().removeprefix("```").removesuffix("```").strip()
            return new_prompt
        elif "Error:" in new_prompt:
             print(f"LLM call failed: {new_prompt}")
             return old_prompt # Return old prompt on LLM error
        else:
            print("LLM did not provide a substantially different prompt.")
            return old_prompt # Return old prompt if LLM response is empty or same
    except Exception as e:
        print(f"Error during LLM call for prompt update: {e}")
        return old_prompt # Return old prompt on exception


if __name__ == "__main__":

    class TestArgumentSchema(BaseModel):
        x: int

    print("\n--- Memory Store Example with Recency & Feedback ---")
    episodic_memory_store = EpisodicMemoryStore()
    procedural_memory_store = ProceduralMemoryStore()
    now = datetime.datetime.now(datetime.timezone.utc)

    # Add memories with varying timestamps and feedback
    episodic_memory_store.add_memory(
        "User likes apples.",
        metadata={"timestamp": (now - datetime.timedelta(days=2)).isoformat()}, # Old
        feedback_score=0.9, # High quality memory
        feedback_text="Confirmed user preference."
    )
    episodic_memory_store.add_memory(
        "User mentioned liking fruit, especially bananas.",
        metadata={"timestamp": (now - datetime.timedelta(hours=1)).isoformat()}, # Recent
        feedback_score=0.7, # Medium quality
        feedback_text="User seemed positive about bananas."
    )
    episodic_memory_store.add_memory(
        "Agent discussed fruit preferences with the user.",
        metadata={"timestamp": (now - datetime.timedelta(minutes=5)).isoformat()}, # Very Recent
        feedback_score=0.4, # Low quality - maybe agent misunderstood
        feedback_text="Agent response was okay, but could be clearer."
    )
    episodic_memory_store.add_memory(
        "The weather is nice today.",
        metadata={"timestamp": (now - datetime.timedelta(hours=5)).isoformat()} # Less Recent
        # No feedback score provided for this one
    )

    print(f"\nTotal memories stored: {len(episodic_memory_store.memories)}")

    query = "What fruit does the user like?"

    print("\nRetrieval (Balanced Weights):")
    weights_balanced = {'relevance': 0.4, 'recency': 0.4, 'feedback': 0.2}
    memories_balanced = episodic_memory_store.retrieve_memories(query, top_n=3, weights=weights_balanced)
    for score, mem in memories_balanced: # Note: retrieve_memories returns List[EpisodicMemory], not scores directly
        print(f"- Content: {mem.content} | Score: {mem.metadata.get('feedback_score', 'N/A')} | Timestamp: {mem.metadata.get('timestamp')}")
        if 'feedback_text' in mem.metadata: print(f"  Feedback: {mem.metadata['feedback_text']}")


    print("\nRetrieval (Prioritizing Feedback):")
    weights_feedback = {'relevance': 0.2, 'recency': 0.2, 'feedback': 0.6}
    memories_feedback = episodic_memory_store.retrieve_memories(query, top_n=3, weights=weights_feedback)
    for mem in memories_feedback:
        print(f"- Content: {mem.content} | Score: {mem.metadata.get('feedback_score', 'N/A')} | Timestamp: {mem.metadata.get('timestamp')}")
        if 'feedback_text' in mem.metadata: print(f"  Feedback: {mem.metadata['feedback_text']}")

    print("\nRetrieval (Prioritizing Relevance):")
    weights_relevance = {'relevance': 0.7, 'recency': 0.2, 'feedback': 0.1}
    memories_relevance = episodic_memory_store.retrieve_memories(query, top_n=3, weights=weights_relevance)
    for mem in memories_relevance:
        print(f"- Content: {mem.content} | Score: {mem.metadata.get('feedback_score', 'N/A')} | Timestamp: {mem.metadata.get('timestamp')}")
        if 'feedback_text' in mem.metadata: print(f"  Feedback: {mem.metadata['feedback_text']}")

    print("\nAll stored memories:")
    all_content = episodic_memory_store.get_all_memories_content()
    for content in all_content:
        print(f"- {content}")

    print("\n--- Procedural Memory Store Example ---")
    procedural_store = ProceduralMemoryStore()

    # Adding skills/flaws based on initial prompt or early feedback
    procedural_store.add_or_update_skill(
        skill_description="Speaking like a pirate",
        new_score=0.9,
        initial_prompt_source=True
    )
    procedural_store.add_or_update_skill(
        skill_description="Mathematical calculations",
        new_score=0.3,
        evidence_text="Struggled with basic addition in task_001."
    )
    procedural_store.add_or_update_skill(
        skill_description="Generating creative ideas",
        new_score=0.75,
        evidence_text="Received positive feedback on brainstorming session."
    )

    # Simulate feedback leading to improvement
    print("\nSimulating feedback...")
    procedural_store.add_or_update_skill(
        skill_description="Mathematical calculations",
        new_score=0.55, # Improved score
        evidence_text="Feedback on task_005: Correctly calculated percentages."
    )

    # Get a specific skill
    pirate_skill = procedural_store.get_skill("Speaking like a pirate")
    if pirate_skill:
        print(f"\nDetails for 'Speaking like a pirate':")
        print(f"  Score: {pirate_skill.competency_score}")
        print(f"  Label: {pirate_skill.competency_label}")
        print(f"  Last Updated: {pirate_skill.last_updated}")
        print(f"  Evidence: {pirate_skill.evidence}")

    # Get the summary
    print("\nAgent Self-Assessment Summary:")
    print(procedural_store.get_summary())

    print("\nAgent Self-Assessment Summary (with average):")
    print(procedural_store.get_summary(include_average=True))

    # --- New: Prompt Update Example ---
    print("\n--- Prompt Update Example ---")
    original_agent_prompt = "You are a helpful assistant. You try your best to answer questions accurately."
    print(f"Original Prompt:\n{original_agent_prompt}")
    print("\nSkill Summary for Update:")
    print(procedural_store.get_summary(include_average=True))

    # Ensure OPENAI_API_KEY or equivalent is set in .env for the LLM call
    updated_prompt = update_prompt(original_agent_prompt, procedural_store)

    print(f"\nUpdated Prompt:\n{updated_prompt}")

    # Example with a different initial prompt
    original_pirate_prompt = "Ahoy! Ye be talkin' to Pirate Pete, yer swashbucklin' assistant! Ask me anythin', matey!"
    print(f"\nOriginal Pirate Prompt:\n{original_pirate_prompt}")
    updated_pirate_prompt = update_prompt(original_pirate_prompt, procedural_store)
    print(f"\nUpdated Pirate Prompt:\n{updated_pirate_prompt}")

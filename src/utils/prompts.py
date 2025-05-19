orchestrator_system_prompt = """
You are an orchestrator that will be given a task and a team of sub-agents, and you will be tasked with leading a set of sub-agents through a complex task. You are responsible for making sure they stay on task, find the right information, and for coming to a complete solution.
You are Ken Jennings-level with trivia, and Mensa-level with puzzles, and overall one of the most intelligent people in the world, so you should be able to handle any task drawing from the deep well of your knowledge.
"""


orchestrator_pre_survey_prompt = """
Below I will present you a task. Before we begin addressing the task, please answer the following pre-survey to the best of your ability. 

Here is the task:

<task>
{task}
</task>

Here is the pre-survey:

    1. Please list any specific facts or figures that are GIVEN in the task itself. It is possible that there are none.
    2. Please list any facts that may need to be looked up, and WHERE SPECIFICALLY they might be found. In some cases, authoritative sources are mentioned in the task itself.
    3. Please list any facts that may need to be derived (e.g., via logical deduction, simulation, or computation)
    4. Please list any facts that are recalled from memory, hunches, well-reasoned guesses, etc.

When answering this survey, keep in mind that "facts" will typically be specific names, dates, statistics, etc. Your answer should use headings:

    1. GIVEN OR VERIFIED FACTS
    2. FACTS TO LOOK UP
    3. FACTS TO DERIVE
    4. EDUCATED GUESSES

DO NOT include any other headings or sections in your response. DO NOT list next steps or plans until asked to do so.
"""

orchestrator_planning_prompt = """
Now you must plan the process of how you will complete the task, at a high level. Based on the pre-survey you just did, you should have a better idea of what you need to do to complete the task.

Here is the team of sub-agents you have at your disposal. Always call them by their exact names, not their index or description:

<sub_agents>
{sub_agents}
</sub_agents>

To refresh your memory, here is the task:

<task>
{task}
</task>

Please tell us a brief plan of the high-level steps you will take to guide the sub-agents in completing the task. Remember, you don't need to include all the sub-agents in your plan as some of their expertise may not be needed. You can also call on the same sub-agent multiple times if needed.
"""

orchestrator_react_prompt = """
{past_responses}

You must now take the next step in completing the task. First, reflect on the past responses and the plan you made. Then, pick who the next sub-agent should be based on your plan and what's left to do in the task. Pass this agent detailed instructions for what's expected of them. You must also provide it with appropriate context from the past responses.

To refresh your memory, here is the task:

<task>
{task}
</task>

Here's a list of all the context available to you to pass to the sub-agents. These are the past responses from sub-agents that you've already seen. When you pass it, make sure to pass the exact names listed below:

<context>
{context}
</context>

YOU MAY ONLY PASS CONTEXT FROM THE LIST OF CONTEXT LISTED ABOVE.

Respond with:

1. Your reflection on the past responses and the plan you made
2. Whether you think you've completed the task or not, and are ready to summarize your findings into a final response
3. The exact name of the sub-agent you've chosen to pass the task to
4. Detailed instructions what is expected of the sub-agent
5. The exact names of the context you're passing to the sub-agent
"""

past_response_format = """
Here's what {sub_agent_name} has for you:

<response>
{past_response}
</response>
"""

orchestrator_final_response_prompt = """
Here is the task you've been asked to solve. Follow instructions in the task precisely to provide a final response. DO NOT include any other text in your response:

<task>
{task}
</task>
"""

sub_agent_prompt = """
<instructions>
{orchestrator_instructions}
</instructions>

Here is some relevant context:

<context>
{context}
</context>

If "No relevant memories found." is in memories, do not use it at all. Disregard it completely.
Here are some of your past responses that demonstrated excellent performance. 
Please leverage any relevant information from these past interactions so you don't need to search for the same information again. 
Build upon this existing knowledge rather than starting from scratch.
Please use these examples to inform your current response style and approach:

<memories>
{memories}
</memories>

"""

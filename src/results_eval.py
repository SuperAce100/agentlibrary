from models.llms import llm_call

def evaluate_final_response(agent_response, correct_answer):
    """
    Evaluate if the agent's response matches the correct answer.
    Returns a score of 1 if correct, 0 if incorrect.
    """
    feedback_prompt = f"""
        You are an impartial evaluator assessing whether an AI's answer matches the provided answer.

        **Pair:**
        Correct Answer: "{correct_answer}"
        Agent Response: "{agent_response}"

        **Instructions:**
        1. Extract the agent's answer from the agent's response, which can be more verbose than necessary at times.
        2. If the agent's answer is not exactly the same as the correct answer, return a score of 0. 
        3. If the agent's answer is exactly the same as the correct answer, return a score of 1.
        4. Examples of matching answers that may not be exactly the same (some math equations are written in LaTeX):
        Correct Answer: "1+3x+5x^2+6x^3+5x^4+3x^5+x^6"
        Agent Response: "P(x) = 1+3t+5t^2+6t^3+5t^4+3t^5+t^6"
        Score: 1

        Correct Answer: "Z+Z+Z+Z+Z"
        Agent Response: "\(\widetilde(\Omega)^(\mathrm(Spin))_(12)(BG2)\cong\mathbb(Z)^5\)"
        Score: 1
        
        Return only the score (0 or 1) with no additional text.
        """

    llm_response = llm_call(feedback_prompt)
    return llm_response



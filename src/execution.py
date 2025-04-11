
from tqdm import tqdm
from models.agents import Agent
from utils.diffs import apply_diff


sub_agent_prompt = """
Your name is {name}. Below is a template on which you will need to work. A supervisor has assigned you a section of the template to complete.

You need to complete the first section of the template assigned to you. Use the following format of diff to replace the placeholder prompt given to you with your specialized response.

<<<<<<< SEARCH 
This should be the placeholder prompt for the section you need to complete.
 =======
Put your response here.
>>>>>>> REPLACE 

Make sure to use the correct format for the diff.

ONLY WORK ON YOUR FIRST SECTION OF THE TEMPLATE. DO NOT WORK ON ANY OTHER SECTIONS. IF THERE ARE MULTIPLE SECTIONS ASSIGNED TO YOU, ONLY WORK ON THE FIRST ONE.
"""

def execute_sub_agent(agent: Agent, template: str) -> str:
    """
    Let a sub-agent work on a template
    """
    agent.pass_context(sub_agent_prompt.format(name=agent.name))
    diff = agent.call(template)
    return diff

def execute_sub_agents(agents: list[Agent], agent_order: list[str], template: str) -> str:
    """
    Execute a list of sub-agents in a given order on a template
    """
    for agent_name in tqdm(agent_order, desc="Executing sub-agents"):
        agent = next(a for a in agents if a.name == agent_name)
        diff = execute_sub_agent(agent, template)
        template = apply_diff(diff, template)
    
    return template



if __name__ == "__main__":
    template = """
    ## Market Research: China–US West Coast Air Travel
    ### Market Analyst
    *Conduct a detailed market analysis of the current and projected demand for airline routes between major Chinese cities and major West Coast US cities (e.g., Los Angeles, San Francisco, Seattle, Vancouver). Include key competitors, major route frequencies, known market gaps, airline alliance impacts, and recent trends in traffic/market recovery (especially post-pandemic). Present quantitative demand estimates, identify underserved routes or cities, and summarize relevant government restrictions or bilateral capacity limits. Provide references for data sources used.*

    ## Financial Viability Assessment
    ### Business Analyst
    *Using the market research provided above, prepare high-level pro forma financial projections for a new entrant airline focusing on China–West Coast routes. Include start-up costs (fleet, crew, regulatory, etc.), assumptions on load factors, ticket prices, and operational costs. Estimate break-even timeline and discuss financial risks and sensitivities (fuel cost, currency, bilateral/quota fluctuations). Present as a concise table plus narrative analysis.*

    ## Regulatory and Legal Evaluation
    ### Customs Lawyer
    *Analyze the legal and regulatory framework for launching and operating a new air carrier on China–West Coast US routes. Address bilateral air service agreements, ownership/investment restrictions, air traffic rights, licensing, visa/customs hurdles, and major compliance challenges (e.g., Open Skies, anti-trust concerns, national security review). Highlight any recent areas of tension or shifts in the regulatory environment that could impact the new airline.*

    ## Branding & Differentiation Potential
    ### Branding Expert
    *Develop a branding and positioning strategy for the proposed airline, considering market and financial context. Identify opportunities to differentiate (e.g., cultural inclusivity, premium experience, digital innovation, sustainability, loyalty partnerships). Suggest brand attributes (name, value proposition, target audience) and opportunities for building trust and recognition in both the US and China. Include a sketch of launch marketing tactics or partnership opportunities.*

    ======================END SCRATCH SECTION=====================

    ## Comprehensive Business Proposal for a China–West Coast US Airline

    ### Executive Summary
    ### Business Analyst
    *Using the research and analysis compiled in the sections above, compose an executive summary of the business proposal, including mission, opportunity, concept, and key takeaways on feasibility, risks, and brand potential (approx. 250 words).*

    ### Market Opportunity & Competitive Landscape
    ### Market Analyst
    *Draft a clear, polished section summarizing the demand for China–West Coast airline service, strengths and weaknesses of current competitors, key market gaps, and the strategic rationale for entry. Incorporate data points and insights from your earlier analysis in an accessible format.*

    ### Financial Forecasts & Viability Assessment
    ### Business Analyst
    *Present the business’s projected financial outlook, key assumptions, and an overview of the business model’s strengths/weaknesses. Summarize start-up costs, projected revenues, break-even analysis, and major financial risks. Use visuals/concise tables.*

    ### Legal and Regulatory Challenges
    ### Customs Lawyer
    *Summarize the key legal hurdles, regulatory compliance issues, and bilateral agreement constraints. Offer recommendations for mitigating these risks and highlight any timelines or dependencies that may impact launch.*

    ### Brand Building & Go-to-Market Strategy
    ### Branding Expert
    *Present a compelling brand concept and go-to-market plan, including positioning, proposed brand attributes, and initial marketing steps. Emphasize the airline’s differentiation strategy and resonance with both Chinese and US markets.*

    ### Conclusion & Next Steps
    ### Business Analyst
    *Provide a closing section summarizing overall business potential, priority action items for go/no-go decision, and high-level strategic milestones for the next 12–24 months.*
    """

    sub_agents = [
        Agent(name="Business Analyst", system_prompt="You are a business analyst sub-agent that analyzes a given topic. You will be given a topic and you will need to analyze it and return the results.", tools=[]),
        Agent(name="Customs Lawyer", system_prompt="You are a customs lawyer sub-agent that analyzes a given topic. You will be given a topic and you will need to analyze it and return the results.", tools=[]),
        Agent(name="Market Analyst", system_prompt="You are a market analyst sub-agent that analyzes a given topic. You will be given a topic and you will need to analyze it and return the results.", tools=[]),
        Agent(name="Branding Expert", system_prompt="You are a branding expert sub-agent that analyzes a given topic. You will be given a topic and you will need to analyze it and return the results.", tools=[]),
    ]

    agent_order = ['Market Analyst', 'Business Analyst', 'Customs Lawyer', 'Branding Expert', 'Business Analyst', 'Market Analyst', 'Business Analyst', 'Customs Lawyer', 'Branding Expert', 'Business Analyst']

    new_template = execute_sub_agents(sub_agents, agent_order, template)
    print(new_template)

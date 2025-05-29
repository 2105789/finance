from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from core.llm import get_llm
from core.tools import get_tools

# Get the ReAct prompt
prompt = hub.pull("hwchase17/react")

def get_financial_agent():
    """Initializes and returns the financial Langchain agent."""
    llm = get_llm()
    tools = get_tools()

    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True, # Add this to handle potential output parsing issues
        max_iterations=10, # Increased from 5 to 10
        return_intermediate_steps=False # Explicitly set, though it's default
    )
    return agent_executor

def get_financial_agent_response(user_query: str):
    """Gets a response from the financial agent for a given user query."""
    agent_executor = get_financial_agent()
    try:
        response = agent_executor.invoke({"input": user_query})
        # Check if the agent stopped due to iteration limit or other non-exception stop
        if "output" not in response or response["output"] is None:
            # This can happen if max_iterations is hit and no output is generated
            # or if the agent somehow finishes without a clear output.
            agent_steps = response.get("intermediate_steps", [])
            last_thoughts = ""
            if agent_steps:
                # Attempt to get the last thought or observation
                last_step_action, last_step_observation = agent_steps[-1]
                last_thoughts = f"Last thought process: Action: {last_step_action.tool} with input {last_step_action.tool_input}, Observation: {last_step_observation}"
            
            return f"The agent could not definitively answer your query. It may have reached its processing limit. {last_thoughts} Please try rephrasing or simplifying your question."
        return response["output"]
    except Exception as e:
        print(f"Agent execution error: {e}")
        return f"An error occurred while processing your request: {str(e)}. Please try rephrasing your question."

if __name__ == '__main__':
    # Example usage (for testing)
    # Ensure your API keys are set in core/config.py
    # and core is in PYTHONPATH or this script is run from the project root.
    # test_query = "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"
    test_query = "What is the current stock price of MSFT and what are some recent news about Microsoft?"
    print(f"Testing query: {test_query}")
    response = get_financial_agent_response(test_query)
    print(f"Agent Response: {response}")

    test_query_2 = "What's our risk exposure in Asia tech stocks today, and highlight any earnings surprises?"
    print(f"\nTesting query (complex): {test_query_2}")
    response_2 = get_financial_agent_response(test_query_2)
    print(f"Agent Response: {response_2}") 
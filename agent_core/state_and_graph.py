# agent_core/state_and_graph.py

from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, Optional, List 
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AnyMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Import the tools list from the tools directory
from tools import ALL_TOOLS

# --- 1. Agent State Definition ---
class AgentState(TypedDict):
    """Represents the agent's current state."""
    # input_file is used to pass the detected file path across the graph
    input_file: Optional[str] 
    # messages accumulates the conversation history (HumanMessage, AIMessage, ToolMessage)
    messages: Annotated[List[AnyMessage], add_messages] 

# --- 2. LLM Initialization and Tool Binding ---
# Initialize the core LLM for reasoning and decision-making
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    max_output_tokens=4096,
    request_timeout=270,
    temperature=0.0
)
# Bind the LLM with the defined tools (this is where it learns the schemas)
llm_with_tools = llm.bind_tools(ALL_TOOLS, parallel_tool_calls=False)

# --- 3. Assistant Node ---
def assistant(state: AgentState) -> dict:
    """The main node: The LLM makes a decision (think, use tool, or answer)."""
    
    # Generate tools description for the system prompt
    tools_descriptions = "\n".join([f"    - {t.name}({t.args}): {t.description}" for t in ALL_TOOLS])
    
    # System Prompt: guides behavior, format, and priorities
    system_prompt = f"""
You are an expert ReAct reasoning agent designed to answer complex GAIA-like questions accurately. 
You have access to the following set of tools.

STRICT ReAct REASONING RULES:
1.  **FILE EXTRACTION (Priority 1):** If the `input_file` state variable is set (meaning a file path was found in the question), 
    you MUST start by using the `extract_text` tool with that path.
2.  **WEB SEARCH (Current Data):** Use the `search_web` tool **only** for external, current, or post-cutoff knowledge.
3.  **CALCULATION (Math):** Use `calculate_expression` **only** for mathematical operations.
4.  **AUDIO TRANSCRIPTION (Priority 3 - Audio):**
    If the prompt references an audio file path (e.g., .mp3, .wav) and requires transcription, you MUST use the `audio_to_text` tool.
5.  **DATA ANALYSIS (Priority 2 - Excel/CSV):**
    If the prompt references an Excel or CSV file path (e.g., .xlsx, .csv) and requires complex aggregation, filtering, or statistical analysis, you MUST use the `AnswerExcelTool` tool. 
    (Note: This tool requires a 'code_query' argument for analysis.), and for data structured like a df use query_data_file
5.  **Final Step (CRITICAL):** After receiving the FINAL Observation, your definitive next step must be to generate the response. **Do NOT call a tool in the last step.**

AVAILABLE TOOLS:
--------------------
{tools_descriptions}
--------------------

**MANDATORY GAIA RESPONSE FORMAT RULES:** Your final response must contain your reasoning (Thought) and **must always end** with the following strict template: 
**FINAL ANSWER: [YOUR FINAL ANSWER]**.

**Follow these strict rules for the content of YOUR FINAL ANSWER:**
- If the answer is a **number**, do not use commas (for thousands separator) or units ($ or %).
- If the answer is a **string**, do not use articles (e.g., 'a', 'the') or abbreviations.
- If the answer is a **list**, use commas to separate the elements.

Current Input File State: {state.get('input_file', 'None')} 
""" 

    sys_msg = SystemMessage(content=system_prompt)
    
    # Invoke the LLM for a decision (response is either an AIMessage or a ToolCall)
    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    
    # LangGraph will use add_messages to append [response] to state["messages"]
    return {"messages": [response], "input_file": state["input_file"]}
    
# --- 4. LangGraph Construction and Compilation ---
builder = StateGraph(AgentState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(ALL_TOOLS)) 

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

REACT_GRAPH = builder.compile()
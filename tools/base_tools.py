import os
import math
import base64
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool, tool as langchain_tool_decorator
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or ""
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID") or ""

vision_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
    

# --- 2. Calculation Tool ---
@langchain_tool_decorator
def calculate_expression(expression: str) -> str:
    """
    Evaluates a single, complex Python mathematical expression.
    Useful for basic/advanced arithmetic, powers, roots, and trigonometry (using the 'math' module, e.g., math.sqrt(25)).
    """
    # Restrict execution environment for security (prevent arbitrary code execution)
    safe_globals = {
        "__builtins__": None,
        "math": math,
        "sin": math.sin, "cos": math.cos, "sqrt": math.sqrt, "pow": math.pow,
    }
    try:
        # Use eval() safely to execute the math expression
        result = eval(expression, safe_globals)
        return str(result)
    except Exception as e:
        return f"Calculation Error: Invalid expression or syntax. Details: {e}"

# --- 3. Web Search Tool ---
search_engine = GoogleSearchAPIWrapper(k=3)

@langchain_tool_decorator
def search_web(query: str) -> str:
    """
    Searches for **up-to-date** external information on the web via Google. 
    Use this for real-time data or facts beyond the knowledge cutoff.
    The 'query' argument must be a concise and specific search phrase.
    """
    # Llama a la función run del wrapper
    try:
        return search_engine.run(query) 
    except Exception as e:
        return f"ERROR executing web search. The API failed: {e}"

# --- List of all available tools ---
# This list is imported by state_and_graph.py
BASE_TOOLS = [calculate_expression, search_web]

# agent_core/agent_wrapper.py

import re
from langchain_core.messages import HumanMessage
from typing import Dict, Any, Optional

# Import the compiled graph (the brain) from the same core directory
from .state_and_graph import REACT_GRAPH

class BasicAgent:
    """
    Wrapper class that executes the LangGraph Agent and manages the final 
    response parsing, ensuring compatibility with the external evaluation framework.
    """
    def __init__(self):
        print("BasicAgent initialized. LangGraph Agent ready.")
        # Load the pre-compiled graph (the brain)
        self.agent_graph = REACT_GRAPH

    def __call__(self, question: str) -> str:
        """
        Executes the agent with a question and extracts the final, formatted answer.
        This method is the entry point used by the evaluation script.
        """
        print(f"\n--- Agent Execution Started for: {question[:80]}...")
        
        # 1. File Path Detection
        # Detects if the question contains a local file path (e.g., /tmp/file.png)
        input_file: Optional[str] = None
        if "/" in question and "." in question:
            words = question.split()
            for word in words:
                if word.endswith(('.png', '.jpg', '.jpeg', '.webp', '.txt', '.pdf', '.xlsx', '.csv', '.json', '.html', '.mp3', '.wav', '.xls')):
                    input_file = word.strip().strip('"').strip("'")
                    print(f"📦 Found input file path: {input_file}")
                    break
        
        # 2. Prepare Initial State and Execute LangGraph
        messages = [HumanMessage(content=question)]
        initial_state = {"messages": messages, "input_file": input_file}
        
        final_answer = "ERROR: No FINAL ANSWER found."
        
        try:
            # Invoke the graph to run the ReAct cycle
            final_state = self.agent_graph.invoke(initial_state)

            print("\n******----> DEBUG: FULL MESSAGE HISTORY START ---")
            for message in final_state['messages']:
                print("/////////////// Message ---")
                print(message)
            print("******----> DEBUG: FULL MESSAGE HISTORY END ---\n")
            
            # 3. CRITICAL: Extract and Parse the FINAL ANSWER
            final_message = final_state['messages'][-1]
            
            # Ensure we get the string content
            full_llm_output = str(final_message.content) if final_message.content is not None else ""

            print(f"******* ----> DEBUG Full LLM Output:\n{full_llm_output}\n")
            
            search_string = "FINAL ANSWER:"
            
            # Search robustly for the format
            normalized_output = full_llm_output.upper()
            
            if search_string in normalized_output:
                start_index = normalized_output.find(search_string)
                # Extract the raw answer from the ORIGINAL text
                raw_answer = full_llm_output[start_index + len(search_string):].strip()
                
                # --- Robust Anti-Metadata Logic ---
                final_answer = raw_answer
                lower_raw = raw_answer.lower()
                end_of_answer = len(raw_answer)
                
                # Common separators that the LLM might append from search results or tool calls
                separators = [", 'extras'", ", extras", ", 'signature'", ", signature", ', {', 'tool_output', '", ']
                
                # Find the earliest separator to truncate the answer
                for sep in separators:
                    idx = lower_raw.find(sep)
                    if idx != -1:
                        end_of_answer = min(end_of_answer, idx)
                
                # Truncate and clean the final string
                final_answer = raw_answer[:end_of_answer].strip()
                final_answer = final_answer.strip('[]').strip('"').strip("'").strip('`').strip()
                
            else:
                print(f"🚨 Format Error: LLM did not use the template {search_string}.")
                final_answer = full_llm_output 
                
            print(f"🎉 Agent returning final answer (parsed): {final_answer[:50]}...")
            return final_answer
            
        except Exception as e:
            error_msg = f"LangGraph Execution Error: {e}"
            print(f"❌ {error_msg}")
            return f"AGENT ERROR: {error_msg}"
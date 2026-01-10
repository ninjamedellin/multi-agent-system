# app.py

import os
import sys
from dotenv import load_dotenv
# Import the BasicAgent class (which invokes LangGraph)


# LOAD ENVIRONMENT VARIABLES (Must be the first executable code)
try:
    # Loads the .env file from the .secrets/ folder
    if not load_dotenv(dotenv_path='.secrets/.env'):
        print("⚠️ Warning: Could not load .secrets/.env. Keys must be system variables.")
except Exception as e:
    print(f"❌ Error attempting to load .secrets/.env: {e}")

from agent_core.agent_wrapper import BasicAgent

# --- 1. Agent Initialization ---
try:
    # Initialize your agent
    agent = BasicAgent()
    print("✅ BasicAgent initialized. Ready to test.")
except Exception as e:
    print(f"❌ ERROR: Failed to initialize BasicAgent. Check your API keys and configuration. Details: {e}")
    sys.exit(1) # Exit if initialization fails


# --- 2. Execution Function ---
def run_single_test(prompt: str):
    """Executes the agent with a single prompt and displays the result."""
    
    # Handle case with no argument
    if not prompt or prompt.strip() == "":
        print("\n⚠️ Please provide a question to test.")
        print("Usage:")
        print("  python app.py \"What is the square root of 625?\"")
        print("  python app.py \"Extract text from path /tmp/my_image.png\"")
        return

    # Execute the agent
    print(f"\n--- Starting Agent Test ---")
    print(f"PROMPT: {prompt}")
    
    try:
        # Invokes the __call__ method of BasicAgent
        final_answer = agent(prompt)
        
        # print("\n--- FINAL PARSED RESULT ---")
        # print(f"RESPONSE: {final_answer}")
        # print("--------------------------------")

        print("\n--- FINAL RESULT (RAW/PARSED) ---")
        if not final_answer.strip():
            print("RESPONSE: [Empty String - The LLM returned no useful content]")
        else:
            print(f"RESPONSE: {final_answer}")
            print("--------------------------------------")    
        
    except Exception as e:
        error_msg = f"❌ FATAL ERROR during agent execution: {e}"
        print(error_msg)


# --- 3. Main Execution Block ---
if __name__ == "__main__":
    # The question is taken from command line arguments (sys.argv[1:])
    
    if len(sys.argv) > 1:
        # Joins all arguments after the script name into a single string
        test_prompt = " ".join(sys.argv[1:])
        run_single_test(test_prompt)
    else:
        # Case with no arguments
        run_single_test("")
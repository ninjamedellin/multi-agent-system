
# Multi-Agent System with LangGraph & Gemini 
 
A robust Multi-Agent System architecture designed to orchestrate complex workflows using Google Gemini models and LangGraph. This project implements an intelligent agent capable of routing queries to specialized tools for web research, mathematical calculation, data analysis, and multimedia processing. 
 
## Project Overview 
 
This system moves beyond simple question-answering by using a graph-based orchestration (State Management). The agent decomposes user requests, selects the appropriate tools, and iteratively solves problems that require multiple steps or external data sources. 
 
Key capabilities: 
* Reasoning Engine: Powered by Google Gemini Pro. 
* Orchestration: LangGraph for stateful multi-step execution. 


 
## Features & Tools 
 
The agent is equipped with a modular toolset to handle diverse tasks: 
 
* Web Search: Retrieves real-time information using Google Search API. 
* Mathematical Calculation: Safely evaluates complex mathematical expressions using Python's math library. 
* YouTube Intelligence: Extracts transcripts and summarizes video content. 
* Audio Processing: Converts audio files to text (ASR) using OpenAI Whisper. 
* Data Analysis: Processes Excel files and structured data to extract insights. 
* Multimodal Support: Handles text and image inputs (Vision API). 
 
## Prerequisites 
 
* Python 3.10+ 
* A Google Cloud Project with the following APIs enabled: 
 * Gemini API (Generative Language API) 
 * Custom Search JSON API 
 
## Installation 
 
1. Clone the repository: 
```bash 
git clone https://github.com/ninjamedellin/multi-agent-system.git 
cd multi-agent-system 
``` 
 
2. Create and activate a virtual environment: 
 
 Windows: 
```bash 
python -m venv venv 
source venv/Scripts/activate 
``` 
 
 Mac/Linux: 
```bash 
python3 -m venv venv 
source venv/bin/activate 
``` 
 
3. Install dependencies: 
```bash 
pip install -r requirements.txt 
``` 
 
4. Environment Configuration: 
 Create a folder named .secrets in the root directory and create a file named .env inside it (.secrets/.env). Add your API keys: 



```ini 
GOOGLE_API_KEY="your_gemini_api_key" 
GOOGLE_CSE_ID="your_search_engine_id" 
GOOGLE_API_KEY_SEARCH="your_google_search_api_key" 
``` 
 
## Usage 
 
### Command Line Interface (CLI) 
You can test the agent directly from the terminal with a single prompt. Note that if you want to test with files, you should place them in your directory and reference the path. 
 
```bash 
python app.py "What is the year they mentioned in the file test_assets/audio.mp3" 
``` 
 

## Project Structure 
 
* agent_core/: Contains the core logic for the LangGraph agent, state management, and LLM wrappers. 
* tools/: Definitions for all specialized tools (Math, Search, Excel, YouTube, etc.). 
* app.py: Main entry point for CLI and Gradio interface. 
* test_assets/: Directory to store local files (images, audio, spreadsheets) for testing the agent. You need to create this folder in case you want to test with files. 
* requirements.txt: List of Python dependencies. 
* .secrets/: Folder for storing environment variables (ignored by Git). 
 
--- 
Developed by Andrés M.
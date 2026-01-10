import os
import sys
import math
import base64
from langchain_core.messages import HumanMessage
from langchain_core.tools import Tool, tool as langchain_tool_decorator
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import re
from tempfile import NamedTemporaryFile
import whisper
import pandas as pd
import io
from langchain_core.tools import BaseTool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
# from langchain.agents.agent_types import AgentType
from typing import Optional, Type
from pydantic import BaseModel, Field


os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY") or ""
os.environ["GOOGLE_CSE_ID"] = os.getenv("GOOGLE_CSE_ID") or ""

vision_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


@langchain_tool_decorator
def extract_text(img_path: str) -> str:
    """
    Extracts all readable text from an image file using a multimodal model (Gemini Vision).
    This tool should be used first for any question referencing an image or file path.
    """
    if not img_path:
        return "Error: No image path was provided."
            
    try:
        # Read the image file and encode it to base64 for the API
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        
        message = [
            HumanMessage(
                content = [
                    {
                        "type": "text",
                        "text": ("Extract all the text from this image. "
                                 "Return only the extracted text, without any additional commentary."),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64, {image_base64}"
                        },
                    },
                ]
            )
        ]
        
        response = vision_llm.invoke(message)
        return response.content.strip()
            
    except Exception as e:
        error_msg = f"Error extracting text from {img_path}: {str(e)}"
        return f"Error: {error_msg}"
    

@langchain_tool_decorator
def youtube_transcript(video_url: str) -> str:
    """
    Retrieves, downloads, and concatenates the transcript for a YouTube video using its URL. 
    Use the transcription result to answer questions about the video's content.
    The argument 'video_url' must be the complete URL of the video.
    """
    def extract_video_id(url):
        # Helper function to extract the 11-character video ID
        match = re.search(r'(?:v=|\/embed\/|youtu\.be\/|\/v\/|\/watch\?v=)([a-zA-Z0-9_-]{11})', url)
        return match.group(1) if match else None

    video_id = extract_video_id(video_url)

    if not video_id:
        return "ERROR: Invalid YouTube URL or video ID not found."
    
    try:
        # 1. Instanciar YouTubeTranscriptApi().
        # fetched_transcript es ahora un iterable de FetchedTranscriptSnippet objects.
        fetched_transcript = YouTubeTranscriptApi().fetch(
            video_id, 
            languages=['en', 'es']
        )
        
        # --- CORRECCIÓN AQUÍ: Usar item.text en lugar de item['text'] ---
        # Acceder al atributo 'text' de cada objeto.
        full_transcript = " ".join([item.text for item in fetched_transcript])
        
        # Limitar la salida
        transcript_summary = full_transcript[:15000] 
        
        return f"VIDEO TRANSCRIPT: {transcript_summary} [End of transcript]"

    except TranscriptsDisabled:
        return f"ERROR: Transcription is disabled for this video ({video_id})."
    except Exception as e:
        return f"Unexpected ERROR retrieving the transcript: {e}"    
    
@langchain_tool_decorator
def audio_to_text(audio_path: str) -> str:
    """
    Transcribes the spoken content from an audio file (e.g., MP3, WAV) 
    using the local Whisper model for high-quality, fast transcription.
    This tool should be used first for any question referencing an audio 
    file path or a specific audio file.
    """
    # 1. Verificación de librerías
    if not whisper:
        return "ERROR: Whisper library is not installed. Please install 'openai-whisper' dependencies."

    if not audio_path:
        return "ERROR: No audio file path was provided."
        
    full_path = Path(audio_path).resolve()
    
    # 2. Verificación de existencia de archivo
    if not full_path.exists():
        return f"ERROR: Audio file not found at path: {audio_path}. The file must be accessible."
        
    temp_path = None
    
    try:
        # 3. Leer los bytes del archivo
        with open(full_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        # 4. Escribir los bytes a un archivo temporal
        with NamedTemporaryFile(suffix=full_path.suffix, delete=False) as temp_f:
            temp_f.write(audio_bytes)
            temp_path = temp_f.name
        
        # 5. Transcribir con Whisper (usa el modelo "base")
        model = whisper.load_model("base") 
        output = model.transcribe(temp_path)["text"].strip()
        
        return f"AUDIO TRANSCRIPT: {output}"
            
    except Exception as e:
        # Este bloque capturará errores como la falta de FFmpeg, aunque ya lo resolviste.
        return f"ERROR during Whisper transcription: {str(e)}"
        
    finally:
        # 6. Limpieza del archivo temporal
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@langchain_tool_decorator
def query_data_file(data_path: str, code_query: str) -> str:
    """
    Reads an Excel (.xlsx, .xls) or CSV file into a pandas DataFrame (named 'df'), 
    executes a specific Python 'code_query' against it, and returns the result.
    
    This is a GENERAL PURPOSE tool for complex analysis (e.g., aggregation, 
    filtering, advanced statistics).
    
    The code_query MUST use the variable 'df' (the DataFrame) and the last 
    line MUST be a 'print()' statement for the final result.
    
    Example code_query for correlation: 'print(df[["Col1", "Col2"]].corr().iloc[0, 1])'
    """
    if not data_path:
        return "ERROR: No file path provided."

    full_path = Path(data_path).resolve()

    if not full_path.exists():
        return f"ERROR: Data file not found at path: {data_path}."

    file_extension = full_path.suffix.lower()
    df = None

    try:
        # 1. Load the DataFrame (DF)
        if file_extension in ['.csv']:
            # Read CSV (with common encoding handling)
            df = pd.read_csv(full_path, encoding='utf-8')
            print(f"Loaded CSV with shape: {df.shape}")
        elif file_extension in ['.xlsx', '.xls']:
            # Read Excel files
            df = pd.read_excel(full_path)
            print(f"Loaded Excel with shape: {df.shape}")
        else:
            return f"ERROR: Unsupported file format: {file_extension}. Only CSV, XLSX, and XLS are supported."
        
        # 2. Replace spaces in column names for easier Agent code generation
        # This converts "Operating Status" to "Operating_Status"
        df.columns = df.columns.str.replace(' ', '_', regex=False)

        # 3. Execute the query code (code_query)
        # We capture the 'print' output into a text buffer
        stdout_capture = io.StringIO()
        sys.stdout = stdout_capture
        
        # The Agent must provide valid Python code that uses 'df'
        # We use exec() to execute the code string
        exec(code_query, {'df': df, 'pd': pd, 'os': os}) 
        
        # Restore standard output and get the result
        sys.stdout = sys.__stdout__
        result = stdout_capture.getvalue().strip()
        
        return f"QUERY_RESULT: {result}"

    except Exception as e:
        # 4. Error Handling and Reporting
        # Restore standard output in case of error
        if sys.stdout != sys.__stdout__:
            sys.stdout = sys.__stdout__
            
        # Return a column summary in case of code error so the Agent can correct itself
        column_info = f"Available columns: {list(df.columns)}" if 'df' in locals() else "DataFrame not loaded."
        return f"ERROR executing code: {str(e)}. Please check syntax and column names. {column_info}"


class AnswerExcelToolArgs(BaseModel):
    query: str = Field(description="The full question or analytical query to ask the Excel file (e.g., 'What is the sum of sales for product X?').")
    file_path: str = Field(description="The file path to the Excel (.xlsx, .xls) or CSV file.")

# Defining the AnswerExcelTool class which extends BaseTool
class AnswerExcelTool(BaseTool):
    # This tool will use the internal Pandas Agent (AgentExecutor)
    name : str = "answer_excel_tool"
    description: str = (
        "Given the path to an Excel/CSV file and a query, this tool tries to get an answer "
        "by querying the file. This tool is suitable for complex data analysis, filtering, and aggregation. "
        "Provide the WHOLE question as the 'query' input."
    )
    args_schema: Type[BaseModel] = AnswerExcelToolArgs

    def _run(self, query: str, file_path: str) -> str:
        # Load the file, supporting both Excel and CSV
        try:
            if file_path.lower().endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            elif file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8')
            else:
                return f"ERROR: Unsupported file format for Pandas Agent: {file_path}"
        except Exception as e:
            return f"ERROR loading data file {file_path}: {e}"
        
        # Configure the internal LLM for the Pandas Agent
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

        # Create the specialized Pandas DataFrame agent
        agent_executor = create_pandas_dataframe_agent(
            llm,
            df,
            # ZERO_SHOT_REACT_DESCRIPTION or tool-calling works well here
            agent_type="zero-shot-react-description",
            verbose=False, # Set to True for debugging the internal agent
            allow_dangerous_code=True # IMPORTANT: Required to execute code for analysis
        )

        # Execute the query against the Pandas Agent
        try:
            result = agent_executor.run(query)
            return f"DATA_ANALYSIS_RESULT: {result}"
        except Exception as e:
            return f"PANDAS_AGENT_ERROR: The internal agent failed to execute the query. Error: {e}"



# SPECIALIZED_TOOLS = [extract_text, youtube_transcript, audio_to_text, AnswerExcelTool()]
SPECIALIZED_TOOLS = [extract_text, youtube_transcript, audio_to_text, query_data_file, AnswerExcelTool()]

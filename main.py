# main.py
import autogen
from finrobot.agents.workflow import SingleAssistantRAG
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import traceback
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize the embedding function once
try:
    # Check if ChromaDB is installed
    logger.info("Checking if ChromaDB is installed...")
    import chromadb
    logger.info("ChromaDB is installed.")
    
    # This ensures we only initialize the embedding model once for all requests
    logger.info("Initializing the sentence transformer embedding model...")
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("Embedding model initialized successfully.")
except ImportError as e:
    logger.error(f"Import error: {e}")
    raise ImportError(f"Required dependency not found: {e}. Please install using 'pip install chromadb sentence-transformers'")
except Exception as e:
    logger.error(f"Error initializing embedding model: {e}")
    embedding_function = None  # Will be checked later

# --- Configuration Loading ---
# Ensure the OAI_CONFIG_LIST path is correct relative to where you run the server
# If main.py is in the root FinRobot directory, "../OAI_CONFIG_LIST" becomes "OAI_CONFIG_LIST"
# If main.py is inside tutorials_beginner, then "../OAI_CONFIG_LIST" is correct.
# Adjust as needed. Let's assume it's run from the root directory for now.
CONFIG_FILE_PATH = "OAI_CONFIG_LIST"
REPORT_DIR = "report" # Assuming report dir is in the root

if not os.path.exists(CONFIG_FILE_PATH):
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_FILE_PATH}. Please ensure it exists.")
if not os.path.exists(REPORT_DIR):
     os.makedirs(REPORT_DIR) # Create report dir if not exists, but ensure PDFs are inside
     logger.warning(f"Report directory '{REPORT_DIR}' created, but ensure necessary PDF files are present inside.")


try:
    # Load all configurations first
    all_configs = autogen.config_list_from_json(CONFIG_FILE_PATH)
    if not all_configs:
         raise ValueError(f"No configurations found in {CONFIG_FILE_PATH}")

    # Manually filter for the desired model
    filtered_configs = [config for config in all_configs if config.get("model") == "gpt-3.5-turbo"]
    logger.info(f"Loaded and filtered configs") # Log the filtered list {filtered_configs}

    if not filtered_configs:
        raise ValueError(f"No valid configurations found for the model 'gpt-3.5-turbo' in {CONFIG_FILE_PATH}")

    llm_config = {
        "config_list": filtered_configs,
        "timeout": 120,
        "temperature": 0,
    }
    # Basic check if config list loaded correctly
    # logger.info(f"Loaded llm_config['config_list']: {llm_config.get('config_list')}") # Log the loaded list - Already logged filtered list
    # if not llm_config["config_list"]: # Check is done on filtered_configs now
    #     raise ValueError(f"No valid configurations found for the specified model in {CONFIG_FILE_PATH}")
except Exception as e:
    logger.error(f"Error loading LLM configuration: {e}")
    raise

# --- Retrieve Configurations based on Notebook ---
RETRIEVE_CONFIGS = {
    "annual_report": {
        "task": "qa",
        "vector_db": None,
        "docs_path": [os.path.join(REPORT_DIR, "Microsoft_Annual_Report_2023.pdf")],
        "chunk_token_size": 1000,
        "get_or_create": True,
        "collection_name": "msft_analysis_api", # Use different names than notebook to avoid conflicts if run in same env
        "must_break_at_empty_line": False,
    },
    "10k": {
        "task": "qa",
        "vector_db": None,
        "docs_path": [os.path.join(REPORT_DIR, "2023-07-27_10-K_msft-20230630.htm.pdf")],
        "chunk_token_size": 2000,
        "collection_name": "msft_10k_api", # Use different names
        "get_or_create": True,
        "must_break_at_empty_line": False,
        "embedding_model": "all-MiniLM-L6-v2",  # Specify a default embedding model
        "embedding_function": embedding_function,  # Use the pre-initialized function
    }
}

RAG_DESCRIPTIONS = {
    "annual_report": "Retrieve content from MSFT's 2023 Annual Report PDF for question answering.",
    "10k": "Retrieve content from MSFT's 2023 10-K report for detailed question answering."
}

# --- API Request Model ---
class ChatRequest(BaseModel):
    query: str
    report_type: str # Should be 'annual_report' or '10k'

# --- FastAPI Endpoint ---
@app.post("/chat_rag")
async def chat_rag_endpoint(request: ChatRequest):
    """
    Endpoint to chat with a RAG agent based on a specified report type.
    """
    logger.info(f"Received request for report_type: {request.report_type} with query: '{request.query}'")

    if request.report_type not in RETRIEVE_CONFIGS:
        logger.error(f"Invalid report_type: {request.report_type}")
        raise HTTPException(status_code=400, detail="Invalid report_type. Choose 'annual_report' or '10k'.")

    retrieve_config = RETRIEVE_CONFIGS[request.report_type]
    rag_description = RAG_DESCRIPTIONS[request.report_type]

    try:
        
        # Initialize the RAG agent for each request to ensure clean state
        # Note: This might be slow due to vector DB initialization if not already created.
        # For production, consider managing agent instances differently.
        logger.info("Initializing SingleAssistantRAG...")
        assistant = SingleAssistantRAG(
            "Data_Analyst", # Pass name as the first positional argument (string)
            llm_config,         # Pass llm_config as the second positional argument
            human_input_mode="NEVER", # Crucial for API
            retrieve_config=retrieve_config,
            rag_description=rag_description,
        )
        logger.info("SingleAssistantRAG initialized.")

        # Initiate the chat
        logger.info(f"Initiating chat with query: '{request.query}'")
        try:
            # Use initiate_chat which returns ChatResult
            chat_result = assistant.chat(request.query)
            logger.info("Chat finished.")

            return chat_result
                
        except Exception as e:
            logger.error(f"Error during chat execution: {e}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, 
                detail=f"Error during chat execution. Check server logs for details: {str(e)}")

    except FileNotFoundError as e:
         logger.error(f"File not found during RAG setup: {e}")
         raise HTTPException(status_code=500, detail=f"Required report file not found: {e}")
    except NameError as e:
        # This will catch the "name 'ef' is not defined" error
        error_msg = str(e)
        logger.error(f"Name error during RAG processing: {error_msg}")
        logger.error(traceback.format_exc())
        if "ef" in error_msg:
            raise HTTPException(
                status_code=500, 
                detail="Embedding function error. Make sure sentence-transformers is installed and the embedding_model is valid."
            )
        raise HTTPException(status_code=500, detail=f"Variable name error: {error_msg}")
    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise HTTPException(status_code=500, detail=f"Missing dependency: {e}")
    except ValueError as e:
        logger.error(f"Value error: {e}")
        if "chromadb" in str(e).lower():
            raise HTTPException(status_code=500, detail="ChromaDB initialization error. Check your ChromaDB installation.")
        raise HTTPException(status_code=500, detail=f"Invalid value: {e}")
    except Exception as e:
        logger.error(f"An error occurred during chat processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# --- Root Endpoint for Health Check ---
@app.get("/")
async def root():
    return {"message": "FinRobot RAG API is running"}

# --- Optional: Add uvicorn runner for direct execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server with uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8000) 
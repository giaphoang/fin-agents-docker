# main_up.py
import autogen
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import os
import logging
import traceback
from contextlib import asynccontextmanager
from functools import lru_cache
from finrobot.functional.ragquery import rag_database_earnings_call, rag_database_sec
from autogen import ConversableAgent, register_function

# --- Global Variables for RAG functions (populated at startup) ---
rag_query_functions = {}
llm_config_global = {}
system_message_template = ""
sec_form_names_global = []
earnings_call_quarters_global = []

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI Lifespan for Startup Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application starting up...")
    global llm_config_global, system_message_template
    global rag_query_functions, sec_form_names_global, earnings_call_quarters_global

    # --- Load LLM Configuration ---
    CONFIG_FILE_PATH = "OAI_CONFIG_LIST" # Assuming run from root
    if not os.path.exists(CONFIG_FILE_PATH):
        logger.error(f"Configuration file not found at {CONFIG_FILE_PATH}. Exiting.")
        raise FileNotFoundError(f"Configuration file not found at {CONFIG_FILE_PATH}")

    try:
        # Load all and filter manually - Robust approach
        all_configs = autogen.config_list_from_json(CONFIG_FILE_PATH)
        # Using gpt-3.5-turbo as per the notebook cell [10]
        # Note: Cell [12] uses gpt-4-0125-preview, adjust if needed
        filtered_configs = [config for config in all_configs if config.get("model") == "gpt-3.5-turbo"]

        if not filtered_configs:
            raise ValueError(f"No valid configurations found for 'gpt-3.5-turbo' in {CONFIG_FILE_PATH}")

        llm_config_global = {
            "config_list": filtered_configs,
            "timeout": 120,
            "temperature": 0.5, # Matching notebook cell [46]
             # "temperature": 0 # As per notebook cell [10], using 0.5 from later cells
        }
        logger.info("LLM configuration loaded successfully.")
        logger.debug(f"LLM Config: {llm_config_global}")

    except Exception as e:
        logger.error(f"Fatal error loading LLM configuration: {e}", exc_info=True)
        raise

    # --- Initialize RAG Databases and Query Functions ---
    # Using hardcoded values from notebook cell [6]
    ticker = 'NVDA'
    year = '2023'
    # FROM_MARKDOWN = False (as per notebook cell [29])

    logger.info(f"Initializing RAG for Ticker: {ticker}, Year: {year}")
    try:
        logger.info("Setting up Earnings Call RAG...")
        # This function fetches data, embeds, creates DB, returns query func
        query_earnings, earnings_quarters, _ = rag_database_earnings_call(ticker=ticker, year=year)
        rag_query_functions['earnings'] = query_earnings
        earnings_call_quarters_global = earnings_quarters
        logger.info(f"Earnings Call RAG setup complete. Available Quarters: {earnings_quarters}")

        logger.info("Setting up SEC Filings RAG...")
         # This function fetches data, embeds, creates DB, returns query func
        query_sec, sec_forms = rag_database_sec(ticker=ticker, year=year, FROM_MARKDOWN=False)
        rag_query_functions['sec'] = query_sec
        sec_form_names_global = sec_forms
        logger.info(f"SEC Filings RAG setup complete. Available Forms: {sec_forms}")
        logger.info(f"SEC query function name: {query_sec.__name__}")
        
        # Now check if the function name matches what we expect
        expected_sec_name = "query_database_sec"
        if query_sec.__name__ != expected_sec_name:
            logger.warning(f"SEC query function name '{query_sec.__name__}' does not match expected name '{expected_sec_name}'")
            logger.warning("This might cause issues. Setting explicit name during registration.")
        
        logger.info(f"Earnings query function name: {query_earnings.__name__}")
        expected_earnings_name = "query_database_earnings_call"
        if query_earnings.__name__ != expected_earnings_name:
            logger.warning(f"Earnings query function name '{query_earnings.__name__}' does not match expected name '{expected_earnings_name}'")
            logger.warning("This might cause issues. Setting explicit name during registration.")

        # --- Prepare System Message Template ---
        sec_form_system_msg = ""
        for sec_form in sec_form_names_global:
            if sec_form == "10-K":
                sec_form_system_msg += "10-K for yearly data, "
            elif "10-Q" in sec_form:
                # Extract quarter number if present, otherwise just list 10-Q
                parts = sec_form.split('-')
                quarter = parts[-1] if len(parts) > 1 and parts[-1].isdigit() else None
                if quarter:
                    sec_form_system_msg += f"{sec_form} for Q{quarter} data, "
                else:
                    sec_form_system_msg += f"{sec_form} for quarterly data, " # Fallback if format isn't specific
        sec_form_system_msg = sec_form_system_msg.rstrip(', ')

        earnings_call_system_message = ", ".join(earnings_call_quarters_global)

        system_message_template = f"""You are a helpful financial assistant and your task is to select the sec_filings or earnings_call to best answer the question.
You can use query_database_sec(question, sec_form_name) by passing question and relevant sec_form names like {{{sec_form_system_msg}}}.
You can use query_database_earnings_call(question, quarter) by passing question and relevant quarter names with possible values {{{earnings_call_system_message}}}.
When you are ready to end the conversation, reply TERMINATE."""
        logger.info("System message template prepared.")
        logger.debug(f"System Message: {system_message_template}")


    except Exception as e:
        logger.error(f"Fatal error during RAG setup: {e}", exc_info=True)
        # Depending on severity, you might want the app to fail startup
        raise RuntimeError(f"RAG Setup Failed: {e}")

    logger.info("FastAPI startup initialization complete.")
    yield
    # --- Cleanup (if needed, e.g., close DB connections) ---
    logger.info("FastAPI application shutting down...")

app = FastAPI(lifespan=lifespan)

# --- API Request Model ---
class ChatRequestUp(BaseModel):
    query: str
    data_source_preference: str | None = None # Optional hint, but agent decides

# --- Agent Initialization Cache (Simple Example) ---
# A more robust cache might be needed for high load
@lru_cache(maxsize=2) # Cache based on system message (effectively, once per run)
def get_planner_agent(system_message: str):
     logger.info("Creating PlannerAdmin agent instance...")
     return ConversableAgent(
        name="PlannerAdmin",
        system_message=system_message,
        llm_config=llm_config_global,
        human_input_mode="NEVER",
        code_execution_config=False, # As per notebook
        is_termination_msg=lambda msg: "TERMINATE" in msg.get("content", ""),
    )

@lru_cache(maxsize=1) # Only one type of tool proxy needed
def get_tool_proxy():
    logger.info("Creating Tool_Proxy agent instance...")
    return ConversableAgent(
        name="Tool_Proxy",
        # System message might not be strictly needed if llm_config=False
        # system_message="Analyze the response from user proxy and decide whether the suggested database is suitable. Answer in simple yes or no", # From notebook
        llm_config=False, # Crucial: Makes it act as executor
        human_input_mode="NEVER", # Changed from notebook's ALWAYS
        # Default auto reply might be used if function execution fails, but shouldn't chat back
        default_auto_reply="Function execution failed or no output.",
    )


# --- FastAPI Endpoint ---
@app.post("/chat_rag_up")
async def chat_rag_up_endpoint(request: ChatRequestUp):
    """
    Endpoint to chat with agents using registered RAG tools (SEC/Earnings).
    """
    logger.info(f"Received request with query: '{request.query}'")

    if not rag_query_functions:
        logger.error("RAG query functions not initialized. Server startup likely failed.")
        raise HTTPException(status_code=500, detail="RAG system not initialized.")

    try:
        # Get potentially cached agent instances
        user_proxy = get_planner_agent(system_message_template)
        tool_proxy = get_tool_proxy()

        # Reset agents before use (important for cached agents)
        user_proxy.reset()
        tool_proxy.reset()

        # Clear previous functions if any (agent reset might not do this)
        # This is a workaround for potential state issues with register_function
        try:
            user_proxy.update_function_signature("query_database_sec", is_remove=True)
            user_proxy.update_function_signature("query_database_unstructured_sec", is_remove=True)
            user_proxy.update_function_signature("query_database_earnings_call", is_remove=True)
        except Exception as e:
            # If the functions don't exist yet, that's fine
            logger.debug(f"Function signature removal error (can be ignored): {e}")

        # Register the RAG functions for this specific chat
        register_function(
            rag_query_functions['sec'],
            caller=user_proxy,
            executor=tool_proxy,
            name="query_database_sec",  # Explicit name instead of using __name__
            description="Tool to query SEC filings database (10-K, 10-Q). Provide 'question' and 'sec_form_name'." # Simplified desc
        )
        logger.info(f"Registered tool: query_database_sec")

        # In case the original function in the system message was 'query_database_unstructured_sec',
        # register an alias to that name as well
        try:
            # Register the same function but with a different name as a fallback
            register_function(
                rag_query_functions['sec'],
                caller=user_proxy,
                executor=tool_proxy,
                name="query_database_unstructured_sec",
                description="Tool to query SEC filings database (10-K, 10-Q). Provide 'question' and 'sec_form_name'."
            )
            logger.info("Registered additional alias: query_database_unstructured_sec")
        except Exception as e:
            logger.warning(f"Failed to register function alias: {e}")

        register_function(
            rag_query_functions['earnings'],
            caller=user_proxy,
            executor=tool_proxy,
            name="query_database_earnings_call",  # Explicit name instead of using __name__
            description="Tool to query earnings call transcripts database. Provide 'question' and 'quarter' (e.g., Q1, Q2)." # Simplified desc
        )
        logger.info(f"Registered tool: query_database_earnings_call")


        # Initiate the chat
        logger.info(f"Initiating chat with query: '{request.query}'")
        chat_result = user_proxy.initiate_chat(
            recipient=tool_proxy,
            message=request.query,
            max_turns=10 # Match notebook example
        )
        logger.info("Chat finished.")

        # Basic response structure
        response_data = {
             "chat_history": getattr(chat_result, 'chat_history', None),
             "summary": getattr(chat_result, 'summary', None),
             "cost": getattr(chat_result, 'cost', None),
             "error": getattr(chat_result, 'error', None)
        }
         # Clean up None values for cleaner JSON
        response_data = {k: v for k, v in response_data.items() if v is not None}


        return response_data

    except Exception as e:
        logger.error(f"An error occurred during chat processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- Root Endpoint for Health Check ---
@app.get("/")
async def root():
    if not rag_query_functions:
         return {"status": "error", "message": "RAG system failed to initialize."}
    return {"status": "ok", "message": "FinRobot RAG API (UP Version) is running"}

# --- Optional: Add uvicorn runner for direct execution ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server (UP Version) with uvicorn...")
    # Use reload=True for development, remove for production
    uvicorn.run("main_up:app", host="0.0.0.0", port=8001, reload=True) # Use a different port 
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
# Register tools
from autogen.cache import Cache

from finrobot.utils import get_current_date, register_keys_from_json
from finrobot.data_source import FinnHubUtils, YFinanceUtils
from finrobot.toolkits import register_toolkits

# --- Global Variables for RAG functions (populated at startup) ---
rag_query_functions = {}
llm_config_global = {}
system_message_template = ""
sec_form_names_global = []
earnings_call_quarters_global = []

# Dictionary to cache RAG query functions by ticker and year
rag_cache = {}

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- FastAPI Lifespan for Startup Initialization ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI application starting up...")
    global llm_config_global

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

    # We no longer initialize RAG databases at startup, they will be created on demand

    logger.info("FastAPI startup initialization complete.")
    yield
    # --- Cleanup (if needed, e.g., close DB connections) ---
    logger.info("FastAPI application shutting down...")

app = FastAPI(lifespan=lifespan)

# --- Updated API Request Model ---
class ChatRequestUp(BaseModel):
    query: str
    ticker: str = "NVDA"  # Default to NVDA if not specified
    year: str = "2023"    # Default to 2023 if not specified
    filing_types: list[str] = []  # Optional list of specific filing types to use, empty means use all available
    data_source_preference: str | None = None  # Optional hint, but agent decides

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

# Function to get or create RAG functions for a specific ticker and year
async def get_or_create_rag_functions(ticker: str, year: str, specific_filing_types: list[str] = None):
    cache_key = f"{ticker}_{year}"
    
    # Check if we have this configuration cached
    if cache_key in rag_cache:
        logger.info(f"Using cached RAG functions for {ticker}, {year}")
        return rag_cache[cache_key]
    
    logger.info(f"Initializing RAG for Ticker: {ticker}, Year: {year}")
    result = {
        "sec": None,
        "earnings": None,
        "sec_forms": [],
        "earnings_quarters": []
    }
    
    try:
        # Initialize SEC filings RAG
        logger.info("Setting up SEC Filings RAG...")
        query_sec, sec_forms = rag_database_sec(ticker=ticker, year=year, FROM_MARKDOWN=False)
        result["sec"] = query_sec
        result["sec_forms"] = sec_forms
        logger.info(f"SEC Filings RAG setup complete. Available Forms: {sec_forms}")
        
        # Initialize Earnings Call RAG
        logger.info("Setting up Earnings Call RAG...")
        query_earnings, earnings_quarters, _ = rag_database_earnings_call(ticker=ticker, year=year)
        result["earnings"] = query_earnings
        result["earnings_quarters"] = earnings_quarters
        logger.info(f"Earnings Call RAG setup complete. Available Quarters: {earnings_quarters}")
        
        # Cache the result
        rag_cache[cache_key] = result
        return result
        
    except Exception as e:
        logger.error(f"Error initializing RAG for {ticker}, {year}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize RAG: {str(e)}")

# --- Updated FastAPI Endpoint ---
@app.post("/chat_rag_up")
async def chat_rag_up_endpoint(request: ChatRequestUp):
    """
    Endpoint to chat with agents using registered RAG tools (SEC/Earnings).
    Accepts custom ticker, year, and filing types.
    """
    logger.info(f"Received request with query: '{request.query}', ticker: {request.ticker}, year: {request.year}")

    try:
        # Get or create RAG functions for the requested ticker and year
        rag_data = await get_or_create_rag_functions(request.ticker, request.year, request.filing_types)
        
        # Filter filing types if specified
        sec_forms = rag_data["sec_forms"]
        if request.filing_types:
            sec_forms = [form for form in sec_forms if form in request.filing_types]
            if not sec_forms:
                logger.warning(f"None of the requested filing types {request.filing_types} are available. Using all available types.")
                sec_forms = rag_data["sec_forms"]
        
        # Prepare system message with available filing types
        sec_form_system_msg = ""
        for sec_form in sec_forms:
            if sec_form == "10-K":
                sec_form_system_msg += "10-K for yearly data, "
            elif "10-Q" in sec_form:
                # Extract quarter number if present, otherwise just list 10-Q
                parts = sec_form.split('-')
                quarter = parts[-1] if len(parts) > 1 and parts[-1].isdigit() else None
                if quarter:
                    sec_form_system_msg += f"{sec_form} for Q{quarter} data, "
                else:
                    sec_form_system_msg += f"{sec_form} for quarterly data, "
        sec_form_system_msg = sec_form_system_msg.rstrip(', ')
        
        earnings_call_system_message = ", ".join(rag_data["earnings_quarters"])
        
        system_message = f"""You are a helpful financial assistant analyzing {request.ticker} data from {request.year}. Your task is to select the sec_filings or earnings_call to best answer the question.
You can use query_database_sec(question, sec_form_name) by passing question and relevant sec_form names like {{{sec_form_system_msg}}}.
You can use query_database_earnings_call(question, quarter) by passing question and relevant quarter names with possible values {{{earnings_call_system_message}}}.
When you are ready to end the conversation, reply TERMINATE."""
        
        # Get agent instances
        user_proxy = get_planner_agent(system_message)
        tool_proxy = get_tool_proxy()
        
        # Reset agents
        user_proxy.reset()
        tool_proxy.reset()
        
        # Clear previous functions if any
        try:
            user_proxy.update_function_signature("query_database_sec", is_remove=True)
            user_proxy.update_function_signature("query_database_unstructured_sec", is_remove=True)
            user_proxy.update_function_signature("query_database_earnings_call", is_remove=True)
        except Exception as e:
            logger.debug(f"Function signature removal error (can be ignored): {e}")
        
        # Register the RAG functions for this specific chat
        register_function(
            rag_data["sec"],
            caller=user_proxy,
            executor=tool_proxy,
            name="query_database_sec",
            description=f"Tool to query {request.ticker} SEC filings database ({', '.join(sec_forms)}). Provide 'question' and 'sec_form_name'."
        )
        logger.info(f"Registered tool: query_database_sec for {request.ticker}")
        
        # Optionally register alias
        try:
            register_function(
                rag_data["sec"],
                caller=user_proxy,
                executor=tool_proxy,
                name="query_database_unstructured_sec",
                description=f"Tool to query {request.ticker} SEC filings database ({', '.join(sec_forms)}). Provide 'question' and 'sec_form_name'."
            )
            logger.info("Registered additional alias: query_database_unstructured_sec")
        except Exception as e:
            logger.warning(f"Failed to register function alias: {e}")
        
        register_function(
            rag_data["earnings"],
            caller=user_proxy,
            executor=tool_proxy,
            name="query_database_earnings_call",
            description=f"Tool to query {request.ticker} earnings call transcripts database. Provide 'question' and 'quarter' from ({', '.join(rag_data['earnings_quarters'])})."
        )
        logger.info(f"Registered tool: query_database_earnings_call for {request.ticker}")
        
        # Initiate the chat
        logger.info(f"Initiating chat with query: '{request.query}'")
        chat_result = user_proxy.initiate_chat(
            recipient=tool_proxy,
            message=request.query,
            max_turns=10
        )
        logger.info("Chat finished.")
        
        # Basic response structure
        response_data = {
            "chat_history": getattr(chat_result, 'chat_history', None),
            "summary": getattr(chat_result, 'summary', None),
            "cost": getattr(chat_result, 'cost', None),
            "error": getattr(chat_result, 'error', None),
            "ticker": request.ticker,
            "year": request.year,
            "filing_types_used": sec_forms,
            "earnings_quarters_available": rag_data["earnings_quarters"]
        }
        # Clean up None values for cleaner JSON
        response_data = {k: v for k, v in response_data.items() if v is not None}
        
        return response_data
        
    except Exception as e:
        logger.error(f"An error occurred during chat processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

# --- FinGPT Forecaster Endpoint ---

class ChatRequestFingpt(BaseModel):
    company: str

@app.post("/fingpt_forecaster")
async def fingpt_forecaster(request: ChatRequestFingpt):
    """
    Endpoint for stock price movement prediction using FinGPT Forecaster.
    
    This endpoint implements the functionality from the agent_fingpt_forecaster notebook,
    which analyzes company news and financial data to predict stock price movements.
    """
    try:
        # Validate required parameters
        if not request.company:
            raise HTTPException(status_code=400, detail="Missing required parameter: 'company'")
        
        company = request.company
        logger.info(f"Processing FinGPT Forecaster request for company: {company}")
        
        # Initialize agents
        # gpt-4-0125-preview
        # gpt-3.5-turbo limit data <= 2023
        config_list = autogen.config_list_from_json(
            "OAI_CONFIG_LIST",
            filter_dict={"model": ["gpt-4-0125-preview"]},
        )
        llm_config = {"config_list": config_list, "timeout": 120, "temperature": 0}
        
        # Register API keys if needed
        try:
            register_keys_from_json("config_api_keys")
        except Exception as e:
            logger.warning(f"Failed to register API keys: {e}")
        
        analyst = autogen.AssistantAgent(
            name="Market_Analyst",
            system_message="As a Market Analyst, one must possess strong analytical and problem-solving abilities, collect necessary financial information and aggregate them based on client's requirement."
            "For coding tasks, only use the functions you have been provided with. Reply TERMINATE when the task is done.",
            llm_config=llm_config,
        )
        
        user_proxy = autogen.UserProxyAgent(
            name="User_Proxy",
            is_termination_msg=lambda x: x.get("content", "") and x.get(
                "content", "").endswith("TERMINATE"),
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False,
            },
        )
        
        tools = [
    {
        "function": FinnHubUtils.get_company_profile,
        "name": "get_company_profile",
        "description": "get a company's profile information"
    },
    {
        "function": FinnHubUtils.get_company_news,
        "name": "get_company_news",
        "description": "retrieve market news related to designated company"
    },
    {
        "function": FinnHubUtils.get_basic_financials,
        "name": "get_financial_basics",
        "description": "get latest financial basics for a designated company"
    },
    {
        "function": YFinanceUtils.get_stock_data,
        "name": "get_stock_data",
        "description": "retrieve stock price data for designated ticker symbol"
    }
]
        
        register_toolkits(tools, analyst, user_proxy)
        
        # Run the analysis
        with Cache.disk() as cache:
            chat_result = user_proxy.initiate_chat(
                analyst,
                message=f"Use all the tools provided to retrieve information available for {company} upon {get_current_date()}. Analyze the positive developments and potential concerns of {company} "
                "with 2-4 most important factors respectively and keep them concise. Most factors should be inferred from company related news. "
                f"Then make a rough prediction (e.g. up/down by 2-3%) of the {company} stock price movement for next week. Provide a summary analysis to support your prediction.",
                cache=cache,
            )
        
        # Extract the analysis from chat history
        chat_history = getattr(chat_result, 'chat_history', [])
        analysis = None
        
        # Find the last substantive message from the analyst (before TERMINATE)
        for message in reversed(chat_history):
            if message.get('role') == 'assistant' and message.get('content') != "TERMINATE":
                analysis = message.get('content')
                break
        
        # Prepare response
        response_data = {
            "company": company,
            "analysis": analysis,
            "chat_history": chat_history,
            "timestamp": get_current_date()
        }
        
        logger.info(f"Successfully completed FinGPT Forecaster analysis for {company}")
        return response_data
        
    except Exception as e:
        logger.error(f"An error occurred during FinGPT Forecaster processing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


# --- Root Endpoint for Health Check ---
@app.get("/")
async def root():
    if not llm_config_global:
        return {"status": "error", "message": "LLM configuration failed to initialize."}
    
    return {
        "status": "ok", 
        "message": "FinRobot RAG API (UP Version) is running",
        "endpoints": [
            {
                "path": "/chat_rag_up",
                "method": "POST",
                "description": "Chat with a financial assistant using SEC filings and earnings call data",
                "parameters": {
                    "query": "Your question about the company (required)",
                    "ticker": "Stock ticker symbol (default: NVDA)",
                    "year": "Year for financial data (default: 2023)",
                    "filing_types": "List of specific filing types to use (optional and currently only supports 10-K and 10-Q)",
                    "data_source_preference": "Hint for preferred data source (optional)"
                },
                "example": {
                    "query": "What are Nvidia's strategies for GPU business?",
                    "ticker": "NVDA",
                    "year": "2023"
                }
            }
        ]
    }

# --- Optional: Add uvicorn runner for direct execution ---
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Starting FastAPI server (UP Version) with uvicorn...")
#     # Use reload=True for development, remove for production
#     uvicorn.run("main_up:app", host="0.0.0.0", reload=True, port=80) # Use a different port 
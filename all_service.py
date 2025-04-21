import os
import sys
import logging
import traceback
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
import autogen
import uvicorn
from datetime import datetime
import yfinance as yf
import shutil
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the system path
parent_directory = os.path.abspath(os.path.dirname(__file__))
if parent_directory not in sys.path:
    sys.path.insert(0, parent_directory)

# Import FinRobot modules
from finrobot.agents.workflow import SingleAssistantRAG, SingleAssistant
from finrobot.utils import get_current_date, get_date_n_days_ago, register_keys_from_json
from finrobot.data_source import FinnHubUtils, YFinanceUtils

try:
    # For RAG functionality
    from sentence_transformers import SentenceTransformer
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from finrobot.functional.ragquery import rag_database_earnings_call, rag_database_sec
    
    # Initialize the embedding function once
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    logger.info("Embedding model initialized successfully.")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.warning("Some functionality may be limited due to missing dependencies")
    embedding_function = None

# Create FastAPI app
app = FastAPI(title="FinRobot API", description="Financial analysis and data retrieval API")

# --- Configuration Loading ---
CONFIG_FILE_PATH = "OAI_CONFIG_LIST"
REPORT_DIR = "report"

if not os.path.exists(CONFIG_FILE_PATH):
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_FILE_PATH}. Please ensure it exists.")
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

# Load and register API keys
try:
    register_keys_from_json("config_api_keys")
    logger.info("API keys registered successfully")
except Exception as e:
    logger.error(f"Error loading API keys: {e}")
    raise

# Load OpenAI configuration
try:
    all_configs = autogen.config_list_from_json(CONFIG_FILE_PATH)
    if not all_configs:
        raise ValueError(f"No configurations found in {CONFIG_FILE_PATH}")

    # Filter for GPT-3.5-turbo
    gpt35_configs = [config for config in all_configs if config.get("model") == "gpt-3.5-turbo"]
    if not gpt35_configs:
        raise ValueError(f"No valid configurations found for the model 'gpt-3.5-turbo' in {CONFIG_FILE_PATH}")

    llm_config = {
        "config_list": gpt35_configs,
        "timeout": 120,
        "temperature": 0,
    }
    logger.info(f"Loaded OpenAI configuration for model: gpt-3.5-turbo")
except Exception as e:
    logger.error(f"Error loading LLM configuration: {e}")
    raise

# --- RAG Configurations ---
RETRIEVE_CONFIGS = {
    "annual_report": {
        "task": "qa",
        "vector_db": None,
        "docs_path": [os.path.join(REPORT_DIR, "Microsoft_Annual_Report_2023.pdf")],
        "chunk_token_size": 1000,
        "get_or_create": True,
        "collection_name": "msft_analysis_api",
        "must_break_at_empty_line": False,
        "embedding_function": embedding_function,
    },
    "10k": {
        "task": "qa",
        "vector_db": None,
        "docs_path": [os.path.join(REPORT_DIR, "2023-07-27_10-K_msft-20230630.htm.pdf")],
        "chunk_token_size": 2000,
        "collection_name": "msft_10k_api",
        "get_or_create": True,
        "must_break_at_empty_line": False,
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_function": embedding_function,
    }
}

# --- API Request/Response Models ---
class RagChatRequest(BaseModel):
    query: str
    report_type: str = Field(..., description="Should be 'annual_report' or '10k'")

class ForecastRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL, MSFT)")
    max_news: int = Field(10, description="Maximum number of news items to retrieve")

class StockChartRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL, MSFT)")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")

class StockFunctionRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL, MSFT)")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    end_date: str = Field(..., description="End date in YYYY-MM-DD format")

class RagQueryAdvancedRequest(BaseModel):
    ticker: str = Field(..., description="Stock ticker symbol (e.g., AAPL, MSFT)")
    year: str = Field(..., description="Year for the analysis (e.g., 2023)")
    query: str = Field(..., description="Question to ask about the company")
    source: str = Field("sec", description="Data source: 'sec' or 'earnings_call'")
    form_type: Optional[str] = Field(None, description="SEC form type (e.g., '10-K', '10-Q')")
    quarter: Optional[str] = Field(None, description="Quarter for earnings call (e.g., 'Q1', 'Q2')")

class UploadRequest(BaseModel):
    report_type: str = Field(..., description="Type of report being uploaded ('annual_report' or '10k')")
    company: str = Field(..., description="Company name or ticker symbol")
    year: str = Field(..., description="Year of the report")

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "FinRobot API is running", "endpoints": [
        "/upload_pdf",
        "/chat_rag", 
        "/forecast",
        "/stock_chart",
        "/stock_data",
        "/rag_advanced"
    ]}

@app.post("/chat_rag")
async def chat_rag_endpoint(request: RagChatRequest):
    """
    Endpoint to chat with a RAG agent based on a specified report type (from agent_rag_qa.ipynb).
    """
    logger.info(f"Received RAG request for report_type: {request.report_type} with query: '{request.query}'")

    if request.report_type not in RETRIEVE_CONFIGS:
        logger.error(f"Invalid report_type: {request.report_type}")
        raise HTTPException(status_code=400, detail=f"Invalid report_type. Choose from: {list(RETRIEVE_CONFIGS.keys())}")

    retrieve_config = RETRIEVE_CONFIGS[request.report_type]

    try:
        # Initialize the RAG agent
        logger.info("Initializing SingleAssistantRAG...")
        assistant = SingleAssistantRAG(
            "Data_Analyst",
            llm_config,
            human_input_mode="NEVER",
            retrieve_config=retrieve_config,
        )
        logger.info("SingleAssistantRAG initialized.")

        # Initiate the chat
        logger.info(f"Initiating chat with query: '{request.query}'")
        result = assistant.chat(request.query)
        
        # Extract the last message from the analyst
        chat_history = result.chat_history
        analyst_responses = [msg.get("content", "") for msg in chat_history if 
                            msg.get("role") == "assistant" and 
                            msg.get("content") is not None and 
                            not isinstance(msg.get("content"), dict)]
        
        # Get the last non-empty message
        answer = next((msg for msg in reversed(analyst_responses) if msg), "No response found")
        
        return {
            "query": request.query,
            "report_type": request.report_type,
            "answer": answer,
            "cost": result.cost
        }
        
    except Exception as e:
        logger.error(f"Error in RAG chat: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/forecast")
async def forecast_endpoint(request: ForecastRequest):
    """
    Endpoint for stock forecasting (from agent_fingpt_forecaster.ipynb)
    """
    logger.info(f"Received forecast request for ticker: {request.ticker}")
    
    try:
        # Create stock forecaster agent
        assistant = SingleAssistant(
            "Market_Analyst",
            llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=None,
        )
        
        # Prepare the query
        today = get_current_date()
        query = f"""
        Use all the tools provided to retrieve information available for {request.ticker} upon {today}. 
        Analyze the positive developments and potential concerns of {request.ticker} with 2-4 most important factors respectively and keep them concise. 
        Most factors should be inferred from company related news. 
        Then make a rough prediction (e.g. up/down by 2-3%) of the {request.ticker} stock price movement for next week. 
        Provide a summary analysis to support your prediction.
        """
        
        # Execute the chat
        result = assistant.chat(query)
        
        # Extract the response
        chat_history = result.chat_history
        analyst_responses = [msg.get("content", "") for msg in chat_history if 
                            msg.get("role") == "assistant" and 
                            msg.get("content") is not None and 
                            not isinstance(msg.get("content"), dict)]
        
        # Get the last non-empty message
        answer = next((msg for msg in reversed(analyst_responses) if msg), "No forecast could be generated")
        
        return {
            "ticker": request.ticker,
            "forecast": answer,
            "cost": result.cost
        }
        
    except Exception as e:
        logger.error(f"Error in forecast: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/stock_chart")
async def stock_chart_endpoint(request: StockChartRequest):
    """
    Endpoint to get stock chart data (from ollama stock chart.ipynb)
    """
    logger.info(f"Received stock chart request for {request.ticker} from {request.start_date} to {request.end_date}")
    
    try:
        # Use yfinance directly instead of ollama
        ticker_data = yf.download(
            request.ticker, 
            start=request.start_date, 
            end=request.end_date
        )
        
        if ticker_data.empty:
            return {
                "ticker": request.ticker,
                "message": "No data available for this ticker and date range",
                "data": []
            }
            
        # Convert DataFrame to dictionary for JSON response
        result = []
        for date, row in ticker_data.iterrows():
            result.append({
                "date": date.strftime('%Y-%m-%d'),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": int(row["Volume"])
            })
            
        return {
            "ticker": request.ticker,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "data": result
        }
        
    except Exception as e:
        logger.error(f"Error in stock chart: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/stock_data")
async def stock_function_endpoint(request: StockFunctionRequest):
    """
    Endpoint to get stock data with analysis (from ollama function call.ipynb)
    """
    logger.info(f"Received stock data request for {request.ticker} from {request.start_date} to {request.end_date}")
    
    try:
        # Get stock data using FinRobot's YFinanceUtils
        stock_data = YFinanceUtils.get_stock_data(
            symbol=request.ticker,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        # Create a market analyst to analyze the data
        analyst = SingleAssistant(
            "Market_Analyst",
            llm_config,
            human_input_mode="NEVER",
        )
        
        # Create a query about the stock data
        query = f"Analyze the stock price movement for {request.ticker} from {request.start_date} to {request.end_date}. " \
                f"Identify key trends, significant price movements, and provide a concise analysis."
        
        # Get the analysis
        result = analyst.chat(query)
        
        # Extract the response
        chat_history = result.chat_history
        analyst_responses = [msg.get("content", "") for msg in chat_history if 
                            msg.get("role") == "assistant" and 
                            msg.get("content") is not None and 
                            not isinstance(msg.get("content"), dict)]
        
        # Get the last non-empty message
        analysis = next((msg for msg in reversed(analyst_responses) if msg), "No analysis could be generated")
        
        # Convert stock data to dictionary for JSON response
        stock_dict = []
        if not stock_data.empty:
            for date, row in stock_data.iterrows():
                stock_dict.append({
                    "date": date.strftime('%Y-%m-%d'),
                    "open": float(row["Open"]) if "Open" in row else None,
                    "high": float(row["High"]) if "High" in row else None,
                    "low": float(row["Low"]) if "Low" in row else None,
                    "close": float(row["Close"]) if "Close" in row else None,
                    "volume": int(row["Volume"]) if "Volume" in row else None
                })
        
        return {
            "ticker": request.ticker,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "data": stock_dict,
            "analysis": analysis,
            "cost": result.cost
        }
        
    except Exception as e:
        logger.error(f"Error in stock data analysis: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/rag_advanced")
async def rag_advanced_endpoint(request: RagQueryAdvancedRequest):
    """
    Advanced RAG endpoint using SEC filings or earnings calls (from agent_rag_qa_up.ipynb)
    """
    logger.info(f"Received advanced RAG request for {request.ticker} ({request.year}) with query: '{request.query}'")
    
    try:
        # Based on the source, prepare the appropriate query function
        if request.source == "sec":
            query_database, form_names = rag_database_sec(
                ticker=request.ticker,
                year=request.year,
                FROM_MARKDOWN=False
            )
            
            # Use the specified form type or the first available one
            form_type = request.form_type if request.form_type else form_names[0]
            
            # Execute the query
            result = query_database(request.query, form_type)
            
        elif request.source == "earnings_call":
            query_database, quarter_vals, speaker_dict = rag_database_earnings_call(
                ticker=request.ticker,
                year=request.year
            )
            
            # Use the specified quarter or the first available one
            quarter = request.quarter if request.quarter else quarter_vals[0]
            
            # Execute the query
            result = query_database(request.query, quarter)
            
        else:
            raise HTTPException(status_code=400, detail=f"Invalid source: {request.source}. Choose 'sec' or 'earnings_call'")
        
        # Create an agent to analyze the result
        system_msg = f"""You are a helpful financial assistant analyzing {request.ticker} data from {request.year}.
        Provide a concise and accurate answer to the user's question based solely on the information provided.
        If the information is not available in the provided context, say so clearly."""
        
        user_proxy = autogen.ConversableAgent(
            name = "User",
            system_message=system_msg,
            code_execution_config=False,
            llm_config=llm_config,
            human_input_mode="NEVER",
        )
        
        # Create a prompt with the query and the retrieved context
        prompt = f"""
        Question: {request.query}
        
        Retrieved context:
        {result}
        
        Please provide a concise answer based on the retrieved information.
        """
        
        # Use the agent to analyze the result
        chat_result = user_proxy.initiate_chat(
            autogen.ConversableAgent(
                name="Analyst",
                llm_config=llm_config,
                human_input_mode="NEVER",
            ),
            message=prompt
        )
        
        # Extract the response
        chat_history = chat_result.chat_history
        responses = [msg.get("content", "") for msg in chat_history if 
                    msg.get("role") == "assistant" and 
                    msg.get("content") is not None and 
                    not isinstance(msg.get("content"), dict)]
        
        # Get the last non-empty message
        answer = next((msg for msg in reversed(responses) if msg), "No answer could be generated")
        
        return {
            "ticker": request.ticker,
            "year": request.year,
            "query": request.query,
            "source": request.source,
            "source_detail": request.form_type if request.source == "sec" else request.quarter,
            "answer": answer,
            "context": result
        }
        
    except Exception as e:
        logger.error(f"Error in advanced RAG: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/upload_pdf")
async def upload_pdf_endpoint(
    file: UploadFile = File(...),
    report_type: str = Query(..., description="Type of report ('annual_report' or '10k')"),
    company: str = Query(..., description="Company name or ticker symbol"),
    year: str = Query(..., description="Year of the report")
):
    """
    Upload a PDF file for use with the chat_rag endpoint.
    The file will be saved and made available for RAG operations.
    """
    logger.info(f"Received PDF upload request for {company} {year} {report_type}")
    
    # Validate report type
    if report_type not in ["annual_report", "10k"]:
        raise HTTPException(status_code=400, detail="Invalid report_type. Must be 'annual_report' or '10k'")
    
    # Create directory if it doesn't exist
    report_dir = Path(REPORT_DIR)
    report_dir.mkdir(exist_ok=True)
    
    # Generate filename based on metadata
    filename = f"{company}_{report_type}_{year}.pdf"
    file_path = report_dir / filename
    
    try:
        # Save the uploaded file
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Update the RETRIEVE_CONFIGS with the new file
        RETRIEVE_CONFIGS[report_type] = {
            "task": "qa",
            "vector_db": None,
            "docs_path": [str(file_path)],
            "chunk_token_size": 2000 if report_type == "10k" else 1000,
            "collection_name": f"{company.lower()}_{report_type}_{year}",
            "get_or_create": True,
            "must_break_at_empty_line": False,
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_function": embedding_function,
        }
        
        return {
            "message": "File uploaded successfully",
            "filename": filename,
            "company": company,
            "year": year,
            "report_type": report_type,
            "file_path": str(file_path)
        }
        
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")
    finally:
        file.file.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)

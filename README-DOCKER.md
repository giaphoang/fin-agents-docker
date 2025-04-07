# FinRobot RAG API - Docker Deployment

This README provides instructions for running the FinRobot RAG API using Docker.

## Prerequisites

- Docker and Docker Compose installed
- OpenAI API key (placed in `OAI_CONFIG_LIST` file)
- Report PDF files (placed in the `report` directory)

## Required Files

Before building and running the Docker container, ensure you have:

1. `OAI_CONFIG_LIST` file in the root directory with your OpenAI API key(s)
2. PDF files in the `report` directory:
   - `Microsoft_Annual_Report_2023.pdf` 
   - `2023-07-27_10-K_msft-20230630.htm.pdf`

## Building and Running

```bash
# Build and start the containers
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the containers
docker-compose down

# Rebuild if you make changes
docker-compose up -d --build
```

## API Endpoints

Once running, the following endpoints are available:

- `http://localhost:8001/` - Health check endpoint
- `http://localhost:8001/chat_rag_up` - Main RAG API endpoint (POST)

## Example curl request

```bash
curl -X POST "http://localhost:8001/chat_rag_up" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the strategy of Nvidia for artificial intelligence?"}'
```

## Persistent Data

The Docker setup includes persistent volumes for the vector databases:

- `finrobot-earnings-db` - Earnings call database
- `finrobot-sec-db` - SEC filings database
- `finrobot-sec-md-db` - SEC markdown filings database

These volumes persist even when containers are removed, preserving the generated vector embeddings.

## Troubleshooting

### Initialization Issues

The server may take a while to start up on the first run as it:
1. Downloads embedding models
2. Processes PDF files
3. Creates vector databases

Check the logs for progress: `docker-compose logs -f`

### TensorFlow/Transformers Errors

If you encounter errors related to TensorFlow or transformers, you may need to adjust the versions in the `requirements-docker.txt` file.

### API Key Issues

Make sure your `OAI_CONFIG_LIST` file contains a valid API key for the model specified in the code (`gpt-3.5-turbo`). 
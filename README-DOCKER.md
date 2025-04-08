# FinRobot RAG API - Docker Deployment

This README provides instructions for running the FinRobot RAG API using Docker.

## Prerequisites

- Docker and Docker Compose installed
- OpenAI API key (create `OAI_CONFIG_LIST` file based on the `OAI_CONFIG_LIST.local` sample)
- API keys for financial data sources (create `config_api_keys` file based on the `config_api_keys.local` sample)
- Report PDF files (placed in the `report` directory)

## Required Files

Before building and running the Docker container, ensure you have:

1. `OAI_CONFIG_LIST` file in the root directory with your OpenAI API key(s)
2. PDF files in the `report` directory:
   - `Microsoft_Annual_Report_2023.pdf` 
   - `2023-07-27_10-K_msft-20230630.htm.pdf`

## Building and Running

```bash
# Build docker (need to modify $IMAGE_NAME and $IMAGE_TAG)
docker image build \
  --build-arg username=$USER \
  --build-arg uid=$UID \
  --build-arg gid=$GID \
  --file Dockerfile \
  --tag $IMAGE_NAME:$IMAGE_TAG \
  ./

# Run container expose api (with coressponding $IMAGE_NAME and $IMAGE_TAG)
docker run -p 8888:8888 $IMAGE_NAME:$IMAGE_TAG

# View logs
docker-compose logs -f

# Stop the containers
docker-compose down

# Rebuild if you make changes
docker-compose up -d --build
```

### Can run in local

## Set up environment

Make sure you are in main directory 

# Conda
```bash
# creae conda environment
conda env create -f my_conda.yml

# activate conda environment
conda activate finrobot-test

```

# Python package

```bash

pip install requirements.txt

```

## Run fastapi server to test agents api
Currently main_up.py and main.py is stable for running fastapi request, I still developing a python file which utilize agents power of this repo. Here is the following sample for run in local
```bash
# run main_up.py
uvicorn main_up:app

```

Once running, the following endpoints are available:

main_up.py

- `{base_url}/` - Health check endpoint
- `{base_url}/chat_rag_up` - Main RAG API endpoint (POST)

main.py
- `{base_url}/` - Health check endpoint
- `{base_url}/chat_rag` - Main RAG API endpoint (POST) for querying annual reports and 10-K filings


## Example curl request
main_up.py (however this file have hardcord file uploaded so user can only question and answer that only file, which I need to develop a separate api endpoint but this is for testing the main endpoing)
```bash
curl -X POST "{base_url}/chat_rag" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the strategy of Nvidia for artificial intelligence?"}'
```

main.py
```bash
curl -X POST "{base_url}/chat_rag_up" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the strategy of Nvidia for artificial intelligence?"}'
```

## Persistent Data

The Docker setup includes persistent volumes for the vector databases:

- `finrobot-earnings-db` - Earnings call database
- `finrobot-sec-db` - SEC filings database
- `finrobot-sec-md-db` - SEC markdown filings database

## Current Issue
- Still have conflict in building docker process
- install too many requiremnts.txt due to no exact dependencies cost a lot of time
- still cannot expose api

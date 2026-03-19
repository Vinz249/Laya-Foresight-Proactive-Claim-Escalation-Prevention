# Laya Foresight AI Agent Server (Backend)

This repository contains the backend server for the Laya Foresight project. The server hosts a FastAPI application that runs an AI agent (to simulate support escalation preventions) and streams reasoning events to the frontend dashboard.

## Prerequisites

- **Python 3.8+**
- **pip** package manager

## Setup and Installation

### 1. Clone the Repository
Clone the backend repository to your local machine:
```bash
git clone https://github.com/Laya-hackathon/laya-foresight.git
cd "laya-foresight/AI Agent"
```

### 2. Create a Virtual Environment (Recommended)
It's a best practice to keep your Python dependencies isolated.
```bash
python3 -m venv venv
```

**Activate the virtual environment:**
- **Mac/Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

### 3. Install Dependencies
Install the required packages using pip:
```bash
pip install -r requirements.txt
```
*(Dependencies include FastAPI, Uvicorn, Python-Dotenv, OpenAI SDK, psycopg2)*

### 4. Environment Variables
Create a `.env` file in the root of the backend folder (`AI_Agent`) and add the necessary configuration:
```env
# GitHub Models API token (used to call gpt-4o-mini via Azure inference)
GITHUB_TOKEN=your_github_token_here

# AI model to use (default: gpt-4o-mini)
MODEL=gpt-4o-mini

# PostgreSQL connection string (Supabase or self-hosted)
DATABASE_URL=postgresql://user:password@host:port/dbname

# Brevo (email sending)
EMAIL_API_KEY=your_brevo_api_key_here
EMAIL_ADDRESS=your_sender_email@domain.com
```

## Running Locally

Start the FastAPI application directly:
```bash
python server.py
```

Alternatively, you can run it directly with Uvicorn:
```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

The standard backend API will run on **`http://localhost:8000`**.



## Key API Endpoints
- `GET /` : Serves a built-in demo dashboard (if using the bundled HTML).
- `GET /api/health` : Checks server status, API token configuration, and selected model.
- `POST /api/ingest` : Receives an ML model prediction and stores it in the database.
- `GET /api/scenarios` : Lists all scenarios built from database predictions.
- `GET /api/run/{scenario_id}` : Starts an agent session for a given scenario, streaming output via Server-Sent Events (SSE).
- `GET /api/history/{scenario_id}` : Returns the stored reasoning and tool calls for a completed run.
- `GET /api/stats` : Returns today's dashboard statistics (risk counts, actions taken).
- `GET /api/feed` : Returns the latest activity feed across all action tables.
- `GET /api/chart` : Returns risk score distribution data for the bar chart.

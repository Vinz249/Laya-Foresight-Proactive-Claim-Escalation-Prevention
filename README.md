# Laya Foresight: Proactive Claim Escalation Prevention

An AI-driven system built for Laya Healthcare to predict and prevent costly support call escalations before they happen, giving users peace of mind during their health insurance claim process.

![Laya Foresight Dashboard Preview](https://via.placeholder.com/800x400?text=Laya+Foresight+Dashboard) *(Note: Replace with actual screenshot/GIF if available)*

---

## 🛠 Tech Stack

**Frontend:**
- **React 18** – UI Component Library
- **Vite** – Build Tool and Development Server

**Backend (AI Agent):**
- **Python 3** – Core Agent Logic
- **FastAPI & Uvicorn** – High-performance REST APIs
- **Anthropic API (Claude)** – Large Language Model reasoning for the AI Agent

---

## 💡 What it does

Laya Foresight is a two-component, proactive intervention system. Instead of waiting for users to get anxious about a claim and call support, the system continuously analyzes active claims to identify users who are likely to escalate. When a high risk is detected, the AI Agent either autonomously reaches out to the user with a reassuring update or surfaces a prioritized alert directly to the Laya Healthcare support team.

## ⚙️ How it does it

The system operates in two core components:

### 1. Risk Predictor Model
A binary classification model that scores every active claim on a rolling basis, assigning users into **High, Medium, or Low** risk bands. It evaluates:
- **Behavioural signals:** App logins, status page reloads, document uploads.
- **Claim signals:** Type of claim, days since submission, estimated amount.
- **Historical signals:** Prior escalation history.

### 2. LayaAIAgent (Anthropic Powered)
An intelligent, autonomous agent that acts on the risk scores continuously:
- **Internal Action:** Provides support teams with a ranked, prioritized dashboard of at-risk users, detailing *why* they might call.
- **External Action:** Decides whether messaging the user directly will help. If it determines a message will reduce anxiety (e.g., an update on processing time, or a note about a missing document), it dispatches the message autonomously.

## 🤔 Why it does it & What problem it solves

When a user submits a health insurance claim, they enter a "black box" of uncertainty, driving immense anxiety. This anxiety manifests as constant app refreshing, re-uploading documents, and ultimately—calling support. 

**The Problem:**
- Support calls are highly expensive, consuming vast human resources.
- Most calls are entirely avoidable, stemming from a lack of transparent, timely updates.

**The Solution:**
Laya Foresight flips the paradigm from reactive to highly proactive. By intervening *before* the user picks up the phone, we drastically reduce call center volume, lower operational costs, and create a markedly better, anxiety-free user experience.

## 📊 The Numerics
- **48-Hour Prediction Window:** The model predicts if a user will call support within the next 48 hours.
- **3 Risk Tiers:** Accurately segments users into High, Medium, or Low urgency bands.
- **Measurable Impact:** Designed to measurably decrease inbound call queues while increasing user satisfaction scores.

---

## 🚀 Full Setup Guide

To run the complete Laya Foresight system locally, you need to set up both the backend API and the frontend dashboard.

### 1. Clone the Repository

Clone the master project folder containing both repositories (if applicable):
```bash
git clone https://github.com/Laya-hackathon/laya-foresight
cd LayaForesight
```

### 2. Backend Setup (AI Agent Server)

The backend handles the AI reasoning, mock data generation, and API endpoints.

```bash
# Navigate to the backend directory
cd laya-foresight/AI_Agent

# Create a virtual environment (Recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install dependencies
pip install -r requirements.txt
```

**Environment Variables:**
Create a `.env` file inside the `AI_Agent` folder and add your API keys:
```env
# Example .env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**Run the Server:**
```bash
# Start the FastAPI server using Uvicorn
uvicorn server:app --reload
```
*The backend server will run on `http://localhost:8000`.*

### 3. Frontend Setup (React Dashboard)

The frontend provides a real-time visualization of the AI Agent's reasoning and mock customer support scenarios.

Open a new terminal window:
```bash
# Navigate to the frontend directory
cd laya-foresight-frontend/laya-foresight-frontend

# Install dependencies using npm
npm install
```

**Environment Configuration:**
Create a `.env` file in the root of the frontend directory:
```env
VITE_API_BASE_URL=http://localhost:8000
```
*(If omitted, it defaults to localhost:8000 anyway).*

**Run the Dashboard:**
```bash
# Start the Vite development server
npm run dev
```
*The dashboard will be available at `http://localhost:5173/`.*

---

## 🔮 Status & Next Steps
Currently in the early-stage design and architecture phase. Ongoing collaboration with Laya Healthcare to instrument behavioral event tracking inside their app and compile a labeled dataset for the risk model.

### Planned Orchestration Refactor
While the current AI Agent operates on a custom cyclic while-loop built in core Python, a planned future update includes migrating the tool-binding and orchestration logic to **LangChain** and **LangGraph**. This upgrade will provide:
- **Stateful, Native Cyclic Workflows:** More robust handling of multi-step, human-in-the-loop decision processes.
- **Enhanced Fault Tolerance:** Native rate-limit, error, and retry handling.
- **Standardized Tool Integration:** A unified interface for interacting with the Anthropic LLM and internal APIs.

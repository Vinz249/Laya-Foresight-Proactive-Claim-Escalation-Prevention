
import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

load_dotenv()

from agent import run_agent_streaming


app = FastAPI(title="LayaAIAgent Dashboard", version="1.0.0")

@app.on_event("startup")
async def startup():
    from database import db_setup_hitl_tables
    db_setup_hitl_tables()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(content=html_path.read_text(), status_code=200)


@app.post("/api/ingest")
async def ingest_prediction(request: Request):
    """Receive ML model prediction and store in DB."""
    from database import db_get_claim_details, db_get_user_details, db_log_ml_prediction

    body = await request.json()

    member_id = body.get("member_id", "unknown")
    claim_id = body.get("claim_id")
    probability = float(body.get("risk_probability", 0.0))
    predicted_risk = int(body.get("predicted_risk", 0))
    risk_band = "HIGH" if probability >= 0.7 else ("MEDIUM" if probability >= 0.4 else "LOW")

    db_log_ml_prediction(member_id, claim_id, predicted_risk, probability, risk_band)
    print(f"📥 Ingested: {member_id} | claim={claim_id} | {risk_band} ({probability:.4f})")
    return {"status": "ok", "scenario_id": f"pred_{member_id}", "risk_band": risk_band}


@app.get("/api/scenarios")
async def list_scenarios():
    from database import db_get_scenarios
    return {"scenarios": db_get_scenarios()}


@app.get("/api/run/{scenario_id}")
async def run_scenario_stream(scenario_id: str, request: Request):
    from database import db_get_scenarios
    scenarios = db_get_scenarios()
    scenario = next((s for s in scenarios if s["id"] == scenario_id), None)
    if not scenario:
        return {"error": f"Scenario {scenario_id} not found"}

    async def event_generator():
        import queue
        import threading

        q = queue.Queue()

        def run_in_thread():
            try:
                for event in run_agent_streaming(scenario):
                    q.put(event)
            except Exception as e:
                q.put({"type": "error", "data": {"message": str(e)}})
            finally:
                q.put(None)

        thread = threading.Thread(target=run_in_thread, daemon=True)
        thread.start()

        while True:
            if await request.is_disconnected():
                break

            try:
                event = q.get_nowait()
            except queue.Empty:
                yield ": keepalive\n\n"
                await asyncio.sleep(0.1)
                continue

            if event is None:
                break

            payload = json.dumps(event)
            padding = " " * max(0, 1024 - len(payload))
            yield f"data: {payload}{padding}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        }
    )


@app.get("/api/claims")
async def get_claims():
    from database import db_get_all_claims
    return {"claims": db_get_all_claims()}


@app.get("/api/claims/{claim_id}/activity")
async def get_claim_activity(claim_id: str, user_id: str):
    from database import db_get_app_logs
    logs = db_get_app_logs(user_id, claim_id)
    return {"logs": logs if isinstance(logs, list) else []}


@app.get("/api/history/{scenario_id}")
async def get_history(scenario_id: str):
    from database import db_get_run_history
    return {"events": db_get_run_history(scenario_id)}


@app.get("/api/stats")
async def get_stats():
    from database import db_get_stats
    return db_get_stats()


@app.get("/api/feed")
async def get_feed():
    from database import db_get_feed
    return {"feed": db_get_feed()}


@app.get("/api/chart")
async def get_chart():
    from database import db_get_chart_data
    return {"bins": db_get_chart_data()}


@app.get("/api/reports")
async def get_reports():
    from database import db_get_reports
    return {"reports": db_get_reports()}


@app.get("/api/reports/{run_id}")
async def get_report_detail(run_id: int):
    from database import db_get_report_detail
    return db_get_report_detail(run_id)


@app.get("/api/alerts")
async def get_alerts():
    from database import db_get_all_alerts
    return {"alerts": db_get_all_alerts()}


@app.get("/api/questions/pending")
async def get_pending_questions():
    from database import db_get_pending_questions
    return {"questions": db_get_pending_questions()}


@app.post("/api/questions/{question_id}/respond")
async def respond_to_question(question_id: int, request: Request):
    body = await request.json()
    employee_response = body.get("response", "").strip()
    if not employee_response:
        return {"error": "Response cannot be empty"}

    from database import db_respond_to_question, db_get_question
    db_respond_to_question(question_id, employee_response)
    question = db_get_question(question_id)
    return {"status": "ok", "scenario_id": question.get("scenario_id")}


@app.get("/api/resume/{scenario_id}")
async def resume_scenario(scenario_id: str, question_id: int, request: Request):
    from agent import resume_agent_streaming

    async def event_generator():
        import queue, threading
        q = queue.Queue()

        def run_in_thread():
            try:
                for event in resume_agent_streaming(question_id):
                    q.put(event)
            except Exception as e:
                q.put({"type": "error", "data": {"message": str(e)}})
            finally:
                q.put(None)

        threading.Thread(target=run_in_thread, daemon=True).start()

        while True:
            if await request.is_disconnected():
                break
            try:
                event = q.get_nowait()
            except queue.Empty:
                yield ": keepalive\n\n"
                await asyncio.sleep(0.1)
                continue
            if event is None:
                break
            payload = json.dumps(event)
            padding = " " * max(0, 1024 - len(payload))
            yield f"data: {payload}{padding}\n\n"
            await asyncio.sleep(0)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache, no-store", "Connection": "keep-alive", "X-Accel-Buffering": "no"}
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "api_key_configured": bool(os.getenv("GITHUB_TOKEN")),
        "model": os.getenv("MODEL", "gpt-4o-mini"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


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


# In-memory store for ML-ingested scenarios
_ingested_scenarios: list[dict] = []


def _get_scenario(scenario_id: str) -> dict | None:
    return next((s for s in _ingested_scenarios if s["id"] == scenario_id), None)


app = FastAPI(title="LayaAIAgent Dashboard", version="1.0.0")

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
    """Receive ML model prediction, enrich with DB data, and store as a scenario."""
    from database import db_get_claim_details, db_get_user_details, db_get_app_logs, db_log_ml_prediction

    body = await request.json()

    member_id = body.get("member_id", "unknown")
    claim_id = body.get("claim_id")
    probability = float(body.get("risk_probability", 0.0))
    predicted_risk = int(body.get("predicted_risk", 0))
    risk_band = "HIGH" if probability >= 0.7 else ("MEDIUM" if probability >= 0.4 else "LOW")

    claim = db_get_claim_details(claim_id) if claim_id else {}
    user = db_get_user_details(member_id) if member_id else {}
    app_logs = db_get_app_logs(member_id, claim_id) if member_id and claim_id else []

    scenario_id = f"pred_{member_id}"
    scenario = {
        "id": scenario_id,
        "risk_band": risk_band,
        "risk_score": probability,
        "claim_id": claim_id,
        "user_id": member_id,
        "claim": claim if "error" not in claim else {},
        "user": user if "error" not in user else {},
        "app_logs": app_logs if isinstance(app_logs, list) else [],
    }

    existing = next((i for i, s in enumerate(_ingested_scenarios) if s["id"] == scenario_id), None)
    if existing is not None:
        _ingested_scenarios[existing] = scenario
    else:
        _ingested_scenarios.append(scenario)

    db_log_ml_prediction(member_id, claim_id, predicted_risk, probability, risk_band)
    print(f"📥 Ingested: {member_id} | claim={claim_id} | {risk_band} ({probability:.4f})")
    return {"status": "ok", "scenario_id": scenario_id, "risk_band": risk_band}


@app.get("/api/scenarios")
async def list_scenarios():
    return {"scenarios": _ingested_scenarios}


@app.post("/api/reset")
async def reset_scenarios():
    _ingested_scenarios.clear()
    return {"status": "ok"}


@app.get("/api/run/{scenario_id}")
async def run_scenario_stream(scenario_id: str, request: Request):
    scenario = _get_scenario(scenario_id)
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

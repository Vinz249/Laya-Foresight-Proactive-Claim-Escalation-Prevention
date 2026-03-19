
import os
import json
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return psycopg2.connect(os.getenv("DATABASE_URL"), cursor_factory=psycopg2.extras.RealDictCursor)


# ---------------------------------------------------------------------------
# READ
# ---------------------------------------------------------------------------

def db_get_claim_details(claim_id: str) -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM claims WHERE claim_id = %s", (claim_id,))
            row = cur.fetchone()
    if row:
        return dict(row)
    return {"error": f"Claim {claim_id} not found"}


def db_get_user_details(user_id: str) -> dict:
    # TODO: switch back to 'users' once first_name, last_name, email columns are added to the real table
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM users_backup WHERE user_id = %s", (user_id,))
            row = cur.fetchone()
    if row:
        return dict(row)
    return {"error": f"User {user_id} not found"}


def db_get_app_logs(user_id: str, claim_id: str) -> list:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT * FROM app_logs
                WHERE user_id = %s AND claim_id = %s
                ORDER BY timestamp ASC
            """, (user_id, claim_id))
            rows = cur.fetchall()
    if rows:
        return [dict(row) for row in rows]
    return {"error": f"No app logs found for user {user_id} and claim {claim_id}"}


# ---------------------------------------------------------------------------
# aa_ml_predictions
# ---------------------------------------------------------------------------

def db_log_ml_prediction(member_id: str, claim_id: str, predicted_risk: int, risk_probability: float, risk_band: str) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO aa_ml_predictions (member_id, claim_id, predicted_risk, risk_probability, risk_band)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING prediction_id
            """, (member_id, claim_id, predicted_risk, risk_probability, risk_band))
            prediction_id = cur.fetchone()["prediction_id"]
        conn.commit()
    return prediction_id


# ---------------------------------------------------------------------------
# aa_agent_runs
# ---------------------------------------------------------------------------

def db_create_agent_run(scenario_id: str, member_id: str, claim_id: str, risk_score: float, risk_band: str, model_name: str) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO aa_agent_runs (scenario_id, member_id, claim_id, risk_score, risk_band, model_name, status)
                VALUES (%s, %s, %s, %s, %s, %s, 'running')
                RETURNING run_id
            """, (scenario_id, member_id, claim_id, risk_score, risk_band, model_name))
            run_id = cur.fetchone()["run_id"]
        conn.commit()
    return run_id


def db_update_agent_run(run_id: int, status: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE aa_agent_runs SET status = %s, ended_at = NOW() WHERE run_id = %s
            """, (status, run_id))
        conn.commit()


# ---------------------------------------------------------------------------
# aa_agent_reasoning_steps
# ---------------------------------------------------------------------------

def db_log_reasoning_step(run_id: int, step_number: int, reasoning_text: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO aa_agent_reasoning_steps (run_id, step_number, reasoning_text)
                VALUES (%s, %s, %s)
            """, (run_id, step_number, reasoning_text))
        conn.commit()


# ---------------------------------------------------------------------------
# aa_agent_tool_calls
# ---------------------------------------------------------------------------

def db_log_tool_call(run_id: int, step_number: int, tool_name: str, tool_input: dict, tool_result: dict):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO aa_agent_tool_calls (run_id, step_number, tool_name, tool_input, tool_result)
                VALUES (%s, %s, %s, %s, %s)
            """, (run_id, step_number, tool_name, json.dumps(tool_input), json.dumps(tool_result)))
        conn.commit()


# ---------------------------------------------------------------------------
# aa_employee_alerts
# ---------------------------------------------------------------------------

def db_log_employee_alert(run_id: int, claim_id: str, message: str, urgency: str, sla_minutes: int):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO aa_employee_alerts (run_id, claim_id, message, urgency, sla_minutes)
                VALUES (%s, %s, %s, %s, %s)
            """, (run_id, claim_id, message, urgency, sla_minutes))
        conn.commit()


# ---------------------------------------------------------------------------
# aa_customer_emails
# ---------------------------------------------------------------------------

def db_log_customer_email(run_id: int, user_id: str, subject: str, body_html: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO aa_customer_emails (run_id, user_id, subject, body_html)
                VALUES (%s, %s, %s, %s)
            """, (run_id, user_id, subject, body_html))
        conn.commit()


# ---------------------------------------------------------------------------
# aa_customer_notifications
# ---------------------------------------------------------------------------

def db_log_customer_notification(run_id: int, user_id: str, title: str, body: str, deep_link: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO aa_customer_notifications (run_id, user_id, title, body, deep_link)
                VALUES (%s, %s, %s, %s, %s)
            """, (run_id, user_id, title, body, deep_link))
        conn.commit()


# ---------------------------------------------------------------------------
# aa_scheduled_callbacks
# ---------------------------------------------------------------------------

def db_log_scheduled_callback(run_id: int, callback_id: str, user_id: str, claim_id: str, priority: str, notes: str, assigned_to: str, scheduled_for: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO aa_scheduled_callbacks (callback_id, run_id, user_id, claim_id, priority, notes, assigned_to, scheduled_for)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (callback_id) DO NOTHING
            """, (callback_id, run_id, user_id, claim_id, priority, notes, assigned_to, scheduled_for))
        conn.commit()


# ---------------------------------------------------------------------------
# aa_interventions
# ---------------------------------------------------------------------------

def db_log_intervention(run_id: int, intervention_id: str, claim_id: str, actions_taken: list, reasoning: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO aa_interventions (intervention_id, run_id, claim_id, actions_taken, reasoning, actions_count)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (intervention_id) DO NOTHING
            """, (intervention_id, run_id, claim_id, json.dumps(actions_taken), reasoning, len(actions_taken)))
        conn.commit()


# ---------------------------------------------------------------------------
# aa_agent_errors
# ---------------------------------------------------------------------------

def db_log_agent_error(run_id: int, error_message: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO aa_agent_errors (run_id, error_message)
                VALUES (%s, %s)
            """, (run_id, error_message))
        conn.commit()


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------

def db_get_stats() -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE risk_band = 'HIGH')   AS high_risk,
                    COUNT(*) FILTER (WHERE risk_band = 'MEDIUM') AS medium_risk,
                    COUNT(*) FILTER (WHERE risk_band = 'LOW')    AS low_risk,
                    COUNT(*)                                      AS total
                FROM aa_ml_predictions
                WHERE DATE(created_at) = CURRENT_DATE
            """)
            predictions = dict(cur.fetchone())

            cur.execute("SELECT COUNT(*) AS cnt FROM aa_interventions WHERE DATE(created_at) = CURRENT_DATE")
            prevented = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) AS cnt FROM aa_customer_emails WHERE DATE(created_at) = CURRENT_DATE")
            emails = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) AS cnt FROM aa_customer_notifications WHERE DATE(created_at) = CURRENT_DATE")
            push = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) AS cnt FROM aa_employee_alerts WHERE DATE(created_at) = CURRENT_DATE")
            slack = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) AS cnt FROM aa_agent_tool_calls WHERE DATE(created_at) = CURRENT_DATE")
            total_actions = cur.fetchone()["cnt"]

    return {
        "high_risk": predictions["high_risk"],
        "medium_risk": predictions["medium_risk"],
        "low_risk": predictions["low_risk"],
        "total_predictions": predictions["total"],
        "calls_prevented": prevented,
        "emails_sent": emails,
        "push_sent": push,
        "slack_sent": slack,
        "total_actions": total_actions,
    }


def db_get_feed(limit: int = 20) -> list:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 'email' AS type, user_id AS ref, subject AS text, created_at FROM aa_customer_emails
                UNION ALL
                SELECT 'alert', claim_id, message, created_at FROM aa_employee_alerts
                UNION ALL
                SELECT 'notification', user_id, title, created_at FROM aa_customer_notifications
                UNION ALL
                SELECT 'callback', user_id, CONCAT('Priority: ', priority), created_at FROM aa_scheduled_callbacks
                UNION ALL
                SELECT 'intervention', claim_id, reasoning, created_at FROM aa_interventions
                UNION ALL
                SELECT 'prediction', member_id, CONCAT('Risk: ', risk_band, ' (', ROUND(risk_probability::numeric, 2), ')'), created_at FROM aa_ml_predictions
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def db_get_chart_data() -> list:
    bins = [
        (0.0, 0.2), (0.2, 0.4), (0.4, 0.5),
        (0.5, 0.6), (0.6, 0.7), (0.7, 0.8),
        (0.8, 0.9), (0.9, 1.01)
    ]
    colors = ['#86efac', '#4ade80', '#fcd34d', '#fbbf24', '#f97316', '#f43f5e', '#e11d48', '#be123c']
    with get_connection() as conn:
        with conn.cursor() as cur:
            result = []
            for (lo, hi), col in zip(bins, colors):
                cur.execute("""
                    SELECT COUNT(*) AS cnt FROM aa_ml_predictions
                    WHERE risk_probability >= %s AND risk_probability < %s
                """, (lo, hi))
                count = cur.fetchone()["cnt"]
                result.append({"c": count, "col": col})
    return result


# ---------------------------------------------------------------------------
# Scenarios (built from aa_ml_predictions + live DB lookups)
# ---------------------------------------------------------------------------

def db_get_scenarios() -> list:
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Latest prediction per member
            cur.execute("""
                SELECT DISTINCT ON (member_id)
                    member_id, claim_id, risk_probability, risk_band
                FROM aa_ml_predictions
                ORDER BY member_id, prediction_id DESC
            """)
            rows = cur.fetchall()

    scenarios = []
    for row in rows:
        member_id = row["member_id"]
        claim_id = row["claim_id"]
        claim = db_get_claim_details(claim_id) if claim_id else {}
        user = db_get_user_details(member_id) if member_id else {}
        app_logs = db_get_app_logs(member_id, claim_id) if member_id and claim_id else []
        scenarios.append({
            "id": f"pred_{member_id}",
            "risk_band": row["risk_band"],
            "risk_score": row["risk_probability"],
            "claim_id": claim_id,
            "user_id": member_id,
            "claim": claim if "error" not in claim else {},
            "user": user if "error" not in user else {},
            "app_logs": app_logs if isinstance(app_logs, list) else [],
        })
    return scenarios


# ---------------------------------------------------------------------------
# Agent run history (for drawer replay on reload)
# ---------------------------------------------------------------------------

def db_get_run_history(scenario_id: str) -> list:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT run_id, status FROM aa_agent_runs
                WHERE scenario_id = %s
                ORDER BY run_id DESC LIMIT 1
            """, (scenario_id,))
            run = cur.fetchone()
            if not run:
                return []
            run_id = run["run_id"]
            status = run["status"]

            cur.execute("""
                SELECT step_number, reasoning_text FROM aa_agent_reasoning_steps
                WHERE run_id = %s ORDER BY step_number
            """, (run_id,))
            reasoning_rows = cur.fetchall()

            cur.execute("""
                SELECT step_number, tool_name, tool_input, tool_result
                FROM aa_agent_tool_calls
                WHERE run_id = %s ORDER BY step_number, tool_call_id
            """, (run_id,))
            tool_rows = cur.fetchall()

    steps = {}
    for r in reasoning_rows:
        steps.setdefault(r["step_number"], {"reasoning": None, "tools": []})
        steps[r["step_number"]]["reasoning"] = r["reasoning_text"]
    for t in tool_rows:
        steps.setdefault(t["step_number"], {"reasoning": None, "tools": []})
        steps[t["step_number"]]["tools"].append(t)

    events = []
    for step_num in sorted(steps.keys()):
        step = steps[step_num]
        if step["reasoning"]:
            events.append({"type": "reasoning", "data": {"step": step_num, "text": step["reasoning"]}})
        for t in step["tools"]:
            events.append({"type": "tool_call", "data": {
                "step": step_num,
                "tool_name": t["tool_name"],
                "tool_input": t["tool_input"],
                "tool_id": f"hist_{step_num}",
            }})
            events.append({"type": "tool_result", "data": {
                "step": step_num,
                "tool_name": t["tool_name"],
                "result": t["tool_result"],
            }})

    msg = "✅ Agent task complete." if status == "complete" else f"⚠️ Agent ended ({status})"
    events.append({"type": "complete", "data": {"message": msg}})
    return events


if __name__ == "__main__":
    user_id = "U00001"
    claim_id = "CLM000001"
    details = db_get_app_logs(user_id, claim_id)
    print(json.dumps(details, indent=4))


import os
import json
import threading
import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Connection pool — reuses TCP connections instead of opening a new one
# on every query (was the primary cause of 14-26s response times).
# ---------------------------------------------------------------------------

_pool: ThreadedConnectionPool | None = None
_pool_lock = threading.Lock()


def _get_pool() -> ThreadedConnectionPool:
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = ThreadedConnectionPool(
                    2, 10,
                    dsn=os.getenv("DATABASE_URL"),
                    cursor_factory=psycopg2.extras.RealDictCursor,
                )
    return _pool


@contextmanager
def get_connection():
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


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
                  AND timestamp::timestamptz >= NOW() - INTERVAL '48 hours'
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
    # Single round-trip: all counts in one query using CROSS JOIN subqueries.
    # created_at >= CURRENT_DATE uses an index range scan; DATE(created_at)
    # applied a function to every row and prevented index use entirely.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    p.high_risk, p.medium_risk, p.low_risk, p.total_predictions,
                    COALESCE(i.calls_prevented, 0)  AS calls_prevented,
                    COALESCE(e.emails_sent,     0)  AS emails_sent,
                    COALESCE(n.push_sent,       0)  AS push_sent,
                    COALESCE(ea.slack_sent,     0)  AS slack_sent,
                    COALESCE(tc.total_actions,  0)  AS total_actions
                FROM (
                    SELECT
                        COUNT(*) FILTER (WHERE risk_band = 'HIGH')   AS high_risk,
                        COUNT(*) FILTER (WHERE risk_band = 'MEDIUM') AS medium_risk,
                        COUNT(*) FILTER (WHERE risk_band = 'LOW')    AS low_risk,
                        COUNT(*)                                      AS total_predictions
                    FROM aa_ml_predictions
                    WHERE created_at >= CURRENT_DATE
                ) p
                CROSS JOIN (SELECT COUNT(*) AS calls_prevented FROM aa_interventions          WHERE created_at >= CURRENT_DATE) i
                CROSS JOIN (SELECT COUNT(*) AS emails_sent     FROM aa_customer_emails        WHERE created_at >= CURRENT_DATE) e
                CROSS JOIN (SELECT COUNT(*) AS push_sent       FROM aa_customer_notifications WHERE created_at >= CURRENT_DATE) n
                CROSS JOIN (SELECT COUNT(*) AS slack_sent      FROM aa_employee_alerts        WHERE created_at >= CURRENT_DATE) ea
                CROSS JOIN (SELECT COUNT(*) AS total_actions   FROM aa_agent_tool_calls       WHERE created_at >= CURRENT_DATE) tc
            """)
            return dict(cur.fetchone())


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
    colors = ['#86efac', '#4ade80', '#fcd34d', '#fbbf24', '#f97316', '#f43f5e', '#e11d48', '#be123c']
    # One table scan with FILTER instead of 8 separate queries in a loop.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) FILTER (WHERE risk_probability >= 0.0 AND risk_probability < 0.2) AS bin0,
                    COUNT(*) FILTER (WHERE risk_probability >= 0.2 AND risk_probability < 0.4) AS bin1,
                    COUNT(*) FILTER (WHERE risk_probability >= 0.4 AND risk_probability < 0.5) AS bin2,
                    COUNT(*) FILTER (WHERE risk_probability >= 0.5 AND risk_probability < 0.6) AS bin3,
                    COUNT(*) FILTER (WHERE risk_probability >= 0.6 AND risk_probability < 0.7) AS bin4,
                    COUNT(*) FILTER (WHERE risk_probability >= 0.7 AND risk_probability < 0.8) AS bin5,
                    COUNT(*) FILTER (WHERE risk_probability >= 0.8 AND risk_probability < 0.9) AS bin6,
                    COUNT(*) FILTER (WHERE risk_probability >= 0.9)                            AS bin7
                FROM aa_ml_predictions
            """)
            row = cur.fetchone()
    return [{"c": row[f"bin{i}"], "col": col} for i, col in enumerate(colors)]


# ---------------------------------------------------------------------------
# All claims (with user info + latest ML prediction if flagged)
# ---------------------------------------------------------------------------

def db_get_all_claims() -> list:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    c.claim_id, c.user_id, c.treatment_type, c.claim_amount,
                    c.submission_timestamp, c.submission_channel,
                    c.missing_documents_flag, c.adjudicator_flag,
                    c.claim_rejected_flag, c.resubmission_flag, c.original_claim_id,
                    u.first_name, u.last_name, u.email,
                    u.age_group, u.region, u.plan_type,
                    u.membership_tenure_years, u.past_escalation_count,
                    u.behavior_archetype,
                    p.risk_band, p.risk_probability,
                    p.created_at AS predicted_at
                FROM claims c
                LEFT JOIN users_backup u ON u.user_id = c.user_id
                LEFT JOIN (
                    SELECT DISTINCT ON (member_id)
                        member_id, risk_band, risk_probability, created_at
                    FROM aa_ml_predictions
                    ORDER BY member_id, prediction_id DESC
                ) p ON p.member_id = c.user_id
                ORDER BY
                    CASE
                        WHEN p.risk_band = 'HIGH'   THEN 1
                        WHEN p.risk_band = 'MEDIUM' THEN 2
                        WHEN p.risk_band = 'LOW'    THEN 3
                        ELSE 4
                    END,
                    p.risk_probability DESC NULLS LAST,
                    c.submission_timestamp DESC
            """)
            rows = cur.fetchall()
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Scenarios (built from aa_ml_predictions + live DB lookups)
# ---------------------------------------------------------------------------

def db_get_scenarios() -> list:
    # Single JOIN query replaces the N+1 loop that opened 3 new DB connections
    # per row (claim + user + app_logs). With 10 scenarios that was 31 connections;
    # with 20 it was 61 — each TCP handshake ~500ms = 10-30s total latency.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    p.member_id, p.claim_id, p.risk_probability, p.risk_band,
                    c.treatment_type, c.claim_amount, c.submission_timestamp,
                    c.submission_channel, c.missing_documents_flag,
                    c.adjudicator_flag, c.claim_rejected_flag,
                    c.resubmission_flag, c.original_claim_id,
                    u.first_name, u.last_name, u.email,
                    u.age_group, u.region, u.plan_type,
                    u.membership_tenure_years, u.past_escalation_count,
                    u.behavior_archetype
                FROM (
                    SELECT DISTINCT ON (member_id)
                        member_id, claim_id, risk_probability, risk_band
                    FROM aa_ml_predictions
                    ORDER BY member_id, prediction_id DESC
                ) p
                LEFT JOIN claims       c ON c.claim_id = p.claim_id
                LEFT JOIN users_backup u ON u.user_id  = p.member_id
            """)
            rows = cur.fetchall()

    scenarios = []
    for row in rows:
        row = dict(row)
        scenarios.append({
            "id":         f"pred_{row['member_id']}",
            "risk_band":  row["risk_band"],
            "risk_score": row["risk_probability"],
            "claim_id":   row["claim_id"],
            "user_id":    row["member_id"],
            "claim": {
                "treatment_type":         row.get("treatment_type"),
                "claim_amount":           row.get("claim_amount"),
                "submission_timestamp":   row.get("submission_timestamp"),
                "submission_channel":     row.get("submission_channel"),
                "missing_documents_flag": row.get("missing_documents_flag"),
                "adjudicator_flag":       row.get("adjudicator_flag"),
                "claim_rejected_flag":    row.get("claim_rejected_flag"),
                "resubmission_flag":      row.get("resubmission_flag"),
                "original_claim_id":      row.get("original_claim_id"),
            } if row.get("treatment_type") else {},
            "user": {
                "first_name":              row.get("first_name"),
                "last_name":               row.get("last_name"),
                "email":                   row.get("email"),
                "age_group":               row.get("age_group"),
                "region":                  row.get("region"),
                "plan_type":               row.get("plan_type"),
                "membership_tenure_years": row.get("membership_tenure_years"),
                "past_escalation_count":   row.get("past_escalation_count"),
                "behavior_archetype":      row.get("behavior_archetype"),
            } if row.get("first_name") else {},
            "app_logs": [],
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


# ---------------------------------------------------------------------------
# Reports (completed agent runs with full context)
# ---------------------------------------------------------------------------

def db_get_reports() -> list:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    ar.run_id, ar.scenario_id, ar.member_id, ar.claim_id,
                    ar.risk_score, ar.risk_band, ar.status,
                    ar.started_at, ar.ended_at,
                    EXTRACT(EPOCH FROM (ar.ended_at - ar.started_at))::int AS duration_sec,
                    u.first_name, u.last_name, u.plan_type, u.region,
                    c.treatment_type, c.claim_amount,
                    i.actions_taken, i.reasoning AS agent_reasoning, i.actions_count,
                    (SELECT COUNT(*) FROM aa_customer_emails      e WHERE e.run_id = ar.run_id) AS emails_sent,
                    (SELECT COUNT(*) FROM aa_customer_notifications n WHERE n.run_id = ar.run_id) AS push_sent,
                    (SELECT COUNT(*) FROM aa_scheduled_callbacks   cb WHERE cb.run_id = ar.run_id) AS callbacks,
                    (SELECT COUNT(*) FROM aa_employee_alerts       ea WHERE ea.run_id = ar.run_id) AS alerts
                FROM aa_agent_runs ar
                LEFT JOIN users_backup u ON u.user_id = ar.member_id
                LEFT JOIN claims       c ON c.claim_id = ar.claim_id
                LEFT JOIN aa_interventions i ON i.run_id = ar.run_id
                WHERE ar.status IN ('complete', 'max_steps')
                ORDER BY ar.started_at DESC
            """)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


def db_get_report_detail(run_id: int) -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            # Final AI summary = last reasoning step text
            cur.execute("""
                SELECT reasoning_text FROM aa_agent_reasoning_steps
                WHERE run_id = %s ORDER BY step_number DESC LIMIT 1
            """, (run_id,))
            last_row = cur.fetchone()

            cur.execute("SELECT COUNT(*) AS cnt FROM aa_agent_tool_calls       WHERE run_id = %s", (run_id,))
            tool_count = cur.fetchone()["cnt"]

            cur.execute("SELECT COUNT(*) AS cnt FROM aa_agent_reasoning_steps WHERE run_id = %s", (run_id,))
            reasoning_count = cur.fetchone()["cnt"]

            cur.execute("SELECT subject, body_html, created_at FROM aa_customer_emails WHERE run_id = %s ORDER BY created_at", (run_id,))
            emails = [dict(r) for r in cur.fetchall()]

            cur.execute("SELECT title, body, created_at FROM aa_customer_notifications WHERE run_id = %s ORDER BY created_at", (run_id,))
            notifications = [dict(r) for r in cur.fetchall()]

            cur.execute("SELECT callback_id, priority, notes, scheduled_for FROM aa_scheduled_callbacks WHERE run_id = %s", (run_id,))
            callbacks = [dict(r) for r in cur.fetchall()]

            cur.execute("SELECT message, urgency, sla_minutes, created_at FROM aa_employee_alerts WHERE run_id = %s ORDER BY created_at", (run_id,))
            alerts = [dict(r) for r in cur.fetchall()]

    return {
        "ai_summary":           last_row["reasoning_text"] if last_row else None,
        "tool_call_count":      tool_count,
        "reasoning_step_count": reasoning_count,
        "emails":               emails,
        "notifications":        notifications,
        "callbacks":            callbacks,
        "alerts":               alerts,
    }


# ---------------------------------------------------------------------------
# Human-in-the-loop tables setup
# ---------------------------------------------------------------------------

def db_setup_hitl_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS aa_employee_questions (
                    question_id  SERIAL PRIMARY KEY,
                    run_id       INTEGER,
                    scenario_id  VARCHAR,
                    claim_id     VARCHAR,
                    question     TEXT NOT NULL,
                    context      TEXT,
                    tool_call_id VARCHAR NOT NULL,
                    response     TEXT,
                    status       VARCHAR DEFAULT 'pending',
                    created_at   TIMESTAMPTZ DEFAULT NOW(),
                    responded_at TIMESTAMPTZ
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS aa_agent_paused_state (
                    run_id          INTEGER PRIMARY KEY,
                    scenario_id     VARCHAR,
                    message_history JSONB NOT NULL,
                    step            INTEGER DEFAULT 0,
                    created_at      TIMESTAMPTZ DEFAULT NOW()
                )
            """)
        conn.commit()


# ---------------------------------------------------------------------------
# aa_employee_questions
# ---------------------------------------------------------------------------

def db_log_employee_question(run_id: int, scenario_id: str, claim_id: str, question: str, context: str, tool_call_id: str) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO aa_employee_questions (run_id, scenario_id, claim_id, question, context, tool_call_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING question_id
            """, (run_id, scenario_id, claim_id, question, context, tool_call_id))
            qid = cur.fetchone()["question_id"]
        conn.commit()
    return qid


def db_respond_to_question(question_id: int, response: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE aa_employee_questions
                SET response = %s, status = 'responded', responded_at = NOW()
                WHERE question_id = %s
            """, (response, question_id))
        conn.commit()


def db_get_question(question_id: int) -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM aa_employee_questions WHERE question_id = %s", (question_id,))
            row = cur.fetchone()
    return dict(row) if row else {}


def db_get_pending_questions() -> list:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT q.*,
                       u.first_name, u.last_name,
                       c.treatment_type, c.claim_amount,
                       ar.risk_band
                FROM aa_employee_questions q
                LEFT JOIN aa_agent_runs ar ON ar.run_id = q.run_id
                LEFT JOIN users_backup u ON u.user_id = ar.member_id
                LEFT JOIN claims c ON c.claim_id = q.claim_id
                WHERE q.status = 'pending'
                ORDER BY q.created_at DESC
            """)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# aa_agent_paused_state
# ---------------------------------------------------------------------------

def db_save_paused_state(run_id: int, scenario_id: str, messages: list, step: int):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO aa_agent_paused_state (run_id, scenario_id, message_history, step)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (run_id) DO UPDATE
                SET message_history = EXCLUDED.message_history, step = EXCLUDED.step
            """, (run_id, scenario_id, json.dumps(messages, default=str), step))
        conn.commit()


def db_get_paused_state(run_id: int) -> dict:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM aa_agent_paused_state WHERE run_id = %s", (run_id,))
            row = cur.fetchone()
    if not row:
        return {}
    d = dict(row)
    if isinstance(d["message_history"], str):
        d["message_history"] = json.loads(d["message_history"])
    return d


def db_get_all_alerts() -> list:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    ea.run_id, ea.claim_id, ea.message, ea.urgency, ea.sla_minutes, ea.created_at,
                    u.first_name, u.last_name,
                    c.treatment_type, c.claim_amount,
                    ar.risk_band, ar.risk_score
                FROM aa_employee_alerts ea
                LEFT JOIN aa_agent_runs ar ON ar.run_id = ea.run_id
                LEFT JOIN users_backup u ON u.user_id = ar.member_id
                LEFT JOIN claims c ON c.claim_id = ea.claim_id
                ORDER BY ea.created_at DESC
            """)
            rows = cur.fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    user_id = "U00001"
    claim_id = "CLM000001"
    details = db_get_app_logs(user_id, claim_id)
    print(json.dumps(details, indent=4))

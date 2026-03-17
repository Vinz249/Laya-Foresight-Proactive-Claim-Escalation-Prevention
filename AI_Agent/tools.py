
import json
from datetime import datetime

from database import (
    db_get_claim_details, db_get_user_details, db_get_app_logs,
    db_log_employee_alert, db_log_customer_email, db_log_customer_notification,
    db_log_scheduled_callback, db_log_intervention
)

# Set at the start of each agent run by agent.py
_current_run_id = None


def set_current_run_id(run_id: int):
    global _current_run_id
    _current_run_id = run_id


def get_claim_details(claim_id: str) -> dict:
    return db_get_claim_details(claim_id)


def get_customer_behaviour(user_id: str, claim_id: str) -> dict:
    return db_get_app_logs(user_id, claim_id)


def get_customer_history(user_id: str) -> dict:
    return db_get_user_details(user_id)


def alert_employee(claim_id: str, message: str, urgency: str, sla_minutes: int = None) -> dict:
    sla = sla_minutes or (30 if urgency == "URGENT" else 120)
    db_log_employee_alert(_current_run_id, claim_id, message, urgency, sla)
    return {
        "sent": True,
        "channel": "#claims-alerts",
        "urgency": urgency,
        "sla_minutes": sla,
        "timestamp": datetime.now().isoformat(),
        "message_preview": message[:100] + "..." if len(message) > 100 else message
    }


def send_email(user_id: str, subject: str, body_html: str) -> dict:
    db_log_customer_email(_current_run_id, user_id, subject, body_html)
    return {
        "sent": True,
        "recipient_id": user_id,
        "subject": subject,
        "timestamp": datetime.now().isoformat()
    }


def send_in_app_notification(user_id: str, title: str, body: str, deep_link: str = None) -> dict:
    link = deep_link or "laya://claims"
    db_log_customer_notification(_current_run_id, user_id, title, body, link)
    return {
        "sent": True,
        "recipient_id": user_id,
        "title": title,
        "deep_link": link,
        "timestamp": datetime.now().isoformat()
    }


def schedule_callback(user_id: str, claim_id: str, priority: str = "NORMAL", notes: str = "") -> dict:
    callback_id = f"CB-{datetime.now().strftime('%Y%m%d%H%M')}"
    assigned_to = "Claims Team Lead — Aoife Brennan" if priority == "HIGH" else "Available Claims Agent"
    scheduled_for = "Next available slot (within 2 hours)" if priority == "HIGH" else "Next available slot (within 24 hours)"
    db_log_scheduled_callback(_current_run_id, callback_id, user_id, claim_id, priority, notes, assigned_to, scheduled_for)
    return {
        "scheduled": True,
        "callback_id": callback_id,
        "priority": priority,
        "scheduled_for": scheduled_for,
        "assigned_to": assigned_to,
        "timestamp": datetime.now().isoformat()
    }


def log_intervention(claim_id: str, actions_taken: list, reasoning: str) -> dict:
    intervention_id = f"INT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    db_log_intervention(_current_run_id, intervention_id, claim_id, actions_taken, reasoning)
    return {
        "logged": True,
        "intervention_id": intervention_id,
        "claim_id": claim_id,
        "actions_count": len(actions_taken),
        "timestamp": datetime.now().isoformat()
    }

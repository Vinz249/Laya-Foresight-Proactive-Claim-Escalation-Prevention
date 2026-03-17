

import json
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
from tools import (
    get_claim_details, get_customer_behaviour, get_customer_history,
    alert_employee, send_email, send_in_app_notification,
    schedule_callback, log_intervention, set_current_run_id
)
from database import (
    db_create_agent_run, db_update_agent_run,
    db_log_reasoning_step, db_log_tool_call, db_log_agent_error
)

load_dotenv()


client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.getenv("GITHUB_TOKEN"),
)

MODEL = os.getenv("MODEL", "gpt-4o-mini")


MIN_DELAY_BETWEEN_CALLS = 5  
_last_api_call_time = 0


def _rate_limit_wait():
    global _last_api_call_time
    now = time.time()
    elapsed = now - _last_api_call_time
    if elapsed < MIN_DELAY_BETWEEN_CALLS and _last_api_call_time > 0:
        wait_time = MIN_DELAY_BETWEEN_CALLS - elapsed
        time.sleep(wait_time)
    _last_api_call_time = time.time()


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_claim_details",
            "description": "Retrieves full details of a claim: type, amount, status, missing docs, estimated completion.",
            "parameters": {
                "type": "object",
                "properties": {"claim_id": {"type": "string", "description": "The claim ID"}},
                "required": ["claim_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_behaviour",
            "description": "Retrieves real-time app activity: status check counts (1h/6h/24h), acceleration, last app open time.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "claim_id": {"type": "string"}
                },
                "required": ["user_id", "claim_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_history",
            "description": "Returns customer history: tenure, past claims, escalations, segment.",
            "parameters": {
                "type": "object",
                "properties": {"user_id": {"type": "string"}},
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "alert_employee",
            "description": "Sends Slack alert to claims team. You write the message. Use for cases needing human attention.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "message": {"type": "string", "description": "Concise brief on why this needs attention."},
                    "urgency": {"type": "string", "enum": ["URGENT", "ELEVATED", "WATCH"]},
                    "sla_minutes": {"type": "integer"}
                },
                "required": ["claim_id", "message", "urgency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Sends email to customer. Use when NOT currently in app or for claims >€5000.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "subject": {"type": "string"},
                    "body_html": {"type": "string", "description": "Warm, specific HTML email body."}
                },
                "required": ["user_id", "subject", "body_html"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_in_app_notification",
            "description": "Sends push notification. Use when customer is currently active in app (<30 mins).",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "title": {"type": "string"},
                    "body": {"type": "string"},
                    "deep_link": {"type": "string"}
                },
                "required": ["user_id", "title", "body"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_callback",
            "description": "Schedules outbound call. Use for high-value (>€5k), sensitive categories, or when messaging isn't enough.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "claim_id": {"type": "string"},
                    "priority": {"type": "string", "enum": ["HIGH", "NORMAL"]},
                    "notes": {"type": "string"}
                },
                "required": ["user_id", "claim_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "log_intervention",
            "description": "Records actions taken and reasoning. Always call as final action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "actions_taken": {"type": "array", "items": {"type": "string"}},
                    "reasoning": {"type": "string"}
                },
                "required": ["claim_id", "actions_taken", "reasoning"]
            }
        }
    }
]

TOOL_FUNCTIONS = {
    "get_claim_details": get_claim_details,
    "get_customer_behaviour": get_customer_behaviour,
    "get_customer_history": get_customer_history,
    "alert_employee": alert_employee,
    "send_email": send_email,
    "send_in_app_notification": send_in_app_notification,
    "schedule_callback": schedule_callback,
    "log_intervention": log_intervention,
}

SYSTEM_PROMPT = """You are LayaAIAgent for Laya Healthcare Ireland. When a customer is predicted likely to call support about their insurance claim, you assess and act proactively.

Tools available: retrieve claim details, check customer behaviour, get customer history, alert employees, send email/push, schedule callbacks, log actions.

Decision rules:
1. Always get claim details AND customer behaviour first.
2. Customer in app (<30 min) → prefer in-app notification over email.
3. Claims >€5,000 or sensitive medical → schedule callback.
4. Only message if you have genuinely useful info. If status is just "processing" with no update, do NOT message.
5. URGENT cases → alert employee BEFORE messaging customer.
6. Write all messages yourself — specific, empathetic, personalised.
7. Always call log_intervention as your final action.

Be direct, compassionate, professional. The customer is anxious about their healthcare claim.

IMPORTANT: Think step by step. Before each action, explain your reasoning clearly."""


def run_agent_streaming(scenario: dict):

    claim_id = scenario["claim_id"]
    user_id = scenario["user_id"]
    risk_score = scenario["risk_score"]
    risk_band = scenario["risk_band"]

    # Create agent run record in DB
    run_id = db_create_agent_run(
        scenario_id=scenario["id"],
        member_id=user_id,
        claim_id=claim_id,
        risk_score=risk_score,
        risk_band=risk_band,
        model_name=MODEL
    )
    set_current_run_id(run_id)

    yield {
        "type": "status",
        "data": {
            "message": f"🤖 Agent starting for {claim_id}...",
            "scenario_id": scenario["id"],
            "risk_score": risk_score,
            "risk_band": risk_band
        }
    }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"""Customer flagged as high risk of calling support.

Claim ID: {claim_id}
Customer ID: {user_id}
Risk Score: {risk_score:.2f} ({risk_band})

Assess the situation using your tools and take appropriate action. Think step by step and explain your reasoning."""
        }
    ]

    step = 0

    while True:
        step += 1

        yield {
            "type": "api_call",
            "data": {"message": f"⏳ Calling AI model (step {step})... Rate limit: waiting if needed"}
        }

        _rate_limit_wait()

        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                max_tokens=1024,
                temperature=0.3,
            )
        except Exception as e:
            error_msg = str(e)
            db_log_agent_error(run_id, error_msg[:500])
            if "rate" in error_msg.lower() or "429" in error_msg:
                yield {
                    "type": "error",
                    "data": {"message": f"⚠️ Rate limited! Waiting 30s... ({error_msg[:100]})"}
                }
                time.sleep(30)
                continue
            else:
                yield {
                    "type": "error",
                    "data": {"message": f"❌ API Error: {error_msg[:200]}"}
                }
                db_update_agent_run(run_id, "error")
                break

        choice = response.choices[0]
        message = choice.message

        # Show reasoning text
        if message.content:
            db_log_reasoning_step(run_id, step, message.content)
            yield {
                "type": "reasoning",
                "data": {
                    "step": step,
                    "text": message.content
                }
            }

        # Check if model is done (no tool calls)
        if choice.finish_reason == "stop" or not message.tool_calls:
            db_update_agent_run(run_id, "complete")
            yield {
                "type": "complete",
                "data": {"message": "✅ Agent task complete.", "step": step}
            }
            break

        # Process tool calls
        # Add assistant message to history
        messages.append(message)

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            yield {
                "type": "tool_call",
                "data": {
                    "step": step,
                    "tool_name": fn_name,
                    "tool_input": fn_args,
                    "tool_id": tool_call.id
                }
            }

            # Execute the tool
            fn = TOOL_FUNCTIONS.get(fn_name)
            if fn:
                result = fn(**fn_args)
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            db_log_tool_call(run_id, step, fn_name, fn_args, result)

            yield {
                "type": "tool_result",
                "data": {
                    "step": step,
                    "tool_name": fn_name,
                    "result": result
                }
            }

            # Add tool result to messages
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })

        # Safety: max 6 loops
        if step >= 6:
            db_update_agent_run(run_id, "max_steps")
            yield {
                "type": "complete",
                "data": {"message": "⚠️ Agent reached max steps (6). Ending.", "step": step}
            }
            break

    yield {
        "type": "done",
        "data": {"scenario_id": scenario["id"]}
    }
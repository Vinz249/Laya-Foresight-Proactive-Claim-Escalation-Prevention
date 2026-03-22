

import json
import time
import os
from dotenv import load_dotenv
from openai import OpenAI
from tools import (
    get_claim_details, get_customer_behaviour, get_customer_history,
    alert_employee, send_email, send_in_app_notification,
    schedule_callback, log_intervention, request_employee_input,
    set_current_run_id, set_current_scenario_id
)
from database import (
    db_create_agent_run, db_update_agent_run,
    db_log_reasoning_step, db_log_tool_call, db_log_agent_error,
    db_save_paused_state, db_get_paused_state, db_get_question
)

load_dotenv()


client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.getenv("GITHUB_TOKEN"),
)

MODEL = os.getenv("MODEL", "gpt-4o-mini")


def _msg_to_dict(message) -> dict:
    """Convert OpenAI ChatCompletionMessage object to a plain dict safe for JSON serialization."""
    d = {"role": message.role, "content": message.content}
    if message.tool_calls:
        d["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            }
            for tc in message.tool_calls
        ]
    return d


MIN_DELAY_BETWEEN_CALLS = 1
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
                    "body_html": {"type": "string", "description": "Warm, specific HTML email body."},
                    "to_email": {"type": "string", "description": "The recipient's email address."}
                },
                "required": ["user_id", "subject", "body_html", "to_email"]
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
            "name": "request_employee_input",
            "description": "Use when you need information from an employee before you can act. The agent will PAUSE and wait for the employee to respond before continuing. Use for: claims under review with no recent update, unusual patterns you cannot explain, cases where employee context would change your decision.",
            "parameters": {
                "type": "object",
                "properties": {
                    "claim_id": {"type": "string"},
                    "question": {"type": "string", "description": "The specific question you need answered. Be concise and direct."},
                    "context": {"type": "string", "description": "Brief explanation of why you're uncertain and what you've found so far."}
                },
                "required": ["claim_id", "question", "context"]
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
    "request_employee_input": request_employee_input,
}

SYSTEM_PROMPT = """You are LayaAIAgent, an autonomous claims intervention agent for Laya Healthcare Ireland.

Your job: when an ML model flags a customer as high risk of calling support, you investigate the claim, assess the situation, and take the right proactive action — before they have to pick up the phone.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY WORKFLOW — follow this order every time
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

STEP 1 — GATHER (always run both in sequence)
  → get_claim_details(claim_id)
  → get_customer_behaviour(user_id, claim_id)
  → Optionally: get_customer_history(user_id) if tenure, segment, or past escalations are relevant

STEP 2 — ASSESS (reason silently before acting)
  Ask yourself:
  • What is the claim status? Is anything blocked or missing?
  • How many times has the customer checked their status? Are they accelerating?
  • Is the customer currently in the app?
  • How long has this been open? Is there a clear reason for the delay?
  • What is the claim value and category?

STEP 3 — DECIDE (pick one of three paths)

  IMPORTANT: Always communicate with the customer. Even if the claim is simply "processing"
  with no specific update, the customer is anxious and checking their app repeatedly — that
  is exactly why they were flagged. A warm, reassuring message beats a phone call every time.

  PATH A — COMMUNICATE (default for most cases)
    Use when: claim is processing normally, no major flags, no escalation needed.
    • Customer active in app (<30 min since last open) → send_in_app_notification
    • Customer not in app, or claim ≤€5,000 → send_email
    • Even if there is no new update, send a reassuring message: acknowledge their claim,
      confirm it is being processed, give a realistic next step or expected timeframe if available.
    • Personalise every message — reference their specific claim type and situation.

  PATH B — ESCALATE + COMMUNICATE
    Use when: HIGH/CRITICAL risk, claim >€5,000, sensitive medical category, or customer showing
    distress signals (status checks accelerating rapidly, multiple prior escalations).
    Order:
      1. alert_employee(urgency=URGENT or ELEVATED)  ← always first
      2. schedule_callback if claim is high value or medically sensitive
      3. send_email or send_in_app_notification to acknowledge and reassure the customer

  PATH C — PAUSE FOR EMPLOYEE INPUT
    Use when: something is genuinely unclear and the employee's answer would change your decision.
    Examples:
      • Claim has been "under review" >30 days with no documented reason
      • Missing documents flagged but no follow-up on file
      • Unusual pattern (e.g. customer has 5 prior escalations but claim looks routine)
      • You are uncertain whether the claim is sensitive or routine
    → request_employee_input(claim_id, question, context)
    The agent will PAUSE here. You will resume automatically once the employee replies.
    On resume: use the employee's answer to complete your assessment and then communicate.

STEP 4 — LOG (always last)
  → log_intervention(claim_id, actions_taken=[ list every tool you called ], reasoning="...")
  Be specific in reasoning: explain what you found, what you decided, and why.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WRITING GUIDELINES (for all customer messages)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Address the customer by name if available.
• Reference their specific claim type and situation — never generic.
• Warm and human, not corporate. This person is anxious about their health.
• One clear action or update per message. Don't overload.
• Subject lines (email): direct and reassuring, e.g. "Update on your surgical claim" not "Important information".

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GUARDRAILS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Never make promises about outcome or timeline unless the data confirms it.
• If there is no specific update, still message — but be honest: reassure, don't invent progress.
• Never skip log_intervention — it is always your final action.
• If you called request_employee_input, do NOT call log_intervention before pausing. Log after resuming."""


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
    set_current_scenario_id(scenario["id"])

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
            "content": f"""INTERVENTION TRIGGERED

The ML model has flagged this customer as high risk of contacting support.

  Claim ID   : {claim_id}
  Customer ID: {user_id}
  Risk Score : {risk_score:.2f} / 1.00
  Risk Band  : {risk_band}

Begin your investigation now. Follow the mandatory workflow: gather data first, then assess, then decide on the right path. Explain your reasoning at each step before calling a tool."""
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
        messages.append(_msg_to_dict(message))

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
                # Inject tool_call_id for request_employee_input before calling
                if fn_name == "request_employee_input":
                    import tools as _tools_module
                    _tools_module._PENDING_TOOL_CALL_ID = tool_call.id
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

            # ── PAUSE: agent asked for employee input ──
            if fn_name == "request_employee_input" and result.get("status") == "awaiting_response":
                # Save full message history so we can resume later
                # Include the assistant message (already appended above) but NOT the tool result yet
                # The tool result will be injected with the employee's response on resume
                db_save_paused_state(run_id, scenario["id"], messages, step)
                db_update_agent_run(run_id, "paused")
                yield {
                    "type": "waiting_for_input",
                    "data": {
                        "question_id": result["question_id"],
                        "question": result["question"],
                        "claim_id": fn_args.get("claim_id"),
                        "tool_call_id": tool_call.id,
                        "message": "⏸ Agent paused — waiting for employee response"
                    }
                }
                yield {"type": "done", "data": {"scenario_id": scenario["id"]}}
                return

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


def resume_agent_streaming(question_id: int):
    """Resume a paused agent run after an employee has responded."""
    question = db_get_question(question_id)
    if not question:
        yield {"type": "error", "data": {"message": "Question not found"}}
        return
    if question.get("status") != "responded" or not question.get("response"):
        yield {"type": "error", "data": {"message": "No employee response found yet"}}
        return

    run_id       = question["run_id"]
    scenario_id  = question["scenario_id"]
    tool_call_id = question["tool_call_id"]
    employee_response = question["response"]

    paused = db_get_paused_state(run_id)
    if not paused:
        yield {"type": "error", "data": {"message": "No paused state found for this run"}}
        return

    messages = paused["message_history"]
    step = paused.get("step", 0)

    set_current_run_id(run_id)
    set_current_scenario_id(scenario_id)

    # Inject the employee's response as the tool result for request_employee_input
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call_id,
        "content": json.dumps({
            "status": "response_received",
            "employee_response": employee_response,
            "message": "Employee has responded. Continue your assessment with this new information."
        })
    })

    db_update_agent_run(run_id, "running")

    yield {
        "type": "status",
        "data": {"message": f"▶ Agent resuming with employee response for run #{run_id}"}
    }

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
                yield {"type": "error", "data": {"message": f"⚠️ Rate limited! Waiting 30s..."}}
                time.sleep(30)
                continue
            else:
                yield {"type": "error", "data": {"message": f"❌ API Error: {error_msg[:200]}"}}
                db_update_agent_run(run_id, "error")
                break

        choice  = response.choices[0]
        message = choice.message

        if message.content:
            db_log_reasoning_step(run_id, step, message.content)
            yield {"type": "reasoning", "data": {"step": step, "text": message.content}}

        if choice.finish_reason == "stop" or not message.tool_calls:
            db_update_agent_run(run_id, "complete")
            yield {"type": "complete", "data": {"message": "✅ Agent task complete (resumed).", "step": step}}
            break

        messages.append(_msg_to_dict(message))

        for tool_call in message.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            yield {"type": "tool_call", "data": {"step": step, "tool_name": fn_name, "tool_input": fn_args, "tool_id": tool_call.id}}

            fn = TOOL_FUNCTIONS.get(fn_name)
            result = fn(**fn_args) if fn else {"error": f"Unknown tool: {fn_name}"}

            db_log_tool_call(run_id, step, fn_name, fn_args, result)
            yield {"type": "tool_result", "data": {"step": step, "tool_name": fn_name, "result": result}}

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)})

        if step >= 10:
            db_update_agent_run(run_id, "max_steps")
            yield {"type": "complete", "data": {"message": "⚠️ Agent reached max steps. Ending.", "step": step}}
            break

    yield {"type": "done", "data": {"scenario_id": scenario_id}}
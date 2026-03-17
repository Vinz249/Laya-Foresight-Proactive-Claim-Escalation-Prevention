-- ============================================================
-- LayaForesight Agent Database Schema
-- ============================================================


-- ML prediction ingested from the model
CREATE TABLE aa_ml_predictions (
    prediction_id       SERIAL PRIMARY KEY,
    member_id           VARCHAR NOT NULL,
    claim_id            VARCHAR,
    predicted_risk      INTEGER NOT NULL,
    risk_probability    FLOAT NOT NULL,
    risk_band           VARCHAR NOT NULL,
    ingested_at         TIMESTAMP DEFAULT NOW(),
    created_at          TIMESTAMP DEFAULT NOW()
);


-- One row per agent execution
CREATE TABLE aa_agent_runs (
    run_id              SERIAL PRIMARY KEY,
    scenario_id         VARCHAR NOT NULL,
    member_id           VARCHAR,
    claim_id            VARCHAR,
    risk_score          FLOAT,
    risk_band           VARCHAR,
    status              VARCHAR DEFAULT 'running',   -- running | complete | error | max_steps
    started_at          TIMESTAMP DEFAULT NOW(),
    ended_at            TIMESTAMP,
    model_name          VARCHAR,
    created_at          TIMESTAMP DEFAULT NOW()
);


-- Agent reasoning/thinking steps
CREATE TABLE aa_agent_reasoning_steps (
    step_id             SERIAL PRIMARY KEY,
    run_id              INTEGER REFERENCES aa_agent_runs(run_id),
    step_number         INTEGER NOT NULL,
    reasoning_text      TEXT NOT NULL,
    created_at          TIMESTAMP DEFAULT NOW()
);


-- Every tool call and its result
CREATE TABLE aa_agent_tool_calls (
    tool_call_id        SERIAL PRIMARY KEY,
    run_id              INTEGER REFERENCES aa_agent_runs(run_id),
    step_number         INTEGER NOT NULL,
    tool_name           VARCHAR NOT NULL,
    tool_input          JSONB,
    tool_result         JSONB,
    created_at          TIMESTAMP DEFAULT NOW()
);


-- Slack alerts sent to claims team
CREATE TABLE aa_employee_alerts (
    alert_id            SERIAL PRIMARY KEY,
    run_id              INTEGER REFERENCES aa_agent_runs(run_id),
    claim_id            VARCHAR NOT NULL,
    message             TEXT,
    urgency             VARCHAR NOT NULL,            -- URGENT | ELEVATED | WATCH
    sla_minutes         INTEGER,
    created_at          TIMESTAMP DEFAULT NOW()
);


-- Emails sent to customers
CREATE TABLE aa_customer_emails (
    email_id            SERIAL PRIMARY KEY,
    run_id              INTEGER REFERENCES aa_agent_runs(run_id),
    user_id             VARCHAR NOT NULL,
    subject             VARCHAR NOT NULL,
    body_html           TEXT,
    created_at          TIMESTAMP DEFAULT NOW()
);


-- In-app push notifications sent to customers
CREATE TABLE aa_customer_notifications (
    notification_id     SERIAL PRIMARY KEY,
    run_id              INTEGER REFERENCES aa_agent_runs(run_id),
    user_id             VARCHAR NOT NULL,
    title               VARCHAR NOT NULL,
    body                TEXT,
    deep_link           VARCHAR,
    created_at          TIMESTAMP DEFAULT NOW()
);


-- Callbacks scheduled for claims team
CREATE TABLE aa_scheduled_callbacks (
    callback_id         VARCHAR PRIMARY KEY,        -- CB-YYYYMMDDHHmm
    run_id              INTEGER REFERENCES aa_agent_runs(run_id),
    user_id             VARCHAR NOT NULL,
    claim_id            VARCHAR,
    priority            VARCHAR NOT NULL,            -- HIGH | NORMAL
    notes               TEXT,
    assigned_to         VARCHAR,
    scheduled_for       VARCHAR,
    created_at          TIMESTAMP DEFAULT NOW()
);


-- Final intervention log per agent run
CREATE TABLE aa_interventions (
    intervention_id     VARCHAR PRIMARY KEY,        -- INT-YYYYMMDDHHmmss
    run_id              INTEGER REFERENCES aa_agent_runs(run_id),
    claim_id            VARCHAR NOT NULL,
    actions_taken       JSONB,
    reasoning           TEXT,
    actions_count       INTEGER,
    created_at          TIMESTAMP DEFAULT NOW()
);


-- Agent errors for debugging
CREATE TABLE aa_agent_errors (
    error_id            SERIAL PRIMARY KEY,
    run_id              INTEGER REFERENCES aa_agent_runs(run_id),
    error_message       TEXT,
    created_at          TIMESTAMP DEFAULT NOW()
);

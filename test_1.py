import os
import re
import json
import streamlit as st
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableBranch

# -----------------------------------
# 1️⃣ Environment Setup
# -----------------------------------
load_dotenv()

st.set_page_config(page_title="Right2Bill AI Assistant", page_icon="💬", layout="centered")

# -----------------------------------
# 2️⃣ LLM Models
# -----------------------------------
CUSTOMER_LLM = ChatOpenAI(
    model="mistralai/mistral-small-3.1-24b-instruct:free",
    temperature=0.3,
    base_url="https://openrouter.ai/api/v1"
)

MERCHANT_LLM = ChatOpenAI(
    model="openai/gpt-oss-20b:free",
    temperature=0.4,
    base_url="https://openrouter.ai/api/v1"
)

AUTHORITY_LLM = ChatOpenAI(
    model="tngtech/deepseek-r1t-chimera:free",
    temperature=0.2,
    base_url="https://openrouter.ai/api/v1"
)

# -----------------------------------
# 3️⃣ System Context
# -----------------------------------
SYSTEM_CONTEXT = """
You are **Hermit**, the AI compliance assistant for the **Right2Bill** platform.

Mission:
- Help customers, merchants, and tax authorities ensure every transaction is genuine, billed, and GST-compliant.
- Prevent tax leakage by guiding users on proper billing, rewards, and fraud reporting.

Regulatory Guardrails:
1. You are NOT a human advisor. Always say: “I am an AI assistant, not a tax or legal advisor.”
2. Never provide emotional support, personal opinions, or unrelated information.
3. Always warn: “Do not share personal or financial information (PAN, Aadhaar, bank details).”
4. If discussing fraud or non-compliance, maintain a formal and factual tone.
5. Encourage users to use **Right2Bill** tools (bill upload, fraud reporting, reward tracking).

You must operate strictly within this domain.
"""

# -----------------------------------
# 4️⃣ PII Redaction
# -----------------------------------
PII_PATTERNS = [
    r"\b\d{12}\b",
    r"\b\d{10}\b",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",
    r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
    r"(password|otp|pin)\s*[:=]\s*\S+",
]

def pii_redact(text: str) -> str:
    for pat in PII_PATTERNS:
        text = re.sub(pat, "[REDACTED]", text)
    return text

# -----------------------------------
# 5️⃣ Fraud Schema & Flow
# -----------------------------------
class FraudReportSchema(BaseModel):
    report_uuid: str = Field(default_factory=lambda: str(uuid4()))
    transaction_id: str = Field(..., pattern=r"^[A-Za-z0-9\-]{3,64}$")
    merchant_tax_id: str
    non_compliance_type: Literal["FakeBill", "NoBill", "IncorrectRate", "RewardFraud"]
    timestamp_of_event: datetime
    reported_amount_usd: float = Field(..., gt=0)
    evidence_summary: str

    @field_validator("evidence_summary")
    def no_pii(cls, v):
        if any(s in v for s in ["@", "http", "https", "password", "OTP"]):
            raise ValueError("Evidence summary must not include PII or links")
        return v

def submit_fraud_report(data: dict):
    safe_data = {k: (v.isoformat() if isinstance(v, datetime) else v) for k, v in data.items()}
    st.json(safe_data)
    st.success(f"✅ Report submitted successfully! Case ID: {safe_data['report_uuid']}")
    st.info("🎁 You've earned 10 XP for honest reporting!")

def collect_fraud_report_flow():
    st.warning("⚠️ Do not share personal identifiers (Aadhaar, PAN, bank details).")
    with st.form("fraud_report"):
        txn_id = st.text_input("🧾 Transaction ID or short description", "UNKNOWN")
        merchant_id = st.text_input("🏪 Merchant GST ID (if known)", "UNKNOWN")
        issue = st.selectbox("🚨 Issue Type", ["NoBill", "FakeBill", "IncorrectRate", "RewardFraud"])
        date_str = st.text_input("📅 Date of transaction (YYYY-MM-DD)", "")
        amount_inr = st.number_input("💰 Approx amount (₹)", min_value=0.0, value=0.0)
        desc = st.text_area("📝 Describe what happened")

        submitted = st.form_submit_button("Submit Report")
        if submitted:
            try:
                timestamp = datetime.fromisoformat(date_str) if date_str else datetime.now()
                desc_clean = pii_redact(desc)
                report = FraudReportSchema(
                    transaction_id=txn_id,
                    merchant_tax_id=merchant_id,
                    non_compliance_type=issue,
                    timestamp_of_event=timestamp,
                    reported_amount_usd=round(amount_inr / 83.0, 2),
                    evidence_summary=desc_clean
                )
                submit_fraud_report(report.model_dump())
            except Exception as e:
                st.error(f"❌ Error: {e}")

# -----------------------------------
# 6️⃣ Prompts for Each Agent
# -----------------------------------
customer_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_CONTEXT),
    MessagesPlaceholder(variable_name="history"),
    ("user", """
You are the **Customer Agent** of the Right2Bill compliance platform.

Your purpose:
- Help users verify bills, understand rewards, and report merchants who fail to issue bills.
- If a user reports missing or fake bills, ALWAYS mention:
  1. Under GST, bills are mandatory for purchases above ₹200.
  2. Not receiving a bill can be reported under the Right2Bill system.
  3. Offer to start a fraud report in short, clear steps.
- Keep replies under **90 words**.
- Avoid emotional tone, only procedural help.
- End every message with:
  “I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.”

User request: {query}
""")
])

merchant_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_CONTEXT),
    MessagesPlaceholder(variable_name="history"),
    ("user", """
You are the **Merchant Agent** for the Right2Bill compliance platform.

Goals:
- Explain GST compliance duties, merchant incentives, and Input Tax Credit (ITC) rules clearly.
- Mention that:
  - Bills are mandatory for sales over ₹200.
  - ITC can only be claimed with a valid GST invoice or debit note.
  - Right2Bill automates compliant billing, e-invoicing, and digital record-keeping.
- Use concise sentences (under **100 words** total).
- Include a short compliance table **only if** the query involves incentives or billing benefits.
- End with the standard disclaimer and PII warning.

User: {query}
""")
])

authority_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_CONTEXT),
    MessagesPlaceholder(variable_name="history"),
    ("user", """
You are the **Authority Agent** assisting tax officials via Right2Bill.

Guidelines:
- Summarize fraud reports and compliance trends objectively.
- Provide structured summaries (bullet points or JSON-like clarity).
- NEVER reveal or speculate about audit algorithms, selection criteria, or internal models.
- You may explain public policy metrics such as ETR (Effective Tax Rate), fraud incidence, and reward distribution.
- Keep responses formal, concise, and free of emotional tone.
- End with: “Information is for analytical support only, not an official audit finding.”

User: {query}
""")
])

fallback_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_CONTEXT),
    ("user", """
This query is **outside Right2Bill’s domain**.
Politely refuse and say:
“I’m a specialized compliance assistant for GST, tax transparency, and bill verification.
Please ask questions related to bills, merchants, or tax compliance.”

User: {query}
""")
])

# -----------------------------------
# 7️⃣ Model-Specific Chains
# -----------------------------------
customer_chain = customer_prompt | CUSTOMER_LLM
merchant_chain = merchant_prompt | MERCHANT_LLM
authority_chain = authority_prompt | AUTHORITY_LLM
fallback_chain = fallback_prompt | CUSTOMER_LLM

def route_query(inputs):
    q = inputs["query"].lower()
    if not any(k in q for k in ["bill", "gst", "reward", "merchant", "tax", "fake", "report", "invoice", "audit"]):
        return "off_topic"
    if any(k in q for k in ["fake", "report", "fraud", "no bill", "didn't give bill"]):
        return "customer"
    elif any(k in q for k in ["merchant", "seller", "incentive", "benefit"]):
        return "merchant"
    elif any(k in q for k in ["authority", "audit", "kpi"]):
        return "authority"
    else:
        return "customer"

router = RunnableBranch(
    (lambda x: route_query(x) == "customer", customer_chain),
    (lambda x: route_query(x) == "merchant", merchant_chain),
    (lambda x: route_query(x) == "authority", authority_chain),
    (lambda x: route_query(x) == "off_topic", fallback_chain),
    customer_chain
)

# -----------------------------------
# 8️⃣ Stateful Memory
# -----------------------------------
session_store = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

chat_with_memory = RunnableWithMessageHistory(
    router,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history"
)

# -----------------------------------
# 9️⃣ Streamlit Chat UI
# -----------------------------------

# -----------------------------------
# 9️⃣ Streamlit Chat UI (Final Polished Version)
# -----------------------------------

st.markdown("""
    <style>
    /* 🌈 Background */
    body {
        background-color: #0f172a;
        color: #fff;
    }
    /* Title */
    .title {
        text-align: center;
        color: #facc15;
        font-size: 2.3rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.4rem;
    }
    .subtitle {
        text-align: center;
        color: #a3e635;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }
    /* Chat container */
    .chat-container {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        padding: 20px;
        height: 75vh;
        overflow-y: auto;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 16px;
        box-shadow: 0 0 25px rgba(0, 0, 0, 0.3);
    }
    /* Chat messages */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 14px !important;
        margin-bottom: 10px !important;
    }
    .stChatMessage.user {
        background: #1e3a8a;
        color: #fff;
    }
    .stChatMessage.assistant {
        background: #14532d;
        color: #e7e5e4;
    }
    /* Chat input fixed at bottom */
    .stChatInput {
        position: fixed !important;
        bottom: 0 !important;
        left: 50%;
        transform: translateX(-50%);
        width: 60%;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 10px 16px !important;
        box-shadow: 0px 4px 18px rgba(0, 0, 0, 0.5);
        z-index: 100;
    }
    /* Hide scrollbar for clean UI */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-thumb {
        background: #374151;
        border-radius: 10px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #4b5563;
    }
    </style>
""", unsafe_allow_html=True)

# ✨ Title
st.markdown('<div class="title">💬 Right2Bill Compliance Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Assistant for GST Compliance, Merchant Verification & Fraud Reporting</div>', unsafe_allow_html=True)

session_id = "web_user"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

fraud_triggers = [
    "no bill", "did not give bill", "didn't give bill",
    "merchant refused bill", "fake bill", "fake invoice", "report merchant"
]

# --- Chat Display Area ---
chat_container = st.container()
with chat_container:
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(f"🧑‍💼 {chat['user']}")
        with st.chat_message("assistant"):
            st.markdown(f"🤖 {chat['bot']}")

# --- Input Box (Fixed at Bottom) ---
user_input = st.chat_input("Type your message... 💬")

if user_input:
    with st.chat_message("user"):
        st.markdown(f"🧑‍💼 **You:** {user_input}")

    # Fraud case logic
    if any(trigger in user_input.lower() for trigger in fraud_triggers):
        response = chat_with_memory.invoke(
            {"query": pii_redact(user_input)},
            config={"configurable": {"session_id": session_id}}
        )
        reply = response.content.strip() if not isinstance(response, dict) else json.dumps(response, indent=2)

        with st.chat_message("assistant"):
            st.markdown(f"🤖 **Hermit:** {reply}")
            st.info("Would you like to file a fraud report for this incident?")
            col1, col2 = st.columns(2)
            with col1:
                file_now = st.button("🚨 Yes, file complaint now")
            with col2:
                skip = st.button("❌ No, not right now")

            if file_now:
                st.markdown("### 📝 Fraud Report Form")
                collect_fraud_report_flow()
            elif skip:
                st.info("Okay — no report filed. You can always submit one later by typing **report**.")

    elif "report" in user_input.lower():
        st.markdown("### 🧾 File a Fraud Report")
        collect_fraud_report_flow()

    else:
        response = chat_with_memory.invoke(
            {"query": pii_redact(user_input)},
            config={"configurable": {"session_id": session_id}}
        )
        reply = response.content.strip() if not isinstance(response, dict) else json.dumps(response, indent=2)
        st.session_state.chat_history.append({"user": user_input, "bot": reply})

        with st.chat_message("assistant"):
            st.markdown(f"🤖 {reply}")

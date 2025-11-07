import os
import re
import json
import streamlit as st
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from flask import Flask, request, jsonify
from flask_cors import CORS
from threading import Thread

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
# 3️⃣ System Context (unchanged)
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
# 5️⃣ Fraud Schema & Flow (unchanged)
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
# 6️⃣ Prompts & Routing (unchanged)
# -----------------------------------
# (same customer_prompt, merchant_prompt, authority_prompt, fallback_prompt)
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


# -----------------------------------
# 7️⃣ Stateful Memory
# -----------------------------------
session_store = {}
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

customer_chat = RunnableWithMessageHistory(
    customer_chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history"
)
merchant_chat = RunnableWithMessageHistory(
    merchant_chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history"
)
authority_chat = RunnableWithMessageHistory(
    authority_chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="history"
)

router = RunnableBranch(
    (lambda x: route_query(x) == "customer", customer_chat),
    (lambda x: route_query(x) == "merchant", merchant_chat),
    (lambda x: route_query(x) == "authority", authority_chat),
    (lambda x: route_query(x) == "off_topic", fallback_chain),
    customer_chat
)


# -----------------------------------
# 🔥 NEW SECTION: Flask API (Fixed)
# -----------------------------------
flask_app = Flask(__name__)
CORS(flask_app)

@flask_app.route("/chat", methods=["POST"])
def chat_api():
    try:
        data = request.get_json()
        user_input = data.get("message", "")
        role = data.get("role", "Customer")  # Default role = Customer
        session_id = data.get("session_id", "api_user")

        clean_input = pii_redact(user_input)

        # ✅ Select correct memory-enabled model based on role
        selected_model = {
            "Customer": customer_chat,
            "Merchant": merchant_chat,
            "Authority": authority_chat
        }.get(role, customer_chat)

        response = selected_model.invoke(
            {"query": clean_input},
            config={"configurable": {"session_id": f"{session_id}_{role.lower()}"}}
        )

        reply = response.content.strip() if not isinstance(response, dict) else json.dumps(response, indent=2)

        return jsonify({
            "role": role,
            "session_id": f"{session_id}_{role.lower()}",
            "reply": reply
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def run_flask():
    flask_app.run(host="0.0.0.0", port=5000, debug=False)


# ✅ Start Flask in a background thread so Streamlit and Flask can run together
Thread(target=run_flask, daemon=True).start()

# -----------------------------------
# 8️⃣ Streamlit Chat UI (unchanged)
# -----------------------------------
# (your entire Streamlit section goes here as-is)
# The same chat logic, fraud form, styling, etc.
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

# -------------------------------
# CASUAL_RESPONSES (copy-paste)
# -------------------------------
CASUAL_RESPONSES = {
    "hi": "👋 Hello! I’m Hermit, the Right2Bill assistant. Ask me about XP, GST verification, or how to report a merchant. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "hello": "Hey! 👋 I’m Hermit. I can help you verify bills, track XP, and report merchants who don't issue bills. How can I help today? I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "hey": "Hi there! 😊 I’m Hermit — your Right2Bill compliance assistant. Ask me anything about bills, rewards, or reporting. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "good morning": "Good morning! 🌅 Ready to make billing transparent today? Ask me how to earn XP or report a missing bill. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "good afternoon": "Good afternoon! ☀️ Want to check your XP, verify a bill, or report a merchant? I can help. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "good evening": "Good evening! 🌇 I can help with reward queries, GST checks, or filing a report. What would you like to do? I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "who are you": "I’m Hermit — Right2Bill’s AI compliance assistant. I help users verify bills, understand rewards, and report non-compliant merchants. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "what can you do": "I explain GST verification, track XP and impact score, assist with reporting merchants, and guide users through Right2Bill features. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "who made you": "I was built by the Right2Bill hackathon team to demonstrate AI-powered billing transparency. Ask me about features or rewards! I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "thank you": "You’re welcome! 🙌 Glad to help. If you need anything else about bills, XP, or reporting—just ask. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "thanks": "Anytime! 😊 If you want to earn more XP, I can show ways to participate. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "bye": "Goodbye! 👋 Keep supporting transparent billing — and remember to ask for your bill. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "see you": "See you! 👋 Come back anytime to check XP, report merchants, or learn about GST. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "ok": "Got it — let me know if you want to check your XP, report a merchant, or verify a bill. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "how are you": "I’m ready to help! 🤖 Ask me anything about Right2Bill’s features or reporting flow. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "nice": "Thanks! 😊 If you’d like, I can show how to earn XP or how to report a merchant. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "cool": "Glad you think so! 😎 Tell me if you want to check your XP or report an issue. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "great": "Great! 👍 Ready to help with XP, GST verification, or reporting. What would you like to do next? I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "who built you": "A small team built me for the Right2Bill hackathon — to demonstrate automated bill verification and citizen reporting. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "help": "I can explain how Right2Bill works, how to earn XP, how to report merchants, and how GST verification is done. Ask me any specific question to get started. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details."
}

# -------------------------------
# FAQ_RESPONSES (copy-paste)
# -------------------------------
FAQ_RESPONSES = {
    # Basic product
    "what is right2bill": "Right2Bill is an AI-powered platform that connects customers, merchants, and tax authorities to ensure every purchase generates a genuine bill. It rewards honest behavior and enables reporting of non-compliant merchants. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "what does right2bill do": "Right2Bill automates bill collection from merchant POS, verifies invoices via GST checks (or simulated verification in demo), and rewards customers and merchants for compliant behavior. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "how does it work": "When a merchant issues a bill, Right2Bill receives it (via integration or manual input), verifies GST details, credits XP or updates merchant Trust Score, and notifies the customer. Users can also report missing bills. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",

    # XP & rewards
    "how to earn xp": "You earn XP by submitting or verifying genuine bills, reporting merchants who refuse to issue bills, referring friends, and participating in platform events. Higher XP unlocks vouchers and recognition. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "xp points": "XP are reward points for civic participation—valid bill verification, reporting non-compliance, referrals, and engagement. They improve your Impact Score and can be redeemed for vouchers. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "how to redeem xp": "Redeem rules vary by campaign; typically, accumulate required XP and choose a voucher from the Rewards section. For the demo, XP redemption is simulated. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "what rewards are available": "Rewards may include discount vouchers, partner offers, certificates, or leaderboard recognition. In the demo, rewards are simulated to show the flow. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "is xp transferable": "XP is typically non-transferable between users to prevent gaming. Specific transfer rules depend on platform policy. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",

    # Merchant side
    "merchant rewards": "Merchants earn Trust Score increases and rewards for issuing GST-verified bills consistently. High-scoring merchants gain visibility and potential incentives. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "how merchants benefit": "Honest merchants build reputation, get more customer trust, and can unlock platform incentives. The system also reduces fake-billing competition and supports compliant businesses. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "what is trust score": "Trust Score is a 0–100 metric reflecting a merchant’s billing compliance. It increases with verified bills and decreases with confirmed complaints. It helps customers choose trustworthy shops. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",

    # Reporting & fraud
    "how to report merchant": "Type 'report merchant' or use the fraud flow. Provide merchant details, approximate amount, and a short description. We will log the report and, if validated, adjust Trust Scores and reward honest reporters. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "what happens after report": "Reports are stored and reviewed. If multiple reports are received, the merchant may be flagged for review. Verified reports earn customers bonus XP and impact the merchant's Trust Score. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "what is fake bill": "A fake bill is an invoice that lacks valid GST data or IRN, or that is fabricated to claim tax credits/refunds. Right2Bill flags suspicious bills via GST checks and AI pattern analysis. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "what is no bill": "A 'NoBill' case occurs when a merchant refuses to provide a bill. Customers can report this; repeated incidents lower the merchant’s Trust Score. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",

    # GST & verification
    "what is gst verification": "GST verification checks the merchant's GSTIN and invoice details against government records (IRN/e-invoice) to validate authenticity. In demo mode, we simulate verification. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "how does gst verification work": "The system extracts GSTIN/invoice details or IRN from the bill (or merchant POS) and checks it against GST records. A match marks the invoice as verified. In production, this uses official GST or GSP APIs. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "what if gst not found": "If GST verification fails, the bill is flagged as 'suspicious' and may be reviewed. Customers may be asked for more details; multiple failures can trigger merchant scrutiny. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",

    # Privacy & security
    "is my data safe": "Right2Bill stores minimal transaction metadata and avoids sensitive PII in reports. Do not share PAN, Aadhaar, bank details, or OTPs. All data handling follows privacy-first principles (demo environment). I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "what personal info to provide": "Only provide non-sensitive details: merchant name, approximate amount, date, and short description. Never share PAN, Aadhaar, bank details or OTP. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",

    # Platform & onboarding
    "how to onboard merchant": "Merchants can sign up on the platform and connect their billing software or export invoices (CSV/JSON). For the demo we simulate this integration. Production integration may require API keys or a lightweight plugin. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "do merchants need an account": "Yes — merchants should register to authenticate uploads and receive a unique API key or integration token. This prevents unauthorized bill submissions. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "does customer need account": "Customers can optionally create an account to view history and redeem XP. If not registered, bills can be stored by phone or email and claimed later when they sign up. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",

    # Impact & policy
    "how does it prevent tax leakage": "By automatically capturing invoices and rewarding verification/reporting, Right2Bill increases the number of recorded transactions and reduces unbilled sales — improving tax compliance at scale. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "what is impact score": "Impact Score measures a user's contribution to transparency (verified bills + valid reports + referrals). Higher scores reflect greater civic contribution and unlock recognition. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "who uses this data": "Aggregated, anonymized insights can be shared with tax authorities for policy and enforcement. Personal data is not disclosed without legal process. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",

    # Demo specifics
    "is this a demo": "Yes — this hackathon version simulates some integrations (like GST checks and rewards) to demonstrate the flow. Production would use official APIs and hardened security. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "what is simulated": "In the demo we simulate merchant integrations and GST verification to show the automated flow. Core logic, UX, and reward model are implemented for demonstration. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",

    # Troubleshooting & next steps
    "i found an error": "Sorry to hear that — please describe the issue (what you clicked and what happened). We'll log it as feedback. Do not include sensitive data. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "can you show demo steps": "Sure — 1) Merchant issues a bill (simulated), 2) The bill syncs to Right2Bill, 3) System verifies via GST or simulated check, 4) Customer gets XP, 5) Reports adjust Trust Scores. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",

    # Edge cases & policy clarifications
    "what if merchant is honest but gst wrong": "If merchant provides incorrect GST details, the bill may fail verification. We prompt for clarification and allow merchants to correct records; repeated failures affect Trust Score. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "what if customer lies": "False or malicious reports are harmful. Our system uses verification and patterns; proven false reports may result in penalties (XP deduction) to discourage abuse. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "can this integrate with gstn": "Yes — production integration can use GSTN via authorized GSPs or the e-invoice API for IRN verification. For the demo, we use simulated verification. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details.",
    "how fast is verification": "Verification is near real-time if integrated with GST or merchant POS. In demo mode timing is simulated to show the end-to-end flow. I am an AI assistant, not a tax or legal advisor. Do not share personal or financial details."
}


if user_input:
    with st.chat_message("user"):
        st.markdown(f"🧑‍💼 **You:** {user_input}")

    clean_input = pii_redact(user_input.lower())

    # 1️⃣ Check for predefined FAQ or casual responses first
    matched_key = next((k for k in FAQ_RESPONSES if k in clean_input), None)
    casual_key = next((k for k in CASUAL_RESPONSES if k in clean_input), None)

    # if matched_key:
    #     reply = FAQ_RESPONSES[matched_key]
    # elif casual_key:
    #     reply = CASUAL_RESPONSES[casual_key]

    # elif any(trigger in clean_input for trigger in fraud_triggers):
    # # 🧾 Fraud-specific handling
    #     response = router.invoke(
    #         {"query": pii_redact(user_input)},
    #         config={"configurable": {"session_id": session_id}}
    #     )
    
    #     reply = response.content.strip() if not isinstance(response, dict) else json.dumps(response, indent=2)

    #     with st.chat_message("assistant"):
    #         st.markdown(f"🤖 **Hermit:** {reply}")
    #         st.info("Would you like to file a fraud report for this incident?")
    #         col1, col2 = st.columns(2)
    #         with col1:
    #             file_now = st.button("🚨 Yes, file complaint now")
    #         with col2:
    #             skip = st.button("❌ No, not right now")
    #         if file_now:
    #             st.markdown("### 📝 Fraud Report Form")
    #             collect_fraud_report_flow()
    #         elif skip:
    #             st.info("Okay — no report filed. You can always submit one later by typing **report**.")
    #     st.session_state.chat_history.append({"user": user_input, "bot": reply})
    
    if matched_key:
        reply = FAQ_RESPONSES[matched_key]
        # display + store
        with st.chat_message("assistant"):
            st.markdown(f"🤖 **Hermit:** {reply}")
        st.session_state.chat_history.append({"user": user_input, "bot": reply})

    elif casual_key:
        reply = CASUAL_RESPONSES[casual_key]
        # display + store
        with st.chat_message("assistant"):
            st.markdown(f"🤖 **Hermit:** {reply}")
        st.session_state.chat_history.append({"user": user_input, "bot": reply})
    
    elif any(trigger in clean_input for trigger in fraud_triggers):
        # 🧾 Fraud-specific handling (unchanged)
        response = router.invoke(
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
        st.session_state.chat_history.append({"user": user_input, "bot": reply})


    elif "report" in clean_input:
        st.markdown("### 🧾 File a Fraud Report")
        collect_fraud_report_flow()

    else:
        response = router.invoke(
            {"query": pii_redact(user_input)},
            config={"configurable": {"session_id": session_id}}
        )
    
        reply = response.content.strip() if not isinstance(response, dict) else json.dumps(response, indent=2)

        st.session_state.chat_history.append({"user": user_input, "bot": reply})
        with st.chat_message("assistant"):
            st.markdown(f"🤖 {reply}")

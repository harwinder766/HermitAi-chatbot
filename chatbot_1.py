# version 1

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
# import os
# from dotenv import load_dotenv
# load_dotenv()

# # os.environ["OPENAI_API_KEY"] = "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"
# # os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# llm = ChatOpenAI(
#     model = "mistralai/mistral-small-3.1-24b-instruct:free",
#     temperature= 0.3,
#     base_url="https://openrouter.ai/api/v1"
# )

# system_context = """
# You are Hermit, the smart AI assistant for the Right2Bill platform.
# The user will ask you questions about bills, GST, XP points, merchant rewards, and fake bill reporting. 
# Your answer should be friendly, short(under 80 words), easy to understand.

# """

# chat_history = [
#     SystemMessage(content = system_context)
# ]

# prompt = ChatPromptTemplate.from_messages([
#     ('system', system_context),
#     ('user', "{query}")
# ])

# chain = prompt | llm

# print("👋 Hermit is online! Type 'exit' to stop.\n")

# while True:
#     user_input = input("You : ")
#     if user_input == "exit":
#         break
#     chat_history.append(HumanMessage(content= user_input))
#     result = chain.invoke({"query": user_input})
#     chat_history.append(AIMessage(content = result.content))
#     print("Hermit: ",result.content)

# print(chat_history)

# version 2

# import os
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory

# # --- 🔐 Set API key ---
# import os
# from dotenv import load_dotenv
# load_dotenv()
# # --- 🧠 Initialize model ---
# llm = ChatOpenAI(
#     model="mistralai/mistral-small-3.1-24b-instruct:free",
#     temperature=0.3,
#     base_url="https://openrouter.ai/api/v1"
# )

# # --- 🧩 System message ---
# system_context = """
# You are Hermit, the smart AI assistant for the Right2Bill platform.
# You help users with questions about bills, GST, XP points, merchant rewards, and fake bill reporting.
# Your answers should be friendly, short (under 80 words), and easy to understand.
# Remember and use the user's previous questions and your earlier answers
# to provide contextually relevant, consistent replies.
# If a new question refers to something mentioned earlier,
# use that context to make your answer better."""

# prompt = ChatPromptTemplate.from_messages([
#     ("system", system_context),
#     MessagesPlaceholder(variable_name="history"),  # <-- adds previous messages
#     ("user", "{query}")
# ])

# # --- ⚙️ Combine LLM and prompt ---
# chain = prompt | llm

# # --- 🧱 Store user session histories ---
# session_store = {}

# def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
#     """Get or create chat history for a given session"""
#     if session_id not in session_store:
#         session_store[session_id] = InMemoryChatMessageHistory()
#     return session_store[session_id]

# # --- 🔗 Wrap chain with memory ---
# chat_with_memory = RunnableWithMessageHistory(
#     chain,
#     get_session_history,
#     input_messages_key="query",
#     history_messages_key="history"
# )

# # --- 🧑‍💻 Main chat loop ---
# print("👋 Hermit is online! (Context-aware) Type 'exit' to stop.\n")
# session_id = "user1"  # You can set per-user session IDs later

# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         print("👋 Goodbye!")
#         break

#     result = chat_with_memory.invoke(
#         {"query": user_input},
#         config={"configurable": {"session_id": session_id}}
#     )

#     print("Hermit:", result.content)

# version 3

# import os
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.runnables import RunnableParallel, RunnableBranch

# # -----------------------------------
# # 1️⃣ API & LLM setup
# # -----------------------------------
# import os
# from dotenv import load_dotenv
# load_dotenv()

# llm = ChatOpenAI(
#     model="mistralai/mistral-small-3.1-24b-instruct:free",
#     temperature=0.4,
#     base_url="https://openrouter.ai/api/v1"
# )

# # -----------------------------------
# # 2️⃣ System prompt (domain restricted)
# # -----------------------------------
# system_context = """
# You are Hermit, the official AI assistant for the Right2Bill platform.
# You ONLY answer questions related to:
# - Bills and bill verification
# - GST and tax transparency
# - XP points and rewards
# - Merchant benefits and incentives
# - Fake bill reporting
# - Platform features and user help

# If a user asks about any topic NOT related to Right2Bill or its services (like laws of physics, celebrities, math, or general knowledge), 
# politely refuse to answer and remind them that you are only trained to discuss Right2Bill-related topics.
# Keep your answers short (under 80 words), friendly, and easy to understand.
# """

# # -----------------------------------
# # 3️⃣ Sub-prompts for each route
# # -----------------------------------
# faq_prompt = ChatPromptTemplate.from_messages([
#     ("system", system_context),
#     MessagesPlaceholder(variable_name="history"),
#     ("user", "Answer the user's factual Right2Bill question clearly:\n{query}")
# ])

# complaint_prompt = ChatPromptTemplate.from_messages([
#     ("system", system_context),
#     MessagesPlaceholder(variable_name="history"),
#     ("user", "The user might be reporting a fake bill or issue.\n"
#              "Provide a step-by-step action guide and reassure them:\n{query}")
# ])

# merchant_prompt = ChatPromptTemplate.from_messages([
#     ("system", system_context),
#     MessagesPlaceholder(variable_name="history"),
#     ("user", "Explain merchant or seller rewards and benefits clearly:\n{query}")
# ])

# fallback_prompt = ChatPromptTemplate.from_messages([
#     ("system", system_context),
#     ("user", "If this question is not related to Right2Bill, politely refuse to answer and remind them about your domain:\n{query}")
# ])

# faq_chain = faq_prompt | llm
# complaint_chain = complaint_prompt | llm
# merchant_chain = merchant_prompt | llm
# fallback_chain = fallback_prompt | llm

# # -----------------------------------
# # 4️⃣ Parallel reasoning chain (for genuine multi-style response)
# # -----------------------------------
# friendly_prompt = ChatPromptTemplate.from_template(
#     "Answer this Right2Bill question in a friendly and helpful way:\n{query}"
# )
# verify_prompt = ChatPromptTemplate.from_template(
#     "Ensure this answer is factual and aligns with Right2Bill's policies:\n{query}"
# )
# policy_prompt = ChatPromptTemplate.from_template(
#     "Explain the policy or reason behind this feature or action:\n{query}"
# )

# friendly_chain = friendly_prompt | llm
# verify_chain = verify_prompt | llm
# policy_chain = policy_prompt | llm

# parallel_chain = RunnableParallel(
#     friendly=friendly_chain,
#     verify=verify_chain,
#     policy=policy_chain
# )

# # -----------------------------------
# # 5️⃣ Routing logic (intent detection)
# # -----------------------------------
# def route_query(inputs):
#     q = inputs["query"].lower()

#     # Domain filter
#     allowed_keywords = [
#         "bill", "gst", "xp", "reward", "merchant", "seller",
#         "fake", "report", "tax", "invoice", "receipt", "cashback"
#     ]
#     if not any(k in q for k in allowed_keywords):
#         return "off_topic"

#     # Intent detection
#     if any(k in q for k in ["xp", "gst", "bill", "cashback"]):
#         return "faq"
#     elif any(k in q for k in ["fake", "report", "complaint"]):
#         return "complaint"
#     elif any(k in q for k in ["merchant", "seller", "shop"]):
#         return "merchant"
#     else:
#         return "parallel"  # default route for general in-domain questions

# router = RunnableBranch(
#     (lambda x: route_query(x) == "faq", faq_chain),
#     (lambda x: route_query(x) == "complaint", complaint_chain),
#     (lambda x: route_query(x) == "merchant", merchant_chain),
#     (lambda x: route_query(x) == "off_topic", fallback_chain),
#     parallel_chain
# )

# # -----------------------------------
# # 6️⃣ Memory setup
# # -----------------------------------
# session_store = {}

# def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
#     if session_id not in session_store:
#         session_store[session_id] = InMemoryChatMessageHistory()
#     return session_store[session_id]

# chat_with_memory = RunnableWithMessageHistory(
#     router,
#     get_session_history,
#     input_messages_key="query",
#     history_messages_key="history"
# )

# # -----------------------------------
# # 7️⃣ Main chat loop
# # -----------------------------------
# print("🤖 Hermit v3 is online! (Smart + Genuine + Domain-Aware)\nType 'exit' to stop.\n")
# session_id = "user1"

# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         print("👋 Goodbye!")
#         break

#     result = chat_with_memory.invoke(
#         {"query": user_input},
#         config={"configurable": {"session_id": session_id}}
#     )

#     if isinstance(result, dict):  # Parallel output
#         print("\n--- Hermit's Combined Insight ---")
#         print("Friendly:", result['friendly'].content.strip())
#         print("Verified:", result['verify'].content.strip())
#         print("Policy:", result['policy'].content.strip())
#         print("------------------------------\n")
#     else:
#         print("Hermit:", result.content)


# version-3

# import os
# import re
# import json
# from datetime import datetime
# from uuid import uuid4
# from dotenv import load_dotenv
# from pydantic import BaseModel, Field, validator
# from typing import Optional, Literal

# # LangChain imports
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.runnables import RunnableParallel, RunnableBranch

# # -----------------------------------
# # 1️⃣ Environment & LLM Setup
# # -----------------------------------
# load_dotenv()

# llm = ChatOpenAI(
#     model="mistralai/mistral-small-3.1-24b-instruct:free",
#     temperature=0.3,
#     base_url="https://openrouter.ai/api/v1"
# )

# # -----------------------------------
# # 2️⃣ Global System Context
# # -----------------------------------
# SYSTEM_CONTEXT = """
# You are Hermit, the official AI compliance assistant for the Right2Bill platform.
# Your duties are limited to:
# - Bill verification, GST/tax-related guidance, reward inquiries
# - Merchant compliance guidance and incentive explanations
# - Customer fraud/non-compliance reporting
# - Tax Authority reporting and data verification

# ⚠️ Legal & Regulatory Rules:
# 1. You are NOT a human advisor. Remind users: “I am an AI assistant, not a tax or legal advisor.”
# 2. Never provide emotional support, personal opinions, or unrelated knowledge.
# 3. Always warn users: “Do not share personal or financial information (like PAN, Aadhaar, bank details).”
# 4. When collecting fraud data, follow step-by-step questions, validate, and output structured JSON.

# If a question is unrelated to tax, billing, or Right2Bill — refuse politely.
# """

# # -----------------------------------
# # 3️⃣ PII Redaction Middleware
# # -----------------------------------
# PII_PATTERNS = [
#     r"\b\d{12}\b",  # Aadhaar
#     r"\b\d{10}\b",  # phone number
#     r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # email
#     r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",  # PAN
#     r"(password|otp|pin)\s*[:=]\s*\S+",  # password
# ]

# def pii_redact(text: str) -> str:
#     for pat in PII_PATTERNS:
#         text = re.sub(pat, "[REDACTED]", text)
#     return text

# # -----------------------------------
# # 4️⃣ Structured Fraud Report Schema
# # -----------------------------------

# class FraudReportSchema(BaseModel):
#     report_uuid: str = Field(default_factory=lambda: str(uuid4()))
#     transaction_id: str = Field(..., pattern=r"^[A-Za-z0-9\-]{6,64}$")
#     merchant_tax_id: str
#     non_compliance_type: Literal["FakeBill", "NoBill", "IncorrectRate", "RewardFraud"]
#     timestamp_of_event: datetime
#     reported_amount_usd: float = Field(..., gt=0)
#     evidence_summary: str

#     @validator("evidence_summary")
#     def no_pii(cls, v):
#         if any(s in v for s in ["@", "http", "https", "password", "OTP"]):
#             raise ValueError("Evidence summary must not include PII or links")
#         return v

# # -----------------------------------
# # 5️⃣ Sub-Agent Prompts
# # -----------------------------------

# # -- Customer Agent --
# customer_prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_CONTEXT),
#     ("user", """
# You are the Customer Agent. 
# Your role is to help users verify bills, report fake bills, and understand rewards.

# ⚠️ Start every conversation with:
# - AI Disclosure
# - Liability disclaimer
# - PII Warning

# If reporting a fake bill, guide them step-by-step and output a structured JSON
# following this schema:
# {schema}

# User request: {query}
# """)
# ])

# # -- Merchant Agent --
# merchant_prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_CONTEXT),
#     ("user", """
# You are the Merchant Agent.
# You help merchants understand compliance rules, GST benefits, and incentives.
# For all answers:
# - Be factual, brief (under 100 words)
# - Provide a reason or policy source
# - Never commit financial advice

# User: {query}
# """)
# ])

# # -- Authority Agent --
# authority_prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_CONTEXT),
#     ("user", """
# You are the Authority Agent.
# You assist tax authorities with structured fraud data review, audit triggers, and KPI insights.
# You never disclose internal algorithms or risk models.
# Respond formally and only from verified, public data.

# User: {query}
# """)
# ])

# # -- Fallback for Off-topic --
# fallback_prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_CONTEXT),
#     ("user", """
# This user message is outside the Right2Bill domain.
# Politely refuse to answer and say:
# "I’m trained only for Right2Bill-related tax, bill, or merchant questions."

# User: {query}
# """)
# ])

# # Chains
# customer_chain = customer_prompt | llm
# merchant_chain = merchant_prompt | llm
# authority_chain = authority_prompt | llm
# fallback_chain = fallback_prompt | llm

# # -----------------------------------
# # 6️⃣ Router Logic (Intent + Role Detection)
# # -----------------------------------
# def route_query(inputs):
#     q = inputs["query"].lower()

#     if not any(k in q for k in ["bill", "gst", "reward", "merchant", "tax", "fake", "report", "invoice", "audit"]):
#         return "off_topic"

#     if any(k in q for k in ["fake", "report", "fraud"]):
#         return "customer"
#     elif any(k in q for k in ["merchant", "seller", "incentive", "benefit"]):
#         return "merchant"
#     elif any(k in q for k in ["authority", "audit", "kpi"]):
#         return "authority"
#     else:
#         return "customer"

# router = RunnableBranch(
#     (lambda x: route_query(x) == "customer", customer_chain),
#     (lambda x: route_query(x) == "merchant", merchant_chain),
#     (lambda x: route_query(x) == "authority", authority_chain),
#     (lambda x: route_query(x) == "off_topic", fallback_chain),
#     customer_chain
# )

# # -----------------------------------
# # 7️⃣ Stateful Memory per User Session
# # -----------------------------------
# session_store = {}

# def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
#     if session_id not in session_store:
#         session_store[session_id] = InMemoryChatMessageHistory()
#     return session_store[session_id]

# chat_with_memory = RunnableWithMessageHistory(
#     router,
#     get_session_history,
#     input_messages_key="query",
#     history_messages_key="history"
# )

# # -----------------------------------
# # 8️⃣ Helper: Process User Query (PII-safe)
# # -----------------------------------
# def handle_user_query(user_input: str, session_id: str):
#     redacted_input = pii_redact(user_input)
#     result = chat_with_memory.invoke(
#         {"query": redacted_input},
#         config={"configurable": {"session_id": session_id}}
#     )
#     return result

# # -----------------------------------
# # 9️⃣ CLI Chat Interface
# # -----------------------------------
# print("\n🤖 Right2Bill Compliance Assistant is LIVE!")
# print("💡 (Type 'exit' to quit)\n")

# session_id = "demo_user"

# while True:
#     query = input("You: ")
#     if query.lower() == "exit":
#         print("👋 Goodbye! Stay compliant with Right2Bill.")
#         break

#     response = handle_user_query(query, session_id)

#     if isinstance(response, dict):
#         print("\n--- Structured Output ---")
#         print(json.dumps(response, indent=2))
#         print("--------------------------\n")
#     else:
#         print("Hermit:", response.content.strip())

# version 4

# import os
# import re
# import json
# from datetime import datetime
# from uuid import uuid4
# from dotenv import load_dotenv
# from pydantic import BaseModel, Field, field_validator
# from typing import Literal

# # LangChain imports
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_core.runnables import RunnableBranch

# # -----------------------------------
# # 1️⃣ Environment & LLM Setup
# # -----------------------------------
# load_dotenv()

# llm = ChatOpenAI(
#     model="mistralai/mistral-small-3.1-24b-instruct:free",
#     temperature=0.3,
#     base_url="https://openrouter.ai/api/v1"
# )

# # -----------------------------------
# # 2️⃣ Global System Context
# # -----------------------------------
# SYSTEM_CONTEXT = """
# You are Hermit, the official AI compliance assistant for the Right2Bill platform.
# Your duties are limited to:
# - Bill verification, GST/tax-related guidance, reward inquiries
# - Merchant compliance guidance and incentive explanations
# - Customer fraud/non-compliance reporting
# - Tax Authority reporting and data verification

# ⚠️ Legal & Regulatory Rules:
# 1. You are NOT a human advisor. Always remind users: “I am an AI assistant, not a tax or legal advisor.”
# 2. Never provide emotional support, personal opinions, or unrelated knowledge.
# 3. Always warn users: “Do not share personal or financial information (like PAN, Aadhaar, bank details).”
# 4. When collecting fraud data, follow step-by-step questions, validate, and output structured JSON.

# If a question is unrelated to tax, billing, or Right2Bill — politely refuse.
# """

# # -----------------------------------
# # 3️⃣ PII Redaction Middleware
# # -----------------------------------
# PII_PATTERNS = [
#     r"\b\d{12}\b",  # Aadhaar
#     r"\b\d{10}\b",  # phone number
#     r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}",  # email
#     r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",  # PAN
#     r"(password|otp|pin)\s*[:=]\s*\S+",  # password
# ]

# def pii_redact(text: str) -> str:
#     for pat in PII_PATTERNS:
#         text = re.sub(pat, "[REDACTED]", text)
#     return text

# # -----------------------------------
# # 4️⃣ Structured Fraud Report Schema
# # -----------------------------------
# class FraudReportSchema(BaseModel):
#     report_uuid: str = Field(default_factory=lambda: str(uuid4()))
#     transaction_id: str = Field(..., pattern=r"^[A-Za-z0-9\-]{3,64}$")
#     merchant_tax_id: str
#     non_compliance_type: Literal["FakeBill", "NoBill", "IncorrectRate", "RewardFraud"]
#     timestamp_of_event: datetime
#     reported_amount_usd: float = Field(..., gt=0)
#     evidence_summary: str

#     @field_validator("evidence_summary")
#     def no_pii(cls, v):
#         if any(s in v for s in ["@", "http", "https", "password", "OTP"]):
#             raise ValueError("Evidence summary must not include PII or links")
#         return v

# # -----------------------------------
# # 5️⃣ Core API Simulation
# # -----------------------------------
# def submit_fraud_report(data: dict):
#     """Simulate sending the fraud report to backend or GST system."""
#     print("\n📤 Submitting fraud report to Right2Bill server...")
#     print(json.dumps(data, indent=2))
#     print("✅ Report submitted successfully! Your case ID:", data["report_uuid"])
#     print("🎁 You've earned 10 XP for honest reporting!\n")

# # -----------------------------------
# # 6️⃣ Fraud Report Collection Flow
# # -----------------------------------
# def collect_fraud_report_flow():
#     print("\n⚠️ Please do not share personal identifiers like Aadhaar, PAN, or phone numbers.")
#     print("Let's collect the necessary details for your fraud report.\n")

#     try:
#         txn_id = input("🧾 Transaction ID or short description: ").strip() or "UNKNOWN"
#         merchant_id = input("🏪 Merchant GST ID (if known): ").strip() or "UNKNOWN"
#         issue = input("🚨 Issue Type [FakeBill / NoBill / IncorrectRate / RewardFraud]: ").strip() or "NoBill"
#         date_str = input("📅 Date of transaction (YYYY-MM-DD): ").strip()
#         timestamp = datetime.fromisoformat(date_str) if date_str else datetime.now()
#         amount_inr = float(input("💰 Approx amount (₹): ").strip() or 0)
#         desc = input("📝 Describe what happened: ").strip()

#         desc_clean = pii_redact(desc)

#         report = FraudReportSchema(
#             transaction_id=txn_id,
#             merchant_tax_id=merchant_id,
#             non_compliance_type=issue,
#             timestamp_of_event=timestamp,
#             reported_amount_usd=round(amount_inr / 83.0, 2),
#             evidence_summary=desc_clean
#         )

#         submit_fraud_report(report.dict())

#     except Exception as e:
#         print("❌ Error during report collection:", str(e))
#         print("Please try again.\n")

# # -----------------------------------
# # 7️⃣ Sub-Agent Prompts
# # -----------------------------------
# customer_prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_CONTEXT),
#     MessagesPlaceholder(variable_name="history"),
#     ("user", """
# You are the Customer Agent. 
# Your role is to help users verify bills, report fake bills, and understand rewards.

# If the user reports missing or fake bills, say:
# “I can help you file a fraud report. Would you like to continue?”

# User request: {query}
# """)
# ])

# merchant_prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_CONTEXT),
#     MessagesPlaceholder(variable_name="history"),
#     ("user", """
# You are the Merchant Agent.
# Explain GST compliance, merchant incentives, and rewards clearly.

# User: {query}
# """)
# ])

# authority_prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_CONTEXT),
#     MessagesPlaceholder(variable_name="history"),
#     ("user", """
# You are the Authority Agent.
# Provide formal, structured insights based on verified data only.

# User: {query}
# """)
# ])

# fallback_prompt = ChatPromptTemplate.from_messages([
#     ("system", SYSTEM_CONTEXT),
#     ("user", """
# This message is outside the Right2Bill domain.
# Politely refuse and explain that you can only answer GST, billing, or merchant-related questions.

# User: {query}
# """)
# ])

# # Chains
# customer_chain = customer_prompt | llm
# merchant_chain = merchant_prompt | llm
# authority_chain = authority_prompt | llm
# fallback_chain = fallback_prompt | llm

# # -----------------------------------
# # 8️⃣ Router Logic
# # -----------------------------------
# def route_query(inputs):
#     q = inputs["query"].lower()

#     if not any(k in q for k in ["bill", "gst", "reward", "merchant", "tax", "fake", "report", "invoice", "audit"]):
#         return "off_topic"

#     if any(k in q for k in ["fake", "report", "fraud", "no bill", "didn't give bill"]):
#         return "customer"
#     elif any(k in q for k in ["merchant", "seller", "incentive", "benefit"]):
#         return "merchant"
#     elif any(k in q for k in ["authority", "audit", "kpi"]):
#         return "authority"
#     else:
#         return "customer"

# router = RunnableBranch(
#     (lambda x: route_query(x) == "customer", customer_chain),
#     (lambda x: route_query(x) == "merchant", merchant_chain),
#     (lambda x: route_query(x) == "authority", authority_chain),
#     (lambda x: route_query(x) == "off_topic", fallback_chain),
#     customer_chain
# )

# # -----------------------------------
# # 9️⃣ Stateful Memory
# # -----------------------------------
# session_store = {}

# def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
#     if session_id not in session_store:
#         session_store[session_id] = InMemoryChatMessageHistory()
#     return session_store[session_id]

# chat_with_memory = RunnableWithMessageHistory(
#     router,
#     get_session_history,
#     input_messages_key="query",
#     history_messages_key="history"
# )

# # -----------------------------------
# # 🔟 Main Chat Interface
# # -----------------------------------
# print("\n🤖 Right2Bill Compliance Assistant is LIVE!")
# print("💡 Type 'exit' to quit, or 'report' to file a fraud report manually.\n")

# session_id = "demo_user"
    
# while True:
#     user_input = input("You: ").strip()
#     if user_input.lower() == "exit":
#         print("👋 Goodbye! Stay compliant with Right2Bill.")
#         break

#     # Pre-detect fraud-related phrases
#     fraud_phrases = [
#         "no bill", "did not give bill", "didn't give bill",
#         "fake bill", "fake invoice", "wrong gst", "report merchant"
#     ]
#     if any(p in user_input.lower() for p in fraud_phrases):
#         print("\nHermit: Under GST law, merchants must issue a valid bill for purchases over ₹200.")
#         print("Let’s file a quick fraud report so the authorities can verify this transaction.")
#         print("⚠️ Please do not share personal identifiers like Aadhaar, PAN, or bank details.\n")
#         collect_fraud_report_flow()
#         continue

#     # Manual fraud report command
#     if user_input.lower() in ["report", "file report", "fraud report"]:
#         collect_fraud_report_flow()
#         continue

#     # Regular conversational mode
#     response = chat_with_memory.invoke(
#         {"query": pii_redact(user_input)},
#         config={"configurable": {"session_id": session_id}}
#     )

#     reply = response.content.strip() if not isinstance(response, dict) else json.dumps(response, indent=2)
#     print("Hermit:", reply)


# Multi-Model Enhanced Version

import os
import re
import json
from datetime import datetime
from uuid import uuid4
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from typing import Literal

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableBranch

# -----------------------------------
# 1️⃣ Environment & Model Setup
# -----------------------------------
load_dotenv()

# ✅ Separate models for each agent
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
# 3️⃣ PII Redaction
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
# 4️⃣ Fraud Report Schema
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

# -----------------------------------
# 5️⃣ Fraud Report Flow
# -----------------------------------
from datetime import datetime

def submit_fraud_report(data: dict):
    """Simulate sending the fraud report to backend or GST system."""
    print("\n📤 Submitting fraud report to Right2Bill server...")

    # Convert datetime → ISO string
    safe_data = {
        k: (v.isoformat() if isinstance(v, datetime) else v)
        for k, v in data.items()
    }

    print(json.dumps(safe_data, indent=2))
    print("✅ Report submitted successfully! Your case ID:", safe_data["report_uuid"])
    print("🎁 You've earned 10 XP for honest reporting!\n")

def collect_fraud_report_flow():
    print("\n⚠️ Please do not share personal identifiers (Aadhaar, PAN, bank details).")
    print("Let's collect details for your fraud report.\n")

    try:
        txn_id = input("🧾 Transaction ID or short description: ").strip() or "UNKNOWN"
        merchant_id = input("🏪 Merchant GST ID (if known): ").strip() or "UNKNOWN"
        issue = input("🚨 Issue Type [FakeBill / NoBill / IncorrectRate / RewardFraud]: ").strip() or "NoBill"
        date_str = input("📅 Date of transaction (YYYY-MM-DD): ").strip()
        timestamp = datetime.fromisoformat(date_str) if date_str else datetime.now()
        amount_inr = float(input("💰 Approx amount (₹): ").strip() or 0)
        desc = input("📝 Describe what happened: ").strip()

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
        print("❌ Error:", str(e))
        print("Please try again.\n")

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
fallback_chain = fallback_prompt | CUSTOMER_LLM  # safe fallback

# -----------------------------------
# 8️⃣ Routing Logic
# -----------------------------------
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
# 9️⃣ Stateful Memory
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
# 🔟 Main CLI Chat Loop
# -----------------------------------
print("\n🤖 Right2Bill Multi-Model Assistant is LIVE!")
print("💡 Type 'exit' to quit, or 'report' to file a fraud report manually.\n")

session_id = "demo_user"



def is_fraud_trigger(text: str) -> bool:
    text = text.lower()
    patterns = [
        r"no\s+bill",
        r"did\s*not\s+(give|issue).*bill",
        r"didn'?t\s+(give|issue).*bill",
        r"merchant\s+did\s*not\s+bill",
        r"fake\s+bill",
        r"fake\s+invoice",
        r"report\s+merchant",
        r"wrong\s+gst"
    ]
    return any(re.search(p, text) for p in patterns)


# -------------------------------------------
# 🧩 Improved Main Chat Loop
# -------------------------------------------
print("\n🤖 Right2Bill Multi-Model Assistant is LIVE!")
print("💡 Type 'exit' to quit, or 'report' to file a fraud report manually.\n")

session_id = "demo_user"

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "exit":
        print("👋 Goodbye! Stay compliant with Right2Bill.")
        break

    # 1️⃣ Fraud keyword check FIRST
    if is_fraud_trigger(user_input):
        
        short_reply = CUSTOMER_LLM.invoke([
    {"role": "system", "content": SYSTEM_CONTEXT},
    {"role": "user", "content": f"The user said: '{user_input}'. "
                                f"Respond in under 60 words. "
                                f"Politely explain that under GST, merchants must issue a bill for any purchase above ₹200. "
                                f"State that failing to issue a bill is a legal offence that can attract penalties. "
                                f"Then tell the user we’ll start a fraud report to verify this. "
                                f"Keep tone formal, factual, and non-emotional."}
        ])


        print("\nHermit:", short_reply.content.strip())
        print("⚠️ Please do not share personal identifiers like Aadhaar, PAN, or bank details.\n")

        confirm = input("Would you like to file a fraud report now? (yes/no): ").strip().lower()
        if confirm.startswith("y"):
            collect_fraud_report_flow()
        else:
            print("Okay — no report filed. You can still submit it anytime by typing 'report'.")
        continue

    # 2️⃣ Manual fraud report command
    if user_input.lower() in ["report", "file report", "fraud report"]:
        collect_fraud_report_flow()
        continue

    # 3️⃣ Normal chat routing
    response = chat_with_memory.invoke(
        {"query": pii_redact(user_input)},
        config={"configurable": {"session_id": session_id}}
    )

    reply = response.content.strip() if not isinstance(response, dict) else json.dumps(response, indent=2)
    print("Hermit:", reply)

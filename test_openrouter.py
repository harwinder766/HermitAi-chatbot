# import os
# import requests
# from dotenv import load_dotenv

# # Load your OpenRouter API key
# load_dotenv()
# API_KEY = os.getenv("OPENROUTER_API_KEY")

# # Base URL for OpenRouter API
# url = "https://openrouter.ai/api/v1/chat/completions"

# # Model to test
# model = "meta-llama/llama-4-maverick:free"

# #model = "mistralai/mistral-small-3.1-24b-instruct:free"
# # You can also try: "mistralai/mistral-small-3.1-24b-instruct:free"

# # Message to send to model
# payload = {
#     "model": model,
#     "messages": [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Hello! Can you tell me if you’re working?"}
#     ]
# }

# # Headers (include your API key)
# headers = {
#     "Authorization": f"Bearer {API_KEY}",
#     "Content-Type": "application/json"
# }

# # Make API request
# response = requests.post(url, headers=headers, json=payload)

# # Check response
# if response.status_code == 200:
#     reply = response.json()["choices"][0]["message"]["content"]
#     print("✅ Model working! Response:\n", reply)
# else:
#     print("❌ Error:", response.status_code, response.text)



import os
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser

# --- 1. Define Output Schema using Pydantic ---
class SummaryAndPoints(BaseModel):
    """A structured response containing a summary and key points."""
    summary: str = Field(description="Short summary of the text.")
    key_points: list[str] = Field(description="Main bullet points derived from the text.")

# --- 2. Initialize LLM (using correct, modern setup) ---
# Ensure you are using the correct environment variable setup or direct arguments.
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(
    model="tngtech/deepseek-r1t-chimera:free",
    temperature=0.0 # openai/gpt-oss-20b:freeSet temperature lower for reliable structured output
)

# --- 3. Define the Parser ---
parser = JsonOutputParser(pydantic_object=SummaryAndPoints)
format_instructions = parser.get_format_instructions()

# --- 4. Define Prompt and Chain (using LCEL for robustness) ---
# Use a system message to strictly constrain the model's output.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict data formatting bot. Your sole output **MUST** be a valid JSON object. "
            "Do not include any other text, chatter, reasoning, or markdown (e.g., '```json' or '```') outside of the JSON block itself. "
            "The JSON object must strictly adhere to this schema:\n{format_instructions}"
        ),
        ("user", "Text to summarize: {text}"),
    ]
).partial(format_instructions=format_instructions)

# The chain is: Prompt -> LLM -> Parser
chain = prompt | llm | parser

# --- 5. Run and Parse ---
try:
    parsed_output = chain.invoke({
        "text": "LangChain simplifies integration of large language models in apps. It helps developers create data-aware, agentic applications powered by LLMs. It is organized into components, runnable chains, and agents."
    })
    print("✅ Successfully Parsed Output:")
    print(parsed_output)

except Exception as e:
    print(f"❌ Chain failed due to: {e}")
    print("Try increasing the LLM's context or using a more powerful model for structured output.")
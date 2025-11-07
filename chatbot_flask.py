# chatbot_flask.py
import os
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ===============================
# CONFIGURATION
# ===============================

# OpenRouter credentials
os.environ["OPENAI_API_KEY"] = "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"
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# ===============================
# MODEL SETUP
# ===============================

# Initialize the OpenRouter model (Mistral)
llm = ChatOpenAI(
    model="mistralai/mistral-small-3.1-24b-instruct:free",
    temperature=0.3,
)

# System + user prompt setup
system_context = """
You are Billu, the AI assistant for the Right2Bill platform.
Answer user questions about bills, GST, XP points, merchant rewards, and fake bill reporting.
Keep your answers friendly, short (under 80 words), and easy to understand.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_context),
    ("user", "{query}")
])

# Combine into a simple chain
chain = prompt | llm

# ===============================
# CHAT ENDPOINT
# ===============================

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json() or {}
        question = data.get("question") or request.form.get("question")

        if not question:
            return jsonify({"answer": "Please ask a question."}), 400

        result = chain.invoke({"query": question})

        # 🧠 FIX: extract the content from AIMessage
        answer_text = result.content if hasattr(result, "content") else str(result)
        return jsonify({"answer": answer_text})

    except Exception as e:
        return jsonify({"answer": f"Sorry, something went wrong: {str(e)}"}), 500


# ===============================
# HTML CHAT UI (AUTO-SERVED)
# ===============================

@app.route("/", methods=["GET"])
def home():
    # Inline HTML served directly from Flask
    html_code = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8" />
      <title>Right2Bill Chatbot</title>
      <style>
        body { font-family: Arial, sans-serif; background-color: #f8f9fa; padding: 40px; }
        .chat-container { background: white; padding: 20px; border-radius: 15px; width: 450px; margin: auto; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }
        input { width: 80%; padding: 10px; border-radius: 8px; border: 1px solid #ccc; }
        button { padding: 10px 15px; border: none; background-color: #007bff; color: white; border-radius: 8px; cursor: pointer; }
        button:hover { background-color: #0056b3; }
        p { margin-top: 20px; font-weight: bold; color: #2d862d; }
      </style>
    </head>
    <body>
      <div class="chat-container">
        <h2>💬 Chat with Billu</h2>
        <p>Ask me about GST, XP points, or merchants!</p>
        <input id="question" placeholder="Type your question..." />
        <button onclick="askBot()">Ask</button>
        <p id="answer"></p>
      </div>

      <script>
        async function askBot() {
          const question = document.getElementById("question").value;
          if (!question) {
            document.getElementById("answer").innerText = "Please type a question!";
            return;
          }

          document.getElementById("answer").innerText = "⏳ Thinking...";
          const response = await fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ question }),
          });

          const data = await response.json();
          document.getElementById("answer").innerText = data.answer || "No response.";
        }
      </script>
    </body>
    </html>
    """
    return render_template_string(html_code)


# ===============================
# RUN SERVER
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)

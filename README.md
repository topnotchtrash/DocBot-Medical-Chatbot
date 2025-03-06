# 🩺 DocBot - Medical Chatbot

DocBot is an AI-powered medical assistant that provides accurate and reliable answers to medical-related questions using **Mistral-7B** and a **FAISS vector database**.

---

## 🚀 Features
- Provides **fact-based** medical responses.
- Uses **FAISS** for efficient retrieval from a medical knowledge base.
- **Memory-enabled** chatbot for conversational interactions.
- Clean and **interactive UI** using Streamlit.
- Secure **environment variable management** (via `.env` file).

---

## 🛠 Installation

### 1️⃣ **Clone the repository**
```bash
git clone https://github.com/sandavenishubhang/DocBot-Medical-Chatbot.git
cd DocBot-Medical-Chatbot
```

---

### 2️⃣ **Create a Virtual Environment (Recommended)**
A virtual environment helps manage dependencies and prevents conflicts.

```bash
python3 -m venv medical-chatbot-env
```

Activate the virtual environment:
- **macOS/Linux:**
  ```bash
  source medical-chatbot-env/bin/activate
  ```
- **Windows (Command Prompt):**
  ```bash
  medical-chatbot-env\Scripts\activate
  ```
- **Windows (PowerShell):**
  ```powershell
  .\medical-chatbot-env\Scripts\Activate.ps1
  ```

---

### 3️⃣ **Install Dependencies**
Install all required Python packages using:
```bash
pip install -r requirements.txt
```

---

### 4️⃣ **Set Up Environment Variables**
You need to configure your **Hugging Face API Token** to access the Mistral-7B model.

1. **Create a `.env` file** from the example file:
   ```bash
   cp .env.example .env
   ```
2. **Open the `.env` file** and add your Hugging Face API key:
   ```
   HF_TOKEN=your_huggingface_api_token_here
   ```

---

### 5️⃣ **Run the Chatbot**
Start the **Streamlit** app:
```bash
streamlit run docbot.py
```

This will launch the chatbot in your **web browser** at:
```
http://localhost:8501
```

---

### 🎯 **Next Steps**
- Start asking **medical-related questions** in the input box.
- **DocBot will generate a response** based on its knowledge.
- The chatbot **retains conversation history** during the session.

Let me know if you need further assistance! 🚀😊

# AI Chatbot - Streamlit + Groq
 
Simple Streamlit chatbot connected directly to Groq.
 
## Features
- Groq chat completion API
- Conversation history in session
- Minimal setup (no database)
 
## Setup
1. Create a Groq API key from https://console.groq.com/keys
2. Add key in Streamlit secrets:
 
    ```toml
    GROQ_API_KEY = "your_key_here"
    ```
 
    Or add it to a local `.env` file:
 
    ```bash
    GROQ_API_KEY=your_key_here
    ```
 
    Or set environment variable:
 
    ```bash
    export GROQ_API_KEY="your_key_here"
    ```
 
3. Install dependencies:
 
    ```bash
    pip install -r requirements.txt
    ```
 
4. Run app:
 
    ```bash
    streamlit run app.py
    ```
 
## Push to GitHub
1. Initialize git if needed:
 
    ```bash
    git init
    ```
 
2. Add and commit your changes:
 
    ```bash
    git add .
    git commit -m "Add Groq chatbot"
    ```
 
3. Connect your GitHub repo and push:
 
    ```bash
    git branch -M main
    git remote add origin https://github.com/<your-username>/<your-repo>.git
    git push -u origin main
    ```
 
## Deploy on Render
1. Push the project to GitHub first.
2. Go to https://render.com and create a new Web Service.
3. Connect your GitHub repository.
4. Use these settings:
 
    ```text
    Build Command: pip install -r requirements.txt
    Start Command: streamlit run app.py --server.port $PORT --server.address 0.0.0.0
    ```
 
5. Add the environment variable:
 
    ```text
    GROQ_API_KEY=your_key_here
    ```
 
6. Deploy the service.
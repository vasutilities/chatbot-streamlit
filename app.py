import os
import time
from dotenv import dotenv_values, load_dotenv
from groq import Groq
import streamlit as st

# ============================================
# 1. PAGE SETUP & CONFIGURATION
# ============================================
st.set_page_config(page_title="AI Chatbot Studio", page_icon="⚡", layout="wide")
st.markdown("<style>.stApp{max-width:1050px;margin:0 auto;}</style>", unsafe_allow_html=True)

load_dotenv()

def get_api_key():
    """Retrieve Groq API key securely from multiple fallback sources."""
    # 1. Check local .env file
    env_file_key = dotenv_values(".env").get("GROQ_API_KEY")
    if env_file_key:
        return env_file_key

    # 2. Check Streamlit cloud secrets
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    # 3. Check OS environment variables
    return os.getenv("GROQ_API_KEY")

GROQ_API_KEY = get_api_key()

# ============================================
# 2. SIDEBAR CONTROLS & MODEL SETTINGS
# ============================================
with st.sidebar:
    st.title("⚡ Settings & Controls")
    
    # API Key input fallback if not present in env
    if not GROQ_API_KEY:
        GROQ_API_KEY = st.text_input("Enter Groq API Key", type="password")
        if not GROQ_API_KEY:
            st.warning("⚠️ Please provide an API key to start.")
            st.stop()
    
    # Model Selection
    model = st.selectbox(
        "Model",
        [
            "openai/gpt-oss-20b",
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant"
        ],
        index=0
    )
    
    # Persona / System Prompt Selector
    st.subheader("🎭 AI Persona")
    persona_choice = st.selectbox(
        "Choose Persona",
        [
            "Helpful Tutor (Explains concepts simply)",
            "Python Coding Mentor (Clean code & explanations)",
            "Ultra Concise (Answers in 1-2 bullet points)",
            "Custom Persona"
        ]
    )
    
    personas = {
        "Helpful Tutor (Explains concepts simply)": "You are an empathetic, clear university tutor. Use simple analogies and step-by-step reasoning.",
        "Python Coding Mentor (Clean code & explanations)": "You are a senior Python software engineer. Always provide robust, idiomatic Python code with brief explanations.",
        "Ultra Concise (Answers in 1-2 bullet points)": "You are a concise assistant. Provide answers strictly in 1 or 2 bullet points without filler."
    }
    
    if persona_choice == "Custom Persona":
        system_instruction = st.text_area("System Prompt", value="You are a helpful assistant.")
    else:
        system_instruction = personas[persona_choice]
        st.caption(f"_{system_instruction}_")
    
    # Hyperparameters
    st.subheader("⚙️ Hyperparameters")
    temperature = st.slider("Temperature (Creativity)", min_value=0.0, max_value=1.5, value=0.2, step=0.1)
    max_tokens = st.slider("Max Completion Tokens", min_value=100, max_value=2000, value=500, step=50)

    # Chat Actions
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# Initialize Groq Client
try:
    groq_client = Groq(api_key=GROQ_API_KEY)
except Exception as e:
    st.error(f"Failed to initialize Groq Client: {e}")
    st.stop()

# ============================================
# 3. CONVERSATION STATE MANAGEMENT
# ============================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Main Title Area
st.title("💬 Interactive LLM Chat Studio")
st.caption(f"Running **{model}** via Groq Cloud | Optimized for latency & live token streaming")

# Display Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Export Chat History Button
with col2:
    if st.session_state.messages:
        chat_export = "\n\n".join([f"**{m['role'].upper()}**: {m['content']}" for m in st.session_state.messages])
        st.download_button(
            "💾 Export",
            data=chat_export,
            file_name="chat_transcript.md",
            mime="text/markdown",
            use_container_width=True
        )

# ============================================
# 4. STREAMING CHAT PIPELINE (Universal Fallback)
# ============================================
if user_prompt := st.chat_input("Ask a question or request code..."):
    # 1. Append and render user input
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # 2. Render Assistant response via streaming
    with st.chat_message("assistant"):
        # Assemble message payload (System Prompt + Recent History)
        api_messages = [{"role": "system", "content": system_instruction}]
        for m in st.session_state.messages[-8:]:  # Sliding memory window of last 8 turns
            api_messages.append({"role": m["role"], "content": m["content"]})

        try:
            start_time = time.time()

            # Streaming API call to Groq
            stream = groq_client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            # Universal streaming loop (compatible with all Streamlit versions)
            response_placeholder = st.empty()
            full_response = ""

            for chunk in stream:
                delta = chunk.choices[0].delta
                content = delta.content or ""
                full_response += content
                # Render intermediate text with a blinking cursor indicator
                response_placeholder.markdown(full_response + "▌")

            # Final clean render without the cursor
            response_placeholder.markdown(full_response)

            elapsed = time.time() - start_time

            # Show response telemetry
            st.caption(f"⏱️ Generated in `{elapsed:.2f}s` using `{model}`")

            # Persist completed response to memory
            st.session_state.messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            st.error(f"❌ API Request Failed: {e}")

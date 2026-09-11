import os

import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="ElectroStore", page_icon="🛍️", layout="wide")

# Setup Groq Client
GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "--- IGNORE ---",
)
client = Groq(api_key=GROQ_API_KEY)

if "messages" not in st.session_state:
    st.session_state.messages = []

# ============================================
# 1. E-COMMERCE STOREFRONT
# ============================================

header_col, btn_col = st.columns([8, 2])

with header_col:
    st.title("🛍️ ElectroStore")
    st.caption("Latest electronics delivered to your doorstep")

with btn_col:
    st.write("")  # Alignment spacer
    open_chat = st.button("💬 Chat with AI Support", type="primary", use_container_width=True)

# Product Grid Display
PRODUCTS = [
    {"name": "Wireless Noise-Canceling Headphones", "price": "$199", "tag": "Audio", "img": "🎧"},
    {"name": "Mechanical Gaming Keyboard RGB", "price": "$89", "tag": "Gaming", "img": "⌨️"},
    {"name": "Ultra-Slim 4K Monitor 27-inch", "price": "$349", "tag": "Displays", "img": "🖥️"},
    {"name": "Ergonomic Wireless Mouse", "price": "$49", "tag": "Accessories", "img": "🖱️"},
]

cols = st.columns(4)

for i, item in enumerate(PRODUCTS):
    with cols[i]:
        st.markdown(f"### {item['img']}")
        st.subheader(item["name"])
        st.write(f"**Category:** {item['tag']}")
        st.write(f"**Price:** {item['price']}")
        st.button("Add to Cart", key=f"cart_{i}", use_container_width=True)

# ============================================
# 2. POPUP CHATBOT DIALOG
# ============================================

@st.dialog("ElectroStore AI Assistant 🤖", width="large")
def chat_modal():
    # Chat container for scrollable history
    chat_box = st.container(height=350)

    for msg in st.session_state.messages:
        with chat_box.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat Input
    if user_input := st.chat_input("Ask about products, orders, or specs..."):
        st.session_state.messages.append({"role": "user", "content": user_input})

        with chat_box.chat_message("user"):
            st.write(user_input)

        # AI Response
        with chat_box.chat_message("assistant"):
            sys_prompt = (
                "You are a concise e-commerce sales assistant for ElectroStore. "
                "Catalog: Headphones ($199), Keyboard ($89), 4K Monitor ($349), Mouse ($49)."
            )
            payload = [{"role": "system", "content": sys_prompt}] + st.session_state.messages[-6:]

            try:
                stream = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=payload,
                    temperature=0.2,
                    max_tokens=2000,
                    stream=True,
                )

                placeholder = st.empty()
                full_reply = ""

                for chunk in stream:
                    full_reply += chunk.choices[0].delta.content or ""
                    placeholder.markdown(full_reply + "▌")

                placeholder.markdown(full_reply)
                st.session_state.messages.append({"role": "assistant", "content": full_reply})

            except Exception as e:
                st.error(f"Error: {e}")

if open_chat:
    chat_modal()

import streamlit as st
from PIL import Image
from io import BytesIO
import requests
from openai import OpenAI
import replicate
import os

# --- Set your API keys here or via Streamlit secrets ---
OPENAI_API_KEY = st.secrets["openai_api_key"]
REPLICATE_API_TOKEN = st.secrets["replicate_api_token"]

# --- OpenAI client setup ---
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(page_title="Damney AI", layout="centered")
st.title("🤖 Damney AI")
st.markdown("Your personal mobile AI assistant ✨")

# --- Tabs ---
tab1, tab2 = st.tabs(["🧠 Chat", "🖼️ Enhance Image"])

# --- Chat Tab ---
with tab1:
    st.subheader("Ask Me Anything")
    user_input = st.text_input("You:", placeholder="Type your message here...")

    if user_input:
        with st.spinner("Thinking..."):
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are Damney AI, a helpful, stylish assistant."},
                    {"role": "user", "content": user_input}
                ]
            )
            reply = response.choices[0].message.content
            st.markdown(f"**Damney AI:** {reply}")

# --- Enhance Image Tab ---
with tab2:
    st.subheader("Upload an image to enhance")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        enhance_button = st.button("Enhance Image with AI")

        if enhance_button:
            with st.spinner("Enhancing..."):
                response = replicate.run(
                    "tencentarc/gfpgan",
                    input={"img": uploaded_file, "scale": 2, "version": "1.3"},
                    api_token=REPLICATE_API_TOKEN
                )

                # Load the output image
                enhanced_url = response
                if isinstance(response, list):
                    enhanced_url = response[0]

                enhanced_image = Image.open(requests.get(enhanced_url, stream=True).raw)
                st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)

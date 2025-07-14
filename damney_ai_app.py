
import streamlit as st
import openai
import replicate
from PIL import Image
import requests
from io import BytesIO

# --- Configuration ---
openai.api_key = "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"
replicate_api = "r8_4yKBHcPg3Y5372miRThSb0orPHR1pUm1edu6s"

# --- Page Config ---
st.set_page_config(page_title="Damney AI", layout="centered")
st.title("🤖 Damney AI")
st.subheader("Your personal mobile AI assistant")
st.markdown("---")

# --- Tabs for Functionality ---
tab1, tab2 = st.tabs(["💬 Chat", "🖼️ Enhance Image"])

# --- Tab 1: Chat ---
with tab1:
    st.write("Talk to Damney AI")

    user_input = st.text_input("You:", placeholder="Ask anything...")
    if user_input:
        with st.spinner("Thinking..."):
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are Damney AI, a helpful and smart assistant."},
                    {"role": "user", "content": user_input},
                ]
            )
            st.markdown("**Damney AI:** " + response["choices"][0]["message"]["content"])

# --- Tab 2: Enhance Image ---
with tab2:
    uploaded_image = st.file_uploader("Upload an image to enhance", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Original Image", use_column_width=True)

        if st.button("✨ Enhance"):
            with st.spinner("Enhancing..."):
                replicate_client = replicate.Client(api_token=replicate_api)
                output = replicate_client.run(
                    "cjwbw/real-esrgan:1f8c088c3ef07917311a57a3d4d7b3f420e6843706d6e1b1116d67fa7dcd9303",
                    input={"image": uploaded_image}
                )
                enhanced_url = output["output"]
                response = requests.get(enhanced_url)
                enhanced_img = Image.open(BytesIO(response.content))
                st.image(enhanced_img, caption="Enhanced Image", use_column_width=True)
                st.markdown(f"[Download Enhanced Image]({enhanced_url})")

st.markdown("---")
st.caption("Powered by GPT-4 and Replicate • Built by you, assisted by ChatGPT")

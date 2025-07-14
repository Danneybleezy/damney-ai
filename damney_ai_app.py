import streamlit as st
from PIL import Image
import requests
from openai import OpenAI
import replicate

# --- Load API keys securely from Streamlit secrets ---
OPENAI_API_KEY = st.secrets["openai_api_key"]
REPLICATE_API_TOKEN = st.secrets["replicate_api_token"]

# --- Set up OpenAI client ---
client = OpenAI(api_key=OPENAI_API_KEY)

# --- Streamlit page config ---
st.set_page_config(page_title="🤖 Damney AI", layout="centered")
st.title("🤖 Damney AI")
st.markdown("Your personal mobile AI assistant ✨")

# --- Tabs: Chat and Image Enhancer ---
tab1, tab2 = st.tabs(["🧠 Chat", "🖼️ Enhance Image"])

# --- Chat Tab ---
with tab1:
    st.subheader("Ask Me Anything")
    user_input = st.text_input("You:", placeholder="Type your message...")

    if user_input:
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",  # safer default
                    messages=[
                        {"role": "system", "content": "You are Damney AI, a helpful mobile assistant."},
                        {"role": "user", "content": user_input}
                    ]
                )
                reply = response.choices[0].message.content
                st.markdown(f"**Damney AI:** {reply}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# --- Image Enhancement Tab ---
with tab2:
    st.subheader("Upload an image to enhance")
    uploaded_file = st.file_uploader("Choose a photo", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        if st.button("✨ Enhance Image with AI"):
            with st.spinner("Enhancing... please wait"):
                try:
                    output_url = replicate.run(
                        "tencentarc/gfpgan",
                        input={"img": uploaded_file, "scale": 2, "version": "1.3"},
                        api_token=REPLICATE_API_TOKEN
                    )

                    # If replicate returns a list, use the first item
                    if isinstance(output_url, list):
                        output_url = output_url[0]

                    enhanced_image = Image.open(requests.get(output_url, stream=True).raw)
                    st.image(enhanced_image, caption="Enhanced Image", use_column_width=True)
                except Exception as e:
                    st.error(f"Image enhancement failed: {str(e)}")

import streamlit as st
import requests

st.set_page_config(page_title="SMS Spam Detection", page_icon="📩", layout="centered")

st.title("📩 SMS Spam Detection App")
st.write("This app uses a trained Machine Learning model deployed via FastAPI to predict whether a message is **Spam** or **Not Spam**.")

st.divider()

# API endpoint
API_URL = "https://sms-spam-detection-api.onrender.com/predict"

# Input text
text_input = st.text_area("✉️ Enter your message:", placeholder="Type a message to check if it's spam or not...", height=150)

if st.button("🚀 Check Message"):
    if text_input.strip() == "":
        st.warning("Please enter a message to analyze.")
    else:
        try:
            response = requests.post(API_URL, json={"text": text_input})
            if response.status_code == 200:
                result = response.json()
                label = result.get("label", "Unknown")
                prob = result.get("probability", 0)

                if label == "SPAM":
                    st.error(f"🚨 **Spam Detected!** (Probability: {prob:.2f})")
                else:
                    st.success(f"✅ **Not Spam** (Confidence: {prob:.2f})")

            else:
                st.error(f"API returned an error: {response.status_code}")
        except Exception as e:
            st.error(f"Error contacting API: {e}")

st.caption("Powered by Streamlit + FastAPI 🚀 | Developed by Nithin Chowdary")

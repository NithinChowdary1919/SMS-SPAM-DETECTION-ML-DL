import streamlit as st
import requests


st.set_page_config(page_title="SMS Spam Detector", page_icon="📩")
st.title("SMS Spam Detection")


st.markdown("Enter an SMS text below and click Predict. This app calls the FastAPI backend to get predictions.")


text = st.text_area("Message", height=150)


api_url = st.text_input("API URL (keep blank for local http://localhost:8000)", value="http://localhost:8000")


if st.button("Predict"):
if text.strip()=="":
st.warning("Please enter a message to classify.")
else:
try:
resp = requests.post(f"{api_url}/predict", json={"text": text}, timeout=10)
if resp.status_code==200:
data = resp.json()
st.success(f"Prediction: {data['label']}")
st.info(f"Spam probability: {data['probability']:.4f}")
else:
st.error(f"API error: {resp.status_code} {resp.text}")
except Exception as e:
st.error(f"Request failed: {e}")
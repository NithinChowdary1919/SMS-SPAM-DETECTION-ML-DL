import streamlit as st
import requests
from streamlit_lottie import st_lottie
import json

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="SMS Spam Detection",
    page_icon="📩",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -------------------- STYLING --------------------
st.markdown("""
    <style>
        .main {
            background: linear-gradient(135deg, #f0f2f6, #d9e4f5);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0px 4px 20px rgba(0,0,0,0.15);
        }
        .stTextArea textarea {
            border-radius: 12px !important;
            border: 1px solid #5a5a5a !important;
            font-size: 16px !important;
            padding: 1rem !important;
        }
        .stButton>button {
            background-color: #2563eb;
            color: white;
            border-radius: 10px;
            padding: 0.75rem 1.5rem;
            font-size: 16px;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #1e40af;
            transform: scale(1.05);
        }
        .result-box {
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.2rem;
            margin-top: 1rem;
        }
        .spam {
            background-color: #fee2e2;
            color: #b91c1c;
            border: 2px solid #b91c1c;
        }
        .ham {
            background-color: #dcfce7;
            color: #166534;
            border: 2px solid #166534;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------- LOTTIE ANIMATION --------------------
# -------------------- LOTTIE ANIMATION (Safe Loader) --------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"⚠️ Could not load Lottie animation: {e}")
    return None  # Return None if loading fails

# Try to load animation
spam_anim = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_ydo1amjm.json")

# Display animation only if loaded
if spam_anim:
    from streamlit_lottie import st_lottie
    st_lottie(spam_anim, height=200, key="spam-animation")
else:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/565/565547.png",
        width=120,
        caption="SMS Spam Detection"
    )

# -------------------- API URL --------------------
API_URL = "https://sms-spam-detection-api.onrender.com/predict"

# -------------------- INPUT AREA --------------------
with st.container():
    st.markdown("<div class='main'>", unsafe_allow_html=True)
    text_input = st.text_area(
        "✉️ Enter your message:",
        placeholder="Type your message here...",
        height=150
    )

    if st.button("🚀 Analyze Message"):
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
                        st.markdown(
                            f"<div class='result-box spam'>🚨 <b>Spam Detected!</b><br>Probability: {prob:.2f}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='result-box ham'>✅ <b>Not Spam</b><br>Confidence: {prob:.2f}</div>",
                            unsafe_allow_html=True
                        )
                else:
                    st.error(f"API Error: {response.status_code}")
            except Exception as e:
                st.error(f"Error connecting to API: {e}")

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("""
    <hr>
    <p style="text-align:center; font-size:15px;">
    Built with ❤️ using <b>Streamlit</b> & <b>FastAPI</b><br>
    <span style="font-size:13px; color:gray;">© 2025 Nithin Chowdary | SMS Spam Detection</span>
    </p>
""", unsafe_allow_html=True)


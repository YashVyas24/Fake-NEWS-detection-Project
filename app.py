import streamlit as st
import joblib

vectorizer = joblib.load("vectorizer.jb")
model = joblib.load("lr_model.jb")

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #89f7fe, #66a6ff);
        background-size: cover;
        min-height: 100vh;
    }
    .check-button {
        display: flex;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style='text-align: center;'>
        <h1 style='color:#ff4b4b;'>📰 Fake News Detection System</h1>
        <p style='font-size:18px;'>Instantly check if a news article is <b>Real</b> or <b>Fake</b>.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("---")

# Sample news options: 2 real + 2 fake
sample_articles = [
    "",
    "As U.S. budget fight looms, Republicans flip their fiscal script. WASHINGTON (Reuters) - The head of a conservative Republican faction in the U.S. Congress, who voted this month for a huge expansion of the national debt to pay for tax cuts, called himself a 'fiscal conservative' on Sunday and urged budget restraint in 2018.",
    "U.S. military to accept transgender recruits on Monday: Pentagon. WASHINGTON (Reuters) - Transgender people will be allowed for the first time to enlist in the U.S. military starting on Monday as ordered by federal courts.",
    "Chaos Ensues After Man Accidentally Shoots Himself And Wife In Church During Gun Safety Talk. An 81-year-old churchgoer accidentally discharged his gun during a safety talk, critically injuring himself and his wife and causing panic in the congregation.",
    "New Accuser Confirms She Got Roy Moore Banned From The Mall. A woman claims Senate candidate Roy Moore was banned from the Gadsden Mall for harassing young girls, adding to his mounting allegations."
]

selected_sample = st.selectbox("Or pick a sample news article to try it out:", sample_articles)

default_text = selected_sample if selected_sample else ""

st.subheader("Enter a News Article 👇")
news_input = st.text_area(
    "Paste your news article here:",
    height=200,
    value=default_text,
    placeholder="Type or paste your news content..."
)

st.markdown("<div class='check-button'>", unsafe_allow_html=True)
if st.button("🔎 Check News"):
    if news_input.strip():
        with st.spinner("Analyzing the article..."):
            transform_input = vectorizer.transform([news_input])
            prediction = model.predict(transform_input)

        if prediction[0] == 1:
            st.success("✅ The news article appears to be **REAL**.")
        else:
            st.error("🚨 The news article appears to be **FAKE**.")
    else:
        st.warning("⚠️ Please enter some text to analyze.")
st.markdown("</div>", unsafe_allow_html=True)

st.write("---")
st.markdown(
    "<p style='text-align:center; font-size:12px;'>Made by yash ❤️</p>",
    unsafe_allow_html=True
)

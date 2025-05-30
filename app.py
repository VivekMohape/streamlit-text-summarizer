import streamlit as st
import requests

# --- CONFIG ---
st.set_page_config(page_title="AI Assistant", layout="centered")

# --- SESSION STATE ---
if "output" not in st.session_state:
    st.session_state.output = ""
if "feature" not in st.session_state:
    st.session_state.feature = "Text Summarizer"

# --- Model Mapping ---
model_map = {
    "LLaMA 3 (8B) 🧠": "llama3-8b-8192",
    "LLaMA 3 (70B) 🚀": "llama3-70b-8192",
    "Gemma2 (9B) 🩺": "gemma2-9b-it"
}

# --- Header ---
st.title("🧠 AI Assistant")
st.markdown("Choose a feature below to get started:")

# --- Feature Selector ---
feature = st.radio("Choose a Feature", ["Text Summarizer", "Medical Term Explainer"], index=["Text Summarizer", "Medical Term Explainer"].index(st.session_state.feature))
st.session_state.feature = feature

# --- Feature Hints ---
if feature == "Text Summarizer":
    st.info("This tool helps you summarize long articles, notes, or documents into key points.")
elif feature == "Medical Term Explainer":
    st.info("Paste medical reports or test results, and the assistant will explain them in layman's terms.")

# --- Shared Inputs ---
text_input = st.text_area("Enter Text", placeholder="Paste your content here...", height=200)
selected_model_name = st.selectbox("Select Model", list(model_map.keys()))
selected_model = model_map[selected_model_name]

# --- Optional Input ---
summary_length = 100  # Default
if feature == "Text Summarizer":
    summary_length = st.slider("Summary Length (approx words)", 30, 300, 100)

run_button = st.button("Run")

# --- Groq API Call ---
@st.cache_data(show_spinner=False)
def call_groq_api(feature, text, model, word_limit=100):
    headers = {
        "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }

    if feature == "Text Summarizer":
        prompt = f"Summarize the following text in approximately {word_limit} words:\n\n{text}"
    elif feature == "Medical Term Explainer":
        prompt = f"Explain this medical report in simple, layman terms. Be clear and patient-friendly:\n\n{text}"
    else:
        prompt = text  # fallback

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# --- Main Action ---
if run_button:
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner(f"Running {feature} with {selected_model_name}..."):
            try:
                output = call_groq_api(feature, text_input, selected_model, summary_length)
                st.session_state.output = output
                st.success("✅ Done!")
            except Exception as e:
                st.error(f"Groq API error: {e}")

# --- Output ---
if st.session_state.output:
    st.subheader("📝 Result")
    st.write(st.session_state.output)

# --- Footer ---
st.markdown("---")
st.caption("Built with ❤️ by Vivek Mohape")

import streamlit as st
import requests

# --- CONFIG ---
st.set_page_config(page_title="AI-Powered Summarizer", layout="centered")

# --- SESSION STATE ---
if "summary" not in st.session_state:
    st.session_state.summary = ""

# --- UI ---
st.title("🧠 AI-Powered Text Summarizer with Groq")
st.markdown("Paste your text below, choose a summary length, and click 'Summarize'.")

text_input = st.text_area("Enter Text", placeholder="Paste your text here...", height=200)
summary_length = st.slider("Summary Length (approx words)", 30, 300, 100)

# --- Model selector ---
selected_model_name = st.selectbox(
    "Select Model",
    ["LLaMA 3 (8B)", "LLaMA 3 (70B)", "Mixtral (8x7B)"]
)

# Map readable name to Groq model ID
model_map = {
    "LLaMA 3 (8B)": "llama3-8b-8192",
    "LLaMA 3 (70B)": "llama3-70b-8192",
    "Mixtral (8x7B)": "mixtral-8x7b-32768"
}
selected_model = model_map[selected_model_name]

summarize_button = st.button("Summarize")

# --- Groq API Call ---
@st.cache_data(show_spinner=False)
def call_groq_api(text, word_limit, model):
    headers = {
        "Authorization": f"Bearer {st.secrets['GROQ_API_KEY']}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert text summarizer."},
            {"role": "user", "content": f"Summarize the following text in approximately {word_limit} words:\n\n{text}"}
        ]
    }

    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

# --- Action ---
if summarize_button:
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner(f"Summarizing using {selected_model_name}..."):
            try:
                summary = call_groq_api(text_input, summary_length, selected_model)
                st.session_state.summary = summary
                st.success("✅ Summary complete!")
            except Exception as e:
                st.error(f"Groq API error: {e}")

# --- Output ---
if st.session_state.summary:
    st.subheader("📋 Summary")
    st.write(st.session_state.summary)

# --- Footer ---
st.markdown("---")
st.caption("Powered by Streamlit + Groq API")

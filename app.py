import streamlit as st
import requests
import instructor
from pydantic import BaseModel
import json
import openai  # Added for instructor client

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
    "Gemma2 (9B) 🦩": "gemma2-9b-it"
}

# --- Pydantic Schema for Structured Output ---
class UserInfo(BaseModel):
    name: str
    age: int

# --- Header ---
st.title("🧠 AI Assistant")
st.markdown("Choose a feature below to get started:")

# --- Feature Selector ---
feature = st.radio(
    "Choose a Feature",
    ["Text Summarizer", "Medical Term Explainer", "Structured Info Extractor"],
    index=["Text Summarizer", "Medical Term Explainer", "Structured Info Extractor"].index(st.session_state.feature)
)
st.session_state.feature = feature

# --- Feature Hints ---
if feature == "Text Summarizer":
    st.info("This tool helps you summarize long articles, notes, or documents into key points.")
elif feature == "Medical Term Explainer":
    st.info("Paste medical reports or test results, and the assistant will explain them in layman's terms.")
elif feature == "Structured Info Extractor":
    st.info("Provide sentences like 'John Doe is 30 years old.' to extract structured info.")

# --- Shared Inputs ---
text_input = st.text_area("Enter Text", placeholder="Paste your content here...", height=200)

if feature != "Structured Info Extractor":
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
    elif feature == "Structured Info Extractor":
        # fallback in case instructor is not used
        prompt = f"Extract the name and age from the following sentence and return it in JSON format like {{\"name\": string, \"age\": int}}:\n\n{text}"
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

# --- Setup instructor client with Groq API ---
openai.api_key = st.secrets["GROQ_API_KEY"]
openai.base_url = "https://api.groq.com/openai/v1"  # override for Groq endpoint
client = instructor.from_openai(openai)

# --- Main Action ---
if run_button:
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner(f"Running {feature}..."):
            try:
                if feature == "Structured Info Extractor":
                    try:
                        structured = client.chat.completions.create(
                            model="llama3-8b-8192",
                            response_model=UserInfo,
                            messages=[
                                {"role": "system", "content": "You are an extractor that converts sentences into structured data."},
                                {"role": "user", "content": text_input}
                            ]
                        )
                        st.session_state.output = structured.model_dump()
                    except Exception as parse_error:
                        st.session_state.output = f"❌ Parsing failed: {parse_error}"
                        st.warning("Could not parse structured output using instructor. Showing error message.")
                else:
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

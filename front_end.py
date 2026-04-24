import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Define paths to available models
MODEL_PATHS = {
    "Twitter RoBERTa Base": "twitter-roberta-base-irony/cv_sarcasm_optuna/final_saved_model",
    "Twitter RoBERTa Irony": "twitter-roberta-base-irony/cv_sarcasm_optuna/final_saved_model",
    "Bertweet Sarcoji": "model_saved/bertweet_sarcoji_finetuned"
}

@st.cache_resource
def load_model(model_name):
    """Loads the model and tokenizer, caching them so they aren't reloaded on every interaction."""
    path = MODEL_PATHS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path)
    model.eval()
    return tokenizer, model

def predict_sarcasm(text, tokenizer, model):
    """Runs inference on the provided text using the loaded model."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        prediction = torch.argmax(logits, dim=-1).item()
        
    # Assuming config.json has id2label mapping (0: not_sarcastic, 1: sarcastic)
    label = model.config.id2label[prediction]
    confidence = probabilities[0][prediction].item()
    return label, confidence

# --- Streamlit UI Configuration ---
st.set_page_config(page_title="Sarcasm Detector", page_icon="🤖", layout="centered")

# Sidebar for model selection
with st.sidebar:
    st.header("Settings")
    selected_model_name = st.selectbox("Choose a Model", list(MODEL_PATHS.keys()))
    st.markdown("---")
    st.markdown("""
    **About:**
    This interface analyzes your text and predicts whether it contains sarcasm/irony based on the selected fine-tuned model.
    """)

st.title("🤖 Sarcasm Detector AI")
st.caption("Chat with the model to test its sarcasm detection capabilities.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load the selected model
try:
    with st.spinner("Loading model..."):
        tokenizer, model = load_model(selected_model_name)
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model from {MODEL_PATHS[selected_model_name]}: {e}")
    model_loaded = False

# React to user input
if prompt := st.chat_input("Enter a message to analyze for sarcasm..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    if model_loaded:
        # Get prediction
        label, confidence = predict_sarcasm(prompt, tokenizer, model)
        
        # Format assistant response
        if "not" in label.lower() or label == "0":
            emoji = "😐"
            formatted_label = "Not Sarcastic"
        else:
            emoji = "🎭"
            formatted_label = "Sarcastic"

        response = f"**Prediction:** {formatted_label} {emoji}\n\n**Confidence:** {confidence:.2%}"

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

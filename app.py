import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch
from langchain_openai import ChatOpenAI
import base64

# Set up the sentiment analysis model and tokenizer
base_model_checkpoint = "distilbert-base-uncased"
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(base_model_checkpoint)

# Load the PEFT model with the saved LoRA layers
model = PeftModel.from_pretrained(base_model, "Model")

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize the ChatOpenAI instance for summarization using GPT-3.5
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # Specify the model you want to use
    base_url="https://api.avalai.ir/v1",  # Custom base URL for the API
    api_key="aa-ihTCi18jbmQdBLb9HiZKP2qAWRH860PvhnuH39pAPb4TAXgA"  # Your API key
)

# Function to encode image as base64
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Encode your background image
img = get_img_as_base64("desktop.jpg")

# CSS to apply the background image
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("data:image/jpg;base64,{img}");
background-size: cover;
background-position: top left;
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/jpg;base64,{img}");
background-position: center; 
background-repeat: no-repeat;
background-attachment: fixed;
}}

[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}

[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Title Section
st.markdown('<div style="background-color: rgba(49, 48, 49, 0.8); padding: 20px; border-radius: 10px; text-align: center; color: #f1c40f;"><h1>Sentiment Analysis and Summarization</h1></div>', unsafe_allow_html=True)

# Text input area with updated background color and text style
st.markdown('<div style="background-color: rgba(241, 196, 15, 0.8); padding: 20px; border-radius: 10px; color: #f1c40f; font-weight: bold;">', unsafe_allow_html=True)
text_input = st.text_area("", height=200)
st.markdown('</div>', unsafe_allow_html=True)

if st.button("Analyze and Summarize"):
    text_list = text_input.strip().split("\n")
    
    positives = []
    negatives = []

    # Step 1: Perform Sentiment Analysis and gather negative comments
    for index, text in enumerate(text_list):
        text = text.strip()  # Remove any leading/trailing whitespace
        if not text:  # Skip empty lines
            continue

        inputs = tokenizer.encode(text, return_tensors="pt").to(device)
        logits = model(inputs).logits
        predictions = torch.max(logits, 1).indices
        sentiment = "Negative" if predictions.tolist()[0] == 0 else "Positive"

        if sentiment == "Positive":
            positives.append(text)
        else:
            negatives.append(f"Customer {len(negatives) + 1}: {text}")

    # Combine all negative comments into a single block of text
    if negatives:
        negative_comments = "\n".join(negatives)
    else:
        negative_comments = "No negative comments."

    # Human-crafted summary as a placeholder
    #human_summary = "Customers are frustrated with the poor battery life, subpar camera quality, and issues with shipping and packaging, it has software issues and memory size is small, leading to an overall disappointing experience with the product."

    # Prompt creation for summarization
    prompt = f"""
    Summarize the following conversation in a more abstract way, focusing on the overall sentiment and key points.

    {negative_comments}

    Summary:
    
    """

    # Generate output from GPT-3.5 via Aval AI
    response = llm.invoke(prompt)
    output = response.content

    num_positive = len(positives)
    num_negative = len(negatives)

    # Display results in Streamlit with styled sections
    st.markdown(f"""
    <div style="background-color: rgba(118, 215, 196, 0.8); color: white; padding: 10px; border-radius: 5px; font-weight: bold;">
        <strong>Positive Comments:</strong> {num_positive}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color: rgba(236, 112, 99, 0.8); color: white; padding: 10px; border-radius: 5px; font-weight: bold;">
        <strong>Negative Comments:</strong> {num_negative}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color: rgba(44, 44, 44, 0.8); color: #f1c40f; padding: 10px; border-radius: 5px; font-weight: bold;">
        <strong>Negative Comments Detected:</strong><br>{negative_comments}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color: rgba(44, 44, 44, 0.8); color: #f1c40f; padding: 10px; border-radius: 5px; font-weight: bold;">
        <strong>Generated Summary:</strong><br>{output}
    </div>
    """, unsafe_allow_html=True)

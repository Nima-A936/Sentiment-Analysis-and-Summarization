import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch
from langchain_openai import ChatOpenAI
import base64
from PIL import Image

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
    model="gpt-3.5-turbo",
    base_url="https://api.avalai.ir/v1",
    api_key="aa-ihTCi18jbmQdBLb9HiZKP2qAWRH860PvhnuH39pAPb4TAXgA"
)

# Function to encode image as base64
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Encode your background image
img = get_img_as_base64("desktop.jpg")
image = Image.open("customer.png")

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
.image-container {{
    position: absolute;
    top: -90px;
    left: 50%;
    transform: translateX(-50%);
    z-index: 1;
}}

img.customer-image {{
    background: transparent;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown('<div class="image-container"><img src="data:image/png;base64,' + get_img_as_base64("customer.png") + '" alt="Customer Image" class="customer-image" width="130"></div></div>', unsafe_allow_html=True)


# Title Section with the new text added
st.markdown('<div style="background-color: rgba(49, 48, 49, 0.8); padding: 21px; border-radius: 10px; text-align: center; color: #f1c40f;"><h1>Customer Reviews Summarization</h1></div>', unsafe_allow_html=True)


# Text input area with updated background color, font style, and text size
st.markdown('<div style="background-color: rgba(241, 196, 15, 0.8); padding: 20px; border-radius: 10px; color: black; font-weight: bold; font-family: sans-serif; font-size: 16px;">Please enter your text in the area below</div>', unsafe_allow_html=True)
text_input = st.text_area("", height=200)

# Add a dropdown for context selection without "Customer Service"
context = st.selectbox(
    "Choose the context for summarization:",
    options=["Overall Sentiment", "Product Quality", "Delivery Experience"],
    format_func=lambda x: f"🔧 {x}" if x == "Product Quality" else (f"📦 {x}" if x == "Delivery Experience" else f"📝 {x}")
)

# Include the Google Font in the HTML
st.sidebar.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# README Section in the Sidebar with custom font and increased text size
# Include the Google Font in the HTML
st.sidebar.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# README Section in the Sidebar with custom font, increased text size, and added depth
st.sidebar.markdown('<div style="font-family: \'Lato\', sans-serif; font-size: 22px; color: #e74c3c ; font-weight: bold; text-shadow: 4px 4px 4px rgba(0, 0, 0, 0.9);">📖 README</div>', unsafe_allow_html=True)
st.sidebar.markdown("""
<div style="font-family: 'Lato', sans-serif; font-size: 18px; color: #f4d03f  ; line-height: 1.6; text-shadow: 3px 3px 3px rgba(241, 196, 15, 0.4); font-weight: 550;">
**Customer Reviews Summarization** is a web application that allows managers to analyze and summarize customer feedback. The app focuses on providing context-based summaries, allowing managers to concentrate on specific aspects like product quality or delivery experience.

### Features
- Context-based summarization.
- Sentiment analysis of customer feedback.
- Customizable interface with background images.

### Contributing
Feel free to submit pull requests or open issues to contribute to the project. Please make sure to follow the contribution guidelines.

</div>
""", unsafe_allow_html=True)




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

    total_comments = len(text_list)
    positive_percentage = (len(positives) / total_comments) * 100 if total_comments > 0 else 0


    # Modify the prompt based on the selected context
    if context == "Overall Sentiment":
        prompt_context = "Summarize the following conversation in a more abstract way, focusing on the overall sentiment and key points."
        summary_context = "Overall Sentiment Summary:"
        summary_label = "Overall Sentiment Summary:"
    elif context == "Product Quality":
        prompt_context = "Only summarize the quality issues, not delivery."
        summary_context = "Product Quality Summary:"
        summary_label = "Product Quality Summary:"
    elif context == "Delivery Experience":
        prompt_context = "Only summarize the delivery and packaging issues."
        summary_context = "Delivery Experience Summary:"
        summary_label = "Delivery Experience Summary:"

    # Combine the context with the negative comments
    prompt = f"""
    {prompt_context}

    {negative_comments}

    Summary:
    {summary_context}
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
    <div style="background-color: rgba(52, 152, 219 , 0.8); color: #f1c40f; padding: 10px; border-radius: 5px; font-weight: bold;">
        📊 Positive Feedback: {positive_percentage:.2f}%
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="background-color: rgba(44, 44, 44, 0.8); color: #f1c40f; padding: 15px; border-radius: 5px; font-weight: bold;">
        <strong>{summary_label}</strong><br>{output}
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background-color: rgba(44, 44, 44, 0.8); color: #f1c40f; padding: 15px; border-radius: 5px; font-weight: bold;">
        <strong>Negative Comments Detected:</strong><br>{negative_comments}
    </div>
    """, unsafe_allow_html=True)

import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch
from langchain_openai import ChatOpenAI
import base64
from PIL import Image
import os
import pandas as pd

# Set up the sentiment analysis model and tokenizer
base_model_checkpoint = "distilbert-base-uncased"
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(base_model_checkpoint)

# Load the PEFT model with the saved LoRA layers
model = PeftModel.from_pretrained(base_model, "Model")

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
api_key = os.getenv('API_KEY')

# Initialize the ChatOpenAI instance for summarization using GPT-3.5
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    base_url="https://api.avalai.ir/v1",
    api_key=api_key
)

# Function to encode image as base64
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Encode your background image
img = get_img_as_base64("Pictures/desktop.jpg")
#image = Image.open("customer.png")
img1 = get_img_as_base64("Pictures/Applications.png")
img2 = get_img_as_base64("Pictures/E-Commerce.png")

#sidebar_image1 = f"""
#    <div style="text-align: center;">
#        <img src="data:image/png;base64,{img1}" alt="E-commerce Image" style="width:300px;height:300px;">
#    </div>
#"""
#idebar_image2 = f"""
#    <div style="text-align: center;">
#      
#    </div>
#"""

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

.image-container1 {{
    position: absolute;
    top: -92px;
    left: -6%;
    transform: translateX(50%);
    z-index: 1;
}}

.image-container2 {{
    position: absolute;
    top: -108px;
    right: -6%;
    transform: translateX(-50%);
    z-index: 1;
}}

img.customer-image {{
    background: transparent;
}}

</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown('<div class="image-container"><img src="data:image/png;base64,' + get_img_as_base64("Pictures/customer.png") + '" alt="Customer Image" class="customer-image" width="130"></div></div>', unsafe_allow_html=True)

st.markdown('<div class="image-container1"><img src="data:image/png;base64,' + get_img_as_base64("Pictures/like_thumb.png") + '" alt="Like_thumb Image" class="like-image" width="65"></div></div>', unsafe_allow_html=True)

st.markdown('<div class="image-container2"><img src="data:image/png;base64,' + get_img_as_base64("Pictures/dislike_thumb.png") + '" alt="Dislike_thumb Image" class="Dislike-image" width="65"></div></div>', unsafe_allow_html=True)


# Title Section with the new text added
st.markdown('<div style="background-color: rgba(49, 48, 49, 0.8); padding: 21px; border-radius: 10px; text-align: center; color: #f1c40f;"><h1>Customer Reviews Summarization</h1></div>', unsafe_allow_html=True)


# Text input area with updated background color, font style, and text size
st.markdown('<div style="background-color: rgba(241, 196, 15, 0.8); padding: 20px; border-radius: 10px; color: black; font-weight: bold; font-family: sans-serif; font-size: 16px;">Please enter your text in the area below</div>', unsafe_allow_html=True)
text_input = st.text_area("Prompt Window", height=200)


# File Upload Section (Optional)
#st.markdown('<div style="color: white; font-weight: bold; margin-top: 20px;">Or, choose a CSV file containing customer reviews:</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a CSV file (Optional)", type="csv")

# Process the CSV file if uploaded
reviews = []
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("File uploaded successfully!")
    
    # Show the column names and allow the user to select the product column
    product_columns = df.columns.tolist()
    selected_product_column = st.selectbox("Select the Product ID column to analyze:", product_columns)
    
    if selected_product_column:
        # Extract all reviews under the selected product column
        reviews = df[selected_product_column].dropna().tolist()
        st.write(f"Found {len(reviews)} reviews for Product ID '{selected_product_column}'.")

# If no CSV is uploaded, use the manual input
if not reviews and text_input:
    reviews = text_input.split("\n")  # Split the manual input into lines

# Add a dropdown for context selection without "Customer Service"
context = st.selectbox(
    "Choose the context for summarization:",
    options=["Overall Sentiment", "Product Quality", "Delivery Experience", "Ongoing Concern"],
    format_func=lambda x: (f"🔧 {x}" if x == "Product Quality" 
                           else (f"📦 {x}" if x == "Delivery Experience" 
                           else (f"⚠️ {x}" if x == "Ongoing Concern" 
                           else f"📝 {x}")))
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
st.sidebar.markdown(f"""
<div style="font-family: 'Lato', sans-serif; font-size: 18px; color: #f4d03f; line-height: 1.6; text-shadow: 3px 3px 3px rgba(241, 196, 15, 0.4); font-weight: 550;">
<strong>Customer Reviews Summarization</strong> is a web application that allows managers to analyze and summarize customer feedback. The app focuses on providing context-based summaries, allowing managers to concentrate on specific aspects like product quality or delivery experience.

<h3>Features:</h3>
<ul>
<li>Context-based summarization.</li>
<li>Sentiment analysis of customer feedback.</li>
<li>CSV File Upload Option.</li>
<li>Customizable interface with background images.</li>
</ul>

<h3>Contributing:</h3>
Feel free to submit pull requests or open issues to contribute to the project. Please make sure to follow the contribution guidelines.

<h3>Applications:</h3>
This app is ideal for e-commerce businesses to identify common product quality issues, delivery challenges, and overall customer sentiment. 
With this tool, businesses can gain insights into areas that need improvement, enabling them to enhance their services, 
optimize product offerings, and boost customer satisfaction by focusing on what matters most to their consumers.
<br><img src="data:image/png;base64,{img2}" alt="Applications Image" style="width:300px;height:300px;">

<h3>Social Media:</h3>
This app is highly suitable for <strong> social network applications</strong> where businesses can leverage <strong>Customer Reviews Summarization</strong> to analyze and understand customer feedback at scale.
For <strong>social network applications</strong>, where many online shopping businesses are active, this tool allows companies to assess feedback from platforms like Instagram, Facebook, and other social media channels.
<br><img src='data:image/png;base64,{img1}' alt='E-commerce Image' style='width:300px;height:300px;'>
</div>
""", unsafe_allow_html=True)


#st.sidebar.markdown(sidebar_image, unsafe_allow_html=True)


if st.button("Analyze and Summarize"):
    if not reviews:
        st.error("No reviews to analyze. Please enter text or upload a CSV file.")
    else:
        positives = []
        negatives = []

        # Perform Sentiment Analysis and gather negative comments
        for text in reviews:
            text = str(text).strip()  # Convert to string and remove any leading/trailing whitespace
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

        total_comments = len(reviews)
        positive_percentage = (len(positives) / total_comments) * 100 if total_comments > 0 else 0

        # Modify the prompt based on the selected context
        if context == "Overall Sentiment":
            prompt_context = "Summarize the following conversation in a more abstract way, focusing on the overall sentiment."
            summary_context = "Overall Sentiment Summary:"
            summary_label = "Overall Sentiment Summary:"
        elif context == "Ongoing Concern":
            prompt_context = "of all the problems which on was the most repeated."
            summary_context = "Most Repeated Problem:"
            summary_label = "Ongoing Concern:"
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

import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch
from langchain_openai import ChatOpenAI
import base64
from PIL import Image
import os
import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Set up the sentiment analysis model and tokenizer
base_model_checkpoint = "distilbert-base-uncased"
base_model = AutoModelForSequenceClassification.from_pretrained(base_model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(base_model_checkpoint)

# Load the PEFT model with the saved LoRA layers
model = PeftModel.from_pretrained(base_model, "Model")

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
load_dotenv()
api_key = os.getenv('API_KEY')

# Initialize the ChatOpenAI instance for summarization using GPT-3.5
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    base_url="https://api.avalai.ir/v1",
    api_key="aa-gAp2CUy0mkUSBkHkJ8HpVNnVW099QOZgkKG99LN8gpxc5fwT"
)

# Function to encode image as base64
@st.cache_data
def get_img_as_base64(file):
    with open(file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Display average rating and review counts
def display_html_ratings(avg_rating, review_counts):
    html_code = f'''
    <span class="fa fa-star {'checked' if avg_rating >= 1 else ''}"></span>
    <span class="fa fa-star {'checked' if avg_rating >= 2 else ''}"></span>
    <span class="fa fa-star {'checked' if avg_rating >= 3 else ''}"></span>
    <span class="fa fa-star {'checked' if avg_rating >= 4 else ''}"></span>
    <span class="fa fa-star {'checked' if avg_rating >= 5 else ''}"></span>

    <div class="row">
      <div class="side"><div>5 star</div></div>
      <div class="middle"><div class="bar-container"><div class="bar-5" style="width:{(review_counts[5]/sum(review_counts.values()))*100}%"></div></div></div>
      <div class="side right"><div>{review_counts[5]}</div></div>

      <div class="side"><div>4 star</div></div>
      <div class="middle"><div class="bar-container"><div class="bar-4" style="width:{(review_counts[4]/sum(review_counts.values()))*100}%"></div></div></div>
      <div class="side right"><div>{review_counts[4]}</div></div>

      <div class="side"><div>3 star</div></div>
      <div class="middle"><div class="bar-container"><div class="bar-3" style="width:{(review_counts[3]/sum(review_counts.values()))*100}%"></div></div></div>
      <div class="side right"><div>{review_counts[3]}</div></div>

      <div class="side"><div>2 star</div></div>
      <div class="middle"><div class="bar-container"><div class="bar-2" style="width:{(review_counts[2]/sum(review_counts.values()))*100}%"></div></div></div>
      <div class="side right"><div>{review_counts[2]}</div></div>

      <div class="side"><div>1 star</div></div>
      <div class="middle"><div class="bar-container"><div class="bar-1" style="width:{(review_counts[1]/sum(review_counts.values()))*100}%"></div></div></div>
      <div class="side right"><div>{review_counts[1]}</div></div>
      <div style="text-align: center;">
        <div style="font-size: 2em; font-weight: bold; color: gold;">{avg_rating:.2f} / 5</div>
        <div class="stars-outer" style="position: relative; display: inline-block; font-size: 3em; color: lightgray;">
            <div class="stars-inner" style="position: absolute; top: 0; left: 0; white-space: nowrap; overflow: hidden; color: gold; width: {avg_rating / 5 * 100}%;">
                &#9733;&#9733;&#9733;&#9733;&#9733;
            </div>
            &#9733;&#9733;&#9733;&#9733;&#9733;
        </div>
    </div>
    <style>
    .fa {{ font-size: 25px; }}
    .checked {{ color: orange; }}
    .side {{ float: left; width: 15%; margin-top:10px; }}
    .middle {{ float: left; width: 70%; margin-top:10px; }}
    .right {{ text-align: right; }}
    .row:after {{ content: ""; display: table; clear: both; }}
    .bar-container {{ width: 100%; background-color: #313031; text-align: center; color: #313031; border-radius: 20px }}
    .bar-5 {{ height: 18px; background-color: #f1c40f; border-radius: 20px }}
    .bar-4 {{ height: 18px; background-color: #f1c40f; border-radius: 20px }}
    .bar-3 {{ height: 18px; background-color: #f1c40f; border-radius: 20px }}
    .bar-2 {{ height: 18px; background-color: #f1c40f; border-radius: 20px }}
    .bar-1 {{ height: 18px; background-color: #f1c40f; border-radius: 20px }}
    </style>
    '''
    st.markdown(html_code, unsafe_allow_html=True)


def calculate_average_rating(stars):
    if stars:
        return sum(stars) / len(stars)
    return 0


# Encode your Images
img = get_img_as_base64("Pictures/desktop.jpg")
img1 = get_img_as_base64("Pictures/Applications.png")
img2 = get_img_as_base64("Pictures/E-Commerce.png")



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
    top: -45px;
    left: -6%;
    transform: translateX(50%);
    z-index: 1;
}}

.image-container2 {{
    position: absolute;
    top: -61px;
    right: -6%;
    transform: translateX(-50%);
    z-index: 1;
}}

img.customer-image {{
    background: transparent;
}}

</style>
"""
#Padding the upload file section
css = '''
<style>
/* Target the file uploader text to replace "Drag and drop file here" */
[data-testid="stFileUploadDropzone"]::before {
    content: "Upload Your Custom Text Here"; /* Replace with your own text */
    font-size: 100px;
    color: white;
    font-weight: bold;
}

/* Optional: Adjust the style for the whole file uploader */
[data-testid="stFileUploader"] {
    background-color: #111; /* Change background if needed */
    padding: 30px;
    border-radius: 10px;
    font-size: 20px
    color: white
}
</style>
'''



# Inject the custom CSS into the Streamlit app
st.markdown(css, unsafe_allow_html=True)

st.markdown(page_bg_img, unsafe_allow_html=True)

st.markdown('<div class="image-container"><img src="data:image/png;base64,' + get_img_as_base64("Pictures/customer.png") + '" alt="Customer Image" class="customer-image" width="130"></div></div>', unsafe_allow_html=True)

st.markdown('<div class="image-container1"><img src="data:image/png;base64,' + get_img_as_base64("Pictures/like_thumb.png") + '" alt="Like_thumb Image" class="like-image" width="65"></div></div>', unsafe_allow_html=True)

st.markdown('<div class="image-container2"><img src="data:image/png;base64,' + get_img_as_base64("Pictures/dislike_thumb.png") + '" alt="Dislike_thumb Image" class="Dislike-image" width="65"></div></div>', unsafe_allow_html=True)

st.markdown('<div style="background-color: rgba(49, 48, 49, 0.8); padding: 21px; border-radius: 10px; text-align: center; color: #f1c40f;"><h1>Customer Reviews Summarization</h1></div>', unsafe_allow_html=True)


# Text input area with updated background color, font style, and text size
st.markdown('<div style="background-color: rgba(241, 196, 15, 0.8); padding: 20px; border-radius: 10px; color: black; font-weight: bold; font-family: sans-serif; font-size: 16px;">Please enter your text in the area below</div>', unsafe_allow_html=True)
text_input = st.text_area("Prompt Window", height=200)

col1, col2  = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Choose a CSV file for summarization (Optional)", type="csv")
    if uploaded_file is not None:
        st.write(f"File {uploaded_file.name} uploaded successfully!")
        
with col2:
    uploaded_stars = st.file_uploader("Choose a CSV file for task ratings (Optional2)", type="csv")
    if uploaded_file is not None:
        st.write(f"File {uploaded_file.name} uploaded successfully!")

reviews = []
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    product_columns = df.columns.tolist()
    selected_product_column = st.selectbox("Select the Product ID column to analyze:", product_columns)
    
    if selected_product_column:
        reviews = df[selected_product_column].dropna().tolist()
        st.write(f"Found {len(reviews)} reviews for Product ID '{selected_product_column}'.")

# Process the stars CSV if uploaded
stars = []
if uploaded_stars is not None:
    # Read the CSV file
    stars_df = pd.read_csv(uploaded_stars)
    selected_stars_column = st.selectbox("Select the Star Rating column:", stars_df.columns.tolist())
    
    if selected_stars_column:
        stars = stars_df[selected_stars_column].dropna().astype(int).tolist()
        avg_rating = calculate_average_rating(stars)

        star_counts = pd.Series(stars).value_counts().sort_index().to_dict()

        for i in range(1, 6):
            if i not in star_counts:
                star_counts[i] = 0

        display_html_ratings(avg_rating, star_counts)

        # Add the "Percentage Chart" button after the rating results
        if st.button("Percentage Chart"):
            if stars:
                def summary_plot(star_ratings):
                    plt.style.use("dark_background")
                    summary_rating = pd.Series(star_ratings).value_counts().sort_index()
                    total_ratings = sum(summary_rating.values)
                    star_percentage = (summary_rating / total_ratings) * 100
                    plt.clf()
                    fig, ax = plt.subplots()
                    bars = plt.bar(star_percentage.index, star_percentage.values, color="gold", width=0.08, edgecolor="black")

                    for bar in bars:
                        bar_width = bar.get_width()
                        bar_height = bar.get_height()
                        ax.add_patch(FancyBboxPatch(
                            (bar.get_x(), bar.get_y()),
                            bar_width,
                            bar_height,
                            boxstyle="round,pad=0.3",
                            edgecolor="black",
                            mutation_scale=0.87,
                            mutation_aspect=15,
                            facecolor=bar.get_facecolor(),
                        ))
                        ax.text(
                            bar.get_x() + bar.get_width() / 2,
                            bar.get_height() - 4,
                            f'{bar_height:.2f}%',
                            horizontalalignment='center',
                            color="black",
                            weight='bold',
                            va='bottom',
                            fontsize=8.4,
                            rotation=45
                        )

                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.spines['bottom'].set_visible(False)
                    plt.xticks(rotation=45, ha='right')
                    plt.ylabel("Percentage %")
                    plt.xlabel("Star Rating")
                    plt.title("Star Rating Distribution")
                    return fig

                st.pyplot(summary_plot(stars))



        
# Check if the stars file is not uploaded
if uploaded_stars is None:
    if not reviews and text_input:
        reviews = text_input.split("\n") 

    # Add a dropdown for context selection without "Customer Service"
    context = st.selectbox(
        "Choose the context for summarization:",
        options=["Overall Sentiment", "Product Quality", "Delivery Experience", "Ongoing Concern"],
        format_func=lambda x: (f"🔧 {x}" if x == "Product Quality" 
                               else (f"📦 {x}" if x == "Delivery Experience" 
                               else (f"⚠️ {x}" if x == "Ongoing Concern" 
                               else f"📝 {x}")))
    )

    if st.button("Analyze and Summarize"):
        if not reviews:
            st.error("No reviews to analyze. Please enter text or upload a CSV file.")
        else:
            positives = []
            negatives = []

            # Perform Sentiment Analysis and gather negative comments
            for text in reviews:
                text = str(text).strip()
                if not text: 
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
                prompt_context = "Only summarize the delivery and packaging issues , do not mention quality issues."
                summary_context = "Only late arrival,damaged packing."
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

        
# README Section in the Sidebar with custom font and increased text size
st.sidebar.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

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

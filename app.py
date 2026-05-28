import streamlit as st
from predict import predict_sentiment

# Configure the Streamlit page
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎭")

st.markdown("""
<style>
    /* 1. Dark premium background, 3. text color, 10. dark theme */
    .stApp {
        background-color: #0f0f0f;
    }
    /* General text */
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: #e5e5e5 !important;
    }
    /* 9. Sidebar */
    [data-testid="stSidebar"] {
        background-color: #121212 !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #7c3aed !important;
    }
    /* 4. Text area */
    .stTextArea textarea {
        background-color: #1a1a1a !important;
        color: white !important;
        border-radius: 12px !important;
        border: 1px solid #333 !important;
    }
    .stTextArea textarea:focus {
        border-color: #7c3aed !important;
        box-shadow: 0 0 0 1px #7c3aed !important;
    }
    /* 5. Analyze button */
    .stButton > button {
        width: 100%;
        background-color: #7c3aed !important;
        color: white !important;
        font-weight: bold !important;
        padding: 14px !important;
        border-radius: 8px !important;
        border: none !important;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #8b5cf6 !important;
        color: white !important;
        border-color: #8b5cf6 !important;
    }
    /* Sidebar example buttons styling */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #1a1a1a !important;
        border: 1px solid #7c3aed !important;
        color: white !important;
        padding: 8px !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #2a2a2a !important;
    }
    /* 6. Sentiment badge animation */
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }
    .sentiment-badge {
        text-align: center;
        padding: 15px 40px;
        border-radius: 50px;
        font-weight: bold;
        font-size: 28px;
        margin: 20px auto;
        display: block;
        width: fit-content;
        animation: popIn 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards;
    }
    .badge-positive {
        background-color: #16a34a;
        color: #bbf7d0 !important;
    }
    .badge-negative {
        background-color: #dc2626;
        color: #fecaca !important;
    }
    /* 7. Confidence bar */
    .confidence-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-top: 10px;
    }
    .confidence-track {
        background-color: #1a1a1a;
        border-radius: 10px;
        flex-grow: 1;
        height: 12px;
        overflow: hidden;
        margin-right: 15px;
    }
    .confidence-fill {
        background-color: #7c3aed;
        height: 100%;
        border-radius: 10px;
        transition: width 1s ease-in-out;
    }
    .confidence-text {
        font-weight: bold;
        color: #e5e5e5;
        min-width: 50px;
        text-align: right;
    }
    /* 8. Score breakdown cards */
    .score-card {
        background-color: #1a1a1a;
        border-radius: 12px;
        padding: 20px;
        border-left: 4px solid #7c3aed;
        text-align: center;
    }
    .score-card h3 {
        margin: 0;
        font-size: 16px;
        color: #a3a3a3 !important;
        margin-bottom: 10px;
    }
    .score-card p {
        margin: 0;
        font-size: 28px;
        font-weight: bold;
        color: white !important;
    }
    /* Custom divider */
    .custom-divider {
        height: 1px;
        background-color: #7c3aed;
        margin: 30px 0;
        opacity: 0.3;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for text area
if "user_text" not in st.session_state:
    st.session_state.user_text = ""

def set_example_text(text):
    st.session_state.user_text = text

with st.sidebar:
    st.title("About")
    st.write(
        "BERT (Bidirectional Encoder Representations from Transformers) is a powerful language model developed by Google. "
        "It reads text bidirectionally to understand the context of words in a sentence, making it highly effective for natural language tasks like sentiment analysis."
    )
    
    st.write("---")
    st.write("### Try these examples")
    
    st.button("The movie was absolutely fantastic!", on_click=set_example_text, args=("The movie was absolutely fantastic!",))
    st.button("Terrible experience, would not recommend.", on_click=set_example_text, args=("Terrible experience, would not recommend.",))
    st.button("It was okay, nothing special.", on_click=set_example_text, args=("It was okay, nothing special.",))

# App Header
st.title("Sentiment Analyzer 🎬")
st.subheader("Fine-tuned BERT on IMDB Movie Reviews")

# Input area
text_input = st.text_area(
    "Input text", 
    key="user_text",
    placeholder="Paste any text here...", 
    height=150, 
    label_visibility="collapsed"
)

# Analyze Button
if st.button("Analyze", type="primary"):
    if not text_input.strip():
        # Handle empty edge case
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing sentiment..."):
            # Run prediction
            result = predict_sentiment(text_input)
            
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            
            label = result["label"]
            confidence = result["confidence"]
            scores = result["scores"]
            
            # Big colored badge for the predicted label
            if label == "positive":
                st.markdown(
                    "<div class='sentiment-badge badge-positive'>POSITIVE</div>", 
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    "<div class='sentiment-badge badge-negative'>NEGATIVE</div>", 
                    unsafe_allow_html=True
                )
                
            # Confidence progress bar
            st.write("**Confidence**")
            st.markdown(
                f"""
                <div class='confidence-container'>
                    <div class='confidence-track'>
                        <div class='confidence-fill' style='width: {confidence:.2f}%;'></div>
                    </div>
                    <div class='confidence-text'>{confidence:.2f}%</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            st.markdown("<div class='custom-divider'></div>", unsafe_allow_html=True)
            st.write("### Score Breakdown")
            
            # Two-column score breakdown
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"""
                    <div class='score-card'>
                        <h3>Negative</h3>
                        <p>{scores['negative']:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col2:
                st.markdown(
                    f"""
                    <div class='score-card'>
                        <h3>Positive</h3>
                        <p>{scores['positive']:.2f}%</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# Light gray info box
st.markdown(
    """
    <div style="background-color: #1a1a1a; padding: 15px; border-radius: 10px; margin-top: 50px; border: 1px solid #333;">
        <p style="margin: 0; color: #a3a3a3; font-size: 14px;">
            <b>Note:</b> Model: bert-base-uncased fine-tuned on IMDB Movie Reviews dataset
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

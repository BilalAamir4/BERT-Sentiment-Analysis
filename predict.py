import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Conditionally import streamlit to allow running as a standalone script without issues
try:
    import streamlit as st
    cache_decorator = st.cache_resource
except ImportError:
    def cache_decorator(func):
        return func

@cache_decorator
def load_model_and_tokenizer(model_path="./saved_model"):
    """Loads the model and tokenizer from the specified path and caches them."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def predict_sentiment(text: str) -> dict:
    """
    Predicts the sentiment of the input text using the fine-tuned BERT model.
    """
    model, tokenizer = load_model_and_tokenizer()
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    # Run model inference
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits
    # Apply softmax to get probabilities
    probabilities = torch.softmax(logits, dim=-1).squeeze()
    
    # Handle single element or batched properly
    if probabilities.dim() == 0:
        prob_neg = 0.0
        prob_pos = probabilities.item()
    else:
        prob_neg = probabilities[0].item()
        prob_pos = probabilities[1].item()
        
    confidence = max(prob_neg, prob_pos) * 100
    label = "positive" if prob_pos > prob_neg else "negative"
    
    return {
        "label": label,
        "confidence": confidence,
        "scores": {
            "negative": prob_neg * 100,
            "positive": prob_pos * 100
        }
    }

if __name__ == "__main__":
    # Test the function with some sample sentences
    sentences = [
        "This movie is absolutely wonderful, I loved it!",
        "What a terrible and boring experience. Do not recommend.",
        "It was okay, nothing special but not bad either."
    ]
    
    print("Testing Sentiment Analyzer...\n")
    for s in sentences:
        result = predict_sentiment(s)
        print(f"Text: '{s}'")
        print(f"Result: {result}\n")

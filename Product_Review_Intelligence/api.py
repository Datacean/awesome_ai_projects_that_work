"""
Product Review Intelligence System
Uses pretrained models to analyze product reviews for sentiment, topics, and urgency
"""

# First, install required packages:
# pip install transformers datasets fastapi uvicorn torch

from transformers import pipeline
from datasets import load_dataset
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn

# ============================================================================
# PART 1: Load Models (do this once at startup)
# ============================================================================

print("Loading models... this may take a minute...")

# Sentiment Analysis - using a robust model
sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Zero-shot classification for topics/aspects
topic_classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Define the aspects/topics we want to extract
TOPICS = ["product quality", "shipping and delivery", "price and value", "customer service", "packaging"]

# For urgency detection, we'll use the same zero-shot classifier
URGENCY_LABELS = ["urgent response needed", "routine feedback"]

print("Models loaded successfully!")

# ============================================================================
# PART 2: Analysis Functions
# ============================================================================

def analyze_review(review_text: str) -> Dict:
    """
    Analyze a single review for sentiment, topics, and urgency
    """
    # Sentiment
    sentiment_result = sentiment_analyzer(review_text[:512])[0]  # Truncate to model max length
    sentiment = sentiment_result['label'].lower()
    sentiment_score = sentiment_result['score']
    
    # Topics/Aspects mentioned
    topic_result = topic_classifier(
        review_text[:512],
        candidate_labels=TOPICS,
        multi_label=True
    )
    
    # Filter topics with score > 0.5 (reasonably confident)
    relevant_topics = [
        {"topic": label, "confidence": score}
        for label, score in zip(topic_result['labels'], topic_result['scores'])
        if score > 0.5
    ]
    
    # Urgency detection
    urgency_result = topic_classifier(
        review_text[:512],
        candidate_labels=URGENCY_LABELS,
        multi_label=False
    )
    
    needs_response = urgency_result['labels'][0] == "urgent response needed"
    urgency_score = urgency_result['scores'][0] if needs_response else 1 - urgency_result['scores'][0]
    
    return {
        "sentiment": sentiment,
        "sentiment_confidence": round(sentiment_score, 3),
        "topics": relevant_topics,
        "needs_urgent_response": needs_response,
        "urgency_confidence": round(urgency_score, 3),
        "review_text": review_text
    }

# ============================================================================
# PART 3: Test with Amazon Polarity Dataset
# ============================================================================

def test_with_dataset(num_samples: int = 5):
    """
    Load and analyze some reviews from Amazon Polarity dataset
    """
    print(f"\nLoading {num_samples} reviews from Amazon Polarity dataset...")
    
    # Load dataset (this will download it first time)
    dataset = load_dataset("amazon_polarity", split="test", streaming=True)
    
    results = []
    for i, example in enumerate(dataset):
        if i >= num_samples:
            break
        
        review_text = example['content']
        print(f"\n--- Review {i+1} ---")
        print(f"Original Label: {example['label']} (0=negative, 1=positive)")
        print(f"Review: {review_text[:200]}...")
        
        analysis = analyze_review(review_text)
        results.append(analysis)
        
        print(f"Detected Sentiment: {analysis['sentiment']} ({analysis['sentiment_confidence']})")
        print(f"Topics: {[t['topic'] for t in analysis['topics']]}")
        print(f"Needs Response: {analysis['needs_urgent_response']}")
    
    return results

# ============================================================================
# PART 4: FastAPI Application
# ============================================================================

app = FastAPI(
    title="Product Review Intelligence API",
    description="Analyze product reviews for sentiment, topics, and urgency",
    version="1.0.0"
)

class ReviewRequest(BaseModel):
    review_text: str

class ReviewResponse(BaseModel):
    sentiment: str
    sentiment_confidence: float
    topics: List[Dict[str, float]]
    needs_urgent_response: bool
    urgency_confidence: float

class BatchReviewRequest(BaseModel):
    reviews: List[str]

@app.get("/")
def read_root():
    return {
        "message": "Product Review Intelligence API",
        "endpoints": {
            "/analyze": "POST - Analyze a single review",
            "/analyze-batch": "POST - Analyze multiple reviews",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": True}

@app.post("/analyze", response_model=ReviewResponse)
def analyze_single_review(request: ReviewRequest):
    """
    Analyze a single product review
    """
    try:
        if not request.review_text or len(request.review_text.strip()) == 0:
            raise HTTPException(status_code=400, detail="Review text cannot be empty")
        
        result = analyze_review(request.review_text)
        
        return ReviewResponse(
            sentiment=result['sentiment'],
            sentiment_confidence=result['sentiment_confidence'],
            topics=result['topics'],
            needs_urgent_response=result['needs_urgent_response'],
            urgency_confidence=result['urgency_confidence']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-batch")
def analyze_batch_reviews(request: BatchReviewRequest):
    """
    Analyze multiple product reviews at once
    """
    try:
        if not request.reviews or len(request.reviews) == 0:
            raise HTTPException(status_code=400, detail="Reviews list cannot be empty")
        
        if len(request.reviews) > 50:
            raise HTTPException(status_code=400, detail="Maximum 50 reviews per batch")
        
        results = []
        for review_text in request.reviews:
            if review_text and len(review_text.strip()) > 0:
                result = analyze_review(review_text)
                results.append(result)
        
        return {"results": results, "count": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# PART 5: Run Everything
# ============================================================================

if __name__ == "__main__":
    # Test with dataset first
    print("\n" + "="*80)
    print("TESTING WITH AMAZON POLARITY DATASET")
    print("="*80)
    test_results = test_with_dataset(num_samples=3)
    
    print("\n" + "="*80)
    print("STARTING FASTAPI SERVER")
    print("="*80)
    print("\nAPI will be available at: http://localhost:8000")
    print("Interactive docs at: http://localhost:8000/docs")
    print("\nExample curl command:")
    print('''
curl -X POST "http://localhost:8000/analyze" \\
  -H "Content-Type: application/json" \\
  -d '{"review_text": "Great product but shipping took forever! Really disappointed with delivery time."}'
    ''')
    
    # Start the API server
    uvicorn.run(app, host="0.0.0.0", port=8000)
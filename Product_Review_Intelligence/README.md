# 🎯 Product Review Intelligence System

AI-powered system to automatically analyze product reviews for sentiment, topics, and urgency using Hugging Face transformers and FastAPI.

## What It Does

- **😊😞 Sentiment Analysis**: Detects if reviews are positive or negative
- **🏷️ Topic Detection**: Identifies what customers talk about (quality, shipping, price, etc.)
- **⚡ Urgency Detection**: Flags reviews that need immediate attention
- **🚀 REST API**: Easy integration with any application

## Quick Start

### 1. Install Dependencies

```bash
uv venv datacean --python 3.12
source datacean/bin/activate
uv pip install -r requirements.txt
```

### 2. Run the API

```bash
python api.py
```

The API will be available at `http://localhost:8000` with interactive docs at `http://localhost:8000/docs`

### 3. Test It

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"review_text": "Great product but shipping was slow!"}'
```

## Learn More

📓 **For a complete step-by-step guide**, open [Product_Review.ipynb](Product_Review.ipynb) which explains:
- How the AI models work
- Detailed examples and testing
- How to customize and extend the system
- Performance optimization tips
- Real-world use cases

## Models Used

- **DistilBERT** - Fast sentiment analysis (positive/negative)
- **BART** - Flexible zero-shot classification for topics and urgency

## Example Response

```json
{
  "sentiment": "positive",
  "sentiment_confidence": 0.89,
  "topics": [
    {"topic": "product quality", "confidence": 0.92},
    {"topic": "shipping and delivery", "confidence": 0.78}
  ],
  "needs_urgent_response": false,
  "urgency_confidence": 0.85
}
```

---

**Note**: Models will download automatically on first run (~500MB). See the notebook for detailed documentation.

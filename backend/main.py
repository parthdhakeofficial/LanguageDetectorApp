from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pickle
import os

app = FastAPI()

LANGUAGE_MAP = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "sw": "Swahili",
    "th": "Thai",
    "tr": "Turkish",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese"
}
# CORS Configuration
origins = [
    "http://localhost:3000",  
    "http://127.0.0.1:3000",
    "*",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Load models safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_model(path):
    try:
        return pickle.load(open(path, "rb"))
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {path} -> {e}")

nb_model = load_model(os.path.join(BASE_DIR, "models/model_nb.pkl"))
lr_tfidf = load_model(os.path.join(BASE_DIR, "models/model_tfidf.pkl"))
lr_ngram = load_model(os.path.join(BASE_DIR, "models/model_ngram.pkl"))

ngram_vectorizer = load_model(os.path.join(BASE_DIR, "models/ngram_vectorizer.pkl"))
tfidf_vectorizer = load_model(os.path.join(BASE_DIR, "models/tfidf_vectorizer.pkl"))

# 📦 Request Schema
class PredictRequest(BaseModel):
    text: str
    model: str

# 🧠 Prediction logic
def predict(text: str, model_name: str):
    text = [text]

    if model_name == "nb":
        vec = ngram_vectorizer.transform(text)
        return nb_model.predict(vec)[0]

    elif model_name == "tfidf":
        vec = tfidf_vectorizer.transform(text)
        return lr_tfidf.predict(vec)[0]

    elif model_name == "ngram":
        vec = ngram_vectorizer.transform(text)
        return lr_ngram.predict(vec)[0]

    else:
        raise ValueError("Invalid model")

# ✅ Route 1: Health check
@app.get("/")
def home():
    return {"message": "API is working 🚀"}

@app.get("/languages")
def get_languages():
    return LANGUAGE_MAP
# ✅ Route 2: Get models
@app.get("/models")
def get_models():
    return {"models": ["nb", "tfidf", "ngram"]}

# ✅ Route 3: Prediction
@app.post("/predict")
def predict_api(request: PredictRequest):
    try:
        prediction_code = predict(request.text, request.model)
        prediction_name = LANGUAGE_MAP.get(prediction_code, "Unknown")

        return {
            "model": request.model,
            "text": request.text,
            "prediction_code": prediction_code,
            "prediction": prediction_name
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# evaluate.py
import os
import mlflow
from fastapi import FastAPI, Form
from typing import Optional
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import accuracy_score
from huggingface_hub import login

app = FastAPI()

# Set the MLflow tracking URI
mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-tracking-service:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

@app.post("/evaluate")
def evaluate(
    hf_token: Optional[str] = Form(None),
    model_dir: str = Form("/model"),
    dataset_name: str = Form("imdb"),
    split: str = Form("test[:200]"),
    max_length: int = Form(256)
):
    """
    Evaluate the model and log metrics to MLflow.

    Parameters:
    - hf_token (str): Hugging Face API token.
    - model_dir (str): Path to the saved model directory.
    - dataset_name (str): Name of the dataset to evaluate on.
    - split (str): Dataset split to use.
    - max_length (int): Maximum length for text generation.
    """
    # Login to Hugging Face if token is provided
    if hf_token:
        login(token=hf_token)

    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_dir, use_auth_token=hf_token).eval()

    # Load the dataset
    dataset = load_dataset(dataset_name, split=split, use_auth_token=hf_token)
    texts = dataset['text']
    labels = dataset['label']

    # Initialize predictions
    predictions = []

    # Evaluate model
    with mlflow.start_run(run_name="Model-Evaluation"):
        # Log parameters
        mlflow.log_params({
            "model_dir": model_dir,
            "dataset_name": dataset_name,
            "split": split,
            "max_length": max_length
        })

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            outputs = model.generate(**inputs, max_length=max_length)
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Simple heuristic for sentiment analysis
            if "negative" in pred_text.lower():
                pred_label = 0
            elif "positive" in pred_text.lower():
                pred_label = 1
            else:
                pred_label = 1  # Default to positive

            predictions.append(pred_label)

        # Calculate accuracy
        accuracy = accuracy_score(labels, predictions)
        mlflow.log_metric("accuracy", accuracy)

    return {"accuracy": accuracy}
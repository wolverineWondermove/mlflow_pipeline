import os
import mlflow
from typing import Optional
from huggingface_hub import login
from fastapi import FastAPI, Form
from datasets import load_dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM

app = FastAPI()

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-tracking-service:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)

@app.post("/train")
def train(
    hf_token: Optional[str] = Form(None),
    model_name: str = Form("gpt2"),
    data_path: str = Form("/data/train_data.json"),
    output_dir: str = Form("/model"),
    per_device_train_batch_size: int = Form(4),
    num_train_epochs: int = Form(1),
    learning_rate: float = Form(5e-5)
):
    """
    Train a model with the specified parameters.

    Parameters:
    - hf_token (str): Hugging Face API token.
    - model_name (str): Name or path of the pretrained model.
    - data_path (str): Path to the training data.
    - output_dir (str): Directory to save the trained model.
    - per_device_train_batch_size (int): Batch size per device.
    - num_train_epochs (int): Number of training epochs.
    - learning_rate (float): Learning rate.
    """

    if hf_token:
        login(token=hf_token)

    dataset = load_dataset('json', data_files=data_path)['train']

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        save_steps=10,
        logging_steps=10,
        evaluation_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset
    )

    with mlflow.start_run(run_name="Model-Training"):
        mlflow.log_params({
            "model_name": model_name,
            "data_path": data_path,
            "output_dir": output_dir,
            "per_device_train_batch_size": per_device_train_batch_size,
            "num_train_epochs": num_train_epochs,
            "learning_rate": learning_rate
        })

        trainer.train()

        os.makedirs(output_dir, exist_ok=True)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        mlflow.log_artifacts(output_dir)

    return {"status": "Model Trained", "output_dir": output_dir}
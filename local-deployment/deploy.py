import os
from fastapi import FastAPI, Form
from huggingface_hub import login, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI()

model_dir = os.getenv("MODEL_DIR", "/model")
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir).eval()

@app.post("/predict")
def predict(
    input_text: str = Form(...),
    max_length: int = Form(256),
    temperature: float = Form(1.0),
    num_return_sequences: int = Form(1),
    top_p: float = Form(0.9),
    top_k: int = Form(50),
    do_sample: bool = Form(True)
):
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    outputs = model.generate(
        inputs, max_length=max_length, temperature=temperature,
        num_return_sequences=num_return_sequences, top_p=top_p,
        top_k=top_k, do_sample=do_sample
    )
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return {"generated_texts": generated_texts}

@app.post("/upload_to_huggingface")
def upload_to_huggingface(
    hf_repo_id: str = Form(...),
    hf_token: str = Form(...),
    model_dir: str = Form("/model")
):
    login(token=hf_token)
    api = HfApi()
    api.upload_folder(
        folder_path=model_dir,
        path_in_repo="",
        repo_id=hf_repo_id,
        repo_type="model"
    )

    return {"status": "uploaded", "repo_id": hf_repo_id}
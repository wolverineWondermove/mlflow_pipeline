# convert.py
import os
import mlflow
from fastapi import FastAPI, Form
from typing import Optional
from huggingface_hub import login, HfApi
import subprocess

app = FastAPI()

mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-tracking-service:5000")
mlflow.set_tracking_uri(mlflow_tracking_uri)


@app.post("/convert")
def convert(
        hf_token: str = Form(...),
        model_dir: str = Form("/model"),
        output_dir: str = Form("/model/converted"),
        hf_repo_id: str = Form(...),
        conversion_script_url: str = Form(
            "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/convert_graph_to_onnx.py")
):
    """
    Convert the model and upload to Hugging Face Hub.

    Parameters:
    - hf_token (str): Hugging Face API token.
    - model_dir (str): Path to the model directory.
    - output_dir (str): Output directory for the converted model.
    - hf_repo_id (str): Hugging Face repository ID to upload the model.
    - conversion_script_url (str): URL to download the conversion script.
    """
    with mlflow.start_run(run_name="Model-Conversion-and-Upload"):
        mlflow.log_params({
            "model_dir": model_dir,
            "output_dir": output_dir,
            "hf_repo_id": hf_repo_id,
            "conversion_script_url": conversion_script_url
        })

        conversion_script = "convert_graph_to_onnx.py"
        if not os.path.exists(conversion_script):
            subprocess.run(["wget", conversion_script_url, "-O", conversion_script], check=True)

        subprocess.run(["python", conversion_script, "--model", model_dir, "--output", f"{output_dir}/model.onnx"],
                       check=True)

        login(token=hf_token)

        api = HfApi()
        api.upload_folder(
            folder_path=output_dir,
            path_in_repo="converted_model",
            repo_id=hf_repo_id,
            repo_type="model"
        )

        mlflow.log_artifacts(output_dir)

    return {"status": "converted and uploaded", "repository": hf_repo_id}
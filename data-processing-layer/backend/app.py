import os
import re
import json
import shutil
import pandas as pd
import mlflow
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from datasets import Dataset
from huggingface_hub import login, HfApi

app = FastAPI()

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow-tracking-service:5001')
mlflow.set_tracking_uri(mlflow_tracking_uri)


def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s#가-힣.,!?%()+\-=/<>]', '', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()


@app.post("/uploadfile/")
async def upload_file(
        file: UploadFile = File(...),
        repo_name: str = Form(...),
        hf_token: str = Form(...)
):
    filename = file.filename
    file_location = os.path.join(UPLOAD_DIR, filename)
    os.makedirs(UPLOAD_DIR, exist_ok=True)

    try:
        # Save the uploaded file
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        data_records = []

        # Process file based on extension
        if filename.endswith(".xlsx"):
            excel = pd.ExcelFile(file_location)
            for sheet_name in excel.sheet_names:
                df = excel.parse(sheet_name=sheet_name)
                if set(df.columns) != {"Instruction", "Output"}:
                    raise ValueError(
                        f"Excel sheet '{sheet_name}' must have exactly columns 'Instruction' and 'Output'.")
                df.dropna(subset=["Instruction", "Output"], inplace=True)
                for _, row in df.iterrows():
                    instruction = preprocess_text(row["Instruction"])
                    output = preprocess_text(row["Output"])
                    if instruction and output:
                        data_records.append({"type": sheet_name, "instruction": instruction, "output": output})
        elif filename.endswith(".csv"):
            df = pd.read_csv(file_location)
            type_name = os.path.splitext(filename)[0]
            if set(df.columns) != {"Instruction", "Output"}:
                raise ValueError("CSV file must have exactly columns 'Instruction' and 'Output'.")
            df.dropna(subset=["Instruction", "Output"], inplace=True)
            for _, row in df.iterrows():
                instruction = preprocess_text(row["Instruction"])
                output = preprocess_text(row["Output"])
                if instruction and output:
                    data_records.append({"type": type_name, "instruction": instruction, "output": output})
        else:
            raise ValueError("Invalid file format. File must be .xlsx or .csv")

        # Check if any valid records were found
        if not data_records:
            raise ValueError("No valid records found in the file.")

        # Save processed data as JSONL
        jsonl_path = os.path.join(UPLOAD_DIR, "processed_dataset.jsonl")
        with open(jsonl_path, "w", encoding="utf-8") as jf:
            for rec in data_records:
                json.dump(rec, jf, ensure_ascii=False)
                jf.write("\n")

        # MLflow logging (wrapped to avoid complete failure on MLflow errors)
        try:
            with mlflow.start_run(run_name="upload_to_hf_hub"):
                mlflow.log_param("repo_name", repo_name)
                mlflow.log_metric("total_records", len(data_records))
                mlflow.log_artifact(jsonl_path)
        except Exception as mlflow_e:
            print("MLflow logging failed:", mlflow_e)

        # Upload to Hugging Face Hub
        login(token=hf_token)
        api = HfApi()
        api.create_repo(repo_id=repo_name, exist_ok=True, repo_type="dataset")
        dataset = Dataset.from_list(data_records)
        dataset.push_to_hub(repo_id=repo_name, split="train", token=hf_token)

        # Clean up temporary files
        os.remove(file_location)
        os.remove(jsonl_path)

        return {"status": "success", "uploaded_records": len(data_records),
                "url": f"https://huggingface.co/datasets/{repo_name}"}

    except Exception as e:
        if os.path.exists(file_location):
            os.remove(file_location)
        print("Error in /uploadfile/:", str(e))
        return JSONResponse({"status": "error", "message": str(e)}, status_code=400)
